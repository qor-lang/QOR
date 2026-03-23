// ── Persistent NeuronStore — O(1) Snapshots via HAMT ─────────────────────
//
// Feature-gated behind `--features persistent`.
//
// Uses the `im` crate's persistent data structures (Hash Array Mapped Tries)
// with structural sharing. Clone/snapshot is O(1) instead of O(n) because
// unchanged subtrees are shared between the original and the clone.
//
// This is critical for `test_hypothesis()` which snapshots the entire store,
// injects facts, chains, measures, then restores. With HAMT, the snapshot
// and restore steps are essentially free.
//
// Public API mirrors NeuronStore exactly — drop-in replacement.

#![cfg(feature = "persistent")]

use std::collections::HashMap;
use im::{Vector as ImVector, HashMap as ImHashMap};
use roaring::RoaringBitmap;

use qor_core::neuron::{Neuron, StoredNeuron};
use qor_core::truth_value::TruthValue;

use crate::store::has_variables;

// ── Persistent Trie Node ──────────────────────────────────────────────

/// A trie node using persistent (structural-sharing) data structures.
/// Clone is O(1) — shares unchanged subtrees with the original.
#[derive(Clone)]
struct PersistentTrieNode {
    children: ImHashMap<String, PersistentTrieNode>,
    values: RoaringBitmap,
}

impl Default for PersistentTrieNode {
    fn default() -> Self {
        PersistentTrieNode {
            children: ImHashMap::new(),
            values: RoaringBitmap::new(),
        }
    }
}

/// Key type for trie queries.
enum QueryKey {
    Concrete(String),
    Wildcard,
}

fn trie_insert(node: &mut PersistentTrieNode, path: &[String], idx: u32) {
    if path.is_empty() {
        node.values.insert(idx);
        return;
    }
    let mut child = node.children.get(&path[0]).cloned().unwrap_or_default();
    trie_insert(&mut child, &path[1..], idx);
    node.children.insert(path[0].clone(), child);
}

fn trie_query(node: &PersistentTrieNode, path: &[QueryKey], results: &mut Vec<u32>) {
    if path.is_empty() {
        results.extend(node.values.iter());
        return;
    }
    match &path[0] {
        QueryKey::Concrete(key) => {
            if let Some(child) = node.children.get(key) {
                trie_query(child, &path[1..], results);
            }
        }
        QueryKey::Wildcard => {
            for (_, child) in node.children.iter() {
                trie_query(child, &path[1..], results);
            }
        }
    }
}

fn neuron_to_insert_path(neuron: &Neuron) -> Option<Vec<String>> {
    match neuron {
        Neuron::Expression(children) if !children.is_empty() => {
            Some(children.iter().map(|c| c.to_string()).collect())
        }
        _ => None,
    }
}

fn neuron_to_query_path(neuron: &Neuron) -> Option<Vec<QueryKey>> {
    match neuron {
        Neuron::Expression(children) if !children.is_empty() => {
            Some(
                children
                    .iter()
                    .map(|c| match c {
                        Neuron::Variable(_) => QueryKey::Wildcard,
                        Neuron::Expression(_) if has_variables(c) => QueryKey::Wildcard,
                        other => QueryKey::Concrete(other.to_string()),
                    })
                    .collect(),
            )
        }
        _ => None,
    }
}

fn matches_pattern(pattern: &Neuron, target: &Neuron) -> bool {
    match (pattern, target) {
        (Neuron::Variable(_), _) => true,
        (Neuron::Symbol(a), Neuron::Symbol(b)) => a == b,
        (Neuron::Expression(pat_children), Neuron::Expression(tgt_children)) => {
            pat_children.len() == tgt_children.len()
                && pat_children
                    .iter()
                    .zip(tgt_children.iter())
                    .all(|(p, t)| matches_pattern(p, t))
        }
        (Neuron::Value(a), Neuron::Value(b)) => a == b,
        _ => false,
    }
}

fn collect_all_indices(node: &PersistentTrieNode, out: &mut Vec<u32>) {
    out.extend(node.values.iter());
    for (_, child) in node.children.iter() {
        collect_all_indices(child, out);
    }
}

// ── PersistentStore ─────────────────────────────────────────────────────

/// NeuronStore with O(1) clone/snapshot via HAMT structural sharing.
///
/// Drop-in replacement for NeuronStore when `--features persistent` is enabled.
/// All public methods have identical signatures and semantics.
#[derive(Clone)]  // O(1) clone via structural sharing!
pub struct PersistentStore {
    neurons: ImVector<StoredNeuron>,
    trie: PersistentTrieNode,
}

impl PersistentStore {
    pub fn new() -> Self {
        PersistentStore {
            neurons: ImVector::new(),
            trie: PersistentTrieNode::default(),
        }
    }

    /// O(1) snapshot — the whole point of this module.
    pub fn snapshot(&self) -> Self {
        self.clone()
    }

    pub fn insert(&mut self, neuron: Neuron, tv: TruthValue) {
        self.insert_inner(neuron, tv, false, None);
    }

    pub fn insert_with_decay(&mut self, neuron: Neuron, tv: TruthValue, decay: Option<f64>) {
        self.insert_inner(neuron, tv, false, decay);
    }

    pub fn insert_inferred(&mut self, neuron: Neuron, tv: TruthValue) {
        self.insert_inner(neuron, tv, true, None);
    }

    fn insert_inner(
        &mut self,
        neuron: Neuron,
        tv: TruthValue,
        is_inferred: bool,
        decay: Option<f64>,
    ) {
        // Try to find existing neuron via trie
        if let Some(path) = neuron_to_insert_path(&neuron) {
            let query_path: Vec<QueryKey> = path
                .iter()
                .map(|s| QueryKey::Concrete(s.clone()))
                .collect();
            let mut candidates = Vec::new();
            trie_query(&self.trie, &query_path, &mut candidates);

            for &idx in &candidates {
                let i = idx as usize;
                if self.neurons[i].neuron == neuron {
                    // Belief revision: merge evidence
                    let mut sn = self.neurons[i].clone();
                    sn.tv = sn.tv.revision(&tv);
                    if let Some(rate) = decay {
                        sn.decay_rate = Some(rate);
                    }
                    self.neurons.set(i, sn);
                    return;
                }
            }
        } else {
            // Fallback: linear scan for non-expression neurons
            for i in 0..self.neurons.len() {
                if self.neurons[i].neuron == neuron {
                    let mut sn = self.neurons[i].clone();
                    sn.tv = sn.tv.revision(&tv);
                    if let Some(rate) = decay {
                        sn.decay_rate = Some(rate);
                    }
                    self.neurons.set(i, sn);
                    return;
                }
            }
        }

        // New neuron — add to store and trie
        let idx = self.neurons.len() as u32;
        if let Some(path) = neuron_to_insert_path(&neuron) {
            trie_insert(&mut self.trie, &path, idx);
        }
        self.neurons.push_back(StoredNeuron {
            neuron,
            tv,
            timestamp: None,
            decay_rate: decay,
            inferred: is_inferred,
        });
    }

    pub fn query(&self, pattern: &Neuron) -> Vec<&StoredNeuron> {
        if let Some(query_path) = neuron_to_query_path(pattern) {
            let mut indices = Vec::new();
            trie_query(&self.trie, &query_path, &mut indices);
            indices
                .iter()
                .map(|&i| &self.neurons[i as usize])
                .filter(|sn| matches_pattern(pattern, &sn.neuron))
                .collect()
        } else {
            self.neurons
                .iter()
                .filter(|sn| matches_pattern(pattern, &sn.neuron))
                .collect()
        }
    }

    pub fn contains(&self, neuron: &Neuron) -> bool {
        if let Some(path) = neuron_to_insert_path(neuron) {
            let query_path: Vec<QueryKey> =
                path.iter().map(|s| QueryKey::Concrete(s.clone())).collect();
            let mut indices = Vec::new();
            trie_query(&self.trie, &query_path, &mut indices);
            indices.iter().any(|&i| self.neurons[i as usize].neuron == *neuron)
        } else {
            self.neurons.iter().any(|sn| sn.neuron == *neuron)
        }
    }

    pub fn get_exact(&self, neuron: &Neuron) -> Option<&StoredNeuron> {
        if let Some(path) = neuron_to_insert_path(neuron) {
            let query_path: Vec<QueryKey> =
                path.iter().map(|s| QueryKey::Concrete(s.clone())).collect();
            let mut indices = Vec::new();
            trie_query(&self.trie, &query_path, &mut indices);
            for &i in &indices {
                if self.neurons[i as usize].neuron == *neuron {
                    return Some(&self.neurons[i as usize]);
                }
            }
            None
        } else {
            self.neurons.iter().find(|sn| sn.neuron == *neuron)
        }
    }

    pub fn len(&self) -> usize {
        self.neurons.len()
    }

    pub fn is_empty(&self) -> bool {
        self.neurons.is_empty()
    }

    /// Convert from regular NeuronStore for snapshotting scenarios.
    pub fn from_regular(store: &crate::store::NeuronStore) -> Self {
        let mut persistent = PersistentStore::new();
        for sn in store.all() {
            if sn.inferred {
                persistent.insert_inferred(sn.neuron.clone(), sn.tv);
            } else if let Some(rate) = sn.decay_rate {
                persistent.insert_with_decay(sn.neuron.clone(), sn.tv, Some(rate));
            } else {
                persistent.insert(sn.neuron.clone(), sn.tv);
            }
        }
        persistent
    }

    /// Convert back to regular NeuronStore.
    pub fn to_regular(&self) -> crate::store::NeuronStore {
        let mut store = crate::store::NeuronStore::new();
        for sn in self.neurons.iter() {
            if sn.inferred {
                store.insert_inferred(sn.neuron.clone(), sn.tv);
            } else if let Some(rate) = sn.decay_rate {
                store.insert_with_decay(sn.neuron.clone(), sn.tv, Some(rate));
            } else {
                store.insert(sn.neuron.clone(), sn.tv);
            }
        }
        store
    }
}

impl Default for PersistentStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_persistent_basic() {
        let mut store = PersistentStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("eagle")]),
            TruthValue::from_strength(0.95),
        );

        let results = store.query(&Neuron::expression(vec![
            Neuron::symbol("bird"),
            Neuron::variable("x"),
        ]));
        assert_eq!(results.len(), 2);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_persistent_snapshot_isolation() {
        let mut store = PersistentStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]),
            TruthValue::from_strength(0.99),
        );

        // Take snapshot
        let snapshot = store.snapshot();

        // Modify original
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("eagle")]),
            TruthValue::from_strength(0.95),
        );

        // Original has 2, snapshot still has 1
        assert_eq!(store.len(), 2);
        assert_eq!(snapshot.len(), 1);

        // Snapshot doesn't see the new fact
        let snap_results = snapshot.query(&Neuron::expression(vec![
            Neuron::symbol("bird"),
            Neuron::variable("x"),
        ]));
        assert_eq!(snap_results.len(), 1);
    }

    #[test]
    fn test_persistent_belief_revision() {
        let mut store = PersistentStore::new();
        let neuron = Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]);

        store.insert(neuron.clone(), TruthValue::new(0.70, 0.40));
        assert_eq!(store.len(), 1);

        store.insert(neuron.clone(), TruthValue::new(0.95, 0.80));
        assert_eq!(store.len(), 1); // still 1 — revised, not duplicated

        let results = store.query(&neuron);
        assert!(results[0].tv.confidence > 0.80);
    }

    #[test]
    fn test_persistent_contains() {
        let mut store = PersistentStore::new();
        let bird = Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]);
        let fish = Neuron::expression(vec![Neuron::symbol("fish"), Neuron::symbol("salmon")]);

        store.insert(bird.clone(), TruthValue::from_strength(0.99));
        assert!(store.contains(&bird));
        assert!(!store.contains(&fish));
    }

    #[test]
    fn test_persistent_get_exact() {
        let mut store = PersistentStore::new();
        let bird = Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]);
        store.insert(bird.clone(), TruthValue::from_strength(0.99));

        let result = store.get_exact(&bird);
        assert!(result.is_some());
        assert!((result.unwrap().tv.strength - 0.99).abs() < 0.01);
    }

    #[test]
    fn test_persistent_from_regular_roundtrip() {
        let mut regular = crate::store::NeuronStore::new();
        regular.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]),
            TruthValue::from_strength(0.99),
        );
        regular.insert_inferred(
            Neuron::expression(vec![Neuron::symbol("flies"), Neuron::symbol("tweety")]),
            TruthValue::from_strength(0.94),
        );

        // Convert to persistent
        let persistent = PersistentStore::from_regular(&regular);
        assert_eq!(persistent.len(), 2);

        // Convert back
        let back = persistent.to_regular();
        assert_eq!(back.len(), 2);
        assert!(back.contains(&Neuron::expression(vec![
            Neuron::symbol("bird"), Neuron::symbol("tweety"),
        ])));
    }

    #[test]
    fn test_persistent_snapshot_speed() {
        // Verify that snapshot is cheap even with many facts
        let mut store = PersistentStore::new();
        for i in 0..1000 {
            store.insert(
                Neuron::expression(vec![
                    Neuron::symbol("fact"),
                    Neuron::symbol(&format!("item{}", i)),
                ]),
                TruthValue::from_strength(0.99),
            );
        }

        // Snapshot should be near-instant (O(1))
        let start = std::time::Instant::now();
        let _snapshot = store.snapshot();
        let elapsed = start.elapsed();

        // Should be well under 1ms (typically < 1μs)
        assert!(elapsed.as_millis() < 10, "snapshot took too long: {:?}", elapsed);
        assert_eq!(store.len(), 1000);
    }
}
