use ahash::AHashMap;
use roaring::RoaringBitmap;
use petgraph::graph::{UnGraph, NodeIndex};
use petgraph::algo::astar;

use qor_core::neuron::{Neuron, StoredNeuron};
use qor_core::truth_value::TruthValue;

// ── MORK-inspired Prefix Trie ───────────────────────────────────────────
//
// Instead of linear scan or simple HashMap, QOR uses a prefix trie
// (inspired by MORK's PathMap) for O(k) neuron lookup where k = expression depth.
//
// (bird tweety)     → trie path: ["bird", "tweety"]
// (bird eagle)      → trie path: ["bird", "eagle"]
// (claim text src)  → trie path: ["claim", "text", "src"]
//
// Querying (bird $x) follows ["bird"] then branches to ALL children → O(1 + fan-out).
// Querying (bird tweety) follows ["bird"]["tweety"] exactly → O(2).

/// A node in the prefix trie. Each level corresponds to one element
/// of an expression's children.
#[derive(Default, Clone)]
struct TrieNode {
    /// Children keyed by stringified neuron element.
    children: AHashMap<String, TrieNode>,
    /// Indices into the main neurons Vec for neurons stored at this path.
    /// RoaringBitmap gives O(1) membership test and fast sorted iteration.
    values: RoaringBitmap,
}

/// Key type for trie queries: either a concrete string or a wildcard (variable).
enum QueryKey {
    Concrete(String),
    Wildcard,
}

/// Insert a neuron index at a path in the trie.
fn trie_insert(node: &mut TrieNode, path: &[String], idx: u32) {
    if path.is_empty() {
        node.values.insert(idx);
        return;
    }
    let child = node
        .children
        .entry(path[0].clone())
        .or_default();
    trie_insert(child, &path[1..], idx);
}

/// Query the trie with a path that may contain wildcards.
/// Wildcards branch to ALL children at that level.
fn trie_query(node: &TrieNode, path: &[QueryKey], results: &mut Vec<u32>) {
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
            // Branch to ALL children — variable matches anything
            for child in node.children.values() {
                trie_query(child, &path[1..], results);
            }
        }
    }
}

/// Extract an insert path (all concrete) from a ground neuron.
fn neuron_to_insert_path(neuron: &Neuron) -> Option<Vec<String>> {
    match neuron {
        Neuron::Expression(children) if !children.is_empty() => {
            Some(children.iter().map(|c| c.to_string()).collect())
        }
        _ => None,
    }
}

/// Extract a query path (may contain wildcards) from a pattern.
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

pub(crate) fn has_variables(neuron: &Neuron) -> bool {
    match neuron {
        Neuron::Variable(_) => true,
        Neuron::Expression(children) => children.iter().any(has_variables),
        _ => false,
    }
}

// ── NeuronStore ─────────────────────────────────────────────────────────

/// In-memory knowledge store with MORK-style prefix trie indexing.
///
/// - O(k) exact lookup (k = expression depth)
/// - O(k + fan-out) pattern queries with variables
/// - Automatic belief revision on duplicate neurons
/// - Temporal decay support
/// - Roaring bitmap indices for O(1) membership and fast intersection
pub struct NeuronStore {
    neurons: Vec<StoredNeuron>,
    trie: TrieNode,
}

impl NeuronStore {
    pub fn new() -> Self {
        NeuronStore {
            neurons: Vec::new(),
            trie: TrieNode::default(),
        }
    }

    /// Insert a fact (asserted, not inferred).
    pub fn insert(&mut self, neuron: Neuron, tv: TruthValue) {
        self.insert_inner(neuron, tv, false, None);
    }

    /// Insert a fact with a temporal decay rate.
    pub fn insert_with_decay(&mut self, neuron: Neuron, tv: TruthValue, decay: Option<f64>) {
        self.insert_inner(neuron, tv, false, decay);
    }

    /// Insert a derived fact (from forward chaining).
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
                    self.neurons[i].tv = self.neurons[i].tv.revision(&tv);
                    if let Some(rate) = decay {
                        self.neurons[i].decay_rate = Some(rate);
                    }
                    return;
                }
            }
        } else {
            // Fallback: linear scan for non-expression neurons
            if let Some(existing) = self.neurons.iter_mut().find(|sn| sn.neuron == neuron) {
                existing.tv = existing.tv.revision(&tv);
                if let Some(rate) = decay {
                    existing.decay_rate = Some(rate);
                }
                return;
            }
        }

        // New neuron — add to store and trie
        let idx = self.neurons.len() as u32;
        if let Some(path) = neuron_to_insert_path(&neuron) {
            trie_insert(&mut self.trie, &path, idx);
        }
        self.neurons.push(StoredNeuron {
            neuron,
            tv,
            timestamp: None,
            decay_rate: decay,
            inferred: is_inferred,
        });
    }

    /// Insert a fact for a "functional" predicate (one value per subject).
    /// For 3-element expressions `(pred subject value)`, if an existing fact
    /// `(pred subject other_value)` exists, keeps the higher-confidence one.
    /// Use this for predicates like `color`, `size`, `position` where only one value is valid.
    pub fn insert_functional(&mut self, neuron: Neuron, tv: TruthValue) {
        if let Neuron::Expression(parts) = &neuron {
            if parts.len() == 3 {
                if let Neuron::Symbol(pred) = &parts[0] {
                    let pattern = Neuron::expression(vec![
                        Neuron::Symbol(pred.clone()),
                        parts[1].clone(),
                        Neuron::variable("_contra"),
                    ]);
                    if let Some(qp) = neuron_to_query_path(&pattern) {
                        let mut indices = Vec::new();
                        trie_query(&self.trie, &qp, &mut indices);
                        for &idx in &indices {
                            let i = idx as usize;
                            if let Neuron::Expression(ex_parts) = &self.neurons[i].neuron {
                                if ex_parts.len() == 3
                                    && ex_parts[0] == parts[0]
                                    && ex_parts[1] == parts[1]
                                    && ex_parts[2] != parts[2]
                                {
                                    if tv.confidence > self.neurons[i].tv.confidence {
                                        self.neurons[i].neuron = neuron;
                                        self.neurons[i].tv = tv;
                                    } else {
                                        // Equal or lower confidence — apply belief revision
                                        self.neurons[i].tv = self.neurons[i].tv.revision(&tv);
                                    }
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }
        // No contradiction found — normal insert
        self.insert(neuron, tv);
    }

    /// Find contradictions: returns pairs of neurons with same predicate+subject
    /// but different values (3-element expressions only).
    pub fn find_contradictions(&self) -> Vec<(&StoredNeuron, &StoredNeuron)> {
        let mut result = Vec::new();
        let mut seen: AHashMap<(String, String), usize> = AHashMap::new();
        for (i, sn) in self.neurons.iter().enumerate() {
            if let Neuron::Expression(parts) = &sn.neuron {
                if parts.len() == 3 {
                    let key = (parts[0].to_string(), parts[1].to_string());
                    if let Some(&prev) = seen.get(&key) {
                        if self.neurons[prev].neuron != sn.neuron {
                            result.push((&self.neurons[prev], sn));
                        }
                    } else {
                        seen.insert(key, i);
                    }
                }
            }
        }
        result
    }

    /// Query: find all neurons matching a pattern.
    /// Uses the prefix trie for fast candidate lookup, then verifies with pattern matching.
    pub fn query(&self, pattern: &Neuron) -> Vec<&StoredNeuron> {
        if let Some(query_path) = neuron_to_query_path(pattern) {
            let mut indices = Vec::new();
            trie_query(&self.trie, &query_path, &mut indices);
            // RoaringBitmap iterates in sorted order — no sort needed
            indices
                .iter()
                .map(|&i| &self.neurons[i as usize])
                .filter(|sn| matches_pattern(pattern, &sn.neuron))
                .collect()
        } else {
            // Fallback: linear scan for non-expression patterns
            self.neurons
                .iter()
                .filter(|sn| matches_pattern(pattern, &sn.neuron))
                .collect()
        }
    }

    /// Check if a neuron already exists in the store.
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

    /// MORK-style O(1) exact lookup: get a ground neuron's stored entry.
    /// For fully-bound conditions, this avoids query+iterate+unify overhead.
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

    /// Iterate over all stored neurons.
    pub fn all(&self) -> &[StoredNeuron] {
        &self.neurons
    }

    pub fn len(&self) -> usize {
        self.neurons.len()
    }

    pub fn is_empty(&self) -> bool {
        self.neurons.is_empty()
    }

    /// Apply temporal decay to all neurons with a decay rate.
    pub fn apply_decay(&mut self) -> bool {
        let mut changed = false;
        for sn in &mut self.neurons {
            if let Some(rate) = sn.decay_rate {
                if sn.tv.confidence > 0.001 {
                    sn.tv.confidence = (sn.tv.confidence - rate).max(0.0);
                    changed = true;
                }
            }
        }
        changed
    }

    /// Remove all neurons whose first element (predicate) matches any of the given predicates.
    /// Rebuilds the trie after removal. Returns the count of removed neurons.
    pub fn remove_by_predicate(&mut self, predicates: &[&str]) -> usize {
        let before = self.neurons.len();
        self.neurons.retain(|sn| {
            if let Neuron::Expression(parts) = &sn.neuron {
                if let Some(Neuron::Symbol(pred)) = parts.first() {
                    return !predicates.contains(&pred.as_str());
                }
            }
            true
        });
        let removed = before - self.neurons.len();
        if removed > 0 {
            self.trie = TrieNode::default();
            for (idx, sn) in self.neurons.iter().enumerate() {
                if let Some(path) = neuron_to_insert_path(&sn.neuron) {
                    trie_insert(&mut self.trie, &path, idx as u32);
                }
            }
        }
        removed
    }

    /// Remove only INFERRED facts matching a given predicate.
    /// Keeps asserted facts intact — important when source and target share a predicate.
    pub fn remove_inferred_by_predicate(&mut self, predicate: &str) -> usize {
        let before = self.neurons.len();
        self.neurons.retain(|sn| {
            if sn.inferred {
                if let Neuron::Expression(parts) = &sn.neuron {
                    if let Some(Neuron::Symbol(pred)) = parts.first() {
                        return pred != predicate;
                    }
                }
            }
            true
        });
        let removed = before - self.neurons.len();
        if removed > 0 {
            self.trie = TrieNode::default();
            for (idx, sn) in self.neurons.iter().enumerate() {
                if let Some(path) = neuron_to_insert_path(&sn.neuron) {
                    trie_insert(&mut self.trie, &path, idx as u32);
                }
            }
        }
        removed
    }

    /// Stats: number of trie nodes (for diagnostics).
    pub fn trie_node_count(&self) -> usize {
        count_trie_nodes(&self.trie)
    }
}

fn count_trie_nodes(node: &TrieNode) -> usize {
    1 + node
        .children
        .values()
        .map(count_trie_nodes)
        .sum::<usize>()
}

impl Clone for NeuronStore {
    fn clone(&self) -> Self {
        NeuronStore {
            neurons: self.neurons.clone(),
            trie: self.trie.clone(),
        }
    }
}

impl Default for NeuronStore {
    fn default() -> Self {
        Self::new()
    }
}

// ── Iterators ──────────────────────────────────────────────────────────
//
// Three iterator types for different traversal patterns:
// - StoreIter:      flat iteration over all neurons (O(n))
// - PredicateIter:  trie-accelerated filtering by first element (O(fan-out))
// - TrieWalker:     depth-first trie walk yielding (path, &StoredNeuron)

/// Flat iterator over all stored neurons.
pub struct StoreIter<'a> {
    inner: std::slice::Iter<'a, StoredNeuron>,
}

impl<'a> Iterator for StoreIter<'a> {
    type Item = &'a StoredNeuron;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a> ExactSizeIterator for StoreIter<'a> {}

/// Iterator that yields only neurons with a given predicate (first element).
/// Uses the trie to skip irrelevant branches — O(fan-out) not O(n).
pub struct PredicateIter<'a> {
    store: &'a NeuronStore,
    indices: Vec<u32>,
    pos: usize,
}

impl<'a> Iterator for PredicateIter<'a> {
    type Item = &'a StoredNeuron;
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.indices.len() {
            let idx = self.indices[self.pos];
            self.pos += 1;
            Some(&self.store.neurons[idx as usize])
        } else {
            None
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.indices.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for PredicateIter<'a> {}

/// Depth-first trie walker yielding (path, &StoredNeuron) pairs.
/// Walks the trie structure, showing how neurons are organized.
pub struct TrieWalker<'a> {
    store: &'a NeuronStore,
    // Stack of (node, path, value_position) for DFS
    stack: Vec<TrieFrame<'a>>,
}

struct TrieFrame<'a> {
    path: Vec<String>,
    /// Pre-collected indices from the bitmap for position-based access.
    values: Vec<u32>,
    value_pos: usize,
    children: Vec<(&'a String, &'a TrieNode)>,
    child_pos: usize,
}

impl<'a> Iterator for TrieWalker<'a> {
    type Item = (Vec<String>, &'a StoredNeuron);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let frame = self.stack.last_mut()?;

            // Yield values at this node first
            if frame.value_pos < frame.values.len() {
                let idx = frame.values[frame.value_pos] as usize;
                frame.value_pos += 1;
                return Some((frame.path.clone(), &self.store.neurons[idx]));
            }

            // Then descend to next child
            if frame.child_pos < frame.children.len() {
                let (key, child) = frame.children[frame.child_pos];
                frame.child_pos += 1;
                let mut child_path = frame.path.clone();
                child_path.push(key.clone());
                let children: Vec<_> = child.children.iter().collect();
                let values: Vec<u32> = child.values.iter().collect();
                self.stack.push(TrieFrame {
                    path: child_path,
                    values,
                    value_pos: 0,
                    children,
                    child_pos: 0,
                });
            } else {
                // Done with this node
                self.stack.pop();
            }
        }
    }
}

/// Collect all value indices from a trie node and its descendants.
fn collect_all_indices(node: &TrieNode, out: &mut Vec<u32>) {
    out.extend(node.values.iter());
    for child in node.children.values() {
        collect_all_indices(child, out);
    }
}

impl NeuronStore {
    /// Iterate over all neurons in insertion order.
    pub fn iter(&self) -> StoreIter<'_> {
        StoreIter {
            inner: self.neurons.iter(),
        }
    }

    /// Iterate over neurons with a specific predicate (first element).
    /// Uses the trie for O(fan-out) lookup instead of O(n) scan.
    pub fn iter_predicate(&self, predicate: &str) -> PredicateIter<'_> {
        let mut indices = Vec::new();
        if let Some(child) = self.trie.children.get(predicate) {
            collect_all_indices(child, &mut indices);
        }
        PredicateIter {
            store: self,
            indices,
            pos: 0,
        }
    }

    /// Walk the trie structure depth-first, yielding (path, neuron) pairs.
    /// Useful for inspecting how the trie organizes knowledge.
    pub fn walk_trie(&self) -> TrieWalker<'_> {
        let children: Vec<_> = self.trie.children.iter().collect();
        let values: Vec<u32> = self.trie.values.iter().collect();
        TrieWalker {
            store: self,
            stack: vec![TrieFrame {
                path: Vec::new(),
                values,
                value_pos: 0,
                children,
                child_pos: 0,
            }],
        }
    }

    /// Count of neurons matching a predicate (trie-accelerated).
    pub fn count_predicate(&self, predicate: &str) -> usize {
        self.iter_predicate(predicate).count()
    }

    /// Get all distinct predicates (first elements) in the store.
    pub fn predicates(&self) -> Vec<String> {
        self.trie.children.keys().cloned().collect()
    }

    /// Graph walk: BFS through fact triples to find a path from `from` to `to`.
    /// Treats 3-element expressions `(pred A B)` as edges A→B (and B→A).
    /// Returns the path as a vec of (predicate, node) pairs, or None if no path within max_hops.
    pub fn find_path(&self, from: &str, to: &str, max_hops: usize) -> Option<Vec<(String, String)>> {
        if from == to {
            return Some(vec![]);
        }

        let (graph, node_map) = self.to_graph();
        let from_idx = node_map.get(from)?;
        let to_idx = node_map.get(to)?;

        let result = astar(&graph, *from_idx, |n| n == *to_idx, |_| 1usize, |_| 0usize)?;
        let (cost, path) = result;
        if cost > max_hops {
            return None;
        }

        let mut result_path = Vec::new();
        for window in path.windows(2) {
            let edge = graph.find_edge(window[0], window[1])
                .or_else(|| graph.find_edge(window[1], window[0]))?;
            let pred = graph.edge_weight(edge)?.clone();
            let node_name = graph.node_weight(window[1])?.clone();
            result_path.push((pred, node_name));
        }
        Some(result_path)
    }

    /// Build a petgraph from all 3-element expressions.
    /// Returns (graph, node_index_map).
    pub fn to_graph(&self) -> (UnGraph<String, String>, AHashMap<String, NodeIndex>) {
        let mut graph = UnGraph::<String, String>::new_undirected();
        let mut node_map: AHashMap<String, NodeIndex> = AHashMap::new();

        for sn in &self.neurons {
            if let Neuron::Expression(parts) = &sn.neuron {
                if parts.len() == 3 {
                    let pred = parts[0].to_string();
                    let a = parts[1].to_string();
                    let b = parts[2].to_string();
                    let ai = *node_map.entry(a.clone()).or_insert_with(|| graph.add_node(a));
                    let bi = *node_map.entry(b.clone()).or_insert_with(|| graph.add_node(b));
                    graph.add_edge(ai, bi, pred);
                }
            }
        }
        (graph, node_map)
    }

    /// Find connected components in the fact graph.
    pub fn connected_components(&self) -> Vec<Vec<String>> {
        let (graph, node_map) = self.to_graph();
        if graph.node_count() == 0 {
            return vec![];
        }

        use petgraph::visit::Bfs;

        let mut result: Vec<Vec<String>> = Vec::new();
        let mut seen: AHashMap<NodeIndex, usize> = AHashMap::new();
        let idx_to_name: AHashMap<NodeIndex, &String> = node_map.iter().map(|(n, &i)| (i, n)).collect();

        for (_, &start) in &node_map {
            if seen.contains_key(&start) { continue; }
            let comp_idx = result.len();
            result.push(Vec::new());
            let mut bfs = Bfs::new(&graph, start);
            while let Some(node) = bfs.next(&graph) {
                if seen.insert(node, comp_idx).is_none() {
                    if let Some(name) = idx_to_name.get(&node) {
                        result[comp_idx].push((*name).clone());
                    }
                }
            }
        }
        result
    }

    /// Shortest distance (hop count) between two entities.
    pub fn shortest_distance(&self, from: &str, to: &str) -> Option<usize> {
        if from == to { return Some(0); }
        let (graph, node_map) = self.to_graph();
        let from_idx = node_map.get(from)?;
        let to_idx = node_map.get(to)?;
        let (cost, _) = astar(&graph, *from_idx, |n| n == *to_idx, |_| 1usize, |_| 0usize)?;
        Some(cost)
    }
}

impl<'a> IntoIterator for &'a NeuronStore {
    type Item = &'a StoredNeuron;
    type IntoIter = StoreIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Pattern matching: does a pattern (with variables) match a concrete neuron?
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_query_exact() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]),
            TruthValue::from_strength(0.99),
        );

        let results = store.query(&Neuron::expression(vec![
            Neuron::symbol("bird"),
            Neuron::symbol("tweety"),
        ]));
        assert_eq!(results.len(), 1);
        assert!((results[0].tv.strength - 0.99).abs() < 0.01);
        assert!(!results[0].inferred);
    }

    #[test]
    fn test_query_with_variable() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("eagle")]),
            TruthValue::from_strength(0.95),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("fish"), Neuron::symbol("salmon")]),
            TruthValue::from_strength(0.99),
        );

        // ? (bird $x) → tweety and eagle, not salmon
        let results = store.query(&Neuron::expression(vec![
            Neuron::symbol("bird"),
            Neuron::variable("x"),
        ]));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_no_match() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]),
            TruthValue::from_strength(0.99),
        );

        let results = store.query(&Neuron::expression(vec![
            Neuron::symbol("mammal"),
            Neuron::variable("x"),
        ]));
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_belief_revision_on_duplicate() {
        let mut store = NeuronStore::new();
        let neuron = Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]);

        store.insert(neuron.clone(), TruthValue::new(0.70, 0.40));
        assert_eq!(store.len(), 1);

        store.insert(neuron.clone(), TruthValue::new(0.95, 0.80));
        assert_eq!(store.len(), 1); // still 1, not 2

        let results = store.query(&neuron);
        let tv = results[0].tv;
        assert!(tv.confidence > 0.80);
        assert!(tv.strength > 0.80);
    }

    #[test]
    fn test_pattern_matching_nested() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![
                Neuron::symbol("claim"),
                Neuron::str_val("eagles can fly"),
                Neuron::symbol("wikipedia"),
            ]),
            TruthValue::from_strength(0.92),
        );

        let results = store.query(&Neuron::expression(vec![
            Neuron::symbol("claim"),
            Neuron::variable("x"),
            Neuron::symbol("wikipedia"),
        ]));
        assert_eq!(results.len(), 1);

        let results = store.query(&Neuron::expression(vec![
            Neuron::symbol("claim"),
            Neuron::variable("x"),
            Neuron::symbol("twitter"),
        ]));
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_insert_inferred() {
        let mut store = NeuronStore::new();
        let neuron = Neuron::expression(vec![Neuron::symbol("flies"), Neuron::symbol("tweety")]);

        store.insert_inferred(neuron.clone(), TruthValue::from_strength(0.94));

        let results = store.query(&neuron);
        assert_eq!(results.len(), 1);
        assert!(results[0].inferred);
    }

    #[test]
    fn test_asserted_stays_asserted_after_inferred_revision() {
        let mut store = NeuronStore::new();
        let neuron = Neuron::expression(vec![Neuron::symbol("flies"), Neuron::symbol("tweety")]);

        store.insert(neuron.clone(), TruthValue::new(0.80, 0.70));
        assert!(!store.query(&neuron)[0].inferred);

        store.insert_inferred(neuron.clone(), TruthValue::new(0.94, 0.73));
        assert!(!store.query(&neuron)[0].inferred); // stays asserted
        assert!(store.query(&neuron)[0].tv.confidence > 0.70); // but TV revised
    }

    #[test]
    fn test_contains() {
        let mut store = NeuronStore::new();
        let bird = Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]);
        let fish = Neuron::expression(vec![Neuron::symbol("fish"), Neuron::symbol("salmon")]);

        store.insert(bird.clone(), TruthValue::from_strength(0.99));
        assert!(store.contains(&bird));
        assert!(!store.contains(&fish));
    }

    #[test]
    fn test_temporal_decay() {
        let mut store = NeuronStore::new();
        let neuron = Neuron::expression(vec![Neuron::symbol("news"), Neuron::symbol("headline")]);
        store.insert_with_decay(neuron.clone(), TruthValue::new(0.95, 0.90), Some(0.05));

        for _ in 0..5 {
            store.apply_decay();
        }

        let results = store.query(&neuron);
        assert!((results[0].tv.confidence - 0.65).abs() < 0.01);
        assert!((results[0].tv.strength - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_decay_bottoms_at_zero() {
        let mut store = NeuronStore::new();
        store.insert_with_decay(
            Neuron::expression(vec![Neuron::symbol("temp"), Neuron::symbol("data")]),
            TruthValue::new(0.90, 0.10),
            Some(0.05),
        );

        for _ in 0..100 {
            store.apply_decay();
        }

        let results = store.query(&Neuron::expression(vec![
            Neuron::symbol("temp"),
            Neuron::variable("x"),
        ]));
        assert!((results[0].tv.confidence - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_trie_performance() {
        let mut store = NeuronStore::new();
        // Insert 1000 neurons with different predicates
        for i in 0..1000 {
            store.insert(
                Neuron::expression(vec![
                    Neuron::symbol(&format!("pred{}", i)),
                    Neuron::symbol("value"),
                ]),
                TruthValue::from_strength(0.99),
            );
        }
        // Query for a specific predicate — trie gives O(2) lookup
        let results = store.query(&Neuron::expression(vec![
            Neuron::symbol("pred500"),
            Neuron::variable("x"),
        ]));
        assert_eq!(results.len(), 1);

        // Trie should have ~1001 nodes (root + 1000 pred + 1000 value)
        assert!(store.trie_node_count() > 1000);
    }

    #[test]
    fn test_trie_shared_prefix() {
        let mut store = NeuronStore::new();
        // All share the "bird" prefix
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("eagle")]),
            TruthValue::from_strength(0.95),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("robin")]),
            TruthValue::from_strength(0.90),
        );

        // Trie: root → bird → {tweety, eagle, robin}
        // Query (bird $x) follows root → bird → ALL children = 3 results
        let results = store.query(&Neuron::expression(vec![
            Neuron::symbol("bird"),
            Neuron::variable("x"),
        ]));
        assert_eq!(results.len(), 3);

        // Trie nodes: root(1) + bird(1) + tweety(1) + eagle(1) + robin(1) = 5
        assert_eq!(store.trie_node_count(), 5);
    }

    // ── Iterator tests ──

    #[test]
    fn test_store_iter() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("eagle")]),
            TruthValue::from_strength(0.95),
        );

        let names: Vec<String> = store.iter().map(|sn| sn.neuron.to_string()).collect();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"(bird tweety)".to_string()));
        assert!(names.contains(&"(bird eagle)".to_string()));
    }

    #[test]
    fn test_store_into_iterator() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]),
            TruthValue::from_strength(0.99),
        );

        // Should work in for loops
        let mut count = 0;
        for _sn in &store {
            count += 1;
        }
        assert_eq!(count, 1);
    }

    #[test]
    fn test_predicate_iter() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("eagle")]),
            TruthValue::from_strength(0.95),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("fish"), Neuron::symbol("salmon")]),
            TruthValue::from_strength(0.99),
        );

        assert_eq!(store.iter_predicate("bird").count(), 2);
        assert_eq!(store.iter_predicate("fish").count(), 1);
        assert_eq!(store.iter_predicate("mammal").count(), 0);
    }

    #[test]
    fn test_predicate_iter_exact_size() {
        let mut store = NeuronStore::new();
        for i in 0..5 {
            store.insert(
                Neuron::expression(vec![
                    Neuron::symbol("bird"),
                    Neuron::symbol(&format!("bird{}", i)),
                ]),
                TruthValue::from_strength(0.99),
            );
        }

        let iter = store.iter_predicate("bird");
        assert_eq!(iter.len(), 5);
    }

    #[test]
    fn test_trie_walker() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("eagle")]),
            TruthValue::from_strength(0.95),
        );

        let walked: Vec<_> = store.walk_trie().collect();
        assert_eq!(walked.len(), 2);
        // Each entry has a path
        for (path, _sn) in &walked {
            assert_eq!(path.len(), 2);
            assert_eq!(path[0], "bird");
        }
    }

    #[test]
    fn test_count_predicate() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("eagle")]),
            TruthValue::from_strength(0.95),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("fish"), Neuron::symbol("salmon")]),
            TruthValue::from_strength(0.99),
        );

        assert_eq!(store.count_predicate("bird"), 2);
        assert_eq!(store.count_predicate("fish"), 1);
    }

    #[test]
    fn test_predicates() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("fish"), Neuron::symbol("salmon")]),
            TruthValue::from_strength(0.99),
        );

        let preds = store.predicates();
        assert_eq!(preds.len(), 2);
        assert!(preds.contains(&"bird".to_string()));
        assert!(preds.contains(&"fish".to_string()));
    }

    #[test]
    fn test_remove_by_predicate_basic() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("input"), Neuron::symbol("0"), Neuron::symbol("hello")]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("input"), Neuron::symbol("1"), Neuron::symbol("world")]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]),
            TruthValue::from_strength(0.99),
        );

        let removed = store.remove_by_predicate(&["input"]);
        assert_eq!(removed, 2);
        assert_eq!(store.len(), 1);
        assert!(store.contains(&Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")])));
    }

    #[test]
    fn test_remove_by_predicate_preserves_others() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("intent"), Neuron::symbol("greet")]),
            TruthValue::from_strength(0.95),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("response"), Neuron::symbol("greet"), Neuron::symbol("hello")]),
            TruthValue::from_strength(0.90),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("meaning"), Neuron::symbol("overbought"), Neuron::symbol("high")]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("vocab"), Neuron::symbol("hello"), Neuron::symbol("greeting")]),
            TruthValue::from_strength(0.99),
        );

        let removed = store.remove_by_predicate(&["intent", "response"]);
        assert_eq!(removed, 2);
        assert_eq!(store.len(), 2);
        // Trie should work after rebuild
        let results = store.query(&Neuron::expression(vec![
            Neuron::symbol("meaning"),
            Neuron::variable("x"),
            Neuron::variable("y"),
        ]));
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_trie_three_level() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![
                Neuron::symbol("claim"),
                Neuron::symbol("eagles-fly"),
                Neuron::symbol("wikipedia"),
            ]),
            TruthValue::from_strength(0.92),
        );
        store.insert(
            Neuron::expression(vec![
                Neuron::symbol("claim"),
                Neuron::symbol("earth-round"),
                Neuron::symbol("nasa"),
            ]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![
                Neuron::symbol("claim"),
                Neuron::symbol("eagles-fly"),
                Neuron::symbol("textbook"),
            ]),
            TruthValue::from_strength(0.95),
        );

        // Query: all claims about eagles-fly from any source
        let results = store.query(&Neuron::expression(vec![
            Neuron::symbol("claim"),
            Neuron::symbol("eagles-fly"),
            Neuron::variable("source"),
        ]));
        assert_eq!(results.len(), 2); // wikipedia and textbook

        // Query: all claims from any topic, any source
        let results = store.query(&Neuron::expression(vec![
            Neuron::symbol("claim"),
            Neuron::variable("topic"),
            Neuron::variable("source"),
        ]));
        assert_eq!(results.len(), 3); // all three
    }
}
