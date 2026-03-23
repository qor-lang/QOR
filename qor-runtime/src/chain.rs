use ahash::{AHashMap, AHashSet};
use std::collections::{HashMap, HashSet};

use qor_core::neuron::{self, AggregateOp, ArithmeticOp, ComparisonOp, Condition, Neuron, QorValue, StoredNeuron};
use qor_core::truth_value::TruthValue;
use qor_core::unify::{apply_bindings, extract_variables, unify, Bindings};

use crate::store::{has_variables, NeuronStore};

/// A rule ready for execution (truth value resolved).
#[derive(Debug, Clone)]
pub struct Rule {
    pub head: Neuron,
    pub body: Vec<Condition>,
    pub tv: TruthValue,
    /// Stratum for stratified negation (default 0). Lower strata fire first.
    pub stratum: u32,
}

impl Rule {
    /// Create a rule with default stratum 0.
    pub fn new(head: Neuron, body: Vec<Condition>, tv: TruthValue) -> Self {
        Rule { head, body, tv, stratum: 0 }
    }
}

// ── Body Condition Reordering (query optimization) ──────────────────────
//
// Greedy reorder of rule body conditions for optimal binding propagation:
//   1. Positive conditions first (they BIND variables), prefer more concrete terms
//   2. Arithmetic/Guards next (O(1) once operands are bound)
//   3. Aggregates after their pattern vars are bound
//   4. Negation last (can only narrow, needs all vars pre-bound)

/// Extract variables that a condition REQUIRES to be bound before it can fire.
fn _condition_requires(cond: &Condition) -> HashSet<String> {
    match cond {
        Condition::Positive(_) => HashSet::new(), // Positives generate bindings, require nothing
        Condition::Guard(_, lhs, rhs) => {
            let mut s: HashSet<String> = extract_variables(lhs).into_iter().collect();
            s.extend(extract_variables(rhs));
            s
        }
        Condition::Arithmetic { lhs, rhs, .. } => {
            let mut s: HashSet<String> = extract_variables(lhs).into_iter().collect();
            s.extend(extract_variables(rhs));
            s
        }
        Condition::Aggregate { pattern, bind_var, .. } => {
            let mut s: HashSet<String> = extract_variables(pattern).into_iter().collect();
            s.remove(bind_var); // bind_var is the iteration variable, not required
            s
        }
        Condition::Negated(n) | Condition::NegatedPresent(n) => {
            extract_variables(n).into_iter().collect()
        }
        Condition::Lookup { subject, .. } => {
            extract_variables(subject).into_iter().collect()
        }
        Condition::EndsWith(n, d) => {
            let mut s: HashSet<String> = extract_variables(n).into_iter().collect();
            s.extend(extract_variables(d));
            s
        }
    }
}

/// Extract variables that a condition PROVIDES (binds for downstream conditions).
fn condition_provides(cond: &Condition) -> HashSet<String> {
    match cond {
        Condition::Positive(n) => extract_variables(n).into_iter().collect(),
        Condition::Arithmetic { result_var, .. } => {
            let mut s = HashSet::new();
            s.insert(result_var.clone());
            s
        }
        Condition::Aggregate { result_var, .. } => {
            let mut s = HashSet::new();
            s.insert(result_var.clone());
            s
        }
        Condition::Lookup { result_var, .. } => {
            let mut s = HashSet::new();
            s.insert(result_var.clone());
            s
        }
        _ => HashSet::new(), // Guards, Negated don't provide bindings
    }
}

/// Score a condition for reorder priority. Lower = placed earlier.
/// Returns (priority_class, unbound_count, original_index) for stable sorting.
fn score_condition(cond: &Condition, bound: &HashSet<String>, orig_idx: usize) -> (u32, usize, usize) {
    match cond {
        Condition::Positive(n) => {
            let vars = extract_variables(n);
            let unbound = vars.iter().filter(|v| !bound.contains(*v)).count();
            (0, unbound, orig_idx) // Class 0: generators. Fewer unbound = more selective
        }
        Condition::Arithmetic { lhs, rhs, .. } => {
            let mut needs: HashSet<String> = extract_variables(lhs).into_iter().collect();
            needs.extend(extract_variables(rhs));
            if needs.is_subset(bound) { (1, 0, orig_idx) } else { (99, 0, orig_idx) }
        }
        Condition::Guard(_, lhs, rhs) => {
            let mut needs: HashSet<String> = extract_variables(lhs).into_iter().collect();
            needs.extend(extract_variables(rhs));
            if needs.is_subset(bound) { (1, 0, orig_idx) } else { (99, 0, orig_idx) }
        }
        Condition::Aggregate { pattern, bind_var, .. } => {
            let mut needs: HashSet<String> = extract_variables(pattern).into_iter().collect();
            needs.remove(bind_var);
            if needs.is_subset(bound) { (2, 0, orig_idx) } else { (99, 0, orig_idx) }
        }
        Condition::Negated(n) | Condition::NegatedPresent(n) => {
            let needs: HashSet<String> = extract_variables(n).into_iter().collect();
            if needs.is_subset(bound) { (3, 0, orig_idx) } else { (99, 0, orig_idx) }
        }
        Condition::Lookup { subject, .. } => {
            let needs: HashSet<String> = extract_variables(subject).into_iter().collect();
            if needs.is_subset(bound) { (0, 0, orig_idx) } else { (99, 0, orig_idx) }
        }
        Condition::EndsWith(n, d) => {
            let mut needs: HashSet<String> = extract_variables(n).into_iter().collect();
            needs.extend(extract_variables(d));
            if needs.is_subset(bound) { (1, 0, orig_idx) } else { (99, 0, orig_idx) }
        }
    }
}

/// Reorder rule body conditions for optimal binding propagation.
/// One-time cost per rule, called before the fixed-point loop.
fn reorder_body(body: &[Condition]) -> Vec<Condition> {
    if body.len() <= 1 {
        return body.to_vec();
    }

    let mut bound: HashSet<String> = HashSet::new();
    let mut remaining: Vec<(usize, &Condition)> = body.iter().enumerate().collect();
    let mut result: Vec<Condition> = Vec::with_capacity(body.len());

    while !remaining.is_empty() {
        // Score all remaining conditions
        let mut best_idx = 0;
        let mut best_score = (u32::MAX, usize::MAX, usize::MAX);

        for (i, &(orig_idx, cond)) in remaining.iter().enumerate() {
            let score = score_condition(cond, &bound, orig_idx);
            if score < best_score {
                best_score = score;
                best_idx = i;
            }
        }

        // If best is deadlocked (score 99), fall back to original order
        if best_score.0 == 99 {
            // Append remaining in original index order
            remaining.sort_by_key(|&(orig_idx, _)| orig_idx);
            for (_, cond) in remaining {
                result.push(cond.clone());
            }
            break;
        }

        let (_, cond) = remaining.remove(best_idx);
        // Update bound vars with what this condition provides
        bound.extend(condition_provides(cond));
        result.push(cond.clone());
    }

    result
}

// ── Forward Chaining (PLN continuous reasoning) ─────────────────────────
//
// When a new fact or rule is added, forward chaining fires:
//   1. Try every rule against the current store
//   2. If all body conditions are satisfied, derive the head as a new fact
//   3. Insert derived facts (belief revision if they already exist)
//   4. Repeat until no new facts are derived (fixed point)

/// Run forward chaining to a fixed point.
/// Returns the number of NEW facts derived.
///
/// MORK-style optimizations:
/// - Rule indexing: map body predicates → rule indices (only fire relevant rules)
/// - Delta tracking: subsequent iterations only fire rules touching new predicates
/// - Callback-based body resolution: zero intermediate Vec allocations
/// - Ground-condition shortcut: O(1) trie lookup for fully-bound conditions
///
/// Stratified execution: rules are grouped by stratum (default 0). Each stratum
/// runs to fixed-point before the next stratum begins. This ensures negated
/// conditions in higher strata see a complete set of facts from lower strata.
pub fn forward_chain(rules: &[Rule], store: &mut NeuronStore) -> usize {
    if rules.is_empty() {
        return 0;
    }

    // ── Auto-stratification via Kahn's topological sort ──
    // Analyze negation dependencies between derived predicates and assign
    // strata so that negated predicates reach fixed-point before negation
    // checks them. This fixes race conditions where a rule negates a fact
    // that another rule derives in the same iteration.
    let mut rules = rules.to_vec();
    let _ = crate::stratify::auto_stratify(&mut rules);

    // ── Stratified grouping ──
    // Collect distinct strata and sort ascending.
    let mut strata: Vec<u32> = rules.iter().map(|r| r.stratum).collect::<AHashSet<_>>().into_iter().collect();
    strata.sort();

    // Fast path: single stratum (common case) — skip grouping overhead
    if strata.len() == 1 {
        return forward_chain_stratum(&rules, store);
    }

    // Multi-stratum: run each stratum to fixed-point in order
    let mut total_new = 0;
    for s in &strata {
        let stratum_rules: Vec<&Rule> = rules.iter().filter(|r| r.stratum == *s).collect();
        // Build a temporary slice-compatible vec of cloned rules for the stratum
        let stratum_rules: Vec<Rule> = stratum_rules.into_iter().cloned().collect();
        total_new += forward_chain_stratum(&stratum_rules, store);
    }
    total_new
}

/// Run forward chaining for a single stratum to fixed-point.
fn forward_chain_stratum(rules: &[Rule], store: &mut NeuronStore) -> usize {
    if rules.is_empty() {
        return 0;
    }

    // ── Pre-reorder rule bodies for optimal binding propagation ──
    // One-time cost: greedy reorder puts most-selective conditions first.
    let rules: Vec<Rule> = rules.iter().map(|r| {
        Rule {
            head: r.head.clone(),
            body: reorder_body(&r.body),
            tv: r.tv,
            stratum: r.stratum,
        }
    }).collect();
    let rules = &rules[..]; // rebind as slice for the rest of the function

    let max_iterations = 10;
    let mut total_new = 0;

    // ── MORK-style rule index: body predicate → rule indices ──
    // A rule fires when ANY of its body predicates has new facts.
    let mut rule_by_pred: AHashMap<String, Vec<usize>> = AHashMap::new();
    let mut unindexed: Vec<usize> = Vec::new();

    for (i, rule) in rules.iter().enumerate() {
        let mut indexed = false;
        for cond in &rule.body {
            if let Condition::Positive(neuron) = cond {
                if let Neuron::Expression(parts) = neuron {
                    if let Some(Neuron::Symbol(pred)) = parts.first() {
                        rule_by_pred.entry(pred.clone()).or_default().push(i);
                        indexed = true;
                    }
                }
            }
        }
        if !indexed {
            unindexed.push(i);
        }
    }

    let mut delta_preds: AHashSet<String> = AHashSet::new();

    for iter in 0..max_iterations {
        let before = store.len();
        let mut candidates: Vec<(Neuron, TruthValue)> = Vec::new();
        let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

        // Determine which rules to fire (sorted for determinism)
        let mut to_fire: Vec<usize> = if iter == 0 {
            // First iteration: fire all rules
            (0..rules.len()).collect()
        } else {
            // Delta-based: only rules whose body predicates changed
            let mut set: AHashSet<usize> = AHashSet::new();
            for pred in &delta_preds {
                if let Some(indices) = rule_by_pred.get(pred) {
                    for &i in indices {
                        set.insert(i);
                    }
                }
            }
            for &i in &unindexed {
                set.insert(i);
            }
            set.into_iter().collect()
        };
        to_fire.sort(); // deterministic rule firing order

        if to_fire.is_empty() {
            break;
        }

        // MORK-style: callback-based body resolution (zero intermediate Vecs)
        // Track candidate index per key so duplicates keep highest confidence.
        let mut seen_idx: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
        for &rule_idx in &to_fire {
            let rule = &rules[rule_idx];
            resolve_body_cb(
                &rule.body,
                &Bindings::new(),
                store,
                TruthValue::new(1.0, 1.0),
                &mut |bindings, body_tv| {
                    let derived = apply_bindings(&rule.head, bindings);
                    let result_tv = body_tv.deduction(&rule.tv);
                    let key = derived.to_string();
                    if seen.insert(key.clone()) {
                        seen_idx.insert(key, candidates.len());
                        candidates.push((derived, result_tv));
                    } else if let Some(&idx) = seen_idx.get(&key) {
                        // Same fact derived again — keep highest confidence
                        if result_tv.confidence > candidates[idx].1.confidence {
                            candidates[idx].1 = result_tv;
                        }
                    }
                },
            );
        }

        // Filter: only keep truly new facts
        let new_facts: Vec<_> = candidates
            .into_iter()
            .filter(|(neuron, _)| !store.contains(neuron))
            .collect();

        if new_facts.is_empty() {
            break;
        }

        // Track new predicates for delta
        delta_preds.clear();
        for (neuron, _) in &new_facts {
            if let Neuron::Expression(parts) = neuron {
                if let Some(Neuron::Symbol(pred)) = parts.first() {
                    delta_preds.insert(pred.clone());
                }
            }
        }

        for (neuron, tv) in new_facts {
            store.insert_inferred(neuron, tv);
        }

        total_new += store.len() - before;
    }

    total_new
}

// ── Chain Trace (Phase 3B) ───────────────────────────────────────────
//
// Optional trace recording: which rules fired, how many facts each derived.
// Used by the sleep phase to analyze successful reasoning patterns.

/// A single rule firing record.
#[derive(Debug, Clone)]
pub struct RuleFiring {
    pub rule_head: String,
    pub derived_count: usize,
}

/// Trace of a forward chain execution.
#[derive(Debug, Clone, Default)]
pub struct ChainTrace {
    pub firings: Vec<RuleFiring>,
    pub total_derived: usize,
}

/// Forward chain with trace recording.
/// Same as forward_chain but also records which rules fired.
pub fn forward_chain_traced(rules: &[Rule], store: &mut NeuronStore) -> (usize, ChainTrace) {
    let mut trace = ChainTrace::default();

    if rules.is_empty() {
        return (0, trace);
    }

    // Reorder + index (same as forward_chain_stratum)
    let rules: Vec<Rule> = rules.iter().map(|r| {
        Rule {
            head: r.head.clone(),
            body: reorder_body(&r.body),
            tv: r.tv,
            stratum: r.stratum,
        }
    }).collect();

    let max_iterations = 10;
    let mut total_new = 0;

    let mut rule_by_pred: HashMap<String, Vec<usize>> = HashMap::new();
    let mut unindexed: Vec<usize> = Vec::new();

    for (i, rule) in rules.iter().enumerate() {
        let mut indexed = false;
        for cond in &rule.body {
            if let Condition::Positive(neuron) = cond {
                if let Neuron::Expression(parts) = neuron {
                    if let Some(Neuron::Symbol(pred)) = parts.first() {
                        rule_by_pred.entry(pred.clone()).or_default().push(i);
                        indexed = true;
                    }
                }
            }
        }
        if !indexed {
            unindexed.push(i);
        }
    }

    let mut delta_preds: HashSet<String> = HashSet::new();
    let mut per_rule_count: HashMap<usize, usize> = HashMap::new();

    for iter in 0..max_iterations {
        let before = store.len();
        let mut candidates: Vec<(Neuron, TruthValue, usize)> = Vec::new(); // +rule_idx
        let mut seen: HashSet<String> = HashSet::new();

        let to_fire: Vec<usize> = if iter == 0 {
            (0..rules.len()).collect()
        } else {
            let mut set: HashSet<usize> = HashSet::new();
            for pred in &delta_preds {
                if let Some(indices) = rule_by_pred.get(pred) {
                    for &i in indices { set.insert(i); }
                }
            }
            for &i in &unindexed { set.insert(i); }
            set.into_iter().collect()
        };

        if to_fire.is_empty() { break; }

        for &rule_idx in &to_fire {
            let rule = &rules[rule_idx];
            resolve_body_cb(
                &rule.body,
                &Bindings::new(),
                store,
                TruthValue::new(1.0, 1.0),
                &mut |bindings, body_tv| {
                    let derived = apply_bindings(&rule.head, bindings);
                    let result_tv = body_tv.deduction(&rule.tv);
                    let key = derived.to_string();
                    if seen.insert(key) {
                        candidates.push((derived, result_tv, rule_idx));
                    }
                },
            );
        }

        let new_facts: Vec<_> = candidates
            .into_iter()
            .filter(|(neuron, _, _)| !store.contains(neuron))
            .collect();

        if new_facts.is_empty() { break; }

        delta_preds.clear();
        for (neuron, _, rule_idx) in &new_facts {
            if let Neuron::Expression(parts) = neuron {
                if let Some(Neuron::Symbol(pred)) = parts.first() {
                    delta_preds.insert(pred.clone());
                }
            }
            *per_rule_count.entry(*rule_idx).or_insert(0) += 1;
        }

        for (neuron, tv, _) in new_facts {
            store.insert_inferred(neuron, tv);
        }

        total_new += store.len() - before;
    }

    // Build trace
    for (rule_idx, count) in per_rule_count {
        trace.firings.push(RuleFiring {
            rule_head: rules[rule_idx].head.to_string(),
            derived_count: count,
        });
    }
    trace.total_derived = total_new;

    (total_new, trace)
}

// ── Consolidation (belief strengthening through re-reasoning) ───────────
//
// The "heartbeat" of QOR. Even without new data, re-derive existing facts.
// Each cycle, derived facts are re-computed and their TVs are revised.
// Confidence grows toward 1.0 — the system becomes increasingly certain.
// After enough cycles, beliefs are "settled" and reasoning is instant.

/// Consolidate beliefs: apply temporal decay, then re-derive existing facts.
/// Returns true if any TV changed significantly.
///
/// Uses MORK-style callback body resolution for zero intermediate allocations.
pub fn consolidate(rules: &[Rule], store: &mut NeuronStore) -> bool {
    // 1. Temporal decay — facts with @decay lose confidence each cycle
    let mut changed = store.apply_decay();

    // 2. Re-derive facts through rules — strengthens beliefs
    for rule in rules {
        // Collect derivations via callback, then apply (can't mutate store during cb)
        let mut derivations: Vec<(Neuron, TruthValue)> = Vec::new();
        resolve_body_cb(
            &rule.body,
            &Bindings::new(),
            store,
            TruthValue::new(1.0, 1.0),
            &mut |bindings, body_tv| {
                let derived = apply_bindings(&rule.head, bindings);
                let result_tv = body_tv.deduction(&rule.tv);
                derivations.push((derived, result_tv));
            },
        );

        for (derived, result_tv) in derivations {
            // Get current TV before revision (trie-accelerated)
            let old_conf = store
                .query(&derived)
                .first()
                .map(|sn| sn.tv.confidence);

            store.insert_inferred(derived.clone(), result_tv);

            // Check if confidence changed meaningfully
            if let Some(old_c) = old_conf {
                let new_c = store
                    .query(&derived)
                    .first()
                    .map(|sn| sn.tv.confidence)
                    .unwrap_or(0.0);
                if (new_c - old_c).abs() > 0.001 {
                    changed = true;
                }
            }
        }
    }

    changed
}

// ── Backward Chaining (goal-directed reasoning) ─────────────────────────

/// Backward chaining: derive facts matching a query through rules.
pub fn backward_chain(
    query: &Neuron,
    rules: &[Rule],
    store: &NeuronStore,
) -> Vec<StoredNeuron> {
    let mut results = Vec::new();

    for rule in rules {
        if !could_match(&rule.head, query) {
            continue;
        }

        let body_results = resolve_body(&rule.body, &Bindings::new(), store);

        for (bindings, body_tv) in body_results {
            let derived = apply_bindings(&rule.head, &bindings);

            if matches_query(query, &derived) {
                let result_tv = body_tv.deduction(&rule.tv);
                results.push(StoredNeuron {
                    neuron: derived,
                    tv: result_tv,
                    timestamp: None,
                    decay_rate: None,
                    inferred: true,
                });
            }
        }
    }

    results
}

fn could_match(head: &Neuron, query: &Neuron) -> bool {
    match (head, query) {
        (Neuron::Variable(_), _) | (_, Neuron::Variable(_)) => true,
        (Neuron::Symbol(a), Neuron::Symbol(b)) => a == b,
        (Neuron::Expression(a), Neuron::Expression(b)) => {
            a.len() == b.len()
                && a.iter()
                    .zip(b.iter())
                    .all(|(ha, qb)| could_match(ha, qb))
        }
        (Neuron::Value(a), Neuron::Value(b)) => a == b,
        _ => false,
    }
}

fn matches_query(query: &Neuron, derived: &Neuron) -> bool {
    match (query, derived) {
        (Neuron::Variable(_), _) => true,
        (Neuron::Symbol(a), Neuron::Symbol(b)) => a == b,
        (Neuron::Expression(a), Neuron::Expression(b)) => {
            a.len() == b.len()
                && a.iter()
                    .zip(b.iter())
                    .all(|(qa, db)| matches_query(qa, db))
        }
        (Neuron::Value(a), Neuron::Value(b)) => a == b,
        _ => false,
    }
}

// ── MORK-style callback body resolver ────────────────────────────────────
//
// Inspired by MORK's coreferential_transition (space.rs:92-212):
//   - FnMut callback instead of building Vec results at every level
//   - Zero intermediate allocations (no cross-product Vec explosion)
//   - Ground-condition shortcut: O(1) trie lookup for fully-bound conditions
//
// For a rule with N conditions and M matches per condition:
//   OLD: N levels × M intermediate Vecs = O(M^N) Vec allocations
//   NEW: single callback at leaves = O(1) allocation (the callback itself)

fn resolve_body_cb<F>(
    conditions: &[Condition],
    bindings: &Bindings,
    store: &NeuronStore,
    tv: TruthValue,
    cb: &mut F,
)
where
    F: FnMut(&Bindings, TruthValue),
{
    if conditions.is_empty() {
        cb(bindings, tv);
        return;
    }

    match &conditions[0] {
        Condition::Positive(neuron) => {
            let condition = apply_bindings(neuron, bindings);

            // MORK-style ground shortcut: if fully bound, O(1) trie lookup
            // instead of query → iterate → unify (like MORK's direct descend_to)
            if !has_variables(&condition) {
                if let Some(sn) = store.get_exact(&condition) {
                    let combined_tv = tv.and(&sn.tv);
                    resolve_body_cb(&conditions[1..], bindings, store, combined_tv, cb);
                }
                return;
            }

            // Pattern query: trie-accelerated candidate lookup
            let candidates = store.query(&condition);
            for stored in candidates {
                if let Some(new_bindings) = unify(&condition, &stored.neuron) {
                    let mut merged = bindings.clone();
                    let mut conflict = false;
                    for (k, v) in &new_bindings {
                        if let Some(existing) = merged.get(k) {
                            if *existing != *v {
                                conflict = true;
                                break;
                            }
                        } else {
                            merged.insert(k.clone(), v.clone());
                        }
                    }
                    if conflict {
                        continue;
                    }

                    let combined_tv = tv.and(&stored.tv);
                    resolve_body_cb(&conditions[1..], &merged, store, combined_tv, cb);
                }
            }
        }

        Condition::Negated(neuron) => {
            let condition = apply_bindings(neuron, bindings);

            // Ground negation: O(1) contains check
            if !has_variables(&condition) {
                if !store.contains(&condition) {
                    resolve_body_cb(&conditions[1..], bindings, store, tv, cb);
                }
                return;
            }

            // Pattern negation: query + unify
            let candidates = store.query(&condition);
            let has_match = candidates
                .iter()
                .any(|sn| unify(&condition, &sn.neuron).is_some());

            if !has_match {
                resolve_body_cb(&conditions[1..], bindings, store, tv, cb);
            }
        }

        Condition::NegatedPresent(neuron) => {
            let condition = apply_bindings(neuron, bindings);

            // Negation-as-absence: only check non-inferred (base/asserted) facts
            if !has_variables(&condition) {
                let found = store.get_exact(&condition).map_or(false, |sn| !sn.inferred);
                if !found {
                    resolve_body_cb(&conditions[1..], bindings, store, tv, cb);
                }
                return;
            }

            let candidates = store.query(&condition);
            let has_base_match = candidates
                .iter()
                .any(|sn| !sn.inferred && unify(&condition, &sn.neuron).is_some());

            if !has_base_match {
                resolve_body_cb(&conditions[1..], bindings, store, tv, cb);
            }
        }

        Condition::Guard(op, lhs, rhs) => {
            let lhs_bound = apply_bindings(lhs, bindings);
            let rhs_bound = apply_bindings(rhs, bindings);

            match (lhs_bound.as_f64(), rhs_bound.as_f64()) {
                (Some(l), Some(r)) => {
                    let passed = match op {
                        ComparisonOp::Gt => l > r,
                        ComparisonOp::Lt => l < r,
                        ComparisonOp::Ge => l >= r,
                        ComparisonOp::Le => l <= r,
                        ComparisonOp::Eq => (l - r).abs() < f64::EPSILON,
                        ComparisonOp::Ne => (l - r).abs() >= f64::EPSILON,
                    };
                    if passed {
                        resolve_body_cb(&conditions[1..], bindings, store, tv, cb);
                    }
                }
                _ => {} // Can't evaluate — guard fails
            }
        }

        Condition::Aggregate { op, pattern, bind_var, result_var } => {
            let bound_pattern = apply_bindings(pattern, bindings);
            let candidates = store.query(&bound_pattern);

            let result = match op {
                AggregateOp::Count => {
                    candidates.iter()
                        .filter(|sn| unify(&bound_pattern, &sn.neuron).is_some())
                        .count() as f64
                }
                _ => {
                    let values: Vec<f64> = candidates.iter()
                        .filter_map(|sn| {
                            let b = unify(&bound_pattern, &sn.neuron)?;
                            b.get(bind_var)?.as_f64()
                        })
                        .collect();
                    match op {
                        AggregateOp::Sum => values.iter().sum(),
                        AggregateOp::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
                        AggregateOp::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                        AggregateOp::Count => unreachable!(),
                    }
                }
            };

            let result_neuron_agg = Neuron::float_val(result);
            if let Some(existing) = bindings.get(result_var) {
                if *existing != result_neuron_agg {
                    return;
                }
                resolve_body_cb(&conditions[1..], bindings, store, tv, cb);
            } else {
                let mut new_bindings = bindings.clone();
                new_bindings.insert(result_var.clone(), result_neuron_agg);
                resolve_body_cb(&conditions[1..], &new_bindings, store, tv, cb);
            }
        }

        Condition::Arithmetic { op, lhs, rhs, result_var } => {
            let lhs_bound = apply_bindings(lhs, bindings);
            let rhs_bound = apply_bindings(rhs, bindings);

            // Unary ops: only use lhs
            let result_neuron = if neuron::is_unary_arith(op) {
                if let Some(v) = lhs_bound.as_f64() {
                    match op {
                        ArithmeticOp::Sqrt => Neuron::float_val(v.sqrt()),
                        ArithmeticOp::Abs => {
                            if let Neuron::Value(QorValue::Int(a)) = &lhs_bound {
                                Neuron::int_val(a.abs())
                            } else {
                                Neuron::float_val(v.abs())
                            }
                        }
                        ArithmeticOp::DigitSum => {
                            let n = v.abs() as i64;
                            let dsum: i64 = n.to_string().chars()
                                .filter_map(|c| c.to_digit(10))
                                .map(|d| d as i64).sum();
                            Neuron::int_val(dsum)
                        }
                        _ => return,
                    }
                } else { return; }
            } else {
                // Binary ops
                let int_result = match (&lhs_bound, &rhs_bound) {
                    (Neuron::Value(QorValue::Int(a)), Neuron::Value(QorValue::Int(b))) => {
                        match op {
                            ArithmeticOp::Add => a.checked_add(*b).map(Neuron::int_val),
                            ArithmeticOp::Sub => a.checked_sub(*b).map(Neuron::int_val),
                            ArithmeticOp::Mul => a.checked_mul(*b).map(Neuron::int_val),
                            ArithmeticOp::Div => if *b != 0 { a.checked_div(*b).map(Neuron::int_val) } else { None },
                            ArithmeticOp::Mod => if *b != 0 { Some(Neuron::int_val(a % b)) } else { None },
                            ArithmeticOp::Min => Some(Neuron::int_val(*a.min(b))),
                            ArithmeticOp::Max => Some(Neuron::int_val(*a.max(b))),
                            ArithmeticOp::Power => {
                                if *b >= 0 && *b <= 63 { Some(Neuron::int_val(a.pow(*b as u32))) } else { None }
                            }
                            _ => None, // unary ops handled above
                        }
                    }
                    _ => None,
                };
                if let Some(n) = int_result {
                    n
                } else {
                    match (lhs_bound.as_f64(), rhs_bound.as_f64()) {
                        (Some(l), Some(r)) => {
                            let result = match op {
                                ArithmeticOp::Add => l + r,
                                ArithmeticOp::Sub => l - r,
                                ArithmeticOp::Mul => l * r,
                                ArithmeticOp::Div => if r.abs() < f64::EPSILON { return; } else { l / r },
                                ArithmeticOp::Mod => if r.abs() < f64::EPSILON { return; } else { l % r },
                                ArithmeticOp::Min => l.min(r),
                                ArithmeticOp::Max => l.max(r),
                                ArithmeticOp::Power => l.powf(r),
                                _ => return,
                            };
                            Neuron::float_val(result)
                        }
                        _ => return,
                    }
                }
            };

            // If result_var is already bound (e.g. due to body reordering),
            // verify computed value matches — don't silently overwrite.
            if let Some(existing) = bindings.get(result_var) {
                if *existing != result_neuron {
                    return; // mismatch — discard this derivation path
                }
                // Already bound to correct value — proceed without cloning
                resolve_body_cb(&conditions[1..], bindings, store, tv, cb);
            } else {
                let mut new_bindings = bindings.clone();
                new_bindings.insert(result_var.clone(), result_neuron);
                resolve_body_cb(&conditions[1..], &new_bindings, store, tv, cb);
            }
        }

        Condition::EndsWith(num, digit) => {
            let num_bound = apply_bindings(num, bindings);
            let digit_bound = apply_bindings(digit, bindings);
            if let (Some(n), Some(d)) = (num_bound.as_f64(), digit_bound.as_f64()) {
                let last = (n.abs() as i64) % 10;
                if last == d as i64 {
                    resolve_body_cb(&conditions[1..], bindings, store, tv, cb);
                }
            }
        }

        Condition::Lookup { predicate, subject, result_var } => {
            // KB lookup: resolve by querying the NeuronStore for pre-materialized KB facts.
            // Users call session.load_entity() to pull KB facts into the store first.
            // We search for (predicate subject ?) patterns in the store.
            let subj_bound = apply_bindings(subject, bindings);
            let pattern = Neuron::Expression(vec![
                Neuron::Symbol(predicate.clone()),
                subj_bound,
                Neuron::Variable(result_var.clone()),
            ]);

            for stored in store.query(&pattern) {
                if let Some(new_b) = unify(&pattern, &stored.neuron) {
                    let mut merged = bindings.clone();
                    let mut conflict = false;
                    for (k, v) in &new_b {
                        if let Some(existing) = merged.get(k) {
                            if *existing != *v { conflict = true; break; }
                        }
                        merged.insert(k.clone(), v.clone());
                    }
                    if !conflict {
                        let combined_tv = tv.and(&stored.tv);
                        resolve_body_cb(&conditions[1..], &merged, store, combined_tv, cb);
                    }
                }
            }
        }
    }
}

// ── Thin wrapper for backward_chain / explain / tests ────────────────────
//
// resolve_body returns Vec like before, but internally uses resolve_body_cb.
// Only backward_chain and tests call this — forward_chain uses resolve_body_cb directly.

fn resolve_body(
    conditions: &[Condition],
    initial_bindings: &Bindings,
    store: &NeuronStore,
) -> Vec<(Bindings, TruthValue)> {
    let mut results = Vec::new();
    resolve_body_cb(conditions, initial_bindings, store, TruthValue::new(1.0, 1.0), &mut |bindings, tv| {
        results.push((bindings.clone(), tv));
    });
    results
}

// ── Explain support ───────────────────────────────────────────────────
//
// Given a rule body and bindings, find the actual store facts that satisfy each
// condition. Used by `Session::explain()` to trace WHY a fact is believed.

/// Resolve a rule body and return the supporting facts for each positive condition.
/// Returns None if the body can't be satisfied.
/// Each entry: (matched_neuron, truth_value, is_inferred)
pub fn resolve_body_for_explain(
    conditions: &[Condition],
    bindings: &Bindings,
    store: &NeuronStore,
) -> Option<Vec<(Neuron, TruthValue, bool)>> {
    let mut supporting = Vec::new();
    let mut current_bindings = bindings.clone();

    for condition in conditions {
        match condition {
            Condition::Positive(neuron) => {
                let bound = apply_bindings(neuron, &current_bindings);
                // Trie-accelerated: find a matching fact in the store
                let candidates = store.query(&bound);
                let matched = candidates.iter().find(|sn| {
                    if let Some(new_b) = unify(&bound, &sn.neuron) {
                        // Check for binding conflicts
                        for (k, v) in &new_b {
                            if let Some(existing) = current_bindings.get(k) {
                                if *existing != *v {
                                    return false;
                                }
                            }
                        }
                        true
                    } else {
                        false
                    }
                });

                if let Some(sn) = matched {
                    if let Some(new_b) = unify(&bound, &sn.neuron) {
                        for (k, v) in new_b {
                            current_bindings.entry(k).or_insert(v);
                        }
                    }
                    supporting.push((sn.neuron.clone(), sn.tv, sn.inferred));
                } else {
                    return None; // Body condition unsatisfied
                }
            }
            Condition::Guard(op, lhs, rhs) => {
                let lhs_bound = apply_bindings(lhs, &current_bindings);
                let rhs_bound = apply_bindings(rhs, &current_bindings);
                match (lhs_bound.as_f64(), rhs_bound.as_f64()) {
                    (Some(l), Some(r)) => {
                        let passed = match op {
                            ComparisonOp::Gt => l > r,
                            ComparisonOp::Lt => l < r,
                            ComparisonOp::Ge => l >= r,
                            ComparisonOp::Le => l <= r,
                            ComparisonOp::Eq => (l - r).abs() < f64::EPSILON,
                            ComparisonOp::Ne => (l - r).abs() >= f64::EPSILON,
                        };
                        if !passed {
                            return None;
                        }
                        // Guards don't produce supporting facts — they're constraints
                    }
                    _ => return None,
                }
            }
            Condition::Negated(_) | Condition::NegatedPresent(_) => {
                // Negation is a constraint, not evidence — skip for explanation
            }
            Condition::Aggregate { .. } => {
                // Aggregate is a computation, not evidence — skip for explanation
            }
            Condition::Arithmetic { op, lhs, rhs, result_var } => {
                // Arithmetic is a computation — evaluate and bind for downstream conditions
                let lhs_bound = apply_bindings(lhs, &current_bindings);
                let rhs_bound = apply_bindings(rhs, &current_bindings);

                // Unary ops: only use lhs
                let result_neuron = if neuron::is_unary_arith(op) {
                    if let Some(v) = lhs_bound.as_f64() {
                        match op {
                            ArithmeticOp::Sqrt => Neuron::float_val(v.sqrt()),
                            ArithmeticOp::Abs => {
                                if let Neuron::Value(QorValue::Int(a)) = &lhs_bound {
                                    Neuron::int_val(a.abs())
                                } else {
                                    Neuron::float_val(v.abs())
                                }
                            }
                            ArithmeticOp::DigitSum => {
                                let n = v.abs() as i64;
                                let dsum: i64 = n.to_string().chars()
                                    .filter_map(|c| c.to_digit(10))
                                    .map(|d| d as i64).sum();
                                Neuron::int_val(dsum)
                            }
                            _ => return None,
                        }
                    } else { return None; }
                } else {
                    // Binary ops
                    let int_result = match (&lhs_bound, &rhs_bound) {
                        (Neuron::Value(QorValue::Int(a)), Neuron::Value(QorValue::Int(b))) => {
                            match op {
                                ArithmeticOp::Add => a.checked_add(*b).map(Neuron::int_val),
                                ArithmeticOp::Sub => a.checked_sub(*b).map(Neuron::int_val),
                                ArithmeticOp::Mul => a.checked_mul(*b).map(Neuron::int_val),
                                ArithmeticOp::Div => if *b != 0 { a.checked_div(*b).map(Neuron::int_val) } else { None },
                                ArithmeticOp::Mod => if *b != 0 { Some(Neuron::int_val(a % b)) } else { None },
                                ArithmeticOp::Min => Some(Neuron::int_val(*a.min(b))),
                                ArithmeticOp::Max => Some(Neuron::int_val(*a.max(b))),
                                ArithmeticOp::Power => {
                                    if *b >= 0 && *b <= 63 { Some(Neuron::int_val(a.pow(*b as u32))) } else { None }
                                }
                                _ => None,
                            }
                        }
                        _ => None,
                    };
                    if let Some(n) = int_result {
                        n
                    } else if let (Some(l), Some(r)) = (lhs_bound.as_f64(), rhs_bound.as_f64()) {
                        let result = match op {
                            ArithmeticOp::Add => l + r,
                            ArithmeticOp::Sub => l - r,
                            ArithmeticOp::Mul => l * r,
                            ArithmeticOp::Div => if r.abs() < f64::EPSILON { return None; } else { l / r },
                            ArithmeticOp::Mod => if r.abs() < f64::EPSILON { return None; } else { l % r },
                            ArithmeticOp::Min => l.min(r),
                            ArithmeticOp::Max => l.max(r),
                            ArithmeticOp::Power => l.powf(r),
                            _ => return None,
                        };
                        Neuron::float_val(result)
                    } else {
                        return None;
                    }
                };
                current_bindings.insert(result_var.clone(), result_neuron);
            }
            Condition::EndsWith(num, digit) => {
                let num_bound = apply_bindings(num, &current_bindings);
                let digit_bound = apply_bindings(digit, &current_bindings);
                if let (Some(n), Some(d)) = (num_bound.as_f64(), digit_bound.as_f64()) {
                    let last = (n.abs() as i64) % 10;
                    if last != d as i64 {
                        return None;
                    }
                } else {
                    return None;
                }
            }
            Condition::Lookup { .. } => {
                // Lookup is resolved like Positive in explain — skip for now
            }
        }
    }

    Some(supporting)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> NeuronStore {
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
        store
    }

    // ── Backward chaining ──

    #[test]
    fn test_backward_chain_single_rule() {
        let store = make_store();
        let rules = vec![Rule {
            head: Neuron::expression(vec![Neuron::symbol("flies"), Neuron::variable("x")]),
            body: vec![Condition::Positive(Neuron::expression(vec![
                Neuron::symbol("bird"),
                Neuron::variable("x"),
            ]))],
            tv: TruthValue::from_strength(0.95),
            stratum: 0,
        }];

        let query = Neuron::expression(vec![Neuron::symbol("flies"), Neuron::variable("y")]);
        let results = backward_chain(&query, &rules, &store);
        assert_eq!(results.len(), 2);
        assert!(results[0].inferred);
    }

    #[test]
    fn test_backward_chain_concrete_query() {
        let store = make_store();
        let rules = vec![Rule {
            head: Neuron::expression(vec![Neuron::symbol("flies"), Neuron::variable("x")]),
            body: vec![Condition::Positive(Neuron::expression(vec![
                Neuron::symbol("bird"),
                Neuron::variable("x"),
            ]))],
            tv: TruthValue::from_strength(0.95),
            stratum: 0,
        }];

        let query = Neuron::expression(vec![
            Neuron::symbol("flies"),
            Neuron::symbol("tweety"),
        ]);
        let results = backward_chain(&query, &rules, &store);
        assert_eq!(results.len(), 1);
        assert!(results[0].tv.strength > 0.93 && results[0].tv.strength < 0.95);
    }

    #[test]
    fn test_backward_chain_no_match() {
        let store = make_store();
        let rules = vec![Rule {
            head: Neuron::expression(vec![Neuron::symbol("flies"), Neuron::variable("x")]),
            body: vec![Condition::Positive(Neuron::expression(vec![
                Neuron::symbol("bird"),
                Neuron::variable("x"),
            ]))],
            tv: TruthValue::from_strength(0.95),
            stratum: 0,
        }];

        let query = Neuron::expression(vec![Neuron::symbol("swims"), Neuron::variable("y")]);
        assert_eq!(backward_chain(&query, &rules, &store).len(), 0);
    }

    #[test]
    fn test_backward_chain_multi_condition() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("healthy"), Neuron::symbol("tweety")]),
            TruthValue::from_strength(0.90),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("eagle")]),
            TruthValue::from_strength(0.95),
        );

        let rules = vec![Rule {
            head: Neuron::expression(vec![Neuron::symbol("can-fly"), Neuron::variable("x")]),
            body: vec![
                Condition::Positive(Neuron::expression(vec![
                    Neuron::symbol("bird"),
                    Neuron::variable("x"),
                ])),
                Condition::Positive(Neuron::expression(vec![
                    Neuron::symbol("healthy"),
                    Neuron::variable("x"),
                ])),
            ],
            tv: TruthValue::from_strength(0.90),
            stratum: 0,
        }];

        let query = Neuron::expression(vec![
            Neuron::symbol("can-fly"),
            Neuron::variable("y"),
        ]);
        let results = backward_chain(&query, &rules, &store);
        assert_eq!(results.len(), 1);
        assert!(results[0].neuron.to_string().contains("tweety"));
    }

    #[test]
    fn test_truth_value_propagation() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]),
            TruthValue::new(0.99, 0.90),
        );

        let rules = vec![Rule {
            head: Neuron::expression(vec![Neuron::symbol("flies"), Neuron::variable("x")]),
            body: vec![Condition::Positive(Neuron::expression(vec![
                Neuron::symbol("bird"),
                Neuron::variable("x"),
            ]))],
            tv: TruthValue::new(0.95, 0.90),
            stratum: 0,
        }];

        let query = Neuron::expression(vec![
            Neuron::symbol("flies"),
            Neuron::symbol("tweety"),
        ]);
        let results = backward_chain(&query, &rules, &store);
        let tv = results[0].tv;
        assert!((tv.strength - 0.9405).abs() < 0.01);
        assert!((tv.confidence - 0.729).abs() < 0.01);
    }

    // ── Forward chaining ──

    #[test]
    fn test_forward_chain_basic() {
        let mut store = make_store();
        let rules = vec![Rule {
            head: Neuron::expression(vec![Neuron::symbol("flies"), Neuron::variable("x")]),
            body: vec![Condition::Positive(Neuron::expression(vec![
                Neuron::symbol("bird"),
                Neuron::variable("x"),
            ]))],
            tv: TruthValue::from_strength(0.95),
            stratum: 0,
        }];

        let new_count = forward_chain(&rules, &mut store);
        assert_eq!(new_count, 2);
        assert_eq!(store.len(), 5);
    }

    #[test]
    fn test_forward_chain_rule_chaining() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]),
            TruthValue::from_strength(0.99),
        );

        let rules = vec![
            Rule {
                head: Neuron::expression(vec![Neuron::symbol("flies"), Neuron::variable("x")]),
                body: vec![Condition::Positive(Neuron::expression(vec![
                    Neuron::symbol("bird"),
                    Neuron::variable("x"),
                ]))],
                tv: TruthValue::from_strength(0.95),
                stratum: 0,
            },
            Rule {
                head: Neuron::expression(vec![
                    Neuron::symbol("has-wings"),
                    Neuron::variable("x"),
                ]),
                body: vec![Condition::Positive(Neuron::expression(vec![
                    Neuron::symbol("flies"),
                    Neuron::variable("x"),
                ]))],
                tv: TruthValue::from_strength(0.99),
                stratum: 0,
            },
        ];

        let new_count = forward_chain(&rules, &mut store);
        assert_eq!(new_count, 2);

        let results = store.query(&Neuron::expression(vec![
            Neuron::symbol("has-wings"),
            Neuron::symbol("tweety"),
        ]));
        assert_eq!(results.len(), 1);
        assert!(results[0].inferred);
        assert!(results[0].tv.confidence < 0.729);
    }

    #[test]
    fn test_forward_chain_fixed_point() {
        let mut store = make_store();
        let rules = vec![Rule {
            head: Neuron::expression(vec![Neuron::symbol("flies"), Neuron::variable("x")]),
            body: vec![Condition::Positive(Neuron::expression(vec![
                Neuron::symbol("bird"),
                Neuron::variable("x"),
            ]))],
            tv: TruthValue::from_strength(0.95),
            stratum: 0,
        }];

        forward_chain(&rules, &mut store);
        let n2 = forward_chain(&rules, &mut store);
        assert_eq!(n2, 0);
    }

    // ── Negation ──

    #[test]
    fn test_negation_in_rule() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tux")]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("penguin"), Neuron::symbol("tux")]),
            TruthValue::from_strength(0.99),
        );

        // birds fly unless they are penguins
        let rules = vec![Rule {
            head: Neuron::expression(vec![Neuron::symbol("flies"), Neuron::variable("x")]),
            body: vec![
                Condition::Positive(Neuron::expression(vec![
                    Neuron::symbol("bird"),
                    Neuron::variable("x"),
                ])),
                Condition::Negated(Neuron::expression(vec![
                    Neuron::symbol("penguin"),
                    Neuron::variable("x"),
                ])),
            ],
            tv: TruthValue::from_strength(0.95),
            stratum: 0,
        }];

        let new_count = forward_chain(&rules, &mut store);
        // Only tweety should fly (tux is a penguin)
        assert_eq!(new_count, 1);

        assert!(store.contains(&Neuron::expression(vec![
            Neuron::symbol("flies"),
            Neuron::symbol("tweety"),
        ])));
        assert!(!store.contains(&Neuron::expression(vec![
            Neuron::symbol("flies"),
            Neuron::symbol("tux"),
        ])));
    }

    // ── Consolidation ──

    #[test]
    fn test_consolidate_strengthens_beliefs() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]),
            TruthValue::from_strength(0.99),
        );

        let rules = vec![Rule {
            head: Neuron::expression(vec![Neuron::symbol("flies"), Neuron::variable("x")]),
            body: vec![Condition::Positive(Neuron::expression(vec![
                Neuron::symbol("bird"),
                Neuron::variable("x"),
            ]))],
            tv: TruthValue::from_strength(0.95),
            stratum: 0,
        }];

        // First: forward chain to derive (flies tweety)
        forward_chain(&rules, &mut store);
        let initial_conf = store
            .query(&Neuron::expression(vec![
                Neuron::symbol("flies"),
                Neuron::symbol("tweety"),
            ]))[0]
            .tv
            .confidence;

        // Consolidate: re-reason multiple times
        for _ in 0..10 {
            consolidate(&rules, &mut store);
        }

        let final_conf = store
            .query(&Neuron::expression(vec![
                Neuron::symbol("flies"),
                Neuron::symbol("tweety"),
            ]))[0]
            .tv
            .confidence;

        // Confidence should have grown through repeated reasoning
        assert!(final_conf > initial_conf);
    }

    #[test]
    fn test_consolidate_converges() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]),
            TruthValue::from_strength(0.99),
        );

        let rules = vec![Rule {
            head: Neuron::expression(vec![Neuron::symbol("flies"), Neuron::variable("x")]),
            body: vec![Condition::Positive(Neuron::expression(vec![
                Neuron::symbol("bird"),
                Neuron::variable("x"),
            ]))],
            tv: TruthValue::from_strength(0.95),
            stratum: 0,
        }];

        forward_chain(&rules, &mut store);

        // Consolidate 100 times — should approach 1.0
        for _ in 0..100 {
            consolidate(&rules, &mut store);
        }

        let conf = store
            .query(&Neuron::expression(vec![
                Neuron::symbol("flies"),
                Neuron::symbol("tweety"),
            ]))[0]
            .tv
            .confidence;

        assert!(conf > 0.99); // settled — "master" level
    }

    #[test]
    fn test_forward_chain_no_rules() {
        let mut store = make_store();
        let n = forward_chain(&[], &mut store);
        assert_eq!(n, 0);
    }

    // ── Guard conditions ──

    #[test]
    fn test_guard_gt_filters() {
        let mut store = NeuronStore::new();
        // rsi values for different entities
        store.insert(
            Neuron::expression(vec![Neuron::symbol("rsi"), Neuron::symbol("btc"), Neuron::float_val(75.0)]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("rsi"), Neuron::symbol("eth"), Neuron::float_val(45.0)]),
            TruthValue::from_strength(0.99),
        );

        // (overbought $x) if (rsi $x $v) (> $v 70)
        let rules = vec![Rule {
            head: Neuron::expression(vec![Neuron::symbol("overbought"), Neuron::variable("x")]),
            body: vec![
                Condition::Positive(Neuron::expression(vec![
                    Neuron::symbol("rsi"), Neuron::variable("x"), Neuron::variable("v"),
                ])),
                Condition::Guard(ComparisonOp::Gt, Neuron::variable("v"), Neuron::float_val(70.0)),
            ],
            tv: TruthValue::from_strength(0.90),
            stratum: 0,
        }];

        let derived = forward_chain(&rules, &mut store);
        assert_eq!(derived, 1); // only btc (75 > 70)
        assert!(store.contains(&Neuron::expression(vec![
            Neuron::symbol("overbought"), Neuron::symbol("btc"),
        ])));
        assert!(!store.contains(&Neuron::expression(vec![
            Neuron::symbol("overbought"), Neuron::symbol("eth"),
        ])));
    }

    #[test]
    fn test_guard_lt_filters() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("rsi"), Neuron::symbol("btc"), Neuron::float_val(25.0)]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("rsi"), Neuron::symbol("eth"), Neuron::float_val(55.0)]),
            TruthValue::from_strength(0.99),
        );

        // (oversold $x) if (rsi $x $v) (< $v 30)
        let rules = vec![Rule {
            head: Neuron::expression(vec![Neuron::symbol("oversold"), Neuron::variable("x")]),
            body: vec![
                Condition::Positive(Neuron::expression(vec![
                    Neuron::symbol("rsi"), Neuron::variable("x"), Neuron::variable("v"),
                ])),
                Condition::Guard(ComparisonOp::Lt, Neuron::variable("v"), Neuron::float_val(30.0)),
            ],
            tv: TruthValue::from_strength(0.90),
            stratum: 0,
        }];

        let derived = forward_chain(&rules, &mut store);
        assert_eq!(derived, 1); // only btc (25 < 30)
    }

    #[test]
    fn test_guard_with_integers() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("score"), Neuron::symbol("alice"), Neuron::int_val(95)]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("score"), Neuron::symbol("bob"), Neuron::int_val(60)]),
            TruthValue::from_strength(0.99),
        );

        // (top-scorer $x) if (score $x $v) (>= $v 90)
        let rules = vec![Rule {
            head: Neuron::expression(vec![Neuron::symbol("top-scorer"), Neuron::variable("x")]),
            body: vec![
                Condition::Positive(Neuron::expression(vec![
                    Neuron::symbol("score"), Neuron::variable("x"), Neuron::variable("v"),
                ])),
                Condition::Guard(ComparisonOp::Ge, Neuron::variable("v"), Neuron::int_val(90)),
            ],
            tv: TruthValue::from_strength(0.95),
            stratum: 0,
        }];

        let derived = forward_chain(&rules, &mut store);
        assert_eq!(derived, 1); // only alice (95 >= 90)
    }

    // ── Aggregate conditions ──

    #[test]
    fn test_aggregate_count_in_chain() {
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
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("robin")]),
            TruthValue::from_strength(0.90),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("fish"), Neuron::symbol("salmon")]),
            TruthValue::from_strength(0.99),
        );

        // (bird-count $n) if (count (bird $x) $x -> $n)
        let rules = vec![Rule {
            head: Neuron::expression(vec![Neuron::symbol("bird-count"), Neuron::variable("n")]),
            body: vec![Condition::Aggregate {
                op: AggregateOp::Count,
                pattern: Neuron::expression(vec![Neuron::symbol("bird"), Neuron::variable("x")]),
                bind_var: "x".into(),
                result_var: "n".into(),
            }],
            tv: TruthValue::from_strength(0.99),
            stratum: 0,
        }];

        let derived = forward_chain(&rules, &mut store);
        assert_eq!(derived, 1);
        // Should have (bird-count 3.0)
        assert!(store.contains(&Neuron::expression(vec![
            Neuron::symbol("bird-count"),
            Neuron::float_val(3.0),
        ])));
    }

    #[test]
    fn test_aggregate_sum_in_chain() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("score"), Neuron::symbol("alice"), Neuron::int_val(90)]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("score"), Neuron::symbol("bob"), Neuron::int_val(80)]),
            TruthValue::from_strength(0.99),
        );

        let rules = vec![Rule {
            head: Neuron::expression(vec![Neuron::symbol("total-score"), Neuron::variable("t")]),
            body: vec![Condition::Aggregate {
                op: AggregateOp::Sum,
                pattern: Neuron::expression(vec![
                    Neuron::symbol("score"), Neuron::variable("x"), Neuron::variable("v"),
                ]),
                bind_var: "v".into(),
                result_var: "t".into(),
            }],
            tv: TruthValue::from_strength(0.99),
            stratum: 0,
        }];

        forward_chain(&rules, &mut store);
        assert!(store.contains(&Neuron::expression(vec![
            Neuron::symbol("total-score"),
            Neuron::float_val(170.0),
        ])));
    }

    #[test]
    fn test_aggregate_min_max() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("temp"), Neuron::symbol("a"), Neuron::float_val(20.5)]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("temp"), Neuron::symbol("b"), Neuron::float_val(35.0)]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("temp"), Neuron::symbol("c"), Neuron::float_val(15.2)]),
            TruthValue::from_strength(0.99),
        );

        let rules = vec![
            Rule {
                head: Neuron::expression(vec![Neuron::symbol("min-temp"), Neuron::variable("m")]),
                body: vec![Condition::Aggregate {
                    op: AggregateOp::Min,
                    pattern: Neuron::expression(vec![
                        Neuron::symbol("temp"), Neuron::variable("x"), Neuron::variable("v"),
                    ]),
                    bind_var: "v".into(),
                    result_var: "m".into(),
                }],
                tv: TruthValue::from_strength(0.99),
                stratum: 0,
            },
            Rule {
                head: Neuron::expression(vec![Neuron::symbol("max-temp"), Neuron::variable("m")]),
                body: vec![Condition::Aggregate {
                    op: AggregateOp::Max,
                    pattern: Neuron::expression(vec![
                        Neuron::symbol("temp"), Neuron::variable("x"), Neuron::variable("v"),
                    ]),
                    bind_var: "v".into(),
                    result_var: "m".into(),
                }],
                tv: TruthValue::from_strength(0.99),
                stratum: 0,
            },
        ];

        forward_chain(&rules, &mut store);
        assert!(store.contains(&Neuron::expression(vec![
            Neuron::symbol("min-temp"),
            Neuron::float_val(15.2),
        ])));
        assert!(store.contains(&Neuron::expression(vec![
            Neuron::symbol("max-temp"),
            Neuron::float_val(35.0),
        ])));
    }

    #[test]
    fn test_aggregate_with_guard() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("a")]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("b")]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("c")]),
            TruthValue::from_strength(0.99),
        );

        // (many-birds) if (count (bird $x) $x -> $n) (> $n 2)
        let rules = vec![Rule {
            head: Neuron::expression(vec![Neuron::symbol("many-birds")]),
            body: vec![
                Condition::Aggregate {
                    op: AggregateOp::Count,
                    pattern: Neuron::expression(vec![Neuron::symbol("bird"), Neuron::variable("x")]),
                    bind_var: "x".into(),
                    result_var: "n".into(),
                },
                Condition::Guard(ComparisonOp::Gt, Neuron::variable("n"), Neuron::float_val(2.0)),
            ],
            tv: TruthValue::from_strength(0.99),
            stratum: 0,
        }];

        let derived = forward_chain(&rules, &mut store);
        assert_eq!(derived, 1);
        assert!(store.contains(&Neuron::expression(vec![Neuron::symbol("many-birds")])));
    }

    #[test]
    fn test_aggregate_no_matches() {
        let store = NeuronStore::new(); // empty store

        let results = resolve_body(
            &[Condition::Aggregate {
                op: AggregateOp::Count,
                pattern: Neuron::expression(vec![Neuron::symbol("bird"), Neuron::variable("x")]),
                bind_var: "x".into(),
                result_var: "n".into(),
            }],
            &Bindings::new(),
            &store,
        );

        assert_eq!(results.len(), 1);
        // n should be 0.0
        let (bindings, _) = &results[0];
        assert_eq!(bindings.get("n").and_then(|n| n.as_f64()), Some(0.0));
    }

    #[test]
    fn test_aggregate_with_bindings() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![
                Neuron::symbol("grade"), Neuron::symbol("math"), Neuron::symbol("alice"), Neuron::int_val(90),
            ]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![
                Neuron::symbol("grade"), Neuron::symbol("math"), Neuron::symbol("bob"), Neuron::int_val(80),
            ]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![
                Neuron::symbol("grade"), Neuron::symbol("science"), Neuron::symbol("alice"), Neuron::int_val(95),
            ]),
            TruthValue::from_strength(0.99),
        );

        // Count grades where subject is already bound to "math"
        let mut bindings = Bindings::new();
        bindings.insert("subj".into(), Neuron::symbol("math"));

        let results = resolve_body(
            &[Condition::Aggregate {
                op: AggregateOp::Count,
                pattern: Neuron::expression(vec![
                    Neuron::symbol("grade"), Neuron::variable("subj"), Neuron::variable("who"), Neuron::variable("v"),
                ]),
                bind_var: "v".into(),
                result_var: "n".into(),
            }],
            &bindings,
            &store,
        );

        assert_eq!(results.len(), 1);
        // Only 2 math grades, not the science one
        assert_eq!(results[0].0.get("n").and_then(|n| n.as_f64()), Some(2.0));
    }

    // ── Arithmetic conditions ──

    #[test]
    fn test_arithmetic_add_integers() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("pos"), Neuron::symbol("a"), Neuron::int_val(3)]),
            TruthValue::from_strength(0.99),
        );

        // (shifted $name $nr) if (pos $name $r) (+ $r 1 $nr)
        let rules = vec![Rule {
            head: Neuron::expression(vec![Neuron::symbol("shifted"), Neuron::variable("name"), Neuron::variable("nr")]),
            body: vec![
                Condition::Positive(Neuron::expression(vec![
                    Neuron::symbol("pos"), Neuron::variable("name"), Neuron::variable("r"),
                ])),
                Condition::Arithmetic {
                    op: ArithmeticOp::Add,
                    lhs: Neuron::variable("r"),
                    rhs: Neuron::int_val(1),
                    result_var: "nr".into(),
                },
            ],
            tv: TruthValue::from_strength(0.95),
            stratum: 0,
        }];

        let derived = forward_chain(&rules, &mut store);
        assert_eq!(derived, 1);
        // (shifted a 4) — integer preserved
        assert!(store.contains(&Neuron::expression(vec![
            Neuron::symbol("shifted"), Neuron::symbol("a"), Neuron::int_val(4),
        ])));
    }

    #[test]
    fn test_arithmetic_sub_integers() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("cell"), Neuron::int_val(5), Neuron::int_val(3)]),
            TruthValue::from_strength(0.99),
        );

        // (offset $nr $nc) if (cell $r $c) (- $r 2 $nr) (- $c 1 $nc)
        let rules = vec![Rule {
            head: Neuron::expression(vec![Neuron::symbol("offset"), Neuron::variable("nr"), Neuron::variable("nc")]),
            body: vec![
                Condition::Positive(Neuron::expression(vec![
                    Neuron::symbol("cell"), Neuron::variable("r"), Neuron::variable("c"),
                ])),
                Condition::Arithmetic {
                    op: ArithmeticOp::Sub,
                    lhs: Neuron::variable("r"),
                    rhs: Neuron::int_val(2),
                    result_var: "nr".into(),
                },
                Condition::Arithmetic {
                    op: ArithmeticOp::Sub,
                    lhs: Neuron::variable("c"),
                    rhs: Neuron::int_val(1),
                    result_var: "nc".into(),
                },
            ],
            tv: TruthValue::from_strength(0.95),
            stratum: 0,
        }];

        forward_chain(&rules, &mut store);
        // (offset 3 2) — 5-2=3, 3-1=2
        assert!(store.contains(&Neuron::expression(vec![
            Neuron::symbol("offset"), Neuron::int_val(3), Neuron::int_val(2),
        ])));
    }

    #[test]
    fn test_arithmetic_with_guard() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("pos"), Neuron::int_val(2)]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("pos"), Neuron::int_val(0)]),
            TruthValue::from_strength(0.99),
        );

        // (valid $nr) if (pos $r) (- $r 1 $nr) (>= $nr 0)
        let rules = vec![Rule {
            head: Neuron::expression(vec![Neuron::symbol("valid"), Neuron::variable("nr")]),
            body: vec![
                Condition::Positive(Neuron::expression(vec![
                    Neuron::symbol("pos"), Neuron::variable("r"),
                ])),
                Condition::Arithmetic {
                    op: ArithmeticOp::Sub,
                    lhs: Neuron::variable("r"),
                    rhs: Neuron::int_val(1),
                    result_var: "nr".into(),
                },
                Condition::Guard(ComparisonOp::Ge, Neuron::variable("nr"), Neuron::int_val(0)),
            ],
            tv: TruthValue::from_strength(0.95),
            stratum: 0,
        }];

        let derived = forward_chain(&rules, &mut store);
        assert_eq!(derived, 1); // only pos(2) → valid(1), pos(0) → nr=-1 fails guard
        assert!(store.contains(&Neuron::expression(vec![
            Neuron::symbol("valid"), Neuron::int_val(1),
        ])));
        assert!(!store.contains(&Neuron::expression(vec![
            Neuron::symbol("valid"), Neuron::int_val(-1),
        ])));
    }

    #[test]
    fn test_arithmetic_mul_float() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("scale"), Neuron::float_val(2.5)]),
            TruthValue::from_strength(0.99),
        );
        store.insert(
            Neuron::expression(vec![Neuron::symbol("val"), Neuron::int_val(4)]),
            TruthValue::from_strength(0.99),
        );

        // (result $r) if (scale $s) (val $v) (* $v $s $r)
        let rules = vec![Rule {
            head: Neuron::expression(vec![Neuron::symbol("result"), Neuron::variable("r")]),
            body: vec![
                Condition::Positive(Neuron::expression(vec![
                    Neuron::symbol("scale"), Neuron::variable("s"),
                ])),
                Condition::Positive(Neuron::expression(vec![
                    Neuron::symbol("val"), Neuron::variable("v"),
                ])),
                Condition::Arithmetic {
                    op: ArithmeticOp::Mul,
                    lhs: Neuron::variable("v"),
                    rhs: Neuron::variable("s"),
                    result_var: "r".into(),
                },
            ],
            tv: TruthValue::from_strength(0.95),
            stratum: 0,
        }];

        forward_chain(&rules, &mut store);
        // 4 * 2.5 = 10.0
        assert!(store.contains(&Neuron::expression(vec![
            Neuron::symbol("result"), Neuron::float_val(10.0),
        ])));
    }
}
