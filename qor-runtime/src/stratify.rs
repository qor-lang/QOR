// ── Kahn's Topological Sort for Auto-Stratification ─────────────────────
//
// Problem: forward_chain runs all rules in stratum 0 by default. When a rule
// negates a DERIVED predicate, both the positive derivation and the negation
// compute against the same store snapshot → race condition.
//
// Solution: build a DAG of negation dependencies between predicates, then use
// Kahn's topological sort to assign strata. Rules deriving predicate P run in
// an earlier stratum than rules negating P, guaranteeing P reaches fixed-point
// before negation checks it.
//
// Only negation of DERIVED predicates (predicates that appear as some rule's
// head) creates a cross-stratum dependency. Negating BASE predicates (asserted
// directly, never derived) is safe within any stratum.

use std::collections::{HashMap, HashSet, VecDeque};

use qor_core::neuron::{Condition, Neuron};
use crate::chain::Rule;

/// Error when rules contain a negation cycle (unstratifiable program).
#[derive(Debug, Clone)]
pub struct StratifyError {
    pub cycle_preds: Vec<String>,
}

impl std::fmt::Display for StratifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unstratifiable: negation cycle among {:?}", self.cycle_preds)
    }
}

/// Extract the head predicate name from a rule (first symbol of head expression).
fn head_predicate(rule: &Rule) -> Option<String> {
    if let Neuron::Expression(parts) = &rule.head {
        if let Some(Neuron::Symbol(pred)) = parts.first() {
            return Some(pred.clone());
        }
    }
    None
}

/// Extract predicates from positive body conditions.
fn positive_predicates(rule: &Rule) -> Vec<String> {
    let mut preds = Vec::new();
    for cond in &rule.body {
        if let Condition::Positive(n) = cond {
            if let Neuron::Expression(parts) = n {
                if let Some(Neuron::Symbol(pred)) = parts.first() {
                    preds.push(pred.clone());
                }
            }
        }
    }
    preds
}

/// Extract predicates from negated body conditions.
fn negated_predicates(rule: &Rule) -> Vec<String> {
    let mut preds = Vec::new();
    for cond in &rule.body {
        let neuron = match cond {
            Condition::Negated(n) => Some(n),
            Condition::NegatedPresent(n) => Some(n),
            _ => None,
        };
        if let Some(n) = neuron {
            if let Neuron::Expression(parts) = n {
                if let Some(Neuron::Symbol(pred)) = parts.first() {
                    preds.push(pred.clone());
                }
            }
        }
    }
    preds
}

/// Compute strata for a set of rules using Kahn's topological sort on
/// the negation dependency graph.
///
/// Rules whose head predicate doesn't participate in negation stay at stratum 0.
/// Rules whose head is negated by another derived predicate get pushed to a
/// higher stratum, ensuring the negated predicate reaches fixed-point first.
///
/// Returns the max stratum assigned, or an error if a negation cycle is detected.
pub fn auto_stratify(rules: &mut [Rule]) -> Result<u32, StratifyError> {
    if rules.is_empty() {
        return Ok(0);
    }

    // 1. Collect derived predicates (predicates that appear as some rule's head).
    let derived_preds: HashSet<String> = rules.iter()
        .filter_map(|r| head_predicate(r))
        .collect();

    // 2. Build negation dependency DAG.
    //    Edge: neg_pred → head_pred means "head_pred must be in a LATER stratum
    //    than neg_pred" (because some rule deriving head_pred negates neg_pred).
    let mut graph: HashMap<String, HashSet<String>> = HashMap::new();
    let mut in_degree: HashMap<String, usize> = HashMap::new();
    let mut all_nodes: HashSet<String> = HashSet::new();

    for rule in rules.iter() {
        let head = match head_predicate(rule) {
            Some(h) => h,
            None => continue,
        };
        all_nodes.insert(head.clone());

        for neg_pred in negated_predicates(rule) {
            // Only create cross-stratum edge if the negated predicate is derived.
            // Negating a base predicate is safe at any stratum.
            if derived_preds.contains(&neg_pred) {
                all_nodes.insert(neg_pred.clone());
                graph.entry(neg_pred.clone()).or_default().insert(head.clone());
                *in_degree.entry(head.clone()).or_insert(0) += 1;
                in_degree.entry(neg_pred.clone()).or_insert(0);
            }
        }
    }

    // If no negation dependencies, all rules stay at stratum 0.
    if graph.is_empty() {
        return Ok(0);
    }

    // 3. Kahn's topological sort (BFS level = stratum).
    let mut stratum_map: HashMap<String, u32> = HashMap::new();
    let mut queue: VecDeque<String> = VecDeque::new();

    // Start with nodes that have in-degree 0 → stratum 0.
    for node in &all_nodes {
        if *in_degree.get(node).unwrap_or(&0) == 0 {
            queue.push_back(node.clone());
            stratum_map.insert(node.clone(), 0);
        }
    }

    let mut max_stratum: u32 = 0;
    let mut processed = 0;

    while !queue.is_empty() {
        // Process all nodes at the current level.
        let level_size = queue.len();
        for _ in 0..level_size {
            let node = queue.pop_front().unwrap();
            processed += 1;

            if let Some(neighbors) = graph.get(&node) {
                let current_stratum = stratum_map[&node];
                for neighbor in neighbors {
                    let deg = in_degree.get_mut(neighbor).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        let new_stratum = current_stratum + 1;
                        // Take the maximum if already assigned (multiple paths).
                        let entry = stratum_map.entry(neighbor.clone()).or_insert(0);
                        if new_stratum > *entry {
                            *entry = new_stratum;
                        }
                        if new_stratum > max_stratum {
                            max_stratum = new_stratum;
                        }
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }
    }

    // 4. Cycle detection: if not all nodes were processed, there's a cycle.
    if processed < all_nodes.len() {
        let cycle_preds: Vec<String> = all_nodes.into_iter()
            .filter(|n| !stratum_map.contains_key(n))
            .collect();
        return Err(StratifyError { cycle_preds });
    }

    // 5. Propagate through positive dependencies (fixed-point).
    //    If rule R derives head P and positively depends on predicate Q,
    //    then P's stratum must be >= Q's stratum. Combined with negation
    //    constraints (P's stratum > negated Q's stratum), iterate until stable.
    //    This handles cases like: buy depends on vwap-buy (stratum 1 due to
    //    negation) but buy itself has no negation → would stay at stratum 0
    //    without this propagation.
    let propagation_limit = derived_preds.len() + 1;
    for _ in 0..propagation_limit {
        let mut changed = false;
        for rule in rules.iter() {
            let head = match head_predicate(rule) {
                Some(h) => h,
                None => continue,
            };
            let current = stratum_map.get(&head).copied().unwrap_or(0);
            let mut needed = current;

            // Positive body deps: head stratum >= dep stratum.
            for pred in positive_predicates(rule) {
                let dep_s = stratum_map.get(&pred).copied().unwrap_or(0);
                if dep_s > needed {
                    needed = dep_s;
                }
            }

            // Negation body deps: head stratum > dep stratum.
            for neg_pred in negated_predicates(rule) {
                if derived_preds.contains(&neg_pred) {
                    let dep_s = stratum_map.get(&neg_pred).copied().unwrap_or(0);
                    if dep_s + 1 > needed {
                        needed = dep_s + 1;
                    }
                }
            }

            if needed > current {
                stratum_map.insert(head, needed);
                if needed > max_stratum {
                    max_stratum = needed;
                }
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // 6. Assign strata to rules based on their head predicate.
    for rule in rules.iter_mut() {
        // Skip rules that already have a manually-set stratum (non-zero).
        if rule.stratum != 0 {
            continue;
        }
        if let Some(head) = head_predicate(rule) {
            if let Some(&s) = stratum_map.get(&head) {
                rule.stratum = s;
            }
        }
    }

    Ok(max_stratum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use qor_core::neuron::Neuron;
    use qor_core::truth_value::TruthValue;

    fn sym(s: &str) -> Neuron { Neuron::Symbol(s.to_string()) }
    fn var(s: &str) -> Neuron { Neuron::Variable(s.to_string()) }
    fn expr(parts: Vec<Neuron>) -> Neuron { Neuron::Expression(parts) }
    fn tv() -> TruthValue { TruthValue::new(0.9, 0.9) }

    #[test]
    fn test_stratify_no_negation() {
        // (flies $x) if (bird $x)
        // (high-flyer $x) if (flies $x)
        // No negation → all stay stratum 0.
        let mut rules = vec![
            Rule::new(
                expr(vec![sym("flies"), var("x")]),
                vec![Condition::Positive(expr(vec![sym("bird"), var("x")]))],
                tv(),
            ),
            Rule::new(
                expr(vec![sym("high-flyer"), var("x")]),
                vec![Condition::Positive(expr(vec![sym("flies"), var("x")]))],
                tv(),
            ),
        ];
        let max = auto_stratify(&mut rules).unwrap();
        assert_eq!(max, 0);
        assert!(rules.iter().all(|r| r.stratum == 0));
    }

    #[test]
    fn test_stratify_basic() {
        // (flies $x) if (bird $x) → stratum 0
        // (grounded $x) if (bird $x) not (flies $x) → stratum 1
        // Negates derived "flies" → grounded must be stratum 1.
        let mut rules = vec![
            Rule::new(
                expr(vec![sym("flies"), var("x")]),
                vec![Condition::Positive(expr(vec![sym("bird"), var("x")]))],
                tv(),
            ),
            Rule::new(
                expr(vec![sym("grounded"), var("x")]),
                vec![
                    Condition::Positive(expr(vec![sym("bird"), var("x")])),
                    Condition::Negated(expr(vec![sym("flies"), var("x")])),
                ],
                tv(),
            ),
        ];
        let max = auto_stratify(&mut rules).unwrap();
        assert_eq!(max, 1);
        assert_eq!(rules[0].stratum, 0); // flies → stratum 0
        assert_eq!(rules[1].stratum, 1); // grounded (negates flies) → stratum 1
    }

    #[test]
    fn test_stratify_chain() {
        // A derives P (stratum 0)
        // B negates P, derives Q (stratum 1)
        // C negates Q, derives R (stratum 2)
        let mut rules = vec![
            Rule::new(
                expr(vec![sym("p"), var("x")]),
                vec![Condition::Positive(expr(vec![sym("base"), var("x")]))],
                tv(),
            ),
            Rule::new(
                expr(vec![sym("q"), var("x")]),
                vec![
                    Condition::Positive(expr(vec![sym("base"), var("x")])),
                    Condition::Negated(expr(vec![sym("p"), var("x")])),
                ],
                tv(),
            ),
            Rule::new(
                expr(vec![sym("r"), var("x")]),
                vec![
                    Condition::Positive(expr(vec![sym("base"), var("x")])),
                    Condition::Negated(expr(vec![sym("q"), var("x")])),
                ],
                tv(),
            ),
        ];
        let max = auto_stratify(&mut rules).unwrap();
        assert_eq!(max, 2);
        assert_eq!(rules[0].stratum, 0); // p
        assert_eq!(rules[1].stratum, 1); // q (negates p)
        assert_eq!(rules[2].stratum, 2); // r (negates q)
    }

    #[test]
    fn test_stratify_cycle_detection() {
        // A negates B, derives P
        // B negates P, derives Q (= B's head has predicate "q")
        // Wait — need P→Q and Q→P negation cycle.
        // P derives (p $x) negating (q $x)
        // Q derives (q $x) negating (p $x)
        let mut rules = vec![
            Rule::new(
                expr(vec![sym("p"), var("x")]),
                vec![
                    Condition::Positive(expr(vec![sym("base"), var("x")])),
                    Condition::Negated(expr(vec![sym("q"), var("x")])),
                ],
                tv(),
            ),
            Rule::new(
                expr(vec![sym("q"), var("x")]),
                vec![
                    Condition::Positive(expr(vec![sym("base"), var("x")])),
                    Condition::Negated(expr(vec![sym("p"), var("x")])),
                ],
                tv(),
            ),
        ];
        let result = auto_stratify(&mut rules);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.cycle_preds.contains(&"p".to_string()) || err.cycle_preds.contains(&"q".to_string()));
    }

    #[test]
    fn test_stratify_base_pred_safe() {
        // (grounded $x) if (bird $x) not (can-fly $x)
        // "can-fly" is NOT derived by any rule → base predicate.
        // Negating a base predicate is safe at stratum 0.
        let mut rules = vec![
            Rule::new(
                expr(vec![sym("grounded"), var("x")]),
                vec![
                    Condition::Positive(expr(vec![sym("bird"), var("x")])),
                    Condition::Negated(expr(vec![sym("can-fly"), var("x")])),
                ],
                tv(),
            ),
        ];
        let max = auto_stratify(&mut rules).unwrap();
        assert_eq!(max, 0);
        assert_eq!(rules[0].stratum, 0);
    }

    #[test]
    fn test_stratify_multiple_rules_same_head() {
        // Two rules derive "flies" → stratum 0
        // One rule negates "flies" → stratum 1
        let mut rules = vec![
            Rule::new(
                expr(vec![sym("flies"), var("x")]),
                vec![Condition::Positive(expr(vec![sym("bird"), var("x")]))],
                tv(),
            ),
            Rule::new(
                expr(vec![sym("flies"), var("x")]),
                vec![Condition::Positive(expr(vec![sym("plane"), var("x")]))],
                tv(),
            ),
            Rule::new(
                expr(vec![sym("non-flyer"), var("x")]),
                vec![
                    Condition::Positive(expr(vec![sym("thing"), var("x")])),
                    Condition::Negated(expr(vec![sym("flies"), var("x")])),
                ],
                tv(),
            ),
        ];
        let max = auto_stratify(&mut rules).unwrap();
        assert_eq!(max, 1);
        assert_eq!(rules[0].stratum, 0);
        assert_eq!(rules[1].stratum, 0);
        assert_eq!(rules[2].stratum, 1);
    }

    #[test]
    fn test_stratify_positive_dep_propagation() {
        // (p $x) if (base $x)             → stratum 0
        // (q $x) if (base $x) not (p $x)  → stratum 1 (negates p)
        // (r $x) if (q $x)                → stratum 1 (positive dep on q@1)
        // Without positive propagation, r would stay at stratum 0 and never see q.
        let mut rules = vec![
            Rule::new(
                expr(vec![sym("p"), var("x")]),
                vec![Condition::Positive(expr(vec![sym("base"), var("x")]))],
                tv(),
            ),
            Rule::new(
                expr(vec![sym("q"), var("x")]),
                vec![
                    Condition::Positive(expr(vec![sym("base"), var("x")])),
                    Condition::Negated(expr(vec![sym("p"), var("x")])),
                ],
                tv(),
            ),
            Rule::new(
                expr(vec![sym("r"), var("x")]),
                vec![Condition::Positive(expr(vec![sym("q"), var("x")]))],
                tv(),
            ),
        ];
        let max = auto_stratify(&mut rules).unwrap();
        assert_eq!(rules[0].stratum, 0); // p
        assert_eq!(rules[1].stratum, 1); // q (negates p)
        assert_eq!(rules[2].stratum, 1); // r (positive dep on q)
        assert!(max >= 1);
    }

    #[test]
    fn test_stratify_positive_then_negation_chain() {
        // (p $x) if (base $x)             → stratum 0
        // (q $x) if (base $x) not (p $x)  → stratum 1 (negates p)
        // (r $x) if (q $x)                → stratum 1 (positive dep on q@1)
        // (s $x) if (base $x) not (r $x)  → stratum 2 (negates r@1)
        let mut rules = vec![
            Rule::new(
                expr(vec![sym("p"), var("x")]),
                vec![Condition::Positive(expr(vec![sym("base"), var("x")]))],
                tv(),
            ),
            Rule::new(
                expr(vec![sym("q"), var("x")]),
                vec![
                    Condition::Positive(expr(vec![sym("base"), var("x")])),
                    Condition::Negated(expr(vec![sym("p"), var("x")])),
                ],
                tv(),
            ),
            Rule::new(
                expr(vec![sym("r"), var("x")]),
                vec![Condition::Positive(expr(vec![sym("q"), var("x")]))],
                tv(),
            ),
            Rule::new(
                expr(vec![sym("s"), var("x")]),
                vec![
                    Condition::Positive(expr(vec![sym("base"), var("x")])),
                    Condition::Negated(expr(vec![sym("r"), var("x")])),
                ],
                tv(),
            ),
        ];
        let max = auto_stratify(&mut rules).unwrap();
        assert_eq!(rules[0].stratum, 0); // p
        assert_eq!(rules[1].stratum, 1); // q (negates p)
        assert_eq!(rules[2].stratum, 1); // r (positive dep on q)
        assert_eq!(rules[3].stratum, 2); // s (negates r@1)
        assert_eq!(max, 2);
    }

    #[test]
    fn test_stratify_preserves_manual_stratum() {
        // Rule with manually set stratum should not be overwritten.
        let mut rules = vec![
            Rule {
                head: expr(vec![sym("p"), var("x")]),
                body: vec![Condition::Positive(expr(vec![sym("base"), var("x")]))],
                tv: tv(),
                stratum: 5, // manually set
            },
            Rule::new(
                expr(vec![sym("q"), var("x")]),
                vec![
                    Condition::Positive(expr(vec![sym("base"), var("x")])),
                    Condition::Negated(expr(vec![sym("p"), var("x")])),
                ],
                tv(),
            ),
        ];
        let _max = auto_stratify(&mut rules).unwrap();
        assert_eq!(rules[0].stratum, 5); // preserved
        // q negates p (derived) → gets stratum 1 from auto
        assert_eq!(rules[1].stratum, 1);
    }
}
