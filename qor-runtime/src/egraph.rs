// ── E-Graph Rule Deduplication ────────────────────────────────────────
//
// Uses the `egg` crate to detect semantically equivalent rules.
// Equivalent rules (same conditions reordered, redundant guards, etc.)
// get deduplicated, keeping the highest-scoring representative.
//
// DOMAIN AGNOSTIC — works with any QOR rules.

use egg::{rewrite, AstSize, EGraph, Extractor, RecExpr, Runner, SymbolLang};

use crate::chain::Rule;
use crate::mutate::rule_to_qor;
use crate::search::ScoredRule;

/// Algebraic rewrite rules for QOR condition equivalence.
/// These capture semantic identities that don't affect forward chaining.
fn qor_rewrites() -> Vec<egg::Rewrite<SymbolLang, ()>> {
    vec![
        // Condition commutativity: body order doesn't matter in QOR
        rewrite!("and-comm"; "(and ?a ?b)" => "(and ?b ?a)"),
        rewrite!("and-assoc"; "(and ?a (and ?b ?c))" => "(and (and ?a ?b) ?c)"),

        // Duplicate condition removal
        rewrite!("and-idem"; "(and ?a ?a)" => "?a"),

        // Guard simplification
        rewrite!("gt-to-ge"; "(and (gt ?x ?a) (ge ?x ?a))" => "(gt ?x ?a)"),
        rewrite!("ge-to-gt"; "(and (ge ?x ?a) (gt ?x ?a))" => "(gt ?x ?a)"),

        // != redundancy when > is present
        rewrite!("gt-implies-ne-zero"; "(and (gt ?x 0) (ne ?x 0))" => "(gt ?x 0)"),
    ]
}

/// Convert a rule body to an egg S-expression string.
/// Each condition becomes a node; the body is wrapped in nested (and ...) nodes.
fn rule_body_to_sexpr(rule: &Rule) -> String {
    if rule.body.is_empty() {
        return "empty".to_string();
    }
    if rule.body.len() == 1 {
        return condition_to_sexpr(&rule.body[0]);
    }

    // Nest conditions into (and c1 (and c2 (and c3 ...)))
    let mut result = condition_to_sexpr(rule.body.last().unwrap());
    for cond in rule.body.iter().rev().skip(1) {
        result = format!("(and {} {})", condition_to_sexpr(cond), result);
    }
    result
}

/// Convert a single Condition to a deterministic string for e-graph insertion.
fn condition_to_sexpr(cond: &qor_core::neuron::Condition) -> String {
    // Use Display impl, but normalize for e-graph compatibility
    let s = format!("{}", cond);
    // Replace QOR syntax with S-expr-safe form
    s.replace("!=", "ne")
        .replace(">=", "ge")
        .replace("<=", "le")
        .replace('>', "gt")
        .replace('<', "lt")
}

/// Check if two rules are semantically equivalent using e-graph equality saturation.
pub fn rules_equivalent(r1: &Rule, r2: &Rule) -> bool {
    // Quick check: same head required
    if r1.head != r2.head {
        return false;
    }

    let s1 = rule_body_to_sexpr(r1);
    let s2 = rule_body_to_sexpr(r2);

    // Same string = trivially equivalent
    if s1 == s2 {
        return true;
    }

    // Parse into egg RecExprs
    let e1: Result<RecExpr<SymbolLang>, _> = s1.parse();
    let e2: Result<RecExpr<SymbolLang>, _> = s2.parse();

    let (e1, e2) = match (e1, e2) {
        (Ok(a), Ok(b)) => (a, b),
        _ => return false,
    };

    // Build e-graph with both expressions
    let mut egraph = EGraph::<SymbolLang, ()>::default();
    let id1 = egraph.add_expr(&e1);
    let id2 = egraph.add_expr(&e2);

    // Run equality saturation with our rewrite rules
    let runner = Runner::default()
        .with_egraph(egraph)
        .with_iter_limit(5)
        .with_node_limit(1000)
        .run(&qor_rewrites());

    // Check if they ended up in the same e-class
    runner.egraph.find(id1) == runner.egraph.find(id2)
}

/// Canonicalize a rule's body using e-graph extraction.
/// Returns the simplest equivalent form (fewest AST nodes).
pub fn canonicalize_rule(rule: &Rule) -> Rule {
    let sexpr = rule_body_to_sexpr(rule);
    let expr: Result<RecExpr<SymbolLang>, _> = sexpr.parse();

    let expr = match expr {
        Ok(e) => e,
        Err(_) => return rule.clone(), // parse failure = return unchanged
    };

    let mut egraph = EGraph::<SymbolLang, ()>::default();
    let root = egraph.add_expr(&expr);

    let runner = Runner::default()
        .with_egraph(egraph)
        .with_iter_limit(5)
        .with_node_limit(1000)
        .run(&qor_rewrites());

    // Extract the simplest equivalent expression
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (_cost, _best) = extractor.find_best(root);

    // For now, return the original rule — the canonical form is tracked
    // by the e-graph but reconstructing Condition AST from the S-expr
    // is complex. The main value is in dedup_rules (below) which uses
    // equivalence checking, not reconstruction.
    rule.clone()
}

/// Deduplicate scored rules: remove equivalent rules, keeping the highest-scoring.
pub fn dedup_rules(rules: &mut Vec<ScoredRule>) {
    if rules.len() <= 1 {
        return;
    }

    // Sort by score descending (keep best first)
    rules.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // Build e-graph with all rule bodies, track which e-class each belongs to
    let mut egraph = EGraph::<SymbolLang, ()>::default();
    let mut ids = Vec::new();

    for sr in rules.iter() {
        let sexpr = rule_body_to_sexpr(&sr.rule);
        match sexpr.parse::<RecExpr<SymbolLang>>() {
            Ok(expr) => {
                let id = egraph.add_expr(&expr);
                ids.push(Some(id));
            }
            Err(_) => {
                ids.push(None);
            }
        }
    }

    // Run equality saturation
    let runner = Runner::default()
        .with_egraph(egraph)
        .with_iter_limit(5)
        .with_node_limit(5000)
        .run(&qor_rewrites());

    // Group by e-class (canonical ID after saturation)
    let mut seen_classes = std::collections::HashSet::new();
    let mut keep = Vec::new();

    for (i, sr) in rules.iter().enumerate() {
        let dominated = if let Some(id) = ids[i] {
            // Also need same head to be truly equivalent
            let canonical = runner.egraph.find(id);
            let head_key = format!("{}:{}", sr.rule.head, canonical);
            !seen_classes.insert(head_key)
        } else {
            false // unparseable rules are never deduped
        };

        if !dominated {
            keep.push(sr.clone());
        }
    }

    let removed = rules.len() - keep.len();
    if removed > 0 {
        eprintln!("      egraph-dedup: removed {} equivalent rules ({} → {})",
            removed, rules.len(), keep.len());
    }
    *rules = keep;
}

#[cfg(test)]
mod tests {
    use super::*;
    use qor_core::neuron::{Condition, ComparisonOp, Neuron, QorValue};
    use qor_core::truth_value::TruthValue;

    fn make_rule(body: Vec<Condition>) -> Rule {
        Rule {
            head: Neuron::expression(vec![
                Neuron::symbol("test"),
                Neuron::variable("x"),
            ]),
            body,
            tv: TruthValue::new(0.9, 0.9),
            stratum: 0,
        }
    }

    #[test]
    fn test_same_rules_equivalent() {
        let r = make_rule(vec![
            Condition::Positive(Neuron::expression(vec![
                Neuron::symbol("bird"),
                Neuron::variable("x"),
            ])),
        ]);
        assert!(rules_equivalent(&r, &r));
    }

    #[test]
    fn test_different_rules_not_equivalent() {
        let r1 = make_rule(vec![
            Condition::Positive(Neuron::expression(vec![
                Neuron::symbol("bird"),
                Neuron::variable("x"),
            ])),
        ]);
        let r2 = make_rule(vec![
            Condition::Positive(Neuron::expression(vec![
                Neuron::symbol("fish"),
                Neuron::variable("x"),
            ])),
        ]);
        assert!(!rules_equivalent(&r1, &r2));
    }

    #[test]
    fn test_dedup_removes_duplicates() {
        let r1 = make_rule(vec![
            Condition::Positive(Neuron::expression(vec![
                Neuron::symbol("bird"),
                Neuron::variable("x"),
            ])),
        ]);
        let mut rules = vec![
            ScoredRule { rule: r1.clone(), score: 0.9, qor_text: "a".into() },
            ScoredRule { rule: r1.clone(), score: 0.8, qor_text: "b".into() },
        ];
        dedup_rules(&mut rules);
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].score, 0.9); // kept higher score
    }
}
