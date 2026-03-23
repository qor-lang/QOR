// ── QOR Inference Engine — Induction & Abduction ────────────────────────
//
// Discovers NEW rules from existing ones:
// - Induction:  B→A + B→C ⇒ A→C  (shared body term)
// - Abduction:  A→B + C→B ⇒ A→C  (shared head term)
//
// These are the "creative" inference steps — QOR doesn't just apply rules,
// it discovers new relationships between concepts.

use std::collections::HashSet;

use qor_core::neuron::{Condition, Neuron, Statement};
use qor_core::truth_value::TruthValue;

/// Run induction and abduction on existing rules to discover new rules.
/// Returns only NEWLY inferred rules (not duplicates of existing ones).
pub fn infer(rules: &[Statement]) -> Vec<Statement> {
    let parsed: Vec<ParsedRule> = rules.iter().filter_map(|s| ParsedRule::from_stmt(s)).collect();

    let mut new_rules = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    // Register existing rules to avoid duplicates
    for r in &parsed {
        seen.insert(rule_signature(&r.head_pred, &r.body_preds));
    }

    // Induction: shared body → new rule connecting heads
    for i in 0..parsed.len() {
        for j in (i + 1)..parsed.len() {
            let shared_body = shared_terms(&parsed[i].body_preds, &parsed[j].body_preds);
            if !shared_body.is_empty() {
                // B→A + B→C ⇒ A→C
                let sig = rule_signature(&parsed[i].head_pred, &[parsed[j].head_pred.clone()]);
                if !seen.contains(&sig) {
                    seen.insert(sig);
                    let tv = parsed[i].tv.induction(&parsed[j].tv);
                    new_rules.push(make_rule(
                        &parsed[i].head_pred,
                        &[parsed[j].head_pred.clone()],
                        tv,
                    ));
                }

                // Also: C→A (reverse direction)
                let sig_rev = rule_signature(&parsed[j].head_pred, &[parsed[i].head_pred.clone()]);
                if !seen.contains(&sig_rev) {
                    seen.insert(sig_rev);
                    let tv = parsed[j].tv.induction(&parsed[i].tv);
                    new_rules.push(make_rule(
                        &parsed[j].head_pred,
                        &[parsed[i].head_pred.clone()],
                        tv,
                    ));
                }
            }
        }
    }

    // Abduction: shared head → new rule connecting bodies
    for i in 0..parsed.len() {
        for j in (i + 1)..parsed.len() {
            if parsed[i].head_pred == parsed[j].head_pred {
                // A→B + C→B ⇒ infer relationship between A and C bodies
                for body_i in &parsed[i].body_preds {
                    for body_j in &parsed[j].body_preds {
                        if body_i == body_j { continue; }

                        let sig = rule_signature(body_i, &[body_j.clone()]);
                        if !seen.contains(&sig) {
                            seen.insert(sig);
                            let tv = parsed[i].tv.abduction(&parsed[j].tv);
                            new_rules.push(make_rule(body_i, &[body_j.clone()], tv));
                        }
                    }
                }
            }
        }
    }

    new_rules
}

// ── Internal types ──────────────────────────────────────────────────────

struct ParsedRule {
    head_pred: String,
    body_preds: Vec<String>,
    tv: TruthValue,
}

impl ParsedRule {
    fn from_stmt(stmt: &Statement) -> Option<Self> {
        if let Statement::Rule { head, body, tv } = stmt {
            let head_pred = first_symbol(head)?;
            let body_preds: Vec<String> = body.iter().filter_map(|c| {
                match c {
                    Condition::Positive(n) => first_symbol(n),
                    _ => None,
                }
            }).collect();
            if body_preds.is_empty() { return None; }
            let tv = tv.unwrap_or(TruthValue::default_fact());
            Some(ParsedRule { head_pred, body_preds, tv })
        } else {
            None
        }
    }
}

fn first_symbol(neuron: &Neuron) -> Option<String> {
    match neuron {
        Neuron::Symbol(s) => Some(s.clone()),
        Neuron::Expression(parts) if !parts.is_empty() => {
            if let Neuron::Symbol(s) = &parts[0] {
                Some(s.clone())
            } else {
                None
            }
        }
        _ => None,
    }
}

fn shared_terms(a: &[String], b: &[String]) -> Vec<String> {
    a.iter().filter(|t| b.contains(t)).cloned().collect()
}

fn rule_signature(head: &str, body: &[String]) -> String {
    let mut sorted_body = body.to_vec();
    sorted_body.sort();
    format!("{}:if:{}", head, sorted_body.join(":"))
}

fn make_rule(head_pred: &str, body_preds: &[String], tv: TruthValue) -> Statement {
    Statement::Rule {
        head: Neuron::Expression(vec![
            Neuron::Symbol(head_pred.into()),
            Neuron::Variable("x".into()),
        ]),
        body: body_preds.iter().map(|pred| {
            Condition::Positive(Neuron::Expression(vec![
                Neuron::Symbol(pred.clone()),
                Neuron::Variable("x".into()),
            ]))
        }).collect(),
        tv: Some(tv),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(head: &str, body: &[&str], s: f64, c: f64) -> Statement {
        Statement::Rule {
            head: Neuron::Expression(vec![
                Neuron::Symbol(head.into()),
                Neuron::Variable("x".into()),
            ]),
            body: body.iter().map(|b| {
                Condition::Positive(Neuron::Expression(vec![
                    Neuron::Symbol((*b).into()),
                    Neuron::Variable("x".into()),
                ]))
            }).collect(),
            tv: Some(TruthValue::new(s, c)),
        }
    }

    #[test]
    fn test_induction_shared_body() {
        // (flies $x) if (bird $x)
        // (has-feathers $x) if (bird $x)
        // Induction: flies and has-feathers share body "bird"
        // → (flies $x) if (has-feathers $x)
        // → (has-feathers $x) if (flies $x)
        let rules = vec![
            rule("flies", &["bird"], 0.95, 0.90),
            rule("has-feathers", &["bird"], 0.99, 0.85),
        ];

        let inferred = infer(&rules);
        assert!(inferred.len() >= 2, "should infer at least 2 new rules, got {}", inferred.len());

        let sigs: Vec<String> = inferred.iter().map(|s| format!("{:?}", s)).collect();
        let all = sigs.join(" ");
        assert!(all.contains("flies") && all.contains("has-feathers"));
    }

    #[test]
    fn test_abduction_shared_head() {
        // (dangerous $x) if (poisonous $x)
        // (dangerous $x) if (venomous $x)
        // Abduction: both lead to "dangerous"
        // → (poisonous $x) if (venomous $x)
        // → (venomous $x) if (poisonous $x)
        let rules = vec![
            rule("dangerous", &["poisonous"], 0.90, 0.85),
            rule("dangerous", &["venomous"], 0.85, 0.80),
        ];

        let inferred = infer(&rules);
        assert!(!inferred.is_empty(), "should infer new rules via abduction");

        let sigs: Vec<String> = inferred.iter().map(|s| format!("{:?}", s)).collect();
        let all = sigs.join(" ");
        assert!(all.contains("poisonous") && all.contains("venomous"));
    }

    #[test]
    fn test_no_duplicates() {
        let rules = vec![
            rule("flies", &["bird"], 0.95, 0.90),
            rule("has-feathers", &["bird"], 0.99, 0.85),
        ];

        let inferred = infer(&rules);
        let sigs: HashSet<String> = inferred.iter().map(|r| {
            if let Statement::Rule { head, body, .. } = r {
                format!("{}:{}", head, body.iter().map(|c| format!("{}", c)).collect::<Vec<_>>().join(":"))
            } else {
                String::new()
            }
        }).collect();
        assert_eq!(sigs.len(), inferred.len(), "should have no duplicate rules");
    }

    #[test]
    fn test_empty_rules() {
        let inferred = infer(&[]);
        assert!(inferred.is_empty());
    }

    #[test]
    fn test_single_rule_no_inference() {
        let rules = vec![rule("flies", &["bird"], 0.95, 0.90)];
        let inferred = infer(&rules);
        assert!(inferred.is_empty(), "single rule can't produce inferences");
    }

    #[test]
    fn test_conservative_truth_values() {
        let rules = vec![
            rule("flies", &["bird"], 0.80, 0.70),
            rule("has-feathers", &["bird"], 0.90, 0.60),
        ];

        let inferred = infer(&rules);
        for r in &inferred {
            if let Statement::Rule { tv: Some(tv), .. } = r {
                // Inferred rules should have lower confidence than inputs
                assert!(tv.confidence <= 0.70,
                    "inferred confidence {} should be <= min input confidence 0.70", tv.confidence);
            }
        }
    }

    #[test]
    fn test_does_not_duplicate_existing() {
        // If (flies $x) if (has-feathers $x) already exists, don't re-infer it
        let rules = vec![
            rule("flies", &["bird"], 0.95, 0.90),
            rule("has-feathers", &["bird"], 0.99, 0.85),
            rule("flies", &["has-feathers"], 0.80, 0.70), // already exists!
        ];

        let inferred = infer(&rules);
        // The "flies if has-feathers" should NOT appear in inferred
        let has_dup = inferred.iter().any(|r| {
            if let Statement::Rule { head, body, .. } = r {
                first_symbol(head) == Some("flies".into())
                    && body.iter().any(|c| {
                        if let Condition::Positive(n) = c {
                            first_symbol(n) == Some("has-feathers".into())
                        } else { false }
                    })
            } else { false }
        });
        assert!(!has_dup, "should not duplicate existing 'flies if has-feathers' rule");
    }

    #[test]
    fn test_induction_tv_formula() {
        let tv1 = TruthValue::new(0.8, 0.9);
        let tv2 = TruthValue::new(0.7, 0.8);
        let r = tv1.induction(&tv2);
        // s = 0.8 * 0.7 = 0.56
        assert!((r.strength - 0.56).abs() < 0.01);
        // c = w2c(min(c2w(0.9), c2w(0.8))) = w2c(4) = 0.8
        assert!((r.confidence - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_abduction_tv_formula() {
        let tv1 = TruthValue::new(0.9, 0.85);
        let tv2 = TruthValue::new(0.8, 0.80);
        let r = tv1.abduction(&tv2);
        assert!((r.strength - 0.72).abs() < 0.01);
        assert!((r.confidence - 0.8).abs() < 0.01);
    }
}
