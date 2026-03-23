// ── Refinement Search Loop ──────────────────────────────────────────
//
// Time-budgeted evolutionary search for rules.
// Takes near-miss rules, mutates them, scores against training data.
//
// DOMAIN AGNOSTIC — works with any facts and any target predicate.
// Domain knowledge comes from DNA rules loaded in the base session.
//
// Inspired by: SOAR (mutation + hindsight), Poetiq (iterative refinement).

use std::collections::HashSet;
use std::time::{Duration, Instant};

use rand::seq::SliceRandom;
use rand::Rng;

use qor_core::neuron::{Neuron, QorValue, Statement};

use crate::chain::Rule;
use crate::eval::Session;
use crate::mutate::{generate_mutations, crossover_rules, rule_to_qor};

/// Tournament selection: pick `k` random individuals, return the best.
fn tournament_select<'a>(pop: &'a [ScoredRule], k: usize, rng: &mut impl Rng) -> &'a ScoredRule {
    assert!(!pop.is_empty());
    let mut best = &pop[rng.gen_range(0..pop.len())];
    for _ in 1..k {
        let candidate = &pop[rng.gen_range(0..pop.len())];
        if candidate.score > best.score {
            best = candidate;
        }
    }
    best
}

/// Result of a search run.
#[derive(Debug)]
pub struct SearchResult {
    /// Rules that perfectly explain all training pairs (score == 1.0).
    pub solutions: Vec<ScoredRule>,
    /// Best non-perfect rules found.
    pub near_misses: Vec<ScoredRule>,
    /// How many mutations were tried.
    pub mutations_tried: usize,
    /// Time spent searching.
    pub elapsed_ms: u64,
}

/// A rule with its training score.
#[derive(Debug, Clone)]
pub struct ScoredRule {
    pub rule: Rule,
    pub score: f64,
    pub qor_text: String,
}

/// Score a rule against training pairs.
///
/// DOMAIN AGNOSTIC — compares facts by target predicate key matching.
/// For each (input_stmts, expected_output_stmts):
///   1. Clone session, inject rule + input facts, chain
///   2. Extract facts matching target_pred from session
///   3. Compare against expected output facts with same predicate
///   4. Score = fraction of expected facts that match
pub fn score_rule_on_training(
    rule: &Rule,
    training_inputs: &[Vec<Statement>],
    expected_outputs: &[Vec<Statement>],
    target_pred: &str,
    base_session: &Session,
) -> f64 {
    let mut total_score = 0.0;

    for (input_stmts, expected_stmts) in training_inputs.iter().zip(expected_outputs.iter()) {
        let mut session = base_session.clone();

        // Add the candidate rule
        let rule_text = rule_to_qor(rule);
        let _ = session.exec(&rule_text);

        // Add input facts
        let _ = session.exec_statements(input_stmts.clone());

        // Collect predictions matching target predicate
        let predictions = extract_facts_by_pred(&session, target_pred);

        // Collect expected facts matching target predicate
        let expected = extract_facts_from_stmts(expected_stmts, target_pred);
        if expected.is_empty() { continue; }

        let correct = expected.iter().filter(|k| predictions.contains(*k)).count();
        total_score += correct as f64 / expected.len() as f64;
    }

    total_score / training_inputs.len().max(1) as f64
}

/// Extract fact keys for a given predicate from a session.
fn extract_facts_by_pred(session: &Session, pred: &str) -> HashSet<String> {
    let mut set = HashSet::new();
    for sn in session.all_facts() {
        if let Neuron::Expression(parts) = &sn.neuron {
            if let Some(Neuron::Symbol(p)) = parts.first() {
                if p == pred {
                    set.insert(fact_key(parts));
                }
            }
        }
    }
    set
}

/// Extract fact keys for a given predicate from statements.
fn extract_facts_from_stmts(stmts: &[Statement], pred: &str) -> HashSet<String> {
    let mut set = HashSet::new();
    for stmt in stmts {
        if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = stmt {
            if let Some(Neuron::Symbol(p)) = parts.first() {
                if p == pred {
                    set.insert(fact_key(parts));
                }
            }
        }
    }
    set
}

/// Convert fact parts to a comparable key string.
fn fact_key(parts: &[Neuron]) -> String {
    parts.iter().map(|n| match n {
        Neuron::Symbol(s) => s.clone(),
        Neuron::Value(QorValue::Int(i)) => i.to_string(),
        Neuron::Value(QorValue::Float(f)) => format!("{f:.2}"),
        Neuron::Value(QorValue::Str(s)) => format!("\"{s}\""),
        Neuron::Value(QorValue::Bool(b)) => b.to_string(),
        _ => "_".into(),
    }).collect::<Vec<_>>().join(" ")
}

/// Run evolutionary refinement search.
///
/// Takes near-miss rules (almost worked), mutates them,
/// scores against training data, keeps best.
///
/// DOMAIN AGNOSTIC — target_pred identifies what predicate to produce.
pub fn refinement_search(
    seed_rules: &[Rule],
    training_inputs: &[Vec<Statement>],
    expected_outputs: &[Vec<Statement>],
    target_pred: &str,
    base_session: &Session,
    time_budget_ms: u64,
) -> SearchResult {
    let start = Instant::now();
    let budget = Duration::from_millis(time_budget_ms);

    let mut solutions = Vec::new();
    let mut best_near_misses: Vec<ScoredRule> = Vec::new();
    let mut mutations_tried = 0usize;
    let mut seen_outputs = HashSet::new();

    // Score seed rules first
    eprintln!("      refine: scoring {} seed rules...", seed_rules.len());
    let mut population: Vec<ScoredRule> = seed_rules.iter().map(|rule| {
        let score = score_rule_on_training(rule, training_inputs, expected_outputs, target_pred, base_session);
        let qor_text = rule_to_qor(rule);
        ScoredRule { rule: rule.clone(), score, qor_text }
    }).collect();

    population.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // Log seed scores
    for (si, sr) in population.iter().take(5).enumerate() {
        eprintln!("        seed[{}]: {:.1}% rule={}", si, sr.score * 100.0,
            sr.qor_text.chars().take(100).collect::<String>());
    }

    // Check for perfect solutions in seeds
    for sr in &population {
        if sr.score >= 0.999 {
            solutions.push(sr.clone());
        }
    }
    if !solutions.is_empty() {
        eprintln!("      refine: found {} perfect solutions in seeds!", solutions.len());
    }

    // Iterative mutation + crossover loop
    let max_pop = 20;
    let max_mutations_per_parent = 15;
    let mut generation = 0;
    let mut rng = rand::thread_rng();

    while start.elapsed() < budget && generation < 50 {
        generation += 1;

        // Tournament selection: pick 5 parents weighted by fitness
        let parents: Vec<ScoredRule> = if population.len() <= 5 {
            population.clone()
        } else {
            (0..5).map(|_| tournament_select(&population, 3, &mut rng).clone()).collect()
        };

        let mut new_candidates = Vec::new();
        let mut gen_mutations_count = 0;

        // Mutation phase: random sample of mutations per parent
        for sr in &parents {
            let mut mutations = generate_mutations(&sr.rule);
            mutations.shuffle(&mut rng);
            let budget_per_parent = mutations.len().min(max_mutations_per_parent);
            gen_mutations_count += budget_per_parent;

            for mutated in mutations.into_iter().take(budget_per_parent) {
                if start.elapsed() >= budget { break; }
                mutations_tried += 1;

                let score = score_rule_on_training(
                    &mutated, training_inputs, expected_outputs, target_pred, base_session,
                );
                let qor_text = rule_to_qor(&mutated);

                let key = format!("{:.4}-{}", score, qor_text.len());
                if seen_outputs.contains(&key) { continue; }
                seen_outputs.insert(key);

                if score >= 0.999 {
                    eprintln!("      refine-gen{}: PERFECT rule={}", generation, qor_text.chars().take(120).collect::<String>());
                    solutions.push(ScoredRule { rule: mutated, score, qor_text });
                } else if score > 0.0 {
                    new_candidates.push(ScoredRule { rule: mutated, score, qor_text });
                }
            }
        }

        // Crossover phase: combine pairs of parents
        if population.len() >= 2 {
            let num_crossovers = 3.min(population.len() / 2);
            for _ in 0..num_crossovers {
                if start.elapsed() >= budget { break; }
                let p1 = tournament_select(&population, 3, &mut rng);
                let p2 = tournament_select(&population, 3, &mut rng);
                let (child1, child2) = crossover_rules(&p1.rule, &p2.rule, &mut rng);

                for child in [child1, child2] {
                    mutations_tried += 1;
                    let score = score_rule_on_training(
                        &child, training_inputs, expected_outputs, target_pred, base_session,
                    );
                    let qor_text = rule_to_qor(&child);
                    let key = format!("{:.4}-{}", score, qor_text.len());
                    if seen_outputs.contains(&key) { continue; }
                    seen_outputs.insert(key);

                    if score >= 0.999 {
                        eprintln!("      refine-gen{}: PERFECT(xover) rule={}", generation, qor_text.chars().take(120).collect::<String>());
                        solutions.push(ScoredRule { rule: child, score, qor_text });
                    } else if score > 0.0 {
                        new_candidates.push(ScoredRule { rule: child, score, qor_text });
                    }
                }
            }
        }

        population.extend(new_candidates);
        population.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        population.truncate(max_pop);

        if generation <= 3 || generation % 10 == 0 {
            eprintln!("      refine-gen{}: {} mutations+xover, best={:.1}%, pop={}",
                generation, gen_mutations_count,
                population.first().map(|s| s.score * 100.0).unwrap_or(0.0),
                population.len());
        }

        if !solutions.is_empty() { break; }
    }
    eprintln!("      refine-done: {} gens, {} mutations, {} solutions, best={:.1}%",
        generation, mutations_tried, solutions.len(),
        population.first().map(|s| s.score * 100.0).unwrap_or(0.0));

    for sr in population.iter().take(5) {
        if sr.score > 0.5 && sr.score < 0.999 {
            best_near_misses.push(sr.clone());
        }
    }

    SearchResult {
        solutions,
        near_misses: best_near_misses,
        mutations_tried,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qor_core::neuron::{Condition, Neuron, QorValue};
    use qor_core::truth_value::TruthValue;

    fn make_fact(pred: &str, id: &str, a1: i64, a2: i64, val: i64) -> Statement {
        Statement::Fact {
            neuron: Neuron::Expression(vec![
                Neuron::Symbol(pred.into()),
                Neuron::Symbol(id.into()),
                Neuron::Value(QorValue::Int(a1)),
                Neuron::Value(QorValue::Int(a2)),
                Neuron::Value(QorValue::Int(val)),
            ]),
            tv: None,
            decay: None,
        }
    }

    #[test]
    fn test_score_rule_perfect_match() {
        let session = Session::new();

        // Rule: target $id $r $c 7 if source $id $r $c 3
        let rule = Rule::new(
            Neuron::Expression(vec![
                Neuron::Symbol("target".into()),
                Neuron::Variable("id".into()),
                Neuron::Variable("r".into()),
                Neuron::Variable("c".into()),
                Neuron::Value(QorValue::Int(7)),
            ]),
            vec![Condition::Positive(Neuron::Expression(vec![
                Neuron::Symbol("source".into()),
                Neuron::Variable("id".into()),
                Neuron::Variable("r".into()),
                Neuron::Variable("c".into()),
                Neuron::Value(QorValue::Int(3)),
            ]))],
            TruthValue::new(0.99, 0.99),
        );

        let input = vec![make_fact("source", "t0", 0, 0, 3)];
        let expected = vec![make_fact("target", "t0", 0, 0, 7)];

        let score = score_rule_on_training(&rule, &[input], &[expected], "target", &session);
        assert!((score - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_score_rule_no_match() {
        let session = Session::new();

        // Rule predicts value 5, but expected is 7
        let rule = Rule::new(
            Neuron::Expression(vec![
                Neuron::Symbol("target".into()),
                Neuron::Variable("id".into()),
                Neuron::Variable("r".into()),
                Neuron::Variable("c".into()),
                Neuron::Value(QorValue::Int(5)),
            ]),
            vec![Condition::Positive(Neuron::Expression(vec![
                Neuron::Symbol("source".into()),
                Neuron::Variable("id".into()),
                Neuron::Variable("r".into()),
                Neuron::Variable("c".into()),
                Neuron::Value(QorValue::Int(3)),
            ]))],
            TruthValue::new(0.99, 0.99),
        );

        let input = vec![make_fact("source", "t0", 0, 0, 3)];
        let expected = vec![make_fact("target", "t0", 0, 0, 7)];

        let score = score_rule_on_training(&rule, &[input], &[expected], "target", &session);
        assert!(score < 0.01);
    }

    #[test]
    fn test_refinement_finds_solution() {
        let session = Session::new();

        // Seed: predicts value 6, but expected is 7
        // Mutation should change 6 → 7 via ±1
        let seed = Rule::new(
            Neuron::Expression(vec![
                Neuron::Symbol("target".into()),
                Neuron::Variable("id".into()),
                Neuron::Variable("r".into()),
                Neuron::Variable("c".into()),
                Neuron::Value(QorValue::Int(6)),
            ]),
            vec![Condition::Positive(Neuron::Expression(vec![
                Neuron::Symbol("source".into()),
                Neuron::Variable("id".into()),
                Neuron::Variable("r".into()),
                Neuron::Variable("c".into()),
                Neuron::Value(QorValue::Int(3)),
            ]))],
            TruthValue::new(0.99, 0.99),
        );

        let input = vec![make_fact("source", "t0", 0, 0, 3)];
        let expected = vec![make_fact("target", "t0", 0, 0, 7)];

        let result = refinement_search(
            &[seed],
            &[input],
            &[expected],
            "target",
            &session,
            5000,
        );

        assert!(!result.solutions.is_empty() || !result.near_misses.is_empty(),
            "Search should find some candidates, tried {} mutations", result.mutations_tried);
    }
}
