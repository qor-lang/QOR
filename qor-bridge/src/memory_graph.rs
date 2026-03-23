// ── Memory Graph — QOR Reasons About Its Past ───────────────────────────
//
// Converts flat PuzzleRun records into graph-structured QOR facts.
// The agent can then reason about its own experience using the same
// forward chaining engine it uses for everything else.
//
// Graph schema:
//   (puzzle <id>)                                — puzzle node
//   (puzzle-feature <id> <feature>)              — feature edge
//   (attempt <id> <strategy> win|fail <accuracy>) — attempt edge
//   (strategy-for <feature> <strategy>)          — aggregated performance

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use qor_core::neuron::{Neuron, Statement};
use qor_core::truth_value::TruthValue;

fn fact(neuron: Neuron, tv: TruthValue) -> Statement {
    Statement::Fact { neuron, tv: Some(tv), decay: None }
}

/// Convert a single PuzzleRun into QOR fact statements (graph edges).
pub fn run_to_statements(
    puzzle_id: &str,
    strategy: &str,
    correct: bool,
    accuracy: f64,
    features: &[String],
) -> Vec<Statement> {
    let mut stmts = Vec::new();
    let high_tv = TruthValue::new(0.99, 0.99);

    // Puzzle node
    stmts.push(fact(
        Neuron::expression(vec![Neuron::symbol("puzzle"), Neuron::symbol(puzzle_id)]),
        high_tv,
    ));

    // Feature edges
    for feat in features {
        stmts.push(fact(
            Neuron::expression(vec![
                Neuron::symbol("puzzle-feature"),
                Neuron::symbol(puzzle_id),
                Neuron::symbol(feat),
            ]),
            TruthValue::new(0.95, 0.95),
        ));
    }

    // Attempt edge
    let outcome = if correct { "win" } else { "fail" };
    stmts.push(fact(
        Neuron::expression(vec![
            Neuron::symbol("attempt"),
            Neuron::symbol(puzzle_id),
            Neuron::symbol(strategy),
            Neuron::symbol(outcome),
            Neuron::float_val(accuracy),
        ]),
        high_tv,
    ));

    stmts
}

/// Aggregate all runs into condensed strategy-performance facts.
/// Uses simple win-rate as strength, sample-count-based confidence.
pub fn aggregate_strategies(
    runs: &[(String, String, bool, f64, Vec<String>)], // (puzzle_id, strategy, correct, accuracy, features)
) -> Vec<Statement> {
    // Group by (feature, strategy) → (wins, total)
    let mut feature_strategy: HashMap<(String, String), (usize, usize)> = HashMap::new();

    for (_, strategy, correct, _, features) in runs {
        for feat in features {
            let key = (feat.clone(), strategy.clone());
            let entry = feature_strategy.entry(key).or_insert((0, 0));
            entry.1 += 1;
            if *correct {
                entry.0 += 1;
            }
        }
    }

    let mut stmts = Vec::new();
    for ((feature, strategy), (wins, total)) in &feature_strategy {
        let strength = if *total > 0 { *wins as f64 / *total as f64 } else { 0.0 };
        // Confidence from sample size: w2c(n) = n / (n + k), k=2
        let confidence = (*total as f64 / (*total as f64 + 2.0)).min(0.95);
        stmts.push(fact(
            Neuron::expression(vec![
                Neuron::symbol("strategy-for"),
                Neuron::symbol(&feature),
                Neuron::symbol(&strategy),
            ]),
            TruthValue::new(strength, confidence),
        ));
    }

    stmts
}

/// Load memory graph from brain/memory/*.qor files.
pub fn load_memory(memory_dir: &Path) -> Result<Vec<Statement>, String> {
    let mut all = Vec::new();
    if !memory_dir.exists() {
        return Ok(all);
    }
    for entry in std::fs::read_dir(memory_dir)
        .map_err(|e| format!("cannot read memory dir: {}", e))?
    {
        let entry = entry.map_err(|e| format!("read_dir entry: {}", e))?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("qor") {
            let data = std::fs::read_to_string(&path)
                .map_err(|e| format!("read {}: {}", path.display(), e))?;
            let stmts = qor_core::parser::parse(&data)
                .map_err(|e| format!("parse {}: {}", path.display(), e))?;
            all.extend(stmts);
        }
    }
    Ok(all)
}

/// Save memory graph facts to brain/memory/ as .qor files.
pub fn save_memory(
    puzzles: &[Statement],
    attempts: &[Statement],
    strategies: &[Statement],
    memory_dir: &Path,
) -> Result<(), String> {
    std::fs::create_dir_all(memory_dir)
        .map_err(|e| format!("create memory dir: {}", e))?;

    write_statements(&memory_dir.join("puzzles.qor"), puzzles)?;
    write_statements(&memory_dir.join("attempts.qor"), attempts)?;
    write_statements(&memory_dir.join("strategies.qor"), strategies)?;
    Ok(())
}

/// Partition memory statements by predicate into (puzzles, attempts, strategies).
pub fn partition_memory(stmts: &[Statement]) -> (Vec<Statement>, Vec<Statement>, Vec<Statement>) {
    let mut puzzles = Vec::new();
    let mut attempts = Vec::new();
    let mut strategies = Vec::new();

    for stmt in stmts {
        if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = stmt {
            if let Some(Neuron::Symbol(pred)) = parts.first() {
                match pred.as_str() {
                    "puzzle" | "puzzle-feature" => puzzles.push(stmt.clone()),
                    "attempt" => attempts.push(stmt.clone()),
                    "strategy-for" => strategies.push(stmt.clone()),
                    _ => {}
                }
            }
        }
    }

    (puzzles, attempts, strategies)
}

/// Extract feature names from store facts (obs-consistent-* predicates).
pub fn extract_features_from_facts(facts: &[Statement]) -> Vec<String> {
    let mut features = Vec::new();
    let feature_preds = [
        "obs-consistent-same-size",
        "obs-consistent-separator-v",
        "obs-consistent-separator-h",
        "obs-consistent-reflect-h",
        "obs-consistent-reflect-v",
        "obs-consistent-rotate-90",
        "obs-consistent-color-count-same",
        "obs-consistent-obj-count-same",
    ];

    for stmt in facts {
        if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = stmt {
            if let Some(Neuron::Symbol(pred)) = parts.first() {
                if feature_preds.contains(&pred.as_str()) {
                    let feat = pred.strip_prefix("obs-consistent-").unwrap_or(pred);
                    if !features.contains(&feat.to_string()) {
                        features.push(feat.to_string());
                    }
                }
            }
        }
    }

    features
}

/// Memory graph directory path (brain/memory/).
pub fn memory_dir(brain_dir: &Path) -> PathBuf {
    brain_dir.join("memory")
}

fn write_statements(path: &Path, stmts: &[Statement]) -> Result<(), String> {
    let mut lines = Vec::new();
    lines.push(format!(";; Auto-generated memory graph — {}",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown")));
    lines.push(String::new());
    for stmt in stmts {
        lines.push(format_statement(stmt));
    }
    std::fs::write(path, lines.join("\n"))
        .map_err(|e| format!("write {}: {}", path.display(), e))
}

fn format_statement(stmt: &Statement) -> String {
    match stmt {
        Statement::Fact { neuron, tv, .. } => {
            if let Some(tv) = tv {
                format!("{} <{:.2}, {:.2}>", neuron, tv.strength, tv.confidence)
            } else {
                format!("{}", neuron)
            }
        }
        Statement::Rule { head, body, tv } => {
            let body_str: Vec<String> = body.iter().map(|c| format!("{:?}", c)).collect();
            if let Some(tv) = tv {
                format!("{} if {} <{:.2}, {:.2}>", head, body_str.join(" "), tv.strength, tv.confidence)
            } else {
                format!("{} if {}", head, body_str.join(" "))
            }
        }
        Statement::Query { pattern } => {
            format!("? {}", pattern)
        }
        Statement::Test { name, .. } => {
            format!(";; @test {}", name)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_to_statements() {
        let stmts = run_to_statements(
            "abc123",
            "separator-combine",
            true,
            1.0,
            &["same-size".to_string(), "separator-v".to_string()],
        );
        // puzzle node + 2 features + 1 attempt = 4
        assert_eq!(stmts.len(), 4);

        // Check puzzle node exists
        let has_puzzle = stmts.iter().any(|s| {
            if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = s {
                parts.first() == Some(&Neuron::symbol("puzzle"))
            } else { false }
        });
        assert!(has_puzzle);

        // Check attempt has "win"
        let has_win = stmts.iter().any(|s| {
            if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = s {
                parts.contains(&Neuron::symbol("win"))
            } else { false }
        });
        assert!(has_win);
    }

    #[test]
    fn test_aggregate_strategies() {
        let runs = vec![
            ("p1".into(), "reflect-h".into(), true, 1.0, vec!["same-size".into()]),
            ("p2".into(), "reflect-h".into(), true, 1.0, vec!["same-size".into()]),
            ("p3".into(), "reflect-h".into(), false, 0.5, vec!["same-size".into()]),
            ("p4".into(), "color-remap".into(), false, 0.3, vec!["same-size".into()]),
        ];
        let stmts = aggregate_strategies(&runs);
        assert!(!stmts.is_empty());

        // Find strategy-for same-size reflect-h
        let reflect_stmt = stmts.iter().find(|s| {
            if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = s {
                parts.contains(&Neuron::symbol("reflect-h"))
                    && parts.contains(&Neuron::symbol("same-size"))
            } else { false }
        });
        assert!(reflect_stmt.is_some());
        // 2 wins / 3 total = 0.667 strength
        if let Statement::Fact { tv: Some(tv), .. } = reflect_stmt.unwrap() {
            assert!((tv.strength - 0.667).abs() < 0.01);
        }
    }

    #[test]
    fn test_partition_memory() {
        let stmts = vec![
            fact(
                Neuron::expression(vec![Neuron::symbol("puzzle"), Neuron::symbol("p1")]),
                TruthValue::new(0.99, 0.99),
            ),
            fact(
                Neuron::expression(vec![
                    Neuron::symbol("attempt"), Neuron::symbol("p1"),
                    Neuron::symbol("reflect-h"), Neuron::symbol("win"), Neuron::float_val(1.0),
                ]),
                TruthValue::new(0.99, 0.99),
            ),
            fact(
                Neuron::expression(vec![
                    Neuron::symbol("strategy-for"), Neuron::symbol("same-size"),
                    Neuron::symbol("reflect-h"),
                ]),
                TruthValue::new(0.80, 0.90),
            ),
        ];
        let (puzzles, attempts, strategies) = partition_memory(&stmts);
        assert_eq!(puzzles.len(), 1);
        assert_eq!(attempts.len(), 1);
        assert_eq!(strategies.len(), 1);
    }

    #[test]
    fn test_extract_features() {
        let facts = vec![
            fact(
                Neuron::expression(vec![Neuron::symbol("obs-consistent-same-size")]),
                TruthValue::new(0.95, 0.95),
            ),
            fact(
                Neuron::expression(vec![Neuron::symbol("obs-consistent-separator-v")]),
                TruthValue::new(0.90, 0.90),
            ),
            fact(
                Neuron::expression(vec![
                    Neuron::symbol("grid-cell"), Neuron::symbol("ti"),
                    Neuron::float_val(0.0), Neuron::float_val(0.0), Neuron::float_val(1.0),
                ]),
                TruthValue::new(0.99, 0.99),
            ),
        ];
        let features = extract_features_from_facts(&facts);
        assert_eq!(features.len(), 2);
        assert!(features.contains(&"same-size".to_string()));
        assert!(features.contains(&"separator-v".to_string()));
    }

    #[test]
    fn test_save_reload_roundtrip() {
        let tmp_dir = std::env::temp_dir().join("qor_memory_test");
        let _ = std::fs::remove_dir_all(&tmp_dir);

        let puzzles = vec![
            fact(
                Neuron::expression(vec![Neuron::symbol("puzzle"), Neuron::symbol("p1")]),
                TruthValue::new(0.99, 0.99),
            ),
        ];
        let attempts = vec![
            fact(
                Neuron::expression(vec![
                    Neuron::symbol("attempt"), Neuron::symbol("p1"),
                    Neuron::symbol("reflect-h"), Neuron::symbol("win"), Neuron::float_val(1.0),
                ]),
                TruthValue::new(0.99, 0.99),
            ),
        ];
        let strategies = vec![
            fact(
                Neuron::expression(vec![
                    Neuron::symbol("strategy-for"), Neuron::symbol("same-size"),
                    Neuron::symbol("reflect-h"),
                ]),
                TruthValue::new(0.67, 0.60),
            ),
        ];

        // Save
        save_memory(&puzzles, &attempts, &strategies, &tmp_dir).unwrap();

        // Reload
        let loaded = load_memory(&tmp_dir).unwrap();
        assert!(!loaded.is_empty());

        let (p, a, s) = partition_memory(&loaded);
        assert_eq!(p.len(), 1);
        assert_eq!(a.len(), 1);
        assert_eq!(s.len(), 1);

        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }
}
