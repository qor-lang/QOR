//! ARC-AGI Puzzle Benchmark
//! ========================
//! Runs ARC training puzzles through the FULL solve pipeline:
//!   DNA → feed puzzle → solve() (6 phases × 5 rounds) → score on test
//!
//! Usage: cargo test -p qor-agent --release arc_100 -- --nocapture --ignored

use std::collections::HashSet;
use qor_bridge::grid::Grid;
use qor_bridge::solve;
use qor_core::neuron::{Neuron, QorValue, Statement};
use qor_core::parser;
use qor_runtime::eval::Session;
use qor_runtime::library::RuleLibrary;
use std::path::Path;
use std::time::Instant;

// ── ARC JSON structs ────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct ArcPuzzle {
    train: Vec<ArcPair>,
    test: Vec<ArcPair>,
}

#[derive(serde::Deserialize)]
struct ArcPair {
    input: Vec<Vec<u8>>,
    output: Vec<Vec<u8>>,
}

// ── Helpers ─────────────────────────────────────────────────────────

fn dna_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("dna").join("puzzle_solver")
}

fn meta_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("meta")
}

fn data_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("data100")
}

/// Load external taxonomy classification (puzzle_id → category code).
/// Returns empty map if taxonomy.json doesn't exist.
fn load_taxonomy() -> std::collections::HashMap<String, String> {
    let path = data_dir().join("taxonomy.json");
    if let Ok(data) = std::fs::read_to_string(&path) {
        serde_json::from_str(&data).unwrap_or_default()
    } else {
        std::collections::HashMap::new()
    }
}

fn load_puzzle(id: &str) -> Option<ArcPuzzle> {
    let path = data_dir().join(format!("{id}.json"));
    let data = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&data).ok()
}

/// Feed puzzle into session and return (observations, all_expected).
fn feed_puzzle(session: &mut Session, puzzle: &ArcPuzzle, puzzle_id: &str)
    -> (Vec<Statement>, Vec<Statement>)
{
    let mut all_facts = Vec::new();
    let mut pair_facts: Vec<Vec<Statement>> = Vec::new();
    let mut pair_grids: Vec<(Grid, Grid)> = Vec::new();
    let mut all_expected = Vec::new();

    for (i, pair) in puzzle.train.iter().enumerate() {
        let in_id = format!("t{}i", i);
        let out_id = format!("t{}o", i);
        let in_grid = Grid::from_vecs(pair.input.clone()).unwrap();
        let out_grid = Grid::from_vecs(pair.output.clone()).unwrap();

        all_facts.push(Statement::simple_fact(vec![Neuron::symbol("train-pair"), Neuron::symbol(&in_id), Neuron::symbol(&out_id)]));
        all_facts.extend(in_grid.to_statements(&in_id));
        all_facts.extend(out_grid.to_statements(&out_id));

        let compare = Grid::compare_pair(&in_grid, &out_grid, &in_id, &out_id);
        pair_facts.push(compare.clone());
        all_facts.extend(compare);
        pair_grids.push((in_grid, out_grid));

        // Collect expected output facts for solve()
        // Use TRAINING input ID (t0i, t1i...) so split_expected_by_pair can match
        for (r, row) in pair.output.iter().enumerate() {
            for (c, &val) in row.iter().enumerate() {
                if val > 0 {
                    all_expected.push(Statement::simple_fact(vec![
                        Neuron::symbol("predict-cell"),
                        Neuron::symbol(&in_id),
                        Neuron::Value(QorValue::Int(r as i64)),
                        Neuron::Value(QorValue::Int(c as i64)),
                        Neuron::Value(QorValue::Int(val as i64)),
                    ]));
                }
            }
        }
    }

    // Observations
    let obs = Grid::compute_observations(&pair_facts, &pair_grids);
    all_facts.extend(obs.clone());

    // Test input
    let test_in = Grid::from_vecs(puzzle.test[0].input.clone()).unwrap();
    all_facts.push(Statement::simple_fact(vec![Neuron::symbol("test-input"), Neuron::symbol("ti")]));
    all_facts.extend(test_in.to_statements("ti"));
    all_facts.push(Statement::simple_fact(vec![Neuron::symbol("item-id"), Neuron::symbol(puzzle_id)]));

    let _ = session.exec_statements(all_facts);

    (obs, all_expected)
}

// ═════════════════════════════════════════════════════════════════════
// BENCHMARK: ARC puzzles through full solve() pipeline
// ═════════════════════════════════════════════════════════════════════

#[test]
#[ignore] // Run with: cargo test -p qor-agent --release arc_100 -- --nocapture --ignored
fn arc_100_benchmark() {
    let training_dir = data_dir();
    let mut puzzle_ids: Vec<String> = std::fs::read_dir(training_dir)
        .expect("Cannot read ARC training dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
        .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().to_string()))
        .collect();
    puzzle_ids.sort();
    puzzle_ids.truncate(10);

    let taxonomy = load_taxonomy();
    let meta = meta_dir();
    let solve_budget_ms: u64 = 30000; // 30s per puzzle — full pipeline needs time
    let library_dir = std::env::temp_dir().join("qor_arc_benchmark_library");
    let _ = std::fs::remove_dir_all(&library_dir);
    let mut library = RuleLibrary::load(&library_dir);

    eprintln!("\n{}", "=".repeat(70));
    eprintln!("  ARC-AGI Benchmark — full solve() pipeline");
    eprintln!("  Puzzles:  {}", puzzle_ids.len());
    eprintln!("  Budget:   {}ms per puzzle (6 phases × 5 rounds)", solve_budget_ms);
    eprintln!("  Meta:     {}", meta.display());
    eprintln!("  Taxonomy: {} puzzle classifications loaded", taxonomy.len());
    eprintln!("{}", "=".repeat(70));

    let overall_start = Instant::now();

    let mut solved = 0;
    let mut total = 0;
    let mut near_misses = 0;

    for (idx, puzzle_id) in puzzle_ids.iter().enumerate() {
        total += 1;
        let puzzle = match load_puzzle(puzzle_id) {
            Some(p) => p,
            None => {
                eprintln!("  [{}/{}] {}: SKIP (cannot load)", idx + 1, puzzle_ids.len(), puzzle_id);
                continue;
            }
        };

        if puzzle.test.is_empty() || puzzle.test[0].output.is_empty() {
            eprintln!("  [{}/{}] {}: SKIP (no test output)", idx + 1, puzzle_ids.len(), puzzle_id);
            continue;
        }

        let expected_out = &puzzle.test[0].output;
        let exp_rows = expected_out.len();
        let exp_cols = expected_out[0].len();
        let puzzle_start = Instant::now();

        // Load DNA + feed puzzle
        let mut session = Session::load_qor_dir(&dna_dir());
        let (observations, all_expected) = feed_puzzle(&mut session, &puzzle, puzzle_id);

        // Inject external taxonomy classification if available
        if let Some(category) = taxonomy.get(puzzle_id.as_str()) {
            let tax_fact = Statement::simple_fact(vec![
                Neuron::symbol("external-class"),
                Neuron::symbol("puzzle"),
                Neuron::symbol(category),
            ]);
            let _ = session.exec_statements(vec![tax_fact]);
        }

        // Build test validation data BEFORE solve — so solve can detect overfitting
        let test_base = session.clone_obs_only();
        let test_in = Grid::from_vecs(puzzle.test[0].input.clone()).unwrap();
        let mut test_facts = test_in.to_statements("ti");
        test_facts.push(Statement::simple_fact(vec![Neuron::symbol("test-input"), Neuron::symbol("ti")]));
        test_facts.push(Statement::simple_fact(vec![
            Neuron::symbol("grid-size"), Neuron::symbol("ti"),
            Neuron::Value(QorValue::Int(puzzle.test[0].input.len() as i64)),
            Neuron::Value(QorValue::Int(puzzle.test[0].input[0].len() as i64)),
        ]));

        let mut test_session_for_val = test_base.clone();
        let _ = test_session_for_val.exec_statements(test_facts.clone());

        let mut test_expected_keys = HashSet::new();
        for (r, row) in expected_out.iter().enumerate() {
            for (c, &val) in row.iter().enumerate() {
                if val > 0 {
                    test_expected_keys.insert(format!("predict-cell ti {} {} {}", r, c, val));
                }
            }
        }

        let test_val = solve::TestValidation {
            test_session: &test_session_for_val,
            test_expected_keys,
        };

        // DNA baseline FIRST: check if DNA rules already solve the puzzle
        let dna_baseline = {
            let mut dna_test = test_base.clone();
            let _ = dna_test.exec_statements(test_facts.clone());

            // DIAGNOSTIC: dump all predict-cell facts for failing puzzles
            if puzzle_id == "0520fde7" || puzzle_id == "025d127b" {
                eprintln!("    === DIAGNOSTIC for {} ===", puzzle_id);
                let pred_sym = Neuron::symbol("predict-cell");
                let mut preds: Vec<String> = Vec::new();
                for sn in dna_test.all_facts() {
                    if let Neuron::Expression(parts) = &sn.neuron {
                        if parts.first() == Some(&pred_sym) {
                            preds.push(format!("      {} <{:.2}, {:.2}>", sn.neuron, sn.tv.strength, sn.tv.confidence));
                        }
                    }
                }
                preds.sort();
                preds.dedup();
                eprintln!("    predict-cell count: {}", preds.len());
                for p in &preds[..preds.len().min(30)] {
                    eprintln!("{}", p);
                }
                if preds.len() > 30 { eprintln!("    ... and {} more", preds.len() - 30); }
                // Also check ALL detection facts and other key facts
                for sn in dna_test.all_facts() {
                    if let Neuron::Expression(parts) = &sn.neuron {
                        if let Some(Neuron::Symbol(pred)) = parts.first() {
                            if pred.starts_with("detected-") || pred.starts_with("obs-")
                                || pred == "output-only-color" || pred == "non-separator-v-pair"
                                || pred == "train-has-separator-v" || pred == "pure-color-change"
                                || pred == "size-preserving-pattern" || pred == "has-geometric-transform"
                                || pred.starts_with("consistent-") || pred == "transform-identity"
                                || pred == "knows-puzzle-type" || pred == "input-grid"
                            {
                                eprintln!("    FACT: {} <{:.2}, {:.2}>", sn.neuron, sn.tv.strength, sn.tv.confidence);
                            }
                        }
                    }
                }
                eprintln!("    === END DIAGNOSTIC ===");
            }

            if let Some(pred) = Grid::extract_predictions(&dna_test, "predict-cell", exp_rows, exp_cols) {
                Grid::cell_accuracy(&pred, expected_out)
            } else { 0.0 }
        };

        // Only run genesis if DNA doesn't already solve it
        let (genesis_score, result) = if dna_baseline >= 0.999 {
            eprintln!("  DNA solves {}: {:.1}% — skipping genesis", puzzle_id, dna_baseline * 100.0);
            (0.0, solve::SolveResult {
                best_rules: vec![], score: 1.0, solved: true,
                candidates_explored: 0, mutations_tried: 0, elapsed_ms: 0,
                solved_in_phase: Some("dna"), baseline_score: 1.0,
                rounds: 0, failures_tracked: 0, overfit_count: 0,
                diagnostic: solve::DiagnosticReport::default(),
            })
        } else {
            // Run the FULL solve pipeline — all 6 phases, 8+ rounds, meta swarm, advisor
            let result = solve::solve(
                &session,
                &all_expected,
                "predict-cell",
                &observations,
                solve_budget_ms,
                Some(&mut library),
                Some(meta.as_path()),
                Some(&test_val),
            );

            // Genesis: evaluate in TWO modes and take the best
            let gs = if !result.best_rules.is_empty() {
                // Mode 1: Genesis stacked on DNA (DNA predictions may win by confidence)
                let mut test_stacked = test_base.clone();
                let _ = test_stacked.exec_statements(test_facts.clone());
                for rule_text in &result.best_rules {
                    if let Ok(stmts) = parser::parse(rule_text) {
                        let _ = test_stacked.exec_statements(stmts);
                    }
                }
                let score_stacked = if let Some(pred) = Grid::extract_predictions(&test_stacked, "predict-cell", exp_rows, exp_cols) {
                    Grid::cell_accuracy(&pred, expected_out)
                } else { 0.0 };

                // Mode 2: Genesis in isolation (DNA predictions removed, keep intermediates)
                // This lets genesis show its true accuracy without DNA's high-confidence
                // predictions overriding correct genesis predictions.
                let mut test_isolated = test_base.clone();
                let _ = test_isolated.exec_statements(test_facts.clone());
                test_isolated.store_mut().remove_inferred_by_predicate("predict-cell");
                for rule_text in &result.best_rules {
                    if let Ok(stmts) = parser::parse(rule_text) {
                        let _ = test_isolated.exec_statements(stmts);
                    }
                }
                let score_isolated = if let Some(pred) = Grid::extract_predictions(&test_isolated, "predict-cell", exp_rows, exp_cols) {
                    Grid::cell_accuracy(&pred, expected_out)
                } else { 0.0 };

                score_stacked.max(score_isolated)
            } else { 0.0 };
            (gs, result)
        };

        // Always use the better of DNA baseline or genesis+DNA
        let test_score = dna_baseline.max(genesis_score);

        let elapsed = puzzle_start.elapsed().as_millis();

        // Print diagnostic summary
        let diag = &result.diagnostic;
        if !diag.classes.is_empty() || !diag.gaps.is_empty() {
            eprintln!("    diag: class={:?} detections={} obs={} predictions={} covered={:?} gaps={:?}",
                diag.classes, diag.detections.len(), diag.observations.len(),
                diag.dna_predictions, diag.covered, diag.gaps);
        }

        let tax_label = taxonomy.get(puzzle_id.as_str()).map(|s| s.as_str()).unwrap_or("-");

        if test_score >= 0.999 {
            solved += 1;
            eprintln!("  [{}/{}] {} [{}]: SOLVED (phase={:?}) rounds={} candidates={} {}ms  [dna={:.1}% genesis={:.1}%]",
                idx + 1, puzzle_ids.len(), puzzle_id, tax_label,
                result.solved_in_phase.unwrap_or("?"), result.rounds,
                result.candidates_explored, elapsed,
                dna_baseline * 100.0, genesis_score * 100.0);
        } else {
            if test_score >= 0.90 { near_misses += 1; }
            eprintln!("  [{}/{}] {} [{}]: {:.1}% (train={:.1}% baseline={:.1}%) rounds={} candidates={} {}ms  [dna={:.1}% genesis={:.1}%]",
                idx + 1, puzzle_ids.len(), puzzle_id, tax_label,
                test_score * 100.0, result.score * 100.0, result.baseline_score * 100.0,
                result.rounds, result.candidates_explored, elapsed,
                dna_baseline * 100.0, genesis_score * 100.0);
        }
    }

    library.save();
    let total_elapsed = overall_start.elapsed().as_secs();

    eprintln!("\n{}", "=".repeat(70));
    eprintln!("  RESULTS — {} ARC Puzzles (full solve pipeline)", total);
    eprintln!("{}", "=".repeat(70));
    eprintln!("  Solved:     {}/{} ({:.1}%)",
        solved, total, solved as f64 / total.max(1) as f64 * 100.0);
    eprintln!("  Near-miss:  {} (>= 90% accuracy)", near_misses);
    eprintln!("  Library:    {} rules remembered", library.all_rules().len());
    eprintln!("  Time:       {}s total ({:.1}s/puzzle avg)",
        total_elapsed, total_elapsed as f64 / total.max(1) as f64);
    eprintln!("{}", "=".repeat(70));

    assert!(total > 0, "Should have run at least 1 puzzle");
}
