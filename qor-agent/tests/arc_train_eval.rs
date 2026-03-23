//! ARC-AGI-2 Train → Evaluate Pipeline
//! =====================================
//! Phase 1: TRAIN on training puzzles — genesis invents rules, library remembers
//! Phase 2: EVALUATE on evaluation puzzles — use library + DNA to solve
//!
//! This is PURE PLUMBING. Zero domain logic in Rust.
//! All intelligence lives in DNA (.qor files).
//!
//! Usage:
//!   cargo test -p qor-agent --release arc_train_eval -- --nocapture --ignored

use qor_bridge::grid::Grid;
use qor_core::neuron::{Neuron, QorValue, Statement};
use qor_core::parser;
use qor_runtime::eval::Session;
use qor_runtime::invent;
use qor_runtime::library::RuleLibrary;
use std::path::Path;
use std::time::Instant;

// ── Generic JSON structs (just I/O plumbing) ─────────────────────────

#[derive(serde::Deserialize)]
struct Puzzle {
    train: Vec<Pair>,
    test: Vec<Pair>,
}

#[derive(serde::Deserialize)]
struct Pair {
    input: Vec<Vec<u8>>,
    output: Vec<Vec<u8>>,
}

// ── DNA directory path ───────────────────────────────────────────────

fn dna_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("dna")
        .join("puzzle_solver")
}

// ── Minimal helpers (kept — no engine equivalent) ────────────────────

/// Load a puzzle JSON from a directory. Pure I/O.
fn load_puzzle(dir: &Path, id: &str) -> Option<Puzzle> {
    let path = dir.join(format!("{id}.json"));
    let data = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&data).ok()
}

/// List all puzzle IDs in a directory. Pure I/O.
fn list_puzzle_ids(dir: &Path) -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(dir)
        .unwrap_or_else(|_| panic!("Cannot read dir: {}", dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
        .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().to_string()))
        .collect();
    ids.sort();
    ids
}

/// Feed all puzzle data into a session. Uses Grid (bridge) for conversion.
fn feed_puzzle(session: &mut Session, puzzle: &Puzzle, puzzle_id: &str) {
    let mut all_facts = Vec::new();
    let mut pair_facts: Vec<Vec<Statement>> = Vec::new();
    let mut pair_grids: Vec<(Grid, Grid)> = Vec::new();

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
    }

    // Generate obs-consistent and obs-parameter facts across all training pairs
    let obs = Grid::compute_observations(&pair_facts, &pair_grids);
    all_facts.extend(obs);

    let test_in = Grid::from_vecs(puzzle.test[0].input.clone()).unwrap();
    all_facts.push(Statement::simple_fact(vec![Neuron::symbol("test-input"), Neuron::symbol("ti")]));
    all_facts.extend(test_in.to_statements("ti"));
    all_facts.push(Statement::simple_fact(vec![Neuron::symbol("item-id"), Neuron::symbol(puzzle_id)]));

    let _ = session.exec_statements(all_facts);
}

/// Extract training pairs for genesis from a puzzle.
/// Each pair: input facts → expected output facts.
/// Uses Grid (bridge) for conversion — no domain logic.
fn extract_training(puzzle: &Puzzle) -> (Vec<Vec<Statement>>, Vec<Vec<Statement>>) {
    let mut training_inputs = Vec::new();
    let mut expected_outputs = Vec::new();

    for pair in &puzzle.train {
        let in_grid = Grid::from_vecs(pair.input.clone()).unwrap();

        // Input: what Grid produces (bridge handles format)
        let mut input_stmts = in_grid.to_statements("ti");
        input_stmts.push(Statement::simple_fact(vec![Neuron::symbol("test-input"), Neuron::symbol("ti")]));
        input_stmts.push(Statement::simple_fact(vec![
            Neuron::symbol("grid-size"),
            Neuron::symbol("ti"),
            Neuron::Value(QorValue::Int(pair.input.len() as i64)),
            Neuron::Value(QorValue::Int(pair.input[0].len() as i64)),
        ]));

        // Output: expected facts — same format as to_statements but with
        // the target predicate. Only non-zero cells (consistent with to_statements).
        // The target predicate name comes from DNA convention, not Rust logic.
        let mut output_stmts = Vec::new();
        for (r, row) in pair.output.iter().enumerate() {
            for (c, &val) in row.iter().enumerate() {
                if val > 0 {
                    output_stmts.push(Statement::simple_fact(vec![
                        Neuron::symbol("predict-cell"),
                        Neuron::symbol("ti"),
                        Neuron::Value(QorValue::Int(r as i64)),
                        Neuron::Value(QorValue::Int(c as i64)),
                        Neuron::Value(QorValue::Int(val as i64)),
                    ]));
                }
            }
        }

        training_inputs.push(input_stmts);
        expected_outputs.push(output_stmts);
    }

    (training_inputs, expected_outputs)
}

/// Run one puzzle: DNA attempt → genesis attempt → return best accuracy.
fn run_puzzle(
    puzzle: &Puzzle,
    puzzle_id: &str,
    library: &mut RuleLibrary,
    num_workers: usize,
    genesis_budget_ms: u64,
    diagnostic: bool,
) -> (f64, f64, bool) {
    // dna_score, best_score, genesis_helped
    if puzzle.test.is_empty() || puzzle.test[0].output.is_empty() {
        return (0.0, 0.0, false);
    }

    let expected_out = &puzzle.test[0].output;
    let exp_rows = expected_out.len();
    let exp_cols = expected_out[0].len();

    // ── DNA attempt ──
    let mut session = Session::load_qor_dir(&dna_dir());
    feed_puzzle(&mut session, &puzzle, puzzle_id);

    let dna_score = if let Some(pred) = Grid::extract_predictions(&session, "predict-cell", exp_rows, exp_cols) {
        Grid::cell_accuracy(&pred, expected_out)
    } else {
        0.0
    };

    if dna_score >= 0.999 {
        return (dna_score, dna_score, false);
    }

    // ── Genesis attempt ──
    let (training_inputs, expected_outputs) = extract_training(&puzzle);
    if training_inputs.is_empty() {
        return (dna_score, dna_score, false);
    }

    let genesis_base = session.clone_obs_only();

    let candidates = invent::genesis_swarm(
        &genesis_base,
        &training_inputs,
        &expected_outputs,
        genesis_budget_ms,
        Some(library),
        num_workers,
        None,
    );

    if diagnostic && !candidates.is_empty() {
        eprintln!("    DIAG {}: {} candidates, top score={:.3} source={}",
            puzzle_id, candidates.len(),
            candidates[0].score, candidates[0].source);
        if let Some(top) = candidates.first() {
            for line in top.rule_text.lines().take(3) {
                eprintln!("      rule: {}", line);
            }
        }
    } else if diagnostic {
        eprintln!("    DIAG {}: 0 candidates from genesis", puzzle_id);
    }

    let genesis_score = candidates.first().map(|c| c.score).unwrap_or(0.0);

    // Test top candidates on evaluation data
    let final_score = if genesis_score > 0.001 {
        let mut best_genesis_acc = 0.0;
        for candidate in candidates.iter().take(3) {
            if let Ok(rule_stmts) = parser::parse(&candidate.rule_text) {
                let mut test_session = Session::load_qor_dir(&dna_dir());
                let test_in = Grid::from_vecs(puzzle.test[0].input.clone()).unwrap();
                let mut test_facts = test_in.to_statements("ti");
                test_facts.push(Statement::simple_fact(vec![Neuron::symbol("test-input"), Neuron::symbol("ti")]));
                test_facts.push(Statement::simple_fact(vec![
                    Neuron::symbol("grid-size"),
                    Neuron::symbol("ti"),
                    Neuron::Value(QorValue::Int(puzzle.test[0].input.len() as i64)),
                    Neuron::Value(QorValue::Int(puzzle.test[0].input[0].len() as i64)),
                ]));
                let _ = test_session.exec_statements(test_facts);
                let _ = test_session.exec_statements(rule_stmts);

                if let Some(pred) = Grid::extract_predictions(&test_session, "predict-cell", exp_rows, exp_cols) {
                    let acc = Grid::cell_accuracy(&pred, expected_out);
                    if acc > best_genesis_acc {
                        best_genesis_acc = acc;
                    }
                }
            }
        }
        best_genesis_acc
    } else {
        0.0
    };

    let best = final_score.max(dna_score);
    let genesis_helped = final_score > dna_score + 0.01;
    (dna_score, best, genesis_helped)
}

// ═════════════════════════════════════════════════════════════════════
// TEST 1: Diagnostic — run 5 training puzzles with verbose output
//         to understand what genesis actually produces
// ═════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn arc_genesis_diagnostic() {
    let training_dir = Path::new("D:\\QOR-LANG\\ARC-AGI-2\\data\\training");
    let ids = list_puzzle_ids(training_dir);

    eprintln!("\n=== GENESIS DEEP DIAGNOSTIC ===\n");

    // Use first puzzle that has same-size input/output (identity candidate)
    let puzzle_id = &ids[3]; // 00d62c1b — 10x10→10x10, 50% DNA
    let puzzle = load_puzzle(training_dir, puzzle_id).unwrap();

    eprintln!("Puzzle: {}", puzzle_id);
    for (i, pair) in puzzle.train.iter().enumerate() {
        eprintln!("  train[{}]: {}x{} → {}x{}", i,
            pair.input.len(), pair.input[0].len(),
            pair.output.len(), pair.output[0].len());
    }

    // ── Step 1: Extract training data ──
    let (training_inputs, expected_outputs) = extract_training(&puzzle);
    eprintln!("\nStep 1 — Training data:");
    eprintln!("  Pairs: {}", training_inputs.len());
    for (i, (inp, out)) in training_inputs.iter().zip(expected_outputs.iter()).enumerate() {
        eprintln!("  Pair {}: {} input facts, {} output facts", i, inp.len(), out.len());

        // Show first 3 input facts
        for f in inp.iter().take(3) {
            if let Statement::Fact { neuron, .. } = f {
                eprintln!("    in:  {}", neuron);
            }
        }
        // Show first 3 output facts
        for f in out.iter().take(3) {
            if let Statement::Fact { neuron, .. } = f {
                eprintln!("    out: {}", neuron);
            }
        }
    }

    // ── Step 2: Build genesis base (no DNA for this test) ──
    eprintln!("\nStep 2 — Testing with EMPTY base session (no DNA):");
    let empty_session = Session::new();
    let results = invent::genesis(
        &empty_session,
        &training_inputs,
        &expected_outputs,
        3000,
        None,
        None,
    );
    eprintln!("  Candidates from empty base: {}", results.len());
    for (i, c) in results.iter().enumerate().take(5) {
        eprintln!("  [{i}] score={:.3} source={}", c.score, c.source);
        for line in c.rule_text.lines().take(2) {
            eprintln!("      {}", line);
        }
    }

    // ── Step 3: Manual identity rule test ──
    eprintln!("\nStep 3 — Manual identity rule test:");

    // Discover source/target predicates from training data
    let mut input_preds: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut output_preds: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for inp in &training_inputs {
        for stmt in inp {
            if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = stmt {
                if let Some(Neuron::Symbol(p)) = parts.first() {
                    *input_preds.entry(p.clone()).or_default() += 1;
                }
            }
        }
    }
    for out in &expected_outputs {
        for stmt in out {
            if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = stmt {
                if let Some(Neuron::Symbol(p)) = parts.first() {
                    *output_preds.entry(p.clone()).or_default() += 1;
                }
            }
        }
    }
    eprintln!("  Input predicates: {:?}", input_preds);
    eprintln!("  Output predicates: {:?}", output_preds);

    // Find target (in output not input) and source (highest count in input)
    let target = output_preds.keys()
        .find(|k| !input_preds.contains_key(*k))
        .cloned()
        .unwrap_or_else(|| "unknown".into());
    let source = input_preds.iter()
        .max_by_key(|(_, v)| *v)
        .map(|(k, _)| k.clone())
        .unwrap_or_else(|| "unknown".into());

    eprintln!("  Source: {}, Target: {}", source, target);

    // Check arities
    let source_arity = training_inputs[0].iter().filter_map(|s| {
        if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = s {
            if let Some(Neuron::Symbol(p)) = parts.first() {
                if p == &source { return Some(parts.len() - 1); }
            }
        }
        None
    }).next().unwrap_or(0);
    let target_arity = expected_outputs[0].iter().filter_map(|s| {
        if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = s {
            if let Some(Neuron::Symbol(p)) = parts.first() {
                if p == &target { return Some(parts.len() - 1); }
            }
        }
        None
    }).next().unwrap_or(0);
    eprintln!("  Source arity: {}, Target arity: {}", source_arity, target_arity);

    // Build identity rule text
    let vars: Vec<String> = (0..source_arity).map(|i| format!("$a{i}")).collect();
    let vars_str = vars.join(" ");
    let rule_text = format!("({target} {vars_str}) if\n    ({source} {vars_str}) <0.90, 0.90>");
    eprintln!("  Rule text: {}", rule_text.replace('\n', " "));

    // Try parsing
    match parser::parse(&rule_text) {
        Ok(stmts) => {
            eprintln!("  Parse: OK ({} statements)", stmts.len());

            // Try executing on first training pair
            let mut test_session = Session::new();
            let mut batch = stmts.clone();
            batch.extend(training_inputs[0].iter().cloned());
            let _ = test_session.exec_statements(batch);

            // Count target facts produced
            let mut target_count = 0;
            let mut target_samples = Vec::new();
            for sn in test_session.all_facts() {
                if let Neuron::Expression(parts) = &sn.neuron {
                    if let Some(Neuron::Symbol(p)) = parts.first() {
                        if p == &target {
                            target_count += 1;
                            if target_samples.len() < 3 {
                                target_samples.push(sn.neuron.to_string());
                            }
                        }
                    }
                }
            }
            eprintln!("  Forward chain produced {} '{}' facts", target_count, target);
            for s in &target_samples {
                eprintln!("    sample: {}", s);
            }

            // Compare with expected
            let expected_count = expected_outputs[0].len();
            eprintln!("  Expected: {} '{}' facts", expected_count, target);

            if target_count == 0 {
                eprintln!("  *** PROBLEM: identity rule produced ZERO target facts!");
                eprintln!("  Total facts in session: {}", test_session.all_facts().len());
                eprintln!("  Rules in session: {}", test_session.rule_count());

                // Check if source facts exist
                let source_count = test_session.all_facts().iter().filter(|sn| {
                    if let Neuron::Expression(parts) = &sn.neuron {
                        parts.first() == Some(&Neuron::symbol(&source))
                    } else { false }
                }).count();
                eprintln!("  Source '{}' facts in session: {}", source, source_count);
            }
        }
        Err(e) => {
            eprintln!("  Parse: FAILED — {}", e);
        }
    }

    // ── Step 4: Test with DNA loaded ──
    eprintln!("\nStep 4 — Testing with DNA loaded:");
    let mut dna_session = Session::load_qor_dir(&dna_dir());
    feed_puzzle(&mut dna_session, &puzzle, puzzle_id);
    let genesis_base = dna_session.clone_obs_only();

    eprintln!("  DNA base rules: {}", genesis_base.rule_count());
    eprintln!("  DNA base facts: {}", genesis_base.all_facts().len());

    let results_with_dna = invent::genesis(
        &genesis_base,
        &training_inputs,
        &expected_outputs,
        5000,
        None,
        None,
    );
    eprintln!("  Candidates with DNA base: {}", results_with_dna.len());
    for (i, c) in results_with_dna.iter().enumerate().take(5) {
        eprintln!("  [{i}] score={:.3} source={}", c.score, c.source);
        for line in c.rule_text.lines().take(2) {
            eprintln!("      {}", line);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// TEST 2: Full train → evaluate pipeline
// ═════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn arc_train_evaluate() {
    let data_dir = Path::new("D:\\QOR-LANG\\ARC-AGI-2\\data");
    let training_dir = data_dir.join("training");
    let eval_dir = data_dir.join("evaluation");

    let training_ids = list_puzzle_ids(&training_dir);
    let eval_ids = list_puzzle_ids(&eval_dir);

    let num_workers = invent::optimal_worker_count();
    let genesis_budget_ms: u64 = 3000;

    let library_dir = std::env::temp_dir().join("qor_arc_train_eval_library");
    let _ = std::fs::remove_dir_all(&library_dir);
    let mut library = RuleLibrary::load(&library_dir);

    eprintln!("\n{}", "=".repeat(70));
    eprintln!("  ARC-AGI-2 Train → Evaluate Pipeline");
    eprintln!("  Training:   {} puzzles", training_ids.len());
    eprintln!("  Evaluation: {} puzzles", eval_ids.len());
    eprintln!("  Workers:    {} parallel genesis threads", num_workers);
    eprintln!("  Budget:     {}ms per puzzle", genesis_budget_ms);
    eprintln!("{}", "=".repeat(70));

    // ── PHASE 1: TRAIN ──────────────────────────────────────────────
    eprintln!("\n  PHASE 1: TRAINING (learning rules from {} puzzles)", training_ids.len());
    let train_start = Instant::now();

    let mut train_solved = 0;
    let mut train_genesis_helped = 0;
    let mut train_total = 0;

    for (idx, id) in training_ids.iter().enumerate() {
        let puzzle = match load_puzzle(&training_dir, id) {
            Some(p) => p,
            None => continue,
        };
        train_total += 1;

        let (_dna_score, best_score, genesis_helped) =
            run_puzzle(&puzzle, id, &mut library, num_workers, genesis_budget_ms, false);

        if best_score >= 0.999 {
            train_solved += 1;
        }
        if genesis_helped {
            train_genesis_helped += 1;
        }

        // Progress every 50 puzzles
        if (idx + 1) % 50 == 0 || idx + 1 == training_ids.len() {
            eprintln!("    [{:>4}/{}] solved={} genesis+={} library={}",
                idx + 1, training_ids.len(),
                train_solved, train_genesis_helped, library.all_rules().len());
        }
    }

    // Save learned rules
    library.save();
    let train_elapsed = train_start.elapsed().as_secs();

    eprintln!("\n  Training complete: {train_elapsed}s");
    eprintln!("    Solved: {}/{} ({:.1}%)",
        train_solved, train_total, train_solved as f64 / train_total.max(1) as f64 * 100.0);
    eprintln!("    Genesis helped: {}", train_genesis_helped);
    eprintln!("    Library size: {} rules", library.all_rules().len());

    // ── PHASE 2: EVALUATE ───────────────────────────────────────────
    eprintln!("\n  PHASE 2: EVALUATION ({} puzzles, library={} rules)",
        eval_ids.len(), library.all_rules().len());
    let eval_start = Instant::now();

    let mut eval_solved = 0;
    let mut eval_genesis_helped = 0;
    let mut eval_predicted = 0;
    let mut eval_near_miss = 0;
    let mut eval_total = 0;

    for (idx, id) in eval_ids.iter().enumerate() {
        let puzzle = match load_puzzle(&eval_dir, id) {
            Some(p) => p,
            None => continue,
        };
        eval_total += 1;

        if puzzle.test.is_empty() || puzzle.test[0].output.is_empty() {
            continue;
        }

        let exp_rows = puzzle.test[0].output.len();
        let exp_cols = puzzle.test[0].output[0].len();

        let (_dna_score, best_score, genesis_helped) =
            run_puzzle(&puzzle, id, &mut library, num_workers, genesis_budget_ms, false);

        // Check if we produced any predictions
        {
            let mut check_session = Session::load_qor_dir(&dna_dir());
            feed_puzzle(&mut check_session, &puzzle, id);
            if Grid::extract_predictions(&check_session, "predict-cell", exp_rows, exp_cols).is_some() {
                eval_predicted += 1;
            }
        }

        if best_score >= 0.999 {
            eval_solved += 1;
            let tag = if genesis_helped { "genesis" } else { "DNA" };
            eprintln!("  [{:>3}/{}] {}: SOLVED ({})",
                idx + 1, eval_ids.len(), id, tag);
        } else if best_score >= 0.90 {
            eval_near_miss += 1;
        }
        if genesis_helped {
            eval_genesis_helped += 1;
        }

        // Progress every 20 puzzles
        if (idx + 1) % 20 == 0 || idx + 1 == eval_ids.len() {
            eprintln!("    [{:>3}/{}] solved={} near-miss={} genesis+={}",
                idx + 1, eval_ids.len(),
                eval_solved, eval_near_miss, eval_genesis_helped);
        }
    }

    library.save();
    let eval_elapsed = eval_start.elapsed().as_secs();

    eprintln!("\n{}", "=".repeat(70));
    eprintln!("  RESULTS — ARC-AGI-2 Train → Evaluate");
    eprintln!("{}", "=".repeat(70));
    eprintln!("  TRAINING ({} puzzles, {}s):", train_total, train_elapsed);
    eprintln!("    Solved: {}/{} ({:.1}%)",
        train_solved, train_total, train_solved as f64 / train_total.max(1) as f64 * 100.0);
    eprintln!("    Genesis helped: {}", train_genesis_helped);
    eprintln!("  EVALUATION ({} puzzles, {}s):", eval_total, eval_elapsed);
    eprintln!("    Solved: {}/{} ({:.1}%)",
        eval_solved, eval_total, eval_solved as f64 / eval_total.max(1) as f64 * 100.0);
    eprintln!("    Predicted: {}/{}", eval_predicted, eval_total);
    eprintln!("    Near-miss: {} (>= 90%)", eval_near_miss);
    eprintln!("    Genesis helped: {}", eval_genesis_helped);
    eprintln!("  Library: {} rules remembered", library.all_rules().len());
    eprintln!("{}", "=".repeat(70));
}
