//! Web Flow End-to-End Test
//! =========================
//! Tests the FULL agent pipeline with 5 real ARC puzzles:
//!   1. Load puzzle → feed facts → QOR reasons → collect answer
//!   2. If wrong → confidence drops → inject (solve-confidence)
//!   3. When confidence < 0.50 → DNA fires (web-lookup-needed)
//!   4. Simulate web results → new grid facts injected
//!   5. QOR re-reasons with web-enriched facts
//!
//! No actual browser/network — we simulate web responses by injecting
//! the test output grid as if the web returned it. This tests the
//! PLUMBING: does the confidence → web → re-reason loop work end-to-end.

use qor_bridge::grid::Grid;
use qor_core::neuron::{Neuron, Statement};
use qor_runtime::eval::Session;
use std::path::Path;

// ── ARC JSON ────────────────────────────────────────────────────────

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

// ── The actual test data ────────────────────────────────────────────

const PUZZLE_IDS: [&str; 5] = [
    "00576224",  // 2x2→6x6 tiling
    "0520fde7",  // 3x7→3x3 separator mask
    "017c7c7b",  // 3x3→9x3 pattern replication
    "05269061",  // 7x7→7x7 diagonal fill
    "0692e18c",  // 3x3→9x9 grid composition
];

fn load_puzzle(id: &str) -> ArcPuzzle {
    let path = Path::new("D:\\QOR-LANG\\ARC-AGI-2\\data\\training")
        .join(format!("{id}.json"));
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {e}", path.display()));
    serde_json::from_str(&data)
        .unwrap_or_else(|e| panic!("Cannot parse {}: {e}", path.display()))
}

/// Feed an ARC puzzle into a QOR session (training pairs + test input).
fn feed_puzzle(session: &mut Session, puzzle: &ArcPuzzle, puzzle_id: &str) {
    let mut all_facts = Vec::new();

    // Training pairs
    for (i, pair) in puzzle.train.iter().enumerate() {
        let in_id = format!("t{}i", i);
        let out_id = format!("t{}o", i);
        let in_grid = Grid::from_vecs(pair.input.clone()).unwrap();
        let out_grid = Grid::from_vecs(pair.output.clone()).unwrap();

        all_facts.push(Statement::simple_fact(vec![Neuron::symbol("train-pair"), Neuron::symbol(&in_id), Neuron::symbol(&out_id)]));
        all_facts.extend(in_grid.to_statements(&in_id));
        all_facts.extend(out_grid.to_statements(&out_id));

        let compare = Grid::compare_pair(&in_grid, &out_grid, &in_id, &out_id);
        all_facts.extend(compare);
    }

    // Test input
    let test_in = Grid::from_vecs(puzzle.test[0].input.clone()).unwrap();
    all_facts.push(Statement::simple_fact(vec![Neuron::symbol("test-input"), Neuron::symbol("ti")]));
    all_facts.extend(test_in.to_statements("ti"));

    // Item ID
    all_facts.push(Statement::simple_fact(vec![Neuron::symbol("item-id"), Neuron::symbol(puzzle_id)]));

    let _ = session.exec_statements(all_facts);
}

// ═════════════════════════════════════════════════════════════════════
// TEST: Full multi-attempt loop with 5 real ARC puzzles
// ═════════════════════════════════════════════════════════════════════

#[test]
fn web_flow_5_puzzles() {
    let dna_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("dna").join("puzzle_solver");

    let max_attempts = 3;
    let mut results: Vec<(String, Vec<f64>, bool, bool)> = Vec::new();

    for &puzzle_id in &PUZZLE_IDS {
        eprintln!("\n{}", "=".repeat(60));
        eprintln!("  PUZZLE: {puzzle_id}");

        let puzzle = load_puzzle(puzzle_id);
        let expected_out = &puzzle.test[0].output;
        let exp_rows = expected_out.len();
        let exp_cols = expected_out[0].len();

        let mut scores: Vec<f64> = Vec::new();
        let mut web_triggered = false;
        let mut solved = false;

        for attempt in 1..=max_attempts {
            // Fresh session each attempt (like the real agent)
            let mut session = Session::load_qor_dir(&dna_dir);

            // Inject attempt tracking
            let _ = session.exec(&format!("(attempt-number {} {})", puzzle_id, attempt));
            if attempt > 1 {
                let last_score = scores.last().copied().unwrap_or(0.0);
                let _ = session.exec(&format!(
                    "(solve-confidence {} {:.4}) <{:.2}, 0.95>",
                    puzzle_id, last_score, last_score
                ));
                for prev in 1..attempt {
                    let _ = session.exec(&format!("(attempt-failed {} {})", puzzle_id, prev));
                }
            }

            // Feed puzzle data
            feed_puzzle(&mut session, &puzzle, puzzle_id);

            // ── Check: did QOR trigger web lookup? ──
            let wants_web = session.has_fact("web-lookup-needed");
            let has_browser_action = session.has_fact("browser-action");

            if wants_web || has_browser_action {
                web_triggered = true;
                eprintln!("  Attempt {attempt}: WEB TRIGGERED (confidence < 0.50)");

                // Check strategy
                let strategy_web = session.all_facts().iter().any(|f| {
                    f.neuron.to_string().contains("strategy") &&
                    f.neuron.to_string().contains("web-lookup")
                });
                assert!(strategy_web || wants_web,
                    "when web triggers, strategy should be web-lookup");

                // ── Simulate web response ──
                // In reality, the browser would navigate to arcprize.org and
                // extract grid data. We simulate by injecting the test output
                // as "web-discovered" facts — this is what the JS extraction
                // would return from the page.
                eprintln!("  Attempt {attempt}: Simulating web grid data injection...");

                // Mark page visited
                let _ = session.exec(&format!(
                    "(page-url \"https://arcprize.org/play?task={}\") <0.95, 0.90>",
                    puzzle_id
                ));
                // Inject some page elements to trigger web-snapshot-taken
                let _ = session.exec(
                    "(page-element s1 button submit none) <0.95, 0.90>"
                );
                // Mark JS extracted
                let _ = session.exec(
                    "(js-result \"extract-grids\" \"grid-data\") <0.95, 0.90>"
                );

                // Inject web-extracted grid cells as hints
                // (In real flow, browser JS would extract these from the DOM)
                // We inject training output grids as "web-hint" facts
                for (i, pair) in puzzle.train.iter().enumerate() {
                    for (r, row) in pair.output.iter().enumerate() {
                        for (c, &val) in row.iter().enumerate() {
                            let _ = session.exec(&format!(
                                "(web-hint-cell {} {} {} {}) <0.80, 0.80>",
                                i, r, c, val
                            ));
                        }
                    }
                }

                // Run another heartbeat after web data injection
                session.heartbeat();
            }

            // ── Collect answer ──
            let score = if let Some(predicted) = Grid::extract_predictions(&session, "predict-cell", exp_rows, exp_cols) {
                let acc = Grid::cell_accuracy(&predicted, expected_out);
                eprintln!("  Attempt {attempt}: {:.1}% accuracy ({} predict-cells)",
                    acc * 100.0, session.facts_with_predicate("predict-cell").len());
                acc
            } else {
                eprintln!("  Attempt {attempt}: No predict-cell facts produced");
                0.0
            };

            scores.push(score);

            if score >= 0.999 {
                solved = true;
                eprintln!("  SOLVED on attempt {attempt}!");
                break;
            }
        }

        results.push((puzzle_id.to_string(), scores, web_triggered, solved));
    }

    // ── Summary ─────────────────────────────────────────────────────
    eprintln!("\n{}", "=".repeat(60));
    eprintln!("  WEB FLOW E2E SUMMARY — 5 puzzles, {} max attempts", max_attempts);
    eprintln!("{}", "=".repeat(60));

    let mut web_trigger_count = 0;
    let mut solved_count = 0;

    for (id, scores, web_triggered, solved) in &results {
        let best = scores.iter().copied().fold(0.0_f64, f64::max);
        let web_tag = if *web_triggered { " [WEB]" } else { "" };
        let solved_tag = if *solved { " SOLVED" } else { "" };
        eprintln!("  {id}: best={:.1}% attempts={}{}{}",
            best * 100.0, scores.len(), web_tag, solved_tag);
        if *web_triggered { web_trigger_count += 1; }
        if *solved { solved_count += 1; }
    }

    eprintln!("  Web triggered: {web_trigger_count}/5 puzzles");
    eprintln!("  Solved: {solved_count}/5 puzzles");
    eprintln!("{}", "=".repeat(60));

    // ── Assertions ──────────────────────────────────────────────────

    // At least SOME puzzles should trigger web (confidence drops below 0.50)
    // These are hard puzzles — QOR won't solve them on attempt 1
    assert!(web_trigger_count >= 1,
        "At least 1 puzzle should trigger web lookup (confidence < 0.50). Got {web_trigger_count}");

    // Puzzles that produce predictions should have them on every attempt.
    // Puzzles that produce nothing (QOR lacks rules for that transform type)
    // won't trigger web either because score stays at 0.0 (no predictions
    // to be wrong about — confidence undefined, not "low").
    // That's correct behavior: web is for "I tried but scored poorly",
    // not for "I have no applicable rules at all".

    // Verify the flow mechanics work:
    // - Attempt 1: no confidence → strategy try-all
    // - Attempt 2+: if score < 0.50 → web-lookup-needed fires
    for (id, scores, web_triggered, _) in &results {
        if scores.len() >= 2 {
            let first_score = scores[0];
            if first_score < 0.50 && first_score > 0.0 {
                // Score was below 0.50 after attempt 1, web should trigger on attempt 2
                assert!(*web_triggered,
                    "Puzzle {id}: score {:.1}% < 50% but web didn't trigger",
                    first_score * 100.0);
            }
        }
    }
}
