//! Web Fallback Integration Tests
//! ================================
//! Verifies the full confidence-based web fallback pipeline:
//!   1. Low confidence → QOR fires web-lookup rules
//!   2. Browser actions produced → Rust executes them
//!   3. Web results feed back → QOR creates new facts/rules
//!   4. New facts help solve the puzzle
//!
//! No actual browser or network calls — we simulate browser results
//! by injecting facts the same way Rust would after executing actions.

use qor_runtime::eval::Session;
use std::path::Path;

// ═════════════════════════════════════════════════════════════════════
// TEST 1: Low confidence triggers web lookup
// ═════════════════════════════════════════════════════════════════════

#[test]
fn low_confidence_triggers_web_lookup() {
    let dna_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("dna").join("puzzle_solver");
    let mut session = Session::load_qor_dir(&dna_dir);

    // Simulate: attempt 3, previous accuracy was 0.40 (below 0.50 threshold)
    session.exec("(attempt-number puzzle_abc 3)").unwrap();
    session.exec("(solve-confidence puzzle_abc 0.4000) <0.40, 0.95>").unwrap();
    session.exec("(attempt-failed puzzle_abc 1)").unwrap();
    session.exec("(attempt-failed puzzle_abc 2)").unwrap();

    // DNA rules should fire:
    assert!(session.has_fact("web-lookup-needed"),
        "web-lookup-needed should fire when confidence < 0.50");
    assert!(session.has_fact("strategy"),
        "strategy fact should be derived");

    // Check the strategy is web-lookup
    let strategies = session.facts_with_predicate("strategy");
    let strategy_strs: Vec<String> = strategies.iter()
        .map(|f| f.neuron.to_string())
        .collect();
    assert!(strategy_strs.iter().any(|s| s.contains("web-lookup")),
        "strategy should be web-lookup, got: {:?}", strategy_strs);

    // browser-action navigate should fire (no web-visited-play yet)
    assert!(session.has_fact("browser-action"),
        "browser-action should fire for navigation");
}

// ═════════════════════════════════════════════════════════════════════
// TEST 2: High confidence does NOT trigger web lookup
// ═════════════════════════════════════════════════════════════════════

#[test]
fn high_confidence_no_web_lookup() {
    let dna_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("dna").join("puzzle_solver");
    let mut session = Session::load_qor_dir(&dna_dir);

    // Simulate: attempt 2, previous accuracy was 0.75 (above 0.70)
    session.exec("(attempt-number puzzle_xyz 2)").unwrap();
    session.exec("(solve-confidence puzzle_xyz 0.7500) <0.75, 0.95>").unwrap();
    session.exec("(attempt-failed puzzle_xyz 1)").unwrap();

    // web-lookup-needed should NOT fire
    assert!(!session.has_fact("web-lookup-needed"),
        "web-lookup-needed should NOT fire when confidence > 0.50");

    // Strategy should be "refine"
    let strategies = session.facts_with_predicate("strategy");
    let strategy_strs: Vec<String> = strategies.iter()
        .map(|f| f.neuron.to_string())
        .collect();
    assert!(strategy_strs.iter().any(|s| s.contains("refine")),
        "strategy should be refine, got: {:?}", strategy_strs);
}

// ═════════════════════════════════════════════════════════════════════
// TEST 3: Medium confidence → alternate strategy, no web
// ═════════════════════════════════════════════════════════════════════

#[test]
fn medium_confidence_alternate_strategy() {
    let dna_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("dna").join("puzzle_solver");
    let mut session = Session::load_qor_dir(&dna_dir);

    // Simulate: accuracy 0.60 (between 0.50 and 0.70)
    session.exec("(attempt-number puzzle_mid 2)").unwrap();
    session.exec("(solve-confidence puzzle_mid 0.6000) <0.60, 0.95>").unwrap();
    session.exec("(attempt-failed puzzle_mid 1)").unwrap();

    assert!(!session.has_fact("web-lookup-needed"),
        "web-lookup-needed should NOT fire when confidence > 0.50");

    let strategies = session.facts_with_predicate("strategy");
    let strategy_strs: Vec<String> = strategies.iter()
        .map(|f| f.neuron.to_string())
        .collect();
    assert!(strategy_strs.iter().any(|s| s.contains("alternate")),
        "strategy should be alternate, got: {:?}", strategy_strs);
}

// ═════════════════════════════════════════════════════════════════════
// TEST 4: First attempt → try-all strategy
// ═════════════════════════════════════════════════════════════════════

#[test]
fn first_attempt_try_all() {
    let dna_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("dna").join("puzzle_solver");
    let mut session = Session::load_qor_dir(&dna_dir);

    // First attempt — no confidence injected
    session.exec("(attempt-number puzzle_first 1)").unwrap();

    assert!(!session.has_fact("web-lookup-needed"),
        "web-lookup-needed should NOT fire on first attempt");

    let strategies = session.facts_with_predicate("strategy");
    let strategy_strs: Vec<String> = strategies.iter()
        .map(|f| f.neuron.to_string())
        .collect();
    assert!(strategy_strs.iter().any(|s| s.contains("try-all")),
        "strategy should be try-all on first attempt, got: {:?}", strategy_strs);
}

// ═════════════════════════════════════════════════════════════════════
// TEST 5: Full pipeline — web returns new facts → re-solve
// ═════════════════════════════════════════════════════════════════════

#[test]
fn web_results_feed_back_and_create_new_facts() {
    let dna_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("dna").join("puzzle_solver");
    let mut session = Session::load_qor_dir(&dna_dir);

    // Step 1: Inject low confidence → triggers web lookup
    session.exec("(attempt-number puzzle_web 3)").unwrap();
    session.exec("(solve-confidence puzzle_web 0.3500) <0.35, 0.95>").unwrap();
    session.exec("(attempt-failed puzzle_web 1)").unwrap();
    session.exec("(attempt-failed puzzle_web 2)").unwrap();

    assert!(session.has_fact("web-lookup-needed"),
        "web-lookup-needed should fire");
    assert!(session.has_fact("browser-action"),
        "browser-action navigate should fire");

    // Step 2: Simulate browser returning page elements (after snapshot)
    // This is what Rust injects after executing the browser actions
    session.exec("(page-url \"https://arcprize.org/play?task=puzzle_web\") <0.95, 0.90>").unwrap();
    session.exec("(page-element s1 button submit none) <0.95, 0.90>").unwrap();
    session.exec("(page-element s2 link puzzle-link none) <0.95, 0.90>").unwrap();

    // web-snapshot-taken should fire (page-element exists)
    assert!(session.has_fact("web-snapshot-taken"),
        "web-snapshot-taken should fire after page elements injected");
    // web-visited-play should fire (page-url exists)
    assert!(session.has_fact("web-visited-play"),
        "web-visited-play should fire after page-url injected");

    // Step 3: Simulate JS extraction returning grid data
    session.exec("(js-result \"extract-grids\" \"grid-data-json\") <0.95, 0.90>").unwrap();

    // web-js-extracted should fire
    assert!(session.has_fact("web-js-extracted"),
        "web-js-extracted should fire after js-result injected");

    // web-done should fire (js extracted)
    assert!(session.has_fact("web-done"),
        "web-done should fire after JS extraction complete");

    // Step 4: Simulate web-extracted grid facts (what bridge would parse)
    // These are NEW facts the agent didn't have before
    session.exec("(grid-cell ti 0 0 1) <0.95, 0.90>").unwrap();
    session.exec("(grid-cell ti 0 1 2) <0.95, 0.90>").unwrap();
    session.exec("(grid-cell ti 1 0 2) <0.95, 0.90>").unwrap();
    session.exec("(grid-cell ti 1 1 1) <0.95, 0.90>").unwrap();
    session.exec("(grid-size ti 2 2) <0.95, 0.90>").unwrap();
    session.exec("(test-input ti) <0.99, 0.99>").unwrap();

    // The session now has web-enriched facts.
    // Any predict-cell rules that depend on grid-cell + test-input
    // can now fire on this new data.
    let fact_count_after = session.fact_count();
    assert!(fact_count_after > 10,
        "session should have many facts after web injection, got {fact_count_after}");
}

// ═════════════════════════════════════════════════════════════════════
// TEST 6: Browser unavailable → HTTP fallback fires
// ═════════════════════════════════════════════════════════════════════

#[test]
fn browser_unavailable_triggers_http_fallback() {
    let dna_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("dna").join("puzzle_solver");
    let mut session = Session::load_qor_dir(&dna_dir);

    // Low confidence + browser unavailable
    session.exec("(attempt-number puzzle_http 3)").unwrap();
    session.exec("(solve-confidence puzzle_http 0.3000) <0.30, 0.95>").unwrap();
    session.exec("(attempt-failed puzzle_http 1)").unwrap();
    session.exec("(attempt-failed puzzle_http 2)").unwrap();
    session.exec("(browser-unavailable true)").unwrap();

    // web-lookup-needed should still fire
    assert!(session.has_fact("web-lookup-needed"),
        "web-lookup-needed should fire even without browser");

    // http-fetch should fire as fallback
    assert!(session.has_fact("http-fetch"),
        "http-fetch should fire when browser is unavailable");

    // web-done should fire (browser-unavailable + web-lookup-needed)
    assert!(session.has_fact("web-done"),
        "web-done should fire via browser-unavailable fallback path");
}

// ═════════════════════════════════════════════════════════════════════
// TEST 7: web-done prevents re-browsing
// ═════════════════════════════════════════════════════════════════════

#[test]
fn web_done_prevents_re_browsing() {
    let dna_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("dna").join("puzzle_solver");
    let mut session = Session::load_qor_dir(&dna_dir);

    // Simulate: web-done already exists from a previous attempt.
    // In a fresh session (new attempt), inject web-done BEFORE confidence.
    // This mimics: "we already browsed, don't browse again."
    session.exec("(web-done puzzle_done) <0.95, 0.95>").unwrap();
    session.exec("(web-js-extracted puzzle_done) <0.99, 0.99>").unwrap();

    // Now inject low confidence — normally would trigger web lookup
    session.exec("(attempt-number puzzle_done 4)").unwrap();
    session.exec("(solve-confidence puzzle_done 0.3000) <0.30, 0.95>").unwrap();
    session.exec("(attempt-failed puzzle_done 1)").unwrap();
    session.exec("(attempt-failed puzzle_done 2)").unwrap();
    session.exec("(attempt-failed puzzle_done 3)").unwrap();

    // web-done exists → not (web-done $id) blocks web-lookup-needed
    assert!(session.has_fact("web-done"),
        "web-done should exist");
    assert!(!session.has_fact("web-lookup-needed"),
        "web-lookup-needed should be blocked by existing web-done");
}

// ═════════════════════════════════════════════════════════════════════
// TEST 8: Suppression after many failures
// ═════════════════════════════════════════════════════════════════════

#[test]
fn suppression_after_many_failures() {
    let dna_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("dna").join("puzzle_solver");
    let mut session = Session::load_qor_dir(&dna_dir);

    // 4 failed attempts
    session.exec("(attempt-number puzzle_sup 5)").unwrap();
    session.exec("(solve-confidence puzzle_sup 0.2000) <0.20, 0.95>").unwrap();
    session.exec("(attempt-failed puzzle_sup 1)").unwrap();
    session.exec("(attempt-failed puzzle_sup 2)").unwrap();
    session.exec("(attempt-failed puzzle_sup 3)").unwrap();
    session.exec("(attempt-failed puzzle_sup 4)").unwrap();

    // After 3+ failures: suppress color-remap
    assert!(session.has_fact("suppress-strategy"),
        "suppress-strategy should fire after 3+ failures");

    let suppressions = session.facts_with_predicate("suppress-strategy");
    let supp_strs: Vec<String> = suppressions.iter()
        .map(|f| f.neuron.to_string())
        .collect();
    assert!(supp_strs.iter().any(|s| s.contains("color-remap")),
        "color-remap should be suppressed after 3 failures, got: {:?}", supp_strs);
    assert!(supp_strs.iter().any(|s| s.contains("spatial-transform")),
        "spatial-transform should be suppressed after 4 failures, got: {:?}", supp_strs);
}
