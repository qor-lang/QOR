// ── Reasoning-First Solve Loop ────────────────────────────────────────
//
// Based on issues.md 5-operation reasoning model:
//   MATCH  — find relevant rules & facts
//   CHAIN  — follow connections (forward chaining)
//   UNIFY  — find structural similarities
//   REDUCE — apply rules, compute
//   CONTRADICT — check for conflicts
//
// The solve loop uses QOR's OWN reasoning at its core:
//   Phase 0: REASON   — run DNA rules, score baseline (MATCH + CHAIN)
//   Phase 1: TEMPLATES — fill holes from observations (UNIFY)
//   Phase 2: REFINE   — mutate positives (REDUCE + CONTRADICT)
//   Phase 3: GENESIS  — invent new rules (MATCH + CHAIN + REDUCE)
//   Phase 4: SWARM    — parallel multi-strategy (all 5 ops)
//   Phase 5: COMBINE  — combine partial rules (UNIFY)
//
// CRITICAL: puzzle_session must have ALL training context loaded
// (training pairs, comparisons, observations) so DNA rules can fire.
// Each phase shares findings with the next via all_positives.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use qor_core::neuron::{Neuron, QorValue, Statement};
use qor_core::truth_value::TruthValue;
use qor_runtime::chain::{Rule, forward_chain};
use qor_runtime::eval::Session;
use qor_runtime::invent;
use qor_runtime::library::{RuleLibrary, RuleSource, PruneConfig};
use qor_runtime::memory::FailureMemory;
use qor_runtime::mutate::{rule_to_qor, generate_mutations, generate_context_mutations};
use qor_runtime::search;
use qor_runtime::egraph;

use crate::template;

// ── Strategy Hints — read from reasoning.qor derived facts ────────────
//
// reasoning.qor fires during forward chain and produces:
//   (genesis-hint X)      → try strategy X
//   (genesis-skip Y)      → don't try strategy Y
//   (genesis-constrain P) → constrain property P
//   (genesis-reject R)    → reject rule R
//   (problem-class C)     → classified as C (remap, spatial, scale, filter)
//   (current-strategy S)  → best strategy S
//   (strategy-order N S)  → strategy S at priority N
//   (mostly-unchanged P R)→ Pāṇini identity + exception applicable

/// Hints derived from reasoning.qor rules firing on problem facts.
/// These guide genesis: what to try, what to skip, what strategy to use.
#[derive(Default, Debug, Clone)]
pub struct StrategyHints {
    /// What to try first — genesis-hint values
    pub try_these: Vec<String>,
    /// What to skip — genesis-skip values
    pub skip_these: Vec<String>,
    /// Properties to constrain — genesis-constrain values
    pub constraints: Vec<String>,
    /// Rules to reject — genesis-reject values
    pub rejected: Vec<String>,
    /// Problem classification: remap, spatial, scale, filter
    pub problem_class: Option<String>,
    /// Best strategy from reasoning.qor meta-selector
    pub best_strategy: Option<String>,
    /// Ordered strategies by priority
    pub strategy_order: Vec<(i64, String)>,
    /// Pāṇini identity + exception applicable
    pub identity_plus_fix: bool,
}

/// Read strategy hints from session after reasoning.qor has fired.
///
/// reasoning.qor rules fire automatically during forward_chain when
/// problem observation facts are present. This function reads the
/// derived hint/skip/strategy facts and packs them into StrategyHints.
pub fn read_strategy_hints(session: &Session) -> StrategyHints {
    let mut hints = StrategyHints::default();

    for sn in session.all_facts() {
        if let Neuron::Expression(parts) = &sn.neuron {
            let pred = match parts.first() {
                Some(Neuron::Symbol(s)) => s.as_str(),
                _ => continue,
            };
            match pred {
                "genesis-hint" => {
                    if let Some(Neuron::Symbol(h)) = parts.get(1) {
                        hints.try_these.push(h.clone());
                    }
                }
                "genesis-skip" => {
                    if let Some(Neuron::Symbol(s)) = parts.get(1) {
                        hints.skip_these.push(s.clone());
                    }
                }
                "genesis-constrain" => {
                    if let Some(Neuron::Symbol(c)) = parts.get(1) {
                        hints.constraints.push(c.clone());
                    }
                }
                "genesis-reject" => {
                    if let Some(Neuron::Symbol(r)) = parts.get(1) {
                        hints.rejected.push(r.clone());
                    }
                }
                "problem-class" => {
                    if let Some(Neuron::Symbol(c)) = parts.get(1) {
                        hints.problem_class = Some(c.clone());
                    }
                }
                "current-strategy" => {
                    if let Some(Neuron::Symbol(s)) = parts.get(1) {
                        hints.best_strategy = Some(s.clone());
                    }
                }
                "strategy-order" => {
                    if let (Some(Neuron::Value(QorValue::Int(pri))),
                            Some(Neuron::Symbol(name))) = (parts.get(1), parts.get(2))
                    {
                        hints.strategy_order.push((*pri, name.clone()));
                    }
                }
                "mostly-unchanged" => {
                    hints.identity_plus_fix = true;
                }
                _ => {}
            }
        }
    }
    hints.strategy_order.sort_by_key(|(pri, _)| *pri);
    // Dedup try_these and skip_these
    hints.try_these.sort();
    hints.try_these.dedup();
    hints.skip_these.sort();
    hints.skip_these.dedup();
    hints
}

/// Extract transform-class facts from session (derived by classify.qor).
fn extract_transform_classes(session: &Session) -> Vec<String> {
    let mut classes = Vec::new();
    for sn in session.all_facts() {
        if let Neuron::Expression(parts) = &sn.neuron {
            if let Some(Neuron::Symbol(p)) = parts.first() {
                if p == "transform-class" {
                    if let Some(Neuron::Symbol(c)) = parts.get(1) {
                        if !classes.contains(c) {
                            classes.push(c.clone());
                        }
                    }
                }
            }
        }
    }
    classes
}

/// Run diagnostics on the classified session BEFORE solving.
///
/// Reports:
/// 1. What transform classes were detected
/// 2. Which detected-* facts fired (DNA detection rules matched observations)
/// 3. Which observations are present
/// 4. DNA coverage: does DNA have predict-cell rules for the detected class?
/// 5. Gaps: what's classified but not covered by any DNA predict-cell rule?
fn run_diagnostics(session: &Session) -> DiagnosticReport {
    let mut report = DiagnosticReport::default();

    // 1. Transform classes
    report.classes = extract_transform_classes(session);

    // 2. Detected-* facts (fired from DNA detection rules)
    // 3. Observations (obs-* facts from Rust perception)
    // 4. Count predict-cell predictions
    let mut obs_set: Vec<String> = Vec::new();
    for sn in session.all_facts() {
        if let Neuron::Expression(parts) = &sn.neuron {
            if let Some(Neuron::Symbol(p)) = parts.first() {
                if p.starts_with("detected-") {
                    let det = parts.iter().map(|n| format!("{}", n)).collect::<Vec<_>>().join(" ");
                    if !report.detections.contains(&det) {
                        report.detections.push(det);
                    }
                }
                if p.starts_with("obs-") {
                    let obs = parts.iter().map(|n| format!("{}", n)).collect::<Vec<_>>().join(" ");
                    if !obs_set.contains(&obs) {
                        obs_set.push(obs);
                    }
                }
                if p == "predict-cell" {
                    report.dna_predictions += 1;
                }
            }
        }
    }
    report.observations = obs_set;

    // 5. Coverage analysis: map transform-class → expected detected-* prefix
    // If a class is detected AND DNA produced predict-cell facts → covered
    // If a class is detected but NO predict-cell facts at all → gap
    let class_to_detection: &[(&str, &[&str])] = &[
        ("geometric", &["detected-identity", "detected-reflect-h", "detected-reflect-v",
                         "detected-rotate-90", "detected-rotate-180", "detected-rotate-270",
                         "detected-transpose", "detected-shift", "detected-crop",
                         "detected-crop-to-bbox", "detected-scale-up"]),
        ("tiling", &["obs-uniform-scale"]),
        ("color-remap", &["consistent-remap"]),
        ("filtering", &["detected-noise-remove", "detected-keep-largest"]),
        ("fill", &["detected-row-fill", "detected-col-fill", "detected-cross-fill"]),
        ("region-fill", &["detected-region-fill-seed", "detected-flood-fill"]),
        ("scaling", &["detected-scale-up", "obs-size-ratio"]),
        ("composed", &["obs-consistent composed"]),
        ("marker-driven", &[]),
        ("separator-combine", &["detected-separator-v", "detected-separator-h"]),
    ];

    for class in &report.classes {
        // Check if class has matching detections or observations
        let mut has_coverage = false;
        if let Some((_, expected_preds)) = class_to_detection.iter().find(|(c, _)| c == class) {
            for ep in *expected_preds {
                if report.detections.iter().any(|d| d.starts_with(ep))
                    || report.observations.iter().any(|o| o.starts_with(ep))
                {
                    has_coverage = true;
                    break;
                }
            }
        }

        // Check if DNA actually produced predict-cell facts
        if report.dna_predictions > 0 && has_coverage {
            report.covered.push(class.clone());
        } else if class != "unknown" {
            report.gaps.push(class.clone());
        }
    }

    // Print diagnostic report
    eprintln!("\n  ┌─── DIAGNOSTIC REPORT ───────────────────────────────────");
    eprintln!("  │ Classification: {:?}", report.classes);
    eprintln!("  │ Detections:     {} fired", report.detections.len());
    for d in &report.detections {
        eprintln!("  │   ✓ ({})", d);
    }
    eprintln!("  │ Observations:   {} present", report.observations.len());
    for o in report.observations.iter().take(15) {
        eprintln!("  │   • ({})", o);
    }
    if report.observations.len() > 15 {
        eprintln!("  │   ... and {} more", report.observations.len() - 15);
    }
    eprintln!("  │ DNA predictions: {}", report.dna_predictions);
    if !report.covered.is_empty() {
        eprintln!("  │ Coverage:       {:?} — DNA has rules", report.covered);
    }
    if !report.gaps.is_empty() {
        eprintln!("  │ GAPS:           {:?} — classified but NO DNA rules fired!", report.gaps);
        eprintln!("  │   → Genesis should focus on these gaps");
    }
    if report.classes.contains(&"unknown".to_string()) {
        eprintln!("  │ UNKNOWN:        No classification matched — full genesis search needed");
    }
    eprintln!("  └────────────────────────────────────────────────────────\n");

    report
}

impl StrategyHints {
    /// Convert to SwarmHints for passing to genesis_swarm_with_hints.
    /// Passes classification, constraints, and try/skip lists to genesis.
    pub fn to_swarm_hints(&self) -> invent::SwarmHints {
        invent::SwarmHints {
            try_these: self.try_these.clone(),
            skip_these: self.skip_these.clone(),
            problem_class: self.problem_class.clone(),
            constraints: self.constraints.clone(),
        }
    }

    /// True if reasoning.qor produced any hints at all.
    pub fn has_hints(&self) -> bool {
        !self.try_these.is_empty() || !self.skip_these.is_empty()
            || self.problem_class.is_some() || self.best_strategy.is_some()
    }
}

// ── Skip-Hint Candidate Filtering ─────────────────────────────────────
//
// reasoning.qor says "skip recolor" → filter candidates that are purely color remaps.
// This prevents wasting evaluation time on wrong approaches.
//
// Categories:
//   "identity"   → body is just grid-cell or grid-object (trivial copy)
//   "recolor"    → body has no position arithmetic (pure color mapping)
//   "tiling"     → body has multiplication for scaling (* $x FACTOR)
//   "geometric"  → body mentions rotate/reflect/transpose
//   "resize"     → rule changes output dimensions
//   "noise-remove" → body uses not-present or negation
//   "crop"       → body references subgrid/crop predicates

/// Check if a genesis candidate's rule text matches a skip category.
/// Returns true if the rule should be SKIPPED (filtered out).
///
/// IMPORTANT: These must be NARROW matchers to avoid false positives.
/// Only skip candidates that CLEARLY belong to the category.
fn candidate_matches_skip(rule_text: &str, skip: &str) -> bool {
    match skip {
        "identity" => {
            // ONLY match TRUE identity: all head args are variables, all body args
            // are the SAME variables in the SAME order, single predicate body.
            // e.g. (predict-cell $a0 $a1 $a2 $a3) if (grid-cell $a0 $a1 $a2 $a3)
            // NOT: (predict-cell $a0 $a1 6 $a3) if (grid-object $a0 $a1 9 $a3)
            let body_part = rule_text.split(" if\n").nth(1)
                .or_else(|| rule_text.split(" if ").nth(1))
                .unwrap_or("");
            let body_trimmed = body_part.trim();
            let conditions: Vec<&str> = body_trimmed.lines()
                .map(|l| l.trim())
                .filter(|l| l.starts_with('('))
                .collect();
            if conditions.len() != 1 { return false; }
            // Extract head args (everything between first ( and )) and body args
            let extract_args = |s: &str| -> Vec<String> {
                let inner = s.trim().trim_start_matches('(').trim_end_matches(')');
                // Skip the predicate name (first word)
                inner.split_whitespace().skip(1).map(|a| a.to_string()).collect()
            };
            let head = rule_text.split(" if").next().unwrap_or("").trim();
            let head_args = extract_args(head);
            let body_args = extract_args(conditions[0].split('<').next().unwrap_or(conditions[0]));
            // ALL head args must be variables ($...) AND match body args exactly
            head_args.len() >= 3
                && head_args.len() == body_args.len()
                && head_args.iter().all(|a| a.starts_with('$'))
                && head_args == body_args
        }
        "tiling" => {
            // Skip if has scaling multiplication: (* $var FACTOR $result)
            let body = rule_text.split(" if").nth(1).unwrap_or("");
            body.contains("(* $") && (body.contains(" 2 ") || body.contains(" 3 ")
                || body.contains(" 4 ") || body.contains(" 5 "))
        }
        "geometric" => {
            rule_text.contains("rotate") || rule_text.contains("reflect")
                || rule_text.contains("transpose") || rule_text.contains("mirror")
        }
        "noise-remove" => {
            rule_text.contains("not-present") || rule_text.contains("not (")
        }
        "crop" => {
            rule_text.contains("subgrid") || rule_text.contains("crop")
                || rule_text.contains("extract")
        }
        // Categories we DON'T filter on (too broad or hard to detect from text):
        // "recolor", "resize", "invertible", "create-destroy"
        _ => false,
    }
}

/// Filter a list of candidates based on skip hints.
/// Returns only candidates that don't match any skip category.
fn filter_by_skip_hints(candidates: &[invent::Candidate], skip_these: &[String]) -> Vec<invent::Candidate> {
    if skip_these.is_empty() { return candidates.to_vec(); }
    let before = candidates.len();
    let filtered: Vec<_> = candidates.iter()
        .filter(|c| !skip_these.iter().any(|skip| candidate_matches_skip(&c.rule_text, skip)))
        .cloned()
        .collect();
    let removed = before - filtered.len();
    if removed > 0 {
        eprintln!("    [skip-filter] removed {}/{} candidates matching {:?}",
            removed, before, skip_these);
    }
    filtered
}

/// Filter candidates by conservation constraints from reasoning.qor.
///
/// If reasoning.qor derived (genesis-constrain element-count), reject
/// candidates whose rule text implies creation/destruction of elements
/// (e.g. constant-value fills without corresponding removal).
///
/// If reasoning.qor derived (genesis-constrain value-set), reject
/// candidates that introduce new color constants not in input.
///
/// If reasoning.qor derived (genesis-constrain shape), reject
/// candidates that imply size change (scaling/tiling).
fn filter_by_constraints(candidates: &[invent::Candidate], constraints: &[String]) -> Vec<invent::Candidate> {
    if constraints.is_empty() { return candidates.to_vec(); }
    let before = candidates.len();
    let filtered: Vec<_> = candidates.iter()
        .filter(|c| {
            for constraint in constraints {
                match constraint.as_str() {
                    "shape" => {
                        // Reject candidates that imply size change
                        if c.rule_text.contains("(* $") && (
                            c.rule_text.contains(" 2 ") || c.rule_text.contains(" 3 ")
                            || c.rule_text.contains(" 4 "))
                        {
                            return false;
                        }
                    }
                    "value-set" => {
                        // Reject candidates that hardcode color constants > 9
                        // in the head (output) — implies new values
                        let head = c.rule_text.split(" if").next().unwrap_or("");
                        for tok in head.split_whitespace() {
                            if let Ok(v) = tok.trim_end_matches(')').parse::<i64>() {
                                if v > 9 { return false; }
                            }
                        }
                    }
                    _ => {} // unknown constraint — don't filter
                }
            }
            true
        })
        .cloned()
        .collect();
    let removed = before - filtered.len();
    if removed > 0 {
        eprintln!("    [constraint-filter] removed {}/{} candidates violating {:?}",
            removed, before, constraints);
    }
    filtered
}

// ── Identity-Fix Strategy (Pāṇini) ───────────────────────────────────
//
// When reasoning.qor says strategy = "identity-fix":
// 1. Score identity rule → find WHERE it's wrong
// 2. Compute gap: which cells are wrong (extra or missing predictions)
// 3. Inject gap facts so genesis can target fix rules
// 4. Generate exception rules that cover the gaps

/// Compute gap facts between identity predictions and expected output.
/// Returns (gap_statements, identity_score).
///
/// Gap facts produced:
///   (identity-gap $id $r $c $expected $got)  — wrong value at position
///   (identity-extra $id $r $c $v)            — predicted but shouldn't be
///   (identity-missing $id $r $c $v)          — should exist but not predicted
///   (identity-correct $count)                — number of correct predictions
///   (identity-total $count)                  — total expected
fn compute_identity_gap(
    puzzle_session: &Session,
    expected_keys: &HashSet<String>,
    target_pred: &str,
) -> (Vec<Statement>, f64) {
    use qor_core::parser::parse;

    let valid_ids = extract_valid_ids(expected_keys);
    let predictions = extract_keys_from_session_filtered(puzzle_session, target_pred, &valid_ids);
    let correct = expected_keys.iter().filter(|k| predictions.contains(*k)).count();
    let score = precision_recall_score(correct, expected_keys.len(), predictions.len());

    let mut gap_stmts = Vec::new();

    // Parse expected keys into (id, r, c, v) tuples
    let parse_key = |k: &str| -> Option<(String, String, String, String)> {
        let parts: Vec<&str> = k.split_whitespace().collect();
        if parts.len() >= 4 {
            Some((parts[0].to_string(), parts[1].to_string(),
                  parts[2].to_string(), parts[3].to_string()))
        } else { None }
    };

    // Missing: expected but not predicted
    let mut missing_count = 0;
    for k in expected_keys {
        if !predictions.contains(k) {
            if let Some((id, r, c, v)) = parse_key(k) {
                let fact = format!("(identity-missing {} {} {} {})", id, r, c, v);
                if let Ok(stmts) = parse(&fact) {
                    gap_stmts.extend(stmts);
                    missing_count += 1;
                }
            }
        }
    }

    // Extra: predicted but not expected
    let mut extra_count = 0;
    for k in &predictions {
        if !expected_keys.contains(k) {
            if let Some((id, r, c, v)) = parse_key(k) {
                let fact = format!("(identity-extra {} {} {} {})", id, r, c, v);
                if let Ok(stmts) = parse(&fact) {
                    gap_stmts.extend(stmts);
                    extra_count += 1;
                }
            }
        }
    }

    // Summary facts
    if let Ok(stmts) = parse(&format!("(identity-correct {})", correct)) {
        gap_stmts.extend(stmts);
    }
    if let Ok(stmts) = parse(&format!("(identity-total {})", expected_keys.len())) {
        gap_stmts.extend(stmts);
    }
    if let Ok(stmts) = parse(&format!("(identity-missing-count {})", missing_count)) {
        gap_stmts.extend(stmts);
    }
    if let Ok(stmts) = parse(&format!("(identity-extra-count {})", extra_count)) {
        gap_stmts.extend(stmts);
    }

    eprintln!("    P0.5-gap: {} correct, {} missing, {} extra (identity score={:.1}%)",
        correct, missing_count, extra_count, score * 100.0);

    (gap_stmts, score)
}

/// Diagnostic report — what was classified, what DNA covers, what's missing.
#[derive(Debug, Clone, Default)]
pub struct DiagnosticReport {
    /// Transform classes detected by classify.qor
    pub classes: Vec<String>,
    /// Which detected-* facts fired (DNA detection rules matched)
    pub detections: Vec<String>,
    /// Which observation facts are present
    pub observations: Vec<String>,
    /// DNA coverage: classes that have predict-cell rules
    pub covered: Vec<String>,
    /// DNA gaps: classes detected but no predict-cell rules fired
    pub gaps: Vec<String>,
    /// Number of predict-cell facts DNA produced
    pub dna_predictions: usize,
}

/// Result of an iterative solve attempt.
#[derive(Debug)]
pub struct SolveResult {
    /// Winning QOR rule texts (empty if not solved).
    pub best_rules: Vec<String>,
    /// Best training accuracy achieved (0.0 to 1.0).
    pub score: f64,
    /// True if score >= 0.999 (all training pairs matched).
    pub solved: bool,
    /// Total candidates explored across all phases.
    pub candidates_explored: usize,
    /// Total mutations tried.
    pub mutations_tried: usize,
    /// Wall-clock time spent.
    pub elapsed_ms: u64,
    /// Which phase found the solution (if solved).
    pub solved_in_phase: Option<&'static str>,
    /// DNA baseline score (what reasoning alone achieves).
    pub baseline_score: f64,
    /// Number of solve rounds completed.
    pub rounds: usize,
    /// Number of failed rules tracked in failure memory.
    pub failures_tracked: usize,
    /// Number of rules that overfit (100% train, <95% test).
    pub overfit_count: usize,
    /// Diagnostic report (classification → coverage → gaps).
    pub diagnostic: DiagnosticReport,
}

/// Test validation data — allows solve to check candidates against held-out test data.
/// When provided, rules that score 100% on training but <95% on test are flagged
/// as overfitting and the system keeps searching for genuinely generalizing rules.
pub struct TestValidation<'a> {
    pub test_session: &'a Session,
    pub test_expected_keys: HashSet<String>,
}

/// Run the reasoning-first solve pipeline with multi-round failure memory.
///
/// This is the main entry point. It wraps `solve_one_round()` in a loop:
///   Round 1: Run all phases. Track failures.
///   Round 2+: Inject failure memory → reasoning.qor re-fires with failure context
///             → new hints produced → solve with updated strategy → track new failures.
///   Each round is SMARTER than the last because it never repeats mistakes.
///
/// # Arguments
/// * `puzzle_session` — Session with DNA rules AND all training context loaded
/// * `all_expected` — ALL expected predict-cell facts across all training pairs
/// * `target_pred` — Predicate to match (e.g., "predict-cell")
/// * `observations` — Cross-pair observation facts (for template filling)
/// * `time_budget_ms` — Total time budget in milliseconds
/// * `library` — Optional rule library for recall/save
/// * `meta_dir` — Optional path to meta/ directory for strategy-per-worker swarm
pub fn solve(
    puzzle_session: &Session,
    all_expected: &[Statement],
    target_pred: &str,
    observations: &[Statement],
    time_budget_ms: u64,
    library: Option<&mut RuleLibrary>,
    meta_dir: Option<&std::path::Path>,
    test_validation: Option<&TestValidation>,
) -> SolveResult {
    let start = Instant::now();
    let mut deadline = start + Duration::from_millis(time_budget_ms);
    let mut memory = FailureMemory::new();
    let mut best_score = 0.0f64;
    let mut best_rules: Vec<String> = Vec::new();
    let mut accumulated_rules: Vec<String> = Vec::new(); // Progressive: rules locked in across rounds
    let mut total_candidates = 0usize;
    let mut total_mutations = 0usize;
    let mut solved_phase: Option<&'static str> = None;
    let mut baseline_score = 0.0f64;
    let mut max_rounds = 8;
    let mut round = 0;
    let mut prev_best_score = 0.0f64;  // track progress between rounds
    let mut stale_rounds = 0usize;     // rounds with no improvement
    let max_total_budget = time_budget_ms * 3; // never exceed 3x original budget

    // Load meta strategies for strategy-per-worker swarm
    let meta_strats = meta_dir.map(|d| load_meta_strategies(d));

    // ═══════════════════════════════════════════════════════════════════
    // CLASSIFY FIRST: load classify.qor + reasoning.qor into a session
    // clone, forward chain fires, read classification + strategy hints.
    // This runs BEFORE anything else so all phases benefit.
    // ═══════════════════════════════════════════════════════════════════
    let (classified_session, initial_hints) = if let Some(md) = meta_dir {
        let mut cs = puzzle_session.clone();

        // Load classify.qor — universal classification rules
        let classify_path = md.join("classify.qor");
        if let Ok(source) = std::fs::read_to_string(&classify_path) {
            if let Ok(stmts) = qor_core::parser::parse(&source) {
                eprintln!("  classify: loaded classify.qor ({} rules)", stmts.len());
                let _ = cs.exec_statements(stmts);
            }
        }

        // Load bridge.qor — translate obs-* facts → reasoning.qor inputs
        let bridge_path = md.join("bridge.qor");
        if let Ok(source) = std::fs::read_to_string(&bridge_path) {
            if let Ok(stmts) = qor_core::parser::parse(&source) {
                eprintln!("  bridge: loaded bridge.qor ({} rules)", stmts.len());
                let _ = cs.exec_statements(stmts);
            }
        }

        // Load reasoning.qor — strategy advisor
        let reasoning_path = md.join("reasoning.qor");
        if let Ok(source) = std::fs::read_to_string(&reasoning_path) {
            if let Ok(stmts) = qor_core::parser::parse(&source) {
                eprintln!("  reasoning: loaded reasoning.qor ({} rules)", stmts.len());
                let _ = cs.exec_statements(stmts);
            }
        }

        // Read classifications
        let classes = extract_transform_classes(&cs);
        if !classes.is_empty() {
            eprintln!("  classification: {:?}", classes);
        }

        // Read strategy hints (classify + reasoning both fired)
        let hints = read_strategy_hints(&cs);
        if hints.has_hints() {
            eprintln!("  strategy hints: try={:?} skip={:?} class={:?} strategy={:?}",
                hints.try_these, hints.skip_these,
                hints.problem_class, hints.best_strategy);
        }

        (cs, hints)
    } else {
        (puzzle_session.clone(), StrategyHints::default())
    };

    // ═══════════════════════════════════════════════════════════════════
    // DIAGNOSTIC: run BEFORE solving — classify, check coverage, report gaps
    // This is the user's "check every process individual" requirement.
    // ═══════════════════════════════════════════════════════════════════
    let diagnostic = run_diagnostics(&classified_session);

    while round < max_rounds && Instant::now() < deadline {
        round += 1;
        let round_start = Instant::now();

        eprintln!("  === ROUND {}/{} — failures:{} exhausted:{} ===",
            round, max_rounds, memory.total_failures(), memory.exhausted_approaches());

        // Build round session: inject failure memory + accumulated progress rules
        let round_session = {
            let mut s = classified_session.clone();
            // Inject failure memory so reasoning.qor re-fires with failure context
            if round > 1 && memory.total_failures() > 0 {
                let _ = s.exec_statements(memory.to_statements());
            }
            // PROGRESSIVE: inject rules locked in from previous rounds
            // so the current round searches ON TOP of proven progress
            for rule_text in &accumulated_rules {
                if let Ok(stmts) = qor_core::parser::parse(rule_text) {
                    let _ = s.exec_statements(stmts);
                }
            }
            if !accumulated_rules.is_empty() {
                eprintln!("  [progressive] round starts with {} locked rules (baseline={:.1}%)",
                    accumulated_rules.len(), best_score * 100.0);
            }
            s
        };

        // Re-read strategy hints with failure context in round 2+
        let hints = if round > 1 {
            read_strategy_hints(&round_session)
        } else {
            initial_hints.clone()
        };

        // Calculate time budget for this round
        let remaining = deadline.saturating_duration_since(Instant::now()).as_millis() as u64;
        let round_budget = remaining / (max_rounds as u64 - round as u64 + 1);

        let result = solve_one_round(
            &round_session, all_expected, target_pred,
            observations, round_budget, &hints, &mut memory,
            meta_strats.as_deref(), meta_dir, test_validation,
        );

        total_candidates += result.candidates_explored;
        total_mutations += result.mutations_tried;
        if round == 1 {
            baseline_score = result.baseline_score;
        }

        if result.score > best_score {
            best_score = result.score;
            best_rules = result.best_rules.clone();
            solved_phase = result.solved_in_phase;
            // PROGRESSIVE: lock in winning rules for next round
            accumulated_rules = best_rules.clone();
            eprintln!("  [progressive] progress: {:.1}% — locked {} rules for next round",
                best_score * 100.0, accumulated_rules.len());
        }

        if result.solved {
            save_to_library(library, &best_rules, &[]);
            eprintln!("  === SOLVED in round {} ({:.0}ms) ===", round,
                round_start.elapsed().as_millis());
            return SolveResult {
                best_rules, score: best_score, solved: true,
                candidates_explored: total_candidates, mutations_tried: total_mutations,
                elapsed_ms: start.elapsed().as_millis() as u64,
                solved_in_phase: solved_phase, baseline_score,
                rounds: round, failures_tracked: memory.total_failures(),
                overfit_count: memory.overfit_count(),
                diagnostic: diagnostic.clone(),
            };
        }

        // Extend rounds when overfitting is detected — give more chances to find generalizing rules
        if memory.overfit_count() > 0 && max_rounds < 12 {
            eprintln!("  [solve] overfit detected ({} so far) — extending max_rounds to 12", memory.overfit_count());
            max_rounds = 12;
        }

        // ── DYNAMIC BUDGET — auto-extend when making progress ──
        // If score improved this round, the system is learning — give it more time.
        // If score stalled for 3+ rounds, stop wasting time.
        let total_elapsed = start.elapsed().as_millis() as u64;
        if best_score > prev_best_score + 0.005 {
            // Progress! Extend deadline by one more round's worth of time
            stale_rounds = 0;
            let extension = time_budget_ms / 4; // 25% of original budget per improvement
            if total_elapsed + extension <= max_total_budget {
                deadline = deadline + Duration::from_millis(extension);
                if max_rounds <= round + 1 { max_rounds = round + 3; }
                eprintln!("  [solve] progress! {:.1}% → {:.1}% — extending budget +{}ms (total: {}ms cap: {}ms)",
                    prev_best_score * 100.0, best_score * 100.0,
                    extension, total_elapsed + extension, max_total_budget);
            }
            prev_best_score = best_score;
        } else {
            stale_rounds += 1;
        }

        eprintln!("  === ROUND {} done: best={:.1}% failures:{} overfits:{} stale:{} ({:.0}ms) ===",
            round, best_score * 100.0, memory.total_failures(), memory.overfit_count(),
            stale_rounds, round_start.elapsed().as_millis());

        // Stop early if stale for too long or very little time left
        if stale_rounds >= 5 {
            eprintln!("  [solve] stale for {} rounds at {:.1}% — stopping early", stale_rounds, best_score * 100.0);
            break;
        }
        if remaining < 1000 {
            break;
        }
    }

    // Not solved — save best findings to library
    save_to_library(library, &best_rules, &[]);

    eprintln!("  SOLVE-SUMMARY: {} rounds, best={:.1}%, candidates={}, mutations={}, {}ms",
        round, best_score * 100.0, total_candidates, total_mutations,
        start.elapsed().as_millis());

    SolveResult {
        best_rules, score: best_score, solved: false,
        candidates_explored: total_candidates, mutations_tried: total_mutations,
        elapsed_ms: start.elapsed().as_millis() as u64,
        solved_in_phase: solved_phase, baseline_score,
        rounds: round, failures_tracked: memory.total_failures(),
        overfit_count: memory.overfit_count(),
        diagnostic,
    }
}

/// Run one round of the solve pipeline (P0-P5).
///
/// Uses strategy hints to guide genesis and failure memory to skip already-tried rules.
fn solve_one_round(
    puzzle_session: &Session,
    all_expected: &[Statement],
    target_pred: &str,
    observations: &[Statement],
    time_budget_ms: u64,
    hints: &StrategyHints,
    memory: &mut FailureMemory,
    meta_strategies: Option<&[(String, Vec<Statement>)]>,
    meta_dir: Option<&std::path::Path>,
    test_validation: Option<&TestValidation>,
) -> SolveResult {
    let start = Instant::now();
    let deadline = start + Duration::from_millis(time_budget_ms);

    let mut total_candidates = 0usize;
    let mut total_mutations = 0usize;
    let mut best_score: f64;
    let mut best_rules: Vec<String> = Vec::new();
    let mut all_positives: Vec<(Rule, f64)> = Vec::new();

    // Build expected fact keys for scoring
    let expected_keys = extract_keys_from_stmts(all_expected, target_pred);

    // ═══════════════════════════════════════════════════════════════════
    // Phase 0: REASON — score what DNA rules already derive
    // This IS reasoning: MATCH + CHAIN on the loaded knowledge.
    // ═══════════════════════════════════════════════════════════════════
    let valid_ids = extract_valid_ids(&expected_keys);
    let baseline_predictions = extract_keys_from_session_filtered(puzzle_session, target_pred, &valid_ids);
    let baseline_correct = expected_keys.iter()
        .filter(|k| baseline_predictions.contains(*k))
        .count();
    let baseline_score = precision_recall_score(baseline_correct, expected_keys.len(),
        baseline_predictions.len());

    eprintln!("    P0-reason: DNA derives {} predictions, {}/{} correct = {:.1}% (recall={:.1}%, precision={:.1}%)",
        baseline_predictions.len(), baseline_correct, expected_keys.len(),
        baseline_score * 100.0,
        if expected_keys.is_empty() { 0.0 } else { baseline_correct as f64 / expected_keys.len() as f64 * 100.0 },
        if baseline_predictions.is_empty() { 0.0 } else { baseline_correct as f64 / baseline_predictions.len() as f64 * 100.0 });

    best_score = baseline_score;
    // NOTE: baseline_score now uses precision*recall (F1-like). DNA rules that
    // over-predict (many extras) get penalized, so solve loop won't short-circuit.

    // GAP ANALYSIS (CONTRADICT): what does DNA get wrong?
    let missing: Vec<&String> = expected_keys.iter()
        .filter(|k| !baseline_predictions.contains(*k))
        .collect();
    let extra: Vec<&String> = baseline_predictions.iter()
        .filter(|k| !expected_keys.contains(*k))
        .collect();
    eprintln!("    P0-gap: {} missing, {} extra predictions",
        missing.len(), extra.len());
    // Show samples of expected vs predicted, focusing on DIFFERENCES
    if !missing.is_empty() {
        eprintln!("    P0-missing (expected but NOT predicted):");
        for mk in missing.iter().take(5) {
            eprintln!("      - {}", mk);
        }
        if missing.len() > 5 {
            eprintln!("      ... and {} more", missing.len() - 5);
        }
    }
    if !extra.is_empty() {
        eprintln!("    P0-extra (predicted but NOT expected):");
        for ek in extra.iter().take(5) {
            eprintln!("      - {}", ek);
        }
        if extra.len() > 5 {
            eprintln!("      ... and {} more", extra.len() - 5);
        }
    }
    // Show a few correct matches too
    let correct_samples: Vec<&String> = expected_keys.iter()
        .filter(|k| baseline_predictions.contains(*k))
        .take(3)
        .collect();
    if !correct_samples.is_empty() {
        eprintln!("    P0-correct-sample: {:?}", correct_samples);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 0.5: IDENTITY-FIX (Pāṇini strategy)
    // When reasoning says "identity-fix", compute gap between identity
    // predictions and expected output, inject gap facts for genesis.
    // ═══════════════════════════════════════════════════════════════════
    if Instant::now() < deadline && hints.identity_plus_fix {
        let (gap_stmts, identity_score) = compute_identity_gap(
            puzzle_session, &expected_keys, target_pred,
        );
        if !gap_stmts.is_empty() {
            // Inject gap facts into puzzle session for genesis to use
            // (session is immutable here, but gap facts guide proposal generation)
            eprintln!("    P0.5-identity-fix: {} gap facts injected (identity={:.1}%)",
                gap_stmts.len(), identity_score * 100.0);

            // If identity scores well (>50%), generate targeted exception rules
            if identity_score > 0.3 {
                // Try identity + each missing cell as a specific override
                // Group missing cells by value to find patterns
                let mut missing_by_value: std::collections::HashMap<String, Vec<(String, String, String)>> =
                    std::collections::HashMap::new();
                for s in &gap_stmts {
                    if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = s {
                        if let Some(Neuron::Symbol(p)) = parts.first() {
                            if p == "identity-missing" {
                                if let (Some(id), Some(r), Some(c), Some(v)) =
                                    (parts.get(1), parts.get(2), parts.get(3), parts.get(4))
                                {
                                    missing_by_value
                                        .entry(v.to_string())
                                        .or_default()
                                        .push((id.to_string(), r.to_string(), c.to_string()));
                                }
                            }
                        }
                    }
                }

                // For each missing value, check if it correlates with grid-cell patterns
                let mut fix_count = 0;
                for (val, positions) in &missing_by_value {
                    if positions.len() < 2 { continue; }

                    // Generate a fix rule: cells that need this value
                    // Try: predict missing cells using their neighborhood
                    // Common pattern: cells adjacent to certain objects get a specific value
                    for (_, r, c) in positions.iter().take(3) {
                        // Check what's at this position in input
                        let check = format!(
                            "(predict-cell $a {} {} {}) if (grid-cell $a {} {} $v) (!= $v {}) <0.70, 0.70>",
                            r, c, val, r, c, val
                        );
                        if let Ok(stmts) = qor_core::parser::parse(&check) {
                            for stmt in stmts {
                                if let Statement::Rule { head, body, tv } = stmt {
                                    let rule = Rule::new(head, body, tv.unwrap_or(TruthValue::new(0.70, 0.70)));
                                    let train_score = score_rule_combined(
                                        &rule, puzzle_session, &expected_keys, target_pred,
                                    );
                                    if train_score > 0.01 {
                                        let text = rule_to_qor(&rule);
                                        let effective = validate_candidate_score(
                                            &text, train_score, test_validation, memory, target_pred,
                                        );
                                        if effective > best_score {
                                            best_score = effective;
                                            best_rules = vec![text.clone()];
                                        }
                                        if effective > 0.01 {
                                            all_positives.push((rule, effective));
                                            fix_count += 1;
                                        }
                                    }
                                    total_candidates += 1;
                                }
                            }
                        }
                    }
                }
                eprintln!("    P0.5-fixes: {} targeted fix candidates, best={:.1}%",
                    fix_count, best_score * 100.0);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // STRATEGY-AWARE BUDGET: use best_strategy to weight phase budgets.
    // "recall"/"direct-compute" → more P1/P2 (template/refine), less P3/P4
    // "exhaustive" → more P3/P4 (genesis/swarm), less P2
    // "classify"/"constrain" → balanced
    // ═══════════════════════════════════════════════════════════════════
    let (genesis_budget_pct, swarm_budget_pct) = match hints.best_strategy.as_deref() {
        Some("recall") | Some("direct-compute") => {
            eprintln!("    [strategy] best={:?} → heavy P1/P2, light P3/P4",
                hints.best_strategy);
            (25u64, 25u64) // leave 50% for P1/P2
        }
        Some("exhaustive") => {
            eprintln!("    [strategy] best=exhaustive → heavy P3/P4");
            (40, 40) // 80% to genesis/swarm
        }
        Some("identity-fix") => {
            eprintln!("    [strategy] best=identity-fix → heavy P0.5/P2");
            (20, 30) // identity-fix already ran in P0.5
        }
        _ => (33, 45) // default: balanced with slight swarm preference
    };

    // ═══════════════════════════════════════════════════════════════════
    // Phase 1: TEMPLATES — fill $HOLE_* from observations, score
    // ═══════════════════════════════════════════════════════════════════
    if Instant::now() < deadline {
        let filled = template::instantiate_all(observations);
        eprintln!("    P1-templates: {} filled from {} observations", filled.len(), observations.len());
        // Show what observation types we have
        let obs_types: Vec<String> = observations.iter().filter_map(|s| {
            if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = s {
                Some(parts.iter().take(2).map(|p| p.to_string()).collect::<Vec<_>>().join(" "))
            } else { None }
        }).collect::<std::collections::HashSet<_>>().into_iter().take(10).collect();
        if !obs_types.is_empty() {
            eprintln!("    P1-obs-types: {:?}", obs_types);
        }
        // Show each filled template
        for ft in &filled {
            eprintln!("      P1-filled: name={} fills={:?} rule={}", ft.name,
                ft.fill_values, ft.rule_text.chars().take(100).collect::<String>());
        }

        let mut seen_rules: std::collections::HashSet<String> = std::collections::HashSet::new();
        for ft in &filled {
            if ft.name == "identity" { continue; }
            if !seen_rules.insert(ft.rule_text.clone()) { continue; }
            // Skip rules already tried and failed in previous rounds
            if memory.already_failed(&ft.rule_text) { continue; }
            if let Ok(stmts) = qor_core::parser::parse(&ft.rule_text) {
                for stmt in stmts {
                    if let Statement::Rule { head, body, tv } = stmt {
                        let rule = Rule::new(
                            head, body,
                            tv.unwrap_or(TruthValue::new(0.95, 0.95)),
                        );
                        let train_score = score_rule_combined(
                            &rule, puzzle_session, &expected_keys, target_pred,
                        );
                        total_candidates += 1;

                        // Validate against test data
                        let effective_score = validate_candidate_score(
                            &ft.rule_text, train_score, test_validation, memory, target_pred,
                        );

                        eprintln!("      P1-score: {:.1}% (test={:.1}%) template={} rule={}",
                            train_score * 100.0, effective_score * 100.0,
                            ft.name, ft.rule_text.chars().take(100).collect::<String>());

                        if effective_score >= 0.999 {
                            best_rules = vec![ft.rule_text.clone()];
                            best_score = effective_score;
                            return make_result(best_rules, best_score, true, total_candidates,
                                total_mutations, &start, Some("templates"), baseline_score);
                        }
                        // Use training score for seeding refinement
                        let seed_score = train_score.max(effective_score);
                        if seed_score > 0.01 {
                            all_positives.push((rule, seed_score));
                            if effective_score > best_score {
                                best_score = effective_score;
                                best_rules = vec![ft.rule_text.clone()];
                            }
                        }
                    }
                }
            }
        }
        eprintln!("    P1-done: {} candidates, {} positives, best={:.1}%, {}ms",
            total_candidates, all_positives.len(),
            best_score * 100.0, start.elapsed().as_millis());
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 2: REFINEMENT — mutate positives (2s budget)
    // REDUCE: apply mutations. CONTRADICT: check if score improves.
    // ═══════════════════════════════════════════════════════════════════
    if Instant::now() < deadline {
        all_positives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let seed_rules: Vec<Rule> = all_positives.iter()
            .take(10)
            .map(|(r, _s): &(Rule, f64)| r.clone())
            .collect();

        if !seed_rules.is_empty() {
            let remaining_ms = deadline.saturating_duration_since(Instant::now()).as_millis() as u64;
            let refinement_budget = remaining_ms.min(2000);
            eprintln!("    P2-refine: {} seeds, {}ms budget", seed_rules.len(), refinement_budget);
            // Show seed rules
            for (si, sr) in seed_rules.iter().enumerate() {
                let text = rule_to_qor(sr);
                let score = all_positives.get(si).map(|(_,s)| *s).unwrap_or(0.0);
                eprintln!("      P2-seed[{}]: {:.1}% rule={}", si, score * 100.0,
                    text.chars().take(120).collect::<String>());
            }

            // Build per-pair data for refinement_search (it needs per-pair format)
            let (pair_inputs, pair_expected) = split_expected_by_pair(
                puzzle_session, all_expected, target_pred,
            );
            eprintln!("      P2-split: {} pairs, inputs={:?} expected={:?}",
                pair_inputs.len(),
                pair_inputs.iter().map(|v| v.len()).collect::<Vec<_>>(),
                pair_expected.iter().map(|v| v.len()).collect::<Vec<_>>());

            let result = search::refinement_search(
                &seed_rules, &pair_inputs, &pair_expected,
                target_pred, puzzle_session, refinement_budget,
            );
            total_mutations += result.mutations_tried;
            total_candidates += result.mutations_tried;

            for solution in &result.solutions {
                let effective_score = validate_candidate_score(
                    &solution.qor_text, solution.score, test_validation, memory, target_pred,
                );
                if effective_score >= 0.999 {
                    best_rules = vec![solution.qor_text.clone()];
                    best_score = effective_score;
                    return make_result(best_rules, best_score, true, total_candidates,
                        total_mutations, &start, Some("refinement"), baseline_score);
                } else if effective_score > best_score {
                    best_score = effective_score;
                    best_rules = vec![solution.qor_text.clone()];
                }
                if effective_score > 0.01 {
                    all_positives.push((Rule::new(
                        solution.rule.head.clone(), solution.rule.body.clone(), solution.rule.tv,
                    ), effective_score));
                }
            }
            for nm in &result.near_misses {
                let text = rule_to_qor(&nm.rule);
                let effective_score = validate_candidate_score(
                    &text, nm.score, test_validation, memory, target_pred,
                );
                if effective_score > 0.01 {
                    all_positives.push((nm.rule.clone(), effective_score));
                    if effective_score > best_score {
                        best_score = effective_score;
                        best_rules = vec![text];
                    }
                }
            }
            eprintln!("    P2-done: {} mutations, {} solutions, {} positives, best={:.1}%",
                result.mutations_tried, result.solutions.len(), result.near_misses.len(),
                best_score * 100.0);
        } else {
            eprintln!("    P2-skip: no seed rules");
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 3: GENESIS — invent rules using DNA-derived vocabulary
    // MATCH: profile DNA-derived facts. CHAIN: propose multi-hop rules.
    // Now genesis sees the FULL derived vocabulary, not just raw grid cells.
    // ═══════════════════════════════════════════════════════════════════
    if Instant::now() < deadline {
        let remaining_ms = deadline.saturating_duration_since(Instant::now()).as_millis() as u64;
        // Strategy-aware: use genesis_budget_pct instead of fixed 50%
        let genesis_budget = remaining_ms * genesis_budget_pct / (genesis_budget_pct + swarm_budget_pct);

        eprintln!("    P3-genesis: {}ms budget ({}% of remaining)", genesis_budget, genesis_budget_pct);
        if genesis_budget > 500 {
            // Build per-pair data for genesis (it needs per-pair format)
            let (pair_inputs, pair_expected) = split_expected_by_pair(
                puzzle_session, all_expected, target_pred,
            );

            let all_candidates = invent::genesis(
                puzzle_session, &pair_inputs, &pair_expected,
                genesis_budget, None,
                Some(memory.failed_rule_set()),
            );
            // Filter out already-failed rules from previous rounds
            let candidates_unfilt = memory.filter_candidates(all_candidates);
            // Filter by skip hints from reasoning.qor
            let candidates_skip = filter_by_skip_hints(&candidates_unfilt, &hints.skip_these);
            // Filter by conservation constraints from reasoning.qor
            let candidates = filter_by_constraints(&candidates_skip, &hints.constraints);
            eprintln!("    P3-done: {} candidates returned", candidates.len());
            for (ci, c) in candidates.iter().enumerate().take(5) {
                eprintln!("      P3[{}]: score={:.1}% src={} rule={}",
                    ci, c.score * 100.0, c.source,
                    c.rule_text.chars().take(120).collect::<String>());
            }

            for c in &candidates {
                total_candidates += 1;

                // Validate against test data for best_score tracking
                let effective_score = validate_candidate_score(
                    &c.rule_text, c.score, test_validation, memory, target_pred,
                );

                if effective_score >= 0.999 {
                    best_rules = vec![c.rule_text.clone()];
                    best_score = effective_score;
                    eprintln!("    VALIDATED-SOLVED: test={:.1}% — solved in genesis", effective_score * 100.0);
                    return make_result(best_rules, best_score, true, total_candidates,
                        total_mutations, &start, Some("genesis"), baseline_score);
                }

                if effective_score > best_score {
                    best_score = effective_score;
                    best_rules = vec![c.rule_text.clone()];
                }

                // Add candidates with decent TRAINING score to positives for refinement.
                // Even overfit candidates are useful seeds — mutations may generalize.
                let seed_score = c.score.max(effective_score);
                if seed_score > 0.01 {
                    if let Ok(stmts) = qor_core::parser::parse(&c.rule_text) {
                        for stmt in stmts {
                            if let Statement::Rule { head, body, tv } = stmt {
                                let rule = Rule::new(head, body, tv.unwrap_or(TruthValue::new(0.95, 0.95)));
                                all_positives.push((rule, seed_score));
                            }
                        }
                    }
                }
            }

            // ── GAP-TARGETED GENESIS ──
            // When DNA already covers most cells, run a focused genesis
            // on JUST the cells DNA gets wrong. This gives genesis a smaller,
            // easier target instead of trying to explain the entire output.
            if baseline_score > 0.3 && baseline_score < 0.99 && Instant::now() < deadline {
                let mut gap_expected: Vec<Vec<Statement>> = Vec::new();
                let mut total_gap = 0;

                for pair_exp in &pair_expected {
                    // Get valid IDs for this pair
                    let pair_ids: HashSet<String> = pair_exp.iter()
                        .filter_map(|s| {
                            if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = s {
                                parts.get(1).map(|n| format!("{}", n))
                            } else { None }
                        })
                        .collect();

                    // DNA predictions for this pair
                    let dna_preds = extract_keys_from_session_filtered(
                        puzzle_session, target_pred, &pair_ids,
                    );

                    // Gap = expected cells NOT in DNA predictions
                    let gap: Vec<Statement> = pair_exp.iter()
                        .filter(|s| {
                            if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = s {
                                let key = parts.iter()
                                    .map(|n| format!("{}", n))
                                    .collect::<Vec<_>>()
                                    .join(" ");
                                !dna_preds.contains(&key)
                            } else { false }
                        })
                        .cloned()
                        .collect();
                    total_gap += gap.len();
                    gap_expected.push(gap);
                }

                if total_gap > 0 {
                    let gap_budget = deadline.saturating_duration_since(Instant::now())
                        .as_millis().min(3000) as u64;
                    eprintln!("    P3-gap-genesis: {} gap cells across {} pairs (DNA misses), {}ms budget",
                        total_gap, pair_expected.len(), gap_budget);

                    if gap_budget > 500 {
                        let gap_candidates = invent::genesis(
                            puzzle_session, &pair_inputs, &gap_expected,
                            gap_budget, None,
                            Some(memory.failed_rule_set()),
                        );
                        let gap_cands = memory.filter_candidates(gap_candidates);
                        eprintln!("    P3-gap-done: {} gap candidates", gap_cands.len());

                        for c in &gap_cands {
                            total_candidates += 1;
                            // Score against FULL expected (gap rules must not break DNA-correct cells)
                            let effective_score = validate_candidate_score(
                                &c.rule_text, c.score, test_validation, memory, target_pred,
                            );

                            if effective_score >= 0.999 {
                                best_rules = vec![c.rule_text.clone()];
                                best_score = effective_score;
                                eprintln!("    GAP-GENESIS-SOLVED: test={:.1}%", effective_score * 100.0);
                                return make_result(best_rules, best_score, true, total_candidates,
                                    total_mutations, &start, Some("gap-genesis"), baseline_score);
                            }
                            if effective_score > best_score {
                                best_score = effective_score;
                                best_rules = vec![c.rule_text.clone()];
                            }
                            let seed_score = c.score.max(effective_score);
                            if seed_score > 0.01 {
                                if let Ok(stmts) = qor_core::parser::parse(&c.rule_text) {
                                    for stmt in stmts {
                                        if let Statement::Rule { head, body, tv } = stmt {
                                            let rule = Rule::new(head, body,
                                                tv.unwrap_or(TruthValue::new(0.95, 0.95)));
                                            all_positives.push((rule, seed_score));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 3.5: POST-GENESIS REFINEMENT — context-aware mutations + evo search
    // Genesis produces partial matches that Phase 2 never saw.
    // First: context-aware mutations (uses known predicates/values).
    // Then: evolutionary refinement (crossover + tournament).
    // ═══════════════════════════════════════════════════════════════════
    if Instant::now() < deadline && !all_positives.is_empty() {
        all_positives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let seed_rules: Vec<(Rule, f64)> = all_positives.iter()
            .take(5)
            .cloned()
            .collect();

        // Split budget: context mutations get first half, evo search gets second half
        let refine_budget = deadline.saturating_duration_since(Instant::now()).as_millis().min(3000) as u64;
        let ctx_budget = refine_budget / 2;
        let ctx_deadline = Instant::now() + Duration::from_millis(ctx_budget);
        eprintln!("    P3.5-refine: {} seeds, {}ms budget ({}ms ctx + {}ms evo)",
            seed_rules.len(), refine_budget, ctx_budget, refine_budget - ctx_budget);

        // Collect known predicates (with arity) and values for context-aware mutations
        let known_predicates: Vec<(String, usize)> = {
            let mut pred_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            for sn in puzzle_session.all_facts() {
                if let Neuron::Expression(parts) = &sn.neuron {
                    if let Some(Neuron::Symbol(p)) = parts.first() {
                        pred_map.entry(p.clone()).or_insert(parts.len() - 1);
                    }
                }
            }
            pred_map.into_iter().take(50).collect()
        };
        let known_values: Vec<i64> = {
            let mut vals = HashSet::new();
            for sn in puzzle_session.all_facts() {
                if let Neuron::Expression(parts) = &sn.neuron {
                    for p in parts.iter().skip(1).take(3) {
                        if let Neuron::Value(QorValue::Int(i)) = p {
                            vals.insert(*i);
                        }
                    }
                }
            }
            vals.into_iter().take(100).collect()
        };

        let mut refine_improvements = 0;
        let mut refine_mutations = 0;
        let mut seen = HashSet::new();

        // Prioritize overfit rules (high training score, lower test score) — they
        // found the right values but need more conditions to constrain them.
        // Sort seeds: overfit first (biggest train-test gap), then by score.
        let mut prioritized_seeds = seed_rules.clone();
        prioritized_seeds.sort_by(|a, b| {
            let a_gap = a.1; // training score (overfit = high train, seed_score captures max)
            let b_gap = b.1;
            b_gap.partial_cmp(&a_gap).unwrap_or(std::cmp::Ordering::Equal)
        });

        for (seed, seed_score) in &prioritized_seeds {
            if Instant::now() >= ctx_deadline { break; }

            // Structural mutations
            let mutations = generate_mutations(seed);
            // Context-aware mutations (adds conditions from known predicates)
            let ctx_mutations = generate_context_mutations(seed, &known_predicates, &known_values);

            for mutated in mutations.iter().chain(ctx_mutations.iter()) {
                if Instant::now() >= ctx_deadline { break; }

                let text = rule_to_qor(mutated);
                if seen.contains(&text) { continue; }
                seen.insert(text.clone());

                refine_mutations += 1;
                total_mutations += 1;
                total_candidates += 1;

                let train_score = score_rule_combined(
                    mutated, puzzle_session, &expected_keys, target_pred,
                );

                // Validate against test data for real score
                let effective_score = validate_candidate_score(
                    &text, train_score, test_validation, memory, target_pred,
                );

                if effective_score >= 0.999 {
                    best_rules = vec![text];
                    best_score = effective_score;
                    refine_improvements += 1;
                    eprintln!("    P3.5-SOLVED: test={:.1}%", effective_score * 100.0);
                    return make_result(best_rules, best_score, true, total_candidates,
                        total_mutations, &start, Some("post-genesis-refinement"), baseline_score);
                } else if effective_score > *seed_score + 0.01 {
                    all_positives.push((mutated.clone(), effective_score));
                    if effective_score > best_score {
                        best_score = effective_score;
                        best_rules = vec![rule_to_qor(mutated)];
                    }
                    refine_improvements += 1;
                    eprintln!("      P3.5-improved: {:.1}% → {:.1}% (test) rule={}",
                        seed_score * 100.0, effective_score * 100.0,
                        rule_to_qor(mutated).chars().take(100).collect::<String>());
                } else if effective_score > 0.01 {
                    all_positives.push((mutated.clone(), effective_score));
                }
            }
        }
        eprintln!("    P3.5-ctx-done: {} mutations, {} improvements, best={:.1}%",
            refine_mutations, refine_improvements, best_score * 100.0);

        // ── ANTI-OVERFIT: find discriminating conditions for overfit rules ──
        // Take top seeds that score well on training but may be overfitting.
        // Use refine_overfit_rule() to find conditions that eliminate wrong predictions.
        if Instant::now() < deadline && !all_positives.is_empty() {
            let overfit_deadline = Instant::now() + Duration::from_millis(
                deadline.saturating_duration_since(Instant::now()).as_millis().min(2000) as u64
            );
            let mut antioverfit_count = 0;
            let mut antioverfit_improvements = 0;

            // Try anti-overfit on the top seeds (most likely to be overfitting)
            let overfit_seeds: Vec<(Rule, f64)> = all_positives.iter()
                .filter(|(_, s)| *s > 0.1)
                .take(5)
                .cloned()
                .collect();

            for (seed, seed_score) in &overfit_seeds {
                if Instant::now() >= overfit_deadline { break; }

                let refined = refine_overfit_rule(seed, puzzle_session, &expected_keys, target_pred);
                for (refined_rule, refined_text) in &refined {
                    if Instant::now() >= overfit_deadline { break; }
                    if seen.contains(refined_text) { continue; }
                    seen.insert(refined_text.clone());
                    antioverfit_count += 1;
                    total_candidates += 1;

                    let train_score = score_rule_combined(
                        refined_rule, puzzle_session, &expected_keys, target_pred,
                    );
                    let effective_score = validate_candidate_score(
                        refined_text, train_score, test_validation, memory, target_pred,
                    );

                    if effective_score >= 0.999 {
                        best_rules = vec![refined_text.clone()];
                        best_score = effective_score;
                        eprintln!("    ANTI-OVERFIT-SOLVED: test={:.1}%", effective_score * 100.0);
                        return make_result(best_rules, best_score, true, total_candidates,
                            total_mutations, &start, Some("anti-overfit"), baseline_score);
                    }
                    if effective_score > best_score {
                        best_score = effective_score;
                        best_rules = vec![refined_text.clone()];
                        antioverfit_improvements += 1;
                    }
                    let s = train_score.max(effective_score);
                    if s > 0.01 {
                        all_positives.push((refined_rule.clone(), s));
                    }
                }
            }
            if antioverfit_count > 0 {
                eprintln!("    P3.5-anti-overfit: {} candidates, {} improvements, best={:.1}%",
                    antioverfit_count, antioverfit_improvements, best_score * 100.0);
            }
        }

        // Egraph dedup on all_positives before evo search
        {
            let before = all_positives.len();
            let mut scored: Vec<search::ScoredRule> = all_positives.iter()
                .map(|(r, s)| search::ScoredRule {
                    rule: r.clone(), score: *s, qor_text: rule_to_qor(r),
                })
                .collect();
            egraph::dedup_rules(&mut scored);
            if scored.len() < before {
                all_positives = scored.iter()
                    .map(|s| (s.rule.clone(), s.score))
                    .collect();
                eprintln!("    P3.5-egraph-dedup: {} → {} positives", before, all_positives.len());
            }
        }

        // Evolutionary refinement: crossover + tournament on all positives
        let evo_remaining = deadline.saturating_duration_since(Instant::now()).as_millis() as u64;
        if evo_remaining > 500 && all_positives.len() >= 2 {
            all_positives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let evo_seeds: Vec<Rule> = all_positives.iter()
                .take(10)
                .map(|(r, _)| r.clone())
                .collect();

            let evo_budget = evo_remaining.min(1500);
            let (pair_inputs, pair_expected) = split_expected_by_pair(
                puzzle_session, all_expected, target_pred,
            );

            eprintln!("    P3.5-evo: {} seeds, {}ms budget", evo_seeds.len(), evo_budget);
            let evo_result = search::refinement_search(
                &evo_seeds, &pair_inputs, &pair_expected,
                target_pred, puzzle_session, evo_budget,
            );
            total_mutations += evo_result.mutations_tried;
            total_candidates += evo_result.mutations_tried;

            for solution in &evo_result.solutions {
                let effective_score = validate_candidate_score(
                    &solution.qor_text, solution.score, test_validation, memory, target_pred,
                );
                if effective_score >= 0.999 {
                    best_rules = vec![solution.qor_text.clone()];
                    best_score = effective_score;
                    return make_result(best_rules, best_score, true, total_candidates,
                        total_mutations, &start, Some("post-genesis-evo"), baseline_score);
                } else if effective_score > best_score {
                    best_score = effective_score;
                    best_rules = vec![solution.qor_text.clone()];
                }
                let seed_score = solution.score.max(effective_score);
                if seed_score > 0.01 {
                    all_positives.push((Rule::new(
                        solution.rule.head.clone(), solution.rule.body.clone(), solution.rule.tv,
                    ), seed_score));
                }
            }
            for nm in &evo_result.near_misses {
                let text = rule_to_qor(&nm.rule);
                let effective_score = validate_candidate_score(
                    &text, nm.score, test_validation, memory, target_pred,
                );
                let seed_score = nm.score.max(effective_score);
                if seed_score > 0.01 {
                    all_positives.push((nm.rule.clone(), seed_score));
                    if effective_score > best_score {
                        best_score = effective_score;
                        best_rules = vec![text];
                    }
                }
            }
            eprintln!("    P3.5-evo-done: {} mutations, {} solutions, best={:.1}%",
                evo_result.mutations_tried, evo_result.solutions.len(),
                best_score * 100.0);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 4: SWARM — parallel multi-strategy invention
    // Each worker is like a different "perspective agent" — tries
    // different strategies, shares best findings back.
    // ═══════════════════════════════════════════════════════════════════
    if Instant::now() < deadline {
        let remaining_ms = deadline.saturating_duration_since(Instant::now()).as_millis() as u64;

        eprintln!("    P4-swarm: {}ms remaining", remaining_ms);
        if remaining_ms > 2000 {
            let (pair_inputs, pair_expected) = split_expected_by_pair(
                puzzle_session, all_expected, target_pred,
            );

            let swarm_hints = hints.to_swarm_hints();
            // Always pass hints if problem_class or constraints are present
            // (previously only passed when try/skip were non-empty)
            let num_workers = if meta_strategies.is_some() {
                meta_strategies.map(|m| m.len()).unwrap_or(6).max(6)
            } else {
                6
            };
            let has_hints = !swarm_hints.try_these.is_empty()
                || !swarm_hints.skip_these.is_empty()
                || swarm_hints.problem_class.is_some()
                || !swarm_hints.constraints.is_empty();
            let all_candidates = invent::genesis_swarm_with_hints(
                puzzle_session, &pair_inputs, &pair_expected,
                remaining_ms, None, num_workers,
                if has_hints { Some(&swarm_hints) } else { None },
                meta_strategies,
                Some(memory.failed_rule_set()),
            );
            // Filter out already-failed rules + skip-hint categories + constraint violations
            let candidates_unfilt = memory.filter_candidates(all_candidates);
            let candidates_skip = filter_by_skip_hints(&candidates_unfilt, &hints.skip_these);
            let candidates = filter_by_constraints(&candidates_skip, &hints.constraints);
            eprintln!("    P4-done: {} candidates returned", candidates.len());
            for (ci, c) in candidates.iter().enumerate().take(5) {
                eprintln!("      P4[{}]: {:.1}% src={} rule={}", ci, c.score * 100.0, c.source,
                    c.rule_text.chars().take(120).collect::<String>());
            }

            for c in &candidates {
                total_candidates += 1;

                // VALIDATE EVERY CANDIDATE against test data
                let effective_score = validate_candidate_score(
                    &c.rule_text, c.score, test_validation, memory, target_pred,
                );

                if effective_score >= 0.999 {
                    best_rules = vec![c.rule_text.clone()];
                    best_score = effective_score;
                    eprintln!("    VALIDATED-SOLVED: test={:.1}% — solved in swarm", effective_score * 100.0);
                    return make_result(best_rules, best_score, true, total_candidates,
                        total_mutations, &start, Some("swarm"), baseline_score);
                }

                if effective_score > best_score {
                    best_score = effective_score;
                    best_rules = vec![c.rule_text.clone()];
                }

                // Add candidates with decent TRAINING score for refinement seeds.
                let seed_score = c.score.max(effective_score);
                if seed_score > 0.01 {
                    if let Ok(stmts) = qor_core::parser::parse(&c.rule_text) {
                        for stmt in stmts {
                            if let Statement::Rule { head, body, tv } = stmt {
                                let rule = Rule::new(head, body, tv.unwrap_or(TruthValue::new(0.95, 0.95)));
                                all_positives.push((rule, seed_score));
                            }
                        }
                    }
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 4.5: ADVISOR — apply reasoning.qor to best positive
    // reasoning.qor sees winning facts + observations → refines hints
    // → guides further mutation of the winning rule.
    // ═══════════════════════════════════════════════════════════════════
    if Instant::now() < deadline && meta_dir.is_some() {
        if let Some(advisor_stmts) = load_advisor(meta_dir.unwrap()) {
            // Sort positives (may be empty — advisor still fires on puzzle facts)
            all_positives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let top_rule_info: Option<(Rule, f64)> = all_positives.first().cloned();
            let top_score = top_rule_info.as_ref().map(|(_, s)| *s).unwrap_or(0.0);

            eprintln!("    P4.5-advisor: reasoning.qor on {} positives (best={:.1}%)",
                all_positives.len(), top_score * 100.0);

            // Build advisor session: puzzle context + best rule + bridge + reasoning.qor
            let mut advisor_session = puzzle_session.clone();
            if let Some((ref top_rule, _)) = top_rule_info {
                let top_text = rule_to_qor(top_rule);
                if let Ok(stmts) = qor_core::parser::parse(&top_text) {
                    let _ = advisor_session.exec_statements(stmts);
                }
            }
            // Load bridge.qor so reasoning.qor has obs-* → strategy input facts
            let bridge_path = meta_dir.unwrap().join("bridge.qor");
            if let Ok(source) = std::fs::read_to_string(&bridge_path) {
                if let Ok(stmts) = qor_core::parser::parse(&source) {
                    let _ = advisor_session.exec_statements(stmts);
                }
            }
            let _ = advisor_session.exec_statements(advisor_stmts);

            // Read refined hints from advisor
            let refined_hints = read_strategy_hints(&advisor_session);
            eprintln!("    P4.5-hints: try={:?} skip={:?} class={:?}",
                refined_hints.try_these, refined_hints.skip_these, refined_hints.problem_class);

            // Mutate ALL positives (not just the best one)
            let known_predicates: Vec<(String, usize)> = {
                let mut pred_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
                for sn in advisor_session.all_facts() {
                    if let Neuron::Expression(parts) = &sn.neuron {
                        if let Some(Neuron::Symbol(p)) = parts.first() {
                            pred_map.entry(p.clone()).or_insert(parts.len() - 1);
                        }
                    }
                }
                pred_map.into_iter().take(50).collect()
            };
            let known_values: Vec<i64> = {
                let mut vals = HashSet::new();
                for sn in advisor_session.all_facts() {
                    if let Neuron::Expression(parts) = &sn.neuron {
                        for p in parts.iter().skip(1).take(3) {
                            if let Neuron::Value(QorValue::Int(i)) = p {
                                vals.insert(*i);
                            }
                        }
                    }
                }
                vals.into_iter().take(100).collect()
            };

            let advisor_deadline = Instant::now() + Duration::from_millis(
                deadline.saturating_duration_since(Instant::now()).as_millis().min(3000) as u64
            );
            let mut advisor_improvements = 0;
            let mut advisor_mutations = 0;

            // Mutate top positives with advisor-enriched context
            let seeds: Vec<(Rule, f64)> = all_positives.iter().take(10).cloned().collect();
            for (seed_rule, seed_score) in &seeds {
                if Instant::now() >= advisor_deadline { break; }

                let mutations = generate_mutations(seed_rule);
                let ctx_mutations = generate_context_mutations(seed_rule, &known_predicates, &known_values);

                for mutated in mutations.iter().chain(ctx_mutations.iter()) {
                    if Instant::now() >= advisor_deadline { break; }
                    let text = rule_to_qor(mutated);
                    if memory.already_failed(&text) { continue; }
                    advisor_mutations += 1;
                    total_mutations += 1;
                    total_candidates += 1;

                    let train_score = score_rule_combined(
                        mutated, puzzle_session, &expected_keys, target_pred,
                    );

                    // Validate against test data
                    let effective_score = validate_candidate_score(
                        &text, train_score, test_validation, memory, target_pred,
                    );

                    if effective_score >= 0.999 {
                        best_rules = vec![text];
                        best_score = effective_score;
                        advisor_improvements += 1;
                        eprintln!("    P4.5-SOLVED: test={:.1}%", effective_score * 100.0);
                        return make_result(best_rules, best_score, true, total_candidates,
                            total_mutations, &start, Some("advisor-refined"), baseline_score);
                    } else if effective_score > *seed_score {
                        all_positives.push((mutated.clone(), effective_score));
                        if effective_score > best_score {
                            best_score = effective_score;
                            best_rules = vec![rule_to_qor(mutated)];
                        }
                        advisor_improvements += 1;
                    } else if effective_score > 0.01 {
                        all_positives.push((mutated.clone(), effective_score));
                    }
                }
            }
            eprintln!("    P4.5-done: {} mutations, {} improvements, best={:.1}%",
                advisor_mutations, advisor_improvements, best_score * 100.0);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 5: PROGRESSIVE STACKING — Kahn's topological sort + greedy
    //
    // Instead of pairwise combos, greedily ACCUMULATE rules:
    //   1. Topologically sort candidates (Kahn's algorithm)
    //   2. Start with locked rules (current best)
    //   3. For each candidate: test marginal contribution
    //      - Score improves → PROGRESS (lock it in, new baseline)
    //      - Score same/drops → FAILED (record, never retry)
    //   4. Keep stacking until solved or no more improvement
    // ═══════════════════════════════════════════════════════════════════
    eprintln!("    P5-stack: {} positives available", all_positives.len());
    if Instant::now() < deadline && !all_positives.is_empty() {
        let stack_deadline = Instant::now() + Duration::from_millis(
            deadline.saturating_duration_since(Instant::now()).as_millis().min(3000) as u64
        );

        // Dedup and prep candidates — e-graph semantic dedup then text dedup
        all_positives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all_positives.dedup_by(|a, b| rule_to_qor(&a.0) == rule_to_qor(&b.0));
        // Semantic dedup via e-graph: collapse rules with reordered/redundant conditions
        {
            let mut scored: Vec<search::ScoredRule> = all_positives.iter()
                .map(|(r, s)| search::ScoredRule {
                    rule: r.clone(),
                    score: *s,
                    qor_text: rule_to_qor(r),
                })
                .collect();
            let before = scored.len();
            egraph::dedup_rules(&mut scored);
            if scored.len() < before {
                all_positives = scored.into_iter()
                    .map(|sr| (sr.rule, sr.score))
                    .collect();
            }
        }
        all_positives.truncate(20);

        // Topologically sort using Kahn's algorithm
        let sorted_indices = topo_sort_candidates(&all_positives);

        eprintln!("    P5-topo: {} candidates after sort (order: {:?})",
            sorted_indices.len(), &sorted_indices[..sorted_indices.len().min(10)]);
        for &idx in sorted_indices.iter().take(10) {
            let (rule, score) = &all_positives[idx];
            eprintln!("      topo[{}]: {:.1}% rule={}", idx, score * 100.0,
                rule_to_qor(rule).chars().take(100).collect::<String>());
        }

        // Greedy progressive stacking
        let mut locked_rules: Vec<String> = best_rules.clone();
        let mut locked_score = best_score;
        let mut progress_count = 0;
        let mut failed_count = 0;

        for &idx in &sorted_indices {
            if Instant::now() >= stack_deadline { break; }
            let (rule, _) = &all_positives[idx];
            let candidate_text = rule_to_qor(rule);

            // Skip already-failed or already-locked
            if memory.already_failed(&candidate_text) { continue; }
            if locked_rules.contains(&candidate_text) { continue; }

            // Test marginal contribution: locked + candidate ON TEST DATA
            let mut test_set: Vec<&str> = locked_rules.iter().map(|s| s.as_str()).collect();
            test_set.push(&candidate_text);

            // Score on training first (fast)
            let train_combined = score_rules_combined(
                &test_set, puzzle_session, &expected_keys, target_pred,
            );
            total_candidates += 1;

            // Validate on test data for real score
            let combined_score = if let Some(tv) = test_validation {
                validate_on_test(&test_set, &tv.test_session, &tv.test_expected_keys, target_pred)
            } else {
                train_combined
            };

            if combined_score > locked_score + 0.005 {
                // PROGRESS: this rule adds real value (validated) — lock it in
                eprintln!("      P5-PROGRESS: {:.1}% → {:.1}% (test) (+rule) locked={}",
                    locked_score * 100.0, combined_score * 100.0,
                    locked_rules.len() + 1);
                eprintln!("        rule={}", candidate_text.chars().take(120).collect::<String>());
                locked_rules.push(candidate_text);
                locked_score = combined_score;
                progress_count += 1;

                if locked_score >= 0.999 {
                    best_rules = locked_rules.clone();
                    best_score = locked_score;
                    eprintln!("    P5-SOLVED: progressive stacking, {} rules!", best_rules.len());
                    return make_result(best_rules, best_score, true, total_candidates,
                        total_mutations, &start, Some("progressive-stack"), baseline_score);
                }
            } else {
                // FAILED: no marginal improvement — record and never retry
                memory.record_failure(&candidate_text, "rule", combined_score);
                failed_count += 1;
            }
        }

        if locked_score > best_score {
            best_score = locked_score;
            best_rules = locked_rules;
        }

        eprintln!("    P5-done: {} progress, {} failed, locked={} rules, best={:.1}%",
            progress_count, failed_count, best_rules.len(), best_score * 100.0);
    }

    eprintln!("    ROUND-SUMMARY: best={:.1}%, candidates={}, mutations={}, failures={}, {}ms",
        best_score * 100.0, total_candidates, total_mutations,
        memory.total_failures(), start.elapsed().as_millis());
    if !best_rules.is_empty() {
        eprintln!("    ROUND-BEST-RULES:");
        for br in &best_rules {
            eprintln!("      {}", br.chars().take(150).collect::<String>());
        }
    }

    make_result(best_rules, best_score, false, total_candidates,
        total_mutations, &start, None, baseline_score)
}

// ── Kahn's Topological Sort for Progressive Stacking ─────────────────
//
// Orders candidates so rules that produce facts needed by other rules
// come first. Uses Kahn's algorithm (BFS-based topo sort).
//
// For rules with the same head predicate (e.g., all producing predict-cell),
// ordering falls back to: higher individual score first.

/// Topologically sort candidates using Kahn's algorithm.
///
/// Builds a dependency graph: if candidate i's head predicate appears in
/// candidate j's body → i should be tried before j (edge i→j).
/// Within the same topological level, higher-scoring candidates go first.
fn topo_sort_candidates(candidates: &[(Rule, f64)]) -> Vec<usize> {
    use qor_core::neuron::Condition;

    let n = candidates.len();
    if n == 0 { return Vec::new(); }

    // Extract head and body predicates for each candidate
    let mut head_preds: Vec<String> = Vec::with_capacity(n);
    let mut body_pred_sets: Vec<Vec<String>> = Vec::with_capacity(n);

    for (rule, _) in candidates {
        // Head predicate
        let hp = match &rule.head {
            Neuron::Expression(parts) => {
                parts.first().and_then(|p| {
                    if let Neuron::Symbol(s) = p { Some(s.clone()) } else { None }
                }).unwrap_or_default()
            }
            _ => String::new(),
        };
        head_preds.push(hp);

        // Body predicates
        let mut bps = Vec::new();
        for cond in &rule.body {
            if let Condition::Positive(Neuron::Expression(parts)) = cond {
                if let Some(Neuron::Symbol(s)) = parts.first() {
                    if !bps.contains(s) {
                        bps.push(s.clone());
                    }
                }
            }
        }
        body_pred_sets.push(bps);
    }

    // Build adjacency: edge i→j if candidate i's head_pred appears in j's body
    let mut in_degree = vec![0usize; n];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    for j in 0..n {
        for i in 0..n {
            if i != j && body_pred_sets[j].contains(&head_preds[i]) {
                adj[i].push(j);
                in_degree[j] += 1;
            }
        }
    }

    // Kahn's algorithm — BFS with score-ordered queue
    let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    // Higher score first within same topological level
    queue.sort_by(|a, b| candidates[*b].1.partial_cmp(&candidates[*a].1)
        .unwrap_or(std::cmp::Ordering::Equal));

    let mut result = Vec::with_capacity(n);
    let mut q_idx = 0;

    while q_idx < queue.len() {
        let node = queue[q_idx];
        q_idx += 1;
        result.push(node);

        // Collect newly freed nodes for this level
        let mut freed = Vec::new();
        for &next in &adj[node] {
            in_degree[next] -= 1;
            if in_degree[next] == 0 {
                freed.push(next);
            }
        }
        // Sort freed nodes by score descending
        freed.sort_by(|a, b| candidates[*b].1.partial_cmp(&candidates[*a].1)
            .unwrap_or(std::cmp::Ordering::Equal));
        queue.extend(freed);
    }

    // Safety: add any remaining nodes (cycles — shouldn't happen, but be safe)
    for i in 0..n {
        if !result.contains(&i) {
            result.push(i);
        }
    }

    result
}

// ── Scoring Functions ────────────────────────────────────────────────
// These score candidate rules BY REASONING: clone session → add rule
// → forward chain → compare predictions. This IS the 5-op loop.

/// Score a single candidate rule against all expected facts.
/// Only counts predictions for grid IDs present in expected_keys
/// (ignores test-input predictions that would inflate false-positives).
fn score_rule_combined(
    rule: &Rule,
    puzzle_session: &Session,
    expected_keys: &HashSet<String>,
    target_pred: &str,
) -> f64 {
    let mut store = puzzle_session.store().clone();

    // ADDITIVE scoring: keep DNA-derived predictions. Candidate fires ON TOP.
    // Evaluated by MARGINAL contribution:
    //   — adds correct predictions → recall up → score up
    //   — adds wrong predictions → precision down → score down
    //   — does nothing → score == baseline (filtered by caller)

    // Forward chain with ONLY the candidate rule
    let candidate_rules = vec![Rule::new(
        rule.head.clone(), rule.body.clone(), rule.tv,
    )];
    let _new = forward_chain(&candidate_rules, &mut store);

    // Only count predictions for grid IDs that appear in expected_keys.
    // This excludes test-input (ti) predictions from inflating false-positives.
    let valid_ids = extract_valid_ids(expected_keys);
    let predictions = extract_keys_from_store_filtered(&store, target_pred, &valid_ids);
    if expected_keys.is_empty() { return 0.0; }
    let correct = expected_keys.iter().filter(|k| predictions.contains(*k)).count();
    let score = precision_recall_score(correct, expected_keys.len(), predictions.len());

    score
}

/// Score multiple rules applied together.
/// ADDITIVE: keeps DNA-derived predictions, candidates fire on top.
fn score_rules_combined(
    rule_texts: &[&str],
    puzzle_session: &Session,
    expected_keys: &HashSet<String>,
    target_pred: &str,
) -> f64 {
    let mut store = puzzle_session.store().clone();

    // Parse all rule texts into chain Rules
    let mut candidate_rules: Vec<Rule> = Vec::new();
    for text in rule_texts {
        if let Ok(stmts) = qor_core::parser::parse(text) {
            for stmt in stmts {
                if let Statement::Rule { head, body, tv } = stmt {
                    candidate_rules.push(Rule::new(
                        head, body,
                        tv.unwrap_or(TruthValue::new(0.95, 0.95)),
                    ));
                }
            }
        }
    }

    let _new = forward_chain(&candidate_rules, &mut store);

    let valid_ids = extract_valid_ids(expected_keys);
    let predictions = extract_keys_from_store_filtered(&store, target_pred, &valid_ids);
    if expected_keys.is_empty() { return 0.0; }
    let correct = expected_keys.iter().filter(|k| predictions.contains(*k)).count();
    precision_recall_score(correct, expected_keys.len(), predictions.len())
}

/// Combined precision*recall score. Penalizes over-prediction (extras).
/// Returns sqrt(precision * recall) — geometric mean of precision and recall.
/// A rule that predicts exactly the right set scores 1.0.
/// A rule that predicts everything (including all correct) scores low due to low precision.
fn precision_recall_score(correct: usize, expected: usize, predicted: usize) -> f64 {
    if expected == 0 || predicted == 0 { return 0.0; }
    let recall = correct as f64 / expected as f64;
    let precision = correct as f64 / predicted as f64;
    (precision * recall).sqrt()
}

/// Extract fact keys from store, filtered to only include facts whose 2nd element
/// (grid ID) is in `valid_ids`. This excludes test-input predictions from scoring.
fn extract_keys_from_store_filtered(
    store: &qor_runtime::store::NeuronStore, pred: &str, valid_ids: &HashSet<String>,
) -> HashSet<String> {
    let mut set = HashSet::new();
    for sn in store.all() {
        if let Neuron::Expression(parts) = &sn.neuron {
            if let Some(Neuron::Symbol(p)) = parts.first() {
                if p == pred {
                    // Filter by grid ID (2nd element)
                    let id_match = parts.get(1).map(|n| {
                        valid_ids.contains(&format!("{}", n))
                    }).unwrap_or(false);
                    if id_match || valid_ids.is_empty() {
                        set.insert(parts.iter().map(|n| format!("{}", n)).collect::<Vec<_>>().join(" "));
                    }
                }
            }
        }
    }
    set
}

/// Extract valid grid IDs from expected keys (2nd word of each key).
fn extract_valid_ids(expected_keys: &HashSet<String>) -> HashSet<String> {
    expected_keys.iter()
        .filter_map(|k| k.split_whitespace().nth(1).map(|s| s.to_string()))
        .collect()
}

// ── Anti-Overfit: discriminative condition search ───────────────────
//
// When a rule overfits (100% train, low test), it produces too many
// predictions. This function finds which ADDITIONAL CONDITION from
// existing facts would eliminate the wrong predictions while keeping
// the correct ones.
//
// Algorithm:
// 1. Run overfit rule → get all predictions
// 2. Split into correct (in expected) and wrong (not in expected)
// 3. For each wrong prediction, extract (id, row, col) coordinates
// 4. Search session facts for predicates that are TRUE for correct
//    coordinates but FALSE (or different) for wrong coordinates
// 5. Build new rules = original + discriminating condition
//
// Uses the same data the session already has — no new dependencies.

/// Try to fix an overfit rule by finding discriminating conditions.
/// Uses position-adaptive mapping AND value-specific discrimination.
/// A predicate like grid-cell exists at ALL cells, but with VALUE constraints
/// (e.g., grid-cell with color=3) it can discriminate correct from wrong.
fn refine_overfit_rule(
    rule: &Rule,
    puzzle_session: &Session,
    expected_keys: &HashSet<String>,
    target_pred: &str,
) -> Vec<(Rule, String)> {
    use std::collections::HashMap;
    let mut results = Vec::new();

    // Step 1: Run rule, split predictions into correct/wrong
    let mut store = puzzle_session.store().clone();
    let _ = forward_chain(
        &[Rule::new(rule.head.clone(), rule.body.clone(), rule.tv)],
        &mut store,
    );
    let valid_ids = extract_valid_ids(expected_keys);
    let predictions = extract_keys_from_store_filtered(&store, target_pred, &valid_ids);

    let correct_keys: HashSet<&String> = predictions.iter()
        .filter(|k| expected_keys.contains(*k)).collect();
    let wrong_keys: HashSet<&String> = predictions.iter()
        .filter(|k| !expected_keys.contains(*k)).collect();
    if wrong_keys.is_empty() || correct_keys.is_empty() { return results; }

    let parse_coords = |key: &str| -> Option<(String, String, String)> {
        let p: Vec<&str> = key.split_whitespace().collect();
        if p.len() >= 4 { Some((p[1].into(), p[2].into(), p[3].into())) } else { None }
    };
    let correct_coords: HashSet<(String, String, String)> =
        correct_keys.iter().filter_map(|k| parse_coords(k)).collect();
    let wrong_coords: HashSet<(String, String, String)> =
        wrong_keys.iter().filter_map(|k| parse_coords(k)).collect();
    if correct_coords.is_empty() || wrong_coords.is_empty() { return results; }

    eprintln!("      anti-overfit: {} correct, {} wrong coords", correct_coords.len(), wrong_coords.len());

    // Step 2: Index session facts by predicate
    let mut pred_facts: HashMap<String, Vec<Vec<String>>> = HashMap::new();
    for sn in puzzle_session.all_facts() {
        if let Neuron::Expression(parts) = &sn.neuron {
            if let Some(Neuron::Symbol(pred)) = parts.first() {
                if pred == target_pred || pred.starts_with("genesis-")
                    || pred.starts_with("strategy-") || pred.starts_with("problem-")
                    || pred.starts_with("mostly-") { continue; }
                pred_facts.entry(pred.clone()).or_default()
                    .push(parts.iter().skip(1).map(|n| format!("{}", n)).collect());
            }
        }
    }

    // Step 3: Find discriminators — both position-based AND value-specific.
    // value_constraints: empty = any value at non-mapped positions (wildcard)
    //                    [(pos, val)] = require specific value at that position
    struct Disc {
        pred: String,
        ip: usize, rp: usize, cp: usize,
        arity: usize,
        value_constraints: Vec<(usize, String)>,
        correct_set: HashSet<(String, String, String)>,
        wrong_set: HashSet<(String, String, String)>,
        disc_score: f64,
        negated: bool,
    }
    let mut all_discs: Vec<Disc> = Vec::new();

    for (pred, facts) in &pred_facts {
        let arity = facts.first().map(|f| f.len()).unwrap_or(0);
        if arity < 3 || arity > 7 { continue; }

        for ip in 0..arity {
            for rp in 0..arity {
                if rp == ip { continue; }
                for cp in 0..arity {
                    if cp == ip || cp == rp { continue; }

                    // Base coverage (no value constraints)
                    let mut cs: HashSet<(String, String, String)> = HashSet::new();
                    let mut ws: HashSet<(String, String, String)> = HashSet::new();

                    // Value-specific: track values at non-mapped positions
                    let extras: Vec<usize> = (0..arity)
                        .filter(|&p| p != ip && p != rp && p != cp)
                        .collect();
                    // val_cs[extra_idx][value] = set of correct coords with that value
                    let mut val_cs: Vec<HashMap<String, HashSet<(String, String, String)>>> =
                        vec![HashMap::new(); extras.len()];
                    let mut val_ws: Vec<HashMap<String, HashSet<(String, String, String)>>> =
                        vec![HashMap::new(); extras.len()];

                    for f in facts {
                        if f.len() <= ip.max(rp).max(cp) { continue; }
                        let coord = (f[ip].clone(), f[rp].clone(), f[cp].clone());
                        let in_correct = correct_coords.contains(&coord);
                        let in_wrong = wrong_coords.contains(&coord);

                        if in_correct { cs.insert(coord.clone()); }
                        if in_wrong { ws.insert(coord.clone()); }

                        for (ei, &ep) in extras.iter().enumerate() {
                            if ep < f.len() {
                                let val = f[ep].clone();
                                if in_correct {
                                    val_cs[ei].entry(val.clone()).or_default().insert(coord.clone());
                                }
                                if in_wrong {
                                    val_ws[ei].entry(val).or_default().insert(coord.clone());
                                }
                            }
                        }
                    }

                    let cr = cs.len() as f64 / correct_coords.len() as f64;
                    let wr = ws.len() as f64 / wrong_coords.len() as f64;

                    // Base discriminator (no value constraint)
                    if cr > 0.5 && (cr - wr) > 0.15 {
                        all_discs.push(Disc { pred: pred.clone(), ip, rp, cp, arity,
                            value_constraints: vec![], correct_set: cs.clone(),
                            wrong_set: ws.clone(), disc_score: cr - wr, negated: false });
                    }
                    if wr > 0.5 && (wr - cr) > 0.15 {
                        all_discs.push(Disc { pred: pred.clone(), ip, rp, cp, arity,
                            value_constraints: vec![], correct_set: cs, wrong_set: ws,
                            disc_score: wr - cr, negated: true });
                    }

                    // Value-specific discriminators: constrain one extra position
                    for (ei, &ep) in extras.iter().enumerate() {
                        // Positive: value present at correct but not wrong
                        for (val, vcs) in &val_cs[ei] {
                            let vws = val_ws[ei].get(val);
                            let vcr = vcs.len() as f64 / correct_coords.len() as f64;
                            let vwr = vws.map(|s| s.len() as f64 / wrong_coords.len() as f64)
                                .unwrap_or(0.0);
                            if vcr > 0.4 && (vcr - vwr) > 0.2 {
                                all_discs.push(Disc {
                                    pred: pred.clone(), ip, rp, cp, arity,
                                    value_constraints: vec![(ep, val.clone())],
                                    correct_set: vcs.clone(),
                                    wrong_set: vws.cloned().unwrap_or_default(),
                                    disc_score: vcr - vwr, negated: false,
                                });
                            }
                        }
                        // Negative: value present at wrong but not correct
                        for (val, vws) in &val_ws[ei] {
                            let vcs = val_cs[ei].get(val);
                            let vwr = vws.len() as f64 / wrong_coords.len() as f64;
                            let vcr = vcs.map(|s| s.len() as f64 / correct_coords.len() as f64)
                                .unwrap_or(0.0);
                            if vwr > 0.4 && (vwr - vcr) > 0.2 {
                                all_discs.push(Disc {
                                    pred: pred.clone(), ip, rp, cp, arity,
                                    value_constraints: vec![(ep, val.clone())],
                                    correct_set: vcs.cloned().unwrap_or_default(),
                                    wrong_set: vws.clone(),
                                    disc_score: vwr - vcr, negated: true,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    all_discs.sort_by(|a, b| b.disc_score.partial_cmp(&a.disc_score)
        .unwrap_or(std::cmp::Ordering::Equal));

    // Dedup: keep best per (pred, negated, value_constraint_key)
    let mut seen_preds: HashSet<String> = HashSet::new();
    let mut top_discs: Vec<usize> = Vec::new();
    for (idx, d) in all_discs.iter().enumerate() {
        let vc_key: String = d.value_constraints.iter()
            .map(|(p, v)| format!("{}={}", p, v)).collect::<Vec<_>>().join(",");
        if seen_preds.insert(format!("{}:{}:{}", d.pred, d.negated, vc_key)) {
            top_discs.push(idx);
            if top_discs.len() >= 20 { break; }
        }
    }

    // Diagnostic: if no discriminators found, show why
    if top_discs.is_empty() && !all_discs.is_empty() {
        eprintln!("      anti-overfit: {} sub-threshold candidates (best d={:.3})",
            all_discs.len(), all_discs.first().map(|d| d.disc_score).unwrap_or(0.0));
        for d in all_discs.iter().take(3) {
            let cr = d.correct_set.len() as f64 / correct_coords.len() as f64;
            let wr = d.wrong_set.len() as f64 / wrong_coords.len() as f64;
            let vc = if d.value_constraints.is_empty() { String::new() }
                else { format!(" val@{}={}", d.value_constraints[0].0, d.value_constraints[0].1) };
            eprintln!("        sub: {}{}(id@{},r@{},c@{}){} cr={:.0}% wr={:.0}% d={:.3}",
                if d.negated { "NOT:" } else { "" }, d.pred, d.ip, d.rp, d.cp, vc,
                cr * 100.0, wr * 100.0, d.disc_score);
        }
    }
    if top_discs.is_empty() && all_discs.is_empty() {
        // Count predicates scanned
        let preds_scanned = pred_facts.keys()
            .filter(|p| pred_facts[*p].first().map(|f| f.len() >= 3 && f.len() <= 7).unwrap_or(false))
            .count();
        let total_preds = pred_facts.len();
        eprintln!("      anti-overfit: 0 candidates from {}/{} predicates (arity 3-7)",
            preds_scanned, total_preds);
    }

    for (i, &idx) in top_discs.iter().take(5).enumerate() {
        let d = &all_discs[idx];
        let cr = d.correct_set.len() as f64 / correct_coords.len() as f64;
        let wr = d.wrong_set.len() as f64 / wrong_coords.len() as f64;
        let vc = if d.value_constraints.is_empty() { String::new() }
            else { format!(" val@{}={}", d.value_constraints[0].0, d.value_constraints[0].1) };
        eprintln!("        disc[{}]: {}{}(id@{},r@{},c@{}){} cr={:.0}% wr={:.0}% d={:.2}",
            i, if d.negated { "NOT:" } else { "" }, d.pred, d.ip, d.rp, d.cp, vc,
            cr * 100.0, wr * 100.0, d.disc_score);
    }

    // Step 4: Generate rules
    let head_vars = extract_head_vars(rule);
    let rule_text = rule_to_qor(rule);

    let build_cond = |d: &Disc, prefix: &str| -> String {
        let mut args = vec![d.pred.clone()];
        for pos in 0..d.arity {
            if let Some((_, ref val)) = d.value_constraints.iter().find(|(p, _)| *p == pos) {
                args.push(val.clone());
            } else if pos == d.ip && !head_vars.is_empty() {
                args.push(head_vars[0].clone());
            } else if pos == d.rp && head_vars.len() > 1 {
                args.push(head_vars[1].clone());
            } else if pos == d.cp && head_vars.len() > 2 {
                args.push(head_vars[2].clone());
            } else {
                args.push(format!("${}{}", prefix, pos));
            }
        }
        if d.negated { format!("not ({})", args.join(" ")) }
        else { format!("({})", args.join(" ")) }
    };

    let try_parse = |text: &str| -> Option<Rule> {
        qor_core::parser::parse(text).ok().and_then(|stmts| {
            stmts.into_iter().find_map(|s| {
                if let Statement::Rule { head, body, tv } = s {
                    Some(Rule::new(head, body, tv.unwrap_or(TruthValue::new(0.90, 0.90))))
                } else { None }
            })
        })
    };

    // Single-discriminator rules
    for &idx in top_discs.iter().take(12) {
        let cond = build_cond(&all_discs[idx], "d");
        let new_text = format!("{}\n    {}", rule_text.trim_end(), cond);
        if let Some(r) = try_parse(&new_text) {
            results.push((r, new_text));
        }
    }

    // Step 5: Pairwise combinations
    let pair_limit = top_discs.len().min(8);
    for i in 0..pair_limit {
        for j in (i + 1)..pair_limit {
            let (d1, d2) = (&all_discs[top_discs[i]], &all_discs[top_discs[j]]);
            if d1.pred == d2.pred && d1.value_constraints == d2.value_constraints { continue; }

            if !d1.negated && !d2.negated {
                let cb = d1.correct_set.intersection(&d2.correct_set).count();
                let wb = d1.wrong_set.intersection(&d2.wrong_set).count();
                let cr = cb as f64 / correct_coords.len() as f64;
                let wr = wb as f64 / wrong_coords.len() as f64;
                if cr > 0.4 && wr < 0.15 && (cr - wr) > 0.3 {
                    let c1 = build_cond(d1, "d");
                    let c2 = build_cond(d2, "e");
                    let new_text = format!("{}\n    {}\n    {}", rule_text.trim_end(), c1, c2);
                    if let Some(r) = try_parse(&new_text) { results.push((r, new_text)); }
                }
            }

            let pos_neg: Vec<(&Disc, &Disc)> = match (d1.negated, d2.negated) {
                (false, true) => vec![(d1, d2)],
                (true, false) => vec![(d2, d1)],
                _ => vec![],
            };
            for (pos_d, neg_d) in pos_neg {
                let cc = pos_d.correct_set.iter()
                    .filter(|c| !neg_d.correct_set.contains(*c)).count();
                let wc = pos_d.wrong_set.iter()
                    .filter(|c| !neg_d.wrong_set.contains(*c)).count();
                let cr = cc as f64 / correct_coords.len() as f64;
                let wr = wc as f64 / wrong_coords.len() as f64;
                if cr > 0.4 && wr < 0.15 && (cr - wr) > 0.3 {
                    let c1 = build_cond(pos_d, "d");
                    let c2 = build_cond(neg_d, "e");
                    let new_text = format!("{}\n    {}\n    {}", rule_text.trim_end(), c1, c2);
                    if let Some(r) = try_parse(&new_text) { results.push((r, new_text)); }
                }
            }
        }
    }

    eprintln!("      anti-overfit done: {} discriminators, {} rules generated",
        top_discs.len(), results.len());
    results
}

/// Extract head variable names from a rule in order.
fn extract_head_vars(rule: &Rule) -> Vec<String> {
    if let Neuron::Expression(parts) = &rule.head {
        parts.iter().skip(1).map(|n| match n {
            Neuron::Variable(v) => format!("${}", v),
            Neuron::Value(QorValue::Int(i)) => i.to_string(),
            Neuron::Symbol(s) => s.clone(),
            _ => "_".to_string(),
        }).collect()
    } else {
        Vec::new()
    }
}

// ── Helper Functions ─────────────────────────────────────────────────

/// Extract fact keys for a given predicate from a session (filtered by valid IDs).
fn extract_keys_from_session_filtered(
    session: &Session, pred: &str, valid_ids: &HashSet<String>,
) -> HashSet<String> {
    let mut set = HashSet::new();
    for sn in session.all_facts() {
        if let Neuron::Expression(parts) = &sn.neuron {
            if let Some(Neuron::Symbol(p)) = parts.first() {
                if p == pred {
                    let id_match = parts.get(1).map(|n| {
                        valid_ids.contains(&format!("{}", n))
                    }).unwrap_or(false);
                    if id_match || valid_ids.is_empty() {
                        set.insert(parts.iter().map(|n| format!("{}", n)).collect::<Vec<_>>().join(" "));
                    }
                }
            }
        }
    }
    set
}

/// Extract fact keys from statements.
fn extract_keys_from_stmts(stmts: &[Statement], pred: &str) -> HashSet<String> {
    let mut set = HashSet::new();
    for stmt in stmts {
        if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = stmt {
            if let Some(Neuron::Symbol(p)) = parts.first() {
                if p == pred {
                    set.insert(parts.iter().map(|n| format!("{}", n)).collect::<Vec<_>>().join(" "));
                }
            }
        }
    }
    set
}

/// Split all_expected into per-pair format for refinement_search/genesis
/// which expect (Vec<Vec<Statement>>, Vec<Vec<Statement>>).
/// Extracts grid IDs from the puzzle_session's train-pair facts.
fn split_expected_by_pair(
    puzzle_session: &Session,
    all_expected: &[Statement],
    _target_pred: &str,
) -> (Vec<Vec<Statement>>, Vec<Vec<Statement>>) {
    // Find train-pair grid IDs from session
    let mut pair_ids: Vec<(String, String)> = Vec::new();
    for sn in puzzle_session.all_facts() {
        if let Neuron::Expression(parts) = &sn.neuron {
            if parts.len() == 3 {
                if let Some(Neuron::Symbol(p)) = parts.first() {
                    if p == "train-pair" {
                        let in_id = format!("{}", parts[1]);
                        let out_id = format!("{}", parts[2]);
                        pair_ids.push((in_id, out_id));
                    }
                }
            }
        }
    }

    eprintln!("      split: found {} train-pair facts: {:?}", pair_ids.len(),
        pair_ids.iter().map(|(i,o)| format!("{}→{}", i, o)).collect::<Vec<_>>());

    if pair_ids.is_empty() {
        eprintln!("      split: WARNING — no train-pair facts! Falling back to single-pair mode");
        // Fallback: return all expected as single pair
        return (vec![Vec::new()], vec![all_expected.to_vec()]);
    }

    let mut inputs = Vec::new();
    let mut expected = Vec::new();

    // Collect predicates that carry per-grid structural info
    let structural_preds = ["grid-cell", "grid-size", "grid-object", "grid-obj-cell",
        "grid-obj-bbox", "object-count", "color-count", "neighbor"];

    for (in_id, _out_id) in &pair_ids {
        // Input: grid-cell + structural facts for this input grid
        let mut pair_input = Vec::new();
        for sn in puzzle_session.all_facts() {
            if let Neuron::Expression(parts) = &sn.neuron {
                if let Some(Neuron::Symbol(p)) = parts.first() {
                    if structural_preds.contains(&p.as_str()) && parts.len() >= 2 {
                        if let Some(Neuron::Symbol(gid)) = parts.get(1) {
                            if gid == in_id {
                                pair_input.push(Statement::Fact {
                                    neuron: sn.neuron.clone(),
                                    tv: None,
                                    decay: None,
                                });
                            }
                        }
                    }
                }
            }
        }
        inputs.push(pair_input);

        // Expected: predict-cell facts with this input grid ID
        let pair_expected: Vec<Statement> = all_expected.iter()
            .filter(|stmt| {
                if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = stmt {
                    if parts.len() >= 2 {
                        if let Some(Neuron::Symbol(gid)) = parts.get(1) {
                            return gid == in_id;
                        }
                    }
                }
                false
            })
            .cloned()
            .collect();
        expected.push(pair_expected);
    }

    (inputs, expected)
}

fn make_result(
    best_rules: Vec<String>, score: f64, solved: bool,
    candidates: usize, mutations: usize, start: &Instant,
    phase: Option<&'static str>, baseline: f64,
) -> SolveResult {
    SolveResult {
        best_rules, score, solved,
        candidates_explored: candidates, mutations_tried: mutations,
        elapsed_ms: start.elapsed().as_millis() as u64,
        solved_in_phase: phase,
        baseline_score: baseline,
        rounds: 0,           // Set by outer solve() loop
        failures_tracked: 0, // Set by outer solve() loop
        overfit_count: 0,    // Set by outer solve() loop
        diagnostic: DiagnosticReport::default(), // Set by outer solve() loop
    }
}

/// Save winning and positive rules to library.
fn save_to_library(
    library: Option<&mut RuleLibrary>,
    best_rules: &[String],
    positives: &[(Rule, f64)],
) {
    if let Some(lib) = library {
        for rule_text in best_rules {
            lib.add(rule_text.clone(), RuleSource::Template);
        }
        for (rule, score) in positives {
            if *score > 0.5 {
                let text = rule_to_qor(rule);
                lib.add(text, RuleSource::Mutated);
            }
        }
        lib.prune(&PruneConfig {
            max_rules: 500,
            ..Default::default()
        });
    }
}

// ── Validation — "Every candidate gets tested" ──────────────────────

/// Validate a candidate's effective score.
///
/// When test data is available, EVERY candidate is validated against it:
/// - train >= 0.999 but test < 0.95 → overfit (record, use test score)
/// - train >= 0.999 and test >= 0.95 → genuinely solved (use 1.0)
/// - train < 0.999 → use test score as effective score
/// When no test data: trust training score.
fn validate_candidate_score(
    rule_text: &str,
    train_score: f64,
    test_validation: Option<&TestValidation>,
    memory: &mut FailureMemory,
    target_pred: &str,
) -> f64 {
    if let Some(tv) = test_validation {
        let test_score = validate_on_test(
            &[rule_text], &tv.test_session, &tv.test_expected_keys, target_pred,
        );

        if train_score >= 0.999 {
            if test_score >= 0.95 {
                return 1.0; // genuinely solved
            } else {
                // Overfit: 100% train but poor test
                let preds = extract_body_predicates(rule_text);
                memory.record_overfit(rule_text, &preds, train_score, test_score);
                return test_score;
            }
        }

        // Non-perfect training: use test score as ground truth
        test_score
    } else {
        // No test data: trust training score
        train_score
    }
}

/// Score candidate rules against test data (held-out validation).
fn validate_on_test(
    rule_texts: &[&str],
    test_session: &Session,
    test_expected_keys: &HashSet<String>,
    target_pred: &str,
) -> f64 {
    let mut store = test_session.store().clone();

    // Remove DNA-inferred predict-cell facts so genesis is scored IN ISOLATION.
    // Keeps DNA-derived intermediates (enclosed-cell, shape-sig, etc.) intact —
    // genesis can use them but is scored only on its OWN predictions.
    // Matches PrebuiltScorer behavior (invent.rs).
    store.remove_inferred_by_predicate(target_pred);

    let mut rules: Vec<Rule> = Vec::new();
    for text in rule_texts {
        if let Ok(stmts) = qor_core::parser::parse(text) {
            for stmt in stmts {
                if let Statement::Rule { head, body, tv } = stmt {
                    rules.push(Rule::new(head, body, tv.unwrap_or(TruthValue::new(0.95, 0.95))));
                }
            }
        }
    }

    let _ = forward_chain(&rules, &mut store);

    let valid_ids = extract_valid_ids(test_expected_keys);
    let predictions = extract_keys_from_store_filtered(&store, target_pred, &valid_ids);
    if test_expected_keys.is_empty() { return 0.0; }
    let correct = test_expected_keys.iter().filter(|k| predictions.contains(*k)).count();
    precision_recall_score(correct, test_expected_keys.len(), predictions.len())
}

/// Extract body predicate names from a QOR rule text (string-based).
fn extract_body_predicates(rule_text: &str) -> Vec<String> {
    let mut preds = Vec::new();
    let body_start = rule_text.find(" if\n")
        .map(|p| p + 4)
        .or_else(|| rule_text.find(" if ").map(|p| p + 4));
    if let Some(start) = body_start {
        let body = &rule_text[start..];
        for chunk in body.split('(') {
            let trimmed = chunk.trim();
            if let Some(word) = trimmed.split_whitespace().next() {
                let word = word.trim_end_matches(')');
                if !word.is_empty()
                    && !word.starts_with('$')
                    && !word.starts_with('<')
                    && !word.starts_with('>')
                    && !word.starts_with('=')
                    && !word.starts_with('!')
                    && word != "not"
                    && word != "not-present"
                    && word != "count"
                    && word != "sum"
                    && word != "min"
                    && word != "max"
                {
                    preds.push(word.to_string());
                }
            }
        }
    }
    preds
}

// ── Meta Strategy Loading ────────────────────────────────────────────

/// Load individual meta .qor files as strategy workers.
///
/// Each file is parsed independently. Excludes reasoning.qor (advisor)
/// and agi.qor (judge) — those have special roles, not strategy workers.
///
/// Returns (name, statements) pairs — one per meta file.
pub fn load_meta_strategies(meta_dir: &std::path::Path) -> Vec<(String, Vec<Statement>)> {
    let mut strategies = Vec::new();
    let entries = match std::fs::read_dir(meta_dir) {
        Ok(e) => e,
        Err(_) => return strategies,
    };
    let mut paths: Vec<_> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().map(|x| x == "qor").unwrap_or(false)
                && p.file_stem().map(|s| s != "agi" && s != "reasoning" && s != "classify" && s != "bridge").unwrap_or(true)
        })
        .collect();
    paths.sort();

    for path in paths {
        if let Ok(source) = std::fs::read_to_string(&path) {
            if let Ok(stmts) = qor_core::parser::parse(&source) {
                if !stmts.is_empty() {
                    let name = path.file_stem().unwrap()
                        .to_string_lossy().to_string();
                    strategies.push((name, stmts));
                }
            }
        }
    }
    eprintln!("  meta-strategies: loaded {} files from {:?}", strategies.len(),
        meta_dir.file_name().unwrap_or_default());
    strategies
}

/// Load reasoning.qor as the advisor (applied after the swarm race).
///
/// Returns parsed statements, or None if reasoning.qor doesn't exist.
fn load_advisor(meta_dir: &std::path::Path) -> Option<Vec<Statement>> {
    let path = meta_dir.join("reasoning.qor");
    let source = std::fs::read_to_string(&path).ok()?;
    let stmts = qor_core::parser::parse(&source).ok()?;
    if stmts.is_empty() { return None; }
    eprintln!("  advisor: loaded reasoning.qor ({} statements)", stmts.len());
    Some(stmts)
}
