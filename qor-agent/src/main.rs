//! QOR Agent — Generic
//! ====================
//! QOR DECIDES. Rust EXECUTES.
//!
//! Rust knows NOTHING about the domain. It just:
//!   1. Loads DNA (.qor files from a folder)
//!   2. Feeds data (any format via qor-bridge)
//!   3. Runs QOR (forward chain)
//!   4. Reads QOR's answer (any fact matching "answer-*" or configured predicate)
//!   5. Compares answer vs expected
//!   6. If wrong → runs search engine to find better rules
//!   7. Saves discovered rules for future items
//!
//! ALL domain logic lives in DNA. Zero hardcoded domain knowledge in Rust.
//!
//! Usage:
//!   qor-agent <dna-dir> <data-source> [--search-budget N]
//!   qor-agent <dna-dir> --browse <url> [--steps N]

mod act;
mod browser;
mod learn;
mod page;
mod perceive;
mod types;

use anyhow::Result;
use qor_core::neuron::{Neuron, QorValue, Statement};
use qor_core::parser;
use qor_runtime::eval::Session;
use qor_runtime::invent;
use qor_runtime::library::RuleLibrary;
use qor_runtime::search;
use std::path::PathBuf;
use types::AgentStats;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let dna_dir = PathBuf::from(
        args.get(1).map(|s| s.as_str())
            .unwrap_or(r"D:\QOR-LANG\qor\dna\puzzle_solver")
    );

    // ── Check for --browse mode ──
    let browse_url = args.iter()
        .position(|a| a == "--browse")
        .and_then(|i| args.get(i + 1))
        .cloned();

    let max_steps: usize = args.iter()
        .position(|a| a == "--steps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    if let Some(url) = browse_url {
        return browse_mode(&dna_dir, &url, max_steps).await;
    }

    // ── Batch mode ──
    let data_source = args.get(2).map(|s| s.as_str())
        .unwrap_or(".");

    let search_budget_ms: u64 = args.iter()
        .position(|a| a == "--search-budget")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);

    let max_attempts: usize = args.iter()
        .position(|a| a == "--max-attempts")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let memory_path = dna_dir.join("memory.qor");
    let rules_path = dna_dir.join("rules_learned.qor");

    println!("{}", "=".repeat(70));
    println!("  QOR Agent — QOR Decides, Rust Executes");
    println!("{}", "=".repeat(70));
    println!("  DNA:     {}", dna_dir.display());
    println!("  Data:    {data_source}");
    println!("  Search:  {search_budget_ms}ms per failed item");
    println!("  Attempts: {max_attempts} max per item");

    // Worker count for parallel genesis
    let num_workers = invent::optimal_worker_count();
    println!("  Workers: {num_workers} parallel genesis threads");

    // Show memory stats
    let (mem_solved, mem_failed, mem_facts) = learn::memory_stats(&memory_path);
    if mem_facts > 0 {
        println!("  Memory:  {} solved, {} failed, {} facts",
            mem_solved, mem_failed, mem_facts);
    }
    println!("{}", "=".repeat(70));

    let items = perceive::load_items(data_source)?;
    println!("  Loaded {} items", items.len());

    if items.is_empty() {
        println!("  Nothing to process.");
        return Ok(());
    }

    let mut stats = AgentStats::default();
    // Lazy browser — only launched when QOR first requests web lookup
    let mut lazy_browser: Option<browser::Browser> = None;

    // Living library — remembers successful rules across items
    let library_dir = dna_dir.join("library");
    let mut library = RuleLibrary::load(&library_dir);
    if !library.all_rules().is_empty() {
        println!("  Library: {} remembered rules", library.all_rules().len());
    }

    // Load already-solved IDs to skip them
    let solved_ids = learn::load_solved_ids(&memory_path);
    if !solved_ids.is_empty() {
        println!("  Already solved: {} items (will skip)", solved_ids.len());
    }

    for (idx, item) in items.iter().enumerate() {
        println!("\n{}", "-".repeat(70));
        println!("  [{}/{}] {}", idx + 1, items.len(), item.id);

        // Skip already-solved items
        if solved_ids.contains(&item.id) {
            println!("  [SKIP] Already solved");
            stats.correct += 1;
            stats.attempted += 1;
            continue;
        }

        let mut solved = false;
        let mut last_score: f64 = 0.0;
        stats.attempted += 1;

        // ── Multi-attempt loop: confidence drives strategy ──
        for attempt in 1..=max_attempts {
            if attempt > 1 {
                println!("  ── Attempt {attempt}/{max_attempts} (confidence: {:.1}%) ──",
                    last_score * 100.0);
            }

            let mut session = load_dna(&dna_dir)?;

            // Load structured memory (solved, features, failures, best-partials)
            if memory_path.exists() {
                if let Ok(src) = std::fs::read_to_string(&memory_path) {
                    let _ = session.exec(&src);
                }
            }

            // Load previously discovered rules
            if rules_path.exists() {
                if let Ok(src) = std::fs::read_to_string(&rules_path) {
                    let _ = session.exec(&src);
                }
            }

            // Feed the data into QOR
            let _ = session.exec(&format!("(item-id {})", item.id));
            if let Err(e) = session.exec_statements(item.facts.clone()) {
                eprintln!("  [FEED] warning: {e}");
            }

            // Inject attempt tracking facts — DNA reads these
            let _ = session.exec(&format!("(attempt-number {} {})", item.id, attempt));
            if attempt > 1 {
                let _ = session.exec(&format!(
                    "(solve-confidence {} {:.4}) <{:.2}, 0.95>",
                    item.id, last_score, last_score
                ));
                for prev in 1..attempt {
                    let _ = session.exec(&format!("(attempt-failed {} {})", item.id, prev));
                }
            }

            println!("  [FACTS] {} facts, {} rules",
                session.fact_count(), session.rule_count());

            // ── Check if QOR wants to browse (DNA: confidence < 0.50 → web-lookup-needed) ──
            let web_activated = execute_actions(
                &mut session, &mut lazy_browser,
            ).await;
            if web_activated {
                println!("  [WEB] Web lookup completed — re-reasoning...");
            }

            // Read QOR's answer
            let answer_facts = collect_answer_facts(&session);

            // Compare with expected
            match (&answer_facts, &item.expected) {
                (ans, Some(exp)) if !ans.is_empty() => {
                    let score = score_answer(ans, exp);

                    if score >= 0.999 {
                        println!("  [SOLVED] CORRECT (attempt {attempt})");
                        stats.correct += 1;

                        // Extract features + winning strategy from session
                        let features = learn::extract_features(&session);
                        let strategy = learn::extract_winning_strategy(&session);

                        // Record solved with structured data (replaces any failure section)
                        let saved = learn::record_solved(
                            &item.id, &strategy, &features, &memory_path,
                        )?;
                        println!("  [MEMORY] Solved via {strategy} ({saved} facts)");
                        stats.learned += saved;

                        solved = true;
                        break;
                    }

                    println!("  [WRONG] {:.1}% — attempt {attempt}", score * 100.0);
                    last_score = score;

                    // Extract features + tried strategies from session
                    let features = learn::extract_features(&session);
                    let tried_strategies = learn::extract_tried(&session);

                    // Search for better rules — parallel genesis swarm
                    let target = find_dna_predicate(&session, "target-predicate", "predict-cell");
                    let source = find_dna_predicate(&session, "source-predicate", "grid-cell");
                    let mut best_source = "dna-rules".to_string();
                    let mut best_score = score;

                    // Try genesis swarm first (parallel invention)
                    let genesis_result = try_genesis(
                        &session, &item.facts, search_budget_ms,
                        &target, &source, &mut library, num_workers,
                    );
                    // Fall back to refinement search if genesis had no data
                    let search_result = if genesis_result.is_none() {
                        try_search(
                            &session, &item.facts, search_budget_ms, &target, &source,
                        )
                    } else {
                        None
                    };

                    // Process genesis results
                    if let Some((candidates, elapsed)) = genesis_result {
                        let top = candidates.first();
                        let top_score = top.map(|c| c.score).unwrap_or(0.0);
                        let count = candidates.len();

                        if top_score >= 0.95 {
                            println!("  [GENESIS] SOLVED! {count} candidate(s), {elapsed}ms, {num_workers} workers");
                            let rules: Vec<String> = candidates.iter()
                                .filter(|c| c.score >= 0.95)
                                .map(|c| c.rule_text.clone())
                                .collect();
                            let saved = learn::save_winning_rules(
                                &rules, &item.id, &rules_path,
                            )?;
                            stats.learned += saved;
                            best_source = format!("genesis-{}", top.map(|c| c.source.as_str()).unwrap_or("unknown"));
                            best_score = top_score;
                        } else if top_score > score + 0.05 {
                            println!("  [GENESIS] Best {:.1}% ({count} candidates, {elapsed}ms)",
                                top_score * 100.0);
                            let rules: Vec<String> = candidates.iter()
                                .filter(|c| c.score > score)
                                .map(|c| c.rule_text.clone())
                                .collect();
                            let saved = learn::save_winning_rules(
                                &rules, &item.id, &rules_path,
                            )?;
                            if saved > 0 { stats.learned += saved; }
                            if top_score > best_score {
                                best_score = top_score;
                                best_source = format!("genesis-{}", top.map(|c| c.source.as_str()).unwrap_or("unknown"));
                            }
                        } else {
                            println!("  [GENESIS] No improvement ({count} candidates, {elapsed}ms)");
                        }
                    }

                    // Process refinement search results (fallback)
                    if let Some((solutions, near_misses, tried, elapsed)) = search_result {
                        if !solutions.is_empty() {
                            println!("  [SEARCH] Found {} solution(s) in {elapsed}ms ({tried} tried)",
                                solutions.len());
                            let saved = learn::save_winning_rules(
                                &solutions, &item.id, &rules_path,
                            )?;
                            stats.learned += saved;
                            best_source = "search-solved".to_string();
                            best_score = 1.0;
                        } else if !near_misses.is_empty() {
                            let top = &near_misses[0];
                            println!("  [SEARCH] Best {:.1}% ({tried} tried, {elapsed}ms)",
                                top.score * 100.0);
                            if top.score > score + 0.05 {
                                let rules: Vec<String> = near_misses.iter()
                                    .map(|nm| nm.qor_text.clone()).collect();
                                let saved = learn::save_winning_rules(
                                    &rules, &item.id, &rules_path,
                                )?;
                                if saved > 0 { stats.learned += saved; }
                            }
                            if top.score > best_score {
                                best_score = top.score;
                                best_source = "search-near-miss".to_string();
                            }
                        } else {
                            println!("  [SEARCH] No improvement ({tried} tried, {elapsed}ms)");
                        }
                    }

                    // Record structured failure data
                    let saved = learn::record_attempt(
                        &item.id, attempt, best_score, &best_source,
                        &features, &tried_strategies, &memory_path,
                    )?;
                    println!("  [MEMORY] {saved} facts (features: {}, tried: {})",
                        features.len(), tried_strategies.len());
                }
                (ans, None) if !ans.is_empty() => {
                    println!("  [DONE] {} answer facts (no expected)", ans.len());
                    solved = true;
                    break;
                }
                _ => {
                    println!("  [EMPTY] No answer — attempt {attempt}");
                    stats.no_answer += 1;
                    break; // no point retrying if no answer at all
                }
            }
        }

        if !solved && item.expected.is_some() {
            stats.wrong += 1;
        }

        println!("  [STATS] {stats}");
    }

    // Save library for future runs
    library.save();
    if !library.all_rules().is_empty() {
        println!("  Library saved: {} rules", library.all_rules().len());
    }

    println!("\n{}", "=".repeat(70));
    println!("  FINAL: {stats}");
    if stats.attempted > 0 {
        println!("  Accuracy: {:.1}%",
            stats.correct as f64 / stats.attempted as f64 * 100.0);
    }
    println!("{}", "=".repeat(70));

    Ok(())
}

/// Load all .qor files from a DNA directory into a fresh session.
fn load_dna(dna_dir: &PathBuf) -> Result<Session> {
    let mut session = Session::new();
    let mut file_count = 0;
    let mut stmt_count = 0;

    if let Ok(entries) = std::fs::read_dir(dna_dir) {
        let mut paths: Vec<_> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "qor").unwrap_or(false))
            .collect();
        paths.sort();

        for path in &paths {
            if let Ok(source) = std::fs::read_to_string(path) {
                match parser::parse(&source) {
                    Ok(stmts) => {
                        let n = stmts.len();
                        if let Err(e) = session.exec_statements(stmts) {
                            eprintln!("  warn: {}: {e}",
                                path.file_name().unwrap_or_default().to_string_lossy());
                        }
                        stmt_count += n;
                        file_count += 1;
                    }
                    Err(e) => {
                        eprintln!("  warn: {}: {e}",
                            path.file_name().unwrap_or_default().to_string_lossy());
                    }
                }
            }
        }
    }

    eprintln!("  [DNA] {file_count} files, {stmt_count} statements, {} rules",
        session.rule_count());
    Ok(session)
}

/// Collect answer facts from QOR session.
fn collect_answer_facts(session: &Session) -> Vec<(String, Vec<String>)> {
    // Check if DNA specified answer predicates
    let mut answer_preds: Vec<String> = Vec::new();
    for fact in session.all_facts() {
        if let Neuron::Expression(parts) = &fact.neuron {
            if parts.len() >= 2 {
                if let Neuron::Symbol(pred) = &parts[0] {
                    if pred == "answer-predicate" {
                        if let Neuron::Symbol(ap) = &parts[1] {
                            answer_preds.push(ap.clone());
                        }
                    }
                }
            }
        }
    }

    if answer_preds.is_empty() {
        // Default: look for any "answer" or "predict" prefixed predicates
        answer_preds.push("answer".to_string());
    }

    let mut results = Vec::new();
    for fact in session.all_facts() {
        if let Neuron::Expression(parts) = &fact.neuron {
            if let Some(Neuron::Symbol(pred)) = parts.first() {
                if answer_preds.iter().any(|ap| pred == ap || pred.starts_with("answer-")) {
                    let args: Vec<String> = parts.iter().map(|p| p.to_string()).collect();
                    results.push((pred.clone(), args));
                }
            }
        }
    }
    results
}

/// Score answer facts against expected facts.
///
/// Handles `expected-*` prefix: strips it to find the answer predicate name.
/// Handles arity mismatch: if answer has extra prefix args (e.g. grid ID),
/// compares the suffix of answer values against expected values.
fn score_answer(
    answer: &[(String, Vec<String>)],
    expected: &[(String, Vec<String>)],
) -> f64 {
    if expected.is_empty() { return 0.0; }

    let mut matched = 0;
    for (exp_pred, exp_args) in expected {
        let ans_pred = exp_pred.strip_prefix("expected-").unwrap_or(exp_pred);
        let exp_vals: Vec<&String> = exp_args.iter().skip(1).collect();

        let found = answer.iter().any(|(p, a)| {
            if p != ans_pred { return false; }
            let ans_vals: Vec<&String> = a.iter().skip(1).collect();

            if ans_vals.len() == exp_vals.len() {
                ans_vals == exp_vals
            } else if ans_vals.len() > exp_vals.len() {
                let offset = ans_vals.len() - exp_vals.len();
                ans_vals[offset..] == exp_vals[..]
            } else {
                false
            }
        });

        if found { matched += 1; }
    }

    matched as f64 / expected.len() as f64
}

// ── Adaptive search: find better rules when current ones fail ──────────

/// Extract training pair data from item facts for the search engine.
/// Returns: (training_inputs, expected_outputs, target_pred) or None.
///
/// Reads structural markers (train-pair) from QOR facts.
/// Output facts are converted to target-predicate Statements.
fn extract_training_data(
    facts: &[Statement],
    target_pred: &str,
    source_pred: &str,
) -> Option<(Vec<Vec<Statement>>, Vec<Vec<Statement>>)> {
    // Find train-pair facts: (train-pair input_id output_id)
    let mut pairs: Vec<(String, String)> = Vec::new();
    for fact in facts {
        if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = fact {
            if parts.len() >= 3 {
                if let (Neuron::Symbol(pred), Neuron::Symbol(in_id), Neuron::Symbol(out_id)) =
                    (&parts[0], &parts[1], &parts[2])
                {
                    if pred == "train-pair" {
                        pairs.push((in_id.clone(), out_id.clone()));
                    }
                }
            }
        }
    }

    if pairs.is_empty() {
        return None;
    }

    let mut training_inputs = Vec::new();
    let mut expected_outputs = Vec::new();

    for (in_id, out_id) in &pairs {
        let mut input_stmts = Vec::new();
        let mut output_stmts = Vec::new();

        for fact in facts {
            if let Statement::Fact { neuron: Neuron::Expression(parts), tv, decay } = fact {
                if parts.len() >= 2 {
                    if let Neuron::Symbol(fact_id) = &parts[1] {
                        // Input facts: rename ID to "ti" so rules fire
                        if fact_id == in_id {
                            let mut new_parts = parts.clone();
                            new_parts[1] = Neuron::symbol("ti");
                            input_stmts.push(Statement::Fact {
                                neuron: Neuron::Expression(new_parts),
                                tv: *tv,
                                decay: *decay,
                            });
                        }

                        // Output facts: convert source-pred facts to target-pred Statements
                        if fact_id == out_id {
                            if let Some(Neuron::Symbol(pred)) = parts.first() {
                                if pred == source_pred {
                                    let mut target_parts = parts.clone();
                                    target_parts[0] = Neuron::symbol(target_pred);
                                    target_parts[1] = Neuron::symbol("ti");
                                    output_stmts.push(Statement::Fact {
                                        neuron: Neuron::Expression(target_parts),
                                        tv: *tv,
                                        decay: *decay,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        // Add (test-input ti) so rules that check test-input fire
        input_stmts.push(Statement::Fact {
            neuron: Neuron::expression(vec![
                Neuron::symbol("test-input"),
                Neuron::symbol("ti"),
            ]),
            tv: Some(qor_core::truth_value::TruthValue::new(0.99, 0.99)),
            decay: None,
        });

        training_inputs.push(input_stmts);
        expected_outputs.push(output_stmts);
    }

    Some((training_inputs, expected_outputs))
}

/// Run the search engine to find better rules.
/// Returns: Option<(solution_texts, near_miss_scored, mutations_tried, elapsed_ms)>
fn try_search(
    session: &Session,
    item_facts: &[Statement],
    budget_ms: u64,
    target_pred: &str,
    source_pred: &str,
) -> Option<(Vec<String>, Vec<search::ScoredRule>, usize, u64)> {
    let (training_inputs, expected_outputs) =
        extract_training_data(item_facts, target_pred, source_pred)?;

    if training_inputs.is_empty() || expected_outputs.is_empty() {
        return None;
    }

    // Extract existing rules that produce the target predicate as seeds
    let seed_rules: Vec<_> = session.rules()
        .iter()
        .filter(|r| {
            if let Neuron::Expression(parts) = &r.head {
                parts.first()
                    .map(|p| p.to_string().contains(target_pred))
                    .unwrap_or(false)
            } else {
                false
            }
        })
        .cloned()
        .collect();

    if seed_rules.is_empty() {
        return None;
    }

    // Build base session: DNA + derived observation facts
    let mut base = Session::new();
    for stmt in session.rules_as_statements() {
        let _ = base.exec_statements(vec![stmt]);
    }
    // Add derived observation facts (any obs-* or detected-* facts from DNA)
    for fact in session.all_facts() {
        if let Neuron::Expression(parts) = &fact.neuron {
            if let Some(Neuron::Symbol(pred)) = parts.first() {
                if pred.starts_with("obs-") || pred.starts_with("detected-")
                    || pred.starts_with("consistent-")
                {
                    let stmt = Statement::Fact {
                        neuron: fact.neuron.clone(),
                        tv: Some(fact.tv),
                        decay: None,
                    };
                    let _ = base.exec_statements(vec![stmt]);
                }
            }
        }
    }

    let result = search::refinement_search(
        &seed_rules,
        &training_inputs,
        &expected_outputs,
        target_pred,
        &base,
        budget_ms,
    );

    let solution_texts: Vec<String> = result.solutions.iter()
        .map(|s| s.qor_text.clone())
        .collect();

    Some((solution_texts, result.near_misses, result.mutations_tried, result.elapsed_ms))
}

/// Run parallel genesis swarm to invent rules from scratch.
/// Returns: Option<(candidates, elapsed_ms)>
fn try_genesis(
    session: &Session,
    item_facts: &[Statement],
    budget_ms: u64,
    target_pred: &str,
    source_pred: &str,
    library: &mut RuleLibrary,
    num_workers: usize,
) -> Option<(Vec<invent::Candidate>, u64)> {
    let (training_inputs, expected_outputs) =
        extract_training_data(item_facts, target_pred, source_pred)?;

    if training_inputs.is_empty() || expected_outputs.is_empty() {
        return None;
    }

    let start = std::time::Instant::now();

    // Build CLEAN base session: DNA rules + observation facts ONLY.
    // Must NOT include puzzle grid-cell/predict-cell data — genesis uses
    // ID "ti" for training pairs, which would collide with test input facts.
    let mut base = Session::new();
    for stmt in session.rules_as_statements() {
        let _ = base.exec_statements(vec![stmt]);
    }
    // Add derived observation facts (DNA-derived, domain-agnostic)
    for fact in session.all_facts() {
        if let Neuron::Expression(parts) = &fact.neuron {
            if let Some(Neuron::Symbol(pred)) = parts.first() {
                if pred.starts_with("obs-") || pred.starts_with("detected-")
                    || pred.starts_with("consistent-") || pred.starts_with("dna-")
                    || pred == "target-predicate" || pred == "source-predicate"
                    || pred == "answer-predicate"
                {
                    let stmt = Statement::Fact {
                        neuron: fact.neuron.clone(),
                        tv: Some(fact.tv),
                        decay: None,
                    };
                    let _ = base.exec_statements(vec![stmt]);
                }
            }
        }
    }

    let candidates = invent::genesis_swarm(
        &base,
        &training_inputs,
        &expected_outputs,
        budget_ms,
        Some(library),
        num_workers,
        None,
    );

    let elapsed = start.elapsed().as_millis() as u64;

    if candidates.is_empty() {
        return None;
    }

    Some((candidates, elapsed))
}

/// Read a predicate name from DNA facts, with a default fallback.
/// E.g., if DNA has `(target-predicate predict-cell)`, returns "predict-cell".
fn find_dna_predicate(session: &Session, dna_pred: &str, default: &str) -> String {
    for fact in session.all_facts() {
        if let Neuron::Expression(parts) = &fact.neuron {
            if parts.len() >= 2 {
                if let (Neuron::Symbol(pred), Neuron::Symbol(val)) = (&parts[0], &parts[1]) {
                    if pred == dna_pred {
                        return val.clone();
                    }
                }
            }
        }
    }
    default.to_string()
}

// ═══════════════════════════════════════════════════════════════════════
// ACTION EXECUTOR — Runs whatever QOR decided (browser or HTTP)
// ═══════════════════════════════════════════════════════════════════════

/// Execute any actions QOR produced. Pure plumbing — zero domain logic.
/// Reads (browser-action ...) and (http-fetch ...) facts. Executes them.
/// DNA decides WHAT to do. Rust just does it.
/// Returns true if any actions were executed.
async fn execute_actions(
    session: &mut Session,
    lazy_browser: &mut Option<browser::Browser>,
) -> bool {
    let mut executed = false;

    // ── Execute browser actions (DNA decides what/where/how) ──
    let mut browser_steps = 0;
    while let Some(action) = act::next_action(session) {
        if matches!(action, act::AgentAction::Done) { break; }
        if browser_steps >= 20 { break; } // safety valve only

        // Lazy-launch browser on first use
        let b = if let Some(b) = lazy_browser.as_mut() {
            Some(b)
        } else {
            match browser::Browser::launch().await {
                Ok(b) => { *lazy_browser = Some(b); lazy_browser.as_mut() }
                Err(e) => {
                    eprintln!("  [ACT] Browser unavailable: {e}");
                    let _ = session.exec("(browser-unavailable true)");
                    None
                }
            }
        };

        if let Some(b) = b {
            match act::execute(&action, b).await {
                Ok(facts) => {
                    // Clear stale page facts before injecting new ones
                    session.remove_by_predicate(&[
                        "page-element", "page-content",
                        "page-link-count", "page-button-count", "page-form-count",
                        "page-text", "page-url", "page-title",
                        "action-result", "network-request", "js-result",
                    ]);
                    let _ = session.exec_statements(facts);
                    executed = true;
                }
                Err(e) => {
                    eprintln!("  [ACT] Failed: {e}");
                    let _ = session.exec(&format!(
                        "(action-error \"{}\")", e.replace('"', "'")
                    ));
                }
            }
        } else {
            break;
        }
        browser_steps += 1;
    }

    // ── Execute HTTP fetch requests (DNA decides the URLs) ──
    // Read (http-fetch $url) facts — generic plumbing
    let fetch_urls: Vec<String> = session.all_facts().iter().filter_map(|fact| {
        if let Neuron::Expression(parts) = &fact.neuron {
            if parts.len() >= 2 {
                if let Neuron::Symbol(pred) = &parts[0] {
                    if pred == "http-fetch" {
                        return match &parts[1] {
                            Neuron::Value(QorValue::Str(url)) => Some(url.clone()),
                            Neuron::Symbol(url) => Some(url.clone()),
                            _ => None,
                        };
                    }
                }
            }
        }
        None
    }).collect();

    for url in &fetch_urls {
        println!("  [HTTP] Fetching {url}");
        match ureq::get(url).call() {
            Ok(resp) => {
                if let Ok(body) = resp.into_string() {
                    match qor_bridge::feed(&body) {
                        Ok(facts) => {
                            let n = facts.len();
                            let _ = session.exec_statements(facts);
                            println!("  [HTTP] Got {n} facts");
                            executed = true;
                        }
                        Err(e) => eprintln!("  [HTTP] Parse error: {e}"),
                    }
                }
            }
            Err(e) => {
                eprintln!("  [HTTP] Failed: {e}");
                let _ = session.exec(&format!(
                    "(action-error \"{}\")", e.to_string().replace('"', "'")
                ));
            }
        }
    }

    // Clear fetch requests so they don't re-fire
    if !fetch_urls.is_empty() {
        session.remove_by_predicate(&["http-fetch"]);
    }

    executed
}

// ═══════════════════════════════════════════════════════════════════════
// BROWSE MODE — Web agent driven by QOR DNA
// ═══════════════════════════════════════════════════════════════════════

/// Browse mode: DNA decides what to click, fill, navigate. Rust just executes.
async fn browse_mode(dna_dir: &PathBuf, start_url: &str, max_steps: usize) -> Result<()> {
    println!("{}", "=".repeat(70));
    println!("  QOR Agent — Browse Mode");
    println!("{}", "=".repeat(70));
    println!("  DNA:     {}", dna_dir.display());
    println!("  URL:     {start_url}");
    println!("  Steps:   {max_steps}");
    println!("{}", "=".repeat(70));

    // Load DNA into QOR session
    let mut session = load_dna(dna_dir)?;

    // Inject start URL as a QOR fact
    let _ = session.exec(&format!("(start-url \"{}\")", start_url));

    // Try launching Chrome for full browser mode
    let probe = browser::Browser::probe();
    if !probe.chrome_found {
        println!("  Chrome not available — falling back to HTTP-only mode");
        return http_fallback(&mut session, start_url, max_steps).await;
    }

    println!("  Chrome found — launching headless browser...");
    let mut browser = match browser::Browser::launch().await {
        Ok(b) => {
            println!("  Browser connected via CDP");
            b
        }
        Err(e) => {
            eprintln!("  Browser launch failed: {e}");
            println!("  Falling back to HTTP-only mode");
            return http_fallback(&mut session, start_url, max_steps).await;
        }
    };

    // ── Main browse loop: perceive → reason → act ──
    for step in 1..=max_steps {
        println!("\n{}", "-".repeat(50));
        println!("  Step {step}/{max_steps}");

        // Ask QOR what to do next (DNA rules fire on current facts)
        let action = match act::next_action(&session) {
            Some(a) => a,
            None => {
                println!("  No action — DNA produced no browser-action facts");
                break;
            }
        };

        println!("  Action: {:?}", action);

        // Check for done
        if matches!(action, act::AgentAction::Done) {
            println!("  Done — agent finished");
            break;
        }

        // Execute the action on the browser
        match act::execute(&action, &mut browser).await {
            Ok(result_facts) => {
                println!("  Result: {} facts", result_facts.len());

                // Clear previous page-* facts to avoid stale data
                session.remove_by_predicate(&[
                    "page-element", "page-content",
                    "page-link-count", "page-button-count", "page-form-count",
                    "page-text", "page-url", "page-title",
                    "action-result", "network-request", "js-result",
                ]);

                // Feed new facts into QOR and forward-chain
                if let Err(e) = session.exec_statements(result_facts) {
                    eprintln!("  [FEED] warning: {e}");
                }

                println!("  [QOR] {} facts, {} rules",
                    session.fact_count(), session.rule_count());
            }
            Err(e) => {
                eprintln!("  Action failed: {e}");
                // Inject error fact so DNA can decide what to do
                let _ = session.exec(&format!("(action-error \"{}\")",
                    e.replace('"', "'")));
            }
        }
    }

    // Print any findings/answers QOR produced
    println!("\n{}", "=".repeat(70));
    println!("  Browse complete — checking results...");
    let answer_facts = collect_answer_facts(&session);
    if answer_facts.is_empty() {
        println!("  No answer facts produced");
    } else {
        println!("  {} answer facts:", answer_facts.len());
        for (pred, args) in &answer_facts {
            println!("    ({} {})", pred, args[1..].join(" "));
        }
    }

    // Print any findings
    for fact in session.all_facts() {
        if let Neuron::Expression(parts) = &fact.neuron {
            if let Some(Neuron::Symbol(pred)) = parts.first() {
                if pred == "finding" || pred == "result" || pred == "extracted" {
                    let args: Vec<String> = parts.iter().skip(1).map(|p| p.to_string()).collect();
                    println!("  [{pred}] {}", args.join(" "));
                }
            }
        }
    }

    println!("{}", "=".repeat(70));
    Ok(())
}

/// HTTP-only fallback when Chrome is not available.
/// Fetches URL content via ureq, converts to QOR facts, reasons, reports.
async fn http_fallback(session: &mut Session, url: &str, max_steps: usize) -> Result<()> {
    println!("  [HTTP] Fetching {url}...");

    let resp = ureq::get(url).call()
        .map_err(|e| anyhow::anyhow!("HTTP fetch failed: {e}"))?;
    let body = resp.into_string()
        .map_err(|e| anyhow::anyhow!("Read error: {e}"))?;

    // Use qor-bridge to convert the body into QOR facts
    let facts = qor_bridge::feed(&body)
        .map_err(|e| anyhow::anyhow!("Bridge error: {e}"))?;

    println!("  [HTTP] {} facts from page content", facts.len());

    // Also inject page-url and page-text facts
    let _ = session.exec(&format!("(page-url \"{}\")", url));
    // Truncate body for QOR
    let truncated = if body.len() > 4000 { &body[..4000] } else { &body };
    let _ = session.exec(&format!("(page-text \"{}\")", truncated.replace('"', "'")));

    if let Err(e) = session.exec_statements(facts) {
        eprintln!("  [FEED] warning: {e}");
    }

    println!("  [QOR] {} facts, {} rules",
        session.fact_count(), session.rule_count());

    // Run reasoning steps (no browser actions available, just forward chaining)
    for step in 1..=max_steps {
        let action = act::next_action(session);
        match action {
            Some(act::AgentAction::Done) => {
                println!("  Step {step}: Done");
                break;
            }
            Some(act::AgentAction::Navigate(new_url)) => {
                println!("  Step {step}: Navigate (HTTP) → {new_url}");
                match ureq::get(&new_url).call() {
                    Ok(resp) => {
                        if let Ok(body) = resp.into_string() {
                            let _ = session.exec(&format!("(page-url \"{}\")", new_url));
                            let t = if body.len() > 4000 { &body[..4000] } else { &body };
                            let _ = session.exec(&format!("(page-text \"{}\")", t.replace('"', "'")));
                        }
                    }
                    Err(e) => {
                        let _ = session.exec(&format!("(action-error \"{}\")",
                            e.to_string().replace('"', "'")));
                    }
                }
            }
            Some(other) => {
                println!("  Step {step}: {:?} (not available in HTTP mode)", other);
                let _ = session.exec("(action-error \"browser not available\")");
            }
            None => {
                println!("  Step {step}: No action");
                break;
            }
        }
    }

    // Print results
    println!("\n{}", "=".repeat(70));
    let answer_facts = collect_answer_facts(session);
    if !answer_facts.is_empty() {
        println!("  {} answer facts:", answer_facts.len());
        for (pred, args) in &answer_facts {
            println!("    ({} {})", pred, args[1..].join(" "));
        }
    }
    println!("{}", "=".repeat(70));

    Ok(())
}
