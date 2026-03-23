// ── Rule Invention Engine (Genesis) ─────────────────────────────────
//
// Creates QOR rules from scratch when no existing rules produce predictions.
//
// Implements the Creativity Formula (agi.md):
//   1. DECOMPOSE  — Profile fact shapes (predicates, arities, value types)
//   2. CONTRADICT  — Compare input vs output fact sets (what changed?)
//   3. PROPOSE     — Generate candidate rules from discovered patterns
//   4. EVALUATE    — Score each candidate against training data
//   5. COMBINE     — Merge partial rules for better coverage
//   6. EVOLVE      — Mutate best candidates (SCAMPER operators), repeat
//
// This engine is DOMAIN AGNOSTIC. It knows NOTHING about grids, colors,
// puzzles, music, chemistry, or any specific domain. It works with raw
// Statement facts. Domain knowledge comes from .qor DNA files loaded
// into the base_session BEFORE genesis runs.

use crate::chain::Rule;
use crate::egraph;
use crate::eval::Session;
use crate::library::RuleLibrary;
use crate::mutate::{generate_mutations, generate_context_mutations, rule_to_qor};
use crate::search;
use qor_core::neuron::{Neuron, QorValue, Statement};
use qor_core::parser;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A candidate rule with its fitness score.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub rule_text: String,
    pub score: f64,
    pub source: String,
}

impl crate::memory::HasRuleText for Candidate {
    fn rule_text(&self) -> &str { &self.rule_text }
}

/// Profile of ANY fact set — no domain knowledge needed.
struct FactProfile {
    predicates: HashMap<String, Vec<ArgInfo>>,
    constants: HashSet<i64>,
    symbols: HashSet<String>,
}

/// Info about one argument position of a predicate.
struct ArgInfo {
    ints: HashSet<i64>,
    syms: HashSet<String>,
    is_id: bool,
    is_sequential: bool,
    is_categorical: bool,
}

/// General contradiction between ANY two fact sets.
struct GeneralContradiction {
    added: Vec<Vec<Neuron>>,
    removed: Vec<Vec<Neuron>>,
    value_maps: HashMap<(String, usize), HashMap<i64, i64>>,
    new_predicates: HashSet<String>,
    count_changes: HashMap<String, (usize, usize)>,
}

/// Why a candidate scored poorly — guides next mutation.
#[derive(Debug)]
enum FailurePattern {
    AllWrong,
    WrongValues,
    PartialMatch(f64),
    MissingFacts,
    /// High recall but low precision — rule matches too broadly, needs constraining conditions.
    OverPredicting { recall: f64, precision: f64, extra_count: usize },
}

/// Detailed score with failure analysis.
struct ScoreDetail {
    score: f64,
    wrong_keys: Vec<(String, String)>, // (expected_key, got_key)
    missing_keys: Vec<String>,
    pattern: FailurePattern,
}

/// Derived fact parts from a forward-chained session.
struct StoredFact {
    parts: Vec<Neuron>,
}

/// Graph structure insights from petgraph analysis.
struct GraphInsights {
    num_components: usize,
    component_sizes: Vec<usize>,
    hub_predicates: Vec<String>,
}

/// Analyze graph structure of facts using petgraph.
/// Returns component count, sizes, and hub predicates (highest degree nodes).
fn analyze_graph_structure(session: &Session) -> GraphInsights {
    let store = session.store();
    let components = store.connected_components();
    let (graph, node_map) = store.to_graph();

    // Find hub nodes (highest degree) — these are likely important predicates/entities
    let mut degrees: Vec<(String, usize)> = node_map.iter()
        .map(|(name, &idx)| (name.clone(), graph.neighbors(idx).count()))
        .collect();
    degrees.sort_by(|a, b| b.1.cmp(&a.1));
    let hub_predicates: Vec<String> = degrees.into_iter()
        .take(10)
        .map(|(name, _)| name)
        .collect();

    let component_sizes: Vec<usize> = components.iter().map(|c| c.len()).collect();

    GraphInsights {
        num_components: components.len(),
        component_sizes,
        hub_predicates,
    }
}

/// Semantic dedup of candidates using e-graph equality saturation.
/// Catches reordered conditions and redundant guards that text dedup misses.
fn egraph_dedup_candidates(candidates: &mut Vec<Candidate>) {
    let before = candidates.len();
    if before <= 1 { return; }

    let mut scored: Vec<search::ScoredRule> = candidates.iter()
        .filter_map(|c| {
            qor_core::parser::parse(&c.rule_text).ok().and_then(|stmts| {
                stmts.into_iter().find_map(|s| {
                    if let Statement::Rule { head, body, tv } = s {
                        Some(search::ScoredRule {
                            rule: Rule::new(head, body, tv.unwrap_or(
                                qor_core::truth_value::TruthValue::new(0.9, 0.9))),
                            score: c.score,
                            qor_text: c.rule_text.clone(),
                        })
                    } else { None }
                })
            })
        })
        .collect();

    egraph::dedup_rules(&mut scored);

    if scored.len() < before {
        let deduped_texts: HashSet<String> = scored.iter().map(|s| s.qor_text.clone()).collect();
        candidates.retain(|c| deduped_texts.contains(&c.rule_text));
        eprintln!("      egraph-dedup: {} → {} candidates", before, candidates.len());
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════════════

/// Invent rules from scratch using training examples.
///
/// DOMAIN AGNOSTIC — works with any facts. Domain knowledge comes from
/// the DNA rules already loaded in `base_session`.
pub fn genesis(
    base_session: &Session,
    training_inputs: &[Vec<Statement>],
    training_outputs: &[Vec<Statement>],
    time_budget_ms: u64,
    library: Option<&mut RuleLibrary>,
    skip_rules: Option<&HashSet<String>>,
) -> Vec<Candidate> {
    let deadline = Instant::now() + Duration::from_millis(time_budget_ms);

    // ── STEP 0: RECALL — try rules from memory first ─────────────
    let mut candidates = Vec::new();
    if let Some(ref lib) = library {
        let mut past: Vec<_> = lib.all_rules().iter()
            .filter(|r| r.accuracy() > 0.1)
            .collect();
        past.sort_unstable_by(|a, b| b.accuracy().partial_cmp(&a.accuracy())
            .unwrap_or(std::cmp::Ordering::Equal));
        eprintln!("      genesis-S0-recall: {} library rules (>{:.0}% accuracy)", past.len(), 10.0);
        for scored in past.iter().take(10) {
            eprintln!("        recall: acc={:.1}% rule={}", scored.accuracy() * 100.0,
                scored.rule_text.chars().take(100).collect::<String>());
            candidates.push(Candidate {
                rule_text: scored.rule_text.clone(),
                score: 0.0,
                source: "memory-recall".into(),
            });
        }
    } else {
        eprintln!("      genesis-S0-recall: no library");
    }

    // ── STEP 1: DECOMPOSE — profile fact shapes ──────────────────
    let all_inputs: Vec<&Statement> = training_inputs.iter().flat_map(|v| v.iter()).collect();
    let all_outputs: Vec<&Statement> = training_outputs.iter().flat_map(|v| v.iter()).collect();
    let in_profile = profile_facts(&all_inputs);
    let out_profile = profile_facts(&all_outputs);
    let target_pred = find_target_predicate(&out_profile, &in_profile);
    eprintln!("      genesis-S1-decompose: {} input facts, {} output facts, target='{}'",
        all_inputs.len(), all_outputs.len(), target_pred);
    eprintln!("        input predicates: {:?}", in_profile.predicates.keys().take(8).collect::<Vec<_>>());
    eprintln!("        output predicates: {:?}", out_profile.predicates.keys().take(8).collect::<Vec<_>>());
    eprintln!("        input constants: {:?}", in_profile.constants.iter().take(10).collect::<Vec<_>>());

    // ── STEP 2: CONTRADICT — general fact-set comparison ─────────
    let contradictions: Vec<GeneralContradiction> = training_inputs.iter()
        .zip(training_outputs.iter())
        .map(|(inp, out)| find_general_contradiction(inp, out))
        .collect();
    for (ci, c) in contradictions.iter().enumerate() {
        eprintln!("      genesis-S2-contradict[pair{}]: {} added, {} removed, {} value-maps, {} new-preds",
            ci, c.added.len(), c.removed.len(), c.value_maps.len(), c.new_predicates.len());
        for ((pred, pos), vmap) in c.value_maps.iter().take(3) {
            let sample: Vec<String> = vmap.iter().take(5).map(|(k,v)| format!("{}→{}", k, v)).collect();
            eprintln!("        value-map {}.arg{}: [{}]", pred, pos, sample.join(", "));
        }
    }

    // Run DNA rules on each input to get derived facts
    let derived_facts: Vec<Vec<StoredFact>> = training_inputs.iter()
        .enumerate()
        .map(|(di, inp)| {
            let mut session = base_session.clone();
            let _ = session.exec_statements(inp.clone());
            let facts: Vec<StoredFact> = session.all_facts().iter().map(|sn| StoredFact {
                parts: match &sn.neuron {
                    Neuron::Expression(p) => p.clone(),
                    other => vec![other.clone()],
                },
            }).collect();
            eprintln!("      genesis-S2-derived[pair{}]: {} facts after DNA chaining", di, facts.len());
            facts
        })
        .collect();

    // ── STEP 2.5: GRAPH STRUCTURE — petgraph analysis ────────────
    let graph_insights = {
        let mut analysis_session = base_session.clone();
        if !training_inputs.is_empty() {
            let _ = analysis_session.exec_statements(training_inputs[0].clone());
        }
        analyze_graph_structure(&analysis_session)
    };
    eprintln!("      genesis-S2.5-graph: {} components, sizes={:?}, hubs={:?}",
        graph_insights.num_components,
        &graph_insights.component_sizes[..graph_insights.component_sizes.len().min(5)],
        &graph_insights.hub_predicates[..graph_insights.hub_predicates.len().min(3)]);

    // ── STEP 3: PROPOSE — from discovered patterns ───────────────
    let num_pairs = training_inputs.len();
    let pre_propose = candidates.len();
    let pat_candidates = propose_from_patterns(
        &target_pred, &in_profile, &out_profile, &contradictions, &derived_facts, num_pairs,
    );
    eprintln!("      genesis-S3-propose: patterns={}", pat_candidates.len());
    candidates.extend(pat_candidates);

    // Multi-body joins, arithmetic, aggregates, conditional splits
    let join_candidates = propose_multi_join(
        &target_pred, &in_profile, &out_profile, &contradictions, &derived_facts, num_pairs,
    );
    eprintln!("      genesis-S3-propose: multi-join={}", join_candidates.len());
    candidates.extend(join_candidates);

    let arith_candidates = propose_arithmetic_combos(
        &target_pred, &in_profile, &contradictions, num_pairs,
    );
    eprintln!("      genesis-S3-propose: arithmetic={}", arith_candidates.len());
    candidates.extend(arith_candidates);

    let agg_candidates = propose_aggregate_conditions(
        &target_pred, &in_profile, &contradictions, num_pairs,
    );
    eprintln!("      genesis-S3-propose: aggregate={}", agg_candidates.len());
    candidates.extend(agg_candidates);

    let split_candidates = propose_conditional_split(
        &target_pred, &in_profile, &contradictions, num_pairs,
    );
    eprintln!("      genesis-S3-propose: conditional-split={}", split_candidates.len());
    candidates.extend(split_candidates);

    let spatial_candidates = propose_spatial(
        &target_pred, &in_profile, &out_profile, &contradictions, num_pairs,
    );
    eprintln!("      genesis-S3-propose: spatial={}", spatial_candidates.len());
    for (si, c) in spatial_candidates.iter().take(3).enumerate() {
        eprintln!("        spatial[{}]: src={} rule={}", si, c.source,
            c.rule_text.chars().take(140).collect::<String>());
    }
    candidates.extend(spatial_candidates);

    // ── Observation-driven proposals (Vedic/Ramanujan: targeted, not blind) ──
    let observations = extract_observations(&derived_facts);
    let obs_candidates = propose_from_observations(
        &target_pred, &in_profile, &out_profile, &contradictions, &observations, num_pairs,
    );
    eprintln!("      genesis-S3-propose: obs-driven={}", obs_candidates.len());
    for (oi, c) in obs_candidates.iter().take(3).enumerate() {
        eprintln!("        obs[{}]: src={} rule={}", oi, c.source,
            c.rule_text.chars().take(140).collect::<String>());
    }
    candidates.extend(obs_candidates);

    // ── Value tracing — evidence-based rule construction ──
    let trace_candidates = propose_from_value_tracing(
        &target_pred, training_inputs, training_outputs,
        &derived_facts, &graph_insights.hub_predicates, num_pairs,
    );
    eprintln!("      genesis-S3-propose: value-trace={}", trace_candidates.len());
    for (ti, c) in trace_candidates.iter().take(3).enumerate() {
        eprintln!("        trace[{}]: src={} rule={}", ti, c.source,
            c.rule_text.chars().take(140).collect::<String>());
    }
    candidates.extend(trace_candidates);

    // ── Join mining — bottom-up rule discovery via datafrog ──
    #[cfg(feature = "datafrog")]
    {
        let target_arity = out_profile.predicates.get(&target_pred).map(|a| a.len()).unwrap_or(0);
        let jm_candidates = join_mine(
            base_session, training_inputs, training_outputs,
            &target_pred, target_arity,
        );
        eprintln!("      genesis-S3-propose: join-mine={}", jm_candidates.len());
        candidates.extend(jm_candidates);
    }

    let total_proposed = candidates.len() - pre_propose;
    eprintln!("      genesis-S3-total: {} proposed ({} + {} recalled)", total_proposed + pre_propose, total_proposed, pre_propose);
    // Show first 5 proposed rules
    for (pi, c) in candidates.iter().skip(pre_propose).take(5).enumerate() {
        eprintln!("        proposed[{}]: src={} rule={}", pi, c.source,
            c.rule_text.chars().take(120).collect::<String>());
    }

    // ── STEP 4: EVALUATE — parallel scoring via rayon + PrebuiltScorer ──
    // PrebuiltScorer pre-builds per-pair sessions ONCE, then only clones
    // the store (not the full session) for each candidate.
    // rayon parallelizes scoring across all CPU cores.
    let scorer = PrebuiltScorer::new(base_session, training_inputs, training_outputs, &target_pred);
    eprintln!("      genesis-S4-evaluate: scoring {} candidates (rayon parallel)...", candidates.len());
    let eval_start = Instant::now();
    let scored_count = candidates.len();

    // Score ALL candidates in parallel — no more sequential timeout at 10/252
    let scores: Vec<f64> = candidates.par_iter()
        .map(|c| scorer.score(&c.rule_text))
        .collect();
    for (c, s) in candidates.iter_mut().zip(scores.into_iter()) {
        c.score = s;
    }

    let eval_ms = eval_start.elapsed().as_millis();
    let before_retain = candidates.len();
    candidates.retain(|c| c.score > 0.001);
    sort_candidates(&mut candidates);
    eprintln!("      genesis-S4-done: scored {}/{}, {}/{} survived, {}ms ({:.1}ms/candidate), top-5:",
        scored_count, before_retain, candidates.len(), before_retain,
        eval_ms, if scored_count > 0 { eval_ms as f64 / scored_count as f64 } else { 0.0 });
    for (si, c) in candidates.iter().take(5).enumerate() {
        eprintln!("        scored[{}]: {:.1}% src={} rule={}", si, c.score * 100.0, c.source,
            c.rule_text.chars().take(120).collect::<String>());
    }

    // ── STEP 4.5: TEST-FIRE CHECK — penalize training-only rules ──
    // Build set of predicates available at test time (have facts for test input ID).
    // Rules using ONLY test-available predicates can fire on unseen test data.
    // Rules depending on training-only predicates (cell-diff, cell-vacated, etc.)
    // score well on training but produce zero predictions on test → useless.
    let test_filtered = apply_test_fire_filter(&mut candidates, base_session);
    if test_filtered > 0 {
        candidates.retain(|c| c.score > 0.001);
        sort_candidates(&mut candidates);
        eprintln!("      genesis-S4.5-test-fire: {} candidates zeroed (training-only), {} remain",
            test_filtered, candidates.len());
    }

    // ── STEP 5: COMBINE — merge top partials ─────────────────────
    if candidates.len() >= 2 && Instant::now() < deadline {
        eprintln!("      genesis-S5-combine: trying combos of top {} candidates", candidates.len().min(10));
        let combos = combine_top_rules_fast(&candidates, &scorer, &deadline);
        eprintln!("      genesis-S5-done: {} combos found", combos.len());
        for (ci, c) in combos.iter().take(3).enumerate() {
            eprintln!("        combo[{}]: {:.1}% src={}", ci, c.score * 100.0, c.source);
        }
        candidates.extend(combos);
    } else {
        eprintln!("      genesis-S5-skip: {} candidates, deadline_passed={}", candidates.len(), Instant::now() >= deadline);
    }

    // ── STEP 6: EVOLVE — guided mutation + SCAMPER ───────────────
    // Guarantee at least 3 generations even if deadline passed — evolution
    // is essential for improving candidates via condition addition/modification.
    let mut gen = 0;
    let evo_min_gens = 3;
    eprintln!("      genesis-S6-evolve: starting evolution (max 15 gens, best={:.1}%)",
        candidates.first().map(|c| c.score * 100.0).unwrap_or(0.0));
    while (Instant::now() < deadline || gen < evo_min_gens) && gen < 15 {
        sort_candidates(&mut candidates);
        if candidates.first().map(|c| c.score >= 0.999).unwrap_or(false) {
            eprintln!("      genesis-S6-perfect: found 100% at gen {}", gen);
            break;
        }
        candidates.truncate(15);

        let tops: Vec<Candidate> = candidates.iter().take(5).cloned().collect();
        let mut gen_mutations = 0;
        let mut gen_improvements = 0;
        for (top_idx, top) in tops.iter().enumerate() {
            // Allow at least the first candidate per forced generation
            if Instant::now() >= deadline && (gen >= evo_min_gens || top_idx > 0) { break; }
            let detail = score_candidate_detailed(&top.rule_text, &target_pred,
                base_session, training_inputs, training_outputs);

            // ALWAYS run both guided + scamper mutations.
            // Guided: pattern-specific fixes (value changes, relaxation, constraint addition)
            // Scamper: structural mutations (Grow adds conditions, Substitute changes preds)
            // Previously guided blocked scamper — this caused single-body rules to
            // stay single-body forever because Grow never ran.
            let mut mutations = guided_mutate(&top.rule_text, &top.source, &detail);
            let guided_count = mutations.len();
            let scamper = general_scamper_mutate(&top.rule_text, &top.source, &in_profile);
            let scamper_count = scamper.len();
            mutations.extend(scamper);

            gen_mutations += mutations.len();
            for (text, src) in mutations {
                let s = scorer.score(&text);
                if s > 0.001 {
                    if s > top.score { gen_improvements += 1; }
                    candidates.push(Candidate { rule_text: text, score: s, source: src });
                }
            }
            if gen == 0 {
                eprintln!("        evolve-gen0: parent={:.1}% pattern={:?} {} mutations (guided={}, scamper={})",
                    top.score * 100.0, detail.pattern,
                    guided_count + scamper_count, guided_count, scamper_count);
                if !detail.wrong_keys.is_empty() {
                    for (exp, got) in detail.wrong_keys.iter().take(3) {
                        eprintln!("          wrong: expected=[{}] got=[{}]", exp, got);
                    }
                }
                if !detail.missing_keys.is_empty() {
                    for mk in detail.missing_keys.iter().take(3) {
                        eprintln!("          missing: [{}]", mk);
                    }
                }
            }
        }
        gen += 1;
        if gen <= 3 || gen % 5 == 0 {
            sort_candidates(&mut candidates);
            eprintln!("      genesis-S6-gen{}: {} mutations, {} improvements, best={:.1}%",
                gen, gen_mutations, gen_improvements,
                candidates.first().map(|c| c.score * 100.0).unwrap_or(0.0));
        }
    }
    eprintln!("      genesis-S6-done: {} generations, best={:.1}%",
        gen, candidates.first().map(|c| c.score * 100.0).unwrap_or(0.0));

    sort_candidates(&mut candidates);
    candidates.dedup_by(|a, b| a.rule_text == b.rule_text);
    egraph_dedup_candidates(&mut candidates);

    // Filter out already-failed rules BEFORE truncation — this lets novel
    // candidates (multi-body joins, arithmetic) survive instead of being
    // crowded out by high-scoring but overfitting 1-body copy rules.
    if let Some(skip) = skip_rules {
        let before = candidates.len();
        candidates.retain(|c| !skip.contains(&c.rule_text));
        let filtered = before - candidates.len();
        if filtered > 0 {
            eprintln!("      genesis-S6-skip: filtered {} already-failed rules ({} → {})",
                filtered, before, candidates.len());
        }
    }
    candidates.truncate(10);

    // ── STEP 7: REMEMBER — store winning rules back to library ────
    let remember_count = candidates.iter().filter(|c| c.score > 0.8).count();
    if let Some(lib) = library {
        for c in candidates.iter().filter(|c| c.score > 0.8) {
            lib.add(c.rule_text.clone(), crate::library::RuleSource::Composed);
            lib.record_firing(&c.rule_text, true);
        }
    }
    eprintln!("      genesis-S7-remember: {} rules saved to library (>{:.0}% threshold)",
        remember_count, 80.0);
    eprintln!("      genesis-FINAL: returning {} candidates, elapsed={}ms",
        candidates.len(), (Instant::now() - deadline + Duration::from_millis(time_budget_ms)).as_millis());
    for (fi, c) in candidates.iter().take(5).enumerate() {
        eprintln!("        final[{}]: {:.1}% src={} rule={}", fi, c.score * 100.0, c.source,
            c.rule_text.chars().take(120).collect::<String>());
    }

    candidates
}

// ═══════════════════════════════════════════════════════════════════════
// PARALLEL SWARM — Quantum-style: try all strategies, first answer wins
// ═══════════════════════════════════════════════════════════════════════

/// Result from one worker thread.
pub struct WorkerResult {
    pub candidates: Vec<Candidate>,
    pub worker_id: usize,
}

/// Each worker focuses on a different slice of the search space.
/// DOMAIN AGNOSTIC — no domain-specific strategies. All strategies work
/// with raw facts. Domain knowledge comes from .qor DNA files.
#[derive(Clone, Debug)]
pub enum WorkerStrategy {
    /// Identity, value remap, complete remap
    BasicTransforms,
    /// Shift, reverse, swap, modular, categorical substitution
    PositionalTransforms,
    /// DNA-derived fact correlation (clones session, runs chaining)
    DerivedFacts,
    /// Recall from library + evolve past solutions
    MemoryRecall,
    /// Random exploration with a unique seed
    RandomExplore(u64),
    /// Multi-body joins + arithmetic + aggregates (new capabilities)
    MultiJoinArithmetic,
    /// Combinatorial: try ALL predicate combinations as rule bodies
    Combinatorial,
    /// Meta strategy — loads ONE meta .qor file's statements into the session.
    /// Bridge rules fire via forward chain → domain-specific insight facts.
    /// Then proposes rules from the enriched derived vocabulary.
    MetaStrategy(Vec<Statement>),
}

/// Strategy hints from reasoning.qor — controls worker allocation AND behavior.
/// reasoning.qor skip/try/constrain hints guide which GENERIC strategies get
/// workers and how they explore. No domain logic here — the .qor rules decide.
#[derive(Default, Debug, Clone)]
pub struct SwarmHints {
    /// Hint names to try (from genesis-hint facts). Used to allocate more
    /// workers to DerivedFacts/BasicTransforms/etc.
    pub try_these: Vec<String>,
    /// Hint names to skip (from genesis-skip facts). Used to remove
    /// workers from certain generic strategies.
    pub skip_these: Vec<String>,
    /// Problem classification from reasoning.qor (e.g. "remap", "spatial",
    /// "scale", "filter", "region-fill"). Guides worker allocation and
    /// predicate prioritization in Grow/Combinatorial.
    pub problem_class: Option<String>,
    /// Conservation constraints from reasoning.qor (e.g. "element-count",
    /// "shape", "value-set"). Candidates violating these are rejected early.
    pub constraints: Vec<String>,
}

/// Assign strategies to workers, optionally guided by reasoning.qor hints.
///
/// Hints control HOW MANY workers go to each generic strategy — not what
/// the strategies do. The strategies themselves are domain-agnostic.
///
/// - skip hints: remove workers from certain generic strategies
/// - try hints: allocate extra workers to DerivedFacts (DNA-driven)
pub fn assign_strategies_with_hints(
    num_workers: usize,
    hints: Option<&SwarmHints>,
) -> Vec<WorkerStrategy> {
    // Start with the full generic set
    let mut strats: Vec<WorkerStrategy> = vec![
        WorkerStrategy::BasicTransforms,
        WorkerStrategy::Combinatorial,  // Join mining — most important for rule discovery
        WorkerStrategy::DerivedFacts,
        WorkerStrategy::MemoryRecall,
        WorkerStrategy::PositionalTransforms,
        WorkerStrategy::MultiJoinArithmetic,
    ];

    // If reasoning.qor says to skip certain categories, remove workers
    if let Some(h) = hints {
        if h.skip_these.iter().any(|s| s.contains("spatial") || s.contains("positional")) {
            strats.retain(|s| !matches!(s, WorkerStrategy::PositionalTransforms));
        }
        // If there are try-hints, add extra DerivedFacts workers — DNA rules
        // encode all domain logic, so more DerivedFacts = more domain exploration
        let try_count = h.try_these.len();
        if try_count > 2 {
            strats.push(WorkerStrategy::DerivedFacts);
        }
        if try_count > 5 {
            strats.push(WorkerStrategy::DerivedFacts);
        }

        // Use problem_class to focus worker allocation.
        // Each class has a natural affinity for certain strategy types.
        if let Some(ref cls) = h.problem_class {
            match cls.as_str() {
                "color-remap" | "remap" => {
                    // Color remaps need DNA-derived correlation, not spatial transforms
                    strats.retain(|s| !matches!(s, WorkerStrategy::PositionalTransforms));
                    strats.push(WorkerStrategy::DerivedFacts);
                }
                "spatial" | "geometric" => {
                    // Spatial transforms need positional exploration
                    strats.push(WorkerStrategy::PositionalTransforms);
                }
                "tiling" | "scaling" | "scale" => {
                    // Tiling/scaling needs combinatorial + basic transforms
                    strats.retain(|s| !matches!(s, WorkerStrategy::BasicTransforms));
                    strats.push(WorkerStrategy::Combinatorial);
                }
                "filtering" | "filter" => {
                    // Filtering needs DNA-derived + combinatorial
                    strats.push(WorkerStrategy::DerivedFacts);
                    strats.push(WorkerStrategy::Combinatorial);
                }
                "region-fill" | "fill" => {
                    // Region fill needs DNA-derived correlation
                    strats.push(WorkerStrategy::DerivedFacts);
                }
                _ => {} // unknown or other — keep defaults
            }
        }
    }

    // Fill remaining slots with exploration (unique seeds)
    while strats.len() < num_workers {
        let seed = strats.len() as u64 * 42;
        strats.push(WorkerStrategy::RandomExplore(seed));
    }

    strats.truncate(num_workers);
    strats
}

/// Assign one worker per meta .qor file. Each worker loads ONE meta file's
/// statements and races against the others.
///
/// - One MetaStrategy worker per meta file
/// - Skip meta files whose name appears in hints.skip_these
/// - Fill remaining slots (if num_workers > meta count) with generic strategies
pub fn assign_meta_strategies(
    meta_strategies: &[(String, Vec<Statement>)],
    num_workers: usize,
    hints: Option<&SwarmHints>,
) -> Vec<WorkerStrategy> {
    let mut strats: Vec<WorkerStrategy> = Vec::new();

    for (name, stmts) in meta_strategies {
        // Skip meta files that reasoning.qor says to skip
        if let Some(h) = hints {
            if h.skip_these.iter().any(|s| s == name) {
                continue;
            }
        }
        strats.push(WorkerStrategy::MetaStrategy(stmts.clone()));
    }

    // Always include Combinatorial (join mining) and DerivedFacts alongside meta workers
    strats.push(WorkerStrategy::Combinatorial);
    strats.push(WorkerStrategy::DerivedFacts);

    // Fill remaining slots with generic strategies
    if strats.len() < num_workers {
        let generic = assign_strategies_with_hints(
            num_workers - strats.len(), hints,
        );
        strats.extend(generic);
    }

    strats.truncate(num_workers);
    strats
}

/// Detect how many workers to use based on available CPU cores.
pub fn optimal_worker_count() -> usize {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    match cores {
        1 => 1,
        2 => 2,
        3..=4 => 3,
        5..=8 => 4,
        _ => 5,
    }
}

/// Run N workers in parallel. First worker to score >= 0.95 cancels all others.
///
/// Like quantum superposition: all strategies explored simultaneously,
/// first observation (solution found) collapses the search.
///
/// With `hints`: reasoning.qor guides which strategies to use/skip.
/// Without: falls back to default strategy assignment.
pub fn genesis_swarm(
    base_session: &Session,
    training_inputs: &[Vec<Statement>],
    training_outputs: &[Vec<Statement>],
    time_budget_ms: u64,
    library: Option<&mut RuleLibrary>,
    num_workers: usize,
    skip_rules: Option<&HashSet<String>>,
) -> Vec<Candidate> {
    genesis_swarm_with_hints(base_session, training_inputs, training_outputs,
        time_budget_ms, library, num_workers, None, None, skip_rules)
}

/// Run N workers in parallel with optional reasoning.qor hints.
///
/// If `meta_strategies` is Some, each worker loads one meta .qor file (race mode).
/// If None, falls back to generic strategy assignment.
pub fn genesis_swarm_with_hints(
    base_session: &Session,
    training_inputs: &[Vec<Statement>],
    training_outputs: &[Vec<Statement>],
    time_budget_ms: u64,
    library: Option<&mut RuleLibrary>,
    num_workers: usize,
    hints: Option<&SwarmHints>,
    meta_strategies: Option<&[(String, Vec<Statement>)]>,
    skip_rules: Option<&HashSet<String>>,
) -> Vec<Candidate> {
    let cancel = Arc::new(AtomicBool::new(false));
    let strategies = if let Some(metas) = meta_strategies {
        assign_meta_strategies(metas, num_workers, hints)
    } else {
        assign_strategies_with_hints(num_workers, hints)
    };
    eprintln!("    swarm: {} workers, strategies: {:?}",
        strategies.len(),
        strategies.iter().map(|s| match s {
            WorkerStrategy::MetaStrategy(stmts) => format!("Meta({}stmts)", stmts.len()),
            other => format!("{:?}", other),
        }).collect::<Vec<_>>());

    // Recall library rules on main thread before spawning workers
    let mut recalled: Vec<Candidate> = Vec::new();
    if let Some(ref lib) = library {
        let mut past: Vec<_> = lib.all_rules().iter()
            .filter(|r| r.accuracy() > 0.1)
            .collect();
        past.sort_unstable_by(|a, b| b.accuracy().partial_cmp(&a.accuracy())
            .unwrap_or(std::cmp::Ordering::Equal));
        for scored in past.iter().take(10) {
            recalled.push(Candidate {
                rule_text: scored.rule_text.clone(),
                score: 0.0,
                source: "memory-recall".into(),
            });
        }
    }

    // Profile + contradict on main thread (shared, read-only data)
    let all_inputs: Vec<&Statement> = training_inputs.iter().flat_map(|v| v.iter()).collect();
    let all_outputs: Vec<&Statement> = training_outputs.iter().flat_map(|v| v.iter()).collect();
    let in_profile = profile_facts(&all_inputs);
    let out_profile = profile_facts(&all_outputs);
    let target_pred = find_target_predicate(&out_profile, &in_profile);
    let num_pairs = training_inputs.len();

    let contradictions: Vec<GeneralContradiction> = training_inputs.iter()
        .zip(training_outputs.iter())
        .map(|(inp, out)| find_general_contradiction(inp, out))
        .collect();

    // Run workers in parallel using scoped threads
    let results: Vec<WorkerResult> = std::thread::scope(|scope| {
        let handles: Vec<_> = strategies.iter().enumerate().map(|(id, strategy)| {
            let cancel = cancel.clone();
            let session = base_session.clone();
            let inputs = training_inputs.to_vec();
            let outputs = training_outputs.to_vec();
            let strat = strategy.clone();
            let target = target_pred.clone();
            let in_prof = &in_profile;
            let out_prof = &out_profile;
            let contras = &contradictions;
            let recall = if matches!(strat, WorkerStrategy::MemoryRecall) {
                recalled.clone()
            } else {
                Vec::new()
            };

            scope.spawn(move || {
                worker_genesis(
                    id, &session, &inputs, &outputs,
                    time_budget_ms, &strat, &cancel,
                    &target, in_prof, out_prof, contras,
                    num_pairs, recall,
                )
            })
        }).collect();

        handles.into_iter()
            .filter_map(|h| h.join().ok())
            .collect()
    });

    // Merge all results
    let mut all_candidates: Vec<Candidate> = results.into_iter()
        .flat_map(|r| r.candidates)
        .collect();

    sort_candidates(&mut all_candidates);
    all_candidates.dedup_by(|a, b| a.rule_text == b.rule_text);
    egraph_dedup_candidates(&mut all_candidates);

    // Filter out already-failed rules BEFORE truncation
    if let Some(skip) = skip_rules {
        let before = all_candidates.len();
        all_candidates.retain(|c| !skip.contains(&c.rule_text));
        let filtered = before - all_candidates.len();
        if filtered > 0 {
            eprintln!("    swarm-skip: filtered {} already-failed rules ({} → {})",
                filtered, before, all_candidates.len());
        }
    }
    all_candidates.truncate(10);

    // Store winners in library
    if let Some(lib) = library {
        for c in all_candidates.iter().filter(|c| c.score > 0.8) {
            lib.add(c.rule_text.clone(), crate::library::RuleSource::Composed);
            lib.record_firing(&c.rule_text, true);
        }
    }

    all_candidates
}

/// Individual worker thread. Explores one strategy slice of the search space.
/// Checks cancel signal regularly — stops immediately when another worker wins.
fn worker_genesis(
    worker_id: usize,
    base_session: &Session,
    training_inputs: &[Vec<Statement>],
    training_outputs: &[Vec<Statement>],
    time_budget_ms: u64,
    strategy: &WorkerStrategy,
    cancel: &AtomicBool,
    target_pred: &str,
    in_profile: &FactProfile,
    out_profile: &FactProfile,
    contradictions: &[GeneralContradiction],
    num_pairs: usize,
    recalled: Vec<Candidate>,
) -> WorkerResult {
    let deadline = Instant::now() + Duration::from_millis(time_budget_ms);

    // Each worker proposes candidates based on its strategy
    let target_arity = out_profile.predicates.get(target_pred).map(|a| a.len()).unwrap_or(0);
    let mut candidates = match strategy {
        WorkerStrategy::BasicTransforms => {
            propose_basic(target_pred, in_profile, out_profile, contradictions, num_pairs)
        }
        WorkerStrategy::PositionalTransforms => {
            propose_positional(target_pred, in_profile, contradictions, num_pairs, target_arity)
        }
        WorkerStrategy::DerivedFacts => {
            propose_derived_facts(target_pred, in_profile, base_session, training_inputs, num_pairs, target_arity)
        }
        WorkerStrategy::MemoryRecall => recalled,
        WorkerStrategy::RandomExplore(seed) => {
            propose_random(target_pred, in_profile, contradictions, num_pairs, *seed, target_arity)
        }
        WorkerStrategy::MetaStrategy(meta_stmts) => {
            // Inject meta file's statements into a cloned session.
            // Forward chain fires → meta bridge rules produce domain insights.
            // Only use NEWLY DERIVED predicates (from bridge rules), not raw domain facts.

            // 1. Snapshot predicates BEFORE meta injection
            let preds_before: HashSet<String> = base_session.all_facts().iter()
                .filter_map(|sn| match &sn.neuron {
                    Neuron::Expression(p) if !p.is_empty() => {
                        if let Neuron::Symbol(s) = &p[0] { Some(s.clone()) } else { None }
                    }
                    _ => None,
                }).collect();

            // 2. Also collect predicates from the meta statements themselves (raw domain facts)
            let meta_raw_preds: HashSet<String> = meta_stmts.iter()
                .filter_map(|stmt| match stmt {
                    Statement::Fact { neuron: Neuron::Expression(p), .. } if !p.is_empty() => {
                        if let Neuron::Symbol(s) = &p[0] { Some(s.clone()) } else { None }
                    }
                    _ => None,
                }).collect();

            // 3. Inject meta + forward chain
            let mut meta_session = base_session.clone();
            let _ = meta_session.exec_statements(meta_stmts.clone());

            // 4. Find predicates AFTER — only new ones from bridge rules (not raw meta facts)
            let preds_after: HashSet<String> = meta_session.all_facts().iter()
                .filter_map(|sn| match &sn.neuron {
                    Neuron::Expression(p) if !p.is_empty() => {
                        if let Neuron::Symbol(s) = &p[0] { Some(s.clone()) } else { None }
                    }
                    _ => None,
                }).collect();

            // New derived = (after - before) - raw_meta_preds
            let new_preds: HashSet<String> = preds_after.difference(&preds_before)
                .filter(|p| !meta_raw_preds.contains(*p))
                .cloned().collect();

            if new_preds.is_empty() {
                // No new insights from this meta file — return empty
                Vec::new()
            } else {
                // Build a restricted profile that only includes new predicates
                propose_derived_facts_filtered(
                    target_pred, in_profile, &meta_session, training_inputs,
                    num_pairs, target_arity, &new_preds,
                )
            }
        }
        WorkerStrategy::Combinatorial => {
            #[cfg(feature = "datafrog")]
            {
                let mut cands = join_mine(base_session, training_inputs, training_outputs, target_pred, target_arity);
                // Also include some combinatorial proposals as backup
                cands.extend(propose_combinatorial(target_pred, base_session, training_inputs, num_pairs, target_arity));
                cands
            }
            #[cfg(not(feature = "datafrog"))]
            {
                propose_combinatorial(target_pred, base_session, training_inputs, num_pairs, target_arity)
            }
        }
        WorkerStrategy::MultiJoinArithmetic => {
            // Run DNA to get derived facts for multi-join
            let derived_facts: Vec<Vec<StoredFact>> = training_inputs.iter()
                .map(|inp| {
                    let mut session = base_session.clone();
                    let _ = session.exec_statements(inp.clone());
                    session.all_facts().iter().map(|sn| StoredFact {
                        parts: match &sn.neuron {
                            Neuron::Expression(p) => p.clone(),
                            other => vec![other.clone()],
                        },
                    }).collect()
                })
                .collect();
            let mut cands = propose_multi_join(
                target_pred, in_profile, out_profile, contradictions, &derived_facts, num_pairs,
            );
            cands.extend(propose_arithmetic_combos(
                target_pred, in_profile, contradictions, num_pairs,
            ));
            cands.extend(propose_aggregate_conditions(
                target_pred, in_profile, contradictions, num_pairs,
            ));
            cands.extend(propose_conditional_split(
                target_pred, in_profile, contradictions, num_pairs,
            ));
            cands
        }
    };

    // PrebuiltScorer for fast scoring inside worker
    let scorer = PrebuiltScorer::new(base_session, training_inputs, training_outputs, target_pred);

    // Score all candidates in parallel within worker
    let scores: Vec<f64> = candidates.par_iter()
        .map(|c| scorer.score(&c.rule_text))
        .collect();
    for (c, s) in candidates.iter_mut().zip(scores.into_iter()) {
        c.score = s;
        if s >= 0.95 {
            cancel.store(true, Ordering::Relaxed);
        }
    }
    candidates.retain(|c| c.score > 0.001);

    // Evolve best candidates (if time remains and not cancelled)
    let mut gen = 0;
    while !cancel.load(Ordering::Relaxed) && Instant::now() < deadline && gen < 10 {
        sort_candidates(&mut candidates);
        if candidates.first().map(|c| c.score >= 0.95).unwrap_or(false) {
            cancel.store(true, Ordering::Relaxed);
            break;
        }
        candidates.truncate(10);

        let tops: Vec<Candidate> = candidates.iter().take(3).cloned().collect();
        for top in &tops {
            if cancel.load(Ordering::Relaxed) || Instant::now() >= deadline { break; }
            for (text, src) in general_scamper_mutate(&top.rule_text, &top.source, in_profile) {
                if cancel.load(Ordering::Relaxed) { break; }
                let s = scorer.score(&text);
                if s >= 0.95 {
                    candidates.push(Candidate { rule_text: text, score: s, source: src });
                    cancel.store(true, Ordering::Relaxed);
                    break;
                }
                if s > 0.001 {
                    candidates.push(Candidate { rule_text: text, score: s, source: src });
                }
            }
        }
        gen += 1;
    }

    sort_candidates(&mut candidates);
    candidates.truncate(5);

    WorkerResult { candidates, worker_id }
}

// ── Strategy-specific propose functions ──────────────────────────────

/// BasicTransforms: identity + value remap + count-based + fill + eliminate
fn propose_basic(
    target: &str,
    in_profile: &FactProfile,
    out_profile: &FactProfile,
    contradictions: &[GeneralContradiction],
    num_pairs: usize,
) -> Vec<Candidate> {
    let mut out = Vec::new();
    let target_arity = out_profile.predicates.get(target).map(|a| a.len()).unwrap_or(0);
    let (source, source_args, arity, vars, vars_str) = match extract_source(in_profile, contradictions, target_arity) {
        Some(v) => v,
        None => return out,
    };
    let np = num_pairs;

    // Identity
    out.push(mk(
        &format!("({target} {vars_str}) if\n    ({source} {vars_str}) <{}>", conf(0.50, np)),
        "identity",
    ));

    // Value remap from merged maps
    let merged = merge_value_maps(contradictions, &source, arity, &source_args);
    for ((_, pos), mapping) in &merged {
        if *pos == 0 || *pos > arity { continue; }
        let idx = pos - 1;
        if source_args[idx].is_id { continue; }
        for (&from, &to) in mapping {
            let mut ta = vars.clone(); ta[idx] = to.to_string();
            let mut sa = vars.clone(); sa[idx] = from.to_string();
            out.push(mk(
                &format!("({target} {}) if\n    ({source} {}) <{}>",
                    ta.join(" "), sa.join(" "), conf(0.90, np)),
                &format!("remap-pos{pos}-{from}to{to}"),
            ));
        }
        // Complete remap + fallback
        if !mapping.is_empty() {
            let mut combined = String::new();
            for (&from, &to) in mapping {
                let mut ta = vars.clone(); ta[idx] = to.to_string();
                let mut sa = vars.clone(); sa[idx] = from.to_string();
                combined.push_str(&format!("({target} {}) if\n    ({source} {}) <{}>\n",
                    ta.join(" "), sa.join(" "), conf(0.95, np)));
            }
            combined.push_str(&format!("({target} {vars_str}) if\n    ({source} {vars_str}) <{}>", conf(0.40, np)));
            out.push(mk(&combined, "remap-complete"));
        }
    }

    // Count-based (Pattern 5)
    for c in contradictions.iter().take(1) {
        for (pred, (in_c, out_c)) in &c.count_changes {
            if pred == &source && *out_c > *in_c && *in_c > 0 {
                let factor = out_c / in_c;
                if *out_c == factor * in_c {
                    out.push(mk(
                        &format!("({target} {vars_str}) if\n    ({source} {vars_str}) <{}>", conf(0.70, np)),
                        &format!("expand-factor{factor}"),
                    ));
                }
            }
        }
    }

    // Fill from output profile (Pattern 6)
    if let Some(out_args) = out_profile.predicates.get(target) {
        for (i, arg) in out_args.iter().enumerate() {
            if !arg.is_categorical || i >= vars.len() { continue; }
            if let Some(&common_val) = arg.ints.iter().max() {
                let mut ta = vars.clone(); ta[i] = common_val.to_string();
                out.push(mk(
                    &format!("({target} {}) if\n    ({source} {vars_str}) <{}>",
                        ta.join(" "), conf(0.55, np)),
                    &format!("fill-pos{i}-val{common_val}"),
                ));
            }
        }
    }

    // Eliminate (Pattern 7)
    for c in contradictions.iter().take(1) {
        if !c.removed.is_empty() && c.added.is_empty() {
            for (i, arg) in source_args.iter().enumerate() {
                if !arg.is_categorical { continue; }
                if let Some(&min_val) = arg.ints.iter().min() {
                    out.push(mk(
                        &format!("({target} {vars_str}) if\n    ({source} {vars_str})\n    (!= {} {min_val}) <{}>",
                            vars[i], conf(0.65, np)),
                        &format!("eliminate-pos{i}-neq{min_val}"),
                    ));
                }
            }
        }
    }

    out
}

/// PositionalTransforms: swap, shift, reverse, modular, categorical substitution
fn propose_positional(
    target: &str,
    in_profile: &FactProfile,
    contradictions: &[GeneralContradiction],
    num_pairs: usize,
    target_arity: usize,
) -> Vec<Candidate> {
    let mut out = Vec::new();
    let (source, source_args, arity, vars, vars_str) = match extract_source(in_profile, contradictions, target_arity) {
        Some(v) => v,
        None => return out,
    };
    let np = num_pairs;
    if arity < 3 { return out; }

    let seq_positions: Vec<usize> = source_args.iter().enumerate()
        .filter(|(_, a)| a.is_sequential)
        .map(|(i, _)| i)
        .collect();

    // Swap pairs
    for i in 0..seq_positions.len() {
        for j in (i + 1)..seq_positions.len() {
            let (pi, pj) = (seq_positions[i], seq_positions[j]);
            let mut swapped = vars.clone();
            swapped.swap(pi, pj);
            out.push(mk(
                &format!("({target} {}) if\n    ({source} {vars_str}) <{}>",
                    swapped.join(" "), conf(0.80, np)),
                &format!("swap-pos{pi}-pos{pj}"),
            ));
        }
    }

    // Shift
    for &pos in &seq_positions {
        for shift in [-2i64, -1, 1, 2] {
            let sv = format!("$shifted{pos}");
            let op = if shift > 0 { "+" } else { "-" };
            let abs = shift.unsigned_abs();
            let mut ta = vars.clone(); ta[pos] = sv.clone();
            out.push(mk(
                &format!("({target} {}) if\n    ({source} {vars_str})\n    ({op} {} {abs} {sv}) <{}>",
                    ta.join(" "), vars[pos], conf(0.75, np)),
                &format!("shift-pos{pos}-by{shift}"),
            ));
        }
    }

    // Reverse
    for &pos in &seq_positions {
        let max_val = source_args[pos].ints.iter().max().copied().unwrap_or(0);
        if max_val > 0 {
            let rv = format!("$rev{pos}");
            let mut ta = vars.clone(); ta[pos] = rv.clone();
            out.push(mk(
                &format!("({target} {}) if\n    ({source} {vars_str})\n    (- {max_val} {} {rv}) <{}>",
                    ta.join(" "), vars[pos], conf(0.80, np)),
                &format!("reverse-pos{pos}-max{max_val}"),
            ));
        }
    }

    // Modular
    for &pos in &seq_positions {
        let max_val = source_args[pos].ints.iter().max().copied().unwrap_or(0);
        if max_val > 0 {
            let mv = format!("$mod{pos}");
            let dim = max_val + 1;
            let mut lookup = vars.clone(); lookup[pos] = mv.clone();
            out.push(mk(
                &format!("({target} {vars_str}) if\n    (% {} {dim} {mv})\n    ({source} {}) <{}>",
                    vars[pos], lookup.join(" "), conf(0.80, np)),
                &format!("modular-pos{pos}-mod{dim}"),
            ));
        }
    }

    // Categorical substitution (excluding evidence-backed remaps)
    let merged = merge_value_maps(contradictions, &source, arity, &source_args);
    let evidence_positions: HashSet<usize> = merged.iter()
        .filter(|((pred, _), _)| pred == &source)
        .map(|((_, pos), _)| pos - 1)
        .collect();
    for (i, arg) in source_args.iter().enumerate() {
        if !arg.is_categorical || evidence_positions.contains(&i) { continue; }
        let cat_ints: Vec<i64> = arg.ints.iter().copied().collect();
        for &from in &cat_ints {
            for &to in &cat_ints {
                if from == to { continue; }
                let mut ta = vars.clone(); ta[i] = to.to_string();
                let mut sa = vars.clone(); sa[i] = from.to_string();
                out.push(mk(
                    &format!("({target} {}) if\n    ({source} {}) <{}>",
                        ta.join(" "), sa.join(" "), conf(0.70, np)),
                    &format!("cat-sub-pos{i}-{from}to{to}"),
                ));
            }
            if out.len() > 200 { break; }
        }
    }

    out
}

/// DerivedFacts: run DNA rules on each input, correlate derived facts with output
fn propose_derived_facts(
    target: &str,
    in_profile: &FactProfile,
    base_session: &Session,
    training_inputs: &[Vec<Statement>],
    num_pairs: usize,
    target_arity: usize,
) -> Vec<Candidate> {
    let mut out = Vec::new();
    // Prefer same-arity as target predicate
    let source = if target_arity > 0 {
        in_profile.predicates.iter()
            .filter(|(_, args)| args.len() == target_arity)
            .max_by_key(|(_, args)| args.iter().map(|a| a.ints.len() + a.syms.len()).sum::<usize>())
            .map(|(pred, _)| pred.clone())
    } else {
        None
    };
    let source = source.or_else(|| {
        in_profile.predicates.iter()
            .max_by_key(|(_, args)| args.len())
            .map(|(pred, _)| pred.clone())
    });
    let source = match source {
        Some(s) => s,
        None => return out,
    };
    let source_args = match in_profile.predicates.get(&source) {
        Some(a) => a,
        None => return out,
    };
    let arity = source_args.len();
    let vars: Vec<String> = (0..arity).map(|i| format!("$a{i}")).collect();
    let vars_str = vars.join(" ");
    let np = num_pairs;

    // Run DNA rules on each input to get derived facts
    let derived_facts: Vec<Vec<StoredFact>> = training_inputs.iter()
        .map(|inp| {
            let mut session = base_session.clone();
            let _ = session.exec_statements(inp.clone());
            session.all_facts().iter().map(|sn| StoredFact {
                parts: match &sn.neuron {
                    Neuron::Expression(p) => p.clone(),
                    other => vec![other.clone()],
                },
            }).collect()
        }).collect();

    let derived_preds: HashSet<String> = derived_facts.iter()
        .flat_map(|facts| facts.iter())
        .filter_map(|f| {
            if let Some(Neuron::Symbol(s)) = f.parts.first() {
                if !in_profile.predicates.contains_key(s) && s != target {
                    Some(s.clone())
                } else { None }
            } else { None }
        }).collect();

    for dpred in &derived_preds {
        let sample = derived_facts.iter()
            .flat_map(|facts| facts.iter())
            .find(|f| f.parts.first() == Some(&Neuron::Symbol(dpred.clone())));
        if let Some(sf) = sample {
            let d_arity = sf.parts.len() - 1;

            // Skip derived predicates that bind too few target variables.
            // For single-condition rules, we need d_arity >= target_arity
            // to bind ALL head variables. With fewer, unbound head vars
            // cause massive overgeneration (garbage proposals).
            if d_arity < target_arity {
                continue;
            }

            let mut dvars: Vec<String> = (0..d_arity).map(|i| format!("$d{i}")).collect();
            let shared = vars.len().min(dvars.len());
            // Share ALL corresponding variable positions to maximize binding
            for i in 0..shared {
                dvars[i] = vars[i].clone();
            }
            // With derived fact condition
            out.push(mk(
                &format!("({target} {vars_str}) if\n    ({dpred} {}) <{}>",
                    dvars.join(" "), conf(0.70, np)),
                &format!("derived-{dpred}"),
            ));
            // Identity gated by derived fact
            out.push(mk(
                &format!("({target} {vars_str}) if\n    ({source} {vars_str})\n    ({dpred} {}) <{}>",
                    dvars.join(" "), conf(0.65, np)),
                &format!("gated-{dpred}"),
            ));
        }
    }

    out
}

/// DerivedFacts (filtered): only propose from predicates in `allowed` set.
/// Used by MetaStrategy to restrict genesis to bridge-rule outputs (insights),
/// excluding raw domain-knowledge facts injected by meta files.
fn propose_derived_facts_filtered(
    target: &str,
    in_profile: &FactProfile,
    base_session: &Session,
    training_inputs: &[Vec<Statement>],
    num_pairs: usize,
    target_arity: usize,
    allowed: &HashSet<String>,
) -> Vec<Candidate> {
    let mut out = Vec::new();
    let source = if target_arity > 0 {
        in_profile.predicates.iter()
            .filter(|(_, args)| args.len() == target_arity)
            .max_by_key(|(_, args)| args.iter().map(|a| a.ints.len() + a.syms.len()).sum::<usize>())
            .map(|(pred, _)| pred.clone())
    } else {
        None
    };
    let source = source.or_else(|| {
        in_profile.predicates.iter()
            .max_by_key(|(_, args)| args.len())
            .map(|(pred, _)| pred.clone())
    });
    let source = match source {
        Some(s) => s,
        None => return out,
    };
    let source_args = match in_profile.predicates.get(&source) {
        Some(a) => a,
        None => return out,
    };
    let arity = source_args.len();
    let vars: Vec<String> = (0..arity).map(|i| format!("$a{i}")).collect();
    let vars_str = vars.join(" ");
    let np = num_pairs;

    let derived_facts: Vec<Vec<StoredFact>> = training_inputs.iter()
        .map(|inp| {
            let mut session = base_session.clone();
            let _ = session.exec_statements(inp.clone());
            session.all_facts().iter().map(|sn| StoredFact {
                parts: match &sn.neuron {
                    Neuron::Expression(p) => p.clone(),
                    other => vec![other.clone()],
                },
            }).collect()
        }).collect();

    // Only consider predicates in the allowed set
    let derived_preds: HashSet<String> = derived_facts.iter()
        .flat_map(|facts| facts.iter())
        .filter_map(|f| {
            if let Some(Neuron::Symbol(s)) = f.parts.first() {
                if allowed.contains(s) && !in_profile.predicates.contains_key(s) && s != target {
                    Some(s.clone())
                } else { None }
            } else { None }
        }).collect();

    for dpred in &derived_preds {
        let sample = derived_facts.iter()
            .flat_map(|facts| facts.iter())
            .find(|f| f.parts.first() == Some(&Neuron::Symbol(dpred.clone())));
        if let Some(sf) = sample {
            let d_arity = sf.parts.len() - 1;
            // Single-condition rules need d_arity >= target_arity to bind all head vars
            if d_arity < target_arity {
                continue;
            }
            let mut dvars: Vec<String> = (0..d_arity).map(|i| format!("$d{i}")).collect();
            let shared = vars.len().min(dvars.len());
            for i in 0..shared {
                dvars[i] = vars[i].clone();
            }
            out.push(mk(
                &format!("({target} {vars_str}) if\n    ({dpred} {}) <{}>",
                    dvars.join(" "), conf(0.70, np)),
                &format!("derived-{dpred}"),
            ));
            out.push(mk(
                &format!("({target} {vars_str}) if\n    ({source} {vars_str})\n    ({dpred} {}) <{}>",
                    dvars.join(" "), conf(0.65, np)),
                &format!("gated-{dpred}"),
            ));
        }
    }

    out
}

/// RandomExplore: generate random rule variations with different seed
fn propose_random(
    target: &str,
    in_profile: &FactProfile,
    contradictions: &[GeneralContradiction],
    num_pairs: usize,
    seed: u64,
    target_arity: usize,
) -> Vec<Candidate> {
    let mut out = Vec::new();
    let (source, source_args, arity, vars, vars_str) = match extract_source(in_profile, contradictions, target_arity) {
        Some(v) => v,
        None => return out,
    };
    let np = num_pairs;

    // Use seed to create different constant combinations
    let constants: Vec<i64> = in_profile.constants.iter().copied().collect();
    if constants.is_empty() { return out; }

    // Seeded exploration: try swapping different constant pairs
    let start = (seed as usize) % constants.len().max(1);
    for i in 0..constants.len().min(10) {
        let ci = (start + i) % constants.len();
        let cj = (start + i + 1 + (seed as usize / 3)) % constants.len();
        if ci == cj { continue; }
        let from = constants[ci];
        let to = constants[cj];

        // Constant substitution rules
        for pos in 0..arity.min(4) {
            let mut ta = vars.clone(); ta[pos] = to.to_string();
            let mut sa = vars.clone(); sa[pos] = from.to_string();
            out.push(mk(
                &format!("({target} {}) if\n    ({source} {}) <{}>",
                    ta.join(" "), sa.join(" "), conf(0.60, np)),
                &format!("random{seed}-swap-pos{pos}-{from}to{to}"),
            ));
        }
        if out.len() > 50 { break; }
    }

    // Seeded shift combinations
    let shift = ((seed % 5) as i64) - 2; // -2 to +2
    if shift != 0 && arity >= 3 {
        for pos in 0..arity.min(4) {
            if !source_args[pos].is_sequential { continue; }
            let sv = format!("$rs{pos}");
            let op = if shift > 0 { "+" } else { "-" };
            let abs = shift.unsigned_abs();
            let mut ta = vars.clone(); ta[pos] = sv.clone();
            out.push(mk(
                &format!("({target} {}) if\n    ({source} {vars_str})\n    ({op} {} {abs} {sv}) <{}>",
                    ta.join(" "), vars[pos], conf(0.65, np)),
                &format!("random{seed}-shift-pos{pos}-by{shift}"),
            ));
        }
    }

    out
}

// ── Shared helpers for strategy functions ────────────────────────────

/// Extract source predicate info from input profile.
/// Returns (source_name, source_args, arity, vars, vars_str)
fn extract_source<'a>(
    in_profile: &'a FactProfile,
    contradictions: &[GeneralContradiction],
    target_arity: usize,
) -> Option<(String, &'a [ArgInfo], usize, Vec<String>, String)> {
    let new_preds: HashSet<&String> = contradictions.iter()
        .flat_map(|c| c.new_predicates.iter())
        .collect();
    let non_new: Vec<_> = in_profile.predicates.iter()
        .filter(|(pred, _)| !new_preds.contains(pred))
        .collect();
    // Prefer input predicates with SAME ARITY as target — produces correct-shape facts.
    // Among same-arity, prefer more data-rich (more unique values).
    let source_entry = if target_arity > 0 {
        non_new.iter()
            .filter(|(_, args)| args.len() == target_arity)
            .max_by_key(|(_, args)| args.iter().map(|a| a.ints.len() + a.syms.len()).sum::<usize>())
            .map(|&entry| entry)
    } else {
        None
    };
    // Fallback: highest arity
    let source_entry = source_entry
        .or_else(|| non_new.iter().max_by_key(|(_, args)| args.len()).copied())
        .or_else(|| in_profile.predicates.iter().max_by_key(|(_, args)| args.len()));
    let (source, source_args) = source_entry?;
    let arity = source_args.len();
    let vars: Vec<String> = (0..arity).map(|i| format!("$a{i}")).collect();
    let vars_str = vars.join(" ");
    Some((source.clone(), source_args.as_slice(), arity, vars, vars_str))
}

/// Merge and validate value maps across all contradiction pairs.
fn merge_value_maps(
    contradictions: &[GeneralContradiction],
    source: &str,
    _arity: usize,
    source_args: &[ArgInfo],
) -> HashMap<(String, usize), HashMap<i64, i64>> {
    let mut merged: HashMap<(String, usize), HashMap<i64, i64>> = HashMap::new();
    for c in contradictions {
        for ((pred, pos), mapping) in &c.value_maps {
            let entry = merged.entry((pred.clone(), *pos)).or_default();
            for (&from, &to) in mapping { entry.insert(from, to); }
        }
    }
    // Cross-pair consistency
    merged.retain(|key, mapping| {
        if key.0 != source { return true; }
        let idx = key.1.wrapping_sub(1);
        if idx < source_args.len() && source_args[idx].is_id { return false; }
        mapping.retain(|&from, to| {
            let to_val = *to;
            contradictions.iter().all(|c| {
                match c.value_maps.get(key) {
                    Some(m) => m.get(&from).map(|&v| v == to_val).unwrap_or(true),
                    None => true,
                }
            })
        });
        !mapping.is_empty()
    });
    merged
}

fn sort_candidates(candidates: &mut [Candidate]) {
    // Occam's razor: between equal scores, shorter rules generalize better
    // (Kolmogorov complexity / MDL principle)
    candidates.sort_unstable_by(|a, b| {
        let sa = a.score - (a.rule_text.len() as f64 * 0.0001);
        let sb = b.score - (b.rule_text.len() as f64 * 0.0001);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Test-fire filter: penalize candidates that use predicates NOT available at test time.
///
/// Builds a set of "test-available predicates" by scanning base_session for facts
/// that contain the test input ID. Rules using predicates outside this set (e.g.,
/// cell-diff, cell-vacated — which only exist for training pairs) get score=0.
/// This prevents genesis from evolving rules that score well on training but
/// produce zero predictions on unseen test data.
fn apply_test_fire_filter(candidates: &mut [Candidate], base_session: &Session) -> usize {
    // Find the test input ID from (test-input <id>) fact
    let test_id: Option<String> = base_session.all_facts().iter().find_map(|sn| {
        if let Neuron::Expression(parts) = &sn.neuron {
            if parts.len() == 2 && parts[0] == Neuron::symbol("test-input") {
                return Some(parts[1].to_string());
            }
        }
        None
    });
    let test_id = match test_id {
        Some(id) => id,
        None => return 0, // No test input — skip filter
    };

    // Build set of predicates that have facts containing the test ID
    let tid_neuron = Neuron::symbol(&test_id);
    let mut test_available: HashSet<String> = HashSet::new();
    for sn in base_session.all_facts().iter() {
        if let Neuron::Expression(parts) = &sn.neuron {
            if parts.iter().any(|p| p == &tid_neuron) {
                if let Some(pred) = parts.first() {
                    test_available.insert(pred.to_string());
                }
            }
        }
    }
    // obs-* predicates are cross-pair observations (no grid ID) — always available
    for sn in base_session.all_facts().iter() {
        if let Neuron::Expression(parts) = &sn.neuron {
            if let Some(pred) = parts.first().map(|p| p.to_string()) {
                if pred.starts_with("obs-") {
                    test_available.insert(pred);
                }
            }
        }
    }

    let mut filtered = 0;
    for c in candidates.iter_mut() {
        if c.score <= 0.001 { continue; }

        // Extract body predicates from rule text
        let body_preds = extract_rule_body_predicates(&c.rule_text);
        let has_unavailable = body_preds.iter().any(|p| {
            // Skip guards and special keywords — they don't need session facts
            if p.starts_with('>') || p.starts_with('<') || p.starts_with('=')
                || p.starts_with('!') || p == "not" || p == "not-present"
                || p.starts_with("obs-")
            {
                return false;
            }
            // Skip arithmetic operators
            if p == "+" || p == "-" || p == "*" || p == "/" || p == "%" {
                return false;
            }
            !test_available.contains(p.as_str())
        });

        if has_unavailable {
            c.score = 0.0;
            filtered += 1;
        }
    }
    filtered
}

/// Extract body predicate names from a QOR rule text.
/// Returns the first symbol inside each (...) in the body.
fn extract_rule_body_predicates(rule_text: &str) -> Vec<String> {
    let mut preds = Vec::new();
    // Find the body (after " if ")
    let body_start = rule_text.find(" if\n")
        .map(|p| p + 4)
        .or_else(|| rule_text.find(" if ").map(|p| p + 4));
    let body = match body_start {
        Some(start) => &rule_text[start..],
        None => return preds,
    };
    // Split on '(' to find predicate starts
    for chunk in body.split('(') {
        let trimmed = chunk.trim();
        if let Some(word) = trimmed.split_whitespace().next() {
            let word = word.trim_end_matches(')');
            if !word.is_empty() && !word.starts_with('$') && !word.starts_with('<') {
                preds.push(word.to_string());
            }
        }
    }
    preds
}

// ═══════════════════════════════════════════════════════════════════════
// STEP 1: DECOMPOSE — Profile ANY facts
// ═══════════════════════════════════════════════════════════════════════

#[derive(Hash, Eq, PartialEq, Clone)]
enum ArgVal {
    Int(i64),
    Sym(String),
}

fn profile_facts(facts: &[&Statement]) -> FactProfile {
    let mut raw: HashMap<String, Vec<HashMap<ArgVal, usize>>> = HashMap::new();
    let mut constants = HashSet::new();
    let mut symbols = HashSet::new();

    for stmt in facts {
        let parts = match stmt {
            Statement::Fact { neuron: Neuron::Expression(p), .. } => p,
            _ => continue,
        };
        if parts.is_empty() { continue; }
        let pred = match &parts[0] {
            Neuron::Symbol(s) => s.clone(),
            _ => continue,
        };
        symbols.insert(pred.clone());

        let entry = raw.entry(pred).or_default();
        for (i, arg) in parts[1..].iter().enumerate() {
            while entry.len() <= i { entry.push(HashMap::new()); }
            match arg {
                Neuron::Value(QorValue::Int(v)) => {
                    constants.insert(*v);
                    *entry[i].entry(ArgVal::Int(*v)).or_insert(0) += 1;
                }
                Neuron::Symbol(s) => {
                    symbols.insert(s.clone());
                    *entry[i].entry(ArgVal::Sym(s.clone())).or_insert(0) += 1;
                }
                Neuron::Value(QorValue::Float(f)) => {
                    constants.insert(*f as i64);
                    *entry[i].entry(ArgVal::Int(*f as i64)).or_insert(0) += 1;
                }
                Neuron::Value(QorValue::Str(s)) => {
                    *entry[i].entry(ArgVal::Sym(s.clone())).or_insert(0) += 1;
                }
                _ => {}
            }
        }
    }

    let classified = raw.iter().map(|(pred, positions)| {
        let total_facts = facts.iter().filter(|s| {
            matches!(s, Statement::Fact { neuron: Neuron::Expression(p), .. }
                if p.first() == Some(&Neuron::Symbol(pred.clone())))
        }).count();

        let args: Vec<ArgInfo> = positions.iter().map(|pos_vals| {
            let mut ints = HashSet::new();
            let mut syms = HashSet::new();
            for (val, _) in pos_vals {
                match val {
                    ArgVal::Int(i) => { ints.insert(*i); }
                    ArgVal::Sym(s) => { syms.insert(s.clone()); }
                }
            }
            let unique_count = ints.len() + syms.len();
            let is_id = unique_count == 1 && total_facts > 1;
            let is_sequential = !ints.is_empty() && syms.is_empty() && {
                let min = ints.iter().copied().min().unwrap_or(0);
                let max = ints.iter().copied().max().unwrap_or(0);
                max - min + 1 == ints.len() as i64 && min >= 0
            };
            let is_categorical = unique_count > 1 && unique_count <= 10 && !is_id && !is_sequential;
            ArgInfo { ints, syms, is_id, is_sequential, is_categorical }
        }).collect();

        (pred.clone(), args)
    }).collect();

    FactProfile { predicates: classified, constants, symbols }
}

// ═══════════════════════════════════════════════════════════════════════
// STEP 2: CONTRADICT — General fact comparison
// ═══════════════════════════════════════════════════════════════════════

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

fn find_general_contradiction(
    input_facts: &[Statement],
    output_facts: &[Statement],
) -> GeneralContradiction {
    let mut in_by_pred: HashMap<String, Vec<Vec<Neuron>>> = HashMap::new();
    let mut out_by_pred: HashMap<String, Vec<Vec<Neuron>>> = HashMap::new();

    for stmt in input_facts {
        if let Statement::Fact { neuron: Neuron::Expression(p), .. } = stmt {
            if let Some(Neuron::Symbol(pred)) = p.first() {
                in_by_pred.entry(pred.clone()).or_default().push(p.clone());
            }
        }
    }
    for stmt in output_facts {
        if let Statement::Fact { neuron: Neuron::Expression(p), .. } = stmt {
            if let Some(Neuron::Symbol(pred)) = p.first() {
                out_by_pred.entry(pred.clone()).or_default().push(p.clone());
            }
        }
    }

    let new_predicates: HashSet<String> = out_by_pred.keys()
        .filter(|k| !in_by_pred.contains_key(*k))
        .cloned().collect();

    let all_preds: HashSet<&String> = in_by_pred.keys().chain(out_by_pred.keys()).collect();
    let mut count_changes = HashMap::new();
    for pred in &all_preds {
        let in_c = in_by_pred.get(*pred).map(|v| v.len()).unwrap_or(0);
        let out_c = out_by_pred.get(*pred).map(|v| v.len()).unwrap_or(0);
        if in_c != out_c {
            count_changes.insert((*pred).clone(), (in_c, out_c));
        }
    }

    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut value_maps: HashMap<(String, usize), HashMap<i64, i64>> = HashMap::new();

    for pred in &all_preds {
        let in_facts = in_by_pred.get(*pred).cloned().unwrap_or_default();
        let out_facts = out_by_pred.get(*pred).cloned().unwrap_or_default();
        let in_keys: HashSet<String> = in_facts.iter().map(|p| fact_key(p)).collect();
        let out_keys: HashSet<String> = out_facts.iter().map(|p| fact_key(p)).collect();

        for f in &out_facts {
            if !in_keys.contains(&fact_key(f)) { added.push(f.clone()); }
        }
        for f in &in_facts {
            if !out_keys.contains(&fact_key(f)) { removed.push(f.clone()); }
        }

        if in_facts.is_empty() || out_facts.is_empty() { continue; }
        let arity = in_facts[0].len();
        if arity < 2 { continue; }

        for val_pos in 1..arity {
            let mut mapping: HashMap<i64, i64> = HashMap::new();
            let mut consistent = true;

            let mut in_index: HashMap<String, i64> = HashMap::new();
            for f in &in_facts {
                let key: String = f.iter().enumerate()
                    .filter(|(i, _)| *i != val_pos)
                    .map(|(_, n)| match n {
                        Neuron::Value(QorValue::Int(v)) => v.to_string(),
                        Neuron::Symbol(s) => s.clone(),
                        _ => "_".into(),
                    }).collect::<Vec<_>>().join(",");
                if let Some(v) = neuron_to_i64(&f[val_pos]) {
                    in_index.insert(key, v);
                }
            }

            for f in &out_facts {
                let key: String = f.iter().enumerate()
                    .filter(|(i, _)| *i != val_pos)
                    .map(|(_, n)| match n {
                        Neuron::Value(QorValue::Int(v)) => v.to_string(),
                        Neuron::Symbol(s) => s.clone(),
                        _ => "_".into(),
                    }).collect::<Vec<_>>().join(",");

                if let (Some(&in_val), Some(out_val)) =
                    (in_index.get(&key), neuron_to_i64(&f[val_pos]))
                {
                    if in_val != out_val {
                        if let Some(&existing) = mapping.get(&in_val) {
                            if existing != out_val { consistent = false; break; }
                        } else {
                            mapping.insert(in_val, out_val);
                        }
                    }
                }
            }

            if consistent && !mapping.is_empty() {
                value_maps.insert(((*pred).clone(), val_pos), mapping);
            }
        }
    }

    GeneralContradiction { added, removed, value_maps, new_predicates, count_changes }
}

// ═══════════════════════════════════════════════════════════════════════
// STEP 3: PROPOSE — General rule templates from patterns
// ═══════════════════════════════════════════════════════════════════════

fn find_target_predicate(out_profile: &FactProfile, in_profile: &FactProfile) -> String {
    // Prefer predicates in output that don't exist in input
    let mut candidates: Vec<(&String, usize)> = out_profile.predicates.iter()
        .filter(|(pred, _)| !in_profile.predicates.contains_key(*pred))
        .map(|(pred, args)| (pred, args.len()))
        .collect();
    candidates.sort_by(|a, b| b.1.cmp(&a.1));
    if let Some((pred, _)) = candidates.first() {
        return (*pred).clone();
    }
    // Fallback: most-args predicate in output
    out_profile.predicates.iter()
        .max_by_key(|(_, args)| args.len())
        .map(|(pred, _)| pred.clone())
        .unwrap_or_else(|| "predict".into())
}

/// Compute confidence scaled by evidence count (Entropy principle).
/// More training pairs → higher confidence. Caps at 0.95.
fn evidence_confidence(base: f64, num_pairs: usize) -> f64 {
    // sqrt(N)/sqrt(3) scales so that 3 pairs ≈ base confidence,
    // 1 pair = ~58% of base, 5 pairs = ~129% of base
    // Floor at 0.20 so rules remain viable even with 1 training pair
    let factor = (num_pairs as f64).sqrt() / 3.0_f64.sqrt();
    (base * factor).clamp(0.20, 0.95)
}

/// Format a confidence pair for rule text.
fn conf(base: f64, num_pairs: usize) -> String {
    let c = evidence_confidence(base, num_pairs);
    format!("{c:.2}, {c:.2}")
}

fn propose_from_patterns(
    target: &str,
    in_profile: &FactProfile,
    out_profile: &FactProfile,
    contradictions: &[GeneralContradiction],
    derived_facts: &[Vec<StoredFact>],
    num_pairs: usize,
) -> Vec<Candidate> {
    let mut out = Vec::new();

    // Find source predicates — prefer SAME ARITY as target.
    // If input has (data id r c v) [arity 4] and (neighbor id r c r2 c2) [arity 5],
    // and target has arity 4, prefer arity-4 predicates for identity/remap rules.
    let new_preds: HashSet<&String> = contradictions.iter()
        .flat_map(|c| c.new_predicates.iter())
        .collect();
    let target_arity = out_profile.predicates.get(target).map(|a| a.len()).unwrap_or(0);
    let non_new: Vec<(&String, &Vec<ArgInfo>)> = in_profile.predicates.iter()
        .filter(|(pred, _)| !new_preds.contains(pred))
        .collect();

    // Collect ALL same-arity input predicates — each gets identity rules
    let same_arity_sources: Vec<&String> = non_new.iter()
        .filter(|(_, args)| args.len() == target_arity && target_arity > 0)
        .map(|(pred, _)| *pred)
        .collect();

    // Primary source: same arity preferred, then highest arity as fallback
    let source_pred = {
        // First: same arity as target, prefer more unique values (data-rich)
        let same_arity = non_new.iter()
            .filter(|(_, args)| args.len() == target_arity && target_arity > 0)
            .max_by_key(|(_, args)| args.iter().map(|a| a.ints.len() + a.syms.len()).sum::<usize>())
            .map(|(pred, _)| (*pred).clone());
        // Fallback: highest arity
        same_arity.or_else(|| {
            non_new.iter()
                .max_by_key(|(_, args)| args.len())
                .map(|(pred, _)| (*pred).clone())
        }).or_else(|| {
            in_profile.predicates.iter()
                .max_by_key(|(_, args)| args.len())
                .map(|(pred, _)| pred.clone())
        })
    };

    let source = match &source_pred {
        Some(s) => s.as_str(),
        None => return out,
    };
    let source_args = match in_profile.predicates.get(source) {
        Some(a) => a,
        None => return out,
    };
    let arity = source_args.len();

    let vars: Vec<String> = (0..arity).map(|i| format!("$a{i}")).collect();
    let vars_str = vars.join(" ");

    // ── Pattern 1: IDENTITY — try ALL same-arity predicates as source ──
    let np = num_pairs;
    out.push(mk(
        &format!("({target} {vars_str}) if\n    ({source} {vars_str}) <{}>", conf(0.50, np)),
        "identity",
    ));
    // Also try other same-arity predicates (scoring picks the right one)
    for alt_source in &same_arity_sources {
        if **alt_source != source {
            let alt_vars: Vec<String> = (0..target_arity).map(|i| format!("$a{i}")).collect();
            let alt_vars_str = alt_vars.join(" ");
            out.push(mk(
                &format!("({target} {alt_vars_str}) if\n    ({alt_source} {alt_vars_str}) <{}>", conf(0.45, np)),
                &format!("identity-alt-{alt_source}"),
            ));
        }
    }

    // ── Pattern 2: VALUE REMAP — from discovered value_maps ──
    // Merge maps, then validate: only keep mappings consistent across ALL pairs
    let mut merged_maps: HashMap<(String, usize), HashMap<i64, i64>> = HashMap::new();
    for c in contradictions {
        for ((pred, pos), mapping) in &c.value_maps {
            let entry = merged_maps.entry((pred.clone(), *pos)).or_default();
            for (&from, &to) in mapping { entry.insert(from, to); }
        }
    }
    // Cross-pair consistency check: a mapping is only valid if every pair
    // that has this predicate agrees on it (Noether: symmetry → conservation)
    merged_maps.retain(|key, mapping| {
        mapping.retain(|&from, to| {
            let to_val = *to;
            contradictions.iter().all(|c| {
                match c.value_maps.get(key) {
                    Some(m) => m.get(&from).map(|&v| v == to_val).unwrap_or(true),
                    None => true, // predicate absent in this pair — skip
                }
            })
        });
        !mapping.is_empty()
    });

    for ((pred, pos), mapping) in &merged_maps {
        if pred != source { continue; }
        if *pos == 0 || *pos > arity { continue; }
        let idx = pos - 1;

        // Only apply remaps on categorical or value positions, skip IDs
        if source_args[idx].is_id { continue; }

        for (&from, &to) in mapping {
            let mut ta = vars.clone(); ta[idx] = to.to_string();
            let mut sa = vars.clone(); sa[idx] = from.to_string();
            out.push(mk(
                &format!("({target} {}) if\n    ({source} {}) <{}>",
                    ta.join(" "), sa.join(" "), conf(0.90, np)),
                &format!("remap-pos{pos}-{from}to{to}"),
            ));
        }

        // Complete remap + identity fallback
        if !mapping.is_empty() {
            let mut combined = String::new();
            for (&from, &to) in mapping {
                let mut ta = vars.clone(); ta[idx] = to.to_string();
                let mut sa = vars.clone(); sa[idx] = from.to_string();
                combined.push_str(&format!(
                    "({target} {}) if\n    ({source} {}) <{}>\n",
                    ta.join(" "), sa.join(" "), conf(0.95, np)));
            }
            combined.push_str(&format!(
                "({target} {vars_str}) if\n    ({source} {vars_str}) <{}>", conf(0.40, np)));
            out.push(mk(&combined, "remap-complete"));
        }
    }

    // ── Pattern 3: POSITIONAL TRANSFORMS on sequential positions ──
    if arity >= 3 {
        let seq_positions: Vec<usize> = source_args.iter().enumerate()
            .filter(|(_, a)| a.is_sequential)
            .map(|(i, _)| i)
            .collect();

        // Swap pairs (transpose)
        for i in 0..seq_positions.len() {
            for j in (i + 1)..seq_positions.len() {
                let (pi, pj) = (seq_positions[i], seq_positions[j]);
                let mut swapped = vars.clone();
                swapped.swap(pi, pj);
                out.push(mk(
                    &format!("({target} {}) if\n    ({source} {vars_str}) <{}>",
                        swapped.join(" "), conf(0.80, np)),
                    &format!("swap-pos{pi}-pos{pj}"),
                ));
            }
        }

        // Shift by small constants
        for &pos in &seq_positions {
            for shift in [-2i64, -1, 1, 2] {
                let sv = format!("$shifted{pos}");
                let op = if shift > 0 { "+" } else { "-" };
                let abs = shift.unsigned_abs();
                let mut ta = vars.clone(); ta[pos] = sv.clone();
                out.push(mk(
                    &format!("({target} {}) if\n    ({source} {vars_str})\n    ({op} {} {abs} {sv}) <{}>",
                        ta.join(" "), vars[pos], conf(0.75, np)),
                    &format!("shift-pos{pos}-by{shift}"),
                ));
            }
        }

        // Reverse (max - pos)
        for &pos in &seq_positions {
            let max_val = source_args[pos].ints.iter().max().copied().unwrap_or(0);
            if max_val > 0 {
                let rv = format!("$rev{pos}");
                let mut ta = vars.clone(); ta[pos] = rv.clone();
                out.push(mk(
                    &format!("({target} {}) if\n    ({source} {vars_str})\n    (- {max_val} {} {rv}) <{}>",
                        ta.join(" "), vars[pos], conf(0.80, np)),
                    &format!("reverse-pos{pos}-max{max_val}"),
                ));
            }
        }

        // Modular (tiling)
        for &pos in &seq_positions {
            let max_val = source_args[pos].ints.iter().max().copied().unwrap_or(0);
            if max_val > 0 {
                let mv = format!("$mod{pos}");
                let dim = max_val + 1;
                let mut lookup = vars.clone(); lookup[pos] = mv.clone();
                out.push(mk(
                    &format!("({target} {vars_str}) if\n    (% {} {dim} {mv})\n    ({source} {}) <{}>",
                        vars[pos], lookup.join(" "), conf(0.80, np)),
                    &format!("modular-pos{pos}-mod{dim}"),
                ));
            }
        }

        // Categorical substitution — only for positions WITHOUT evidence-backed
        // remaps (Pattern 2 already handles those). This prevents O(n^2) explosion
        // on positions where we already know the correct mapping.
        let evidence_positions: HashSet<usize> = merged_maps.iter()
            .filter(|((pred, _), _)| pred == source)
            .map(|((_, pos), _)| pos - 1) // value_maps uses 1-based positions
            .collect();

        for (i, arg) in source_args.iter().enumerate() {
            if !arg.is_categorical { continue; }
            if evidence_positions.contains(&i) { continue; } // already handled by Pattern 2
            // Integer categorical values
            let cat_ints: Vec<i64> = arg.ints.iter().copied().collect();
            for &from in &cat_ints {
                for &to in &cat_ints {
                    if from == to { continue; }
                    let mut ta = vars.clone(); ta[i] = to.to_string();
                    let mut sa = vars.clone(); sa[i] = from.to_string();
                    out.push(mk(
                        &format!("({target} {}) if\n    ({source} {}) <{}>",
                            ta.join(" "), sa.join(" "), conf(0.70, np)),
                        &format!("cat-sub-pos{i}-{from}to{to}"),
                    ));
                }
                if out.len() > 200 { break; }
            }
            // Symbol categorical values
            let cat_syms: Vec<&String> = arg.syms.iter().collect();
            for from in &cat_syms {
                for to in &cat_syms {
                    if from == to { continue; }
                    let mut ta = vars.clone(); ta[i] = (*to).clone();
                    let mut sa = vars.clone(); sa[i] = (*from).clone();
                    out.push(mk(
                        &format!("({target} {}) if\n    ({source} {}) <{}>",
                            ta.join(" "), sa.join(" "), conf(0.70, np)),
                        &format!("cat-sub-pos{i}-{from}to{to}"),
                    ));
                }
                if out.len() > 200 { break; }
            }
        }
    }

    // ── Pattern 4: DERIVED-FACT CORRELATION ──
    if !derived_facts.is_empty() {
        let derived_preds: HashSet<String> = derived_facts.iter()
            .flat_map(|facts| facts.iter())
            .filter_map(|f| {
                if let Some(Neuron::Symbol(s)) = f.parts.first() {
                    if !in_profile.predicates.contains_key(s) && s != target {
                        Some(s.clone())
                    } else { None }
                } else { None }
            }).collect();

        for dpred in &derived_preds {
            let sample = derived_facts.iter()
                .flat_map(|facts| facts.iter())
                .find(|f| f.parts.first() == Some(&Neuron::Symbol(dpred.clone())));
            if let Some(sf) = sample {
                let d_arity = sf.parts.len() - 1;

                // Skip derived predicates that can't bind enough head variables.
                // Single-condition rules need d_arity >= target_arity (== vars.len())
                // to bind ALL head variables. Anything less produces unbound vars
                // in the head → garbage proposals.
                if d_arity < vars.len() {
                    continue;
                }

                let mut dvars: Vec<String> = (0..d_arity).map(|i| format!("$d{i}")).collect();
                // Share ALL corresponding positions to maximize binding
                let shared = vars.len().min(dvars.len());
                for i in 0..shared {
                    dvars[i] = vars[i].clone();
                }
                out.push(mk(
                    &format!("({target} {vars_str}) if\n    ({dpred} {}) <{}>",
                        dvars.join(" "), conf(0.70, np)),
                    &format!("derived-{dpred}"),
                ));
            }
        }
    }

    // ── Pattern 5: COUNT CHANGES — duplication/reduction ──
    for c in contradictions.iter().take(1) {
        for (pred, (in_c, out_c)) in &c.count_changes {
            if pred == source && *out_c > *in_c && *in_c > 0 {
                let factor = out_c / in_c;
                if *out_c == factor * in_c {
                    out.push(mk(
                        &format!("({target} {vars_str}) if\n    ({source} {vars_str}) <{}>", conf(0.70, np)),
                        &format!("expand-factor{factor}"),
                    ));
                }
            }
        }
    }

    // ── Pattern 6: ADDED FACTS — generate rules that produce specific output facts ──
    // Use out_profile to see what symbols appear in output
    if let Some(out_args) = out_profile.predicates.get(target) {
        // For each categorical position in output, find the most common value
        // and generate a constant-fill rule
        for (i, arg) in out_args.iter().enumerate() {
            if !arg.is_categorical || i >= vars.len() { continue; }
            if let Some(&common_val) = arg.ints.iter().max() {
                let mut ta = vars.clone(); ta[i] = common_val.to_string();
                out.push(mk(
                    &format!("({target} {}) if\n    ({source} {vars_str}) <{}>",
                        ta.join(" "), conf(0.55, np)),
                    &format!("fill-pos{i}-val{common_val}"),
                ));
            }
        }
    }

    // ── Pattern 7: REMOVED FACTS — elimination rules ──
    for c in contradictions.iter().take(1) {
        if !c.removed.is_empty() && c.added.is_empty() {
            // Facts were removed but nothing added — try "keep only foreground"
            // by adding a guard that some value is non-default
            for (i, arg) in source_args.iter().enumerate() {
                if !arg.is_categorical { continue; }
                if let Some(&min_val) = arg.ints.iter().min() {
                    // Keep facts where this position != min_val (likely background)
                    out.push(mk(
                        &format!("({target} {vars_str}) if\n    ({source} {vars_str})\n    (!= {} {min_val}) <{}>",
                            vars[i], conf(0.65, np)),
                        &format!("eliminate-pos{i}-neq{min_val}"),
                    ));
                }
            }
        }
    }

    out
}

/// Propose multi-body rules with 2+ positive conditions sharing variables.
///
/// DOMAIN AGNOSTIC — joins predicates that share an is_id position (likely
/// the same entity). Generates rules with guards (!= to prevent degenerate joins).
fn propose_multi_join(
    target: &str,
    in_profile: &FactProfile,
    out_profile: &FactProfile,
    contradictions: &[GeneralContradiction],
    derived_facts: &[Vec<StoredFact>],
    num_pairs: usize,
) -> Vec<Candidate> {
    let mut out = Vec::new();
    let target_arity = out_profile.predicates.get(target).map(|a| a.len()).unwrap_or(0);
    if target_arity == 0 { return out; }

    // Collect candidate predicates (same arity as target, from input)
    let new_preds: HashSet<&String> = contradictions.iter()
        .flat_map(|c| c.new_predicates.iter()).collect();
    let same_arity: Vec<(&String, &Vec<ArgInfo>)> = in_profile.predicates.iter()
        .filter(|(pred, args)| args.len() == target_arity && !new_preds.contains(pred))
        .collect();

    let np = num_pairs;

    // For each pair of predicates (P1, P2), generate join rules
    for (i, (pred1, args1)) in same_arity.iter().enumerate() {
        let vars_a: Vec<String> = (0..args1.len()).map(|k| format!("$a{k}")).collect();
        let vars_a_str = vars_a.join(" ");

        for (pred2, args2) in same_arity.iter().skip(i) {
            let vars_b: Vec<String> = (0..args2.len()).map(|k| format!("$b{k}")).collect();

            // Find shared id positions (both predicates have is_id at same index)
            let shared_ids: Vec<usize> = (0..args1.len().min(args2.len()))
                .filter(|&k| args1[k].is_id && args2[k].is_id)
                .collect();
            if shared_ids.is_empty() { continue; }

            // Build second condition with shared id variables
            let mut vars_b_joined: Vec<String> = vars_b.clone();
            for &sid in &shared_ids {
                vars_b_joined[sid] = vars_a[sid].clone(); // share the id variable
            }
            let vars_b_str = vars_b_joined.join(" ");

            // Self-join: same predicate with different positional vars
            if *pred1 == *pred2 {
                // Find non-id positions to differentiate
                for pos in 0..args1.len() {
                    if args1[pos].is_id { continue; }
                    // Rule: target from P1, require matching P2 row with != on pos
                    out.push(mk(
                        &format!("({target} {vars_a_str}) if\n    ({pred1} {vars_a_str})\n    ({pred2} {vars_b_str})\n    (!= {} {}) <{}>",
                            vars_a[pos], vars_b_joined[pos], conf(0.70, np)),
                        &format!("multi-join-self-neq-pos{pos}"),
                    ));
                    // Also try == on that position (same value, different cell)
                    if args1.len() > 2 {
                        for pos2 in (pos+1)..args1.len() {
                            if args1[pos2].is_id { continue; }
                            out.push(mk(
                                &format!("({target} {vars_a_str}) if\n    ({pred1} {vars_a_str})\n    ({pred2} {vars_b_str})\n    (== {} {}) (!= {} {}) <{}>",
                                    vars_a[pos], vars_b_joined[pos],
                                    vars_a[pos2], vars_b_joined[pos2],
                                    conf(0.65, np)),
                                &format!("multi-join-self-eq{pos}-neq{pos2}"),
                            ));
                        }
                    }
                }
            } else {
                // Cross-join: different predicates sharing id
                out.push(mk(
                    &format!("({target} {vars_a_str}) if\n    ({pred1} {vars_a_str})\n    ({pred2} {vars_b_str}) <{}>",
                        conf(0.70, np)),
                    &format!("multi-join-{pred1}-{pred2}"),
                ));
                // With != guard on last position (value position)
                let last = args1.len() - 1;
                if !args1[last].is_id && last < vars_b_joined.len() {
                    out.push(mk(
                        &format!("({target} {vars_a_str}) if\n    ({pred1} {vars_a_str})\n    ({pred2} {vars_b_str})\n    (!= {} {}) <{}>",
                            vars_a[last], vars_b_joined[last], conf(0.65, np)),
                        &format!("multi-join-{pred1}-{pred2}-neq"),
                    ));
                }
            }

            if out.len() >= 50 { break; }
        }
        if out.len() >= 50 { break; }
    }

    // Also try joining input predicates with derived-fact predicates
    // (DNA-produced predicates that might carry extra info)
    if !derived_facts.is_empty() {
        let mut derived_preds: HashMap<String, usize> = HashMap::new();
        for df_set in derived_facts {
            for sf in df_set {
                if let Some(Neuron::Symbol(pred)) = sf.parts.first() {
                    derived_preds.entry(pred.clone()).or_insert(sf.parts.len());
                }
            }
        }
        // Remove predicates already in input profile or that ARE the target
        derived_preds.retain(|pred, _| {
            !in_profile.predicates.contains_key(pred) && pred != target
        });

        for (pred1, args1) in &same_arity {
            let vars_a: Vec<String> = (0..args1.len()).map(|k| format!("$a{k}")).collect();
            let vars_a_str = vars_a.join(" ");

            for (dpred, darity) in &derived_preds {
                // Skip derived predicates with too few args to constrain the target
                if *darity < 2 || (target_arity > 2 && *darity < target_arity.saturating_sub(1)) {
                    continue;
                }
                // Build derived condition — share first variable (likely id)
                let mut dargs: Vec<String> = (0..*darity).map(|k| {
                    if k == 0 { vars_a[0].clone() } // share id
                    else { format!("$d{k}") }
                }).collect();
                // If same arity, share more variables
                if *darity == target_arity {
                    for k in 0..target_arity.min(dargs.len()) {
                        dargs[k] = vars_a[k].clone();
                    }
                }
                let dargs_str = dargs.join(" ");
                out.push(mk(
                    &format!("({target} {vars_a_str}) if\n    ({pred1} {vars_a_str})\n    ({dpred} {dargs_str}) <{}>",
                        conf(0.65, np)),
                    &format!("multi-join-derived-{dpred}"),
                ));
                if out.len() >= 60 { break; }
            }
            if out.len() >= 60 { break; }
        }
    }

    out
}

/// Propose rules with arithmetic operations (shift, reflect, difference).
///
/// DOMAIN AGNOSTIC — detects sequential positions in fact profile and
/// generates rules with arithmetic conditions for offset/reflection.
fn propose_arithmetic_combos(
    target: &str,
    in_profile: &FactProfile,
    contradictions: &[GeneralContradiction],
    num_pairs: usize,
) -> Vec<Candidate> {
    let mut out = Vec::new();
    let target_arity = in_profile.predicates.iter()
        .filter(|(_, args)| args.len() > 0)
        .max_by_key(|(_, args)| args.len())
        .map(|(_, args)| args.len())
        .unwrap_or(0);

    let source_info = extract_source(in_profile, contradictions, target_arity);
    let (source, source_args, arity, vars, vars_str) = match source_info {
        Some(s) => s,
        None => return out,
    };

    let np = num_pairs;

    // Find sequential positions (row, col indices)
    let seq_positions: Vec<usize> = source_args.iter().enumerate()
        .filter(|(_, a)| a.is_sequential)
        .map(|(i, _)| i)
        .collect();

    // Shift rules: offset a sequential position by K
    for &pos in &seq_positions {
        for k in &[1i64, 2, -1, -2] {
            let shifted_var = format!("$shifted{pos}");
            let mut target_vars = vars.clone();
            target_vars[pos] = shifted_var.clone();
            out.push(mk(
                &format!("({target} {}) if\n    ({source} {vars_str})\n    (+ {} {k} {shifted_var}) <{}>",
                    target_vars.join(" "), vars[pos], conf(0.70, np)),
                &format!("arith-shift-pos{pos}-k{k}"),
            ));
        }
    }

    // Reflect rules: mirror a position around grid dimension
    for &pos in &seq_positions {
        let refl_var = format!("$refl{pos}");
        let max_var = format!("$max{pos}");
        let mut target_vars = vars.clone();
        target_vars[pos] = refl_var.clone();

        // Requires grid-size for bounds — use a dimension variable
        // If arity >= 4 (likely grid-cell id r c v), positions 1,2 are r,c
        if arity >= 4 && (pos == 1 || pos == 2) {
            let size_dim = if pos == 1 { "$rows" } else { "$cols" };
            out.push(mk(
                &format!("({target} {}) if\n    ({source} {vars_str})\n    (grid-size {} {size_dim} $cols2)\n    (- {size_dim} 1 {max_var})\n    (- {max_var} {} {refl_var}) <{}>",
                    target_vars.join(" "), vars[0], vars[pos], conf(0.70, np)),
                &format!("arith-reflect-pos{pos}"),
            ));
        }
    }

    // Difference between two joined cells (for multi-body arithmetic)
    if seq_positions.len() >= 2 {
        let p1 = seq_positions[0];
        let p2 = seq_positions[1];
        let diff_var = "$diff";
        out.push(mk(
            &format!("({target} {vars_str}) if\n    ({source} {vars_str})\n    ({source} $b0 $b1 $b2 $b3)\n    (- {} {} {diff_var})\n    (== $a0 $b0) <{}>",
                vars[p1], format!("$b{p1}"), conf(0.60, np)),
            &format!("arith-diff-pos{p1}-{p2}"),
        ));
    }

    // Cap output
    out.truncate(30);
    out
}

/// Propose rules with aggregate conditions (count/frequency filters).
///
/// DOMAIN AGNOSTIC — detects count changes between input/output and proposes
/// frequency-based filtering rules.
fn propose_aggregate_conditions(
    target: &str,
    in_profile: &FactProfile,
    contradictions: &[GeneralContradiction],
    num_pairs: usize,
) -> Vec<Candidate> {
    let mut out = Vec::new();
    let target_arity = in_profile.predicates.iter()
        .filter(|(_, args)| args.len() > 0)
        .max_by_key(|(_, args)| args.len())
        .map(|(_, args)| args.len())
        .unwrap_or(0);

    let source_info = extract_source(in_profile, contradictions, target_arity);
    let (source, source_args, arity, vars, vars_str) = match source_info {
        Some(s) => s,
        None => return out,
    };

    let np = num_pairs;

    // Check if output has fewer facts than input (filtering)
    let is_filtering = contradictions.iter().any(|c| {
        c.count_changes.values().any(|(in_count, out_count)| *out_count < *in_count)
    });

    // Find categorical positions (likely color/value columns)
    let cat_positions: Vec<usize> = source_args.iter().enumerate()
        .filter(|(_, a)| a.is_categorical && !a.is_id)
        .map(|(i, _)| i)
        .collect();

    // Frequency filter: count occurrences of a categorical value, keep based on threshold
    for &pos in &cat_positions {
        // Build wildcard pattern for count
        let count_pattern: Vec<String> = (0..arity).map(|k| {
            if k == 0 { vars[0].clone() } // share id
            else if k == pos { vars[pos].clone() } // bind the value we count
            else { "$_".into() } // wildcard
        }).collect();
        let count_pattern_str = count_pattern.join(" ");

        for threshold in &[1i64, 2, 3] {
            // Keep values that appear MORE than threshold times
            out.push(mk(
                &format!("({target} {vars_str}) if\n    ({source} {vars_str})\n    (count ({source} {count_pattern_str}) {} -> $n)\n    (> $n {threshold}) <{}>",
                    vars[pos], conf(0.70, np)),
                &format!("agg-count-pos{pos}-gt{threshold}"),
            ));
            // Keep values that appear EXACTLY threshold times (for unique)
            if *threshold <= 2 {
                out.push(mk(
                    &format!("({target} {vars_str}) if\n    ({source} {vars_str})\n    (count ({source} {count_pattern_str}) {} -> $n)\n    (== $n {threshold}) <{}>",
                        vars[pos], conf(0.65, np)),
                    &format!("agg-count-pos{pos}-eq{threshold}"),
                ));
            }
        }
    }

    // If filtering detected, also propose "keep majority" / "remove minority" rules
    if is_filtering && !cat_positions.is_empty() {
        let pos = *cat_positions.last().unwrap(); // usually the value position
        let count_pattern: Vec<String> = (0..arity).map(|k| {
            if k == 0 { vars[0].clone() }
            else if k == pos { vars[pos].clone() }
            else { "$_".into() }
        }).collect();
        let count_pattern_str = count_pattern.join(" ");

        // Keep value if it appears more than once (non-unique filter)
        out.push(mk(
            &format!("({target} {vars_str}) if\n    ({source} {vars_str})\n    (count ({source} {count_pattern_str}) {} -> $n)\n    (> $n 1) <{}>",
                vars[pos], conf(0.75, np)),
            "agg-keep-non-unique",
        ));
    }

    out.truncate(20);
    out
}

/// Propose conditional split rules — one rule per categorical value.
///
/// DOMAIN AGNOSTIC — when value_maps show different mappings per category,
/// generates separate rules for each value.
fn propose_conditional_split(
    target: &str,
    in_profile: &FactProfile,
    contradictions: &[GeneralContradiction],
    num_pairs: usize,
) -> Vec<Candidate> {
    let mut out = Vec::new();
    let target_arity = in_profile.predicates.iter()
        .filter(|(_, args)| args.len() > 0)
        .max_by_key(|(_, args)| args.len())
        .map(|(_, args)| args.len())
        .unwrap_or(0);

    let source_info = extract_source(in_profile, contradictions, target_arity);
    let (source, source_args, arity, vars, vars_str) = match source_info {
        Some(s) => s,
        None => return out,
    };

    let np = num_pairs;

    // Find categorical positions with value_maps showing different mappings
    let merged_maps = merge_value_maps(contradictions, &source, arity, source_args);
    for ((pred, pos), mapping) in &merged_maps {
        if pred != &source || mapping.len() < 2 { continue; }
        // Generate one rule per mapping entry
        let mut rules_text = Vec::new();
        for (&from, &to) in mapping {
            let mut target_vars = vars.clone();
            let src_pos = pos.wrapping_sub(1).min(arity - 1);
            if src_pos < target_vars.len() {
                target_vars[src_pos] = to.to_string();
            }
            let mut src_vars = vars.clone();
            if src_pos < src_vars.len() {
                src_vars[src_pos] = from.to_string();
            }
            rules_text.push(format!(
                "({target} {}) if\n    ({source} {}) <{}>",
                target_vars.join(" "), src_vars.join(" "), conf(0.80, np)
            ));
        }
        // Also add a default fallback with lower confidence
        rules_text.push(format!(
            "({target} {vars_str}) if\n    ({source} {vars_str}) <{}>",
            conf(0.40, np)
        ));
        // Combine into multi-rule candidate
        let combined = rules_text.join("\n");
        out.push(mk(&combined, &format!("cond-split-pos{pos}")));
    }

    out.truncate(15);
    out
}

/// Propose spatial transformation rules when input/output have different dimensions.
///
/// DOMAIN AGNOSTIC — detects size differences from sequential position ranges,
/// then proposes scaling (division), tiling (modulo), and conditional tiling
/// (the two-lookup pattern seen in puzzles like 007bbfb7).
fn propose_spatial(
    target: &str,
    in_profile: &FactProfile,
    out_profile: &FactProfile,
    contradictions: &[GeneralContradiction],
    num_pairs: usize,
) -> Vec<Candidate> {
    let mut out = Vec::new();

    let target_arity = out_profile.predicates.get(target).map(|a| a.len()).unwrap_or(0);

    // For spatial proposals, we need the source predicate whose sequential positions
    // ALIGN with the target's sequential positions (same indices). This ensures that
    // position comparison is valid (e.g., source pos1=row matches target pos1=row).
    let new_preds: HashSet<&String> = contradictions.iter()
        .flat_map(|c| c.new_predicates.iter())
        .collect();
    let target_seq: Vec<usize> = out_profile.predicates.get(target)
        .map(|args| args.iter().enumerate()
            .filter(|(_, a)| a.is_sequential)
            .map(|(i, _)| i)
            .collect())
        .unwrap_or_default();

    let spatial_source = in_profile.predicates.iter()
        .filter(|(pred, args)| {
            args.len() == target_arity && target_arity >= 3
                && !new_preds.contains(pred)
        })
        .max_by_key(|(_, args)| {
            // Count how many sequential positions MATCH the target's sequential positions
            let matching_seq = args.iter().enumerate()
                .filter(|(i, a)| a.is_sequential && target_seq.contains(i))
                .count();
            // Primary sort: matching sequential positions; secondary: total sequential
            (matching_seq * 100, args.iter().filter(|a| a.is_sequential).count())
        })
        .map(|(pred, args)| {
            let arity = args.len();
            let vars: Vec<String> = (0..arity).map(|i| format!("$a{i}")).collect();
            let vars_str = vars.join(" ");
            (pred.clone(), args.as_slice(), arity, vars, vars_str)
        });

    let (source, source_args, arity, vars, vars_str) = match spatial_source {
        Some(s) => s,
        None => {
            eprintln!("        spatial-debug: no source with >=2 seq positions for arity={}", target_arity);
            return out;
        }
    };
    if arity < 3 {
        eprintln!("        spatial-debug: source={} arity={} < 3, skip", source, arity);
        return out;
    }

    let np = num_pairs;

    // Find sequential positions in source (e.g., row and col positions)
    let seq_positions: Vec<usize> = source_args.iter().enumerate()
        .filter(|(_, a)| a.is_sequential)
        .map(|(i, _)| i)
        .collect();
    eprintln!("        spatial-debug: source={} arity={} seq_positions={:?}", source, arity, seq_positions);
    if seq_positions.len() < 2 {
        eprintln!("        spatial-debug: need >=2 sequential positions, got {}", seq_positions.len());
        return out;
    }

    // Get output arg info
    let out_args = match out_profile.predicates.get(target) {
        Some(a) => a,
        None => {
            eprintln!("        spatial-debug: target {} not in out_profile", target);
            return out;
        }
    };

    // Detect size change per sequential dimension
    // Compare max values in input vs output for each sequential position
    let mut scale_factors: Vec<(usize, i64, i64)> = Vec::new(); // (pos, in_dim, factor)
    for &pos in &seq_positions {
        if pos >= source_args.len() || pos >= out_args.len() { continue; }
        let in_max = source_args[pos].ints.iter().max().copied().unwrap_or(0);
        let out_max = out_args[pos].ints.iter().max().copied().unwrap_or(0);
        eprintln!("        spatial-debug: pos{} in_max={} out_max={} in_dim={} out_dim={}",
            pos, in_max, out_max, in_max + 1, out_max + 1);
        let in_dim = in_max + 1;
        let out_dim = out_max + 1;
        if out_dim > in_dim && in_dim > 0 {
            let factor = out_dim / in_dim;
            if factor * in_dim == out_dim { // exact integer ratio
                scale_factors.push((pos, in_dim, factor));
            }
        }
    }

    if scale_factors.is_empty() {
        // No size change — try object offset proposals instead
        return propose_offsets(target, &source, source_args, &vars, &vars_str, &seq_positions, np);
    }

    eprintln!("      propose_spatial: scale_factors=[{}]",
        scale_factors.iter().map(|(p, d, f)| format!("pos{}:dim{}→{}x", p, d, f)).collect::<Vec<String>>().join(", "));

    // CRITICAL: In forward chaining, fact-match conditions MUST come first
    // to bind variables. Arithmetic conditions can only evaluate when their
    // inputs are already bound. So rules must start with (source ...) match,
    // then compute derived positions with arithmetic.
    //
    // Key technique: use a SELF-JOIN on the source predicate as an iterator.
    // (source $id $tr $tc $iter_val) iterates over all grid positions,
    // providing tile indices or block positions. Then arithmetic computes
    // the output position from source_pos * factor + offset.

    let rp = seq_positions[0]; // row position in source args
    let cp = seq_positions[1]; // col position in source args
    let vp = arity - 1; // value position (typically last)

    // Get scale info for the two key sequential positions
    let sf_r = scale_factors.iter().find(|&&(pos, _, _)| pos == rp);
    let sf_c = scale_factors.iter().find(|&&(pos, _, _)| pos == cp);

    if let (Some(&(_, in_dim_r, factor_r)), Some(&(_, in_dim_c, factor_c))) = (sf_r, sf_c) {
        // Both dimensions scale — full spatial proposals

        // ── TYPE 1: SCALING — each input cell → factor×factor block of same color ──
        // Self-join: iterate (dr, dc) offsets using source positions as iterators
        // Rule: for each source cell (sr,sc,v) and each (dr,dc) in 0..factor:
        //   predict-cell at (sr*factor+dr, sc*factor+dc) = v
        {
            let mut src1 = vars.clone(); // source cell: binds sr, sc, v
            src1[rp] = "$sr".into();
            src1[cp] = "$sc".into();

            let mut iter = vars.clone(); // iterator: binds dr, dc
            iter[rp] = "$dr".into();
            iter[cp] = "$dc".into();
            // iter[vp] can be anything, we don't use it
            iter[vp] = "$_iv".into();

            out.push(mk(
                &format!(
                    "({target} {vars_str}) if\n    ({source} {})\n    ({source} {})\n    (* $sr {factor_r} $br)\n    (+ $br $dr {})\n    (* $sc {factor_c} $bc)\n    (+ $bc $dc {}) <{}>",
                    src1.join(" "), iter.join(" "),
                    vars[rp], vars[cp],
                    conf(0.80, np)),
                "scale-multi",
            ));
        }

        // ── TYPE 2: TILING — output = input repeated factor times in each dim ──
        // Self-join: iterate (tr, tc) tile positions using source positions
        // Rule: for each source cell (lr,lc,v) and each tile pos (tr,tc):
        //   predict-cell at (tr*in_dim+lr, tc*in_dim+lc) = v
        {
            let mut src1 = vars.clone();
            src1[rp] = "$lr".into();
            src1[cp] = "$lc".into();

            let mut iter = vars.clone();
            iter[rp] = "$tr".into();
            iter[cp] = "$tc".into();
            iter[vp] = "$_tv".into();

            out.push(mk(
                &format!(
                    "({target} {vars_str}) if\n    ({source} {})\n    ({source} {})\n    (* $tr {in_dim_r} $base_r)\n    (+ $base_r $lr {})\n    (* $tc {in_dim_c} $base_c)\n    (+ $base_c $lc {}) <{}>",
                    src1.join(" "), iter.join(" "),
                    vars[rp], vars[cp],
                    conf(0.80, np)),
                "tile-multi",
            ));
        }

        // ── TYPE 3: CONDITIONAL TILING (007bbfb7 pattern) ──
        // Two self-joins on source:
        //   1. Block lookup: (source $id $br $bc $bv) with !=$bv 0
        //   2. Local cell: (source $id $lr $lc $v) — the paste value
        // Output at (br*in_dim+lr, bc*in_dim+lc) = v
        {
            let mut block = vars.clone();
            block[rp] = "$br".into();
            block[cp] = "$bc".into();
            block[vp] = "$bv".into();

            let mut local = vars.clone();
            local[rp] = "$lr".into();
            local[cp] = "$lc".into();
            // local[vp] stays as original var (the output value)

            // Non-zero block: paste input grid
            let nz_rule = format!(
                "({target} {vars_str}) if\n    ({source} {}) (!= $bv 0)\n    ({source} {})\n    (* $br {in_dim_r} $base_r)\n    (+ $base_r $lr {})\n    (* $bc {in_dim_c} $base_c)\n    (+ $base_c $lc {}) <{}>",
                block.join(" "), local.join(" "),
                vars[rp], vars[cp],
                conf(0.85, np)
            );

            // Zero block: fill with zeros
            let mut z_block = vars.clone();
            z_block[rp] = "$br".into();
            z_block[cp] = "$bc".into();
            z_block[vp] = "0".into();

            let mut z_local = vars.clone();
            z_local[rp] = "$lr".into();
            z_local[cp] = "$lc".into();
            z_local[vp] = "$_zv".into();

            let mut z_head = vars.clone();
            z_head[vp] = "0".into();

            let z_rule = format!(
                "({target} {}) if\n    ({source} {})\n    ({source} {})\n    (* $br {in_dim_r} $base_r)\n    (+ $base_r $lr {})\n    (* $bc {in_dim_c} $base_c)\n    (+ $base_c $lc {}) <{}>",
                z_head.join(" "),
                z_block.join(" "), z_local.join(" "),
                vars[rp], vars[cp],
                conf(0.85, np)
            );

            out.push(mk(&format!("{nz_rule}\n{z_rule}"), "cond-tile"));

            // Also: block value fill — non-zero blocks filled with the block's value
            let mut bv_head = vars.clone();
            bv_head[vp] = "$bv".into();

            let bv_rule = format!(
                "({target} {}) if\n    ({source} {}) (!= $bv 0)\n    ({source} {})\n    (* $br {in_dim_r} $base_r)\n    (+ $base_r $lr {})\n    (* $bc {in_dim_c} $base_c)\n    (+ $base_c $lc {}) <{}>",
                bv_head.join(" "),
                block.join(" "), z_local.join(" "),
                vars[rp], vars[cp],
                conf(0.80, np)
            );

            out.push(mk(
                &format!("{bv_rule}\n{z_rule}"),
                "cond-tile-blockval",
            ));
        }
    }

    // ── SINGLE-DIM proposals: if only one dimension has a scale factor ──
    for &(pos, in_dim, factor) in &scale_factors {
        // Scale in one dim: source cell at sp is expanded to factor cells
        let sv = format!("$sp");
        let dv = format!("$dp");
        let mut src1 = vars.clone();
        src1[pos] = sv.clone();
        let mut iter = vars.clone();
        iter[pos] = dv.clone();
        iter[vp] = "$_dv".into();

        out.push(mk(
            &format!(
                "({target} {vars_str}) if\n    ({source} {})\n    ({source} {})\n    (* {sv} {factor} $base)\n    (+ $base {dv} {}) <{}>",
                src1.join(" "), iter.join(" "),
                vars[pos], conf(0.75, np)),
            &format!("scale-pos{pos}-{factor}x"),
        ));

        // Tile in one dim: source wraps around at in_dim
        let mut src2 = vars.clone();
        src2[pos] = "$lp".into();
        let mut iter2 = vars.clone();
        iter2[pos] = "$tp".into();
        iter2[vp] = "$_tv2".into();

        out.push(mk(
            &format!(
                "({target} {vars_str}) if\n    ({source} {})\n    ({source} {})\n    (* $tp {in_dim} $base2)\n    (+ $base2 $lp {}) <{}>",
                src2.join(" "), iter2.join(" "),
                vars[pos], conf(0.75, np)),
            &format!("tile-pos{pos}-mod{in_dim}"),
        ));
    }

    out
}

/// Propose offset/shift rules when input and output have the same dimensions.
/// Tries small offsets in sequential positions to handle object movement.
fn propose_offsets(
    target: &str,
    source: &str,
    source_args: &[ArgInfo],
    vars: &[String],
    vars_str: &str,
    seq_positions: &[usize],
    np: usize,
) -> Vec<Candidate> {
    let mut out = Vec::new();
    if seq_positions.len() < 2 { return out; }

    let rp = seq_positions[0];
    let cp = seq_positions[1];
    let vp = vars.len() - 1;

    // Find the value position — skip if it's sequential (probably not the value)
    if source_args.get(vp).map(|a| a.is_sequential).unwrap_or(false) { return out; }

    // Multi-position offsets: shift by (dr, dc) for non-zero cells
    for &dr in &[-3i64, -2, -1, 1, 2, 3] {
        for &dc in &[-3i64, -2, -1, 0, 1, 2, 3] {
            let sr = format!("$or");
            let sc = format!("$oc");
            let mut src_vars = vars.to_vec();
            src_vars[rp] = sr.clone();
            src_vars[cp] = sc.clone();

            let op_r = if dr > 0 { "+" } else { "-" };
            let op_c = if dc > 0 { "+" } else { "-" };
            let abs_r = dr.unsigned_abs();
            let abs_c = dc.unsigned_abs();

            let mut body = format!("({source} {}) (!= {} 0)", src_vars.join(" "), src_vars[vp]);
            body.push_str(&format!("\n    ({op_r} {sr} {abs_r} {})", vars[rp]));
            if dc != 0 {
                body.push_str(&format!("\n    ({op_c} {sc} {abs_c} {})", vars[cp]));
            } else {
                // dc=0 means column stays same; just alias
                let mut src2 = src_vars.clone();
                src2[cp] = vars[cp].clone();
                body = format!("({source} {}) (!= {} 0)", src2.join(" "), src2[vp]);
                body.push_str(&format!("\n    ({op_r} {sr} {abs_r} {})", vars[rp]));
            }

            out.push(mk(
                &format!("({target} {vars_str}) if\n    {body} <{}>", conf(0.70, np)),
                &format!("offset-{dr}-{dc}"),
            ));
        }
        if out.len() > 30 { break; }
    }

    out
}

// ═══════════════════════════════════════════════════════════════════════
// OBSERVATION EXTRACTION — Read DNA-derived observation facts
// ═══════════════════════════════════════════════════════════════════════

/// Structured observations extracted from DNA-derived facts.
/// These guide genesis to propose TARGETED candidates instead of blind search.
#[derive(Default)]
struct Observations {
    // obs-consistent flags (cross-pair invariants from compute_observations)
    same_size: bool,
    output_larger: bool,
    output_smaller: bool,
    color_remap: bool,
    reflect_h: bool,
    reflect_v: bool,
    rotate_90: bool,
    rotate_180: bool,
    rotate_270: bool,
    transpose: bool,
    crop_to_bbox: bool,
    gravity_down: bool,
    row_fill: bool,
    col_fill: bool,
    cross_fill: bool,
    // New problem-class observations (from compute_observations in grid.rs)
    pure_addition: bool,
    pure_removal: bool,
    pure_recolor: bool,
    localized_change: bool,
    output_tiny: bool,
    output_is_subgrid: bool,
    output_binary: bool,
    position_dependent_recolor: bool,
    same_cell_count: bool,
    content_rearranged: bool,
    has_separator: bool,
    // Derived structure observations
    has_enclosed_regions: bool,
    fewer_objects: bool,
    new_values_introduced: bool,
    // obs-parameter values (specific values from observations)
    color_remap_pairs: Vec<(i64, i64)>, // (from, to)
    fill_color: Option<i64>,
    shift_dr: Option<i64>,
    shift_dc: Option<i64>,
    // obs-* value facts (cross-pair invariants with specific values)
    enclosed_fill_color: Option<i64>,         // from obs-enclosed-fill-color
    enclosed_size_fills: Vec<(i64, i64)>,     // from obs-enclosed-size-fill (area, color)
    separator_color: Option<i64>,             // from obs-separator-and
    // grid size analysis
    size_pairs: Vec<(i64, i64, i64, i64)>, // (in_rows, in_cols, out_rows, out_cols)
    // scale-ratio from Rust propose_spatial() or DNA rules
    scale_ratios: Vec<(i64, i64)>, // (row_factor, col_factor)
}

/// Extract structured observations from DNA-derived facts.
/// Reads obs-consistent, obs-parameter, scale-ratio, color-remap, grid-size, train-pair.
fn extract_observations(derived: &[Vec<StoredFact>]) -> Observations {
    let mut obs = Observations::default();
    if derived.is_empty() { return obs; }

    // Scan ALL pairs, not just the first — observations from any pair are valid
    for facts in derived {
    for fact in facts {
        let pred = match fact.parts.first() {
            Some(Neuron::Symbol(s)) => s.as_str(),
            _ => continue,
        };
        match pred {
            "obs-consistent" => {
                if let Some(Neuron::Symbol(what)) = fact.parts.get(1) {
                    match what.as_str() {
                        "same-size" => obs.same_size = true,
                        "output-larger" => obs.output_larger = true,
                        "output-smaller" => obs.output_smaller = true,
                        "color-remap" => obs.color_remap = true,
                        "reflect-h" => obs.reflect_h = true,
                        "reflect-v" => obs.reflect_v = true,
                        "rotate-90" => obs.rotate_90 = true,
                        "rotate-180" => obs.rotate_180 = true,
                        "rotate-270" => obs.rotate_270 = true,
                        "transpose" => obs.transpose = true,
                        "crop-to-bbox" => obs.crop_to_bbox = true,
                        "gravity-down" => obs.gravity_down = true,
                        "row-fill" => obs.row_fill = true,
                        "col-fill" => obs.col_fill = true,
                        "cross-fill" => obs.cross_fill = true,
                        "pure-addition" => obs.pure_addition = true,
                        "pure-removal" => obs.pure_removal = true,
                        "pure-recolor" => obs.pure_recolor = true,
                        "localized-change" => obs.localized_change = true,
                        "output-tiny" => obs.output_tiny = true,
                        "output-is-subgrid" => obs.output_is_subgrid = true,
                        "output-binary" => obs.output_binary = true,
                        "position-dependent-recolor" => obs.position_dependent_recolor = true,
                        "same-cell-count" => obs.same_cell_count = true,
                        "content-rearranged" => obs.content_rearranged = true,
                        "has-separator" => obs.has_separator = true,
                        "has-enclosed-regions" => obs.has_enclosed_regions = true,
                        "fewer-objects" => obs.fewer_objects = true,
                        "new-values-introduced" => obs.new_values_introduced = true,
                        _ => {}
                    }
                }
            }
            "obs-enclosed-fill-color" => {
                obs.enclosed_fill_color = fact.parts.get(1).and_then(neuron_to_i64);
            }
            "obs-enclosed-size-fill" => {
                if let (Some(area), Some(color)) = (
                    fact.parts.get(1).and_then(neuron_to_i64),
                    fact.parts.get(2).and_then(neuron_to_i64),
                ) {
                    if !obs.enclosed_size_fills.contains(&(area, color)) {
                        obs.enclosed_size_fills.push((area, color));
                    }
                }
            }
            "obs-separator-and" => {
                obs.separator_color = fact.parts.get(1).and_then(neuron_to_i64);
            }
            "obs-parameter" => {
                if let Some(Neuron::Symbol(param)) = fact.parts.get(1) {
                    match param.as_str() {
                        "color-remap" => {
                            if let (Some(from), Some(to)) = (
                                fact.parts.get(2).and_then(neuron_to_i64),
                                fact.parts.get(3).and_then(neuron_to_i64),
                            ) {
                                if from != to && !obs.color_remap_pairs.contains(&(from, to)) {
                                    obs.color_remap_pairs.push((from, to));
                                }
                            }
                        }
                        "shift-dr" => {
                            obs.shift_dr = fact.parts.get(2).and_then(neuron_to_i64);
                        }
                        "shift-dc" => {
                            obs.shift_dc = fact.parts.get(2).and_then(neuron_to_i64);
                        }
                        _ => {}
                    }
                }
            }
            "obs-fill-color" => {
                obs.fill_color = fact.parts.get(1).and_then(neuron_to_i64);
            }
            "scale-ratio" => {
                // (scale-ratio $in_id $rf $cf) — from DNA rules if available
                if let (Some(rf), Some(cf)) = (
                    fact.parts.get(2).and_then(neuron_to_i64),
                    fact.parts.get(3).and_then(neuron_to_i64),
                ) {
                    if !obs.scale_ratios.contains(&(rf, cf)) {
                        obs.scale_ratios.push((rf, cf));
                    }
                }
            }
            "color-remap" => {
                // (color-remap $from $to) — from rules.qor detection logic
                if let (Some(from), Some(to)) = (
                    fact.parts.get(2).and_then(neuron_to_i64),
                    fact.parts.get(3).and_then(neuron_to_i64),
                ) {
                    if from != to && !obs.color_remap_pairs.contains(&(from, to)) {
                        obs.color_remap_pairs.push((from, to));
                    }
                }
            }
            _ => {}
        }
    }
    } // end for facts in derived

    // Second pass: grid-size + train-pair → size_pairs (scan all pairs)
    let mut train_pairs: Vec<(String, String)> = Vec::new();
    let mut grid_sizes: HashMap<String, (i64, i64)> = HashMap::new();
    for facts in derived {
    for fact in facts {
        let pred = match fact.parts.first() {
            Some(Neuron::Symbol(s)) => s.as_str(),
            _ => continue,
        };
        match pred {
            "train-pair" => {
                if let (Some(Neuron::Symbol(inp)), Some(Neuron::Symbol(out))) =
                    (fact.parts.get(1), fact.parts.get(2))
                {
                    train_pairs.push((inp.clone(), out.clone()));
                }
            }
            "grid-size" => {
                if let Some(Neuron::Symbol(id)) = fact.parts.get(1) {
                    if let (Some(rows), Some(cols)) = (
                        fact.parts.get(2).and_then(neuron_to_i64),
                        fact.parts.get(3).and_then(neuron_to_i64),
                    ) {
                        grid_sizes.insert(id.clone(), (rows, cols));
                    }
                }
            }
            _ => {}
        }
    }
    } // end for facts in derived
    for (inp, outp) in &train_pairs {
        if let (Some(&(ir, ic)), Some(&(or, oc))) = (grid_sizes.get(inp), grid_sizes.get(outp)) {
            obs.size_pairs.push((ir, ic, or, oc));
        }
    }

    eprintln!("      genesis-obs: consistent=[{}] remap_pairs={} scale_ratios={:?} size_pairs={:?}",
        [
            if obs.same_size { "same-size" } else { "" },
            if obs.output_larger { "output-larger" } else { "" },
            if obs.color_remap { "color-remap" } else { "" },
            if obs.reflect_h { "reflect-h" } else { "" },
            if obs.reflect_v { "reflect-v" } else { "" },
            if obs.rotate_90 { "rotate-90" } else { "" },
            if obs.rotate_180 { "rotate-180" } else { "" },
            if obs.transpose { "transpose" } else { "" },
            if obs.crop_to_bbox { "crop-to-bbox" } else { "" },
            if obs.gravity_down { "gravity-down" } else { "" },
            if obs.row_fill { "row-fill" } else { "" },
            if obs.col_fill { "col-fill" } else { "" },
            if obs.cross_fill { "cross-fill" } else { "" },
            if obs.pure_addition { "pure-addition" } else { "" },
            if obs.pure_removal { "pure-removal" } else { "" },
            if obs.pure_recolor { "pure-recolor" } else { "" },
            if obs.localized_change { "localized-change" } else { "" },
            if obs.output_tiny { "output-tiny" } else { "" },
            if obs.output_is_subgrid { "output-is-subgrid" } else { "" },
            if obs.output_binary { "output-binary" } else { "" },
            if obs.position_dependent_recolor { "pos-dep-recolor" } else { "" },
            if obs.same_cell_count { "same-cell-count" } else { "" },
            if obs.content_rearranged { "content-rearranged" } else { "" },
            if obs.has_separator { "has-separator" } else { "" },
        ].iter().filter(|s| !s.is_empty()).cloned().collect::<Vec<_>>().join(","),
        obs.color_remap_pairs.len(),
        obs.scale_ratios,
        obs.size_pairs);

    obs
}

/// Propose targeted rules directly from observation facts.
///
/// Instead of blind search through transform space, observations tell us
/// EXACTLY which transform to propose. Vedic principle: informed search.
fn propose_from_observations(
    target: &str,
    in_profile: &FactProfile,
    out_profile: &FactProfile,
    contradictions: &[GeneralContradiction],
    obs: &Observations,
    num_pairs: usize,
) -> Vec<Candidate> {
    let target_arity = out_profile.predicates.get(target).map(|a| a.len()).unwrap_or(0);
    let source_info = extract_source(in_profile, contradictions, target_arity);
    let (source, source_args, arity, vars, vars_str) = match source_info {
        Some(s) => s,
        None => return Vec::new(),
    };

    let mut out = Vec::new();
    if arity < 3 { return out; }
    let np = num_pairs;
    let val_pos = arity - 1;

    let seq_positions: Vec<usize> = source_args.iter().enumerate()
        .filter(|(_, a)| a.is_sequential)
        .map(|(i, _)| i)
        .collect();

    // ── COLOR REMAP from observations (Vedic Sutra 2: complement/inverse) ──
    if !obs.color_remap_pairs.is_empty() {
        // Individual remap rules per (from, to) pair
        for &(from, to) in &obs.color_remap_pairs {
            let mut ta = vars.clone(); ta[val_pos] = to.to_string();
            let mut sa = vars.clone(); sa[val_pos] = from.to_string();
            out.push(mk(
                &format!("({target} {}) if\n    ({source} {}) <{}>",
                    ta.join(" "), sa.join(" "), conf(0.95, np)),
                &format!("obs-remap-{from}to{to}"),
            ));
        }

        // Complete remap: all mappings + identity fallback
        let mut combined = String::new();
        for &(from, to) in &obs.color_remap_pairs {
            let mut ta = vars.clone(); ta[val_pos] = to.to_string();
            let mut sa = vars.clone(); sa[val_pos] = from.to_string();
            combined.push_str(&format!(
                "({target} {}) if\n    ({source} {}) <{}>\n",
                ta.join(" "), sa.join(" "), conf(0.95, np)));
        }
        combined.push_str(&format!(
            "({target} {vars_str}) if\n    ({source} {vars_str}) <{}>",
            conf(0.40, np)));
        out.push(mk(&combined, "obs-remap-complete"));
    }

    // ── GEOMETRIC TRANSFORMS from obs-consistent ──
    if seq_positions.len() >= 2 {
        let rp = seq_positions[0];
        let cp = seq_positions[1];
        let max_r = source_args[rp].ints.iter().max().copied().unwrap_or(0);
        let max_c = source_args[cp].ints.iter().max().copied().unwrap_or(0);

        // Reflect-H: reverse rows
        if obs.reflect_h && max_r > 0 {
            let mut ta = vars.clone(); ta[rp] = "$obs_nr".into();
            out.push(mk(
                &format!("({target} {}) if\n    ({source} {vars_str})\n    (- {max_r} {} $obs_nr) <{}>",
                    ta.join(" "), vars[rp], conf(0.95, np)),
                "obs-reflect-h",
            ));
        }

        // Reflect-V: reverse cols
        if obs.reflect_v && max_c > 0 {
            let mut ta = vars.clone(); ta[cp] = "$obs_nc".into();
            out.push(mk(
                &format!("({target} {}) if\n    ({source} {vars_str})\n    (- {max_c} {} $obs_nc) <{}>",
                    ta.join(" "), vars[cp], conf(0.95, np)),
                "obs-reflect-v",
            ));
        }

        // Rotate-90: (r, c) → (c, max_r - r)
        if obs.rotate_90 && max_r > 0 {
            let mut ta = vars.clone();
            ta[rp] = vars[cp].clone();
            ta[cp] = "$obs_nc".into();
            out.push(mk(
                &format!("({target} {}) if\n    ({source} {vars_str})\n    (- {max_r} {} $obs_nc) <{}>",
                    ta.join(" "), vars[rp], conf(0.95, np)),
                "obs-rotate-90",
            ));
        }

        // Rotate-180: (r, c) → (max_r - r, max_c - c)
        if obs.rotate_180 && max_r > 0 && max_c > 0 {
            let mut ta = vars.clone();
            ta[rp] = "$obs_nr".into();
            ta[cp] = "$obs_nc".into();
            out.push(mk(
                &format!("({target} {}) if\n    ({source} {vars_str})\n    (- {max_r} {} $obs_nr)\n    (- {max_c} {} $obs_nc) <{}>",
                    ta.join(" "), vars[rp], vars[cp], conf(0.95, np)),
                "obs-rotate-180",
            ));
        }

        // Rotate-270: (r, c) → (max_c - c, r)
        if obs.rotate_270 && max_c > 0 {
            let mut ta = vars.clone();
            ta[rp] = "$obs_nr".into();
            ta[cp] = vars[rp].clone();
            out.push(mk(
                &format!("({target} {}) if\n    ({source} {vars_str})\n    (- {max_c} {} $obs_nr) <{}>",
                    ta.join(" "), vars[cp], conf(0.95, np)),
                "obs-rotate-270",
            ));
        }

        // Transpose: swap row and col
        if obs.transpose {
            let mut ta = vars.clone();
            ta.swap(rp, cp);
            out.push(mk(
                &format!("({target} {}) if\n    ({source} {vars_str}) <{}>",
                    ta.join(" "), conf(0.95, np)),
                "obs-transpose",
            ));
        }

        // Shift by observed (dr, dc)
        if obs.shift_dr.is_some() || obs.shift_dc.is_some() {
            let dr = obs.shift_dr.unwrap_or(0);
            let dc = obs.shift_dc.unwrap_or(0);
            if dr != 0 || dc != 0 {
                let mut ta = vars.clone();
                let mut body = format!("({source} {vars_str})");
                if dr != 0 {
                    let op = if dr > 0 { "+" } else { "-" };
                    let abs_dr = dr.unsigned_abs();
                    ta[rp] = "$obs_nr".into();
                    body.push_str(&format!("\n    ({op} {} {abs_dr} $obs_nr)", vars[rp]));
                }
                if dc != 0 {
                    let op = if dc > 0 { "+" } else { "-" };
                    let abs_dc = dc.unsigned_abs();
                    ta[cp] = "$obs_nc".into();
                    body.push_str(&format!("\n    ({op} {} {abs_dc} $obs_nc)", vars[cp]));
                }
                out.push(mk(
                    &format!("({target} {}) if\n    {body} <{}>",
                        ta.join(" "), conf(0.95, np)),
                    &format!("obs-shift-{dr}-{dc}"),
                ));
            }
        }
    }

    // ── CROP TO BBOX ──
    if obs.crop_to_bbox {
        out.push(mk(
            &format!("({target} {vars_str}) if\n    ({source} {vars_str})\n    (!= {} 0) <{}>",
                vars[val_pos], conf(0.90, np)),
            "obs-crop-nonzero",
        ));
    }

    // ── FILL operations ──
    if seq_positions.len() >= 2 {
        let rp = seq_positions[0];
        let cp = seq_positions[1];

        if obs.row_fill {
            let mut sa = vars.clone(); sa[cp] = "$obs_any_c".into();
            out.push(mk(
                &format!("({target} {vars_str}) if\n    ({source} {}) (!= {} 0) <{}>",
                    sa.join(" "), sa[val_pos], conf(0.85, np)),
                "obs-row-fill",
            ));
        }

        if obs.col_fill {
            let mut sa = vars.clone(); sa[rp] = "$obs_any_r".into();
            out.push(mk(
                &format!("({target} {vars_str}) if\n    ({source} {}) (!= {} 0) <{}>",
                    sa.join(" "), sa[val_pos], conf(0.85, np)),
                "obs-col-fill",
            ));
        }

        if obs.cross_fill {
            let mut sa_row = vars.clone(); sa_row[cp] = "$obs_any_c".into();
            let mut sa_col = vars.clone(); sa_col[rp] = "$obs_any_r".into();
            let r1 = format!(
                "({target} {vars_str}) if\n    ({source} {}) (!= {} 0) <{}>",
                sa_row.join(" "), sa_row[val_pos], conf(0.85, np));
            let r2 = format!(
                "({target} {vars_str}) if\n    ({source} {}) (!= {} 0) <{}>",
                sa_col.join(" "), sa_col[val_pos], conf(0.85, np));
            out.push(mk(&format!("{r1}\n{r2}"), "obs-cross-fill"));
        }
    }

    // ── PURE-ADDITION: keep existing + fill adjacency ──
    if obs.pure_addition {
        // Identity: keep all existing cells
        out.push(mk(
            &format!("({target} {vars_str}) if\n    ({source} {vars_str}) <{}>", conf(0.85, np)),
            "obs-pure-addition-keep",
        ));
        // Fill: cells with a non-zero neighbor get fill_color
        if let Some(fc) = obs.fill_color {
            if seq_positions.len() >= 2 {
                let rp = seq_positions[0];
                let cp = seq_positions[1];
                let mut ta = vars.clone(); ta[val_pos] = fc.to_string();
                let mut sa = vars.clone(); sa[val_pos] = "$obs_nv".into();
                let mut nb = vars.clone(); nb[rp] = "$obs_nr".into();
                out.push(mk(
                    &format!("({target} {}) if\n    ({source} {})\n    (== {} 0)\n    ({source} {})\n    (!= $obs_nv 0) <{}>",
                        ta.join(" "), vars_str, vars[val_pos], nb.join(" "), conf(0.80, np)),
                    &format!("obs-pure-addition-fill-{fc}"),
                ));
            }
        }
    }

    // ── PURE-RECOLOR: keep positions, change colors ──
    if obs.pure_recolor {
        // Keep zeros
        let mut ta = vars.clone(); ta[val_pos] = "0".into();
        let mut sa = vars.clone(); sa[val_pos] = "0".into();
        out.push(mk(
            &format!("({target} {}) if\n    ({source} {}) <{}>",
                ta.join(" "), sa.join(" "), conf(0.90, np)),
            "obs-pure-recolor-zeros",
        ));
    }

    // ── OUTPUT-IS-SUBGRID: extract non-zero cells ──
    if obs.output_is_subgrid {
        out.push(mk(
            &format!("({target} {vars_str}) if\n    ({source} {vars_str})\n    (!= {} 0) <{}>",
                vars[val_pos], conf(0.90, np)),
            "obs-subgrid-nonzero",
        ));
    }

    // ── OUTPUT-BINARY: reduce to binary ──
    if obs.output_binary {
        // non-zero → 1
        let mut ta = vars.clone(); ta[val_pos] = "1".into();
        out.push(mk(
            &format!("({target} {}) if\n    ({source} {vars_str})\n    (!= {} 0) <{}>",
                ta.join(" "), vars[val_pos], conf(0.85, np)),
            "obs-binary-nonzero",
        ));
        // zero stays zero
        let mut ta0 = vars.clone(); ta0[val_pos] = "0".into();
        let mut sa0 = vars.clone(); sa0[val_pos] = "0".into();
        out.push(mk(
            &format!("({target} {}) if\n    ({source} {}) <{}>",
                ta0.join(" "), sa0.join(" "), conf(0.85, np)),
            "obs-binary-zero",
        ));
    }

    // ── CONTENT-REARRANGED: same cells, different positions ──
    if obs.content_rearranged && obs.same_cell_count {
        // Keep all non-zero values but potentially swap positions
        out.push(mk(
            &format!("({target} {vars_str}) if\n    ({source} {vars_str})\n    (!= {} 0) <{}>",
                vars[val_pos], conf(0.70, np)),
            "obs-rearrange-keep-nonzero",
        ));
    }

    // ── COMBINED REMAP + TRANSFORM ──
    // When we see BOTH color-remap AND a geometric transform,
    // propose the COMBINED rule (remap + transform in one step)
    if !obs.color_remap_pairs.is_empty() && seq_positions.len() >= 2 {
        let rp = seq_positions[0];
        let cp = seq_positions[1];
        let max_r = source_args[rp].ints.iter().max().copied().unwrap_or(0);
        let max_c = source_args[cp].ints.iter().max().copied().unwrap_or(0);

        for &(from, to) in &obs.color_remap_pairs {
            // Remap + reflect-h
            if obs.reflect_h && max_r > 0 {
                let mut ta = vars.clone();
                ta[rp] = "$obs_nr".into();
                ta[val_pos] = to.to_string();
                let mut sa = vars.clone();
                sa[val_pos] = from.to_string();
                out.push(mk(
                    &format!("({target} {}) if\n    ({source} {})\n    (- {max_r} {} $obs_nr) <{}>",
                        ta.join(" "), sa.join(" "), vars[rp], conf(0.90, np)),
                    &format!("obs-remap-{from}to{to}+reflect-h"),
                ));
            }
            // Remap + reflect-v
            if obs.reflect_v && max_c > 0 {
                let mut ta = vars.clone();
                ta[cp] = "$obs_nc".into();
                ta[val_pos] = to.to_string();
                let mut sa = vars.clone();
                sa[val_pos] = from.to_string();
                out.push(mk(
                    &format!("({target} {}) if\n    ({source} {})\n    (- {max_c} {} $obs_nc) <{}>",
                        ta.join(" "), sa.join(" "), vars[cp], conf(0.90, np)),
                    &format!("obs-remap-{from}to{to}+reflect-v"),
                ));
            }
        }
    }

    // ── ENCLOSED-FILL: fill enclosed regions with observed color ──
    if obs.has_enclosed_regions && obs.same_size {
        if let Some(fill_color) = obs.enclosed_fill_color {
            // Single fill color for all enclosed regions
            let bundle = format!(
                "({target} {vars_str}) if\n    (enclosed-cell {} {} {}) (obs-enclosed-fill-color {}) (input-grid {}) <{}>\n\
                 ({target} {vars_str}) if\n    ({source} {vars_str}) (> {} 0) (input-grid {}) <{}>\n\
                 ({target} {vars_str}) if\n    ({source} {vars_str}) (== {} 0) not (enclosed-cell {} {} {}) (input-grid {}) <{}>",
                vars[0], vars[1], vars[2], vars[val_pos], vars[0], conf(0.96, np),
                vars[val_pos], vars[0], conf(0.97, np),
                vars[val_pos], vars[0], vars[1], vars[2], vars[0], conf(0.95, np),
            );
            out.push(mk(&bundle, &format!("obs-enclosed-fill-{fill_color}")));
        }
    }

    // ── SIZE-DEPENDENT ENCLOSED-FILL: different fill per region area ──
    if !obs.enclosed_size_fills.is_empty() {
        let bundle = format!(
            "({target} {vars_str}) if\n    (enclosed-region-cell {} $obs_reg {} {}) (enclosed-region {} $obs_reg $obs_area) (obs-enclosed-size-fill $obs_area {}) (input-grid {}) <{}>\n\
             ({target} {vars_str}) if\n    ({source} {vars_str}) (> {} 0) (input-grid {}) <{}>\n\
             ({target} {vars_str}) if\n    ({source} {vars_str}) (== {} 0) not (enclosed-cell {} {} {}) (input-grid {}) <{}>",
            vars[0], vars[1], vars[2], vars[0], vars[val_pos], vars[0], conf(0.96, np),
            vars[val_pos], vars[0], conf(0.97, np),
            vars[val_pos], vars[0], vars[1], vars[2], vars[0], conf(0.95, np),
        );
        out.push(mk(&bundle, "obs-enclosed-size-fill"));
    }

    // ── POSITION-DEPENDENT RECOLOR via cell-diff ──
    if obs.position_dependent_recolor && obs.same_size {
        // cell-diff provides (cell-diff $id $r $c $old $new) for changed cells
        // Rule 1: cell-diff alone — only predicts CHANGED cells (high precision)
        let diff_only = format!(
            "({target} {vars_str}) if\n    (cell-diff {} {} {} $obs_old {}) (input-grid {}) <{}>",
            vars[0], vars[1], vars[2], vars[val_pos], vars[0], conf(0.90, np),
        );
        out.push(mk(&diff_only, "obs-pos-dep-recolor-diff-only"));

        // Rule 2: cell-diff + keep non-zero originals (covers changed + unchanged non-zero)
        let diff_plus_keep = format!(
            "({target} {vars_str}) if\n    (cell-diff {} {} {} $obs_old {}) (input-grid {}) <{}>\n\
             ({target} {vars_str}) if\n    ({source} {vars_str}) (> {} 0) (input-grid {}) <{}>",
            vars[0], vars[1], vars[2], vars[val_pos], vars[0], conf(0.90, np),
            vars[val_pos], vars[0], conf(0.85, np),
        );
        out.push(mk(&diff_plus_keep, "obs-pos-dep-recolor-diff-keep"));
    }

    // ── SEPARATOR-BASED EXTRACTION ──
    if let Some(sep) = obs.separator_color {
        let sep_rule = format!(
            "({target} {vars_str}) if\n    ({source} {vars_str}) (> {} 0) (!= {} {}) (input-grid {}) <{}>",
            vars[val_pos], vars[val_pos], sep, vars[0], conf(0.88, np),
        );
        out.push(mk(&sep_rule, &format!("obs-separator-extract-{sep}")));

        // ── SEPARATOR-AND: both halves nonzero → marker ──
        // Uses obs-separator-and + separator-col + arithmetic to compute
        // corresponding cell on opposite side of separator.
        if seq_positions.len() >= 2 {
            let cp = seq_positions[1]; // column position
            let and_rule = format!(
                "({target} {vars_str}) if\n\
                \x20   (obs-separator-and {})\n\
                \x20   (detected-separator-v {})\n\
                \x20   (separator-col {} $obs_sep $obs_sc)\n\
                \x20   ({source} {vars_str})\n\
                \x20   (!= {} 0)\n\
                \x20   (< {} $obs_sep)\n\
                \x20   (+ $obs_sep 1 $obs_base)\n\
                \x20   (+ $obs_base {} $obs_rc)\n\
                \x20   ({source} {} {} $obs_rc $obs_rv)\n\
                \x20   (!= $obs_rv 0)\n\
                \x20   (input-grid {}) <{}>",
                vars[val_pos], vars[0], vars[0],
                vars[val_pos], vars[cp],
                vars[cp],
                vars[0], vars[1],
                vars[0], conf(0.95, np),
            );
            out.push(mk(&and_rule, "obs-separator-and-full"));
        }
    }

    out
}

// ═══════════════════════════════════════════════════════════════════════
// STEP 4: EVALUATE
// ═══════════════════════════════════════════════════════════════════════

#[allow(dead_code)]
fn score_candidate(
    rule_text: &str,
    target_pred: &str,
    base_session: &Session,
    training_inputs: &[Vec<Statement>],
    training_outputs: &[Vec<Statement>],
) -> f64 {
    let stmts = match parser::parse(rule_text) {
        Ok(s) if !s.is_empty() => s,
        _ => return 0.0,
    };

    let mut per_pair_scores = Vec::new();
    for (inp, out_stmts) in training_inputs.iter().zip(training_outputs.iter()) {
        let mut batch = stmts.clone();
        batch.extend(inp.iter().cloned());

        let mut session = base_session.clone();
        let _ = session.exec_statements(batch);

        let predictions = extract_target_facts(&session, target_pred);
        let expected = extract_target_from_stmts(out_stmts, target_pred);
        if expected.is_empty() { continue; }

        let correct = expected.iter().filter(|k| predictions.contains(*k)).count();
        let predicted = predictions.len();
        let recall = correct as f64 / expected.len() as f64;
        let precision = if predicted == 0 { 0.0 } else { correct as f64 / predicted as f64 };
        let pair_score = if precision == 0.0 || recall == 0.0 { 0.0 }
            else { (precision * recall).sqrt() };
        per_pair_scores.push(pair_score);
    }
    if per_pair_scores.is_empty() { return 0.0; }
    let avg = per_pair_scores.iter().sum::<f64>() / per_pair_scores.len() as f64;
    let min_score = per_pair_scores.iter().copied()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0);
    // Reward consistency: a rule that works on ALL pairs is better than
    // one that aces some and fails others (Noether: symmetry across pairs)
    let combined = avg * 0.7 + min_score * 0.3;

    // Log only non-zero scores to avoid flooding
    if combined > 0.05 {
        eprintln!("        score: {:.1}% (avg={:.1}% min={:.1}% pairs={:?}) rule={}",
            combined * 100.0, avg * 100.0, min_score * 100.0,
            per_pair_scores.iter().map(|s| format!("{:.0}%", s * 100.0)).collect::<Vec<_>>(),
            rule_text.chars().take(80).collect::<String>());
    }
    combined
}

fn score_candidate_detailed(
    rule_text: &str,
    target_pred: &str,
    base_session: &Session,
    training_inputs: &[Vec<Statement>],
    training_outputs: &[Vec<Statement>],
) -> ScoreDetail {
    let stmts = match parser::parse(rule_text) {
        Ok(s) if !s.is_empty() => s,
        _ => return ScoreDetail {
            score: 0.0, wrong_keys: vec![], missing_keys: vec![],
            pattern: FailurePattern::AllWrong,
        },
    };

    let mut per_pair_scores = Vec::new();
    let mut all_wrong = Vec::new();
    let mut all_missing = Vec::new();
    let mut total_predicted = 0usize;
    let mut total_expected = 0usize;
    let mut total_correct = 0usize;

    for (inp, out_stmts) in training_inputs.iter().zip(training_outputs.iter()) {
        let mut batch = stmts.clone();
        batch.extend(inp.iter().cloned());

        let mut session = base_session.clone();
        let _ = session.exec_statements(batch);

        let predictions = extract_target_facts(&session, target_pred);
        let expected = extract_target_from_stmts(out_stmts, target_pred);
        if expected.is_empty() { continue; }

        total_predicted += predictions.len();
        total_expected += expected.len();

        let mut correct = 0;
        for exp_key in &expected {
            if predictions.contains(exp_key) {
                correct += 1;
            } else {
                let exp_prefix = exp_key.rsplitn(2, ' ').nth(1).unwrap_or("");
                let wrong_match = predictions.iter()
                    .find(|pk| pk.rsplitn(2, ' ').nth(1).unwrap_or("") == exp_prefix);
                if let Some(got) = wrong_match {
                    all_wrong.push((exp_key.clone(), got.clone()));
                } else {
                    all_missing.push(exp_key.clone());
                }
            }
        }
        total_correct += correct;
        per_pair_scores.push(correct as f64 / expected.len() as f64);
    }

    let score = if per_pair_scores.is_empty() {
        0.0
    } else {
        let avg = per_pair_scores.iter().sum::<f64>() / per_pair_scores.len() as f64;
        let min_score = per_pair_scores.iter().copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        avg * 0.7 + min_score * 0.3
    };

    // Detect over-prediction: high recall but low precision
    let recall = if total_expected > 0 { total_correct as f64 / total_expected as f64 } else { 0.0 };
    let precision = if total_predicted > 0 { total_correct as f64 / total_predicted as f64 } else { 0.0 };
    let extra_count = total_predicted.saturating_sub(total_correct);

    let pattern = if recall > 0.5 && precision < 0.3 && extra_count > 5 {
        // High recall but terrible precision — rule is too broad, needs constraints
        FailurePattern::OverPredicting { recall, precision, extra_count }
    } else if score < 0.01 {
        if all_missing.len() > all_wrong.len() { FailurePattern::MissingFacts }
        else { FailurePattern::AllWrong }
    } else if !all_wrong.is_empty() && all_missing.is_empty() {
        FailurePattern::WrongValues
    } else if score > 0.3 {
        FailurePattern::PartialMatch(score)
    } else if !all_missing.is_empty() {
        FailurePattern::MissingFacts
    } else {
        FailurePattern::PartialMatch(score)
    };

    ScoreDetail { score, wrong_keys: all_wrong, missing_keys: all_missing, pattern }
}

fn extract_target_facts(session: &Session, target_pred: &str) -> HashSet<String> {
    let mut set = HashSet::new();
    for sn in session.all_facts() {
        if let Neuron::Expression(parts) = &sn.neuron {
            if let Some(Neuron::Symbol(pred)) = parts.first() {
                if pred == target_pred { set.insert(fact_key(parts)); }
            }
        }
    }
    set
}

fn extract_target_from_stmts(stmts: &[Statement], target_pred: &str) -> HashSet<String> {
    let mut set = HashSet::new();
    for stmt in stmts {
        if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = stmt {
            if let Some(Neuron::Symbol(pred)) = parts.first() {
                if pred == target_pred { set.insert(fact_key(parts)); }
            }
        }
    }
    set
}

// ═══════════════════════════════════════════════════════════════════════
// STEP 5: COMBINE
// ═══════════════════════════════════════════════════════════════════════

fn combine_top_rules_fast(
    candidates: &[Candidate],
    scorer: &PrebuiltScorer,
    deadline: &Instant,
) -> Vec<Candidate> {
    let mut combos = Vec::new();
    let top: Vec<&Candidate> = candidates.iter().take(5).collect();
    for i in 0..top.len() {
        for j in (i + 1)..top.len() {
            if Instant::now() >= *deadline { return combos; }
            let text = format!("{}\n{}", top[i].rule_text, top[j].rule_text);
            let score = scorer.score(&text);
            if score > top[i].score.max(top[j].score) + 0.01 {
                combos.push(Candidate {
                    rule_text: text, score,
                    source: format!("combine-{}+{}", top[i].source, top[j].source),
                });
            }
        }
    }
    combos
}

// ═══════════════════════════════════════════════════════════════════════
// STEP 6: EVOLVE — General SCAMPER mutations
// ═══════════════════════════════════════════════════════════════════════

fn general_scamper_mutate(
    rule_text: &str,
    source: &str,
    profile: &FactProfile,
) -> Vec<(String, String)> {
    general_scamper_mutate_inner(rule_text, source, profile, None)
}

/// Inner implementation with optional problem_class for guided Grow.
fn general_scamper_mutate_inner(
    rule_text: &str,
    source: &str,
    profile: &FactProfile,
    problem_class: Option<&str>,
) -> Vec<(String, String)> {
    let mut results = Vec::new();

    // ── S: Substitute — swap constants discovered in data ──
    let constants: Vec<i64> = profile.constants.iter().copied().collect();
    for &from in &constants {
        for &to in &constants {
            if from == to { continue; }
            let from_s = format!(" {from})");
            let to_s = format!(" {to})");
            if rule_text.contains(&from_s) {
                let new = rule_text.replacen(&from_s, &to_s, 1);
                if new != rule_text {
                    results.push((new, format!("{source}-sub{from}to{to}")));
                }
            }
        }
        if results.len() > 50 { break; }
    }

    // ── P: Put to use — swap symbol references ──
    for sym_a in &profile.symbols {
        for sym_b in &profile.symbols {
            if sym_a == sym_b { continue; }
            let from_s = format!("({sym_a} ");
            let to_s = format!("({sym_b} ");
            if rule_text.contains(&from_s) {
                let new = rule_text.replacen(&from_s, &to_s, 1);
                if new != rule_text {
                    results.push((new, format!("{source}-swap-{sym_a}-{sym_b}")));
                }
            }
            if results.len() > 80 { break; }
        }
        if results.len() > 80 { break; }
    }

    // ── C: Combine — flip direction operators ──
    for (from_op, to_op) in [("(+ $", "(- $"), ("(- $", "(+ $")] {
        if rule_text.contains(from_op) {
            let new = rule_text.replacen(from_op, to_op, 1);
            if new != rule_text {
                results.push((new, format!("{source}-flip-dir")));
            }
        }
    }

    // ── A: Adapt — change confidence thresholds ──
    for conf in ["0.50", "0.70", "0.85", "0.95"] {
        if let Some(pos) = rule_text.find("<0.75, 0.75>") {
            let new = format!("{}<{conf}, {conf}>{}", &rule_text[..pos], &rule_text[pos + 12..]);
            results.push((new, format!("{source}-adapt-{conf}")));
        }
    }

    // ── M: Modify — change numeric constants by ±1 ──
    for &val in &constants {
        if val > 0 && val < 100 {
            let from_s = format!(" {val} ");
            if rule_text.contains(&from_s) {
                let new = rule_text.replacen(&from_s, &format!(" {} ", val + 1), 1);
                results.push((new, format!("{source}-inc{val}")));
                if val > 1 {
                    let new2 = rule_text.replacen(&from_s, &format!(" {} ", val - 1), 1);
                    results.push((new2, format!("{source}-dec{val}")));
                }
            }
        }
    }

    // ── E: Eliminate — remove non-essential body conditions ──
    let lines: Vec<&str> = rule_text.lines().collect();
    if lines.len() > 2 {
        for skip in 1..lines.len() {
            let line = lines[skip].trim();
            if line.starts_with('(') && !line.contains("if") {
                let is_main = profile.predicates.keys()
                    .any(|pred| line.contains(&format!("({pred} ")));
                if is_main { continue; }
            }
            let new: String = lines.iter().enumerate()
                .filter(|(i, _)| *i != skip)
                .map(|(_, l)| *l)
                .collect::<Vec<_>>().join("\n");
            results.push((new, format!("{source}-elim{skip}")));
        }
    }

    // ── R: Reverse — swap arithmetic operators ──
    for (from_op, to_op) in [("(+ ", "(- "), ("(- ", "(+ "), ("(* ", "(/ "), ("(/ ", "(* ")] {
        if rule_text.contains(from_op) {
            let new = rule_text.replacen(from_op, to_op, 1);
            if new != rule_text {
                results.push((new, format!("{source}-rev-op")));
            }
        }
    }

    // ── G: Grow — ADD a new condition from DYNAMIC predicates ──
    // Instead of 9 hardcoded predicates, build from profile.predicates.
    // problem_class guides which predicates to prioritize.
    {
        // Extract existing variables from the rule
        let existing_vars: Vec<String> = rule_text.match_indices("$a")
            .filter_map(|(pos, _)| {
                let rest = &rule_text[pos..];
                let end = rest.find(|c: char| !c.is_alphanumeric() && c != '_' && c != '$')
                    .unwrap_or(rest.len());
                let var = &rest[..end];
                if var.len() > 1 { Some(var.to_string()) } else { None }
            }).collect::<HashSet<_>>().into_iter().collect();

        // Build dynamic predicate list from profile (ALL session predicates)
        // Structural predicates first, then obs-* as gate conditions
        let mut grow_preds: Vec<(String, usize)> = profile.predicates.iter()
            .filter(|(pred, _)| {
                // Exclude meta predicates that are internal bookkeeping
                !pred.starts_with("detected-")
                    && !pred.starts_with("consistent-")
                    && !pred.starts_with("dna-")
                    && !pred.starts_with("save-")
                    && !pred.starts_with("genesis-")
                    && !pred.starts_with("problem-")
                    && !pred.starts_with("transform-")
                    && !pred.starts_with("strategy-")
                    && !pred.starts_with("train-")
                    && !pred.starts_with("test-")
                    && !pred.starts_with("input-grid")
                    && !pred.starts_with("predict-")
                    && pred.as_str() != "pair"
                    // Allow obs-* but NOT obs-consistent (boolean only, no useful binding)
                    && pred.as_str() != "obs-consistent"
            })
            .map(|(pred, args)| (pred.clone(), args.len()))
            .filter(|(_, arity)| *arity >= 2 && *arity <= 6)
            .collect();

        // Sort: structural predicates first (by problem_class relevance), obs-* last
        if let Some(cls) = problem_class {
            let is_relevant = |pred: &str| -> bool {
                match cls {
                    "color-remap" | "remap" => {
                        pred.contains("color") || pred.contains("remap")
                            || pred == "same-color" || pred.contains("recolored")
                    }
                    "spatial" | "geometric" => {
                        pred.contains("neighbor") || pred.contains("cell")
                            || pred.contains("adjacent") || pred.contains("above")
                            || pred.contains("left") || pred.contains("shift")
                    }
                    "filtering" | "filter" => {
                        pred.contains("object") || pred.contains("interior")
                            || pred.contains("border") || pred.contains("largest")
                            || pred.contains("size")
                    }
                    "region-fill" | "fill" => {
                        pred.contains("enclosed") || pred.contains("region")
                            || pred.contains("fill")
                    }
                    "tiling" | "scaling" | "scale" => {
                        pred.contains("grid-cell") || pred.contains("scale")
                            || pred.contains("size") || pred.contains("tile")
                    }
                    _ => false,
                }
            };
            grow_preds.sort_by(|(a, _), (b, _)| {
                let a_obs = a.starts_with("obs-");
                let b_obs = b.starts_with("obs-");
                // Non-obs first, then relevant, then alphabetical
                a_obs.cmp(&b_obs)
                    .then(is_relevant(b).cmp(&is_relevant(a)))
                    .then(a.cmp(b))
            });
        } else {
            grow_preds.sort_by(|(a, _), (b, _)| {
                let a_obs = a.starts_with("obs-");
                let b_obs = b.starts_with("obs-");
                a_obs.cmp(&b_obs).then(a.cmp(b))
            });
        }

        // Cap: 20 structural + 10 obs-* = 30 max
        let obs_start = grow_preds.iter().position(|(p, _)| p.starts_with("obs-")).unwrap_or(grow_preds.len());
        let structural_end = obs_start.min(20);
        let obs_end = (obs_start + 10).min(grow_preds.len());
        let mut capped: Vec<(String, usize)> = grow_preds[..structural_end].to_vec();
        if obs_start < grow_preds.len() {
            capped.extend_from_slice(&grow_preds[obs_start..obs_end]);
        }
        let grow_preds = capped;

        // Find where to insert (before the confidence <...>)
        if let Some(conf_pos) = rule_text.rfind(" <") {
            let before_conf = &rule_text[..conf_pos];
            let conf_part = &rule_text[conf_pos..];

            for (pred, arity) in &grow_preds {
                // Skip if predicate already in rule
                if rule_text.contains(&format!("({pred} ")) { continue; }

                // Build argument list: share as many positions as possible with head vars
                let shared = (*arity).min(existing_vars.len().max(4));
                let mut args: Vec<String> = Vec::new();
                for i in 0..*arity {
                    if i < shared.min(*arity - 1) {
                        args.push(format!("$a{i}"));
                    } else {
                        args.push(format!("$g{}", i));
                    }
                }
                let args_str = args.join(" ");
                let new_cond = format!("({pred} {args_str})");

                // Add as positive condition
                let new_rule = format!("{before_conf}\n    {new_cond}{conf_part}");
                results.push((new_rule, format!("{source}-grow-{pred}")));

                // Try with negation for predicates that denote properties
                if pred.contains("enclosed") || pred.contains("neighbor")
                    || pred.contains("border") || pred.contains("interior")
                    || pred.contains("marker")
                {
                    let neg_rule = format!("{before_conf}\n    not {new_cond}{conf_part}");
                    results.push((neg_rule, format!("{source}-grow-not-{pred}")));
                }

                // For grid-cell–like predicates (4+ arity), try guards on last arg
                if *arity >= 4 {
                    let last_arg = args.last().unwrap();
                    let guarded = format!("{before_conf}\n    {new_cond}\n    (> {last_arg} 0){conf_part}");
                    results.push((guarded, format!("{source}-grow-{pred}-nz")));
                    let guarded0 = format!("{before_conf}\n    ({pred} {} 0){conf_part}",
                        args[..args.len()-1].join(" "));
                    results.push((guarded0, format!("{source}-grow-{pred}-zero")));
                }

                if results.len() > 200 { break; }
            }

            // Also try adding guard conditions on existing variables
            for var in &existing_vars {
                if var.contains("a3") || var.contains("g0") {
                    let guard_nz = format!("{before_conf}\n    (> {var} 0){conf_part}");
                    results.push((guard_nz, format!("{source}-grow-guard-{var}-nz")));
                    // Try specific constants from profile
                    for &c in &[1i64, 2, 3, 5, 8] {
                        let guard_eq = format!("{before_conf}\n    (== {var} {c}){conf_part}");
                        results.push((guard_eq, format!("{source}-grow-guard-{var}-eq{c}")));
                    }
                }
            }
        }
    }

    results
}

fn guided_mutate(rule_text: &str, source: &str, detail: &ScoreDetail) -> Vec<(String, String)> {
    let mut results = Vec::new();

    match &detail.pattern {
        FailurePattern::WrongValues => {
            // Values wrong at right positions — extract wrong values from keys
            for (exp_key, got_key) in &detail.wrong_keys {
                let exp_val = exp_key.rsplit(' ').next().unwrap_or("");
                let got_val = got_key.rsplit(' ').next().unwrap_or("");
                if exp_val != got_val {
                    let from_s = format!(" {got_val})");
                    let to_s = format!(" {exp_val})");
                    if rule_text.contains(&from_s) {
                        let new = rule_text.replace(&from_s, &to_s);
                        if new != rule_text {
                            results.push((new, format!("{source}-fix-{got_val}to{exp_val}")));
                        }
                    }
                }
            }
        }
        FailurePattern::PartialMatch(pct) => {
            if *pct > 0.3 {
                // Partial match — combine with identity fallback
                let lines: Vec<&str> = rule_text.lines().collect();
                if let Some(head) = lines.first() {
                    if let Some(end) = head.find(' ') {
                        let tgt = &head[1..end];
                        // Count args from the head rather than hardcoding 4
                        let arg_count = head.split_whitespace().count().saturating_sub(1);
                        let arg_count = if arg_count == 0 { 4 } else { arg_count };
                        let fallback_vars: String = (0..arg_count)
                            .map(|i| format!("$f{i}"))
                            .collect::<Vec<_>>()
                            .join(" ");
                        for line in &lines[1..] {
                            let trimmed = line.trim().trim_start_matches('(');
                            if let Some(sp) = trimmed.split_whitespace().next() {
                                if sp != tgt {
                                    let fallback = format!(
                                        "{rule_text}\n({tgt} {fallback_vars}) if\n    ({sp} {fallback_vars}) <0.30, 0.30>"
                                    );
                                    results.push((fallback, format!("{source}-fallback")));
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            // Also try score-aware: if score is high (>0.7), be more conservative
            if detail.score > 0.7 {
                // Only try small tweaks on wrong values
                for (exp_key, _) in &detail.wrong_keys {
                    let val = exp_key.rsplit(' ').next().unwrap_or("");
                    if let Ok(v) = val.parse::<i64>() {
                        for delta in [-1i64, 1] {
                            let new = rule_text.replacen(
                                &format!(" {})", v + delta), &format!(" {v})"), 1);
                            if new != rule_text {
                                results.push((new, format!("{source}-tweak-{v}")));
                            }
                        }
                    }
                }
            }
        }
        FailurePattern::MissingFacts => {
            // Rule didn't fire — relax body conditions
            let lines: Vec<&str> = rule_text.lines().collect();
            if lines.len() > 2 {
                for skip in 1..lines.len() {
                    let new: String = lines.iter().enumerate()
                        .filter(|(i, _)| *i != skip)
                        .map(|(_, l)| *l)
                        .collect::<Vec<_>>().join("\n");
                    results.push((new, format!("{source}-relax-{skip}")));
                }
            }
            // Also try: if many missing, the issue might be wrong predicate reference
            if detail.missing_keys.len() > 3 {
                if let Some(first_missing) = detail.missing_keys.first() {
                    let pred = first_missing.split_whitespace().next().unwrap_or("");
                    if !pred.is_empty() && !rule_text.contains(&format!("({pred} ")) {
                        // The target predicate might have a different name
                        results.push((
                            rule_text.to_string(), // re-try as-is (maybe DNA rules will help)
                            format!("{source}-retry"),
                        ));
                    }
                }
            }
        }
        FailurePattern::OverPredicting { recall, precision, extra_count } => {
            // Rule gets many correct but predicts WAY too many extras.
            // Solution: add constraining conditions to filter out wrong predictions.
            // This is the KEY mutation for discovering multi-body rules.

            eprintln!("        over-predict: recall={:.0}% precision={:.0}% extras={}",
                recall * 100.0, precision * 100.0, extra_count);

            // Strategy 1: Add observation-based guards (obs-separator-and, obs-consistent, etc.)
            // These narrow predictions to puzzle-type-specific contexts.
            if let Some(conf_pos) = rule_text.rfind(" <") {
                let before_conf = &rule_text[..conf_pos];
                let conf_part = &rule_text[conf_pos..];

                let obs_guards: &[&str] = &[
                    "(obs-separator-and $obs_val)",
                    "(obs-separator-xor $obs_val)",
                    "(detected-separator-v $a0)",
                    "(obs-consistent same-size)",
                    "(obs-consistent output-smaller)",
                    "(obs-consistent has-separator)",
                ];
                for guard in obs_guards {
                    if !rule_text.contains(guard.split_whitespace().nth(0).unwrap_or("")) {
                        let new = format!("{before_conf}\n    {guard}{conf_part}");
                        results.push((new, format!("{source}-constrain-obs")));
                    }
                }

                // Strategy 2: Add arithmetic guards to limit coordinate ranges
                // Extract variables from the rule head
                let head_vars: Vec<String> = rule_text.lines().next()
                    .unwrap_or("")
                    .match_indices('$')
                    .filter_map(|(pos, _)| {
                        let rest = &rule_text[pos..];
                        let end = rest.find(|c: char| !c.is_alphanumeric() && c != '_' && c != '$')
                            .unwrap_or(rest.len());
                        let var = &rest[..end];
                        if var.len() > 1 { Some(var.to_string()) } else { None }
                    }).collect();

                // For each head variable, try adding a guard (< var N) for small N
                for var in &head_vars {
                    for bound in [3i64, 5, 7, 10] {
                        let guard = format!("(< {var} {bound})");
                        if !rule_text.contains(&guard) {
                            let new = format!("{before_conf}\n    {guard}{conf_part}");
                            results.push((new, format!("{source}-bound-{var}-lt{bound}")));
                        }
                    }
                }

                // Strategy 3: Add a second body condition that links to grid-cell
                // Only for single-body rules (these over-predict most)
                let body_count = rule_text.lines().skip(1)
                    .filter(|l| l.trim().starts_with('(') || l.trim().starts_with("not "))
                    .count();
                if body_count <= 1 {
                    // Try requiring a grid-cell fact at the same coordinates
                    let gc_conds = &[
                        "(grid-cell $a0 $a1 $a2 $g0)\n    (> $g0 0)",
                        "(grid-cell $a0 $a1 $a2 0)",
                        "(cell-added $g1 $a0 $a1 $a3 $a2)",
                        "(recolored $g1 $a0 $a1 $a2 $g2 $a3)",
                    ];
                    for gc in gc_conds {
                        let new = format!("{before_conf}\n    {gc}{conf_part}");
                        results.push((new, format!("{source}-constrain-body")));
                    }
                }
            }
        }
        FailurePattern::AllWrong => {
            // Completely wrong — no guided fix, caller falls back to SCAMPER
        }
    }

    results
}

// ═══════════════════════════════════════════════════════════════════════
// DATAFROG JOIN MINING — Bottom-up rule discovery via relational joins
// ═══════════════════════════════════════════════════════════════════════
//
// Instead of generating candidate rules and testing each one (top-down),
// this engine computes which predicate JOINS actually produce the expected
// output (bottom-up). Only viable joins become QOR rule candidates.
//
// DOMAIN AGNOSTIC — works with whatever predicates exist at runtime.

/// Symbol interner — maps String symbols to i64 for fast numeric comparison.
/// Literals use value + LITERAL_OFFSET to avoid collision with interned IDs.
struct SymbolInterner {
    sym_to_id: HashMap<String, i64>,
    id_to_sym: Vec<String>,
    next_id: i64,
}

const LITERAL_OFFSET: i64 = 1_000_000;

impl SymbolInterner {
    fn new() -> Self {
        SymbolInterner {
            sym_to_id: HashMap::new(),
            id_to_sym: vec![String::new()], // index 0 unused
            next_id: 1,
        }
    }

    fn intern(&mut self, s: &str) -> i64 {
        if let Some(&id) = self.sym_to_id.get(s) {
            return id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.sym_to_id.insert(s.to_string(), id);
        self.id_to_sym.push(s.to_string());
        id
    }

    fn intern_neuron(&mut self, n: &Neuron) -> i64 {
        match n {
            Neuron::Symbol(s) => self.intern(s),
            Neuron::Value(QorValue::Int(i)) => *i + LITERAL_OFFSET,
            Neuron::Value(QorValue::Float(f)) => (*f * 1000.0) as i64 + LITERAL_OFFSET,
            Neuron::Value(QorValue::Bool(b)) => if *b { 1 + LITERAL_OFFSET } else { LITERAL_OFFSET },
            Neuron::Value(QorValue::Str(s)) => self.intern(s),
            _ => 0,
        }
    }

    fn resolve(&self, id: i64) -> String {
        if id >= LITERAL_OFFSET {
            format!("{}", id - LITERAL_OFFSET)
        } else if (id as usize) < self.id_to_sym.len() {
            self.id_to_sym[id as usize].clone()
        } else {
            format!("?{}", id)
        }
    }
}

/// Normalize a NeuronStore into grouped predicate facts.
/// Returns: pred_id → Vec of arg-value rows (excluding predicate name).
fn normalize_store(
    store: &crate::store::NeuronStore,
    interner: &mut SymbolInterner,
) -> HashMap<i64, Vec<Vec<i64>>> {
    let mut result: HashMap<i64, Vec<Vec<i64>>> = HashMap::new();
    for sn in store.all() {
        if let Neuron::Expression(parts) = &sn.neuron {
            if parts.is_empty() { continue; }
            if let Neuron::Symbol(pred) = &parts[0] {
                // Skip internal/meta predicates
                if pred.starts_with("obs-") || pred.starts_with("detected-")
                    || pred.starts_with("consistent-") || pred.starts_with("dna-")
                    || pred.starts_with("save-") || pred == "train-pair"
                    || pred == "test-input" || pred == "input-grid"
                    || pred == "item-id"
                {
                    continue;
                }
                let pred_id = interner.intern(pred);
                let args: Vec<i64> = parts[1..].iter()
                    .map(|n| interner.intern_neuron(n))
                    .collect();
                result.entry(pred_id).or_default().push(args);
            }
        }
    }
    result
}

/// Normalize expected output statements to sets of interned arg tuples.
fn normalize_expected(
    stmts: &[Statement],
    target_pred: &str,
    interner: &mut SymbolInterner,
) -> HashSet<Vec<i64>> {
    let mut result = HashSet::new();
    for stmt in stmts {
        if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = stmt {
            if let Some(Neuron::Symbol(pred)) = parts.first() {
                if pred == target_pred {
                    let args: Vec<i64> = parts[1..].iter()
                        .map(|n| interner.intern_neuron(n))
                        .collect();
                    result.insert(args);
                }
            }
        }
    }
    result
}

/// Join-mine: discover which predicate joins produce the expected output.
/// Uses datafrog's sorted merge-join for efficient pairwise computation.
/// Returns only VIABLE candidates (coverage > threshold on ALL training pairs).
#[cfg(feature = "datafrog")]
fn join_mine(
    base_session: &Session,
    training_inputs: &[Vec<Statement>],
    training_outputs: &[Vec<Statement>],
    target_pred: &str,
    target_arity: usize,
) -> Vec<Candidate> {
    if target_arity < 2 { return Vec::new(); }
    let num_pairs = training_inputs.len();
    if num_pairs == 0 { return Vec::new(); }
    let jm_start = Instant::now();
    let jm_deadline = jm_start + Duration::from_millis(1500); // 1.5s budget for join mining

    let mut interner = SymbolInterner::new();
    let target_pred_id = interner.intern(target_pred);

    // For each training pair: run DNA, normalize enriched facts + expected output
    let mut pair_facts: Vec<HashMap<i64, Vec<Vec<i64>>>> = Vec::new();
    let mut pair_targets: Vec<HashSet<Vec<i64>>> = Vec::new();

    for (inp, out) in training_inputs.iter().zip(training_outputs.iter()) {
        let mut session = base_session.clone();
        let _ = session.exec_statements(inp.clone());
        let mut facts = normalize_store(session.store(), &mut interner);
        facts.remove(&target_pred_id); // Remove target pred from source facts
        pair_facts.push(facts);
        pair_targets.push(normalize_expected(out, target_pred, &mut interner));
    }

    // Collect all predicates with their arities (union across pairs)
    let mut pred_arities: HashMap<i64, usize> = HashMap::new();
    let mut pred_counts: HashMap<i64, usize> = HashMap::new(); // avg fact count for prioritizing
    for facts in &pair_facts {
        for (&pred_id, rows) in facts {
            if let Some(first) = rows.first() {
                pred_arities.entry(pred_id).or_insert(first.len());
                *pred_counts.entry(pred_id).or_insert(0) += rows.len();
            }
        }
    }

    // Pre-filter: only consider predicates with arity >= 2 and reasonable fact count.
    // With 100+ predicates, trying all pairs is O(N²) — limit to top candidates.
    let mut pred_list: Vec<(i64, usize)> = pred_arities.iter()
        .filter(|(_, &a)| a >= 2)
        .map(|(&k, &v)| (k, v))
        .collect();
    // Prioritize predicates with same arity as target, then by fact count
    pred_list.sort_by(|a, b| {
        let a_exact = if a.1 == target_arity { 0 } else { 1 };
        let b_exact = if b.1 == target_arity { 0 } else { 1 };
        a_exact.cmp(&b_exact)
            .then_with(|| pred_counts.get(&b.0).unwrap_or(&0)
                .cmp(pred_counts.get(&a.0).unwrap_or(&0)))
    });
    pred_list.truncate(20); // Cap at 20 predicates → ~190 pairs max

    let mut candidates = Vec::new();

    // ── PHASE 1: Direct hits (1-body) ──
    // For each predicate with matching arity, check if its facts directly match target.
    for (&pred_id, &arity) in &pred_arities {
        if arity != target_arity { continue; }

        let min_cov = min_coverage_direct(&pair_facts, &pair_targets, pred_id, target_arity);
        if min_cov > 0.3 {
            let pred_name = interner.resolve(pred_id);
            let vars: Vec<String> = (0..target_arity).map(|i| format!("$a{i}")).collect();
            let vars_str = vars.join(" ");
            candidates.push(mk(
                &format!("({target_pred} {vars_str}) if\n    ({pred_name} {vars_str}) <{}>", conf(0.70, num_pairs)),
                &format!("jm-direct-{pred_name}"),
            ));
        }

        // Also try with value from last column (if arity > target)
        if arity > target_arity {
            let min_cov = min_coverage_value_end(&pair_facts, &pair_targets, pred_id, target_arity, arity);
            if min_cov > 0.3 {
                let pname = interner.resolve(pred_id);
                let head_vars: Vec<String> = (0..target_arity).map(|i| format!("$a{i}")).collect();
                let mut body_vars: Vec<String> = (0..target_arity - 1).map(|i| format!("$a{i}")).collect();
                for j in (target_arity - 1)..arity - 1 { body_vars.push(format!("$e{j}")); }
                body_vars.push(format!("$a{}", target_arity - 1));
                if body_vars.len() == arity {
                    candidates.push(mk(
                        &format!("({target_pred} {}) if\n    ({pname} {}) <{}>",
                            head_vars.join(" "), body_vars.join(" "), conf(0.65, num_pairs)),
                        &format!("jm-valend-{pname}"),
                    ));
                }
            }
        }
    }

    // ── PHASE 2: 2-body datafrog joins ──
    let preds = &pred_list; // Use pre-filtered, prioritized list
    let shared_cols = if target_arity > 1 { target_arity - 1 } else { 1 };

    for (i, &(p1_id, a1)) in preds.iter().enumerate() {
        if a1 < 2 || Instant::now() > jm_deadline { continue; }

        for &(p2_id, a2) in preds.iter().skip(i) {
            if a2 < 2 || candidates.len() >= 200 || Instant::now() > jm_deadline { break; }

            let p1_name = interner.resolve(p1_id);
            let p2_name = interner.resolve(p2_id);

            // ── Pattern A: Share position columns, value from pred2's last col ──
            // pred1(id,r,c,...) ∧ pred2(id,r,c,...,val) → target(id,r,c,val)
            if a1 >= shared_cols && a2 >= target_arity {
                let min_cov = min_coverage_join_gate(
                    &pair_facts, &pair_targets, p1_id, a1, p2_id, a2,
                    shared_cols, target_arity,
                );
                if min_cov > 0.3 {
                    let head_vars: Vec<String> = (0..target_arity).map(|i| format!("$a{i}")).collect();
                    let mut p1_vars: Vec<String> = (0..shared_cols.min(a1)).map(|i| format!("$a{i}")).collect();
                    for j in shared_cols..a1 { p1_vars.push(format!("$e{j}")); }
                    let mut p2_vars: Vec<String> = (0..shared_cols.min(a2)).map(|i| format!("$a{i}")).collect();
                    for j in shared_cols..a2 - 1 { p2_vars.push(format!("$f{j}")); }
                    p2_vars.push(format!("$a{}", target_arity - 1));

                    if p1_vars.len() == a1 && p2_vars.len() == a2 {
                        candidates.push(mk(
                            &format!("({target_pred} {}) if\n    ({p1_name} {})\n    ({p2_name} {}) <{}>",
                                head_vars.join(" "), p1_vars.join(" "), p2_vars.join(" "), conf(0.70, num_pairs)),
                            &format!("jm-gate-{p1_name}-{p2_name}"),
                        ));
                    }
                }
            }

            // ── Pattern B: Share only first column (id), different spatial vars ──
            // pred1(id,...) ∧ pred2(id,r,c,val) → target(id,r,c,val)
            if a2 >= target_arity && a1 >= 2 {
                let min_cov = min_coverage_join_gate(
                    &pair_facts, &pair_targets, p1_id, a1, p2_id, a2,
                    1, target_arity,
                );
                if min_cov > 0.4 {
                    let head_vars: Vec<String> = (0..target_arity).map(|i| format!("$a{i}")).collect();
                    let mut p1_vars = vec![format!("$a0")];
                    for j in 1..a1 { p1_vars.push(format!("$e{j}")); }
                    let mut p2_vars: Vec<String> = vec![format!("$a0")];
                    for j in 1..target_arity { p2_vars.push(format!("$a{j}")); }
                    for j in target_arity..a2 { p2_vars.push(format!("$f{j}")); }

                    if p1_vars.len() == a1 && p2_vars.len() == a2 {
                        candidates.push(mk(
                            &format!("({target_pred} {}) if\n    ({p1_name} {})\n    ({p2_name} {}) <{}>",
                                head_vars.join(" "), p1_vars.join(" "), p2_vars.join(" "), conf(0.65, num_pairs)),
                            &format!("jm-cross-{p1_name}-{p2_name}"),
                        ));
                    }
                }
            }

            // ── Pattern C: Negation — pred1 ∧ ¬pred2 ──
            // pred1(id,r,c,v) ∧ ¬pred2(id,r,c,...) → target(id,r,c,v)
            if a1 >= target_arity && a2 >= shared_cols {
                let min_cov = min_coverage_antijoin(
                    &pair_facts, &pair_targets, p1_id, a1, p2_id, a2,
                    shared_cols, target_arity,
                );
                if min_cov > 0.3 {
                    let head_vars: Vec<String> = (0..target_arity).map(|i| format!("$a{i}")).collect();
                    let mut p1_vars: Vec<String> = (0..target_arity).map(|i| format!("$a{i}")).collect();
                    for j in target_arity..a1 { p1_vars.push(format!("$e{j}")); }
                    let mut p2_vars: Vec<String> = (0..shared_cols.min(a2)).map(|i| format!("$a{i}")).collect();
                    for j in shared_cols..a2 { p2_vars.push(format!("$f{j}")); }

                    if p1_vars.len() == a1 && p2_vars.len() == a2 {
                        candidates.push(mk(
                            &format!("({target_pred} {}) if\n    ({p1_name} {})\n    not ({p2_name} {}) <{}>",
                                head_vars.join(" "), p1_vars.join(" "), p2_vars.join(" "), conf(0.65, num_pairs)),
                            &format!("jm-neg-{p1_name}-not-{p2_name}"),
                        ));
                    }
                }
            }
        }
    }

    // ── PHASE 3: 3-body extensions ──
    // For viable 2-body candidates, try adding a 3rd predicate condition.
    // Uses datafrog leapjoin for efficient 3-way join.
    if candidates.len() < 100 && Instant::now() < jm_deadline {
        let viable_2body: Vec<(i64, i64)> = candidates.iter()
            .filter_map(|c| {
                if c.source.starts_with("jm-gate-") || c.source.starts_with("jm-cross-") {
                    let parts: Vec<&str> = c.source.splitn(4, '-').collect();
                    if parts.len() >= 4 {
                        let p1 = interner.sym_to_id.get(parts[2]).copied();
                        let p2 = interner.sym_to_id.get(parts[3]).copied();
                        if let (Some(a), Some(b)) = (p1, p2) { return Some((a, b)); }
                    }
                }
                None
            })
            .collect();

        for &(p1_id, p2_id) in &viable_2body {
            for &(p3_id, a3) in preds.iter() {
                if p3_id == p1_id || p3_id == p2_id || a3 < shared_cols { continue; }
                if candidates.len() >= 200 || Instant::now() > jm_deadline { break; }

                let a1 = pred_arities.get(&p1_id).copied().unwrap_or(0);
                let a2 = pred_arities.get(&p2_id).copied().unwrap_or(0);
                if a1 < shared_cols || a2 < target_arity { continue; }

                // Check: does adding p3 as gate improve coverage?
                let min_cov = min_coverage_3body_gate(
                    &pair_facts, &pair_targets, p1_id, a1, p2_id, a2, p3_id, a3,
                    shared_cols, target_arity,
                );
                if min_cov > 0.5 {
                    let p1_name = interner.resolve(p1_id);
                    let p2_name = interner.resolve(p2_id);
                    let p3_name = interner.resolve(p3_id);
                    let head_vars: Vec<String> = (0..target_arity).map(|i| format!("$a{i}")).collect();
                    let mut p1_vars: Vec<String> = (0..shared_cols.min(a1)).map(|i| format!("$a{i}")).collect();
                    for j in shared_cols..a1 { p1_vars.push(format!("$e{j}")); }
                    let mut p2_vars: Vec<String> = (0..shared_cols.min(a2)).map(|i| format!("$a{i}")).collect();
                    for j in shared_cols..a2 - 1 { p2_vars.push(format!("$g{j}")); }
                    p2_vars.push(format!("$a{}", target_arity - 1));
                    let mut p3_vars: Vec<String> = (0..shared_cols.min(a3)).map(|i| format!("$a{i}")).collect();
                    for j in shared_cols..a3 { p3_vars.push(format!("$h{j}")); }

                    if p1_vars.len() == a1 && p2_vars.len() == a2 && p3_vars.len() == a3 {
                        candidates.push(mk(
                            &format!("({target_pred} {}) if\n    ({p1_name} {})\n    ({p2_name} {})\n    ({p3_name} {}) <{}>",
                                head_vars.join(" "), p1_vars.join(" "), p2_vars.join(" "),
                                p3_vars.join(" "), conf(0.70, num_pairs)),
                            &format!("jm-3body-{p1_name}-{p2_name}-{p3_name}"),
                        ));
                    }
                }
            }
        }
    }

    let jm_elapsed = jm_start.elapsed().as_millis();
    eprintln!("  join-mine: {} viable candidates from {} predicates ({} pairs) {}ms",
        candidates.len(), preds.len(), num_pairs, jm_elapsed);
    candidates
}

/// Direct coverage: pred facts that exactly match target tuples.
#[cfg(feature = "datafrog")]
fn min_coverage_direct(
    pair_facts: &[HashMap<i64, Vec<Vec<i64>>>],
    pair_targets: &[HashSet<Vec<i64>>],
    pred_id: i64,
    target_arity: usize,
) -> f64 {
    let mut min_cov = f64::MAX;
    for (facts, target) in pair_facts.iter().zip(pair_targets.iter()) {
        if target.is_empty() { continue; }
        let rows = match facts.get(&pred_id) {
            Some(r) => r,
            None => { return 0.0; }
        };
        let pred_set: HashSet<&[i64]> = rows.iter()
            .filter(|r| r.len() == target_arity)
            .map(|r| r.as_slice())
            .collect();
        let hits = target.iter().filter(|t| pred_set.contains(t.as_slice())).count();
        min_cov = min_cov.min(hits as f64 / target.len() as f64);
    }
    if min_cov == f64::MAX { 0.0 } else { min_cov }
}

/// Coverage with value from predicate's last column mapped to target's last column.
#[cfg(feature = "datafrog")]
fn min_coverage_value_end(
    pair_facts: &[HashMap<i64, Vec<Vec<i64>>>],
    pair_targets: &[HashSet<Vec<i64>>],
    pred_id: i64,
    target_arity: usize,
    pred_arity: usize,
) -> f64 {
    let mut min_cov = f64::MAX;
    for (facts, target) in pair_facts.iter().zip(pair_targets.iter()) {
        if target.is_empty() { continue; }
        let rows = match facts.get(&pred_id) {
            Some(r) => r,
            None => { return 0.0; }
        };
        // Map: first (target_arity-1) columns + last column → target format
        let mapped: HashSet<Vec<i64>> = rows.iter()
            .filter(|r| r.len() == pred_arity)
            .map(|r| {
                let mut t: Vec<i64> = r[..target_arity - 1].to_vec();
                t.push(r[pred_arity - 1]);
                t
            })
            .collect();
        let hits = target.iter().filter(|t| mapped.contains(*t)).count();
        min_cov = min_cov.min(hits as f64 / target.len() as f64);
    }
    if min_cov == f64::MAX { 0.0 } else { min_cov }
}

/// 2-body gate join coverage using datafrog.
/// pred1(shared...) ∧ pred2(shared..., val) → target(shared..., val)
/// Early-terminates if any pair has 0 coverage.
#[cfg(feature = "datafrog")]
fn min_coverage_join_gate(
    pair_facts: &[HashMap<i64, Vec<Vec<i64>>>],
    pair_targets: &[HashSet<Vec<i64>>],
    p1_id: i64, _a1: usize,
    p2_id: i64, _a2: usize,
    shared_cols: usize,
    target_arity: usize,
) -> f64 {
    use datafrog::Relation;
    let mut min_cov = f64::MAX;

    for (facts, target) in pair_facts.iter().zip(pair_targets.iter()) {
        if target.is_empty() { continue; }
        let p1_rows = match facts.get(&p1_id) { Some(r) => r, None => { return 0.0; } };
        let p2_rows = match facts.get(&p2_id) { Some(r) => r, None => { return 0.0; } };

        // Quick pre-check: do p1 and p2 share any key values?
        let p1_keys: HashSet<&[i64]> = p1_rows.iter()
            .filter(|r| r.len() >= shared_cols)
            .map(|r| &r[..shared_cols])
            .collect();
        let has_overlap = p2_rows.iter()
            .any(|r| r.len() >= shared_cols && p1_keys.contains(&r[..shared_cols]));
        if !has_overlap { return 0.0; }

        // Build datafrog Relations keyed on shared columns
        let mut p1_kv: Vec<(Vec<i64>, Vec<i64>)> = p1_rows.iter()
            .filter(|r| r.len() >= shared_cols)
            .map(|r| (r[..shared_cols].to_vec(), r[shared_cols..].to_vec()))
            .collect();
        p1_kv.sort();
        let mut p2_kv: Vec<(Vec<i64>, Vec<i64>)> = p2_rows.iter()
            .filter(|r| r.len() >= shared_cols)
            .map(|r| (r[..shared_cols].to_vec(), r[shared_cols..].to_vec()))
            .collect();
        p2_kv.sort();

        let p1_rel = Relation::from_vec(p1_kv);
        let p2_rel = Relation::from_vec(p2_kv);

        // Join on shared columns, project to target shape
        let joined: Relation<Vec<i64>> = Relation::from_join(
            &p1_rel, &p2_rel,
            |key, _p1_rest, p2_rest| {
                let mut result = key.clone();
                let need = target_arity.saturating_sub(shared_cols);
                if need > 0 && !p2_rest.is_empty() {
                    if need == 1 {
                        result.push(*p2_rest.last().unwrap());
                    } else {
                        for j in 0..need.min(p2_rest.len()) {
                            result.push(p2_rest[j]);
                        }
                    }
                }
                result
            },
        );

        // Use HashSet for O(1) lookups instead of linear scan
        let joined_set: HashSet<&Vec<i64>> = joined.elements.iter().collect();
        let hits = target.iter().filter(|t| joined_set.contains(t)).count();
        let cov = hits as f64 / target.len() as f64;
        if cov < 0.01 { return 0.0; } // Early termination
        min_cov = min_cov.min(cov);
    }
    if min_cov == f64::MAX { 0.0 } else { min_cov }
}

/// Anti-join coverage: pred1 rows whose key is NOT in pred2.
#[cfg(feature = "datafrog")]
fn min_coverage_antijoin(
    pair_facts: &[HashMap<i64, Vec<Vec<i64>>>],
    pair_targets: &[HashSet<Vec<i64>>],
    p1_id: i64, _a1: usize,
    p2_id: i64, _a2: usize,
    shared_cols: usize,
    target_arity: usize,
) -> f64 {
    use datafrog::Relation;
    let mut min_cov = f64::MAX;

    for (facts, target) in pair_facts.iter().zip(pair_targets.iter()) {
        if target.is_empty() { continue; }
        let p1_rows = match facts.get(&p1_id) { Some(r) => r, None => { return 0.0; } };
        let p2_rows = match facts.get(&p2_id) { Some(r) => r, None => { min_cov = min_cov.min(0.0); continue; } };

        // Build keyed relations
        let mut p1_kv: Vec<(Vec<i64>, Vec<i64>)> = p1_rows.iter()
            .filter(|r| r.len() >= target_arity)
            .map(|r| (r[..shared_cols].to_vec(), r[..target_arity].to_vec()))
            .collect();
        p1_kv.sort();

        let mut p2_keys: Vec<Vec<i64>> = p2_rows.iter()
            .filter(|r| r.len() >= shared_cols)
            .map(|r| r[..shared_cols].to_vec())
            .collect();
        p2_keys.sort();
        p2_keys.dedup();

        let p1_rel = Relation::from_vec(p1_kv);
        let p2_keys_rel = Relation::from_vec(p2_keys);

        // Anti-join: p1 rows where key NOT in p2
        let anti: Relation<Vec<i64>> = Relation::from_antijoin(
            &p1_rel, &p2_keys_rel,
            |_key, val| val.clone(),
        );

        let anti_set: HashSet<&Vec<i64>> = anti.elements.iter().collect();
        let hits = target.iter().filter(|t| anti_set.contains(t)).count();
        min_cov = min_cov.min(hits as f64 / target.len() as f64);
    }
    if min_cov == f64::MAX { 0.0 } else { min_cov }
}

/// 3-body gate join: pred1 ∧ pred2 ∧ pred3 → target
#[cfg(feature = "datafrog")]
fn min_coverage_3body_gate(
    pair_facts: &[HashMap<i64, Vec<Vec<i64>>>],
    pair_targets: &[HashSet<Vec<i64>>],
    p1_id: i64, _a1: usize,
    p2_id: i64, _a2: usize,
    p3_id: i64, _a3: usize,
    shared_cols: usize,
    target_arity: usize,
) -> f64 {
    use datafrog::Relation;
    let mut min_cov = f64::MAX;

    for (facts, target) in pair_facts.iter().zip(pair_targets.iter()) {
        if target.is_empty() { continue; }
        let p1_rows = match facts.get(&p1_id) { Some(r) => r, None => { return 0.0; } };
        let p2_rows = match facts.get(&p2_id) { Some(r) => r, None => { return 0.0; } };
        let p3_rows = match facts.get(&p3_id) { Some(r) => r, None => { return 0.0; } };

        // Join p1 ∧ p2 on shared columns first
        let mut p1_kv: Vec<(Vec<i64>, Vec<i64>)> = p1_rows.iter()
            .filter(|r| r.len() >= shared_cols)
            .map(|r| (r[..shared_cols].to_vec(), r[shared_cols..].to_vec()))
            .collect();
        p1_kv.sort();
        let mut p2_kv: Vec<(Vec<i64>, Vec<i64>)> = p2_rows.iter()
            .filter(|r| r.len() >= shared_cols)
            .map(|r| (r[..shared_cols].to_vec(), r[shared_cols..].to_vec()))
            .collect();
        p2_kv.sort();

        let p1_rel = Relation::from_vec(p1_kv);
        let p2_rel = Relation::from_vec(p2_kv);

        // First join: p1 ∧ p2
        let j12: Relation<(Vec<i64>, Vec<i64>)> = Relation::from_join(
            &p1_rel, &p2_rel,
            |key, _p1r, p2r| (key.clone(), p2r.clone()),
        );

        // Second join: (p1∧p2) ∧ p3 on same shared columns
        let mut p3_kv: Vec<(Vec<i64>, Vec<i64>)> = p3_rows.iter()
            .filter(|r| r.len() >= shared_cols)
            .map(|r| (r[..shared_cols].to_vec(), r[shared_cols..].to_vec()))
            .collect();
        p3_kv.sort();
        let p3_rel = Relation::from_vec(p3_kv);

        let joined: Relation<Vec<i64>> = Relation::from_join(
            &j12, &p3_rel,
            |key, p2r, _p3r| {
                let mut result = key.clone();
                let need = target_arity.saturating_sub(shared_cols);
                if need == 1 && !p2r.is_empty() {
                    result.push(*p2r.last().unwrap());
                } else {
                    for j in 0..need.min(p2r.len()) {
                        result.push(p2r[j]);
                    }
                }
                result
            },
        );

        let hits = target.iter()
            .filter(|t| joined.elements.iter().any(|j| j == *t))
            .count();
        min_cov = min_cov.min(hits as f64 / target.len() as f64);
    }
    if min_cov == f64::MAX { 0.0 } else { min_cov }
}

// ═══════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════

// ── Combinatorial Rule Proposal ─────────────────────────────────────
//
// Tries ALL combinations of available predicates as rule body conditions.
// This is the SEARCH that discovers rules from scratch — no hand-coding.
// Domain-agnostic: works with whatever predicates exist in the session.

fn propose_combinatorial(
    target: &str,
    base_session: &Session,
    training_inputs: &[Vec<Statement>],
    num_pairs: usize,
    target_arity: usize,
) -> Vec<Candidate> {
    let mut out = Vec::new();
    if target_arity < 2 { return out; }

    // Run DNA on first training input to discover all available predicates
    let mut enriched = base_session.clone();
    let _ = enriched.exec_statements(training_inputs[0].clone());

    // Collect predicates with their arities (skip target, obs-*, detected-*)
    let mut pred_arities: HashMap<String, usize> = HashMap::new();
    for sn in enriched.all_facts() {
        if let Neuron::Expression(parts) = &sn.neuron {
            if let Some(Neuron::Symbol(pred)) = parts.first() {
                if pred == target
                    || pred.starts_with("obs-")
                    || pred.starts_with("detected-")
                    || pred.starts_with("consistent-")
                    || pred.starts_with("dna-")
                    || pred.starts_with("save-")
                    || pred == "train-pair"
                    || pred == "test-input"
                    || pred == "input-grid"
                {
                    continue;
                }
                pred_arities.entry(pred.clone()).or_insert(parts.len() - 1);
            }
        }
    }

    let np = num_pairs;
    let head_vars: Vec<String> = (0..target_arity).map(|i| format!("$a{i}")).collect();
    let head_str = head_vars.join(" ");

    // ── 1-body rules: each predicate as sole condition ──
    for (pred, &arity) in &pred_arities {
        if arity < 2 { continue; }

        // Binding A: share as many positions as possible with head
        let shared = arity.min(target_arity);
        let mut args: Vec<String> = (0..shared).map(|i| format!("$a{i}")).collect();
        for i in shared..arity {
            args.push(format!("$e{i}"));
        }
        let args_str = args.join(" ");
        out.push(mk(
            &format!("({target} {head_str}) if\n    ({pred} {args_str}) <{}>", conf(0.60, np)),
            &format!("comb1-{pred}"),
        ));

        // If predicate has more args than target, try using last arg as value
        if arity > target_arity {
            let mut args2: Vec<String> = (0..target_arity - 1).map(|i| format!("$a{i}")).collect();
            // Fill middle with extras
            for i in (target_arity - 1)..arity - 1 {
                args2.push(format!("$e{i}"));
            }
            // Last arg maps to target's last var (the value)
            args2.push(format!("$a{}", target_arity - 1));
            if args2.len() == arity {
                out.push(mk(
                    &format!("({target} {head_str}) if\n    ({pred} {}) <{}>", args2.join(" "), conf(0.60, np)),
                    &format!("comb1-{pred}-valend"),
                ));
            }
        }
    }

    // ── 2-body rules: combine pairs of predicates ──
    let preds: Vec<(String, usize)> = pred_arities.into_iter().collect();

    for (i, (p1, a1)) in preds.iter().enumerate() {
        if *a1 < 2 { continue; }

        // P1 bindings: share id + position with head
        let shared1 = (*a1).min(target_arity);
        let p1_args: Vec<String> = (0..shared1).map(|k| format!("$a{k}"))
            .chain((*a1 > shared1).then(|| (shared1..*a1).map(|k| format!("$e{k}")))
                .into_iter().flatten())
            .collect();
        if p1_args.len() != *a1 { continue; }
        let p1_str = p1_args.join(" ");

        for (p2, a2) in preds.iter().skip(i) {
            if *a2 < 2 { continue; }
            if out.len() >= 800 { break; }

            // Pattern A: P2 shares id + same position (filter/gate)
            {
                let shared2 = (*a2).min(target_arity);
                let p2_args: Vec<String> = (0..shared2).map(|k| format!("$a{k}"))
                    .chain((*a2 > shared2).then(|| (shared2..*a2).map(|k| format!("$f{k}")))
                        .into_iter().flatten())
                    .collect();
                if p2_args.len() == *a2 {
                    out.push(mk(
                        &format!("({target} {head_str}) if\n    ({p1} {p1_str})\n    ({p2} {}) <{}>",
                            p2_args.join(" "), conf(0.65, np)),
                        &format!("comb2-{p1}-{p2}-gate"),
                    ));
                }
            }

            // Pattern B: P2 shares id but different spatial vars (cross-cell lookup)
            if *a2 >= 3 {
                let mut p2_args_cross: Vec<String> = vec![format!("$a0")]; // share id
                for k in 1..*a2 {
                    p2_args_cross.push(format!("$x{k}"));
                }
                // Map P2's last arg to head's value position
                if let Some(last) = p2_args_cross.last_mut() {
                    *last = format!("$a{}", target_arity - 1);
                }
                out.push(mk(
                    &format!("({target} {head_str}) if\n    ({p1} {p1_str})\n    ({p2} {}) <{}>",
                        p2_args_cross.join(" "), conf(0.60, np)),
                    &format!("comb2-{p1}-{p2}-cross"),
                ));
            }

            // Pattern C: P2 with negation (exclude cells matching P2)
            if *a2 >= 2 && *a2 <= target_arity {
                let shared2 = (*a2).min(target_arity);
                let p2_args_neg: Vec<String> = (0..shared2).map(|k| format!("$a{k}"))
                    .chain((*a2 > shared2).then(|| (shared2..*a2).map(|k| format!("$f{k}")))
                        .into_iter().flatten())
                    .collect();
                if p2_args_neg.len() == *a2 {
                    out.push(mk(
                        &format!("({target} {head_str}) if\n    ({p1} {p1_str})\n    not ({p2} {}) <{}>",
                            p2_args_neg.join(" "), conf(0.60, np)),
                        &format!("comb2-{p1}-not-{p2}"),
                    ));
                }
            }
        }

        // Guard variants on the primary predicate
        if *a1 == target_arity {
            // (> $a_last 0) — nonzero value
            let last_var = format!("$a{}", target_arity - 1);
            out.push(mk(
                &format!("({target} {head_str}) if\n    ({p1} {p1_str})\n    (> {last_var} 0) <{}>", conf(0.60, np)),
                &format!("comb1-{p1}-nz"),
            ));
            // (== $a_last 0) — zero value
            out.push(mk(
                &format!("({target} {head_str}) if\n    ({p1} {p1_str})\n    (== {last_var} 0) <{}>", conf(0.60, np)),
                &format!("comb1-{p1}-zero"),
            ));
        }
    }

    // ── 3-body rules: combine triples of the most promising predicates ──
    // Only when we haven't hit the cap yet, and we have enough predicates.
    // Use top-10 predicates (by arity match to target) to keep combinations small.
    if out.len() < 1200 && preds.len() >= 3 {
        // Select top predicates: prefer those whose arity matches target_arity
        let mut ranked: Vec<(String, usize)> = preds.iter()
            .filter(|(_, a)| *a >= 2)
            .map(|(p, a)| (p.clone(), *a))
            .collect();
        ranked.sort_by_key(|(_, a)| (*a as i64 - target_arity as i64).unsigned_abs());
        ranked.truncate(8);

        for i in 0..ranked.len() {
            if out.len() >= 1200 { break; }
            let (ref p1, a1) = ranked[i];
            let shared1 = a1.min(target_arity);
            let p1_args: Vec<String> = (0..shared1).map(|k| format!("$a{k}"))
                .chain((shared1..a1).map(|k| format!("$e{k}")))
                .collect();
            if p1_args.len() != a1 { continue; }
            let p1_str = p1_args.join(" ");

            for j in (i+1)..ranked.len() {
                if out.len() >= 1200 { break; }
                let (ref p2, a2) = ranked[j];
                let shared2 = a2.min(target_arity);
                let p2_args: Vec<String> = (0..shared2).map(|k| format!("$a{k}"))
                    .chain((shared2..a2).map(|k| format!("$f{k}")))
                    .collect();
                if p2_args.len() != a2 { continue; }
                let p2_str = p2_args.join(" ");

                for k in (j+1)..ranked.len() {
                    if out.len() >= 1200 { break; }
                    let (ref p3, a3) = ranked[k];
                    let shared3 = a3.min(target_arity);
                    let p3_args: Vec<String> = (0..shared3).map(|k| format!("$a{k}"))
                        .chain((shared3..a3).map(|k| format!("$h{k}")))
                        .collect();
                    if p3_args.len() != a3 { continue; }
                    let p3_str = p3_args.join(" ");

                    // 3-body: P1 + P2 + P3 all sharing id/position with head
                    out.push(mk(
                        &format!("({target} {head_str}) if\n    ({p1} {p1_str})\n    ({p2} {p2_str})\n    ({p3} {p3_str}) <{}>",
                            conf(0.60, np)),
                        &format!("comb3-{p1}-{p2}-{p3}"),
                    ));

                    // 3-body with P3 negated
                    if a3 <= target_arity {
                        out.push(mk(
                            &format!("({target} {head_str}) if\n    ({p1} {p1_str})\n    ({p2} {p2_str})\n    not ({p3} {p3_str}) <{}>",
                                conf(0.55, np)),
                            &format!("comb3-{p1}-{p2}-not{p3}"),
                        ));
                    }
                }
            }
        }
    }

    // ── GATED variants: add obs-* as gate conditions to best proposals ──
    // obs-* facts are cross-pair invariants — gating on them helps rules generalize.
    // Collect obs-* predicates with their full argument strings from session.
    let base_count = out.len();
    let mut gate_entries: Vec<(String, Vec<String>)> = Vec::new(); // (pred+args string, value_args)
    {
        let mut seen_gates: HashSet<String> = HashSet::new();
        for sn in enriched.all_facts() {
            if let Neuron::Expression(parts) = &sn.neuron {
                if let Some(Neuron::Symbol(pred)) = parts.first() {
                    if !pred.starts_with("obs-") { continue; }
                    // Skip pure boolean obs-consistent — no value to bind
                    if pred == "obs-consistent" { continue; }
                    // Build the full fact string for use as a gate condition
                    let args: Vec<String> = parts[1..].iter().map(|p| p.to_string()).collect();
                    let gate_key = format!("({} {})", pred, args.join(" "));
                    if seen_gates.insert(gate_key.clone()) {
                        gate_entries.push((gate_key, args));
                    }
                }
            }
        }
    }

    eprintln!("      genesis-gates: {} obs-* gate predicates found, {} base rules", gate_entries.len(), base_count);
    for (gi, (gs, _)) in gate_entries.iter().enumerate().take(5) {
        eprintln!("        gate[{gi}]: {gs}");
    }

    if !gate_entries.is_empty() {
        // Pre-collect base rule data to avoid borrow conflicts
        let gate_limit = gate_entries.len().min(10);
        let base_limit = base_count.min(15);
        let base_snapshots: Vec<(String, String)> = out[..base_limit].iter()
            .filter_map(|c| {
                let conf_pos = c.rule_text.rfind(" <")?;
                Some((c.rule_text[..conf_pos].to_string(), c.source.clone()))
            })
            .collect();

        let mut gated_count = 0usize;
        for (before_conf, base_source) in &base_snapshots {
            if gated_count >= 200 { break; }

            for gi in 0..gate_limit {
                if gated_count >= 200 { break; }
                let (gate_str, gate_args) = &gate_entries[gi];

                // Try binding gate value to head's last var (the value)
                if !gate_args.is_empty() {
                    let last_arg = &gate_args[gate_args.len() - 1];
                    if last_arg.parse::<i64>().is_ok() {
                        let val_var = format!("$a{}", target_arity - 1);
                        let gate_with_var = gate_str.replacen(last_arg, &val_var, 1);
                        let bound_rule = format!("{}\n    {} <{}>", before_conf, gate_with_var, conf(0.70, np));
                        out.push(mk(&bound_rule, &format!("{}-gated-bound-{}", base_source, gate_str)));
                        gated_count += 1;
                    }
                }

                let gated_rule = format!("{}\n    {} <{}>", before_conf, gate_str, conf(0.65, np));
                out.push(mk(&gated_rule, &format!("{}-gated-{}", base_source, gate_str)));
                gated_count += 1;
            }
        }
    }

    out
}

// ═══════════════════════════════════════════════════════════════════════
// VALUE TRACING — Evidence-based rule construction
//
// Instead of guessing rule shapes, traces WHERE output values come from
// in input/derived facts and BUILDS rules from the evidence.
//
// For each output position, asks: "which input predicate+position
// consistently provides this exact value across all training pairs?"
// When it finds a match, it constructs a rule connecting them.
//
// DOMAIN AGNOSTIC — works with any predicates and any value types.
// ═══════════════════════════════════════════════════════════════════════

/// A discovered mapping: output position ← (source_predicate, source_position)
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ValueSource {
    out_pos: usize,
    source_pred: String,
    source_pos: usize,
    consistency: f64,  // fraction of output facts matched
}

/// Trace where output values come from and build rules from evidence.
fn propose_from_value_tracing(
    target_pred: &str,
    training_inputs: &[Vec<Statement>],
    training_outputs: &[Vec<Statement>],
    derived_facts: &[Vec<StoredFact>],
    hub_predicates: &[String],
    num_pairs: usize,
) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    if training_inputs.is_empty() || training_outputs.is_empty() { return candidates; }

    // Collect output facts per pair
    let output_facts_per_pair: Vec<Vec<Vec<Neuron>>> = training_outputs.iter()
        .map(|stmts| {
            stmts.iter().filter_map(|s| {
                if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = s {
                    if let Some(Neuron::Symbol(p)) = parts.first() {
                        if p == target_pred { return Some(parts.clone()); }
                    }
                }
                None
            }).collect()
        })
        .collect();

    if output_facts_per_pair.is_empty() || output_facts_per_pair[0].is_empty() {
        return candidates;
    }
    let target_arity = output_facts_per_pair[0][0].len(); // includes predicate at pos 0
    if target_arity < 2 { return candidates; }

    // Build value-to-source index for each training pair
    // Maps: value_string → Vec<(predicate, position)>
    let pair_indices: Vec<HashMap<String, Vec<(String, usize)>>> = training_inputs.iter()
        .enumerate()
        .map(|(pi, input_stmts)| {
            let mut idx: HashMap<String, Vec<(String, usize)>> = HashMap::new();

            // Index raw input facts
            for stmt in input_stmts {
                if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = stmt {
                    if let Some(Neuron::Symbol(pred)) = parts.first() {
                        if pred == target_pred { continue; } // skip target pred in input
                        for (pos, part) in parts.iter().enumerate().skip(1) {
                            let val_str = format!("{}", part);
                            idx.entry(val_str).or_default().push((pred.clone(), pos));
                        }
                    }
                }
            }

            // Also index derived facts (from DNA forward chaining)
            if pi < derived_facts.len() {
                for df in &derived_facts[pi] {
                    if let Some(Neuron::Symbol(pred)) = df.parts.first() {
                        if pred == target_pred { continue; }
                        for (pos, part) in df.parts.iter().enumerate().skip(1) {
                            let val_str = format!("{}", part);
                            idx.entry(val_str).or_default().push((pred.clone(), pos));
                        }
                    }
                }
            }

            // Dedup sources per value
            for sources in idx.values_mut() {
                sources.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
                sources.dedup();
            }

            idx
        })
        .collect();

    // For each output position, find sources that consistently provide the value
    let mut sources_per_pos: Vec<Vec<ValueSource>> = vec![Vec::new(); target_arity];

    for out_pos in 1..target_arity {
        // Count how often each (source_pred, source_pos) provides the right value
        let mut source_hits: HashMap<(String, usize), usize> = HashMap::new();
        let mut total_output_facts = 0usize;

        for (pi, pair_outputs) in output_facts_per_pair.iter().enumerate() {
            if pi >= pair_indices.len() { continue; }
            let idx = &pair_indices[pi];

            for out_fact in pair_outputs {
                total_output_facts += 1;
                let out_val = format!("{}", &out_fact[out_pos]);

                if let Some(sources) = idx.get(&out_val) {
                    for (pred, pos) in sources {
                        *source_hits.entry((pred.clone(), *pos)).or_insert(0) += 1;
                    }
                }
            }
        }

        if total_output_facts == 0 { continue; }

        // Keep sources that match >= 80% of output facts
        let threshold = (total_output_facts as f64 * 0.8) as usize;
        for ((pred, pos), hits) in &source_hits {
            if *hits >= threshold.max(1) {
                let consistency = *hits as f64 / total_output_facts as f64;
                sources_per_pos[out_pos].push(ValueSource {
                    out_pos, source_pred: pred.clone(), source_pos: *pos, consistency,
                });
            }
        }

        // Sort by consistency (best first), then prefer hub predicates
        sources_per_pos[out_pos].sort_by(|a, b| {
            let a_hub = hub_predicates.contains(&a.source_pred);
            let b_hub = hub_predicates.contains(&b.source_pred);
            b.consistency.partial_cmp(&a.consistency).unwrap_or(std::cmp::Ordering::Equal)
                .then(b_hub.cmp(&a_hub))
        });
    }

    // Log what we found
    for out_pos in 1..target_arity {
        if !sources_per_pos[out_pos].is_empty() {
            let top: Vec<String> = sources_per_pos[out_pos].iter().take(3)
                .map(|vs| format!("{}.{} ({:.0}%)", vs.source_pred, vs.source_pos, vs.consistency * 100.0))
                .collect();
            eprintln!("        value-trace pos{}: sources=[{}]", out_pos, top.join(", "));
        }
    }

    // ── BUILD RULES from discovered sources ──
    // Strategy: for each combination of source predicates, build a rule
    // that binds shared positions as variables

    // Group sources by source predicate
    let mut preds_at_pos: Vec<Vec<&str>> = vec![Vec::new(); target_arity];
    for out_pos in 1..target_arity {
        for vs in &sources_per_pos[out_pos] {
            if !preds_at_pos[out_pos].contains(&vs.source_pred.as_str()) {
                preds_at_pos[out_pos].push(&vs.source_pred);
            }
        }
    }

    // Find if a SINGLE source predicate can provide ALL positions
    let mut all_source_preds: HashSet<String> = HashSet::new();
    for pos_sources in &sources_per_pos[1..] {
        for vs in pos_sources {
            all_source_preds.insert(vs.source_pred.clone());
        }
    }

    for source_pred in &all_source_preds {
        // Check which output positions this predicate can provide
        let mut provides: Vec<(usize, usize)> = Vec::new(); // (out_pos, source_pos)
        for out_pos in 1..target_arity {
            for vs in &sources_per_pos[out_pos] {
                if &vs.source_pred == source_pred {
                    provides.push((out_pos, vs.source_pos));
                    break; // take best match per position
                }
            }
        }

        if provides.len() < 2 { continue; } // need at least 2 position matches

        // Build the rule
        let coverage = provides.len() as f64 / (target_arity - 1) as f64;

        // Create variable names for each position
        let mut head_args: Vec<String> = vec!["_".into(); target_arity];
        let mut body_args: Vec<String> = Vec::new();
        head_args[0] = target_pred.to_string(); // predicate name doesn't use variable

        // Find max source position to size body args
        let max_source_pos = provides.iter().map(|(_, sp)| *sp).max().unwrap_or(0);
        let source_arity = max_source_pos + 1;
        body_args.resize(source_arity, "_".into());

        let mut var_counter = 0;
        for &(out_pos, source_pos) in &provides {
            let var_name = format!("$v{}", var_counter);
            var_counter += 1;
            head_args[out_pos] = var_name.clone();
            if source_pos < body_args.len() {
                body_args[source_pos] = var_name;
            }
        }

        // Replace remaining "_" with unique variables (free positions)
        for arg in head_args.iter_mut().skip(1) {
            if arg == "_" {
                *arg = format!("$v{}", var_counter);
                var_counter += 1;
            }
        }
        for arg in body_args.iter_mut() {
            if arg == "_" {
                *arg = format!("$v{}", var_counter);
                var_counter += 1;
            }
        }

        let head_str = head_args[1..].join(" ");
        let body_str = body_args.join(" ");
        let conf_val = evidence_confidence(0.85 * coverage, num_pairs);

        let rule_text = format!(
            "({target_pred} {head_str}) if\n    ({source_pred} {body_str}) <{conf_val:.2}, {conf_val:.2}>"
        );

        candidates.push(mk(&rule_text, &format!("value-trace-{source_pred}-{}/{}", provides.len(), target_arity - 1)));

        // If coverage < 100%, try adding a second body predicate for missing positions
        if provides.len() < target_arity - 1 {
            let missing_positions: Vec<usize> = (1..target_arity)
                .filter(|op| !provides.iter().any(|(p, _)| p == op))
                .collect();

            for missing_pos in &missing_positions {
                for vs in &sources_per_pos[*missing_pos] {
                    if &vs.source_pred == source_pred { continue; } // already tried

                    // Build 2-body rule: main source + supplementary source
                    let mut head2 = head_args.clone();
                    let shared_var = format!("$v{}", var_counter);
                    var_counter += 1;
                    head2[*missing_pos] = shared_var.clone();

                    let supp_arity = vs.source_pos + 1;
                    let mut supp_args: Vec<String> = (0..supp_arity)
                        .map(|i| if i == vs.source_pos { shared_var.clone() } else { format!("$w{}", i) })
                        .collect();

                    let head2_str = head2[1..].join(" ");
                    let body2_str = body_args.join(" ");
                    let supp_str = supp_args.join(" ");
                    let conf2 = evidence_confidence(0.80, num_pairs);

                    let rule_text2 = format!(
                        "({target_pred} {head2_str}) if\n    ({source_pred} {body2_str})\n    ({} {supp_str}) <{conf2:.2}, {conf2:.2}>",
                        vs.source_pred
                    );

                    candidates.push(mk(&rule_text2, &format!("value-trace-2body-{source_pred}+{}", vs.source_pred)));
                    break; // one supplementary source per missing position
                }
            }
        }
    }

    // ── CROSS-PREDICATE rules: different source preds for different positions ──
    // If no single predicate covers all positions, try combining top sources
    if target_arity > 2 && candidates.is_empty() {
        // Take the best source per position
        let mut best_per_pos: Vec<Option<&ValueSource>> = vec![None; target_arity];
        for out_pos in 1..target_arity {
            best_per_pos[out_pos] = sources_per_pos[out_pos].first();
        }

        let unique_preds: HashSet<&str> = best_per_pos.iter()
            .filter_map(|o| o.map(|vs| vs.source_pred.as_str()))
            .collect();

        if unique_preds.len() >= 2 && unique_preds.len() <= 3 {
            // Build a multi-body rule with one body condition per source predicate
            let mut head_args: Vec<String> = vec!["_".into(); target_arity];
            let mut body_parts: Vec<String> = Vec::new();
            let mut var_counter = 0;

            // Group by source predicate
            let mut pred_positions: HashMap<&str, Vec<(usize, usize)>> = HashMap::new();
            for out_pos in 1..target_arity {
                if let Some(vs) = best_per_pos[out_pos] {
                    pred_positions.entry(&vs.source_pred)
                        .or_default()
                        .push((out_pos, vs.source_pos));
                }
            }

            for (pred, positions) in &pred_positions {
                let max_sp = positions.iter().map(|(_, sp)| *sp).max().unwrap_or(0);
                let mut bargs: Vec<String> = (0..=max_sp).map(|_| { var_counter += 1; format!("$z{}", var_counter) }).collect();

                for &(out_pos, source_pos) in positions {
                    let var = format!("$x{}", out_pos);
                    head_args[out_pos] = var.clone();
                    if source_pos < bargs.len() {
                        bargs[source_pos] = var;
                    }
                }
                body_parts.push(format!("({} {})", pred, bargs.join(" ")));
            }

            // Fill remaining head positions
            for arg in head_args.iter_mut().skip(1) {
                if arg == "_" {
                    var_counter += 1;
                    *arg = format!("$x{}", var_counter);
                }
            }

            let head_str = head_args[1..].join(" ");
            let body_str = body_parts.join("\n    ");
            let conf_val = evidence_confidence(0.75, num_pairs);

            let rule_text = format!(
                "({target_pred} {head_str}) if\n    {body_str} <{conf_val:.2}, {conf_val:.2}>"
            );
            candidates.push(mk(&rule_text, &format!("value-trace-cross-{}preds", unique_preds.len())));
        }
    }

    candidates
}

fn mk(text: &str, source: &str) -> Candidate {
    Candidate { rule_text: text.into(), score: 0.0, source: source.into() }
}

fn neuron_to_i64(n: &Neuron) -> Option<i64> {
    match n {
        Neuron::Value(QorValue::Int(i)) => Some(*i),
        Neuron::Value(QorValue::Float(f)) => Some(*f as i64),
        _ => None,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// EVOLUTION — Iterative refinement of near-miss candidates
// ═══════════════════════════════════════════════════════════════════════

/// Parse a candidate's rule_text into a chain::Rule for mutation.
fn parse_to_rule(rule_text: &str) -> Option<Rule> {
    let stmts = parser::parse(rule_text).ok()?;
    for stmt in stmts {
        if let Statement::Rule { head, body, tv } = stmt {
            return Some(Rule {
                head,
                body,
                tv: tv.unwrap_or(qor_core::truth_value::TruthValue::new(0.90, 0.90)),
                stratum: 0,
            });
        }
    }
    None
}

/// Pre-built scorer: sessions with DNA+inputs already at fixed-point.
/// Avoids redundant DNA forward-chaining on every score call.
struct PrebuiltScorer {
    /// Pre-built sessions: DNA rules ran, training inputs loaded, chained to fixed-point.
    pair_sessions: Vec<Session>,
    /// Expected output facts (keys) per training pair.
    expected: Vec<HashSet<String>>,
    target_pred: String,
}

impl PrebuiltScorer {
    fn new(
        base_session: &Session,
        training_inputs: &[Vec<Statement>],
        training_outputs: &[Vec<Statement>],
        target_pred: &str,
    ) -> Self {
        let mut pair_sessions = Vec::with_capacity(training_inputs.len());
        let mut expected = Vec::with_capacity(training_outputs.len());

        for (inp, out_stmts) in training_inputs.iter().zip(training_outputs.iter()) {
            // Build session with DNA + inputs, forward chain to fixed-point
            let mut session = base_session.clone();
            let _ = session.exec_statements(inp.clone());
            pair_sessions.push(session);

            // Pre-extract expected output keys
            expected.push(extract_target_from_stmts(out_stmts, target_pred));
        }

        PrebuiltScorer {
            pair_sessions,
            expected,
            target_pred: target_pred.to_string(),
        }
    }

    /// Score a candidate rule — clones pre-built session, adds candidate rule,
    /// runs only the candidate rule against the existing fact store.
    /// Much faster than score_candidate() because DNA facts are already derived.
    fn score(&self, rule_text: &str) -> f64 {
        let stmts = match parser::parse(rule_text) {
            Ok(s) if !s.is_empty() => s,
            _ => return 0.0,
        };

        // Extract candidate rules from parsed text
        let candidate_rules: Vec<Rule> = stmts.iter().filter_map(|s| {
            if let Statement::Rule { head, body, tv } = s {
                Some(Rule::new(
                    head.clone(), body.clone(),
                    tv.unwrap_or(qor_core::truth_value::TruthValue::new(0.90, 0.90)),
                ))
            } else {
                None
            }
        }).collect();

        if candidate_rules.is_empty() {
            return 0.0;
        }

        let mut per_pair_scores = Vec::new();
        for (_pi, (session, exp)) in self.pair_sessions.iter()
            .zip(self.expected.iter())
            .enumerate()
        {
            if exp.is_empty() { continue; }

            // Clone pre-built store and REMOVE DNA-inferred predictions so candidate
            // is scored on its OWN predictions, not contaminated by DNA over-prediction.
            // Uses remove_inferred to keep asserted input facts (important when
            // source and target share the same predicate).
            let mut store = session.store().clone();
            store.remove_inferred_by_predicate(&self.target_pred);

            // Forward chain with ONLY candidate rules against clean store
            let _new_derived = crate::chain::forward_chain(&candidate_rules, &mut store);

            // Extract candidate-only predictions (DNA predictions were removed)
            let candidate_preds = extract_target_facts_from_store(&store, &self.target_pred);

            // Score candidate on its own precision-recall
            let correct = exp.iter().filter(|k| candidate_preds.contains(*k)).count();
            let predicted = candidate_preds.len();
            let expected_count = exp.len();

            let recall = correct as f64 / expected_count as f64;
            let precision = if predicted == 0 { 0.0 }
                else { correct as f64 / predicted as f64 };
            let pair_score = if precision == 0.0 || recall == 0.0 { 0.0 }
                else { (precision * recall).sqrt() };
            per_pair_scores.push(pair_score);
        }

        if per_pair_scores.is_empty() { return 0.0; }
        let avg = per_pair_scores.iter().sum::<f64>() / per_pair_scores.len() as f64;
        let min_score = per_pair_scores.iter().copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        // Generalization-aware scoring: min-score dominant to penalize overfitting.
        // A rule that scores 100% on 3 pairs but 10% on 1 pair is BAD (overfitted).
        // Geometric mean ensures ALL pairs must score well.
        let base_score = avg * 0.5 + min_score * 0.5;

        // Simplicity bonus: fewer body conditions = more likely to generalize.
        // Count body conditions (lines starting with whitespace in rule text)
        // Applied as a tiny tie-breaker, not a major factor.
        base_score
    }
}

/// Extract target-predicate fact keys from a NeuronStore directly.
fn extract_target_facts_from_store(
    store: &crate::store::NeuronStore,
    target_pred: &str,
) -> HashSet<String> {
    let mut out = HashSet::new();
    for sn in store.all() {
        if let Neuron::Expression(parts) = &sn.neuron {
            if let Some(Neuron::Symbol(pred)) = parts.first() {
                if pred == target_pred {
                    let key: Vec<String> = parts.iter().map(|p| p.to_string()).collect();
                    out.insert(key.join(" "));
                }
            }
        }
    }
    out
}

/// AlphaEvolve-style iterative refinement of genesis candidates.
///
/// Takes the best candidates from genesis_swarm, mutates them,
/// scores mutations, keeps improvements, repeats until solved or
/// budget exhausted. This is the key insight: an 80% rule is
/// ALMOST there — a few mutations can push it to 100%.
///
/// DOMAIN AGNOSTIC — uses score_candidate() which compares
/// predicted vs expected facts by key matching.
pub fn evolve_candidates(
    candidates: &[Candidate],
    target_pred: &str,
    base_session: &Session,
    training_inputs: &[Vec<Statement>],
    training_outputs: &[Vec<Statement>],
    time_budget_ms: u64,
) -> Vec<Candidate> {
    let start = Instant::now();
    let budget = Duration::from_millis(time_budget_ms);

    // Only evolve candidates that show promise (> 30% score)
    let mut population: Vec<Candidate> = candidates.iter()
        .filter(|c| c.score > 0.30)
        .take(10)
        .cloned()
        .collect();

    if population.is_empty() {
        return candidates.to_vec();
    }

    // Pre-build scorer: DNA+inputs at fixed-point (built ONCE, reused for all mutations)
    let scorer = PrebuiltScorer::new(base_session, training_inputs, training_outputs, target_pred);

    // Extract context from base_session for context-aware mutations:
    // known predicates (name, arity) and known integer values from data
    let (known_preds, known_vals) = extract_session_context(base_session, training_inputs);

    let mut seen = HashSet::new();
    for c in &population {
        seen.insert(c.rule_text.clone());
    }

    let max_pop = 30;
    let mut generation = 0;
    let max_generations = 50;
    let mut stall_count = 0;

    while start.elapsed() < budget && generation < max_generations {
        generation += 1;
        let mut improved = false;

        // Take top-5 for mutation
        let top: Vec<Candidate> = population.iter().take(5).cloned().collect();

        for parent in &top {
            if start.elapsed() >= budget { break; }

            let rule = match parse_to_rule(&parent.rule_text) {
                Some(r) => r,
                None => continue,
            };

            // Phase A: Structural mutations (swap constants, remove conds, etc.)
            let structural = generate_mutations(&rule);
            // Phase B: Context-aware mutations (add conditions, specialize vars, etc.)
            let contextual = generate_context_mutations(&rule, &known_preds, &known_vals);

            // Interleave both — structural first (cheaper), then contextual
            let all_mutations: Vec<_> = structural.into_iter()
                .chain(contextual.into_iter())
                .collect();

            for mutated in all_mutations {
                if start.elapsed() >= budget { break; }

                let text = rule_to_qor(&mutated);
                if seen.contains(&text) { continue; }
                seen.insert(text.clone());

                let score = scorer.score(&text);

                if score > parent.score {
                    improved = true;
                }

                if score > 0.0 {
                    population.push(Candidate {
                        rule_text: text,
                        score,
                        source: format!("evolve-g{}", generation),
                    });
                }

                // Early exit if solved
                if score >= 0.999 {
                    population.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
                    return population;
                }
            }
        }

        // Sort and truncate population
        population.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        population.truncate(max_pop);

        // Try multi-rule combinations every 5 generations
        if generation % 5 == 0 && population.len() >= 2 && start.elapsed() < budget {
            let combos = try_multi_rule_combos(
                &population, &scorer, generation,
            );
            for combo in combos {
                if combo.score > population[0].score {
                    improved = true;
                }
                if !seen.contains(&combo.rule_text) {
                    seen.insert(combo.rule_text.clone());
                    population.push(combo);
                }
            }
            population.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            population.truncate(max_pop);
        }

        if !improved {
            stall_count += 1;
            // Allow a few stall generations before giving up —
            // context mutations might unlock new paths
            if stall_count >= 3 { break; }
        } else {
            stall_count = 0;
        }
    }

    population.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    population
}

/// Extract known predicates and integer values from session + training data.
/// Used by context-aware mutations to propose meaningful new conditions.
fn extract_session_context(
    base_session: &Session,
    training_inputs: &[Vec<Statement>],
) -> (Vec<(String, usize)>, Vec<i64>) {
    let mut pred_set: HashMap<String, usize> = HashMap::new();
    let mut val_set: HashSet<i64> = HashSet::new();

    // From base_session (DNA-derived facts)
    for sn in base_session.all_facts() {
        if let Neuron::Expression(parts) = &sn.neuron {
            if let Some(Neuron::Symbol(name)) = parts.first() {
                pred_set.entry(name.clone()).or_insert(parts.len());
            }
            for p in parts {
                if let Neuron::Value(QorValue::Int(v)) = p {
                    val_set.insert(*v);
                }
            }
        }
    }

    // From training data
    for input in training_inputs {
        for stmt in input {
            if let Statement::Fact { neuron, .. } = stmt {
                if let Neuron::Expression(parts) = neuron {
                    if let Some(Neuron::Symbol(name)) = parts.first() {
                        pred_set.entry(name.clone()).or_insert(parts.len());
                    }
                    for p in parts {
                        if let Neuron::Value(QorValue::Int(v)) = p {
                            val_set.insert(*v);
                        }
                    }
                }
            }
        }
    }

    // Cap values to avoid explosion (keep only most common range)
    let mut vals: Vec<i64> = val_set.into_iter().collect();
    vals.sort();
    // Keep at most 20 values — the most likely relevant ones
    if vals.len() > 20 {
        // Keep first 10, last 10 (covers min/max range)
        let first: Vec<_> = vals[..10].to_vec();
        let last: Vec<_> = vals[vals.len()-10..].to_vec();
        vals = first.into_iter().chain(last.into_iter()).collect();
        vals.sort();
        vals.dedup();
    }

    let preds: Vec<(String, usize)> = pred_set.into_iter().collect();
    (preds, vals)
}

/// Try combining pairs of top rules into multi-rule sets.
/// Two rules together might cover different training pairs better than either alone.
fn try_multi_rule_combos(
    population: &[Candidate],
    scorer: &PrebuiltScorer,
    generation: usize,
) -> Vec<Candidate> {
    let mut combos = Vec::new();
    let top = population.iter().take(5).collect::<Vec<_>>();

    for i in 0..top.len() {
        for j in (i+1)..top.len() {
            // Combine two rules' text — both fire, highest-confidence wins
            let combined_text = format!("{}\n{}", top[i].rule_text, top[j].rule_text);
            let score = scorer.score(&combined_text);
            if score > top[i].score.max(top[j].score) {
                combos.push(Candidate {
                    rule_text: combined_text,
                    score,
                    source: format!("combo-g{}", generation),
                });
            }
        }
    }

    combos
}

// ═══════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use qor_core::truth_value::TruthValue;

    fn make_fact(parts: Vec<Neuron>) -> Statement {
        Statement::Fact {
            neuron: Neuron::Expression(parts),
            tv: Some(TruthValue::new(0.99, 0.99)),
            decay: None,
        }
    }

    fn sym(s: &str) -> Neuron { Neuron::Symbol(s.into()) }
    fn int(v: i64) -> Neuron { Neuron::Value(QorValue::Int(v)) }

    fn data_facts(pred: &str, id: &str, data: &[(i64, i64, i64)]) -> Vec<Statement> {
        data.iter().map(|&(p1, p2, v)| {
            make_fact(vec![sym(pred), sym(id), int(p1), int(p2), int(v)])
        }).collect()
    }

    #[test]
    fn test_profile_facts_discovers_predicates() {
        let facts = vec![
            make_fact(vec![sym("item"), sym("a"), int(0), int(0), int(5)]),
            make_fact(vec![sym("item"), sym("a"), int(0), int(1), int(3)]),
            make_fact(vec![sym("item"), sym("a"), int(1), int(0), int(7)]),
            make_fact(vec![sym("size"), sym("a"), int(2), int(2)]),
        ];
        let refs: Vec<&Statement> = facts.iter().collect();
        let profile = profile_facts(&refs);

        assert!(profile.predicates.contains_key("item"));
        assert!(profile.predicates.contains_key("size"));
        assert_eq!(profile.predicates["item"].len(), 4);
        assert!(profile.predicates["item"][0].is_id, "pos 0 = ID");
        assert!(profile.predicates["item"][1].is_sequential, "pos 1 = sequential");
        assert!(profile.symbols.contains("item"));
        assert!(profile.constants.contains(&5));
    }

    #[test]
    fn test_find_general_contradiction_value_maps() {
        let input = data_facts("item", "a", &[(0, 0, 1), (0, 1, 2), (1, 0, 1), (1, 1, 2)]);
        let output = data_facts("item", "a", &[(0, 0, 3), (0, 1, 2), (1, 0, 3), (1, 1, 2)]);
        let c = find_general_contradiction(&input, &output);

        let map = c.value_maps.get(&("item".into(), 4));
        assert!(map.is_some(), "should find value map at position 4");
        assert_eq!(map.unwrap().get(&1), Some(&3));
        assert!(!c.added.is_empty() || !c.removed.is_empty());
    }

    #[test]
    fn test_find_target_predicate() {
        let in_facts = vec![make_fact(vec![sym("source"), sym("a"), int(0)])];
        let out_facts = vec![make_fact(vec![sym("target"), sym("a"), int(0)])];
        let in_p = profile_facts(&in_facts.iter().collect::<Vec<_>>());
        let out_p = profile_facts(&out_facts.iter().collect::<Vec<_>>());
        assert_eq!(find_target_predicate(&out_p, &in_p), "target");
    }

    #[test]
    fn test_contradiction_new_predicates() {
        let input = vec![make_fact(vec![sym("source"), sym("a"), int(1)])];
        let output = vec![make_fact(vec![sym("target"), sym("a"), int(1)])];
        let c = find_general_contradiction(&input, &output);
        assert!(c.new_predicates.contains("target"));
    }

    #[test]
    fn test_genesis_identity() {
        let session = Session::new();
        let input = data_facts("source", "a", &[(0, 0, 5), (0, 1, 3), (1, 0, 7), (1, 1, 2)]);
        let output = data_facts("target", "a", &[(0, 0, 5), (0, 1, 3), (1, 0, 7), (1, 1, 2)]);

        let results = genesis(&session, &[input], &[output], 3000, None, None);
        assert!(!results.is_empty());
        assert!(results[0].score >= 0.999,
            "identity: {:.3} ({})", results[0].score, results[0].source);
    }

    #[test]
    fn test_genesis_value_remap() {
        let session = Session::new();
        let input = data_facts("source", "a", &[(0, 0, 1), (0, 1, 2), (1, 0, 1), (1, 1, 2)]);
        let output = data_facts("target", "a", &[(0, 0, 3), (0, 1, 2), (1, 0, 3), (1, 1, 2)]);

        let results = genesis(&session, &[input], &[output], 3000, None, None);
        assert!(!results.is_empty());
        assert!(results[0].score >= 0.5,
            "remap: {:.1}% ({})", results[0].score * 100.0, results[0].source);
    }

    #[test]
    fn test_genesis_multiple_pairs() {
        let session = Session::new();
        let input1 = data_facts("source", "a", &[(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)]);
        let output1 = data_facts("target", "a", &[(0, 0, 5), (0, 1, 0), (1, 0, 0), (1, 1, 5)]);
        let input2 = data_facts("source", "b", &[(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]);
        let output2 = data_facts("target", "b", &[(0, 0, 0), (0, 1, 5), (1, 0, 5), (1, 1, 0)]);

        let results = genesis(&session, &[input1, input2], &[output1, output2], 3000, None, None);
        assert!(!results.is_empty());
        assert!(results[0].score >= 0.5,
            "multi-pair: {:.1}%", results[0].score * 100.0);
    }

    #[test]
    fn test_genesis_shift() {
        let session = Session::new();
        let input = data_facts("source", "a", &[
            (0, 0, 5), (0, 1, 3), (0, 2, 7), (1, 0, 2), (1, 1, 4), (1, 2, 6),
        ]);
        let output = data_facts("target", "a", &[
            (0, 1, 5), (0, 2, 3), (0, 3, 7), (1, 1, 2), (1, 2, 4), (1, 3, 6),
        ]);

        let results = genesis(&session, &[input], &[output], 3000, None, None);
        assert!(!results.is_empty());
        assert!(results.iter().any(|c| c.source.contains("shift")) || results[0].score > 0.3,
            "shift: {:.1}% ({})", results[0].score * 100.0, results[0].source);
    }

    #[test]
    fn test_genesis_reflect() {
        let session = Session::new();
        let input = data_facts("source", "a", &[(0, 0, 1), (0, 1, 2), (0, 2, 3)]);
        let output = data_facts("target", "a", &[(0, 2, 1), (0, 1, 2), (0, 0, 3)]);

        let results = genesis(&session, &[input], &[output], 3000, None, None);
        assert!(!results.is_empty());
        assert!(results[0].score >= 0.3,
            "reflect: {:.1}% ({})", results[0].score * 100.0, results[0].source);
    }

    #[test]
    fn test_scamper_mutate() {
        let profile = FactProfile {
            predicates: HashMap::new(),
            constants: vec![0, 1, 2, 3].into_iter().collect(),
            symbols: vec!["source".into(), "target".into()].into_iter().collect(),
        };
        let rule = "(target $a0 $a1 $a2 1) if\n    (source $a0 $a1 $a2 $v) <0.75, 0.75>";
        let mutations = general_scamper_mutate(rule, "test", &profile);
        assert!(!mutations.is_empty());
        assert!(mutations.iter().any(|(t, _)| t.contains(" 2)") || t.contains(" 3)")));
    }

    #[test]
    fn test_guided_mutate_missing() {
        let detail = ScoreDetail {
            score: 0.0,
            wrong_keys: vec![],
            missing_keys: vec!["target a 0 0 1".into()],
            pattern: FailurePattern::MissingFacts,
        };
        let rule = "(target $a0 $a1 $a2 9) if\n    (source $a0 $a1 $a2 99)\n    (> $a1 5) <0.90, 0.90>";
        let mutations = guided_mutate(rule, "test", &detail);
        assert!(!mutations.is_empty());
    }

    #[test]
    fn test_guided_mutate_wrong_values() {
        let detail = ScoreDetail {
            score: 0.5,
            wrong_keys: vec![("target a 0 0 3".into(), "target a 0 0 5".into())],
            missing_keys: vec![],
            pattern: FailurePattern::WrongValues,
        };
        let rule = "(target $a0 $a1 $a2 5) if\n    (source $a0 $a1 $a2 $v) <0.90, 0.90>";
        let mutations = guided_mutate(rule, "test", &detail);
        assert!(!mutations.is_empty());
        assert!(mutations.iter().any(|(t, _)| t.contains(" 3)")),
            "should fix 5→3");
    }

    #[test]
    fn test_genesis_same_predicate_remap() {
        // Same predicate in both input and output — value remap (not new pred)
        let session = Session::new();
        let input = data_facts("item", "a", &[(0, 0, 1), (0, 1, 2)]);
        let output = data_facts("item", "a", &[(0, 0, 3), (0, 1, 2)]);

        let results = genesis(&session, &[input], &[output], 3000, None, None);
        assert!(!results.is_empty());
        assert!(results[0].score >= 0.5,
            "same-pred remap: {:.1}% ({})", results[0].score * 100.0, results[0].source);
    }

    #[test]
    fn test_consistency_scoring_prefers_uniform() {
        // Two rules: one scores 100%+0% (avg=50%), other scores 60%+60% (avg=60%)
        // The consistent rule should score higher
        let session = Session::new();
        let input1 = data_facts("source", "a", &[(0, 0, 1)]);
        let output1 = data_facts("target", "a", &[(0, 0, 1)]);
        let input2 = data_facts("source", "b", &[(0, 0, 2)]);
        let output2 = data_facts("target", "b", &[(0, 0, 2)]);

        // Identity rule should score well on both pairs
        let rule = "(target $a0 $a1 $a2 $a3) if\n    (source $a0 $a1 $a2 $a3) <0.90, 0.90>";
        let score = score_candidate(rule, "target", &session,
            &[input1, input2], &[output1, output2]);
        // Should get 100% on both → avg=1.0, min=1.0 → 0.7*1.0 + 0.3*1.0 = 1.0
        assert!(score >= 0.99, "consistent rule: {:.3}", score);
    }

    #[test]
    fn test_score_detail_tracks_missing() {
        let session = Session::new();
        let input = data_facts("source", "a", &[(0, 0, 1), (0, 1, 2)]);
        let output = data_facts("target", "a", &[(0, 0, 1), (0, 1, 2)]);
        // Rule that won't match anything
        let rule = "(target $a0 $a1 $a2 9) if\n    (source $a0 $a1 $a2 99) <0.90, 0.90>";
        let detail = score_candidate_detailed(rule, "target", &session, &[input], &[output]);
        assert_eq!(detail.score, 0.0);
        assert!(!detail.missing_keys.is_empty());
    }

    #[test]
    fn test_genesis_swarm_identity() {
        // Swarm should find identity rule with multiple workers
        let session = Session::new();
        let input = data_facts("source", "a", &[(0, 0, 1), (0, 1, 2), (1, 0, 3)]);
        let output = data_facts("target", "a", &[(0, 0, 1), (0, 1, 2), (1, 0, 3)]);

        let results = genesis_swarm(
            &session, &[input], &[output], 2000, None, 3, None,
        );
        assert!(!results.is_empty(), "swarm should find at least one candidate");
        assert!(results[0].score >= 0.95,
            "swarm should find identity: got {:.3}", results[0].score);
    }

    #[test]
    fn test_genesis_swarm_remap() {
        // Swarm should find value remap with parallel workers.
        // With 1 training pair the remap may not reach 0.95 (needs remap+identity combo).
        // The key test is that workers run in parallel and find SOMETHING useful.
        let session = Session::new();
        let input = data_facts("source", "a", &[(0, 0, 1), (0, 1, 2), (1, 0, 1)]);
        let output = data_facts("target", "a", &[(0, 0, 5), (0, 1, 2), (1, 0, 5)]);

        let results = genesis_swarm(
            &session, &[input], &[output], 2000, None, 4, None,
        );
        assert!(!results.is_empty(), "swarm should find at least one candidate");
        assert!(results[0].score >= 0.5,
            "swarm should find partial remap: got {:.3}", results[0].score);
    }

    #[test]
    fn test_optimal_worker_count() {
        let n = optimal_worker_count();
        assert!(n >= 1 && n <= 8, "worker count should be reasonable: got {n}");
    }

    #[test]
    fn test_assign_strategies() {
        assert_eq!(assign_strategies_with_hints(1, None).len(), 1);
        assert_eq!(assign_strategies_with_hints(3, None).len(), 3);
        assert_eq!(assign_strategies_with_hints(5, None).len(), 5);
        assert_eq!(assign_strategies_with_hints(8, None).len(), 8);
    }

    #[test]
    fn test_assign_strategies_with_hints() {
        let hints = SwarmHints {
            try_these: vec!["class-remap".into(), "identity".into(), "focus".into()],
            skip_these: vec!["spatial".into()],
            problem_class: None,
            constraints: Vec::new(),
        };
        let strats = assign_strategies_with_hints(6, Some(&hints));
        assert_eq!(strats.len(), 6);
        // Spatial strategies should be removed (skip_these)
        assert!(!strats.iter().any(|s| matches!(s, WorkerStrategy::PositionalTransforms)));
        // Extra DerivedFacts should be added (3 try_these > 2 threshold)
        let df_count = strats.iter().filter(|s| matches!(s, WorkerStrategy::DerivedFacts)).count();
        assert!(df_count >= 2, "Expected extra DerivedFacts workers, got {}", df_count);
    }

}
