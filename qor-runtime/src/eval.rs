use qor_core::neuron::{Neuron, Statement, StoredNeuron};
use qor_core::parser;
use qor_core::truth_value::TruthValue;
use qor_core::unify::{extract_variables, unify};
use std::collections::HashMap;
use std::sync::Arc;

use crate::chain;
use crate::kb::KnowledgeBase;
use crate::store::NeuronStore;

// ── Explanation types ─────────────────────────────────────────────────

/// Why a fact is believed — traces the chain of reasoning.
#[derive(Debug)]
pub struct Explanation {
    pub fact: String,
    pub tv: TruthValue,
    pub reason: ExplanationReason,
}

/// The reason a fact exists.
#[derive(Debug)]
pub enum ExplanationReason {
    /// Directly asserted by the user or data ingestion.
    Asserted,
    /// Derived from a rule + supporting facts.
    Derived {
        rule: String,
        from: Vec<Explanation>,
    },
}

/// Result of testing a hypothesis.
#[derive(Debug)]
pub struct HypothesisResult {
    pub hypothesis: String,
    pub new_facts: usize,
    pub confidence: f64,
    pub facts: Vec<String>,
}

/// Result of running a .qor program.
#[derive(Debug)]
pub struct QueryResult {
    pub pattern: String,
    pub results: Vec<String>,
}

/// Result of executing a single statement in a session.
#[derive(Debug)]
pub enum ExecResult {
    Stored { neuron: String, derived: usize },
    RuleAdded { derived: usize },
    Query(QueryResult),
}

// ── Session (persistent state for REPL) ─────────────────────────────────

#[derive(Clone)]
pub struct Session {
    store: NeuronStore,
    rules: Vec<chain::Rule>,
    consolidation_cycles: usize,
    stored_queries: Vec<qor_core::neuron::Neuron>,
    kb: Option<Arc<KnowledgeBase>>,
}

impl Session {
    pub fn new() -> Self {
        Session {
            store: NeuronStore::new(),
            rules: Vec::new(),
            consolidation_cycles: 0,
            stored_queries: Vec::new(),
            kb: None,
        }
    }

    /// Attach a binary knowledge base to this session.
    /// Uses Arc so cloned sessions share the same KB without copying.
    pub fn set_kb(&mut self, kb: Arc<KnowledgeBase>) {
        self.kb = Some(kb);
    }

    /// Read-only access to the attached knowledge base.
    pub fn kb(&self) -> Option<&KnowledgeBase> {
        self.kb.as_deref()
    }

    /// Pull all facts about an entity from the KB into the NeuronStore.
    /// This materializes binary data as QOR facts for forward chaining.
    pub fn load_entity(&mut self, name: &str) {
        if let Some(kb) = &self.kb {
            let stmts = kb.to_qor_facts(name);
            if !stmts.is_empty() {
                let _ = self.exec_statements(stmts);
            }
        }
    }

    pub fn exec(&mut self, input: &str) -> Result<Vec<ExecResult>, String> {
        let (stmts, _warnings, strata) =
            parser::parse_with_strata(input).map_err(|e| e.to_string())?;
        self.exec_statements_with_strata(stmts, strata)
    }

    /// Execute pre-parsed statements directly (no re-parsing needed).
    /// Used by the ingestion bridge to avoid serialization round-trips.
    pub fn exec_statements(&mut self, stmts: Vec<Statement>) -> Result<Vec<ExecResult>, String> {
        let strata = vec![None; stmts.len()];
        self.exec_statements_with_strata(stmts, strata)
    }

    /// Execute statements with explicit strata annotations.
    /// strata[i] is the @stratum(N) annotation for stmts[i], if any.
    pub fn exec_statements_with_strata(
        &mut self,
        stmts: Vec<Statement>,
        strata: Vec<Option<u32>>,
    ) -> Result<Vec<ExecResult>, String> {
        let mut results = Vec::new();
        let mut needs_chain = false;

        for (i, stmt) in stmts.into_iter().enumerate() {
            match stmt {
                Statement::Fact { neuron, tv, decay } => {
                    let tv = tv.unwrap_or(TruthValue::default_fact());
                    let neuron_str = neuron.to_string();
                    self.store.insert_with_decay(neuron, tv, decay);
                    needs_chain = true;
                    results.push(ExecResult::Stored {
                        neuron: neuron_str,
                        derived: 0,
                    });
                }
                Statement::Rule { head, body, tv } => {
                    let tv = tv.unwrap_or(TruthValue::default_fact());
                    let mut rule = chain::Rule::new(head, body, tv);
                    if let Some(s) = strata.get(i).copied().flatten() {
                        rule.stratum = s;
                    }
                    self.rules.push(rule);
                    needs_chain = true;
                    results.push(ExecResult::RuleAdded { derived: 0 });
                }
                Statement::Test { .. } => {
                    // Tests are not executed during normal eval — use `qor test`
                }
                Statement::Query { pattern } => {
                    // Chain before queries so results are up-to-date
                    if needs_chain {
                        chain::forward_chain(&self.rules, &mut self.store);
                        needs_chain = false;
                    }
                    // Store query patterns for later use by `think`
                    if !self.stored_queries.iter().any(|q| *q == pattern) {
                        self.stored_queries.push(pattern.clone());
                    }
                    let qr = format_query(&pattern, &self.store);
                    results.push(ExecResult::Query(qr));
                }
            }
        }

        // Batch chain: one forward_chain at the end instead of per-fact
        if needs_chain && !self.rules.is_empty() {
            chain::forward_chain(&self.rules, &mut self.store);
        }

        Ok(results)
    }

    pub fn heartbeat(&mut self) -> bool {
        if self.rules.is_empty() && !self.store.all().iter().any(|sn| sn.decay_rate.is_some()) {
            return false;
        }
        let changed = chain::consolidate(&self.rules, &mut self.store);
        if changed {
            self.consolidation_cycles += 1;
        }
        changed
    }

    pub fn fact_count(&self) -> usize {
        self.store.len()
    }

    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    pub fn consolidation_cycles(&self) -> usize {
        self.consolidation_cycles
    }

    pub fn all_facts(&self) -> &[qor_core::neuron::StoredNeuron] {
        self.store.all()
    }

    /// Explain WHY a fact is believed — trace the reasoning chain.
    pub fn explain(&self, pattern: &qor_core::neuron::Neuron) -> Vec<Explanation> {
        let matches = self.store.query(pattern);
        matches.iter().map(|sn| self.explain_fact(sn, 0)).collect()
    }

    fn explain_fact(&self, sn: &qor_core::neuron::StoredNeuron, depth: usize) -> Explanation {
        if depth > 10 || !sn.inferred {
            return Explanation {
                fact: sn.neuron.to_string(),
                tv: sn.tv,
                reason: ExplanationReason::Asserted,
            };
        }

        // Try each rule to find which one derived this fact
        for rule in &self.rules {
            if let Some(bindings) = qor_core::unify::unify(&rule.head, &sn.neuron) {
                if let Some(supporting) = chain::resolve_body_for_explain(&rule.body, &bindings, &self.store) {
                    // Found the rule that derived this — explain supporting facts recursively
                    let body_str: Vec<String> = rule.body.iter().map(|c| format!("{}", c)).collect();
                    let rule_str = format!("{} if {}", rule.head, body_str.join(" "));

                    let from: Vec<Explanation> = supporting.iter().map(|(neuron, tv, inferred)| {
                        if *inferred {
                            // Recursively explain inferred facts
                            let sub_sn = qor_core::neuron::StoredNeuron {
                                neuron: neuron.clone(),
                                tv: *tv,
                                timestamp: None,
                                decay_rate: None,
                                inferred: true,
                            };
                            self.explain_fact(&sub_sn, depth + 1)
                        } else {
                            Explanation {
                                fact: neuron.to_string(),
                                tv: *tv,
                                reason: ExplanationReason::Asserted,
                            }
                        }
                    }).collect();

                    return Explanation {
                        fact: sn.neuron.to_string(),
                        tv: sn.tv,
                        reason: ExplanationReason::Derived { rule: rule_str, from },
                    };
                }
            }
        }

        // No rule found — must be asserted
        Explanation {
            fact: sn.neuron.to_string(),
            tv: sn.tv,
            reason: ExplanationReason::Asserted,
        }
    }

    /// Convert internal rules back to Statement::Rule for use by inference engine.
    /// Access the raw Rule objects (for search/mutation engine).
    pub fn rules(&self) -> &[chain::Rule] {
        &self.rules
    }

    pub fn rules_as_statements(&self) -> Vec<Statement> {
        self.rules
            .iter()
            .map(|r| Statement::Rule {
                head: r.head.clone(),
                body: r.body.clone(),
                tv: Some(r.tv),
            })
            .collect()
    }

    /// Clear ephemeral chat facts (input tokens, intents, responses) between turns.
    pub fn clear_turn(&mut self) {
        self.store.remove_by_predicate(&[
            "input", "compound", "intent", "intent-define", "intent-explain",
            "intent-count", "intent-list", "response",
        ]);
    }

    /// Get response facts from the store as string vectors (parts after "response").
    pub fn response_facts(&self) -> Vec<Vec<String>> {
        self.store
            .iter_predicate("response")
            .map(|sn| {
                if let qor_core::neuron::Neuron::Expression(parts) = &sn.neuron {
                    parts.iter().skip(1).map(|p| match p {
                        qor_core::neuron::Neuron::Symbol(s) => s.clone(),
                        qor_core::neuron::Neuron::Value(qor_core::neuron::QorValue::Str(s)) => s.clone(),
                        other => other.to_string(),
                    }).collect()
                } else {
                    vec![]
                }
            })
            .collect()
    }

    /// Public read-only accessor for the store.
    pub fn store(&self) -> &NeuronStore {
        &self.store
    }

    /// Mutable accessor for the store (for direct fact insertion).
    pub fn store_mut(&mut self) -> &mut NeuronStore {
        &mut self.store
    }

    /// Remove all facts matching the given predicate names.
    pub fn remove_by_predicate(&mut self, predicates: &[&str]) -> usize {
        self.store.remove_by_predicate(predicates)
    }

    /// Test a hypothesis: snapshot store, inject hypothesis facts/rules,
    /// forward chain, measure new derivations, then restore original store.
    pub fn test_hypothesis(&mut self, hypothesis: &str) -> Result<HypothesisResult, String> {
        let snapshot = self.store.clone();
        let snapshot_rules_len = self.rules.len();

        let stmts = parser::parse(hypothesis).map_err(|e| e.to_string())?;
        for stmt in stmts {
            match stmt {
                Statement::Fact { neuron, tv, decay } => {
                    let tv = tv.unwrap_or(TruthValue::default_fact());
                    self.store.insert_with_decay(neuron, tv, decay);
                }
                Statement::Rule { head, body, tv } => {
                    let tv = tv.unwrap_or(TruthValue::default_fact());
                    self.rules.push(chain::Rule::new(head, body, tv));
                }
                Statement::Test { .. } => {} // ignore tests in hypothesis
                Statement::Query { .. } => {} // ignore queries in hypothesis
            }
        }

        chain::forward_chain(&self.rules, &mut self.store);

        // Find new facts: those in current store but not in snapshot
        let mut new_facts = Vec::new();
        let mut total_conf = 0.0;
        for sn in self.store.all() {
            if !snapshot.contains(&sn.neuron) {
                new_facts.push(sn.neuron.to_string());
                total_conf += sn.tv.confidence;
            }
        }

        let new_count = new_facts.len();
        let avg_conf = if new_count > 0 { total_conf / new_count as f64 } else { 0.0 };

        // Restore
        self.store = snapshot;
        self.rules.truncate(snapshot_rules_len);

        Ok(HypothesisResult {
            hypothesis: hypothesis.to_string(),
            new_facts: new_count,
            confidence: avg_conf,
            facts: new_facts,
        })
    }

    /// Get stored query patterns (accumulated from loaded .qor files).
    pub fn queries(&self) -> &[qor_core::neuron::Neuron] {
        &self.stored_queries
    }

    /// Execute all stored queries and return their results.
    pub fn run_queries(&self) -> Vec<QueryResult> {
        self.stored_queries
            .iter()
            .map(|pattern| format_query(pattern, &self.store))
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Multi-Perspective Reasoning — 11 domain swarm + AGI combiner
// ═══════════════════════════════════════════════════════════════════════

/// A domain of knowledge (one meta file).
#[derive(Clone, Debug)]
pub struct MetaDomain {
    pub name: String,
    pub statements: Vec<Statement>,
}

/// What one domain perspective saw in the problem.
#[derive(Debug)]
pub struct DomainInsight {
    pub domain: String,
    pub inferred_count: usize,
    /// Predicates this domain's rules derived, with counts.
    pub predicates: Vec<(String, usize)>,
    /// Unique facts this domain derived that other domains did NOT.
    pub unique_facts: Vec<String>,
    /// All inferred facts from this domain (as StoredNeuron snapshots).
    pub all_inferred: Vec<(String, TruthValue)>,
    pub elapsed_ms: u64,
}

/// 6D State Vector from agi.qor — scores every insight across 6 dimensions.
///
/// R⃗ = (f, n, s, e, c, i) where:
///   f = feasibility    — does it match observed evidence?
///   n = novelty        — is this a unique perspective?
///   s = simplicity     — Occam's razor (atomic facts = max simple)
///   e = ethics         — respects constraints? (gate, not score)
///   c = consistency    — confidence across sources
///   i = information    — how much of the problem does it cover?
#[derive(Debug, Clone)]
pub struct StateVector6D {
    pub feasibility: f64,
    pub novelty: f64,
    pub simplicity: f64,
    pub ethics: f64,
    pub consistency: f64,
    pub information: f64,
}

impl StateVector6D {
    /// Combined score: project 6D → 1D for ranking.
    /// Weights from agi.qor Section 6:
    ///   0.35*f + 0.20*c + 0.15*s + 0.15*n + 0.10*i + 0.05*e
    pub fn combined_score(&self) -> f64 {
        0.35 * self.feasibility
            + 0.20 * self.consistency
            + 0.15 * self.simplicity
            + 0.15 * self.novelty
            + 0.10 * self.information
            + 0.05 * self.ethics
    }

    /// Tensor product ⊗ for chain propagation (agi.qor Section 4).
    /// Each dimension has a DIFFERENT operator:
    ///   feasibility:  multiply   — chain weakens confidence
    ///   novelty:      max        — novelty of the most novel part
    ///   simplicity:   min        — complexity of the most complex part
    ///   ethics:       min        — weakest ethical link breaks chain
    ///   consistency:  multiply   — inconsistency compounds
    ///   information:  prob-OR    — combined info: 1-(1-a)(1-b)
    pub fn chain(&self, other: &StateVector6D) -> StateVector6D {
        StateVector6D {
            feasibility: self.feasibility * other.feasibility,
            novelty: self.novelty.max(other.novelty),
            simplicity: self.simplicity.min(other.simplicity),
            ethics: self.ethics.min(other.ethics),
            consistency: self.consistency * other.consistency,
            information: self.information + other.information
                - self.information * other.information,
        }
    }
}

/// An insight scored through the 6D framework.
#[derive(Debug, Clone)]
pub struct ScoredInsight {
    /// The fact string.
    pub fact: String,
    /// 6D state vector.
    pub state: StateVector6D,
    /// Combined 1D score (projection of state vector).
    pub combined_score: f64,
    /// Which domains produced this insight.
    pub source_domains: Vec<String>,
    /// Accepted (above threshold + ethics gate) or rejected.
    pub accepted: bool,
}

/// Combined result of multi-perspective reasoning.
#[derive(Debug)]
pub struct ReasonResult {
    /// Per-domain breakdown — what each perspective contributed.
    pub insights: Vec<DomainInsight>,
    /// 6D-scored insights — ranked by combined score, filtered by ethics gate.
    pub scored_insights: Vec<ScoredInsight>,
    /// Facts agreed upon by multiple domains (consensus).
    pub consensus_facts: Vec<(String, usize, f64)>, // (fact, domain_count, avg_strength)
    /// Total unique inferred facts across all domains.
    pub total_unique_inferred: usize,
    /// Total time.
    pub elapsed_ms: u64,
}

impl ReasonResult {
    /// Print full reasoning report: per-domain insights + 6D AGI scores.
    pub fn print_report(&self) {
        // Per-domain insights
        eprintln!("  {:<25} {:>6} {:>8} {:>6}  top predicates",
            "DOMAIN", "INFER", "UNIQUE", "TIME");
        eprintln!("  {:-<80}", "");

        for insight in &self.insights {
            let top_preds: String = insight.predicates.iter()
                .take(5)
                .map(|(p, c)| format!("{}:{}", p, c))
                .collect::<Vec<_>>()
                .join(" ");

            eprintln!("  {:<25} {:>6} {:>8} {:>4}ms  {}",
                insight.domain,
                insight.inferred_count,
                insight.unique_facts.len(),
                insight.elapsed_ms,
                top_preds);
        }

        // Unique contributions
        eprintln!("\n  Unique Contributions:");
        for insight in &self.insights {
            if !insight.unique_facts.is_empty() {
                eprintln!("    {} ({}):", insight.domain, insight.unique_facts.len());
                for fact in insight.unique_facts.iter().take(5) {
                    eprintln!("      {}", fact);
                }
                if insight.unique_facts.len() > 5 {
                    eprintln!("      ... +{} more", insight.unique_facts.len() - 5);
                }
            }
        }

        // 6D AGI scoring
        let accepted_count = self.scored_insights.iter().filter(|s| s.accepted).count();
        let rejected_count = self.scored_insights.len() - accepted_count;

        eprintln!("\n  6D AGI Scoring ({} accepted, {} rejected):", accepted_count, rejected_count);
        eprintln!("  {:<50} {:>5} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4}  {}",
            "INSIGHT", "SCORE", "F", "N", "S", "E", "C", "I", "DOMAINS");
        eprintln!("  {:-<120}", "");

        for si in self.scored_insights.iter().take(20) {
            let mark = if si.accepted { "+" } else { "-" };
            let domains_str: String = si.source_domains.iter()
                .map(|d| abbreviate_domain(d))
                .collect::<Vec<_>>()
                .join(",");

            let fact_short = if si.fact.len() > 48 {
                format!("{}...", &si.fact[..45])
            } else {
                si.fact.clone()
            };

            eprintln!("  {} {:<48} {:>5.3} {:>.2} {:>.2} {:>.2} {:>.2} {:>.2} {:>.2}  [{}]",
                mark, fact_short, si.combined_score,
                si.state.feasibility, si.state.novelty, si.state.simplicity,
                si.state.ethics, si.state.consistency, si.state.information,
                domains_str);
        }
        if self.scored_insights.len() > 20 {
            eprintln!("  ... +{} more insights", self.scored_insights.len() - 20);
        }

        // Consensus
        if !self.consensus_facts.is_empty() {
            eprintln!("\n  Consensus ({}):", self.consensus_facts.len());
            for (fact, count, avg_str) in self.consensus_facts.iter().take(10) {
                eprintln!("    [{}x] {:.2} — {}", count, avg_str, fact);
            }
        }

        eprintln!("\n  Summary: {} unique | {} accepted | {} consensus | {}ms",
            self.total_unique_inferred, accepted_count,
            self.consensus_facts.len(), self.elapsed_ms);
    }
}

fn abbreviate_domain(d: &str) -> &str {
    match d {
        "math" => "Ma", "physics" => "Ph", "chemistry" => "Ch",
        "biology" => "Bi", "computer_science" => "CS", "astronomy_earth" => "As",
        "general" => "Ge", "genesis" => "Gn", "perception" => "Pe",
        "sanskrit" => "Sa", "vedic_science" => "Ve",
        other => if other.len() >= 2 { &other[..2] } else { other },
    }
}

impl Session {
    /// Multi-perspective reasoning: run the same problem through N domain lenses
    /// in parallel, then combine insights.
    ///
    /// Each domain gets its OWN session clone with its meta loaded on top.
    /// All domains see the same problem data and DNA rules.
    /// Results are combined: facts seen by multiple domains get boosted confidence.
    ///
    /// `domains`: the 11 meta domain perspectives (math, physics, etc.)
    /// `problem`: the problem data as statements (grid facts, user input, etc.)
    /// `heartbeats`: how many reasoning cycles each worker runs
    pub fn reason(
        &self,
        domains: &[MetaDomain],
        problem: Vec<Statement>,
        heartbeats: usize,
    ) -> ReasonResult {
        let start = std::time::Instant::now();

        // Snapshot base fact keys before spawning domain workers
        let base_fact_keys: std::collections::HashSet<String> = self.store.all().iter()
            .map(|sn| sn.neuron.to_string())
            .collect();

        // Run each domain perspective in parallel
        let domain_results: Vec<(String, Vec<(String, TruthValue)>, u64)> = {
            let base_keys = std::sync::Arc::new(base_fact_keys);
            let handles: Vec<_> = domains.iter().map(|domain| {
                let mut worker = self.clone();
                let domain_stmts = domain.statements.clone();
                let domain_name = domain.name.clone();
                let problem_clone = problem.clone();
                let hb = heartbeats;
                let keys = base_keys.clone();

                std::thread::spawn(move || {
                    let t0 = std::time::Instant::now();

                    // Load this domain's meta knowledge
                    let _ = worker.exec_statements(domain_stmts);

                    // Feed the problem
                    let _ = worker.exec_statements(problem_clone);

                    // Reason: heartbeat cycles
                    for _ in 0..hb {
                        worker.heartbeat();
                    }

                    // Collect all new facts (inferred + domain-asserted, excluding base)
                    let inferred: Vec<(String, TruthValue)> = worker.store.all().iter()
                        .filter(|sn| sn.inferred || !keys.contains(&sn.neuron.to_string()))
                        .map(|sn| (sn.neuron.to_string(), sn.tv))
                        .collect();

                    let elapsed = t0.elapsed().as_millis() as u64;
                    (domain_name, inferred, elapsed)
                })
            }).collect();

            handles.into_iter().filter_map(|h| h.join().ok()).collect()
        };

        // Also run the base session (no extra meta) as control
        let mut base_worker = self.clone();
        let _ = base_worker.exec_statements(problem.clone());
        for _ in 0..heartbeats {
            base_worker.heartbeat();
        }
        let base_inferred: std::collections::HashSet<String> = base_worker.store.all().iter()
            .filter(|sn| sn.inferred)
            .map(|sn| sn.neuron.to_string())
            .collect();

        // Build per-domain insights
        let mut all_domain_facts: HashMap<String, Vec<(String, TruthValue)>> = HashMap::new();
        let mut insights = Vec::new();

        for (domain_name, inferred, elapsed) in &domain_results {
            // Count predicates
            let mut pred_counts: HashMap<String, usize> = HashMap::new();
            for (fact_str, _tv) in inferred {
                // Extract predicate (first symbol in the expression)
                if let Some(pred) = fact_str.strip_prefix('(')
                    .and_then(|s| s.split_whitespace().next())
                {
                    *pred_counts.entry(pred.to_string()).or_default() += 1;
                }
            }
            let mut preds: Vec<(String, usize)> = pred_counts.into_iter().collect();
            preds.sort_by(|a, b| b.1.cmp(&a.1));

            // Find facts unique to THIS domain (not in base)
            let unique: Vec<String> = inferred.iter()
                .filter(|(f, _)| !base_inferred.contains(f))
                .map(|(f, _)| f.clone())
                .collect();

            // Track for consensus analysis
            for (fact_str, tv) in inferred {
                all_domain_facts.entry(fact_str.clone())
                    .or_default()
                    .push((domain_name.clone(), *tv));
            }

            insights.push(DomainInsight {
                domain: domain_name.clone(),
                inferred_count: inferred.len(),
                predicates: preds,
                unique_facts: unique,
                all_inferred: inferred.clone(),
                elapsed_ms: *elapsed,
            });
        }

        // Consensus: facts seen by 2+ domains (excluding base-common facts)
        let mut consensus: Vec<(String, usize, f64)> = all_domain_facts.iter()
            .filter(|(fact, domains)| {
                domains.len() >= 2 && !base_inferred.contains(fact.as_str())
            })
            .map(|(fact, domains)| {
                let avg_str = domains.iter().map(|(_, tv)| tv.strength).sum::<f64>()
                    / domains.len() as f64;
                (fact.clone(), domains.len(), avg_str)
            })
            .collect();
        consensus.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)));

        // Total unique inferred across all domains (excluding base)
        let total_unique = all_domain_facts.keys()
            .filter(|f| !base_inferred.contains(f.as_str()))
            .count();

        // ═══════════════════════════════════════════════════════════════
        // 6D SCORING — agi.qor pipeline implemented in Rust
        // Score every unique insight through the 6-dimensional framework
        // ═══════════════════════════════════════════════════════════════

        let total_domains = domains.len() as f64;

        let mut scored_insights: Vec<ScoredInsight> = all_domain_facts.iter()
            .filter(|(fact, _)| !base_inferred.contains(fact.as_str()))
            .map(|(fact, sources)| {
                let source_count = sources.len() as f64;

                // Average TV strength across all domains that produced this
                let avg_strength = sources.iter()
                    .map(|(_, tv)| tv.strength).sum::<f64>() / source_count;

                // Worst-case confidence (consistency = weakest link)
                let min_confidence = sources.iter()
                    .map(|(_, tv)| tv.confidence)
                    .fold(f64::MAX, f64::min);

                // Relevance: does this insight relate to the problem?
                // Domain-tagged insights (*-insight) are relevant.
                // Internal domain artifacts (syllable-weight, sandhi, etc.) are not.
                let is_insight_pred = fact.contains("-insight")
                    || fact.starts_with("(genesis-hint")
                    || fact.starts_with("(try-genesis")
                    || fact.starts_with("(no-predictions");
                let relevance = if is_insight_pred { 1.0 } else { 0.3 };

                let state = StateVector6D {
                    // Feasibility: evidence strength × relevance
                    // Weighted: 70% avg strength + 30% agreement ratio
                    feasibility: relevance * (0.70 * avg_strength
                        + 0.30 * (source_count / total_domains)),

                    // Novelty: unique perspectives are more valuable
                    // 1 domain → 0.91 novel, all 11 → 0.0
                    novelty: 1.0 - (source_count / total_domains),

                    // Simplicity: insight facts = simple, artifacts = less simple
                    simplicity: if is_insight_pred { 1.0 } else { 0.5 },

                    // Ethics: observations are neutral (gate = pass)
                    ethics: 1.0,

                    // Consistency: worst-case confidence across sources
                    consistency: min_confidence,

                    // Information: coverage × relevance
                    information: relevance * (source_count / total_domains),
                };

                let score = state.combined_score();
                let source_names: Vec<String> = sources.iter()
                    .map(|(d, _)| d.clone()).collect();

                // Acceptance: ethics gate + score > 0.50 (fixed threshold for insights)
                // agi.qor's adaptive threshold is for rule candidates (few proposals).
                // For insights we use a fixed bar — accept useful signal, reject noise.
                let accepted = state.ethics > 0.5 && score > 0.50;

                ScoredInsight {
                    fact: fact.clone(),
                    state,
                    combined_score: score,
                    source_domains: source_names,
                    accepted,
                }
            })
            .collect();

        // Rank by combined score (highest first)
        scored_insights.sort_by(|a, b| {
            b.combined_score.partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        ReasonResult {
            insights,
            scored_insights,
            consensus_facts: consensus,
            total_unique_inferred: total_unique,
            elapsed_ms: start.elapsed().as_millis() as u64,
        }
    }

    /// Load meta domain files from a directory. Returns Vec<MetaDomain>.
    /// Skips agi.qor (that's the combiner, loaded separately).
    pub fn load_meta_domains(meta_dir: &std::path::Path) -> Vec<MetaDomain> {
        let mut domains = Vec::new();
        if let Ok(entries) = std::fs::read_dir(meta_dir) {
            let mut paths: Vec<_> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.extension().map(|x| x == "qor").unwrap_or(false)
                        && p.file_stem().map(|s| s != "agi").unwrap_or(true)
                })
                .collect();
            paths.sort();
            for path in paths {
                if let Ok(source) = std::fs::read_to_string(&path) {
                    match parser::parse(&source) {
                        Ok(stmts) if !stmts.is_empty() => {
                            let name = path.file_stem().unwrap()
                                .to_string_lossy().to_string();
                            domains.push(MetaDomain { name, statements: stmts });
                        }
                        _ => {}
                    }
                }
            }
        }
        domains
    }

    /// Load all .qor files from a directory into a new session.
    /// Files are loaded in sorted order.
    pub fn load_qor_dir(dir: &std::path::Path) -> Self {
        let mut session = Session::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            let mut paths: Vec<_> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map(|x| x == "qor").unwrap_or(false))
                .collect();
            paths.sort();
            for path in &paths {
                if let Ok(source) = std::fs::read_to_string(path) {
                    if let Ok(stmts) = parser::parse(&source) {
                        let _ = session.exec_statements(stmts);
                    }
                }
            }
        }
        session
    }

    /// Check if any fact with the given predicate exists.
    pub fn has_fact(&self, predicate: &str) -> bool {
        self.store.all().iter().any(|f| {
            if let Neuron::Expression(parts) = &f.neuron {
                if let Some(Neuron::Symbol(p)) = parts.first() {
                    return p == predicate;
                }
            }
            false
        })
    }

    /// Get all facts matching a given predicate name.
    pub fn facts_with_predicate(&self, predicate: &str) -> Vec<&StoredNeuron> {
        self.store.all().iter()
            .filter(|f| {
                if let Neuron::Expression(parts) = &f.neuron {
                    if let Some(Neuron::Symbol(p)) = parts.first() {
                        return p == predicate;
                    }
                }
                false
            })
            .collect()
    }

    /// Clone session keeping only rules + observation/detection facts.
    /// Produces a clean base for genesis (no raw data facts).
    pub fn clone_obs_only(&self) -> Self {
        let mut base = Session::new();
        for stmt in self.rules_as_statements() {
            let _ = base.exec_statements(vec![stmt]);
        }
        for fact in self.store.all() {
            if let Neuron::Expression(parts) = &fact.neuron {
                if let Some(Neuron::Symbol(pred)) = parts.first() {
                    if pred.starts_with("obs-") || pred.starts_with("detected-")
                        || pred.starts_with("consistent-") || pred.starts_with("dna-")
                        || pred == "target-predicate" || pred == "source-predicate"
                        || pred == "answer-predicate"
                        || pred.starts_with("save-known-") || pred.starts_with("item-feature")
                        || pred == "train-pair" || pred == "tile-block"
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
        base
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

// ── Query formatting ────────────────────────────────────────────────────

/// Format query results, showing variable bindings when the pattern has variables.
fn format_query(pattern: &qor_core::neuron::Neuron, store: &NeuronStore) -> QueryResult {
    let matches = store.query(pattern);
    let pattern_str = pattern.to_string();
    let vars = extract_variables(pattern);

    if matches.is_empty() {
        return QueryResult {
            pattern: pattern_str,
            results: vec!["no results".to_string()],
        };
    }

    let result_strings: Vec<String> = if vars.is_empty() {
        // No variables — show full expression
        matches.iter().map(|sn| sn.to_string()).collect()
    } else {
        // Has variables — show bindings
        matches
            .iter()
            .filter_map(|sn| {
                let bindings = unify(pattern, &sn.neuron)?;
                let binding_strs: Vec<String> = vars
                    .iter()
                    .filter_map(|v| bindings.get(v).map(|val| format!("${} = {}", v, val)))
                    .collect();
                let inferred_tag = if sn.inferred { " (inferred)" } else { "" };
                let decay_tag = match sn.decay_rate {
                    Some(r) => format!(" @decay {:.2}", r),
                    None => String::new(),
                };
                Some(format!(
                    "{}  {}{}{}",
                    binding_strs.join(", "),
                    sn.tv,
                    decay_tag,
                    inferred_tag
                ))
            })
            .collect()
    };

    QueryResult {
        pattern: pattern_str,
        results: result_strings,
    }
}

// ── Batch mode ──────────────────────────────────────────────────────────

pub fn run(source: &str) -> Result<Vec<QueryResult>, String> {
    let (statements, _warnings, strata) =
        parser::parse_with_strata(source).map_err(|e| e.to_string())?;
    let mut store = NeuronStore::new();
    let mut rules: Vec<chain::Rule> = Vec::new();
    let mut query_results = Vec::new();
    let mut needs_forward_chain = false;

    for (i, stmt) in statements.into_iter().enumerate() {
        match stmt {
            Statement::Fact { neuron, tv, decay } => {
                let tv = tv.unwrap_or(TruthValue::default_fact());
                store.insert_with_decay(neuron, tv, decay);
                if !rules.is_empty() {
                    needs_forward_chain = true;
                }
            }
            Statement::Rule { head, body, tv } => {
                let tv = tv.unwrap_or(TruthValue::default_fact());
                let mut rule = chain::Rule::new(head, body, tv);
                if let Some(s) = strata.get(i).copied().flatten() {
                    rule.stratum = s;
                }
                rules.push(rule);
                needs_forward_chain = true;
            }
            Statement::Test { .. } => {} // tests not executed in run mode
            Statement::Query { pattern } => {
                if needs_forward_chain {
                    chain::forward_chain(&rules, &mut store);
                    needs_forward_chain = false;
                }
                query_results.push(format_query(&pattern, &store));
            }
        }
    }

    Ok(query_results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_milestone_program() {
        let input = r#"
            (bird tweety)     <0.99>
            (bird eagle)      <0.95>
            (fish salmon)     <0.99>

            ? (bird tweety)
            ? (bird $x)
            ? (fish $x)
            ? (mammal $x)
        "#;

        let results = run(input).unwrap();
        assert_eq!(results.len(), 4);

        // Concrete query — shows full expression
        assert_eq!(results[0].results.len(), 1);
        assert!(results[0].results[0].contains("bird tweety"));

        // Variable query — shows bindings (order not guaranteed)
        assert_eq!(results[1].results.len(), 2);
        let bird_all = results[1].results.join(" | ");
        assert!(bird_all.contains("$x = tweety"), "missing tweety in: {bird_all}");
        assert!(bird_all.contains("$x = eagle"), "missing eagle in: {bird_all}");

        assert_eq!(results[3].results[0], "no results");
    }

    #[test]
    fn test_run_belief_revision() {
        let input = r#"
            (bird tweety) <0.70, 0.40>
            (bird tweety) <0.95, 0.80>
            ? (bird tweety)
        "#;

        let results = run(input).unwrap();
        let result = &results[0].results[0];
        assert!(result.contains("bird tweety"));
        assert!(result.contains("0.88") || result.contains("0.87") || result.contains("0.89"));
    }

    #[test]
    fn test_run_with_rules_forward_chain() {
        let input = r#"
            (bird tweety) <0.99>
            (bird eagle)  <0.95>
            (fish salmon) <0.99>
            (flies $x) if (bird $x) <0.95>
            (swims $x) if (fish $x) <0.98>

            ? (flies $x)
            ? (swims $x)
            ? (swims eagle)
        "#;

        let results = run(input).unwrap();

        // ? (flies $x) — two bindings (order not guaranteed)
        assert_eq!(results[0].results.len(), 2);
        let flies_all = results[0].results.join(" | ");
        assert!(flies_all.contains("$x = tweety"), "missing tweety in: {flies_all}");
        assert!(flies_all.contains("$x = eagle"), "missing eagle in: {flies_all}");
        assert!(flies_all.contains("inferred"), "missing inferred tag in: {flies_all}");

        // ? (swims eagle) — no results
        assert_eq!(results[2].results[0], "no results");
    }

    #[test]
    fn test_run_rule_chaining() {
        let input = r#"
            (bird tweety) <0.99>
            (flies $x) if (bird $x) <0.95>
            (has-wings $x) if (flies $x) <0.99>
            ? (has-wings $x)
        "#;

        let results = run(input).unwrap();
        assert_eq!(results[0].results.len(), 1);
        assert!(results[0].results[0].contains("$x = tweety"));
        assert!(results[0].results[0].contains("inferred"));
    }

    #[test]
    fn test_run_negation() {
        let input = r#"
            (bird tweety)  <0.99>
            (bird tux)     <0.99>
            (penguin tux)  <0.99>
            (can-fly $x) if (bird $x) not (penguin $x) <0.95>
            ? (can-fly $x)
        "#;

        let results = run(input).unwrap();
        assert_eq!(results[0].results.len(), 1);
        assert!(results[0].results[0].contains("tweety"));
        assert!(!results[0].results[0].contains("tux"));
    }

    #[test]
    fn test_run_multi_condition_rule() {
        let input = r#"
            (bird tweety)     <0.99>
            (healthy tweety)  <0.90>
            (bird eagle)      <0.95>
            (can-fly $x) if (bird $x) (healthy $x) <0.90>
            ? (can-fly $x)
        "#;

        let results = run(input).unwrap();
        assert_eq!(results[0].results.len(), 1);
        assert!(results[0].results[0].contains("tweety"));
    }

    #[test]
    fn test_run_direct_and_inferred_merge() {
        let input = r#"
            (flies tweety) <0.80>
            (bird tweety) <0.99>
            (flies $x) if (bird $x) <0.95>
            ? (flies tweety)
        "#;

        let results = run(input).unwrap();
        assert_eq!(results[0].results.len(), 1);
        assert!(!results[0].results[0].contains("inferred"));
    }

    #[test]
    fn test_run_temporal_decay() {
        let input = r#"
            (trending bitcoin) <0.95> @decay 0.10
            (bird tweety) <0.99>
            ? (trending $x)
        "#;

        let results = run(input).unwrap();
        assert_eq!(results[0].results.len(), 1);
        assert!(results[0].results[0].contains("$x = bitcoin"));
        assert!(results[0].results[0].contains("@decay"));
    }

    // ── Session tests ──

    #[test]
    fn test_session_persistent_state() {
        let mut session = Session::new();

        session.exec("(bird tweety) <0.99>").unwrap();
        session.exec("(bird eagle) <0.95>").unwrap();
        assert_eq!(session.fact_count(), 2);

        let results = session.exec("? (bird $x)").unwrap();
        match &results[0] {
            ExecResult::Query(qr) => assert_eq!(qr.results.len(), 2),
            _ => panic!("expected query result"),
        }
    }

    #[test]
    fn test_session_forward_chain_on_insert() {
        let mut session = Session::new();

        session.exec("(flies $x) if (bird $x) <0.95>").unwrap();
        session.exec("(bird tweety) <0.99>").unwrap();
        // Batch chaining: derived fact (flies tweety) is produced after all facts
        assert_eq!(session.fact_count(), 2); // bird tweety + flies tweety
    }

    #[test]
    fn test_session_stores_queries() {
        let mut session = Session::new();
        session.exec("(bird tweety) <0.99>").unwrap();
        session.exec("? (bird $x)").unwrap();
        session.exec("? (flies $x)").unwrap();
        assert_eq!(session.queries().len(), 2);
    }

    #[test]
    fn test_session_run_queries() {
        let mut session = Session::new();
        session.exec("(bird tweety) <0.99>").unwrap();
        session.exec("(bird eagle) <0.95>").unwrap();
        session.exec("? (bird $x)").unwrap();

        let results = session.run_queries();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].results.len(), 2);
    }

    #[test]
    fn test_session_no_duplicate_queries() {
        let mut session = Session::new();
        session.exec("? (bird $x)").unwrap();
        session.exec("? (bird $x)").unwrap();
        assert_eq!(session.queries().len(), 1);
    }

    #[test]
    fn test_explain_asserted_fact() {
        let mut session = Session::new();
        session.exec("(bird tweety) <0.99>").unwrap();

        let pattern = qor_core::neuron::Neuron::expression(vec![
            qor_core::neuron::Neuron::symbol("bird"),
            qor_core::neuron::Neuron::symbol("tweety"),
        ]);
        let explanations = session.explain(&pattern);
        assert_eq!(explanations.len(), 1);
        assert!(matches!(explanations[0].reason, ExplanationReason::Asserted));
        assert!(explanations[0].fact.contains("tweety"));
    }

    #[test]
    fn test_explain_derived_fact() {
        let mut session = Session::new();
        session.exec("(bird tweety) <0.99>").unwrap();
        session.exec("(flies $x) if (bird $x) <0.95>").unwrap();

        let pattern = qor_core::neuron::Neuron::expression(vec![
            qor_core::neuron::Neuron::symbol("flies"),
            qor_core::neuron::Neuron::symbol("tweety"),
        ]);
        let explanations = session.explain(&pattern);
        assert_eq!(explanations.len(), 1);
        match &explanations[0].reason {
            ExplanationReason::Derived { rule, from } => {
                assert!(rule.contains("flies"));
                assert!(rule.contains("bird"));
                assert_eq!(from.len(), 1);
                assert!(from[0].fact.contains("tweety"));
            }
            _ => panic!("expected Derived explanation"),
        }
    }

    #[test]
    fn test_explain_chain() {
        let mut session = Session::new();
        session.exec("(bird tweety) <0.99>").unwrap();
        session.exec("(flies $x) if (bird $x) <0.95>").unwrap();
        session.exec("(has-wings $x) if (flies $x) <0.99>").unwrap();

        let pattern = qor_core::neuron::Neuron::expression(vec![
            qor_core::neuron::Neuron::symbol("has-wings"),
            qor_core::neuron::Neuron::symbol("tweety"),
        ]);
        let explanations = session.explain(&pattern);
        assert_eq!(explanations.len(), 1);
        match &explanations[0].reason {
            ExplanationReason::Derived { from, .. } => {
                assert_eq!(from.len(), 1);
                // The supporting fact (flies tweety) should also be Derived
                assert!(matches!(from[0].reason, ExplanationReason::Derived { .. }));
            }
            _ => panic!("expected Derived explanation"),
        }
    }

    #[test]
    fn test_explain_no_match() {
        let mut session = Session::new();
        session.exec("(bird tweety) <0.99>").unwrap();

        let pattern = qor_core::neuron::Neuron::expression(vec![
            qor_core::neuron::Neuron::symbol("fish"),
            qor_core::neuron::Neuron::symbol("salmon"),
        ]);
        let explanations = session.explain(&pattern);
        assert!(explanations.is_empty());
    }

    #[test]
    fn test_session_rules_as_statements() {
        let mut session = Session::new();
        session.exec("(flies $x) if (bird $x) <0.95>").unwrap();
        session.exec("(swims $x) if (fish $x) <0.98>").unwrap();

        let stmts = session.rules_as_statements();
        assert_eq!(stmts.len(), 2);
        for stmt in &stmts {
            assert!(matches!(stmt, qor_core::neuron::Statement::Rule { .. }));
        }
    }

    #[test]
    fn test_session_rules_as_statements_preserves_guards() {
        let mut session = Session::new();
        session.exec("(overbought $x) if (rsi $x $v) (> $v 70) <0.90>").unwrap();

        let stmts = session.rules_as_statements();
        assert_eq!(stmts.len(), 1);
        if let qor_core::neuron::Statement::Rule { body, .. } = &stmts[0] {
            let has_guard = body.iter().any(|c| matches!(c, qor_core::neuron::Condition::Guard(..)));
            assert!(has_guard, "should preserve guard condition");
        } else {
            panic!("expected Rule");
        }
    }

    #[test]
    fn test_session_rules_as_statements_empty() {
        let session = Session::new();
        assert!(session.rules_as_statements().is_empty());
    }

    #[test]
    fn test_clear_turn_removes_ephemeral() {
        let mut session = Session::new();
        session.exec("(input 0 hello) <0.99>").unwrap();
        session.exec("(input 1 world) <0.99>").unwrap();
        session.exec("(intent greet) <0.95>").unwrap();
        session.exec("(response greet hello) <0.90>").unwrap();
        session.exec("(meaning overbought high) <0.99>").unwrap();

        assert_eq!(session.fact_count(), 5);
        session.clear_turn();
        assert_eq!(session.fact_count(), 1); // only meaning preserved
    }

    #[test]
    fn test_response_facts_extracts() {
        let mut session = Session::new();
        session.exec("(response greet hello) <0.95>").unwrap();
        session.exec("(response define overbought high-pressure) <0.90>").unwrap();
        session.exec("(bird tweety) <0.99>").unwrap();

        let responses = session.response_facts();
        assert_eq!(responses.len(), 2);
        assert!(responses.iter().any(|r| r[0] == "greet"));
        assert!(responses.iter().any(|r| r[0] == "define"));
    }

    #[test]
    fn test_run_guard_conditions() {
        let input = r#"
            (rsi btc 75) <0.99>
            (rsi eth 45) <0.99>
            (rsi sol 82) <0.99>
            (overbought $x) if (rsi $x $v) (> $v 70) <0.90>
            ? (overbought $x)
        "#;

        let results = run(input).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].results.len(), 2); // btc (75>70) and sol (82>70)
        let all = results[0].results.join(" ");
        assert!(all.contains("btc"));
        assert!(all.contains("sol"));
        assert!(!all.contains("eth")); // 45 is not > 70
    }

    #[test]
    fn test_run_guard_with_negation() {
        let input = r#"
            (rsi btc 25) <0.99>
            (rsi eth 15) <0.99>
            (blacklisted eth) <0.99>
            (buy-signal $x) if (rsi $x $v) (< $v 30) not (blacklisted $x) <0.85>
            ? (buy-signal $x)
        "#;

        let results = run(input).unwrap();
        assert_eq!(results[0].results.len(), 1); // only btc (eth is blacklisted)
        assert!(results[0].results[0].contains("btc"));
    }

    #[test]
    fn test_run_aggregate_count() {
        let input = r#"
            (bird tweety) <0.99>
            (bird eagle) <0.95>
            (bird robin) <0.90>
            (fish salmon) <0.99>
            (bird-count $n) if (count (bird $x) $x -> $n) <0.99>
            ? (bird-count $n)
        "#;

        let results = run(input).unwrap();
        assert_eq!(results[0].results.len(), 1);
        assert!(results[0].results[0].contains("3")); // 3 birds
    }

    #[test]
    fn test_aggregate_in_explain() {
        let mut session = Session::new();
        session.exec("(bird tweety) <0.99>").unwrap();
        session.exec("(bird eagle) <0.95>").unwrap();
        session.exec("(bird-count $n) if (count (bird $x) $x -> $n) <0.99>").unwrap();

        let pattern = qor_core::neuron::Neuron::expression(vec![
            qor_core::neuron::Neuron::symbol("bird-count"),
            qor_core::neuron::Neuron::variable("n"),
        ]);
        let explanations = session.explain(&pattern);
        // Should find the derived fact, explain doesn't crash on Aggregate
        assert!(!explanations.is_empty());
    }

    #[test]
    fn test_session_heartbeat_with_decay() {
        let mut session = Session::new();
        session.exec("(news flash) <0.95> @decay 0.10").unwrap();
        session.exec("(bird tweety) <0.99>").unwrap();

        // Run heartbeats — news should decay, bird should not
        for _ in 0..5 {
            session.heartbeat();
        }

        let results = session.exec("? (news $x)").unwrap();
        match &results[0] {
            ExecResult::Query(qr) => {
                // Confidence should have dropped from 0.90 to ~0.40
                assert!(qr.results[0].contains("0.4") || qr.results[0].contains("0.3"));
            }
            _ => panic!(),
        }

        // Bird should be unchanged
        let results = session.exec("? (bird $x)").unwrap();
        match &results[0] {
            ExecResult::Query(qr) => {
                assert!(qr.results[0].contains("0.90")); // default confidence
            }
            _ => panic!(),
        }
    }

    // ── Store clone tests ──

    #[test]
    fn test_store_clone() {
        let mut store = crate::store::NeuronStore::new();
        store.insert(
            qor_core::neuron::Neuron::expression(vec![
                qor_core::neuron::Neuron::symbol("bird"),
                qor_core::neuron::Neuron::symbol("tweety"),
            ]),
            qor_core::truth_value::TruthValue::from_strength(0.99),
        );
        store.insert(
            qor_core::neuron::Neuron::expression(vec![
                qor_core::neuron::Neuron::symbol("fish"),
                qor_core::neuron::Neuron::symbol("salmon"),
            ]),
            qor_core::truth_value::TruthValue::from_strength(0.95),
        );

        let cloned = store.clone();
        assert_eq!(cloned.len(), 2);
        assert!(cloned.contains(&qor_core::neuron::Neuron::expression(vec![
            qor_core::neuron::Neuron::symbol("bird"),
            qor_core::neuron::Neuron::symbol("tweety"),
        ])));
    }

    #[test]
    fn test_store_clone_independent() {
        let mut store = crate::store::NeuronStore::new();
        store.insert(
            qor_core::neuron::Neuron::expression(vec![
                qor_core::neuron::Neuron::symbol("bird"),
                qor_core::neuron::Neuron::symbol("tweety"),
            ]),
            qor_core::truth_value::TruthValue::from_strength(0.99),
        );

        let mut cloned = store.clone();
        cloned.insert(
            qor_core::neuron::Neuron::expression(vec![
                qor_core::neuron::Neuron::symbol("fish"),
                qor_core::neuron::Neuron::symbol("salmon"),
            ]),
            qor_core::truth_value::TruthValue::from_strength(0.95),
        );

        // Original unchanged
        assert_eq!(store.len(), 1);
        assert_eq!(cloned.len(), 2);
    }

    #[test]
    fn test_store_clone_trie_works() {
        let mut store = crate::store::NeuronStore::new();
        store.insert(
            qor_core::neuron::Neuron::expression(vec![
                qor_core::neuron::Neuron::symbol("bird"),
                qor_core::neuron::Neuron::symbol("tweety"),
            ]),
            qor_core::truth_value::TruthValue::from_strength(0.99),
        );
        store.insert(
            qor_core::neuron::Neuron::expression(vec![
                qor_core::neuron::Neuron::symbol("bird"),
                qor_core::neuron::Neuron::symbol("eagle"),
            ]),
            qor_core::truth_value::TruthValue::from_strength(0.95),
        );

        let cloned = store.clone();
        // Trie queries should work on clone
        let results = cloned.query(&qor_core::neuron::Neuron::expression(vec![
            qor_core::neuron::Neuron::symbol("bird"),
            qor_core::neuron::Neuron::variable("x"),
        ]));
        assert_eq!(results.len(), 2);
    }

    // ── Hypothesis testing ──

    #[test]
    fn test_hypothesis_derives_new() {
        let mut session = Session::new();
        session.exec("(flies $x) if (bird $x) <0.95>").unwrap();

        let result = session.test_hypothesis("(bird tweety) <0.99>").unwrap();
        assert!(result.new_facts > 0); // should derive (flies tweety)
        assert!(result.facts.iter().any(|f| f.contains("flies")));
    }

    #[test]
    fn test_hypothesis_restores() {
        let mut session = Session::new();
        session.exec("(flies $x) if (bird $x) <0.95>").unwrap();
        let before = session.fact_count();

        session.test_hypothesis("(bird tweety) <0.99>").unwrap();

        // Store should be restored
        assert_eq!(session.fact_count(), before);
        assert!(!session.store().contains(&qor_core::neuron::Neuron::expression(vec![
            qor_core::neuron::Neuron::symbol("bird"),
            qor_core::neuron::Neuron::symbol("tweety"),
        ])));
    }

    #[test]
    fn test_hypothesis_reports_count() {
        let mut session = Session::new();
        session.exec("(bird tweety) <0.99>").unwrap();
        session.exec("(flies $x) if (bird $x) <0.95>").unwrap();
        session.exec("(has-wings $x) if (flies $x) <0.99>").unwrap();

        // Adding another bird should derive (bird eagle) + (flies eagle) + (has-wings eagle)
        let result = session.test_hypothesis("(bird eagle) <0.95>").unwrap();
        assert_eq!(result.new_facts, 3); // bird eagle + flies eagle + has-wings eagle
    }

    #[test]
    fn test_hypothesis_no_effect() {
        let mut session = Session::new();
        session.exec("(bird tweety) <0.99>").unwrap();

        // Adding a fish fact with no rules about fish
        let result = session.test_hypothesis("(fish salmon) <0.99>").unwrap();
        assert_eq!(result.new_facts, 1); // only the hypothesis itself is new
    }

    #[test]
    fn test_hypothesis_with_rules() {
        let mut session = Session::new();
        session.exec("(flies $x) if (bird $x) not (penguin $x) <0.95>").unwrap();
        session.exec("(penguin tux) <0.99>").unwrap();

        // Adding (bird tux) should NOT derive (flies tux) because tux is a penguin
        let result = session.test_hypothesis("(bird tux) <0.99>").unwrap();
        assert!(!result.facts.iter().any(|f| f.contains("flies")));
    }

    #[test]
    fn test_hypothesis_confidence() {
        let mut session = Session::new();
        session.exec("(flies $x) if (bird $x) <0.95>").unwrap();

        let result = session.test_hypothesis("(bird tweety) <0.99>").unwrap();
        assert!(result.confidence > 0.0);
        assert!(result.confidence <= 1.0);
    }

    #[test]
    fn test_store_accessor() {
        let mut session = Session::new();
        session.exec("(bird tweety) <0.99>").unwrap();

        let store = session.store();
        assert_eq!(store.len(), 1);
        assert!(store.contains(&qor_core::neuron::Neuron::expression(vec![
            qor_core::neuron::Neuron::symbol("bird"),
            qor_core::neuron::Neuron::symbol("tweety"),
        ])));
    }

    #[test]
    fn test_auto_truth_value() {
        let mut session = Session::new();
        session.exec("(bird tweety) <0.99>").unwrap();
        session.exec("(flies $x) if (bird $x) <auto>").unwrap();
        let results = session.exec("? (flies $x)").unwrap();
        assert!(!results.is_empty());
        // <auto> = (1.0, 1.0) — rule TV is passthrough, result TV comes from evidence
        let facts: Vec<_> = session.store().query(
            &qor_core::neuron::Neuron::expression(vec![
                qor_core::neuron::Neuron::symbol("flies"),
                qor_core::neuron::Neuron::Variable("x".into()),
            ]),
        );
        assert_eq!(facts.len(), 1);
        // Derived TV should reflect the input fact's strength (0.99)
        assert!(facts[0].tv.strength > 0.9);
    }

    #[test]
    fn test_stratum_annotation() {
        let mut session = Session::new();
        // Stratum 0 rule fires first, then stratum 1
        session.exec(r#"
            (bird tweety)
            @stratum(0)
            (can-fly $x) if (bird $x) not (heavy $x)
            @stratum(1)
            (grounded $x) if (bird $x) not (can-fly $x)
        "#).unwrap();
        // tweety can fly (not heavy), so grounded should NOT fire for tweety
        let store = session.store();
        assert!(store.contains(&qor_core::neuron::Neuron::expression(vec![
            qor_core::neuron::Neuron::symbol("can-fly"),
            qor_core::neuron::Neuron::symbol("tweety"),
        ])));
        assert!(!store.contains(&qor_core::neuron::Neuron::expression(vec![
            qor_core::neuron::Neuron::symbol("grounded"),
            qor_core::neuron::Neuron::symbol("tweety"),
        ])));
    }

    #[test]
    fn test_module_prefixing() {
        let results = run(r#"
            @module animals
            (bird tweety)
            (flies $x) if (bird $x)
            @end
            ? (animals.flies $x)
        "#).unwrap();
        assert_eq!(results.len(), 1);
        assert!(!results[0].results.is_empty(), "should find animals.flies tweety");
    }

    #[test]
    fn test_import_resolution() {
        let results = run(r#"
            @module zoo
            (animal tiger)
            @end
            @import zoo.animal
            (dangerous $x) if (animal $x)
            (animal tiger)
            ? (dangerous $x)
        "#).unwrap();
        // @import zoo.animal → "animal" resolves to "zoo.animal"
        // so the rule body (animal $x) becomes (zoo.animal $x)
        // and the fact (animal tiger) also becomes (zoo.animal tiger)
        assert_eq!(results.len(), 1);
        assert!(!results[0].results.is_empty(), "import should resolve animal to zoo.animal");
    }
}
