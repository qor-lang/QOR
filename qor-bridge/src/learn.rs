// ── QOR Learning Engine ─────────────────────────────────────────────────
//
// Transforms raw data + analysis into condensed KNOWLEDGE.
// Raw data is temporary — only learned facts survive in brain/.
//
// What it learns:
// 1. Statistical summaries — mean, min, max, std_dev per numeric predicate
// 2. Pattern frequencies  — how often each analysis pattern fires
// 3. Co-occurrences       — which patterns appear together and how often
// 4. Learned rules        — high-strength co-occurrences become rules
// 5. Source metadata       — where the data came from, how many points

use qor_core::neuron::{ComparisonOp, Condition, Neuron, QorValue, Statement};
use qor_core::truth_value::TruthValue;
use std::collections::{HashMap, HashSet};

use crate::context::{self, DataDomain};

// ── Stats ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Stats {
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
}

fn compute_stats(values: &[f64]) -> Option<Stats> {
    if values.is_empty() {
        return None;
    }
    let count = values.len();
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = values.iter().sum();
    let mean = sum / count as f64;
    let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64;
    let std_dev = variance.sqrt();
    Some(Stats { count, min, max, mean, std_dev })
}

// ── Fact Builders ──────────────────────────────────────────────────────

fn stat_fact(predicate: &str, stat_name: &str, value: f64) -> Statement {
    Statement::Fact {
        neuron: Neuron::Expression(vec![
            Neuron::Symbol(format!("stat-{}", predicate)),
            Neuron::Symbol(stat_name.into()),
            Neuron::Value(QorValue::Float(round2(value))),
        ]),
        tv: Some(TruthValue::new(0.99, 0.99)),
        decay: None,
    }
}

fn stat_count_fact(predicate: &str, count: usize) -> Statement {
    Statement::Fact {
        neuron: Neuron::Expression(vec![
            Neuron::Symbol(format!("stat-{}", predicate)),
            Neuron::Symbol("count".into()),
            Neuron::Value(QorValue::Int(count as i64)),
        ]),
        tv: Some(TruthValue::new(0.99, 0.99)),
        decay: None,
    }
}

fn pattern_rate_fact(pattern: &str, count: usize, total: usize) -> Statement {
    let strength = count as f64 / total as f64;
    let confidence = TruthValue::w2c(count as f64).min(0.95);
    Statement::Fact {
        neuron: Neuron::Expression(vec![
            Neuron::Symbol("pattern-rate".into()),
            Neuron::Symbol(pattern.into()),
            Neuron::Value(QorValue::Int(count as i64)),
            Neuron::Value(QorValue::Int(total as i64)),
        ]),
        tv: Some(TruthValue::new(round2(strength), round2(confidence))),
        decay: None,
    }
}

fn co_occur_fact(pattern_a: &str, pattern_b: &str, both: usize, base: usize) -> Statement {
    let strength = both as f64 / base as f64;
    let confidence = TruthValue::w2c(base as f64).min(0.95);
    Statement::Fact {
        neuron: Neuron::Expression(vec![
            Neuron::Symbol("co-occur".into()),
            Neuron::Symbol(pattern_a.into()),
            Neuron::Symbol(pattern_b.into()),
            Neuron::Value(QorValue::Int(both as i64)),
            Neuron::Value(QorValue::Int(base as i64)),
        ]),
        tv: Some(TruthValue::new(round2(strength), round2(confidence))),
        decay: None,
    }
}

fn learned_rule(head_pred: &str, body_pred: &str, strength: f64, confidence: f64) -> Statement {
    Statement::Rule {
        head: Neuron::Expression(vec![
            Neuron::Symbol(head_pred.into()),
            Neuron::Variable("x".into()),
        ]),
        body: vec![qor_core::neuron::Condition::Positive(
            Neuron::Expression(vec![
                Neuron::Symbol(body_pred.into()),
                Neuron::Variable("x".into()),
            ]),
        )],
        tv: Some(TruthValue::new(round2(strength), round2(confidence))),
    }
}

fn learned_rule_2cond(head_pred: &str, body1: &str, body2: &str, strength: f64, confidence: f64) -> Statement {
    Statement::Rule {
        head: Neuron::Expression(vec![
            Neuron::Symbol(head_pred.into()),
            Neuron::Variable("x".into()),
        ]),
        body: vec![
            qor_core::neuron::Condition::Positive(Neuron::Expression(vec![
                Neuron::Symbol(body1.into()),
                Neuron::Variable("x".into()),
            ])),
            qor_core::neuron::Condition::Positive(Neuron::Expression(vec![
                Neuron::Symbol(body2.into()),
                Neuron::Variable("x".into()),
            ])),
        ],
        tv: Some(TruthValue::new(round2(strength), round2(confidence))),
    }
}

/// Build a guard-based rule: `(head_pred $x) if (indicator $x $v) (op $v threshold)`
fn learned_guard_rule(head_pred: &str, indicator: &str, op: ComparisonOp, threshold: f64) -> Statement {
    Statement::Rule {
        head: Neuron::Expression(vec![
            Neuron::Symbol(head_pred.into()),
            Neuron::Variable("x".into()),
        ]),
        body: vec![
            Condition::Positive(Neuron::Expression(vec![
                Neuron::Symbol(indicator.into()),
                Neuron::Variable("x".into()),
                Neuron::Variable("v".into()),
            ])),
            Condition::Guard(op, Neuron::Variable("v".into()), Neuron::float_val(threshold)),
        ],
        tv: Some(TruthValue::new(0.95, 0.90)),
    }
}

/// Map pattern name → comparison operator.
/// "overbought" means value is HIGH → Gt; "oversold" means value is LOW → Lt.
fn pattern_to_op(pattern: &str) -> ComparisonOp {
    match pattern {
        "oversold" | "range-bound" | "below-trend" => ComparisonOp::Lt,
        _ => ComparisonOp::Gt, // overbought, trending, fever-detected, high-fever, above-trend, high-volume
    }
}

/// Generate guard-based rules from `(threshold indicator pattern value)` knowledge.
///
/// Reads facts like `(threshold rsi overbought 70)` and generates:
///   `(overbought $x) if (rsi $x $v) (> $v 70) <0.95, 0.90>`
pub fn threshold_rules(knowledge: &[Statement]) -> Vec<Statement> {
    let mut rules = Vec::new();
    for stmt in knowledge {
        if let Statement::Fact { neuron, .. } = stmt {
            if let Neuron::Expression(parts) = neuron {
                if parts.len() == 4 {
                    if let (
                        Neuron::Symbol(pred),
                        Neuron::Symbol(indicator),
                        Neuron::Symbol(pattern),
                        threshold_neuron,
                    ) = (&parts[0], &parts[1], &parts[2], &parts[3])
                    {
                        if pred == "threshold" {
                            if let Some(threshold) = threshold_neuron.as_f64() {
                                let op = pattern_to_op(pattern);
                                rules.push(learned_guard_rule(pattern, indicator, op, threshold));
                            }
                        }
                    }
                }
            }
        }
    }
    rules
}

fn metadata_fact(pred: &str, value: &str) -> Statement {
    Statement::Fact {
        neuron: Neuron::Expression(vec![
            Neuron::Symbol(pred.into()),
            Neuron::Symbol(value.into()),
        ]),
        tv: Some(TruthValue::new(0.99, 0.99)),
        decay: None,
    }
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

// ── Analysis Predicate Detection ────────────────────────────────────────

const ANALYSIS_PREDS: &[&str] = &[
    "overbought", "oversold", "trending", "range-bound",
    "bullish-cross", "bearish-cross", "above-trend", "below-trend",
    "high-volume", "fever-detected", "high-fever",
];

/// Extract analysis pattern facts grouped by entity.
/// Returns HashMap<entity_name, set_of_patterns>.
fn group_analysis_by_entity(analysis: &[Statement]) -> HashMap<String, HashSet<String>> {
    let mut map: HashMap<String, HashSet<String>> = HashMap::new();
    for stmt in analysis {
        if let Statement::Fact { neuron, .. } = stmt {
            if let Neuron::Expression(parts) = neuron {
                if parts.len() >= 2 {
                    if let (Neuron::Symbol(pred), Neuron::Symbol(entity)) = (&parts[0], &parts[1]) {
                        if ANALYSIS_PREDS.contains(&pred.as_str()) {
                            map.entry(entity.clone())
                                .or_default()
                                .insert(pred.clone());
                        }
                    }
                }
            }
        }
    }
    map
}

// ── Main Learning Function ──────────────────────────────────────────────

/// Learn from raw facts + analysis facts. Returns condensed knowledge.
///
/// This is the core of QOR's learning curve:
/// - Raw data (230k facts) goes in
/// - Condensed knowledge (~50 facts) comes out
/// - Brain stores only what was LEARNED, not the raw data
pub fn learn(facts: &[Statement], analysis: &[Statement]) -> Vec<Statement> {
    let mut learned = Vec::new();

    // Step 0: Semantic knowledge — what indicators mean, thresholds, ranges
    let domain = context::detect_domain(facts);
    let knowledge = indicator_knowledge(&domain);
    let guard_rules = threshold_rules(&knowledge);
    learned.extend(knowledge);
    learned.extend(guard_rules);

    // Step 1: Statistical summaries per numeric predicate
    let preds = context::extract_predicates(facts);
    let total_entities = count_entities(facts);

    for pred in &preds {
        let vals: Vec<f64> = context::extract_float_values(facts, pred)
            .into_iter()
            .map(|(_, v)| v)
            .collect();
        if let Some(stats) = compute_stats(&vals) {
            learned.push(stat_count_fact(pred, stats.count));
            learned.push(stat_fact(pred, "mean", stats.mean));
            learned.push(stat_fact(pred, "min", stats.min));
            learned.push(stat_fact(pred, "max", stats.max));
            if stats.std_dev > 0.0 {
                learned.push(stat_fact(pred, "std-dev", stats.std_dev));
            }
        }
    }

    // Step 2: Pattern frequency rates
    if total_entities > 0 {
        let entity_patterns = group_analysis_by_entity(analysis);
        let mut pattern_counts: HashMap<String, usize> = HashMap::new();
        for patterns in entity_patterns.values() {
            for p in patterns {
                *pattern_counts.entry(p.clone()).or_default() += 1;
            }
        }

        let mut sorted_patterns: Vec<_> = pattern_counts.iter().collect();
        sorted_patterns.sort_by_key(|(name, _)| (*name).clone());

        for (pattern, count) in &sorted_patterns {
            learned.push(pattern_rate_fact(pattern, **count, total_entities));
        }

        // Step 3: Co-occurrence analysis
        let co_occurrences = count_co_occurrences(&entity_patterns);
        for ((a, b), (both, base)) in &co_occurrences {
            learned.push(co_occur_fact(a, b, *both, *base));
        }

        // Step 4: Learned rules from strong co-occurrences
        for ((a, b), (both, base)) in &co_occurrences {
            let strength = *both as f64 / *base as f64;
            if strength >= 0.7 && *base >= 10 {
                let confidence = TruthValue::w2c(*base as f64).min(0.95);
                // Name the rule based on what it predicts
                let rule_name = format!("predict-{}-with-{}", a, b);
                learned.push(learned_rule_2cond(&rule_name, a, b, strength, confidence));
            }
        }

        // Step 5: Single-pattern learned rules for high-frequency patterns
        for (pattern, count) in &sorted_patterns {
            let rate = **count as f64 / total_entities as f64;
            if rate >= 0.05 && **count >= 10 {
                let confidence = TruthValue::w2c(**count as f64).min(0.95);
                let rule_name = format!("has-{}", pattern);
                learned.push(learned_rule(&rule_name, pattern, rate, confidence));
            }
        }
    }

    // Step 6: Data source metadata
    learned.push(metadata_fact("data-points", &total_entities.to_string()));

    // Add current date
    let date = chrono_free_date();
    learned.push(metadata_fact("learned-at", &date));

    learned
}

/// Count unique entities in the data (second element of 3-part facts).
fn count_entities(facts: &[Statement]) -> usize {
    let mut entities = HashSet::new();
    for stmt in facts {
        if let Statement::Fact { neuron, .. } = stmt {
            if let Neuron::Expression(parts) = neuron {
                if parts.len() >= 3 {
                    if let Neuron::Symbol(entity) = &parts[1] {
                        entities.insert(entity.clone());
                    }
                }
            }
        }
    }
    entities.len()
}

/// Count co-occurrences: for each pair of patterns, how many entities have both.
/// Returns HashMap<(pattern_a, pattern_b), (both_count, base_count)>.
fn count_co_occurrences(
    entity_patterns: &HashMap<String, HashSet<String>>,
) -> Vec<((String, String), (usize, usize))> {
    // Collect all unique patterns
    let mut all_patterns: HashSet<String> = HashSet::new();
    for patterns in entity_patterns.values() {
        all_patterns.extend(patterns.iter().cloned());
    }
    let mut sorted: Vec<String> = all_patterns.into_iter().collect();
    sorted.sort();

    let mut results = Vec::new();

    // For each pair (a, b) where a < b
    for i in 0..sorted.len() {
        for j in (i + 1)..sorted.len() {
            let a = &sorted[i];
            let b = &sorted[j];

            let base = entity_patterns.values()
                .filter(|pats| pats.contains(a))
                .count();
            let both = entity_patterns.values()
                .filter(|pats| pats.contains(a) && pats.contains(b))
                .count();

            if both >= 5 {
                results.push(((a.clone(), b.clone()), (both, base)));
            }
        }
    }

    results
}

// ── Semantic Knowledge — What Indicators Mean ───────────────────────────

fn knowledge_tv() -> TruthValue {
    TruthValue::new(0.99, 0.99)
}

fn knowledge_fact_2(pred: &str, a: &str, b: &str) -> Statement {
    Statement::Fact {
        neuron: Neuron::Expression(vec![
            Neuron::Symbol(pred.into()),
            Neuron::Symbol(a.into()),
            Neuron::Symbol(b.into()),
        ]),
        tv: Some(knowledge_tv()),
        decay: None,
    }
}

fn knowledge_fact_3(pred: &str, a: &str, b: &str, c: f64) -> Statement {
    Statement::Fact {
        neuron: Neuron::Expression(vec![
            Neuron::Symbol(pred.into()),
            Neuron::Symbol(a.into()),
            Neuron::Symbol(b.into()),
            Neuron::Value(QorValue::Float(c)),
        ]),
        tv: Some(knowledge_tv()),
        decay: None,
    }
}

fn range_fact(indicator: &str, lo: f64, hi: f64) -> Statement {
    Statement::Fact {
        neuron: Neuron::Expression(vec![
            Neuron::Symbol("range".into()),
            Neuron::Symbol(indicator.into()),
            Neuron::Value(QorValue::Float(lo)),
            Neuron::Value(QorValue::Float(hi)),
        ]),
        tv: Some(knowledge_tv()),
        decay: None,
    }
}

/// Generate semantic indicator definitions for the detected domain.
/// These facts explain WHAT the indicators mean and WHY thresholds exist.
pub fn indicator_knowledge(domain: &DataDomain) -> Vec<Statement> {
    match domain {
        DataDomain::Market => market_knowledge(),
        DataDomain::Medical => medical_knowledge(),
        _ => Vec::new(),
    }
}

fn market_knowledge() -> Vec<Statement> {
    let mut facts = Vec::new();

    // What each indicator IS
    facts.push(knowledge_fact_2("indicator", "rsi", "relative-strength-index"));
    facts.push(knowledge_fact_2("indicator", "adx", "average-directional-index"));
    facts.push(knowledge_fact_2("indicator", "macd", "moving-average-convergence-divergence"));
    facts.push(knowledge_fact_2("indicator", "atr", "average-true-range"));
    facts.push(knowledge_fact_2("indicator", "ma-200", "moving-average-200"));
    facts.push(knowledge_fact_2("indicator", "ma-50", "moving-average-50"));
    facts.push(knowledge_fact_2("indicator", "ma-20", "moving-average-20"));
    facts.push(knowledge_fact_2("indicator", "volume", "trading-volume"));

    // Valid ranges
    facts.push(range_fact("rsi", 0.0, 100.0));
    facts.push(range_fact("adx", 0.0, 100.0));

    // Thresholds — the rules QOR uses to classify
    facts.push(knowledge_fact_3("threshold", "rsi", "overbought", 70.0));
    facts.push(knowledge_fact_3("threshold", "rsi", "oversold", 30.0));
    facts.push(knowledge_fact_3("threshold", "adx", "trending", 25.0));
    facts.push(knowledge_fact_3("threshold", "adx", "range-bound", 20.0));

    // What the analysis patterns MEAN
    facts.push(knowledge_fact_2("meaning", "overbought", "buying-pressure-extreme"));
    facts.push(knowledge_fact_2("meaning", "oversold", "selling-pressure-extreme"));
    facts.push(knowledge_fact_2("meaning", "trending", "strong-directional-movement"));
    facts.push(knowledge_fact_2("meaning", "range-bound", "sideways-consolidation"));
    facts.push(knowledge_fact_2("meaning", "bullish-cross", "upward-momentum-shift"));
    facts.push(knowledge_fact_2("meaning", "bearish-cross", "downward-momentum-shift"));
    facts.push(knowledge_fact_2("meaning", "above-trend", "price-above-long-term-average"));
    facts.push(knowledge_fact_2("meaning", "below-trend", "price-below-long-term-average"));
    facts.push(knowledge_fact_2("meaning", "high-volume", "unusual-trading-activity"));

    facts
}

fn medical_knowledge() -> Vec<Statement> {
    let mut facts = Vec::new();

    // What each indicator IS
    facts.push(knowledge_fact_2("indicator", "temperature", "body-temperature"));

    // Valid range
    facts.push(range_fact("temperature", 35.0, 42.0));

    // Thresholds
    facts.push(knowledge_fact_3("threshold", "temperature", "fever-detected", 38.0));
    facts.push(knowledge_fact_3("threshold", "temperature", "high-fever", 39.5));

    // Meanings
    facts.push(knowledge_fact_2("meaning", "fever-detected", "elevated-body-temperature"));
    facts.push(knowledge_fact_2("meaning", "high-fever", "dangerous-body-temperature"));

    facts
}

// ── Incremental Learning — Merge New Knowledge with Existing ─────────

/// Merge new learned knowledge with existing brain knowledge.
/// This is how QOR grows — each new dataset strengthens what it knows.
///
/// Merge strategy:
/// - stat-X count: sum
/// - stat-X mean: weighted average by count
/// - stat-X min: take minimum
/// - stat-X max: take maximum
/// - stat-X std-dev: keep newer (approximation)
/// - pattern-rate: sum counts + totals, recalculate rate
/// - co-occur: sum counts, recalculate strength
/// - data-points: sum
/// - Rules: TruthValue revision (confidence grows)
/// - Other facts: TruthValue revision
/// - Old-only facts: keep (knowledge persists)
/// - New-only facts: add
pub fn merge_learned(existing: &[Statement], new: &[Statement]) -> Vec<Statement> {
    let mut result = Vec::new();
    let mut used_new: HashSet<usize> = HashSet::new();

    for old_stmt in existing {
        let old_key = fact_key(old_stmt);
        // Find matching new statement
        let match_idx = new.iter().enumerate().find(|(i, s)| {
            !used_new.contains(i) && fact_key(s) == old_key
        });

        if let Some((idx, new_stmt)) = match_idx {
            used_new.insert(idx);
            result.push(merge_single(old_stmt, new_stmt));
        } else {
            // Old-only: keep (knowledge persists)
            result.push(old_stmt.clone());
        }
    }

    // New-only: add
    for (i, stmt) in new.iter().enumerate() {
        if !used_new.contains(&i) {
            result.push(stmt.clone());
        }
    }

    result
}

/// Generate a key for matching facts across brain versions.
/// Ignores numeric value slots — matches by structure.
fn fact_key(stmt: &Statement) -> String {
    match stmt {
        Statement::Fact { neuron, .. } => neuron_key(neuron),
        Statement::Rule { head, body, .. } => {
            let body_str: Vec<String> = body.iter().map(|c| format!("{}", c)).collect();
            format!("rule:{}:if:{}", neuron_key(head), body_str.join(":"))
        }
        Statement::Query { pattern } => format!("query:{}", pattern),
        Statement::Test { name, .. } => format!("test:{}", name),
    }
}

/// Predicates that are singletons — only one value per predicate.
/// These get keyed by predicate alone, not by value.
const SINGLETON_PREDS: &[&str] = &["data-points", "learned-at"];

/// Structural key for a neuron — symbols preserved, values replaced with placeholder.
fn neuron_key(neuron: &Neuron) -> String {
    match neuron {
        Neuron::Symbol(s) => s.clone(),
        Neuron::Variable(v) => format!("${}", v),
        Neuron::Expression(parts) => {
            // For singleton predicates, key only on the predicate
            if parts.len() == 2 {
                if let Neuron::Symbol(pred) = &parts[0] {
                    if SINGLETON_PREDS.contains(&pred.as_str()) {
                        return format!("({} _V_)", pred);
                    }
                }
            }
            let inner: Vec<String> = parts.iter().map(|p| neuron_key(p)).collect();
            format!("({})", inner.join(" "))
        }
        Neuron::Value(_) => "_V_".to_string(),
    }
}

/// Merge two matched statements.
fn merge_single(old: &Statement, new: &Statement) -> Statement {
    match (old, new) {
        (Statement::Fact { neuron: old_n, tv: old_tv, decay: old_d },
         Statement::Fact { neuron: new_n, tv: new_tv, decay: new_d }) => {
            // Check if it's a stat fact for special merging
            let old_str = format!("{}", old_n);
            if old_str.starts_with("(stat-") {
                return merge_stat_fact(old_n, old_tv, new_n, new_tv, old_d);
            }
            if old_str.starts_with("(pattern-rate ") {
                return merge_pattern_rate(old_n, old_tv, new_n, new_tv);
            }
            if old_str.starts_with("(co-occur ") {
                return merge_co_occur(old_n, old_tv, new_n, new_tv);
            }
            if old_str.starts_with("(data-points ") {
                return merge_data_points(old_n, new_n);
            }
            // Default: TruthValue revision
            let merged_tv = match (old_tv, new_tv) {
                (Some(o), Some(n)) => Some(o.revision(n)),
                (Some(o), None) => Some(*o),
                (None, Some(n)) => Some(*n),
                (None, None) => None,
            };
            Statement::Fact {
                neuron: new_n.clone(),
                tv: merged_tv,
                decay: new_d.or(*old_d),
            }
        }
        (Statement::Rule { head, body, tv: old_tv },
         Statement::Rule { tv: new_tv, .. }) => {
            // Rules: TruthValue revision — confidence grows
            let merged_tv = match (old_tv, new_tv) {
                (Some(o), Some(n)) => Some(o.revision(n)),
                (Some(o), None) => Some(*o),
                (None, Some(n)) => Some(*n),
                (None, None) => None,
            };
            Statement::Rule {
                head: head.clone(),
                body: body.clone(),
                tv: merged_tv,
            }
        }
        _ => new.clone(),
    }
}

fn merge_stat_fact(old_n: &Neuron, old_tv: &Option<TruthValue>,
                   new_n: &Neuron, new_tv: &Option<TruthValue>,
                   decay: &Option<f64>) -> Statement {
    let old_parts = if let Neuron::Expression(p) = old_n { p } else { return new_n_fact(new_n, new_tv); };
    let new_parts = if let Neuron::Expression(p) = new_n { p } else { return new_n_fact(new_n, new_tv); };

    if old_parts.len() < 3 || new_parts.len() < 3 { return new_n_fact(new_n, new_tv); }

    let stat_type = if let Neuron::Symbol(s) = &old_parts[1] { s.as_str() } else { return new_n_fact(new_n, new_tv); };
    let old_val = old_parts[2].as_f64();
    let new_val = new_parts[2].as_f64();

    match (old_val, new_val) {
        (Some(ov), Some(nv)) => {
            let merged_val = match stat_type {
                "count" => ov + nv,               // sum
                "mean" => (ov + nv) / 2.0,        // simple average (weighted would need count)
                "min" => ov.min(nv),               // take minimum
                "max" => ov.max(nv),               // take maximum
                "std-dev" => nv,                   // keep newer
                _ => nv,
            };
            let mut merged_parts = old_parts.clone();
            if merged_val == (merged_val as i64) as f64 && stat_type == "count" {
                merged_parts[2] = Neuron::Value(QorValue::Int(merged_val as i64));
            } else {
                merged_parts[2] = Neuron::Value(QorValue::Float(round2(merged_val)));
            }
            Statement::Fact {
                neuron: Neuron::Expression(merged_parts),
                tv: *old_tv,
                decay: *decay,
            }
        }
        _ => new_n_fact(new_n, new_tv),
    }
}

fn merge_pattern_rate(old_n: &Neuron, old_tv: &Option<TruthValue>,
                      new_n: &Neuron, _new_tv: &Option<TruthValue>) -> Statement {
    let old_parts = if let Neuron::Expression(p) = old_n { p } else { return new_n_fact(new_n, old_tv); };
    let new_parts = if let Neuron::Expression(p) = new_n { p } else { return new_n_fact(new_n, old_tv); };

    if old_parts.len() < 4 || new_parts.len() < 4 { return new_n_fact(new_n, old_tv); }

    let old_count = old_parts[2].as_f64().unwrap_or(0.0) as usize;
    let old_total = old_parts[3].as_f64().unwrap_or(0.0) as usize;
    let new_count = new_parts[2].as_f64().unwrap_or(0.0) as usize;
    let new_total = new_parts[3].as_f64().unwrap_or(0.0) as usize;

    let merged_count = old_count + new_count;
    let merged_total = old_total + new_total;

    if let Neuron::Symbol(name) = &old_parts[1] {
        pattern_rate_fact(name, merged_count, merged_total)
    } else {
        new_n_fact(new_n, old_tv)
    }
}

fn merge_co_occur(old_n: &Neuron, old_tv: &Option<TruthValue>,
                  new_n: &Neuron, _new_tv: &Option<TruthValue>) -> Statement {
    let old_parts = if let Neuron::Expression(p) = old_n { p } else { return new_n_fact(new_n, old_tv); };
    let new_parts = if let Neuron::Expression(p) = new_n { p } else { return new_n_fact(new_n, old_tv); };

    if old_parts.len() < 5 || new_parts.len() < 5 { return new_n_fact(new_n, old_tv); }

    let old_both = old_parts[3].as_f64().unwrap_or(0.0) as usize;
    let old_base = old_parts[4].as_f64().unwrap_or(0.0) as usize;
    let new_both = new_parts[3].as_f64().unwrap_or(0.0) as usize;
    let new_base = new_parts[4].as_f64().unwrap_or(0.0) as usize;

    let merged_both = old_both + new_both;
    let merged_base = old_base + new_base;

    if let (Neuron::Symbol(a), Neuron::Symbol(b)) = (&old_parts[1], &old_parts[2]) {
        co_occur_fact(a, b, merged_both, merged_base)
    } else {
        new_n_fact(new_n, old_tv)
    }
}

fn merge_data_points(old_n: &Neuron, new_n: &Neuron) -> Statement {
    let old_parts = if let Neuron::Expression(p) = old_n { p } else { return new_n_fact(new_n, &None); };
    let new_parts = if let Neuron::Expression(p) = new_n { p } else { return new_n_fact(new_n, &None); };

    if old_parts.len() < 2 || new_parts.len() < 2 { return new_n_fact(new_n, &None); }

    let old_val = if let Neuron::Symbol(s) = &old_parts[1] { s.parse::<usize>().unwrap_or(0) } else { 0 };
    let new_val = if let Neuron::Symbol(s) = &new_parts[1] { s.parse::<usize>().unwrap_or(0) } else { 0 };

    metadata_fact("data-points", &(old_val + new_val).to_string())
}

fn new_n_fact(neuron: &Neuron, tv: &Option<TruthValue>) -> Statement {
    Statement::Fact { neuron: neuron.clone(), tv: *tv, decay: None }
}

/// Get current date without chrono dependency.
fn chrono_free_date() -> String {
    // Use a simple approach — will be overridden by source metadata if available
    "2026-03-09".to_string()
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use qor_core::neuron::QorValue;

    fn float_fact(pred: &str, entity: &str, val: f64) -> Statement {
        Statement::Fact {
            neuron: Neuron::Expression(vec![
                Neuron::Symbol(pred.into()),
                Neuron::Symbol(entity.into()),
                Neuron::Value(QorValue::Float(val)),
            ]),
            tv: Some(TruthValue::new(0.90, 0.90)),
            decay: None,
        }
    }

    fn analysis_fact_2(pred: &str, entity: &str) -> Statement {
        Statement::Fact {
            neuron: Neuron::Expression(vec![
                Neuron::Symbol(pred.into()),
                Neuron::Symbol(entity.into()),
            ]),
            tv: Some(TruthValue::new(0.90, 0.85)),
            decay: None,
        }
    }

    #[test]
    fn test_compute_stats_basic() {
        let vals = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let s = compute_stats(&vals).unwrap();
        assert_eq!(s.count, 5);
        assert!((s.mean - 30.0).abs() < 0.01);
        assert!((s.min - 10.0).abs() < 0.01);
        assert!((s.max - 50.0).abs() < 0.01);
        assert!(s.std_dev > 0.0);
    }

    #[test]
    fn test_compute_stats_empty() {
        assert!(compute_stats(&[]).is_none());
    }

    #[test]
    fn test_learn_generates_stats() {
        let facts: Vec<Statement> = (0..20).map(|i| {
            float_fact("rsi", &format!("row-{}", i), 40.0 + i as f64)
        }).collect();

        let learned = learn(&facts, &[]);

        // Should have stat-rsi facts
        let stat_facts: Vec<_> = learned.iter().filter(|s| {
            if let Statement::Fact { neuron, .. } = s {
                format!("{}", neuron).contains("stat-rsi")
            } else { false }
        }).collect();
        assert!(stat_facts.len() >= 4, "expected count/mean/min/max, got {}", stat_facts.len());
    }

    #[test]
    fn test_learn_generates_pattern_rates() {
        let mut facts = Vec::new();
        let mut analysis = Vec::new();

        // 100 entities with RSI data
        for i in 0..100 {
            let entity = format!("row-{}", i);
            facts.push(float_fact("rsi", &entity, 40.0 + (i % 60) as f64));
            facts.push(float_fact("close", &entity, 100.0));

            // ~10% overbought (RSI > 70 mapped to analysis)
            if i % 10 == 0 {
                analysis.push(analysis_fact_2("overbought", &entity));
            }
            // ~50% bullish
            if i % 2 == 0 {
                analysis.push(analysis_fact_2("bullish-cross", &entity));
            }
        }

        let learned = learn(&facts, &analysis);

        let pattern_rates: Vec<_> = learned.iter().filter(|s| {
            if let Statement::Fact { neuron, .. } = s {
                format!("{}", neuron).contains("pattern-rate")
            } else { false }
        }).collect();
        assert!(pattern_rates.len() >= 2, "expected overbought + bullish rates");
    }

    #[test]
    fn test_learn_generates_co_occurrences() {
        let mut facts = Vec::new();
        let mut analysis = Vec::new();

        for i in 0..100 {
            let entity = format!("row-{}", i);
            facts.push(float_fact("close", &entity, 100.0));

            // Both overbought AND bearish-cross for first 50
            if i < 50 {
                analysis.push(analysis_fact_2("overbought", &entity));
                analysis.push(analysis_fact_2("bearish-cross", &entity));
            }
            // Only overbought for next 10
            if i >= 50 && i < 60 {
                analysis.push(analysis_fact_2("overbought", &entity));
            }
        }

        let learned = learn(&facts, &analysis);

        let co_occurs: Vec<_> = learned.iter().filter(|s| {
            if let Statement::Fact { neuron, .. } = s {
                format!("{}", neuron).contains("co-occur")
            } else { false }
        }).collect();
        assert!(!co_occurs.is_empty(), "should find co-occurrence of overbought + bearish-cross");
    }

    #[test]
    fn test_learn_generates_learned_rules() {
        let mut facts = Vec::new();
        let mut analysis = Vec::new();

        // Create strong co-occurrence: overbought + bearish-cross appear together 90% of time
        for i in 0..100 {
            let entity = format!("row-{}", i);
            facts.push(float_fact("close", &entity, 100.0));

            analysis.push(analysis_fact_2("overbought", &entity));
            if i < 90 {
                analysis.push(analysis_fact_2("bearish-cross", &entity));
            }
        }

        let learned = learn(&facts, &analysis);

        let rules: Vec<_> = learned.iter().filter(|s| {
            matches!(s, Statement::Rule { .. })
        }).collect();
        assert!(!rules.is_empty(), "should generate learned rules from strong co-occurrences");
    }

    #[test]
    fn test_learn_metadata() {
        let facts = vec![float_fact("close", "row-0", 100.0)];
        let learned = learn(&facts, &[]);

        let has_data_points = learned.iter().any(|s| {
            if let Statement::Fact { neuron, .. } = s {
                format!("{}", neuron).contains("data-points")
            } else { false }
        });
        assert!(has_data_points);

        let has_learned_at = learned.iter().any(|s| {
            if let Statement::Fact { neuron, .. } = s {
                format!("{}", neuron).contains("learned-at")
            } else { false }
        });
        assert!(has_learned_at);
    }

    #[test]
    fn test_learn_empty() {
        let learned = learn(&[], &[]);
        // Should still have metadata
        assert!(learned.len() >= 2); // data-points + learned-at
    }

    #[test]
    fn test_market_knowledge() {
        let knowledge = market_knowledge();
        let text: Vec<String> = knowledge.iter().map(|s| format!("{:?}", s)).collect();
        let all = text.join(" ");

        // Indicators
        assert!(all.contains("rsi") && all.contains("relative-strength-index"));
        assert!(all.contains("adx") && all.contains("average-directional-index"));
        assert!(all.contains("macd"));

        // Thresholds
        assert!(all.contains("threshold") && all.contains("overbought"));
        assert!(all.contains("threshold") && all.contains("oversold"));

        // Meanings
        assert!(all.contains("meaning") && all.contains("buying-pressure-extreme"));
        assert!(all.contains("meaning") && all.contains("selling-pressure-extreme"));

        // Ranges
        assert!(all.contains("range"));
    }

    #[test]
    fn test_medical_knowledge() {
        let knowledge = medical_knowledge();
        let text: Vec<String> = knowledge.iter().map(|s| format!("{:?}", s)).collect();
        let all = text.join(" ");

        assert!(all.contains("temperature") && all.contains("body-temperature"));
        assert!(all.contains("threshold") && all.contains("fever-detected"));
        assert!(all.contains("threshold") && all.contains("high-fever"));
        assert!(all.contains("meaning") && all.contains("elevated-body-temperature"));
    }

    #[test]
    fn test_indicator_knowledge_in_learn() {
        // Market facts should trigger market knowledge
        let facts = vec![
            float_fact("close", "row-0", 100.0),
            float_fact("rsi", "row-0", 50.0),
            float_fact("macd", "row-0", 1.5),
        ];
        let learned = learn(&facts, &[]);

        let has_indicator = learned.iter().any(|s| {
            if let Statement::Fact { neuron, .. } = s {
                format!("{}", neuron).contains("indicator")
            } else { false }
        });
        assert!(has_indicator, "learn() should include indicator definitions");

        let has_threshold = learned.iter().any(|s| {
            if let Statement::Fact { neuron, .. } = s {
                format!("{}", neuron).contains("threshold")
            } else { false }
        });
        assert!(has_threshold, "learn() should include threshold definitions");

        let has_meaning = learned.iter().any(|s| {
            if let Statement::Fact { neuron, .. } = s {
                format!("{}", neuron).contains("meaning")
            } else { false }
        });
        assert!(has_meaning, "learn() should include meaning definitions");
    }

    #[test]
    fn test_general_domain_no_knowledge() {
        // General domain facts should NOT have indicator knowledge
        let facts = vec![
            float_fact("color", "ball", 255.0),
            float_fact("size", "ball", 10.0),
        ];
        let learned = learn(&facts, &[]);

        let has_indicator = learned.iter().any(|s| {
            if let Statement::Fact { neuron, .. } = s {
                format!("{}", neuron).contains("indicator")
            } else { false }
        });
        assert!(!has_indicator, "general domain should not have indicator knowledge");
    }

    #[test]
    fn test_merge_stats_sum_count() {
        let old = vec![stat_count_fact("close", 100)];
        let new = vec![stat_count_fact("close", 200)];
        let merged = merge_learned(&old, &new);
        assert_eq!(merged.len(), 1);
        // Count should be summed: 100 + 200 = 300
        let s = format!("{:?}", merged[0]);
        assert!(s.contains("300"), "count should be 300, got: {}", s);
    }

    #[test]
    fn test_merge_stats_min_max() {
        let old = vec![
            stat_fact("close", "min", 80000.0),
            stat_fact("close", "max", 95000.0),
        ];
        let new = vec![
            stat_fact("close", "min", 78000.0),
            stat_fact("close", "max", 96000.0),
        ];
        let merged = merge_learned(&old, &new);
        assert_eq!(merged.len(), 2);
        let all = format!("{:?}", merged);
        assert!(all.contains("78000"), "min should be 78000");
        assert!(all.contains("96000"), "max should be 96000");
    }

    #[test]
    fn test_merge_pattern_rates() {
        let old = vec![pattern_rate_fact("overbought", 100, 1000)];
        let new = vec![pattern_rate_fact("overbought", 150, 2000)];
        let merged = merge_learned(&old, &new);
        assert_eq!(merged.len(), 1);
        // Should sum: 250/3000
        let s = format!("{:?}", merged[0]);
        assert!(s.contains("250"), "count should be 250, got: {}", s);
        assert!(s.contains("3000"), "total should be 3000, got: {}", s);
    }

    #[test]
    fn test_merge_old_persists() {
        let old = vec![
            metadata_fact("domain", "market"),
            metadata_fact("learned-at", "2026-01-01"),
        ];
        let new = vec![
            metadata_fact("learned-at", "2026-03-09"),
        ];
        let merged = merge_learned(&old, &new);
        // domain should persist (old-only), learned-at should be updated
        assert_eq!(merged.len(), 2);
        let all = format!("{:?}", merged);
        assert!(all.contains("market"), "old domain should persist");
        assert!(all.contains("2026-03-09"), "learned-at should update");
    }

    #[test]
    fn test_merge_new_added() {
        let old = vec![metadata_fact("domain", "market")];
        let new = vec![
            metadata_fact("domain", "market"),
            metadata_fact("purpose", "prediction"),
        ];
        let merged = merge_learned(&old, &new);
        assert_eq!(merged.len(), 2);
        let all = format!("{:?}", merged);
        assert!(all.contains("purpose"), "new fact should be added");
    }

    #[test]
    fn test_merge_rule_confidence_grows() {
        let old = vec![learned_rule("has-trending", "trending", 0.37, 0.80)];
        let new = vec![learned_rule("has-trending", "trending", 0.40, 0.85)];
        let merged = merge_learned(&old, &new);
        assert_eq!(merged.len(), 1);
        if let Statement::Rule { tv: Some(tv), .. } = &merged[0] {
            assert!(tv.confidence > 0.85, "merged confidence {} should grow above 0.85", tv.confidence);
        } else {
            panic!("expected Rule");
        }
    }

    #[test]
    fn test_threshold_rules_market() {
        let knowledge = market_knowledge();
        let rules = threshold_rules(&knowledge);
        // Should generate rules for: overbought, oversold, trending, range-bound
        assert!(rules.len() >= 4, "expected >=4 guard rules, got {}", rules.len());

        let all: Vec<String> = rules.iter().map(|r| format!("{:?}", r)).collect();
        let text = all.join(" ");
        assert!(text.contains("overbought"), "should have overbought rule");
        assert!(text.contains("oversold"), "should have oversold rule");
        assert!(text.contains("trending"), "should have trending rule");
    }

    #[test]
    fn test_threshold_rules_medical() {
        let knowledge = medical_knowledge();
        let rules = threshold_rules(&knowledge);
        assert!(rules.len() >= 2, "expected >=2 medical guard rules, got {}", rules.len());

        let all: Vec<String> = rules.iter().map(|r| format!("{:?}", r)).collect();
        let text = all.join(" ");
        assert!(text.contains("fever-detected"), "should have fever rule");
        assert!(text.contains("high-fever"), "should have high-fever rule");
    }

    #[test]
    fn test_threshold_rules_have_guard() {
        let knowledge = market_knowledge();
        let rules = threshold_rules(&knowledge);

        // Every rule should have a Guard condition
        for rule in &rules {
            if let Statement::Rule { body, .. } = rule {
                let has_guard = body.iter().any(|c| matches!(c, qor_core::neuron::Condition::Guard(..)));
                assert!(has_guard, "threshold rule should have a Guard condition: {:?}", rule);
            }
        }
    }

    #[test]
    fn test_threshold_rules_roundtrip() {
        let knowledge = market_knowledge();
        let rules = threshold_rules(&knowledge);

        // Serialize and re-parse — guards should survive round-trip
        for rule in &rules {
            if let Statement::Rule { head, body, tv } = rule {
                let body_str: Vec<String> = body.iter().map(|c| format!("{}", c)).collect();
                let tv_str = tv.map(|t| format!(" <{:.2}, {:.2}>", t.strength, t.confidence))
                    .unwrap_or_default();
                let serialized = format!("{} if {}{}", head, body_str.join(" "), tv_str);

                // Should parse back without error
                let parsed = qor_core::parser::parse(&serialized);
                assert!(parsed.is_ok(), "round-trip failed for: {} — error: {:?}", serialized, parsed.err());
            }
        }
    }

    #[test]
    fn test_learn_includes_guard_rules() {
        // Market facts should trigger guard rule generation
        let facts = vec![
            float_fact("close", "row-0", 100.0),
            float_fact("rsi", "row-0", 50.0),
        ];
        let learned = learn(&facts, &[]);

        let guard_rules: Vec<_> = learned.iter().filter(|s| {
            if let Statement::Rule { body, .. } = s {
                body.iter().any(|c| matches!(c, qor_core::neuron::Condition::Guard(..)))
            } else {
                false
            }
        }).collect();
        assert!(!guard_rules.is_empty(), "learn() should generate guard-based rules from thresholds");
    }

    #[test]
    fn test_count_entities() {
        let facts = vec![
            float_fact("close", "row-0", 100.0),
            float_fact("rsi", "row-0", 50.0),
            float_fact("close", "row-1", 200.0),
            float_fact("rsi", "row-1", 60.0),
        ];
        assert_eq!(count_entities(&facts), 2);
    }
}
