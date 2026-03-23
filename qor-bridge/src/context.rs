use qor_core::neuron::{Condition, Neuron, QorValue, Statement};
use qor_core::truth_value::TruthValue;
use std::collections::{HashMap, HashSet};

/// Detected domain of the ingested data.
#[derive(Debug, Clone, PartialEq)]
pub enum DataDomain {
    Market,
    Medical,
    News,
    Social,
    Science,
    Bio,
    Config,
    General,
}

impl std::fmt::Display for DataDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataDomain::Market => write!(f, "market"),
            DataDomain::Medical => write!(f, "medical"),
            DataDomain::News => write!(f, "news"),
            DataDomain::Social => write!(f, "social"),
            DataDomain::Science => write!(f, "science"),
            DataDomain::Bio => write!(f, "biology"),
            DataDomain::Config => write!(f, "config"),
            DataDomain::General => write!(f, "general"),
        }
    }
}

const MARKET_KEYWORDS: &[&str] = &[
    "open", "high", "low", "close", "volume", "price",
    "rsi", "macd", "signal", "histogram", "atr", "adx",
    "ma-20", "ma-50", "ma-200", "trendline",
    "bl-upper", "bl-lower", "mn-upper", "mn-lower",
    "bid", "ask", "spread", "ticker", "symbol",
    "timestamp", "ohlc", "candle", "trade",
];

const MEDICAL_KEYWORDS: &[&str] = &[
    "symptom", "diagnosis", "patient", "temperature", "admitted",
    "blood-pressure", "heart-rate", "pulse", "medication", "dosage",
    "allergy", "condition", "treatment", "lab-result", "bmi",
];

const NEWS_KEYWORDS: &[&str] = &[
    "title", "headline", "author", "content", "source",
    "published", "category", "summary", "article", "date",
    "url", "body", "tags", "reporter",
];

const SOCIAL_KEYWORDS: &[&str] = &[
    "user", "post", "comment", "likes", "followers",
    "hashtag", "retweet", "share", "reply", "mention",
    "username", "handle", "tweet", "status", "feed",
];

const SCIENCE_KEYWORDS: &[&str] = &[
    "experiment", "measurement", "sample", "result", "hypothesis",
    "control", "variable", "observation", "trial", "data-point",
    "wavelength", "frequency", "mass", "velocity", "concentration",
];

const BIO_KEYWORDS: &[&str] = &[
    "species", "genus", "habitat", "can-fly", "weight",
    "kingdom", "phylum", "class", "order", "family",
    "population", "ecosystem", "predator", "prey",
];

const CONFIG_KEYWORDS: &[&str] = &[
    "env", "port", "debug", "cache", "host", "timeout",
    "database", "password", "secret", "api-key", "url",
    "log-level", "max-retries", "workers", "mode",
];

/// Extract all unique predicates (first symbol) from facts.
pub(crate) fn extract_predicates(facts: &[Statement]) -> HashSet<String> {
    let mut preds = HashSet::new();
    for stmt in facts {
        if let Statement::Fact { neuron, .. } = stmt {
            if let Neuron::Expression(parts) = neuron {
                if let Some(Neuron::Symbol(s)) = parts.first() {
                    preds.insert(s.clone());
                }
            }
        }
    }
    preds
}

/// Score how many predicates match a keyword set.
fn score(preds: &HashSet<String>, keywords: &[&str]) -> usize {
    keywords.iter().filter(|kw| preds.contains(**kw)).count()
}

/// Detect the domain of the data based on its predicates.
pub fn detect_domain(facts: &[Statement]) -> DataDomain {
    let preds = extract_predicates(facts);

    let scores = [
        (DataDomain::Market, score(&preds, MARKET_KEYWORDS)),
        (DataDomain::Medical, score(&preds, MEDICAL_KEYWORDS)),
        (DataDomain::News, score(&preds, NEWS_KEYWORDS)),
        (DataDomain::Social, score(&preds, SOCIAL_KEYWORDS)),
        (DataDomain::Science, score(&preds, SCIENCE_KEYWORDS)),
        (DataDomain::Bio, score(&preds, BIO_KEYWORDS)),
        (DataDomain::Config, score(&preds, CONFIG_KEYWORDS)),
    ];

    // Need at least 2 keyword matches to claim a domain
    let best = scores.iter().max_by_key(|(_, s)| s).unwrap();
    if best.1 >= 2 {
        best.0.clone()
    } else {
        DataDomain::General
    }
}

fn ctx_tv() -> TruthValue {
    TruthValue::new(0.90, 0.90)
}

fn rule_tv() -> TruthValue {
    TruthValue::new(0.90, 0.80)
}

fn domain_fact(domain: &str) -> Statement {
    Statement::Fact {
        neuron: Neuron::Expression(vec![
            Neuron::Symbol("domain".into()),
            Neuron::Symbol(domain.into()),
        ]),
        tv: Some(TruthValue::new(0.95, 0.90)),
        decay: None,
    }
}

fn purpose_fact(purpose: &str) -> Statement {
    Statement::Fact {
        neuron: Neuron::Expression(vec![
            Neuron::Symbol("purpose".into()),
            Neuron::Symbol(purpose.into()),
        ]),
        tv: Some(ctx_tv()),
        decay: None,
    }
}

fn make_rule(head_pred: &str, body_pred: &str) -> Statement {
    Statement::Rule {
        head: Neuron::Expression(vec![
            Neuron::Symbol(head_pred.into()),
            Neuron::Variable("x".into()),
        ]),
        body: vec![Condition::Positive(Neuron::Expression(vec![
            Neuron::Symbol(body_pred.into()),
            Neuron::Variable("x".into()),
            Neuron::Variable("v".into()),
        ]))],
        tv: Some(rule_tv()),
    }
}

/// Rule with 2-part body: (head $x) if (pred $x)
/// For matching analysis facts like (overbought row-0) that have no value slot.
fn make_rule_simple(head_pred: &str, body_pred: &str) -> Statement {
    Statement::Rule {
        head: Neuron::Expression(vec![
            Neuron::Symbol(head_pred.into()),
            Neuron::Variable("x".into()),
        ]),
        body: vec![Condition::Positive(Neuron::Expression(vec![
            Neuron::Symbol(body_pred.into()),
            Neuron::Variable("x".into()),
        ]))],
        tv: Some(rule_tv()),
    }
}

fn make_rule_2cond(head_pred: &str, body1: &str, body2: &str) -> Statement {
    Statement::Rule {
        head: Neuron::Expression(vec![
            Neuron::Symbol(head_pred.into()),
            Neuron::Variable("x".into()),
        ]),
        body: vec![
            Condition::Positive(Neuron::Expression(vec![
                Neuron::Symbol(body1.into()),
                Neuron::Variable("x".into()),
            ])),
            Condition::Positive(Neuron::Expression(vec![
                Neuron::Symbol(body2.into()),
                Neuron::Variable("x".into()),
            ])),
        ],
        tv: Some(TruthValue::new(0.85, 0.75)),
    }
}

/// Generate market-domain rules.
fn market_rules(preds: &HashSet<String>) -> Vec<Statement> {
    let mut rules = vec![
        domain_fact("market"),
        purpose_fact("analysis"),
        purpose_fact("prediction"),
    ];

    if preds.contains("close") {
        rules.push(make_rule("price-bar", "close"));
    }
    if preds.contains("rsi") {
        rules.push(make_rule("has-indicator", "rsi"));
    }
    if preds.contains("macd") {
        rules.push(make_rule("has-trend", "macd"));
    }
    if preds.contains("atr") {
        rules.push(make_rule("has-volatility", "atr"));
    }
    if preds.contains("adx") {
        rules.push(make_rule("has-strength", "adx"));
    }
    if preds.contains("volume") {
        rules.push(make_rule("has-volume", "volume"));
    }
    if preds.contains("ma-200") {
        rules.push(make_rule("has-long-trend", "ma-200"));
    }
    if preds.contains("ma-50") {
        rules.push(make_rule("has-mid-trend", "ma-50"));
    }

    // Composite rules
    if preds.contains("close") && preds.contains("rsi") {
        rules.push(make_rule_2cond("fully-analyzed", "price-bar", "has-indicator"));
    }
    if preds.contains("close") && preds.contains("volume") {
        rules.push(make_rule_2cond("has-liquidity", "price-bar", "has-volume"));
    }

    // Signal rules — operate on analysis facts from auto_analysis()
    rules.push(make_rule_2cond("buy-signal", "oversold", "bullish-cross"));
    rules.push(make_rule_2cond("sell-signal", "overbought", "bearish-cross"));
    rules.push(make_rule_simple("anomaly", "high-volume"));
    rules.push(make_rule_2cond("strong-setup", "trending", "above-trend"));

    rules
}

/// Generate medical-domain rules.
fn medical_rules(preds: &HashSet<String>) -> Vec<Statement> {
    let mut rules = vec![
        domain_fact("medical"),
        purpose_fact("diagnostics"),
    ];

    if preds.contains("symptom") {
        rules.push(make_rule("has-symptom", "symptom"));
    }
    if preds.contains("temperature") {
        rules.push(make_rule("has-vitals", "temperature"));
    }
    if preds.contains("diagnosis") {
        rules.push(make_rule("has-diagnosis", "diagnosis"));
    }
    if preds.contains("medication") {
        rules.push(make_rule("on-medication", "medication"));
    }
    if preds.contains("admitted") {
        rules.push(make_rule("is-admitted", "admitted"));
    }

    if preds.contains("symptom") && preds.contains("temperature") {
        rules.push(make_rule_2cond("needs-review", "has-symptom", "has-vitals"));
    }

    // Signal rules — operate on analysis facts from auto_analysis()
    if preds.contains("temperature") && preds.contains("admitted") {
        rules.push(make_rule_2cond("critical-patient", "high-fever", "is-admitted"));
    }

    rules
}

/// Generate news-domain rules.
fn news_rules(preds: &HashSet<String>) -> Vec<Statement> {
    let mut rules = vec![
        domain_fact("news"),
        purpose_fact("summary"),
        purpose_fact("analysis"),
    ];

    if preds.contains("content") || preds.contains("body") {
        let pred = if preds.contains("content") { "content" } else { "body" };
        rules.push(make_rule("has-content", pred));
    }
    if preds.contains("source") || preds.contains("author") {
        let pred = if preds.contains("source") { "source" } else { "author" };
        rules.push(make_rule("has-source", pred));
    }
    if preds.contains("title") || preds.contains("headline") {
        let pred = if preds.contains("title") { "title" } else { "headline" };
        rules.push(make_rule("has-title", pred));
    }

    rules
}

/// Generate social-domain rules.
fn social_rules(preds: &HashSet<String>) -> Vec<Statement> {
    let mut rules = vec![
        domain_fact("social"),
        purpose_fact("sentiment"),
        purpose_fact("trends"),
    ];

    if preds.contains("post") || preds.contains("tweet") || preds.contains("comment") {
        let pred = if preds.contains("post") {
            "post"
        } else if preds.contains("tweet") {
            "tweet"
        } else {
            "comment"
        };
        rules.push(make_rule("has-content", pred));
    }
    if preds.contains("likes") {
        rules.push(make_rule("has-engagement", "likes"));
    }
    if preds.contains("user") || preds.contains("username") {
        let pred = if preds.contains("user") { "user" } else { "username" };
        rules.push(make_rule("has-author", pred));
    }

    rules
}

/// Generate science-domain rules.
fn science_rules(preds: &HashSet<String>) -> Vec<Statement> {
    let mut rules = vec![
        domain_fact("science"),
        purpose_fact("research"),
        purpose_fact("analysis"),
    ];

    if preds.contains("measurement") || preds.contains("result") {
        let pred = if preds.contains("measurement") { "measurement" } else { "result" };
        rules.push(make_rule("has-data", pred));
    }
    if preds.contains("sample") {
        rules.push(make_rule("has-sample", "sample"));
    }
    if preds.contains("experiment") {
        rules.push(make_rule("has-experiment", "experiment"));
    }

    rules
}

/// Generate biology-domain rules.
fn bio_rules(preds: &HashSet<String>) -> Vec<Statement> {
    let mut rules = vec![
        domain_fact("biology"),
        purpose_fact("classification"),
    ];

    if preds.contains("species") {
        rules.push(make_rule("has-species", "species"));
    }
    if preds.contains("habitat") {
        rules.push(make_rule("has-habitat", "habitat"));
    }
    if preds.contains("weight") {
        rules.push(make_rule("has-measurement", "weight"));
    }
    if preds.contains("can-fly") {
        rules.push(make_rule("has-ability", "can-fly"));
    }

    if preds.contains("species") && preds.contains("habitat") {
        rules.push(make_rule_2cond("classified", "has-species", "has-habitat"));
    }

    rules
}

/// Generate config-domain rules.
fn config_rules(preds: &HashSet<String>) -> Vec<Statement> {
    let mut rules = vec![
        domain_fact("config"),
        purpose_fact("configuration"),
    ];

    if preds.contains("env") {
        rules.push(make_rule("has-environment", "env"));
    }
    if preds.contains("port") {
        rules.push(make_rule("has-network", "port"));
    }
    if preds.contains("debug") {
        rules.push(make_rule("has-debug", "debug"));
    }

    rules
}

/// Generate fallback rules for unknown domains.
/// Creates a `(has-<pred> $x)` rule for each unique predicate.
fn general_rules(preds: &HashSet<String>) -> Vec<Statement> {
    let mut rules = vec![domain_fact("general")];

    // Skip meta-predicates
    let skip = ["text", "domain", "purpose"];

    let mut sorted: Vec<&String> = preds.iter()
        .filter(|p| !skip.contains(&p.as_str()))
        .collect();
    sorted.sort();

    for pred in sorted {
        let head = format!("has-{}", pred);
        rules.push(make_rule(&head, pred));
    }

    rules
}

/// Auto-generate context rules based on the data structure.
/// This is the main entry point — call after ingestion.
pub fn auto_context(facts: &[Statement]) -> Vec<Statement> {
    if facts.is_empty() {
        return Vec::new();
    }

    let preds = extract_predicates(facts);
    let domain = detect_domain(facts);

    match domain {
        DataDomain::Market => market_rules(&preds),
        DataDomain::Medical => medical_rules(&preds),
        DataDomain::News => news_rules(&preds),
        DataDomain::Social => social_rules(&preds),
        DataDomain::Science => science_rules(&preds),
        DataDomain::Bio => bio_rules(&preds),
        DataDomain::Config => config_rules(&preds),
        DataDomain::General => general_rules(&preds),
    }
}

// ── Numeric Analysis ────────────────────────────────────────────────────

/// Extract float values for a given predicate from facts.
/// Returns Vec<(entity_name, value)>.
pub(crate) fn extract_float_values(facts: &[Statement], predicate: &str) -> Vec<(String, f64)> {
    let mut values = Vec::new();
    for stmt in facts {
        if let Statement::Fact { neuron, .. } = stmt {
            if let Neuron::Expression(parts) = neuron {
                if parts.len() >= 3 {
                    if let Neuron::Symbol(pred) = &parts[0] {
                        if pred == predicate {
                            let entity = match &parts[1] {
                                Neuron::Symbol(s) => s.clone(),
                                _ => continue,
                            };
                            let val = match &parts[2] {
                                Neuron::Value(QorValue::Float(f)) => *f,
                                Neuron::Value(QorValue::Int(n)) => *n as f64,
                                Neuron::Symbol(s) => s.parse::<f64>().unwrap_or(f64::NAN),
                                _ => continue,
                            };
                            if !val.is_nan() {
                                values.push((entity, val));
                            }
                        }
                    }
                }
            }
        }
    }
    values
}

fn analysis_fact(pred: &str, entity: &str, tv: TruthValue) -> Statement {
    Statement::Fact {
        neuron: Neuron::Expression(vec![
            Neuron::Symbol(pred.into()),
            Neuron::Symbol(entity.into()),
        ]),
        tv: Some(tv),
        decay: None,
    }
}

fn analysis_tv() -> TruthValue {
    TruthValue::new(0.90, 0.85)
}

fn weak_tv() -> TruthValue {
    TruthValue::new(0.85, 0.75)
}

/// Analyze raw data values and generate analysis facts.
/// Market: overbought/oversold, trending, bullish/bearish cross, volume anomalies.
/// Medical: fever detection.
pub fn auto_analysis(facts: &[Statement]) -> Vec<Statement> {
    if facts.is_empty() {
        return Vec::new();
    }

    let domain = detect_domain(facts);
    match domain {
        DataDomain::Market => market_analysis(facts),
        DataDomain::Medical => medical_analysis(facts),
        _ => Vec::new(),
    }
}

fn market_analysis(facts: &[Statement]) -> Vec<Statement> {
    let mut analysis = Vec::new();

    // RSI analysis
    let rsi_vals = extract_float_values(facts, "rsi");
    for (entity, val) in &rsi_vals {
        if *val > 70.0 {
            analysis.push(analysis_fact("overbought", entity, analysis_tv()));
        } else if *val < 30.0 {
            analysis.push(analysis_fact("oversold", entity, analysis_tv()));
        }
    }

    // ADX analysis
    let adx_vals = extract_float_values(facts, "adx");
    for (entity, val) in &adx_vals {
        if *val > 25.0 {
            analysis.push(analysis_fact("trending", entity, analysis_tv()));
        } else if *val < 20.0 {
            analysis.push(analysis_fact("range-bound", entity, analysis_tv()));
        }
    }

    // MACD vs Signal crossover
    let macd_vals: HashMap<String, f64> = extract_float_values(facts, "macd").into_iter().collect();
    let signal_vals: HashMap<String, f64> = extract_float_values(facts, "signal").into_iter().collect();
    for (entity, macd_val) in &macd_vals {
        if let Some(sig_val) = signal_vals.get(entity) {
            if macd_val > sig_val {
                analysis.push(analysis_fact("bullish-cross", entity, weak_tv()));
            } else {
                analysis.push(analysis_fact("bearish-cross", entity, weak_tv()));
            }
        }
    }

    // Close vs MA-200 trend position
    let close_vals: HashMap<String, f64> = extract_float_values(facts, "close").into_iter().collect();
    let ma200_vals: HashMap<String, f64> = extract_float_values(facts, "ma-200").into_iter().collect();
    for (entity, close_val) in &close_vals {
        if let Some(ma_val) = ma200_vals.get(entity) {
            if close_val > ma_val {
                analysis.push(analysis_fact("above-trend", entity, weak_tv()));
            } else {
                analysis.push(analysis_fact("below-trend", entity, weak_tv()));
            }
        }
    }

    // Volume anomaly (> 2x mean)
    let vol_vals = extract_float_values(facts, "volume");
    if !vol_vals.is_empty() {
        let mean: f64 = vol_vals.iter().map(|(_, v)| v).sum::<f64>() / vol_vals.len() as f64;
        let threshold = mean * 2.0;
        for (entity, val) in &vol_vals {
            if *val > threshold {
                analysis.push(analysis_fact("high-volume", entity, analysis_tv()));
            }
        }
    }

    analysis
}

fn medical_analysis(facts: &[Statement]) -> Vec<Statement> {
    let mut analysis = Vec::new();

    let temp_vals = extract_float_values(facts, "temperature");
    for (entity, val) in &temp_vals {
        if *val > 39.5 {
            analysis.push(analysis_fact("high-fever", entity, TruthValue::new(0.95, 0.90)));
        } else if *val > 38.0 {
            analysis.push(analysis_fact("fever-detected", entity, analysis_tv()));
        }
    }

    analysis
}

// ── Auto Queries ───────────────────────────────────────────────────────

fn make_query(pred: &str) -> Statement {
    Statement::Query {
        pattern: Neuron::Expression(vec![
            Neuron::Symbol(pred.into()),
            Neuron::Variable("x".into()),
        ]),
    }
}

/// Query with 2 variables: (pred $x $y) — for indicator, meaning facts.
fn make_query_3(pred: &str) -> Statement {
    Statement::Query {
        pattern: Neuron::Expression(vec![
            Neuron::Symbol(pred.into()),
            Neuron::Variable("x".into()),
            Neuron::Variable("y".into()),
        ]),
    }
}

/// Query with 3 variables: (pred $x $y $z) — for pattern-rate facts.
fn make_query_4(pred: &str) -> Statement {
    Statement::Query {
        pattern: Neuron::Expression(vec![
            Neuron::Symbol(pred.into()),
            Neuron::Variable("x".into()),
            Neuron::Variable("y".into()),
            Neuron::Variable("z".into()),
        ]),
    }
}

/// Query with 4 variables: (pred $a $b $c $d) — for co-occur facts.
fn make_query_5(pred: &str) -> Statement {
    Statement::Query {
        pattern: Neuron::Expression(vec![
            Neuron::Symbol(pred.into()),
            Neuron::Variable("a".into()),
            Neuron::Variable("b".into()),
            Neuron::Variable("c".into()),
            Neuron::Variable("d".into()),
        ]),
    }
}

/// Auto-generate queries based on the data domain.
/// These queries will auto-execute when the .qor file is run.
pub fn auto_queries(facts: &[Statement]) -> Vec<Statement> {
    if facts.is_empty() {
        return Vec::new();
    }

    let domain = detect_domain(facts);
    let mut queries = vec![
        make_query("domain"),
        make_query("purpose"),
    ];

    match domain {
        DataDomain::Market => {
            queries.push(make_query("buy-signal"));
            queries.push(make_query("sell-signal"));
            queries.push(make_query("anomaly"));
            queries.push(make_query("strong-setup"));
            queries.push(make_query_4("pattern-rate"));
            queries.push(make_query_5("co-occur"));
            queries.push(make_query_3("indicator"));
            queries.push(make_query_4("threshold"));
            queries.push(make_query_3("meaning"));
        }
        DataDomain::Medical => {
            queries.push(make_query("needs-review"));
            queries.push(make_query("critical-patient"));
            queries.push(make_query_3("indicator"));
            queries.push(make_query_4("threshold"));
            queries.push(make_query_3("meaning"));
        }
        DataDomain::Bio => {
            queries.push(make_query("classified"));
            queries.push(make_query("has-species"));
        }
        DataDomain::News => {
            queries.push(make_query("has-content"));
            queries.push(make_query("article"));
        }
        DataDomain::Social => {
            queries.push(make_query("has-engagement"));
            queries.push(make_query("has-author"));
        }
        DataDomain::Science => {
            queries.push(make_query("has-data"));
            queries.push(make_query("has-experiment"));
        }
        DataDomain::Config => {
            queries.push(make_query("has-environment"));
            queries.push(make_query("has-network"));
        }
        DataDomain::General => {
            // Query each auto-generated has-<pred> rule
            let preds = extract_predicates(facts);
            let skip = ["text", "domain", "purpose"];
            let mut sorted: Vec<&String> = preds.iter()
                .filter(|p| !skip.contains(&p.as_str()))
                .collect();
            sorted.sort();
            for pred in sorted.iter().take(5) {
                let head = format!("has-{}", pred);
                queries.push(make_query(&head));
            }
        }
    }

    queries
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sanitize::make_fact;
    use qor_core::neuron::Neuron;

    fn fact(pred: &str, entity: &str, val: &str) -> Statement {
        make_fact(vec![
            Neuron::Symbol(pred.into()),
            Neuron::Symbol(entity.into()),
            Neuron::Symbol(val.into()),
        ])
    }

    #[test]
    fn test_detect_market() {
        let facts = vec![
            fact("open", "row-0", "100"),
            fact("close", "row-0", "105"),
            fact("rsi", "row-0", "65"),
            fact("macd", "row-0", "1.5"),
        ];
        assert_eq!(detect_domain(&facts), DataDomain::Market);
    }

    #[test]
    fn test_detect_medical() {
        let facts = vec![
            fact("symptom", "alice", "fever"),
            fact("temperature", "alice", "38"),
            fact("admitted", "alice", "true"),
        ];
        assert_eq!(detect_domain(&facts), DataDomain::Medical);
    }

    #[test]
    fn test_detect_bio() {
        let facts = vec![
            fact("species", "tweety", "canary"),
            fact("can-fly", "tweety", "true"),
            fact("weight", "tweety", "0.5"),
        ];
        assert_eq!(detect_domain(&facts), DataDomain::Bio);
    }

    #[test]
    fn test_detect_general_fallback() {
        let facts = vec![
            fact("foo", "a", "1"),
            fact("bar", "a", "2"),
        ];
        assert_eq!(detect_domain(&facts), DataDomain::General);
    }

    #[test]
    fn test_market_rules_generated() {
        let facts = vec![
            fact("close", "row-0", "100"),
            fact("rsi", "row-0", "65"),
            fact("volume", "row-0", "1000"),
        ];
        let ctx = auto_context(&facts);

        // Should have domain + purpose facts
        let has_domain = ctx.iter().any(|s| {
            if let Statement::Fact { neuron, .. } = s {
                format!("{}", neuron).contains("domain") && format!("{}", neuron).contains("market")
            } else { false }
        });
        assert!(has_domain);

        // Should have rules
        let rule_count = ctx.iter().filter(|s| matches!(s, Statement::Rule { .. })).count();
        assert!(rule_count >= 3); // price-bar, has-indicator, has-volume at minimum
    }

    #[test]
    fn test_medical_rules_generated() {
        let facts = vec![
            fact("symptom", "alice", "fever"),
            fact("temperature", "alice", "38"),
        ];
        let ctx = auto_context(&facts);

        let has_domain = ctx.iter().any(|s| {
            if let Statement::Fact { neuron, .. } = s {
                format!("{}", neuron).contains("medical")
            } else { false }
        });
        assert!(has_domain);

        // Should have needs-review composite rule
        let has_composite = ctx.iter().any(|s| {
            if let Statement::Rule { head, .. } = s {
                format!("{}", head).contains("needs-review")
            } else { false }
        });
        assert!(has_composite);
    }

    #[test]
    fn test_general_creates_has_rules() {
        let facts = vec![
            fact("color", "ball", "red"),
            fact("size", "ball", "large"),
        ];
        let ctx = auto_context(&facts);

        let rule_heads: Vec<String> = ctx.iter().filter_map(|s| {
            if let Statement::Rule { head, .. } = s {
                Some(format!("{}", head))
            } else { None }
        }).collect();

        assert!(rule_heads.iter().any(|h| h.contains("has-color")));
        assert!(rule_heads.iter().any(|h| h.contains("has-size")));
    }

    #[test]
    fn test_empty_facts_no_context() {
        let ctx = auto_context(&[]);
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_news_detected() {
        let facts = vec![
            fact("title", "art-1", "breaking"),
            fact("source", "art-1", "bbc"),
            fact("content", "art-1", "something happened"),
        ];
        assert_eq!(detect_domain(&facts), DataDomain::News);
    }

    #[test]
    fn test_social_detected() {
        let facts = vec![
            fact("user", "p1", "alice"),
            fact("post", "p1", "hello"),
            fact("likes", "p1", "42"),
        ];
        assert_eq!(detect_domain(&facts), DataDomain::Social);
    }

    #[test]
    fn test_config_detected() {
        let facts = vec![
            fact("env", "cfg", "production"),
            fact("port", "cfg", "8080"),
            fact("debug", "cfg", "false"),
        ];
        assert_eq!(detect_domain(&facts), DataDomain::Config);
    }

    // ── Analysis tests ──

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

    #[test]
    fn test_market_analysis_overbought() {
        let facts = vec![
            float_fact("open", "row-0", 100.0),
            float_fact("close", "row-0", 105.0),
            float_fact("rsi", "row-0", 75.0),
            float_fact("rsi", "row-1", 25.0),
            float_fact("rsi", "row-2", 50.0),
        ];
        let analysis = auto_analysis(&facts);
        let overbought: Vec<_> = analysis.iter().filter(|s| {
            format!("{:?}", s).contains("overbought")
        }).collect();
        let oversold: Vec<_> = analysis.iter().filter(|s| {
            format!("{:?}", s).contains("oversold")
        }).collect();
        assert_eq!(overbought.len(), 1); // row-0
        assert_eq!(oversold.len(), 1);   // row-1
    }

    #[test]
    fn test_market_analysis_crossover() {
        let facts = vec![
            float_fact("open", "row-0", 100.0),
            float_fact("close", "row-0", 105.0),
            float_fact("macd", "row-0", 10.0),
            float_fact("signal", "row-0", 5.0),
            float_fact("macd", "row-1", 3.0),
            float_fact("signal", "row-1", 8.0),
        ];
        let analysis = auto_analysis(&facts);
        let bullish: Vec<_> = analysis.iter().filter(|s| {
            format!("{:?}", s).contains("bullish")
        }).collect();
        let bearish: Vec<_> = analysis.iter().filter(|s| {
            format!("{:?}", s).contains("bearish")
        }).collect();
        assert_eq!(bullish.len(), 1); // row-0: macd > signal
        assert_eq!(bearish.len(), 1); // row-1: macd < signal
    }

    #[test]
    fn test_market_analysis_adx_trending() {
        let facts = vec![
            float_fact("open", "row-0", 100.0),
            float_fact("close", "row-0", 105.0),
            float_fact("adx", "row-0", 35.0),
            float_fact("adx", "row-1", 15.0),
        ];
        let analysis = auto_analysis(&facts);
        let trending: Vec<_> = analysis.iter().filter(|s| {
            format!("{:?}", s).contains("trending")
        }).collect();
        let range: Vec<_> = analysis.iter().filter(|s| {
            format!("{:?}", s).contains("range-bound")
        }).collect();
        assert_eq!(trending.len(), 1);
        assert_eq!(range.len(), 1);
    }

    #[test]
    fn test_medical_analysis_fever() {
        let facts = vec![
            fact("symptom", "alice", "fever"),
            float_fact("temperature", "alice", 39.8),
            float_fact("temperature", "bob", 37.0),
            float_fact("temperature", "carol", 38.5),
            fact("admitted", "alice", "true"),
        ];
        let analysis = auto_analysis(&facts);
        let high: Vec<_> = analysis.iter().filter(|s| {
            format!("{:?}", s).contains("high-fever")
        }).collect();
        let fever: Vec<_> = analysis.iter().filter(|s| {
            format!("{:?}", s).contains("fever-detected")
        }).collect();
        assert_eq!(high.len(), 1);  // alice: 39.8
        assert_eq!(fever.len(), 1); // carol: 38.5
    }

    #[test]
    fn test_auto_analysis_empty() {
        assert!(auto_analysis(&[]).is_empty());
    }

    #[test]
    fn test_auto_analysis_general_no_analysis() {
        let facts = vec![
            fact("color", "ball", "red"),
            fact("size", "ball", "large"),
        ];
        assert!(auto_analysis(&facts).is_empty());
    }

    // ── Query tests ──

    #[test]
    fn test_market_queries() {
        let facts = vec![
            float_fact("open", "row-0", 100.0),
            float_fact("close", "row-0", 105.0),
            float_fact("rsi", "row-0", 65.0),
        ];
        let queries = auto_queries(&facts);
        assert!(queries.len() >= 4); // domain, purpose, buy-signal, sell-signal, ...
        let query_pats: Vec<String> = queries.iter().filter_map(|s| {
            if let Statement::Query { pattern } = s {
                Some(format!("{}", pattern))
            } else { None }
        }).collect();
        assert!(query_pats.iter().any(|q| q.contains("domain")));
        assert!(query_pats.iter().any(|q| q.contains("buy-signal")));
        assert!(query_pats.iter().any(|q| q.contains("sell-signal")));
        assert!(query_pats.iter().any(|q| q.contains("anomaly")));
    }

    #[test]
    fn test_medical_queries() {
        let facts = vec![
            fact("symptom", "alice", "fever"),
            fact("temperature", "alice", "38"),
            fact("admitted", "alice", "true"),
        ];
        let queries = auto_queries(&facts);
        let query_pats: Vec<String> = queries.iter().filter_map(|s| {
            if let Statement::Query { pattern } = s {
                Some(format!("{}", pattern))
            } else { None }
        }).collect();
        assert!(query_pats.iter().any(|q| q.contains("needs-review")));
        assert!(query_pats.iter().any(|q| q.contains("critical-patient")));
    }

    #[test]
    fn test_auto_queries_empty() {
        assert!(auto_queries(&[]).is_empty());
    }
}
