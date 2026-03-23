//! Real-time perception engine.
//!
//! Fetches live data from configured sources (HTTP APIs) and converts to
//! QOR facts tagged with temporal decay. Short-term memory — fades unless
//! QOR reasoning promotes it to long-term storage.
//!
//! Source configuration lives in .qor files:
//!   `(source <id> <url> <format> <interval-secs>)`
//!
//! Rust is PLUMBING only — what to fetch, how to interpret, what to keep
//! is all defined in .qor rules.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use qor_core::neuron::{Neuron, QorValue, Statement};
use qor_core::truth_value::TruthValue;

/// A configured data source parsed from `(source ...)` facts.
#[derive(Debug, Clone)]
pub struct SourceConfig {
    pub id: String,
    pub url: String,
    pub format: String,     // "json", "csv", "text"
    pub interval: Duration, // how often to fetch
}

/// Tracks last-fetch times per source.
#[derive(Debug, Default)]
pub struct PerceptionState {
    last_fetch: HashMap<String, Instant>,
}

/// Result of a single perception cycle.
#[derive(Debug)]
pub struct PerceptionResult {
    pub source_id: String,
    pub facts: Vec<Statement>,
    pub raw_size: usize,
    pub error: Option<String>,
}

/// Default decay rate for short-term perception facts.
/// At 0.15/heartbeat, facts drop below 0.10 confidence in ~6 heartbeats
/// (~30 minutes with 5-min interval) unless refreshed.
pub const SHORT_TERM_DECAY: f64 = 0.15;

/// Default truth value for perceived facts (high strength, moderate confidence).
pub const PERCEPTION_TV: TruthValue = TruthValue {
    strength: 0.90,
    confidence: 0.85,
};

/// Parse `(source <id> <url> <format> <interval>)` facts from a list of statements.
pub fn parse_sources(statements: &[Statement]) -> Vec<SourceConfig> {
    let mut sources = Vec::new();
    for stmt in statements {
        if let Statement::Fact { neuron, .. } = stmt {
            if let Neuron::Expression(parts) = neuron {
                if parts.len() >= 5 {
                    if let Some(Neuron::Symbol(pred)) = parts.first() {
                        if pred == "source" {
                            let id = match &parts[1] {
                                Neuron::Symbol(s) => s.clone(),
                                _ => continue,
                            };
                            let url = match &parts[2] {
                                Neuron::Value(QorValue::Str(s)) => s.clone(),
                                Neuron::Symbol(s) => s.clone(),
                                _ => continue,
                            };
                            let format = match &parts[3] {
                                Neuron::Symbol(s) => s.clone(),
                                _ => "json".to_string(),
                            };
                            let interval_secs = match &parts[4] {
                                Neuron::Value(QorValue::Int(n)) => *n as u64,
                                Neuron::Value(QorValue::Float(f)) => *f as u64,
                                _ => 300, // default 5 minutes
                            };
                            sources.push(SourceConfig {
                                id,
                                url,
                                format,
                                interval: Duration::from_secs(interval_secs),
                            });
                        }
                    }
                }
            }
        }
    }
    sources
}

impl PerceptionState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Check which sources are due for a fetch (interval elapsed).
    pub fn sources_due<'a>(&self, sources: &'a [SourceConfig]) -> Vec<&'a SourceConfig> {
        let now = Instant::now();
        sources
            .iter()
            .filter(|s| {
                match self.last_fetch.get(&s.id) {
                    Some(last) => now.duration_since(*last) >= s.interval,
                    None => true, // never fetched — due now
                }
            })
            .collect()
    }

    /// Fetch a single source and return QOR facts.
    pub fn fetch_source(&mut self, source: &SourceConfig) -> PerceptionResult {
        let mut result = PerceptionResult {
            source_id: source.id.clone(),
            facts: Vec::new(),
            raw_size: 0,
            error: None,
        };

        // HTTP GET
        let body = match ureq::get(&source.url)
            .timeout(std::time::Duration::from_secs(10))
            .call()
        {
            Ok(resp) => match resp.into_string() {
                Ok(s) => s,
                Err(e) => {
                    result.error = Some(format!("read error: {e}"));
                    return result;
                }
            },
            Err(e) => {
                result.error = Some(format!("fetch error: {e}"));
                return result;
            }
        };

        result.raw_size = body.len();

        // Parse based on format
        let raw_facts = match source.format.as_str() {
            "json" => parse_json_to_facts(&source.id, &body),
            "csv" => parse_csv_to_facts(&source.id, &body),
            _ => parse_text_to_facts(&source.id, &body),
        };

        // Tag each fact with decay and source metadata
        for neuron in raw_facts {
            result.facts.push(Statement::Fact {
                neuron,
                tv: Some(PERCEPTION_TV),
                decay: Some(SHORT_TERM_DECAY),
            });
        }

        // Add source metadata fact (when was it fetched)
        let ts = chrono_timestamp();
        result.facts.push(Statement::Fact {
            neuron: Neuron::Expression(vec![
                Neuron::Symbol("perceived".to_string()),
                Neuron::Symbol(source.id.clone()),
                Neuron::Value(QorValue::Str(ts)),
            ]),
            tv: Some(TruthValue::new(1.0, 0.99)),
            decay: Some(SHORT_TERM_DECAY),
        });

        // Mark fetch time
        self.last_fetch.insert(source.id.clone(), Instant::now());

        result
    }

    /// Run a full perception cycle: fetch all due sources.
    pub fn perceive(&mut self, sources: &[SourceConfig]) -> Vec<PerceptionResult> {
        let due: Vec<SourceConfig> = self.sources_due(sources)
            .into_iter()
            .cloned()
            .collect();

        let mut results = Vec::new();
        for source in &due {
            results.push(self.fetch_source(source));
        }
        results
    }
}

/// Parse JSON response into flat QOR facts.
/// Handles objects (key-value pairs) and arrays of objects.
fn parse_json_to_facts(source_id: &str, body: &str) -> Vec<Neuron> {
    let mut facts = Vec::new();
    let Ok(val) = serde_json::from_str::<serde_json::Value>(body) else {
        return facts;
    };

    match &val {
        serde_json::Value::Object(map) => {
            flatten_json_object(source_id, "", map, &mut facts, 0);
        }
        serde_json::Value::Array(arr) => {
            // For arrays, take first 50 items max
            for (i, item) in arr.iter().take(50).enumerate() {
                if let serde_json::Value::Object(map) = item {
                    let prefix = format!("item-{i}");
                    flatten_json_object(source_id, &prefix, map, &mut facts, 0);
                } else {
                    // Simple array: store as indexed values
                    let neuron = json_value_to_neuron(item);
                    facts.push(Neuron::Expression(vec![
                        Neuron::Symbol(format!("{source_id}-item")),
                        Neuron::Value(QorValue::Int(i as i64)),
                        neuron,
                    ]));
                }
            }
        }
        _ => {}
    }

    facts
}

/// Recursively flatten a JSON object into QOR facts.
fn flatten_json_object(
    source_id: &str,
    prefix: &str,
    map: &serde_json::Map<String, serde_json::Value>,
    facts: &mut Vec<Neuron>,
    depth: usize,
) {
    if depth > 3 {
        return; // cap recursion
    }
    for (key, val) in map {
        let pred = if prefix.is_empty() {
            format!("{source_id}-{key}")
        } else {
            format!("{source_id}-{prefix}-{key}")
        };
        // Sanitize predicate: replace spaces/special chars
        let pred = sanitize_predicate(&pred);

        match val {
            serde_json::Value::Object(inner) => {
                flatten_json_object(source_id, &format!("{}{key}-", if prefix.is_empty() { "" } else { prefix }), inner, facts, depth + 1);
            }
            serde_json::Value::Array(arr) => {
                for (i, item) in arr.iter().take(20).enumerate() {
                    let sub_pred = format!("{pred}-{i}");
                    facts.push(Neuron::Expression(vec![
                        Neuron::Symbol(sub_pred),
                        json_value_to_neuron(item),
                    ]));
                }
            }
            _ => {
                facts.push(Neuron::Expression(vec![
                    Neuron::Symbol(pred),
                    json_value_to_neuron(val),
                ]));
            }
        }
    }
}

/// Convert a JSON value to a QOR Neuron.
fn json_value_to_neuron(val: &serde_json::Value) -> Neuron {
    match val {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Neuron::Value(QorValue::Int(i))
            } else if let Some(f) = n.as_f64() {
                Neuron::Value(QorValue::Float(f))
            } else {
                Neuron::Value(QorValue::Str(n.to_string()))
            }
        }
        serde_json::Value::String(s) => Neuron::Value(QorValue::Str(s.clone())),
        serde_json::Value::Bool(b) => Neuron::Value(QorValue::Bool(*b)),
        serde_json::Value::Null => Neuron::Symbol("null".to_string()),
        _ => Neuron::Value(QorValue::Str(val.to_string())),
    }
}

/// Parse CSV response into QOR facts.
fn parse_csv_to_facts(source_id: &str, body: &str) -> Vec<Neuron> {
    let mut facts = Vec::new();
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(body.as_bytes());

    let headers: Vec<String> = match rdr.headers() {
        Ok(h) => h.iter().map(|s| sanitize_predicate(&format!("{source_id}-{s}"))).collect(),
        Err(_) => return facts,
    };

    for (row_idx, result) in rdr.records().take(100).enumerate() {
        let Ok(record) = result else { continue };
        for (col_idx, field) in record.iter().enumerate() {
            if col_idx >= headers.len() {
                break;
            }
            let neuron = if let Ok(n) = field.parse::<i64>() {
                Neuron::Value(QorValue::Int(n))
            } else if let Ok(f) = field.parse::<f64>() {
                Neuron::Value(QorValue::Float(f))
            } else {
                Neuron::Value(QorValue::Str(field.to_string()))
            };
            facts.push(Neuron::Expression(vec![
                Neuron::Symbol(headers[col_idx].clone()),
                Neuron::Value(QorValue::Int(row_idx as i64)),
                neuron,
            ]));
        }
    }

    facts
}

/// Parse plain text response into QOR facts (one fact per non-empty line).
fn parse_text_to_facts(source_id: &str, body: &str) -> Vec<Neuron> {
    body.lines()
        .take(100)
        .filter(|line| !line.trim().is_empty())
        .enumerate()
        .map(|(i, line)| {
            Neuron::Expression(vec![
                Neuron::Symbol(format!("{source_id}-line")),
                Neuron::Value(QorValue::Int(i as i64)),
                Neuron::Value(QorValue::Str(line.trim().to_string())),
            ])
        })
        .collect()
}

/// Sanitize a string into a valid QOR predicate name.
fn sanitize_predicate(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

/// Simple timestamp string (no chrono dependency — use system time).
fn chrono_timestamp() -> String {
    use std::time::SystemTime;
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{secs}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use qor_core::parser;

    #[test]
    fn test_parse_sources() {
        let input = r#"
            (source weather "https://api.open-meteo.com/v1/forecast" json 300)
            (source news "https://hn.algolia.com/api/v1/search" json 600)
        "#;
        let stmts = parser::parse(input).unwrap();
        let sources = parse_sources(&stmts);
        assert_eq!(sources.len(), 2);
        assert_eq!(sources[0].id, "weather");
        assert_eq!(sources[0].format, "json");
        assert_eq!(sources[0].interval, Duration::from_secs(300));
        assert_eq!(sources[1].id, "news");
        assert_eq!(sources[1].interval, Duration::from_secs(600));
    }

    #[test]
    fn test_parse_json_flat_object() {
        let facts = parse_json_to_facts("test", r#"{"temp": 25.5, "wind": 10}"#);
        assert_eq!(facts.len(), 2);
    }

    #[test]
    fn test_parse_json_nested() {
        let facts = parse_json_to_facts("btc", r#"{"bitcoin":{"usd":67000,"usd_24h_change":2.5}}"#);
        assert!(facts.len() >= 2);
    }

    #[test]
    fn test_parse_json_array() {
        let facts = parse_json_to_facts("items", r#"[1, 2, 3]"#);
        assert_eq!(facts.len(), 3);
    }

    #[test]
    fn test_parse_csv() {
        let facts = parse_csv_to_facts("data", "name,value\nalpha,100\nbeta,200\n");
        assert_eq!(facts.len(), 4); // 2 rows × 2 columns
    }

    #[test]
    fn test_parse_text() {
        let facts = parse_text_to_facts("log", "hello world\nfoo bar\n\nbaz\n");
        assert_eq!(facts.len(), 3);
    }

    #[test]
    fn test_sanitize_predicate() {
        assert_eq!(sanitize_predicate("btc-price"), "btc-price");
        assert_eq!(sanitize_predicate("some key!@#"), "some-key---");
    }

    #[test]
    fn test_sources_due_first_time() {
        let state = PerceptionState::new();
        let sources = vec![SourceConfig {
            id: "test".into(),
            url: "http://example.com".into(),
            format: "json".into(),
            interval: Duration::from_secs(60),
        }];
        let due = state.sources_due(&sources);
        assert_eq!(due.len(), 1); // never fetched = due
    }

    #[test]
    fn test_short_term_decay_value() {
        // After 6 heartbeats: 0.85 - 6*0.15 = -0.05 → clamped to 0.0
        // So facts survive ~5-6 heartbeats (25-30 min at 5-min interval)
        let mut conf = PERCEPTION_TV.confidence;
        for _ in 0..6 {
            conf = (conf - SHORT_TERM_DECAY).max(0.0);
        }
        assert!(conf < 0.01, "should be near zero after 6 decays");
    }
}
