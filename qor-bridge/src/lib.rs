// ── QOR Bridge — Universal Data Ingestion ──────────────────────────────
//
// Feed QOR any data format and it auto-converts to neurons.
// No formatting needed — feed anything, it organizes the way it needs.
//
// Supported formats:
// - JSON (serde_json)     — objects, arrays, nested structures
// - CSV  (csv crate)      — header-based datasets
// - Text (regex)          — natural language pattern extraction
// - KV   (nom)            — key=value, key: value, key\tvalue
// - Parquet (parquet)     — columnar binary data files
//
// Usage:
//   let facts = qor_bridge::feed(raw_data)?;
//   let facts = qor_bridge::feed_file(path)?;  // handles binary formats too

pub mod detect;
pub mod sanitize;
pub mod json;
pub mod csv_ingest;
pub mod text;
pub mod kv;
pub mod parquet_ingest;
pub mod context;
pub mod learn;
pub mod language;
pub mod llguidance;
pub mod dna;
pub mod grid;
pub mod template;
pub mod text_hint;
pub mod web_rules;
pub mod web_fetch;
#[cfg(feature = "web")]
pub mod web_search;
pub mod memory_graph;
pub mod kb_build;
pub mod oeis;
pub mod perceive;
pub mod dharmic;
pub mod solve;

use qor_core::neuron::Statement;
pub use detect::DataFormat;
pub use context::DataDomain;

/// Feed raw data to QOR — auto-detects format and converts to facts.
///
/// This is the primary entry point for data ingestion.
///
/// # Examples
/// ```
/// let facts = qor_bridge::feed(r#"{"name": "alice", "age": 30}"#).unwrap();
/// assert!(!facts.is_empty());
/// ```
pub fn feed(data: &str) -> Result<Vec<Statement>, String> {
    let format = detect::detect_format(data);
    let raw = feed_raw(data, format)?;
    Ok(enrich(raw))
}

/// Feed a file by path — auto-detects format including binary (parquet).
pub fn feed_file(path: &std::path::Path) -> Result<Vec<Statement>, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let raw = if ext == "parquet" {
        parquet_ingest::from_parquet_file(path)?
    } else {
        let data = std::fs::read_to_string(path)
            .map_err(|e| format!("could not read '{}': {}", path.display(), e))?;
        let format = detect::detect_format(&data);
        feed_raw(&data, format)?
    };
    Ok(enrich(raw))
}

/// Feed data without auto-context (raw parse only).
fn feed_raw(data: &str, format: DataFormat) -> Result<Vec<Statement>, String> {
    match format {
        DataFormat::Json => json::from_json(data),
        DataFormat::Csv => csv_ingest::from_csv(data),
        DataFormat::KeyValue => kv::from_kv(data),
        DataFormat::Text => text::from_text(data),
    }
}

/// Feed data with an explicit format (bypass auto-detection).
pub fn feed_as(data: &str, format: DataFormat) -> Result<Vec<Statement>, String> {
    let raw = feed_raw(data, format)?;
    Ok(enrich(raw))
}

/// Enrich raw facts: analyze, learn, return KNOWLEDGE (not raw data).
///
/// Pipeline:
/// 1. Raw facts come in (230k rows of data)
/// 2. auto_context → structural rules + domain/purpose facts
/// 3. auto_analysis → pattern detection (overbought, trending, etc.)
/// 4. learn → statistical summaries, pattern rates, co-occurrences, learned rules
/// 5. auto_queries → queries for learned predicates
/// 6. For grid data: deep perception (pair analysis, observations)
///
/// Returns: learned knowledge + context + queries (NOT raw data)
fn enrich(facts: Vec<Statement>) -> Vec<Statement> {
    if facts.is_empty() {
        return facts;
    }

    // Grid perception facts are knowledge — preserve them as-is.
    // Tabular data gets condensed through the learning pipeline.
    let grid_preds = [
        "grid-size", "grid-cell", "grid-object", "grid-obj-cell", "grid-obj-bbox",
        "grid-neighbor", "has-color", "color-count", "color-cell-count",
        "content-bbox", "object-count", "bbox-width", "bbox-height",
        "above", "left-of", "h-aligned", "v-aligned", "contains",
        "same-color", "bbox-overlap", "separator-row", "separator-col",
    ];
    let (grid_facts, data_facts): (Vec<Statement>, Vec<Statement>) =
        facts.into_iter().partition(|stmt| {
            if let Statement::Fact { neuron: qor_core::neuron::Neuron::Expression(parts), .. } = stmt {
                if let Some(qor_core::neuron::Neuron::Symbol(pred)) = parts.first() {
                    return grid_preds.contains(&pred.as_str());
                }
            }
            false
        });

    let ctx = context::auto_context(&data_facts);
    let analysis = context::auto_analysis(&data_facts);
    let learned = learn::learn(&data_facts, &analysis);
    let queries = context::auto_queries(&data_facts);

    let mut result = Vec::new();
    // Deep grid perception: remap IDs, add train-pair/test-input facts,
    // run compare_pair + compute_observations for ARC-like data.
    result.extend(grid::deep_grid_perception(grid_facts));
    result.extend(learned);    // stats + patterns + co-occurrences + learned rules
    result.extend(ctx);        // domain + purpose + structural rules
    result.extend(queries);    // auto queries
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feed_json_autodetect() {
        let facts = feed(r#"{"name": "alice", "age": 30}"#).unwrap();
        assert!(!facts.is_empty());
    }

    #[test]
    fn test_feed_csv_autodetect() {
        let data = "name,age,city\nalice,30,nyc\nbob,25,sf";
        let format = detect::detect_format(data);
        let facts = feed_raw(data, format).unwrap();
        assert_eq!(facts.len(), 4);
    }

    #[test]
    fn test_feed_kv_autodetect() {
        let data = "name=alice\nage=30\ncity=nyc";
        let format = detect::detect_format(data);
        let facts = feed_raw(data, format).unwrap();
        assert_eq!(facts.len(), 3);
    }

    #[test]
    fn test_feed_text_autodetect() {
        let data = "Tweety is a bird";
        let format = detect::detect_format(data);
        let facts = feed_raw(data, format).unwrap();
        assert_eq!(facts.len(), 1);
    }

    #[test]
    fn test_feed_as_explicit() {
        let facts = feed_as(r#"{"x": 1}"#, DataFormat::Json).unwrap();
        assert!(!facts.is_empty());
    }

    #[test]
    fn test_feed_medical_json() {
        let json = r#"[
            {"name": "alice", "symptom": "fever", "temperature": 38.5},
            {"name": "bob", "symptom": "cough", "temperature": 37.0}
        ]"#;
        let facts = feed_raw(json, DataFormat::Json).unwrap();
        assert_eq!(facts.len(), 4);
    }

    #[test]
    fn test_feed_empty_text() {
        let facts = feed("").unwrap();
        assert!(facts.is_empty());
    }

    #[test]
    fn test_feed_includes_auto_context() {
        // feed() should return raw facts PLUS auto-generated context
        let json = r#"[
            {"name": "alice", "symptom": "fever", "temperature": 38.5},
            {"name": "bob", "symptom": "cough", "temperature": 37.0}
        ]"#;
        let raw = feed_raw(json, DataFormat::Json).unwrap();
        let with_ctx = feed(json).unwrap();
        assert!(with_ctx.len() > raw.len(), "feed() should add context statements");
    }
}
