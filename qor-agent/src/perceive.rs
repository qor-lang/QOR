//! PERCEIVE — Generic data loader.
//!
//! Loads data items from files or API. Knows NOTHING about the domain.
//! Uses qor-bridge to convert any format (JSON, CSV, text) into QOR facts.
//! The DNA rules decide what to do with those facts.

use anyhow::{anyhow, Result};
use qor_core::neuron::Statement;
use std::path::Path;

use crate::types::DataItem;

/// Load data items from a source — directory, single file, or ID (fetched via HTTP).
pub fn load_items(source: &str) -> Result<Vec<DataItem>> {
    let path = Path::new(source);
    if path.is_dir() {
        load_from_directory(path)
    } else if path.is_file() {
        Ok(vec![load_from_file(path)?])
    } else {
        // Treat as a URL or ID — try HTTP fetch
        load_from_url(source)
    }
}

fn load_from_directory(dir: &Path) -> Result<Vec<DataItem>> {
    let mut items = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|x| x == "json" || x == "csv" || x == "txt" || x == "qor")
                .unwrap_or(false)
        })
        .collect();
    entries.sort();

    for path in &entries {
        match load_from_file(path) {
            Ok(item) => items.push(item),
            Err(e) => eprintln!("  [SKIP] {}: {e}", path.display()),
        }
    }
    Ok(items)
}

fn load_from_file(path: &Path) -> Result<DataItem> {
    let id = path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    // Use qor-bridge for generic format handling
    let facts = if ext == "qor" {
        // .qor files are already QOR statements — parse directly
        let source = std::fs::read_to_string(path)?;
        match qor_core::parser::parse(&source) {
            Ok(stmts) => stmts,
            Err(e) => return Err(anyhow!("parse error: {e}")),
        }
    } else {
        // JSON, CSV, text — let qor-bridge auto-detect and convert
        qor_bridge::feed_file(path).map_err(|e| anyhow!("{e}"))?
    };

    // Check if the data contains expected output (for scoring)
    // This is generic — looks for facts with "expected-" prefix
    let expected = extract_expected(&facts);

    Ok(DataItem { id, facts, expected })
}

fn load_from_url(source: &str) -> Result<Vec<DataItem>> {
    // Try as a direct URL
    let url = if source.starts_with("http") {
        source.to_string()
    } else {
        // Assume it's an ARC task ID — DNA could configure this URL pattern
        // but for now use a reasonable default
        format!("https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/training/{source}.json")
    };

    let resp = ureq::get(&url).call()
        .map_err(|e| anyhow!("Could not fetch {url}: {e}"))?;
    let body = resp.into_string()
        .map_err(|e| anyhow!("Read error: {e}"))?;
    let facts = qor_bridge::feed(&body).map_err(|e| anyhow!("{e}"))?;
    let expected = extract_expected(&facts);
    let id = source.split('/').last().unwrap_or(source)
        .trim_end_matches(".json").to_string();
    Ok(vec![DataItem { id, facts, expected }])
}

/// Extract expected-output facts from the data (if present).
/// Looks for any fact with predicate starting with "expected-".
fn extract_expected(facts: &[Statement]) -> Option<Vec<(String, Vec<String>)>> {
    use qor_core::neuron::Neuron;

    let mut expected = Vec::new();
    for fact in facts {
        if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = fact {
            if let Some(Neuron::Symbol(pred)) = parts.first() {
                if pred.starts_with("expected-") {
                    let args: Vec<String> = parts.iter().map(|p| p.to_string()).collect();
                    expected.push((pred.clone(), args));
                }
            }
        }
    }

    if expected.is_empty() { None } else { Some(expected) }
}
