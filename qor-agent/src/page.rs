//! Page-to-QOR — Convert browser page data into QOR facts.
//!
//! Pure data conversion. Zero domain logic.
//! Takes browser output (elements, text, network) → QOR Statement facts.
//! DNA rules fire on these facts to decide what the agent does next.

use qor_core::neuron::{Neuron, QorValue, Statement};
use qor_core::truth_value::TruthValue;
use serde_json::Value;

use crate::browser::PageElement;

/// Convert navigation result into QOR facts.
pub fn nav_to_facts(url: &str, title: &str) -> Vec<Statement> {
    vec![
        fact(vec![sym("page-url"), str_val(url)]),
        fact(vec![sym("page-title"), str_val(title)]),
    ]
}

/// Convert page text into a QOR fact.
pub fn text_to_facts(text: &str) -> Vec<Statement> {
    if text.is_empty() {
        return vec![];
    }
    // Truncate to avoid flooding the store
    let truncated = if text.len() > 4000 { &text[..4000] } else { text };
    vec![fact(vec![sym("page-text"), str_val(truncated)])]
}

/// Convert accessibility snapshot elements into QOR facts.
pub fn snapshot_to_facts(elements: &[PageElement]) -> Vec<Statement> {
    let mut facts = Vec::new();

    let mut link_count = 0u32;
    let mut button_count = 0u32;
    let mut form_count = 0u32;

    for elem in elements {
        // Count interactive elements
        match elem.role.as_str() {
            "link" => link_count += 1,
            "button" => button_count += 1,
            "textbox" | "searchbox" => form_count += 1,
            _ => {}
        }

        // Only emit facts for elements with UIDs (interactive) or meaningful content
        if let Some(uid) = &elem.uid {
            // (page-element <uid> <role> <name> <href-or-empty>)
            let href = elem.properties.as_ref()
                .and_then(|p| p.get("url"))
                .and_then(|v| v.as_str())
                .unwrap_or("");

            facts.push(fact(vec![
                sym("page-element"),
                sym(uid),
                sym(&elem.role),
                str_val(&elem.name),
                str_val(href),
            ]));
        } else if !elem.name.is_empty() {
            // Non-interactive but named elements (headings, paragraphs)
            match elem.role.as_str() {
                "heading" | "paragraph" | "banner" | "navigation"
                | "main" | "contentinfo" => {
                    facts.push(fact(vec![
                        sym("page-content"),
                        sym(&elem.role),
                        str_val(&elem.name),
                    ]));
                }
                _ => {}
            }
        }
    }

    // Summary counts
    facts.push(fact(vec![sym("page-link-count"), int_val(link_count as i64)]));
    facts.push(fact(vec![sym("page-button-count"), int_val(button_count as i64)]));
    facts.push(fact(vec![sym("page-form-count"), int_val(form_count as i64)]));

    facts
}

/// Convert network entries JSON into QOR facts.
pub fn network_to_facts(entries: &Value) -> Vec<Statement> {
    let mut facts = Vec::new();
    if let Value::Array(arr) = entries {
        for (i, entry) in arr.iter().enumerate() {
            let name = entry["name"].as_str().unwrap_or_default();
            let rtype = entry["type"].as_str().unwrap_or("unknown");
            let duration = entry["duration"].as_i64().unwrap_or(0);

            facts.push(fact(vec![
                sym("network-request"),
                str_val(&format!("r{}", i + 1)),
                sym(rtype),
                str_val(name),
                int_val(duration),
            ]));
        }
    }
    facts
}

/// Convert JavaScript eval result into a QOR fact.
pub fn js_result_to_facts(script: &str, result: &str) -> Vec<Statement> {
    vec![fact(vec![
        sym("js-result"),
        str_val(script),
        str_val(result),
    ])]
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn sym(s: &str) -> Neuron { Neuron::Symbol(s.to_string()) }
fn str_val(s: &str) -> Neuron { Neuron::Value(QorValue::Str(s.to_string())) }
fn int_val(i: i64) -> Neuron { Neuron::Value(QorValue::Int(i)) }

fn fact(parts: Vec<Neuron>) -> Statement {
    Statement::Fact {
        neuron: Neuron::Expression(parts),
        tv: Some(TruthValue::new(0.95, 0.90)),
        decay: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nav_to_facts() {
        let facts = nav_to_facts("https://example.com", "Example");
        assert_eq!(facts.len(), 2);
    }

    #[test]
    fn test_text_to_facts_empty() {
        let facts = text_to_facts("");
        assert!(facts.is_empty());
    }

    #[test]
    fn test_text_to_facts_normal() {
        let facts = text_to_facts("Hello world");
        assert_eq!(facts.len(), 1);
    }

    #[test]
    fn test_snapshot_to_facts() {
        let elements = vec![
            PageElement {
                role: "link".into(),
                name: "About".into(),
                uid: Some("s1".into()),
                properties: None,
            },
            PageElement {
                role: "button".into(),
                name: "Submit".into(),
                uid: Some("s2".into()),
                properties: None,
            },
            PageElement {
                role: "heading".into(),
                name: "Welcome".into(),
                uid: None,
                properties: None,
            },
        ];
        let facts = snapshot_to_facts(&elements);
        // 2 page-element + 1 page-content + 3 counts = 6
        assert_eq!(facts.len(), 6);
    }

    #[test]
    fn test_network_to_facts() {
        let entries = serde_json::json!([
            {"name": "https://api.example.com/data", "type": "fetch", "duration": 150},
            {"name": "https://cdn.example.com/style.css", "type": "css", "duration": 50},
        ]);
        let facts = network_to_facts(&entries);
        assert_eq!(facts.len(), 2);
    }

    #[test]
    fn test_js_result_to_facts() {
        let facts = js_result_to_facts("document.title", "Example");
        assert_eq!(facts.len(), 1);
    }
}
