// ── Text → QOR Facts (regex extraction) ────────────────────────────────
//
// Extracts facts from plain text using pattern matching.
// "X is a Y" → (is-a X Y), "X has Y" → (has X Y), etc.
// Fallback: (text "raw sentence")

use qor_core::neuron::{Neuron, QorValue, Statement};
use crate::sanitize::{sanitize_symbol, infer_value, ingested_tv};

use regex::Regex;

/// Convert natural text into QOR facts using pattern extraction.
pub fn from_text(text: &str) -> Result<Vec<Statement>, String> {
    let patterns = build_patterns();
    let mut facts = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let mut matched = false;
        for (regex, predicate) in &patterns {
            if let Some(caps) = regex.captures(line) {
                let subject = caps.get(1).map_or("", |m| m.as_str()).trim();
                let object = caps.get(2).map_or("", |m| m.as_str()).trim();

                if !subject.is_empty() && !object.is_empty() {
                    let subj = sanitize_symbol(subject);
                    let obj_neuron = infer_value(object);

                    facts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol(predicate),
                            Neuron::symbol(&subj),
                            obj_neuron,
                        ]),
                        tv: Some(ingested_tv()),
                        decay: None,
                    });
                    matched = true;
                    break; // first matching pattern wins
                }
            }
        }

        if !matched {
            // Fallback: store raw sentence as text fact
            facts.push(Statement::Fact {
                neuron: Neuron::expression(vec![
                    Neuron::symbol("text"),
                    Neuron::Value(QorValue::Str(line.to_string())),
                ]),
                tv: Some(ingested_tv()),
                decay: None,
            });
        }
    }

    Ok(facts)
}

/// Build ordered extraction patterns. More specific patterns first.
fn build_patterns() -> Vec<(Regex, String)> {
    vec![
        // "X is a Y" / "X is an Y" → (is-a X Y)
        (
            Regex::new(r"(?i)^([A-Za-z][\w\s-]*?)\s+is\s+an?\s+(.+?)\.?$").unwrap(),
            "is-a".to_string(),
        ),
        // "X are Y" → (is X Y)
        (
            Regex::new(r"(?i)^([A-Za-z][\w\s-]*?)\s+are\s+(.+?)\.?$").unwrap(),
            "is".to_string(),
        ),
        // "X is Y" → (is X Y)
        (
            Regex::new(r"(?i)^([A-Za-z][\w\s-]*?)\s+is\s+(.+?)\.?$").unwrap(),
            "is".to_string(),
        ),
        // "X has a Y" → (has X Y)
        (
            Regex::new(r"(?i)^([A-Za-z][\w\s-]*?)\s+has\s+an?\s+(.+?)\.?$").unwrap(),
            "has".to_string(),
        ),
        // "X has Y" → (has X Y)
        (
            Regex::new(r"(?i)^([A-Za-z][\w\s-]*?)\s+has\s+(.+?)\.?$").unwrap(),
            "has".to_string(),
        ),
        // "X can Y" → (can X Y)
        (
            Regex::new(r"(?i)^([A-Za-z][\w\s-]*?)\s+can\s+(.+?)\.?$").unwrap(),
            "can".to_string(),
        ),
        // "X contains Y" → (contains X Y)
        (
            Regex::new(r"(?i)^([A-Za-z][\w\s-]*?)\s+contains?\s+(.+?)\.?$").unwrap(),
            "contains".to_string(),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_is_a_pattern() {
        let facts = from_text("Tweety is a bird").unwrap();
        assert_eq!(facts.len(), 1);
        let s = fact_str(&facts[0]);
        assert!(s.contains("is-a"));
        assert!(s.contains("tweety"));
        assert!(s.contains("bird"));
    }

    #[test]
    fn test_text_is_pattern() {
        let facts = from_text("The sky is blue").unwrap();
        assert_eq!(facts.len(), 1);
        let s = fact_str(&facts[0]);
        assert!(s.contains("is"));
        assert!(s.contains("blue"));
    }

    #[test]
    fn test_text_has_pattern() {
        let facts = from_text("Alice has a cat").unwrap();
        assert_eq!(facts.len(), 1);
        let s = fact_str(&facts[0]);
        assert!(s.contains("has"));
        assert!(s.contains("alice"));
        assert!(s.contains("cat"));
    }

    #[test]
    fn test_text_can_pattern() {
        let facts = from_text("Birds can fly").unwrap();
        assert_eq!(facts.len(), 1);
        let s = fact_str(&facts[0]);
        assert!(s.contains("can"));
        assert!(s.contains("birds"));
        assert!(s.contains("fly"));
    }

    #[test]
    fn test_text_are_pattern() {
        let facts = from_text("Dogs are loyal").unwrap();
        assert_eq!(facts.len(), 1);
        let s = fact_str(&facts[0]);
        assert!(s.contains("is")); // "are" maps to "is"
        assert!(s.contains("dogs"));
        assert!(s.contains("loyal"));
    }

    #[test]
    fn test_text_fallback() {
        let facts = from_text("random gibberish here 123").unwrap();
        assert_eq!(facts.len(), 1);
        let s = fact_str(&facts[0]);
        assert!(s.contains("text"));
        assert!(s.contains("random gibberish"));
    }

    #[test]
    fn test_text_multiple_lines() {
        let facts = from_text("Tweety is a bird\nAlice has a cat\nHello world").unwrap();
        assert_eq!(facts.len(), 3);
    }

    #[test]
    fn test_text_empty_lines_skipped() {
        let facts = from_text("Tweety is a bird\n\n\nAlice has a cat").unwrap();
        assert_eq!(facts.len(), 2);
    }

    #[test]
    fn test_text_case_insensitive() {
        let facts = from_text("TWEETY IS A BIRD").unwrap();
        assert_eq!(facts.len(), 1);
        let s = fact_str(&facts[0]);
        assert!(s.contains("is-a"));
    }

    #[test]
    fn test_text_trailing_period() {
        let facts = from_text("Tweety is a bird.").unwrap();
        assert_eq!(facts.len(), 1);
        let s = fact_str(&facts[0]);
        assert!(s.contains("is-a"));
        assert!(!s.contains(".")); // period stripped
    }

    fn fact_str(stmt: &Statement) -> String {
        match stmt {
            Statement::Fact { neuron, .. } => neuron.to_string(),
            _ => String::new(),
        }
    }
}
