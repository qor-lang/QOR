// ── Web Rule Extraction ─────────────────────────────────────────────
//
// Extract candidate QOR rules and facts from plain text (web content).
// Uses regex patterns to find causal, conditional, and definitional
// structures in natural language and convert to QOR statements.
//
// No web feature needed — this is pure text processing.

use regex::Regex;
use qor_core::neuron::{Neuron, QorValue, Statement};
use qor_core::truth_value::TruthValue;

use crate::sanitize::sanitize_symbol;

/// A candidate rule extracted from web text.
#[derive(Debug, Clone)]
pub struct ExtractedRule {
    pub rule_text: String,
    pub source_url: String,
    pub source_sentence: String,
    pub confidence: f64,
}

/// A fact extracted from web text.
#[derive(Debug, Clone)]
pub struct ExtractedFact {
    pub statement: Statement,
    pub source_url: String,
    pub source_sentence: String,
}

/// Extract candidate QOR rules from text.
///
/// Looks for causal/conditional/logical patterns:
///   - "if X then Y"
///   - "X causes Y"  / "X leads to Y"
///   - "when X, Y"
///   - "X implies Y"
///   - "X results in Y"
///   - "X therefore Y"
pub fn extract_rules(text: &str, source_url: &str) -> Vec<ExtractedRule> {
    let mut rules = Vec::new();

    let patterns: Vec<(Regex, f64, &str)> = vec![
        // "if X then Y" → (Y) if (X)
        (Regex::new(r"(?i)\bif\s+(.+?)\s+then\s+(.+?)[\.\;\,]").unwrap(), 0.75, "conditional"),
        // "when X, Y" → (Y) if (X)
        (Regex::new(r"(?i)\bwhen\s+(.+?)\s*,\s*(.+?)[\.\;]").unwrap(), 0.70, "conditional"),
        // "X causes Y" → (Y) if (X)
        (Regex::new(r"(?i)\b(.+?)\s+causes?\s+(.+?)[\.\;\,]").unwrap(), 0.65, "causal"),
        // "X leads to Y" → (Y) if (X)
        (Regex::new(r"(?i)\b(.+?)\s+leads?\s+to\s+(.+?)[\.\;\,]").unwrap(), 0.65, "causal"),
        // "X results in Y" → (Y) if (X)
        (Regex::new(r"(?i)\b(.+?)\s+results?\s+in\s+(.+?)[\.\;\,]").unwrap(), 0.65, "causal"),
        // "X implies Y" → (Y) if (X)
        (Regex::new(r"(?i)\b(.+?)\s+implies?\s+(.+?)[\.\;\,]").unwrap(), 0.70, "logical"),
        // "X therefore Y" → (Y) if (X)
        (Regex::new(r"(?i)\b(.+?)\s*,?\s+therefore\s+(.+?)[\.\;]").unwrap(), 0.65, "logical"),
    ];

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.len() < 10 || line.len() > 300 {
            continue;
        }

        for (re, confidence, _kind) in &patterns {
            if let Some(caps) = re.captures(line) {
                let antecedent = caps.get(1).map(|m| m.as_str().trim()).unwrap_or("");
                let consequent = caps.get(2).map(|m| m.as_str().trim()).unwrap_or("");

                // Skip if either side is too short or too long
                if antecedent.len() < 3 || consequent.len() < 3 {
                    continue;
                }
                if antecedent.len() > 100 || consequent.len() > 100 {
                    continue;
                }

                let ante_sym = text_to_predicate(antecedent);
                let cons_sym = text_to_predicate(consequent);

                if ante_sym.is_empty() || cons_sym.is_empty() {
                    continue;
                }

                let rule_text = format!(
                    "({}) if ({}) <{:.2}, {:.2}>",
                    cons_sym, ante_sym, confidence, confidence
                );

                rules.push(ExtractedRule {
                    rule_text,
                    source_url: source_url.to_string(),
                    source_sentence: line.to_string(),
                    confidence: *confidence,
                });

                break; // One rule per line
            }
        }
    }

    rules
}

/// Extract factual statements from text.
///
/// Looks for:
///   - "X is defined as Y" / "X means Y" → (meaning X "Y")
///   - "X is a Y" / "X are Y" → (is-a X Y)
///   - "X has Y" / "X contains Y" → (has X Y)
///   - "X = N" / "X is N" (numeric) → (web-fact X N)
pub fn extract_facts(text: &str, source_url: &str) -> Vec<ExtractedFact> {
    let mut facts = Vec::new();

    let def_re = Regex::new(r"(?i)\b(\w[\w\s]{1,30}?)\s+(?:is defined as|means|refers to)\s+(.+?)[\.\;]").unwrap();
    let isa_re = Regex::new(r"(?i)\b(\w[\w\s]{1,20}?)\s+(?:is a|is an|are)\s+(\w[\w\s]{1,30}?)[\.\;,]").unwrap();
    let has_re = Regex::new(r"(?i)\b(\w[\w\s]{1,20}?)\s+(?:has|contains|includes)\s+(\w[\w\s]{1,30}?)[\.\;,]").unwrap();
    let num_re = Regex::new(r"(?i)\b(\w[\w\s]{1,20}?)\s+(?:is|=|equals)\s+(\d+(?:\.\d+)?)[\.\;,\s]").unwrap();
    let prop_re = Regex::new(r"(?i)\b(\w[\w\s]{1,20}?)\s+(?:is|are)\s+([\w\-]+)[\.\;,]").unwrap();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.len() < 8 || line.len() > 300 {
            continue;
        }

        // Definition patterns → (meaning term "definition")
        if let Some(caps) = def_re.captures(line) {
            let term = caps.get(1).map(|m| m.as_str().trim()).unwrap_or("");
            let def = caps.get(2).map(|m| m.as_str().trim()).unwrap_or("");
            if term.len() >= 2 && def.len() >= 3 {
                let sym = sanitize_symbol(term);
                facts.push(ExtractedFact {
                    statement: Statement::Fact {
                        neuron: Neuron::Expression(vec![
                            Neuron::Symbol("meaning".into()),
                            Neuron::Symbol(sym),
                            Neuron::str_val(def),
                        ]),
                        tv: Some(TruthValue::new(0.70, 0.70)),
                        decay: None,
                    },
                    source_url: source_url.to_string(),
                    source_sentence: line.to_string(),
                });
                continue;
            }
        }

        // Numeric facts → (web-fact term value)
        if let Some(caps) = num_re.captures(line) {
            let term = caps.get(1).map(|m| m.as_str().trim()).unwrap_or("");
            let val_str = caps.get(2).map(|m| m.as_str()).unwrap_or("");
            if term.len() >= 2 {
                let sym = sanitize_symbol(term);
                let val = if val_str.contains('.') {
                    Neuron::Value(QorValue::Float(val_str.parse().unwrap_or(0.0)))
                } else {
                    Neuron::Value(QorValue::Int(val_str.parse().unwrap_or(0)))
                };
                facts.push(ExtractedFact {
                    statement: Statement::Fact {
                        neuron: Neuron::Expression(vec![
                            Neuron::Symbol("web-fact".into()),
                            Neuron::Symbol(sym),
                            val,
                        ]),
                        tv: Some(TruthValue::new(0.65, 0.65)),
                        decay: None,
                    },
                    source_url: source_url.to_string(),
                    source_sentence: line.to_string(),
                });
                continue;
            }
        }

        // "is a" patterns → (is-a term category)
        if let Some(caps) = isa_re.captures(line) {
            let term = caps.get(1).map(|m| m.as_str().trim()).unwrap_or("");
            let cat = caps.get(2).map(|m| m.as_str().trim()).unwrap_or("");
            if term.len() >= 2 && cat.len() >= 2 {
                let t_sym = sanitize_symbol(term);
                let c_sym = sanitize_symbol(cat);
                facts.push(ExtractedFact {
                    statement: Statement::Fact {
                        neuron: Neuron::Expression(vec![
                            Neuron::Symbol("is-a".into()),
                            Neuron::Symbol(t_sym),
                            Neuron::Symbol(c_sym),
                        ]),
                        tv: Some(TruthValue::new(0.70, 0.70)),
                        decay: None,
                    },
                    source_url: source_url.to_string(),
                    source_sentence: line.to_string(),
                });
                continue;
            }
        }

        // "has" patterns → (has term property)
        if let Some(caps) = has_re.captures(line) {
            let term = caps.get(1).map(|m| m.as_str().trim()).unwrap_or("");
            let prop = caps.get(2).map(|m| m.as_str().trim()).unwrap_or("");
            if term.len() >= 2 && prop.len() >= 2 {
                let t_sym = sanitize_symbol(term);
                let p_sym = sanitize_symbol(prop);
                facts.push(ExtractedFact {
                    statement: Statement::Fact {
                        neuron: Neuron::Expression(vec![
                            Neuron::Symbol("has".into()),
                            Neuron::Symbol(t_sym),
                            Neuron::Symbol(p_sym),
                        ]),
                        tv: Some(TruthValue::new(0.65, 0.65)),
                        decay: None,
                    },
                    source_url: source_url.to_string(),
                    source_sentence: line.to_string(),
                });
                continue;
            }
        }

        // Property patterns → (is term property)
        if let Some(caps) = prop_re.captures(line) {
            let term = caps.get(1).map(|m| m.as_str().trim()).unwrap_or("");
            let prop = caps.get(2).map(|m| m.as_str().trim()).unwrap_or("");
            if term.len() >= 2 && prop.len() >= 2 && prop.len() <= 20 {
                let t_sym = sanitize_symbol(term);
                let p_sym = sanitize_symbol(prop);
                facts.push(ExtractedFact {
                    statement: Statement::Fact {
                        neuron: Neuron::Expression(vec![
                            Neuron::Symbol("is".into()),
                            Neuron::Symbol(t_sym),
                            Neuron::Symbol(p_sym),
                        ]),
                        tv: Some(TruthValue::new(0.60, 0.60)),
                        decay: None,
                    },
                    source_url: source_url.to_string(),
                    source_sentence: line.to_string(),
                });
            }
        }
    }

    facts
}

/// Convert a natural language phrase to a QOR predicate-style symbol.
///
/// "the temperature rises" → "temperature-rises"
/// "high blood pressure" → "high-blood-pressure"
fn text_to_predicate(text: &str) -> String {
    let text = text.trim().to_lowercase();

    // Remove common stop words
    let stop_words = ["the", "a", "an", "is", "are", "was", "were", "be",
                      "been", "being", "have", "has", "had", "do", "does",
                      "did", "will", "would", "could", "should", "may",
                      "might", "shall", "can", "very", "quite", "rather",
                      "that", "this", "these", "those", "it", "its"];

    let words: Vec<&str> = text.split_whitespace()
        .filter(|w| !stop_words.contains(w))
        .collect();

    if words.is_empty() {
        return String::new();
    }

    // Take first 4 words max, join with hyphens
    let result: Vec<&str> = words.into_iter().take(4).collect();
    let joined = result.join("-");

    sanitize_symbol(&joined)
}

/// Build search URLs for a topic from configured source bases.
pub fn build_search_urls(topic: &str, source_bases: &[String]) -> Vec<String> {
    let encoded = topic.replace(' ', "+");
    let slug = topic.replace(' ', "_");

    let mut urls = Vec::new();

    for base in source_bases {
        if base.contains("wikipedia.org") {
            urls.push(format!("{}wiki/{}", base, slug));
        } else if base.contains("arxiv.org") {
            urls.push(format!("{}?query={}&searchtype=all", base, encoded));
        } else if base.contains("stackoverflow.com") {
            urls.push(format!("{}?q={}", base, encoded));
        } else {
            // Generic: append encoded query
            let sep = if base.ends_with('/') { "" } else { "/" };
            urls.push(format!("{}{}{}", base, sep, encoded));
        }
    }

    urls
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_if_then_rule() {
        let text = "If the temperature rises then the pressure increases.";
        let rules = extract_rules(text, "https://example.com");
        assert!(!rules.is_empty(), "Should extract 'if...then' rule");
        let rule = &rules[0];
        assert!(rule.rule_text.contains("if"));
        assert!(rule.rule_text.contains("pressure"));
        assert!(rule.rule_text.contains("temperature"));
        assert!(rule.confidence >= 0.70);
    }

    #[test]
    fn test_extract_when_rule() {
        let text = "When water freezes, it expands.";
        let rules = extract_rules(text, "https://example.com");
        assert!(!rules.is_empty(), "Should extract 'when' rule");
        assert!(rules[0].rule_text.contains("if"));
    }

    #[test]
    fn test_extract_causes_rule() {
        let text = "Smoking causes lung cancer.";
        let rules = extract_rules(text, "https://example.com");
        assert!(!rules.is_empty(), "Should extract 'causes' rule");
        assert!(rules[0].rule_text.contains("lung-cancer") || rules[0].rule_text.contains("lung"));
    }

    #[test]
    fn test_extract_definition_fact() {
        let text = "Photosynthesis is defined as the process by which plants make food.";
        let facts = extract_facts(text, "https://example.com");
        assert!(!facts.is_empty(), "Should extract definition");
        let fact_str = format!("{:?}", facts[0].statement);
        assert!(fact_str.contains("meaning"));
        assert!(fact_str.contains("photosynthesis"));
    }

    #[test]
    fn test_extract_isa_fact() {
        let text = "A penguin is a bird.";
        let facts = extract_facts(text, "https://example.com");
        assert!(facts.iter().any(|f| {
            let s = format!("{:?}", f.statement);
            s.contains("is-a") && s.contains("penguin")
        }), "Should extract 'is a' fact");
    }

    #[test]
    fn test_extract_numeric_fact() {
        let text = "The speed of light is 299792458.";
        let facts = extract_facts(text, "https://example.com");
        assert!(facts.iter().any(|f| {
            let s = format!("{:?}", f.statement);
            s.contains("web-fact") && s.contains("299792458")
        }), "Should extract numeric fact");
    }

    #[test]
    fn test_extract_has_fact() {
        let text = "Water has hydrogen atoms.";
        let facts = extract_facts(text, "https://example.com");
        assert!(facts.iter().any(|f| {
            let s = format!("{:?}", f.statement);
            s.contains("has") && s.contains("water")
        }), "Should extract 'has' fact");
    }

    #[test]
    fn test_text_to_predicate() {
        assert_eq!(text_to_predicate("the temperature rises"), "temperature-rises");
        assert_eq!(text_to_predicate("high blood pressure"), "high-blood-pressure");
        assert_eq!(text_to_predicate(""), "");
    }

    #[test]
    fn test_skip_short_lines() {
        let text = "Hi.\nOK.\nYes.";
        let rules = extract_rules(text, "");
        assert!(rules.is_empty(), "Short lines should be skipped");
    }

    #[test]
    fn test_build_search_urls() {
        let sources = vec![
            "https://en.wikipedia.org/".to_string(),
            "https://arxiv.org/search/".to_string(),
        ];
        let urls = build_search_urls("grid transformation", &sources);
        assert_eq!(urls.len(), 2);
        assert!(urls[0].contains("wiki/grid_transformation"));
        assert!(urls[1].contains("query=grid+transformation"));
    }

    #[test]
    fn test_extract_implies_rule() {
        let text = "Poverty implies limited access to education.";
        let rules = extract_rules(text, "https://example.com");
        assert!(!rules.is_empty(), "Should extract 'implies' rule");
    }

    #[test]
    fn test_multiple_extractions() {
        let text = "\
If rain falls then the ground gets wet.
Water is a liquid.
The boiling point is 100.
Plants have roots.
Gravity causes objects to fall.";
        let rules = extract_rules(text, "");
        let facts = extract_facts(text, "");
        assert!(rules.len() >= 2, "Should extract at least 2 rules, got {}", rules.len());
        assert!(facts.len() >= 2, "Should extract at least 2 facts, got {}", facts.len());
    }
}
