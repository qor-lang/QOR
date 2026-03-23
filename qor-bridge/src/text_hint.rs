// ── Text Hint Parser ────────────────────────────────────────────────
//
// Phase 5A: Parse natural language descriptions into semantic hint facts.
// Text hints join grid facts in the same NeuronStore — cross-modal
// reasoning is automatic through forward chaining.

use qor_core::neuron::{Neuron, QorValue, Statement};
use qor_core::truth_value::TruthValue;

/// Keyword-to-hint mapping entry.
struct HintMapping {
    keywords: &'static [&'static str],
    hint_name: &'static str,
    confidence: f64,
}

/// All known keyword → hint mappings.
const HINT_MAPPINGS: &[HintMapping] = &[
    HintMapping { keywords: &["reflect", "mirror", "flip"], hint_name: "reflect", confidence: 0.90 },
    HintMapping { keywords: &["rotate", "turn", "spin"], hint_name: "rotate", confidence: 0.90 },
    HintMapping { keywords: &["gravity", "fall", "drop", "sink"], hint_name: "gravity", confidence: 0.90 },
    HintMapping { keywords: &["fill", "flood"], hint_name: "flood-fill", confidence: 0.90 },
    HintMapping { keywords: &["crop", "extract", "cut", "trim"], hint_name: "crop", confidence: 0.90 },
    HintMapping { keywords: &["scale", "enlarge", "zoom", "resize", "magnify"], hint_name: "scale", confidence: 0.90 },
    HintMapping { keywords: &["color", "recolor", "paint", "dye"], hint_name: "recolor", confidence: 0.85 },
    HintMapping { keywords: &["symmetry", "symmetric", "symmetrical"], hint_name: "symmetry", confidence: 0.90 },
    HintMapping { keywords: &["shift", "move", "slide", "translate"], hint_name: "shift", confidence: 0.90 },
    HintMapping { keywords: &["copy", "duplicate", "clone", "replicate"], hint_name: "copy", confidence: 0.85 },
    HintMapping { keywords: &["tile", "repeat", "pattern", "tessellate"], hint_name: "tile", confidence: 0.85 },
    HintMapping { keywords: &["border", "frame", "outline", "edge"], hint_name: "border", confidence: 0.85 },
    HintMapping { keywords: &["sort", "order", "arrange"], hint_name: "sort", confidence: 0.85 },
    HintMapping { keywords: &["count", "number", "quantity", "how many"], hint_name: "count", confidence: 0.80 },
    HintMapping { keywords: &["largest", "biggest", "maximum", "most"], hint_name: "select-largest", confidence: 0.85 },
    HintMapping { keywords: &["smallest", "minimum", "least", "tiny"], hint_name: "select-smallest", confidence: 0.85 },
    HintMapping { keywords: &["horizontal", "row", "left-right"], hint_name: "horizontal", confidence: 0.80 },
    HintMapping { keywords: &["vertical", "column", "up-down"], hint_name: "vertical", confidence: 0.80 },
    HintMapping { keywords: &["diagonal"], hint_name: "diagonal", confidence: 0.80 },
    HintMapping { keywords: &["overlap", "intersection", "common"], hint_name: "overlap", confidence: 0.80 },
];

/// Parse natural language text into semantic hint facts.
///
/// Returns `(text-hint <hint_name> <confidence>)` facts for each
/// detected keyword pattern.
pub fn parse_text_hints(text: &str) -> Vec<Statement> {
    let lower = text.to_lowercase();
    let mut hints = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for mapping in HINT_MAPPINGS {
        for &keyword in mapping.keywords {
            if lower.contains(keyword) && !seen.contains(mapping.hint_name) {
                seen.insert(mapping.hint_name);
                hints.push(Statement::Fact {
                    neuron: Neuron::Expression(vec![
                        Neuron::Symbol("text-hint".into()),
                        Neuron::Symbol(mapping.hint_name.into()),
                        Neuron::Value(QorValue::Float(mapping.confidence)),
                    ]),
                    tv: Some(TruthValue::new(
                        mapping.confidence,
                        mapping.confidence,
                    )),
                    decay: None,
                });
                break; // One hint per mapping
            }
        }
    }

    // Also extract directional modifiers
    if lower.contains("horizontal") || lower.contains("left") || lower.contains("right") {
        if !seen.contains("dir-horizontal") {
            seen.insert("dir-horizontal");
            hints.push(make_hint("dir-horizontal", 0.85));
        }
    }
    if lower.contains("vertical") || lower.contains("up") || lower.contains("down") {
        if !seen.contains("dir-vertical") {
            seen.insert("dir-vertical");
            hints.push(make_hint("dir-vertical", 0.85));
        }
    }
    if lower.contains("clockwise") || lower.contains("90") {
        if !seen.contains("dir-clockwise") {
            seen.insert("dir-clockwise");
            hints.push(make_hint("dir-clockwise", 0.85));
        }
    }

    hints
}

fn make_hint(name: &str, confidence: f64) -> Statement {
    Statement::Fact {
        neuron: Neuron::Expression(vec![
            Neuron::Symbol("text-hint".into()),
            Neuron::Symbol(name.into()),
            Neuron::Value(QorValue::Float(confidence)),
        ]),
        tv: Some(TruthValue::new(confidence, confidence)),
        decay: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hint_names(hints: &[Statement]) -> Vec<String> {
        hints.iter().filter_map(|s| {
            if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = s {
                if let Some(Neuron::Symbol(name)) = parts.get(1) {
                    return Some(name.clone());
                }
            }
            None
        }).collect()
    }

    #[test]
    fn test_reflect_keywords() {
        let hints = parse_text_hints("mirror the grid horizontally");
        let names = hint_names(&hints);
        assert!(names.contains(&"reflect".to_string()));
        assert!(names.contains(&"horizontal".to_string()) || names.contains(&"dir-horizontal".to_string()));
    }

    #[test]
    fn test_rotate_keyword() {
        let hints = parse_text_hints("rotate the grid 90 degrees clockwise");
        let names = hint_names(&hints);
        assert!(names.contains(&"rotate".to_string()));
        assert!(names.contains(&"dir-clockwise".to_string()));
    }

    #[test]
    fn test_multiple_hints() {
        let hints = parse_text_hints("fill the enclosed regions and recolor the border");
        let names = hint_names(&hints);
        assert!(names.contains(&"flood-fill".to_string()));
        assert!(names.contains(&"recolor".to_string()));
        assert!(names.contains(&"border".to_string()));
    }

    #[test]
    fn test_no_duplicates() {
        let hints = parse_text_hints("mirror mirror on the wall, flip it");
        let names = hint_names(&hints);
        let reflect_count = names.iter().filter(|n| *n == "reflect").count();
        assert_eq!(reflect_count, 1, "Should not have duplicate hints");
    }

    #[test]
    fn test_empty_text() {
        let hints = parse_text_hints("");
        assert!(hints.is_empty());
    }

    #[test]
    fn test_no_match() {
        let hints = parse_text_hints("hello world this is a random sentence");
        // Only "horizontal" might not match, no keywords
        assert!(hints.is_empty() || hints.len() <= 1);
    }

    #[test]
    fn test_gravity_drop() {
        let hints = parse_text_hints("drop all colored cells to the bottom");
        let names = hint_names(&hints);
        assert!(names.contains(&"gravity".to_string()));
    }

    #[test]
    fn test_case_insensitive() {
        let hints = parse_text_hints("ROTATE the GRID");
        let names = hint_names(&hints);
        assert!(names.contains(&"rotate".to_string()));
    }
}
