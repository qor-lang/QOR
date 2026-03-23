// ── Template Instantiation Engine ────────────────────────────────────
//
// Phase 1C: Fill QOR rule templates from cross-pair observations,
// verify against training data.
//
// Templates are QOR rules with $HOLE_* placeholders.
// Observations (from grid.rs compute_observations) provide fill values.
// Verification: clone session, inject filled rule + input, chain, compare.

use std::collections::HashMap;
use qor_core::neuron::{Neuron, Statement};

/// A template with typed holes.
#[derive(Debug, Clone)]
pub struct Template {
    pub name: String,
    pub rule_text: String,
    pub holes: Vec<Hole>,
    /// Which observation triggers this template.
    pub trigger: TemplateTrigger,
}

/// A typed hole in a template.
#[derive(Debug, Clone)]
pub struct Hole {
    pub placeholder: String,
    pub hole_type: HoleType,
}

/// What type of value fills a hole.
#[derive(Debug, Clone, PartialEq)]
pub enum HoleType {
    Color,
    Int,
    Predicate,
    Direction,
    Transform,
}

/// When to try this template.
#[derive(Debug, Clone)]
pub enum TemplateTrigger {
    /// Requires specific obs-consistent facts.
    ObsConsistent(Vec<String>),
    /// Requires specific obs-parameter facts.
    ObsParameter(String),
    /// Always try.
    Always,
}

/// A filled template ready for verification.
#[derive(Debug, Clone)]
pub struct FilledTemplate {
    pub name: String,
    pub rule_text: String,
    pub fill_values: HashMap<String, String>,
    pub score: f64,
}

/// Load templates from a .qor file with $HOLE_* placeholders.
///
/// Parses the file, extracts each rule, scans for $HOLE_* placeholders,
/// auto-detects hole types from naming conventions, and reads triggers
/// from preceding comments.
pub fn load_templates_from_qor(text: &str) -> Vec<Template> {
    let mut templates = Vec::new();
    let mut current_name = String::new();
    let mut current_trigger = TemplateTrigger::Always;

    for line in text.lines() {
        let trimmed = line.trim();

        // Parse template name from comments like ";; ── T1: Direct color remap ──"
        if let Some(rest) = trimmed.strip_prefix(";; ── T") {
            if let Some(name_part) = rest.split(':').nth(1) {
                current_name = name_part.trim()
                    .trim_end_matches('─')
                    .trim()
                    .to_lowercase()
                    .replace(' ', "-");
                // Reset trigger for each new template section
                current_trigger = TemplateTrigger::Always;
            }
            continue;
        }

        // Parse trigger from comments like ";; Triggered by: (obs-consistent reflect-h)"
        if let Some(rest) = trimmed.strip_prefix(";; Triggered by:") {
            let trigger_text = rest.trim();
            if trigger_text.contains("obs-consistent") {
                if let Some(tail) = trigger_text.split("obs-consistent").nth(1) {
                    let name = tail.trim()
                        .trim_start_matches(')')
                        .trim()
                        .trim_end_matches(')')
                        .trim();
                    if !name.is_empty() {
                        current_trigger = TemplateTrigger::ObsConsistent(vec![name.to_string()]);
                    }
                }
            } else if trigger_text.contains("obs-parameter") {
                if let Some(tail) = trigger_text.split("obs-parameter").nth(1) {
                    let name = tail.trim()
                        .trim_end_matches(')')
                        .trim()
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .to_string();
                    if !name.is_empty() {
                        current_trigger = TemplateTrigger::ObsParameter(name);
                    }
                }
            }
            continue;
        }

        // Skip other comments and empty lines
        if trimmed.starts_with(";;") || trimmed.is_empty() {
            continue;
        }

        // This is a rule line — scan for $HOLE_* placeholders
        if trimmed.starts_with('(') {
            let holes = extract_holes_from_rule(trimmed);
            let name = if current_name.is_empty() {
                format!("qor-template-{}", templates.len())
            } else {
                current_name.clone()
            };

            templates.push(Template {
                name,
                rule_text: trimmed.to_string(),
                holes,
                trigger: current_trigger.clone(),
            });
        }
    }

    templates
}

/// Scan a rule text for $HOLE_* placeholders and create typed Hole entries.
fn extract_holes_from_rule(rule_text: &str) -> Vec<Hole> {
    let mut holes = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for word in rule_text.split(|c: char| !c.is_alphanumeric() && c != '$' && c != '_') {
        if word.starts_with("$HOLE_") && seen.insert(word.to_string()) {
            let name_part = word.trim_start_matches("$HOLE_");
            let hole_type = if name_part == "COLOR" || name_part.starts_with("COLOR_") || name_part.ends_with("_COLOR") {
                HoleType::Color
            } else if name_part == "DIR" || name_part.starts_with("DIR_") || name_part.ends_with("_DIR") {
                HoleType::Direction
            } else if name_part == "TRANS" || name_part.starts_with("TRANS_") || name_part.ends_with("_TRANS") {
                HoleType::Transform
            } else if name_part == "PRED" || name_part.starts_with("PRED_") || name_part.ends_with("_PRED") {
                HoleType::Predicate
            } else {
                HoleType::Int
            };
            holes.push(Hole {
                placeholder: word.to_string(),
                hole_type,
            });
        }
    }

    holes
}

/// Build the built-in template library.
pub fn builtin_templates() -> Vec<Template> {
    vec![
        Template {
            name: "color-remap".into(),
            rule_text: "(predict-cell $id $r $c $HOLE_TO) if (grid-cell $id $r $c $HOLE_FROM) <0.99, 0.99>".into(),
            holes: vec![
                Hole { placeholder: "$HOLE_FROM".into(), hole_type: HoleType::Color },
                Hole { placeholder: "$HOLE_TO".into(), hole_type: HoleType::Color },
            ],
            trigger: TemplateTrigger::ObsParameter("color-remap".into()),
        },
        Template {
            name: "shift".into(),
            rule_text: "(predict-cell $id $r $c $v) if (- $r $HOLE_DR $sr) (- $c $HOLE_DC $sc) (grid-cell $id $sr $sc $v) <0.95, 0.95>".into(),
            holes: vec![
                Hole { placeholder: "$HOLE_DR".into(), hole_type: HoleType::Int },
                Hole { placeholder: "$HOLE_DC".into(), hole_type: HoleType::Int },
            ],
            trigger: TemplateTrigger::ObsParameter("shift-dr".into()),
        },
        Template {
            name: "extract-by-color".into(),
            rule_text: "(predict-cell $id $r $c $v) if (grid-obj-cell $id $obj $r $c $v) (grid-object $id $obj $HOLE_COLOR $sz) <0.95, 0.95>".into(),
            holes: vec![
                Hole { placeholder: "$HOLE_COLOR".into(), hole_type: HoleType::Color },
            ],
            trigger: TemplateTrigger::ObsConsistent(vec!["output-smaller".into()]),
        },
        Template {
            name: "fill-enclosed".into(),
            rule_text: "(predict-cell $id $r $c $HOLE_FILL) if (grid-cell $id $r $c 0) (enclosed-region $id $r $c) <0.95, 0.95>".into(),
            holes: vec![
                Hole { placeholder: "$HOLE_FILL".into(), hole_type: HoleType::Color },
            ],
            trigger: TemplateTrigger::ObsConsistent(vec!["has-enclosed-regions".into()]),
        },
        Template {
            name: "identity".into(),
            rule_text: "(predict-cell $id $r $c $v) if (grid-cell $id $r $c $v) <0.50, 0.50>".into(),
            holes: vec![],
            trigger: TemplateTrigger::Always,
        },
    ]
}

/// Extract candidate fill values from observation facts.
///
/// Returns a map: obs-key → list of (placeholder-mapping) values.
/// E.g., "color-remap" → [{"$HOLE_FROM": "3", "$HOLE_TO": "7"}, ...]
pub fn extract_fill_candidates(
    observations: &[Statement],
) -> HashMap<String, Vec<HashMap<String, String>>> {
    let mut candidates: HashMap<String, Vec<HashMap<String, String>>> = HashMap::new();

    for stmt in observations {
        if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = stmt {
            let pred = match parts.first() {
                Some(Neuron::Symbol(s)) => s.as_str(),
                _ => continue,
            };

            match pred {
                "obs-parameter" => {
                    if parts.len() >= 3 {
                        let key = parts[1].to_string();
                        match key.as_str() {
                            "color-remap" if parts.len() >= 4 => {
                                let from = parts[2].to_string();
                                let to = parts[3].to_string();
                                // Builtin template holes
                                let mut fill = HashMap::new();
                                fill.insert("$HOLE_FROM".into(), from.clone());
                                fill.insert("$HOLE_TO".into(), to.clone());
                                candidates.entry("color-remap".into())
                                    .or_default().push(fill);
                                // templates.qor holes (T1, T2, T7)
                                let mut fill2 = HashMap::new();
                                fill2.insert("$HOLE_COLOR_FROM".into(), from.clone());
                                fill2.insert("$HOLE_COLOR_TO".into(), to.clone());
                                candidates.entry("color-remap".into())
                                    .or_default().push(fill2);
                                // Color swap (T7) — bidirectional
                                let mut fill3 = HashMap::new();
                                fill3.insert("$HOLE_COLOR_A".into(), from.clone());
                                fill3.insert("$HOLE_COLOR_B".into(), to.clone());
                                candidates.entry("color-remap".into())
                                    .or_default().push(fill3);
                            }
                            "shift-dr" => {
                                let val = parts[2].to_string();
                                let mut fill = HashMap::new();
                                fill.insert("$HOLE_DR".into(), val);
                                candidates.entry("shift-dr".into())
                                    .or_default().push(fill);
                            }
                            "shift-dc" => {
                                let val = parts[2].to_string();
                                let mut fill = HashMap::new();
                                fill.insert("$HOLE_DC".into(), val);
                                candidates.entry("shift-dc".into())
                                    .or_default().push(fill);
                            }
                            "extract-color" => {
                                let val = parts[2].to_string();
                                let mut fill = HashMap::new();
                                fill.insert("$HOLE_COLOR".into(), val.clone());
                                candidates.entry("extract-color".into())
                                    .or_default().push(fill);
                                // Also map to KEEP_COLOR (T18)
                                let mut fill2 = HashMap::new();
                                fill2.insert("$HOLE_KEEP_COLOR".into(), val);
                                candidates.entry("extract-color".into())
                                    .or_default().push(fill2);
                            }
                            "fill-color" => {
                                let val = parts[2].to_string();
                                let mut fill = HashMap::new();
                                fill.insert("$HOLE_FILL".into(), val.clone());
                                candidates.entry("fill-color".into())
                                    .or_default().push(fill);
                                // Also map to FILL_COLOR (T3 in templates.qor)
                                let mut fill2 = HashMap::new();
                                fill2.insert("$HOLE_FILL_COLOR".into(), val);
                                candidates.entry("fill-color".into())
                                    .or_default().push(fill2);
                            }
                            "scale-factor" => {
                                let val = parts[2].to_string();
                                let mut fill = HashMap::new();
                                fill.insert("$HOLE_SCALE".into(), val.clone());
                                fill.insert("$HOLE_INT".into(), val);
                                candidates.entry("scale-factor".into())
                                    .or_default().push(fill);
                            }
                            "tile-factor" => {
                                let val = parts[2].to_string();
                                let mut fill = HashMap::new();
                                fill.insert("$HOLE_INT".into(), val);
                                candidates.entry("tile-factor".into())
                                    .or_default().push(fill);
                            }
                            "border-color" => {
                                let val = parts[2].to_string();
                                let mut fill = HashMap::new();
                                fill.insert("$HOLE_BORDER_COLOR".into(), val);
                                candidates.entry("border-color".into())
                                    .or_default().push(fill);
                            }
                            _ => {}
                        }
                    }
                }
                "obs-consistent" => {
                    if parts.len() >= 2 {
                        let key = parts[1].to_string();
                        candidates.entry(format!("consistent-{}", key))
                            .or_default(); // just mark it present
                    }
                }
                _ => {}
            }
        }
    }

    candidates
}

/// Try to instantiate a template from observations.
///
/// Returns the filled rule text if all holes can be filled.
pub fn instantiate_template(
    template: &Template,
    candidates: &HashMap<String, Vec<HashMap<String, String>>>,
) -> Vec<FilledTemplate> {
    let mut results = Vec::new();

    // Check trigger
    let triggered = match &template.trigger {
        TemplateTrigger::Always => true,
        TemplateTrigger::ObsConsistent(keys) => {
            keys.iter().all(|k| candidates.contains_key(&format!("consistent-{}", k)))
        }
        TemplateTrigger::ObsParameter(key) => candidates.contains_key(key),
    };

    if !triggered {
        return results;
    }

    if template.holes.is_empty() {
        // No holes — return as-is
        results.push(FilledTemplate {
            name: template.name.clone(),
            rule_text: template.rule_text.clone(),
            fill_values: HashMap::new(),
            score: 0.0,
        });
        return results;
    }

    // Collect fill values from candidates
    let mut all_fills: Vec<HashMap<String, String>> = vec![HashMap::new()];

    for hole in &template.holes {
        let mut new_fills = Vec::new();
        let mut found = false;

        // Search all candidate groups for this hole's placeholder
        for (_key, fill_list) in candidates.iter() {
            for fill_map in fill_list {
                if let Some(val) = fill_map.get(&hole.placeholder) {
                    found = true;
                    for existing in &all_fills {
                        let mut merged = existing.clone();
                        merged.insert(hole.placeholder.clone(), val.clone());
                        new_fills.push(merged);
                    }
                }
            }
        }

        if !found {
            return results; // Can't fill this hole
        }
        all_fills = new_fills;
    }

    // Generate filled templates
    for fill_values in all_fills {
        let mut rule_text = template.rule_text.clone();
        for (placeholder, value) in &fill_values {
            rule_text = rule_text.replace(placeholder, value);
        }
        results.push(FilledTemplate {
            name: template.name.clone(),
            rule_text,
            fill_values,
            score: 0.0,
        });
    }

    results
}

/// Try all templates against observations, return all filled candidates.
pub fn instantiate_all(
    observations: &[Statement],
) -> Vec<FilledTemplate> {
    let templates = builtin_templates();
    let candidates = extract_fill_candidates(observations);
    let mut filled = Vec::new();

    for template in &templates {
        filled.extend(instantiate_template(template, &candidates));
    }

    filled
}

/// Try builtin + extra templates against observations.
/// Extra templates typically come from `load_templates_from_qor()`.
pub fn instantiate_all_with_extra(
    observations: &[Statement],
    extra_templates: &[Template],
) -> Vec<FilledTemplate> {
    let mut all_templates = builtin_templates();
    all_templates.extend(extra_templates.iter().cloned());
    let candidates = extract_fill_candidates(observations);
    let mut filled = Vec::new();

    for template in &all_templates {
        filled.extend(instantiate_template(template, &candidates));
    }

    filled
}

#[cfg(test)]
mod tests {
    use super::*;
    use qor_core::truth_value::TruthValue;

    fn obs_fact(parts: Vec<&str>) -> Statement {
        Statement::Fact {
            neuron: Neuron::Expression(parts.iter().map(|s| Neuron::Symbol(s.to_string())).collect()),
            tv: Some(TruthValue::new(0.95, 0.95)),
            decay: None,
        }
    }

    #[test]
    fn test_extract_color_remap_candidates() {
        let obs = vec![
            Statement::Fact {
                neuron: Neuron::Expression(vec![
                    Neuron::Symbol("obs-parameter".into()),
                    Neuron::Symbol("color-remap".into()),
                    Neuron::Value(qor_core::neuron::QorValue::Int(3)),
                    Neuron::Value(qor_core::neuron::QorValue::Int(7)),
                ]),
                tv: Some(TruthValue::new(0.99, 0.99)),
                decay: None,
            },
        ];
        let cands = extract_fill_candidates(&obs);
        assert!(cands.contains_key("color-remap"));
        let fills = &cands["color-remap"];
        assert!(fills.len() >= 1);
        // Should have both old-style and new-style mappings
        assert!(fills.iter().any(|f| f.contains_key("$HOLE_FROM") && f.contains_key("$HOLE_TO")));
        assert!(fills.iter().any(|f| f.contains_key("$HOLE_COLOR_FROM") && f.contains_key("$HOLE_COLOR_TO")));
    }

    #[test]
    fn test_instantiate_color_remap_template() {
        let obs = vec![
            Statement::Fact {
                neuron: Neuron::Expression(vec![
                    Neuron::Symbol("obs-parameter".into()),
                    Neuron::Symbol("color-remap".into()),
                    Neuron::Value(qor_core::neuron::QorValue::Int(3)),
                    Neuron::Value(qor_core::neuron::QorValue::Int(7)),
                ]),
                tv: Some(TruthValue::new(0.99, 0.99)),
                decay: None,
            },
        ];
        let filled = instantiate_all(&obs);
        // Should get at least the color-remap template + identity
        assert!(filled.len() >= 2);
        let remap = filled.iter().find(|f| f.name == "color-remap").unwrap();
        assert!(remap.rule_text.contains("3"));
        assert!(remap.rule_text.contains("7"));
        assert!(!remap.rule_text.contains("$HOLE"));
    }

    #[test]
    fn test_identity_template_always_present() {
        let obs: Vec<Statement> = vec![];
        let filled = instantiate_all(&obs);
        assert!(filled.iter().any(|f| f.name == "identity"));
    }

    #[test]
    fn test_obs_consistent_trigger() {
        let obs = vec![
            obs_fact(vec!["obs-consistent", "output-smaller"]),
            Statement::Fact {
                neuron: Neuron::Expression(vec![
                    Neuron::Symbol("obs-parameter".into()),
                    Neuron::Symbol("extract-color".into()),
                    Neuron::Value(qor_core::neuron::QorValue::Int(5)),
                ]),
                tv: Some(TruthValue::new(0.95, 0.95)),
                decay: None,
            },
        ];
        let filled = instantiate_all(&obs);
        assert!(filled.iter().any(|f| f.name == "extract-by-color"));
    }
}
