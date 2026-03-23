// ── JSON → QOR Facts ───────────────────────────────────────────────────
//
// Converts JSON objects and arrays into Statement::Fact values.
// - Objects: each key-value → (predicate entity value)
// - Arrays of objects: entity auto-detected from name/id/key fields
// - Nested objects: flattened with hyphenated path prefix
// - Nulls skipped, arrays produce one fact per element

use qor_core::neuron::{Neuron, QorValue, Statement};
use crate::sanitize::{sanitize_symbol, infer_value, make_fact};
use crate::grid::Grid;

use serde_json::Value;

/// Convert a JSON string into QOR facts.
pub fn from_json(json: &str) -> Result<Vec<Statement>, String> {
    let value: Value =
        serde_json::from_str(json).map_err(|e| format!("JSON parse error: {}", e))?;

    let mut facts = Vec::new();

    match &value {
        Value::Object(map) => {
            // Check if this is grid-structured data (has arrays of arrays of ints)
            if has_grid_data(map) {
                grid_object_to_facts(map, "", &mut facts);
            } else {
                let entity = detect_entity(map);
                object_to_facts(map, &entity, "", &mut facts);
            }
        }
        Value::Array(arr) => {
            // Check if top-level is a 2D grid
            if is_grid_value(&Value::Array(arr.clone())) {
                if let Some(grid) = value_to_grid(&Value::Array(arr.clone())) {
                    facts.extend(grid.to_statements("grid"));
                }
            } else {
                for item in arr {
                    if let Value::Object(map) = item {
                        let entity = detect_entity(map);
                        object_to_facts(map, &entity, "", &mut facts);
                    } else {
                        let neuron = json_value_to_neuron(item);
                        facts.push(make_fact(vec![Neuron::symbol("item"), neuron]));
                    }
                }
            }
        }
        other => {
            let neuron = json_value_to_neuron(other);
            facts.push(make_fact(vec![Neuron::symbol("value"), neuron]));
        }
    }

    Ok(facts)
}

/// Check if a JSON value is a 2D grid (array of arrays of numbers).
fn is_grid_value(v: &Value) -> bool {
    if let Value::Array(rows) = v {
        if rows.is_empty() { return false; }
        rows.iter().all(|row| {
            if let Value::Array(cells) = row {
                !cells.is_empty() && cells.iter().all(|c| c.is_number())
            } else {
                false
            }
        })
    } else {
        false
    }
}

/// Convert a 2D JSON array to a Grid.
fn value_to_grid(v: &Value) -> Option<Grid> {
    let rows = v.as_array()?;
    let cells: Vec<Vec<u8>> = rows.iter().map(|row| {
        row.as_array()
            .unwrap_or(&vec![])
            .iter()
            .map(|c| c.as_u64().unwrap_or(0) as u8)
            .collect()
    }).collect();
    Grid::from_vecs(cells).ok()
}

/// Check if a JSON object contains any grid data (2D arrays of numbers).
fn has_grid_data(map: &serde_json::Map<String, Value>) -> bool {
    map.values().any(|v| {
        is_grid_value(v) || match v {
            Value::Array(arr) => arr.iter().any(|item| {
                if let Value::Object(inner) = item {
                    inner.values().any(|iv| is_grid_value(iv))
                } else {
                    false
                }
            }),
            _ => false,
        }
    })
}

/// Recursively convert grid-structured JSON into grid perception facts.
/// Walks the object tree, converting 2D arrays to grid facts with contextual IDs.
fn grid_object_to_facts(
    map: &serde_json::Map<String, Value>,
    prefix: &str,
    facts: &mut Vec<Statement>,
) {
    for (key, value) in map {
        let label = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{}-{}", prefix, key)
        };

        match value {
            v if is_grid_value(v) => {
                // This value is a 2D grid — perceive it
                if let Some(grid) = value_to_grid(v) {
                    facts.extend(grid.to_statements(&label));
                }
            }
            Value::Array(arr) => {
                // Array of objects (like train/test pairs)
                for (i, item) in arr.iter().enumerate() {
                    match item {
                        Value::Object(inner) => {
                            let pair_label = format!("{}-{}", label, i);
                            grid_object_to_facts(inner, &pair_label, facts);
                        }
                        v if is_grid_value(v) => {
                            let grid_label = format!("{}-{}", label, i);
                            if let Some(grid) = value_to_grid(v) {
                                facts.extend(grid.to_statements(&grid_label));
                            }
                        }
                        _ => {}
                    }
                }
            }
            Value::Object(inner) => {
                grid_object_to_facts(inner, &label, facts);
            }
            _ => {
                // Scalar values — emit as regular facts
                let neuron = json_value_to_neuron(value);
                facts.push(make_fact(vec![Neuron::symbol(&label), neuron]));
            }
        }
    }
}

/// Detect entity key from a JSON object (name/id/key/title/label).
fn detect_entity(map: &serde_json::Map<String, Value>) -> Option<String> {
    let candidates = ["name", "id", "key", "title", "label"];
    for c in &candidates {
        if let Some(Value::String(val)) = map.get(*c) {
            return Some(sanitize_symbol(val));
        }
    }
    None
}

/// Convert a JSON object's fields into QOR facts.
fn object_to_facts(
    map: &serde_json::Map<String, Value>,
    entity: &Option<String>,
    prefix: &str,
    facts: &mut Vec<Statement>,
) {
    let entity_key_name = entity.as_ref().and_then(|_| {
        let candidates = ["name", "id", "key", "title", "label"];
        for c in &candidates {
            if let Some(Value::String(_)) = map.get(*c) {
                return Some(c.to_string());
            }
        }
        None
    });

    for (key, value) in map {
        // Skip the entity key itself
        if let Some(ref ek) = entity_key_name {
            if key == ek {
                continue;
            }
        }

        let predicate = if prefix.is_empty() {
            sanitize_symbol(key)
        } else {
            format!("{}-{}", prefix, sanitize_symbol(key))
        };

        match value {
            Value::Null => {} // skip nulls
            Value::Object(nested) => {
                // Flatten nested objects with path prefix
                object_to_facts(nested, entity, &predicate, facts);
            }
            Value::Array(arr) => {
                // One fact per array element
                for item in arr {
                    match item {
                        Value::Object(nested) => {
                            object_to_facts(nested, entity, &predicate, facts);
                        }
                        _ => {
                            let val_neuron = json_value_to_neuron(item);
                            let mut parts = vec![Neuron::symbol(&predicate)];
                            if let Some(ref ent) = entity {
                                parts.push(Neuron::symbol(ent));
                            }
                            parts.push(val_neuron);
                            facts.push(make_fact(parts));
                        }
                    }
                }
            }
            _ => {
                let val_neuron = json_value_to_neuron(value);
                let mut parts = vec![Neuron::symbol(&predicate)];
                if let Some(ref ent) = entity {
                    parts.push(Neuron::symbol(ent));
                }
                parts.push(val_neuron);
                facts.push(make_fact(parts));
            }
        }
    }
}

/// Convert a serde_json Value to a Neuron.
fn json_value_to_neuron(value: &Value) -> Neuron {
    match value {
        Value::String(s) => infer_value(s),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Neuron::Value(QorValue::Int(i))
            } else if let Some(f) = n.as_f64() {
                Neuron::Value(QorValue::Float(f))
            } else {
                Neuron::Value(QorValue::Str(n.to_string()))
            }
        }
        Value::Bool(b) => Neuron::Value(QorValue::Bool(*b)),
        Value::Null => Neuron::symbol("null"),
        _ => Neuron::Value(QorValue::Str(value.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_simple_object() {
        let facts = from_json(r#"{"name": "alice", "age": 30}"#).unwrap();
        // "name" is entity key → skipped. "age" → (age alice 30)
        assert_eq!(facts.len(), 1);
        let s = fact_neuron_str(&facts[0]);
        assert!(s.contains("age"));
        assert!(s.contains("alice"));
        assert!(s.contains("30"));
    }

    #[test]
    fn test_json_object_no_entity() {
        let facts = from_json(r#"{"color": "red", "size": 42}"#).unwrap();
        assert_eq!(facts.len(), 2);
        // No entity key → (color red), (size 42)
        let s0 = fact_neuron_str(&facts[0]);
        let s1 = fact_neuron_str(&facts[1]);
        let both = format!("{} {}", s0, s1);
        assert!(both.contains("color"));
        assert!(both.contains("size"));
    }

    #[test]
    fn test_json_array_of_objects() {
        let facts = from_json(
            r#"[
            {"name": "alice", "age": 30},
            {"name": "bob", "age": 25}
        ]"#,
        )
        .unwrap();
        assert_eq!(facts.len(), 2);
        let s0 = fact_neuron_str(&facts[0]);
        let s1 = fact_neuron_str(&facts[1]);
        assert!(s0.contains("alice"));
        assert!(s1.contains("bob"));
    }

    #[test]
    fn test_json_nested_flatten() {
        let facts =
            from_json(r#"{"name": "alice", "address": {"city": "nyc", "zip": "10001"}}"#)
                .unwrap();
        // (address-city alice nyc), (address-zip alice 10001)
        assert_eq!(facts.len(), 2);
        let all: String = facts.iter().map(|f| fact_neuron_str(f)).collect::<Vec<_>>().join(" ");
        assert!(all.contains("address-city"));
        assert!(all.contains("address-zip"));
    }

    #[test]
    fn test_json_null_skipped() {
        let facts = from_json(r#"{"name": "alice", "middle": null, "age": 30}"#).unwrap();
        // null field skipped → only age
        assert_eq!(facts.len(), 1);
    }

    #[test]
    fn test_json_array_values() {
        let facts =
            from_json(r#"{"name": "alice", "hobbies": ["chess", "music"]}"#).unwrap();
        assert_eq!(facts.len(), 2);
        let all: String = facts.iter().map(|f| fact_neuron_str(f)).collect::<Vec<_>>().join(" ");
        assert!(all.contains("chess"));
        assert!(all.contains("music"));
    }

    #[test]
    fn test_json_boolean_values() {
        let facts = from_json(r#"{"active": true, "deleted": false}"#).unwrap();
        assert_eq!(facts.len(), 2);
    }

    #[test]
    fn test_json_scalar_array() {
        let facts = from_json(r#"["apple", "banana", "cherry"]"#).unwrap();
        assert_eq!(facts.len(), 3);
        let s0 = fact_neuron_str(&facts[0]);
        assert!(s0.contains("item"));
    }

    #[test]
    fn test_json_invalid() {
        assert!(from_json("not json at all").is_err());
    }

    fn fact_neuron_str(stmt: &Statement) -> String {
        match stmt {
            Statement::Fact { neuron, .. } => neuron.to_string(),
            _ => String::new(),
        }
    }
}
