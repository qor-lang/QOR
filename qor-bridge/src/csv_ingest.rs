// ── CSV → QOR Facts ────────────────────────────────────────────────────
//
// Converts CSV data with headers into Statement::Fact values.
// First row = headers (predicates), entity column auto-detected.
// Each cell → (header entity value) with type inference.

use qor_core::neuron::{Neuron, Statement};
use crate::sanitize::{sanitize_symbol, infer_value, ingested_tv};

/// Convert CSV data into QOR facts.
///
/// First row is treated as headers (predicates).
/// Entity column is auto-detected ("name", "id") or defaults to column 0.
pub fn from_csv(csv_data: &str) -> Result<Vec<Statement>, String> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .trim(csv::Trim::All)
        .from_reader(csv_data.as_bytes());

    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| format!("CSV header error: {}", e))?
        .iter()
        .map(|h| sanitize_symbol(h))
        .collect();

    if headers.is_empty() {
        return Err("CSV has no headers".to_string());
    }

    let entity_col = detect_entity_column(&headers);
    let mut facts = Vec::new();

    for result in reader.records() {
        let record = result.map_err(|e| format!("CSV row error: {}", e))?;

        let entity = record
            .get(entity_col)
            .map(|s| sanitize_symbol(s.trim()))
            .unwrap_or_else(|| "unknown".to_string());

        for (i, field) in record.iter().enumerate() {
            if i == entity_col || i >= headers.len() {
                continue;
            }

            let field = field.trim();
            if field.is_empty() {
                continue;
            }

            let value = infer_value(field);
            facts.push(Statement::Fact {
                neuron: Neuron::expression(vec![
                    Neuron::symbol(&headers[i]),
                    Neuron::symbol(&entity),
                    value,
                ]),
                tv: Some(ingested_tv()),
                decay: None,
            });
        }
    }

    Ok(facts)
}

/// Detect which column is the entity identifier.
/// Prefers "name", "id", "key"; defaults to column 0.
fn detect_entity_column(headers: &[String]) -> usize {
    let priority = ["name", "id", "key"];
    for candidate in &priority {
        if let Some(pos) = headers.iter().position(|h| h == *candidate) {
            return pos;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_basic() {
        let csv = "name,age,city\nalice,30,nyc\nbob,25,sf";
        let facts = from_csv(csv).unwrap();
        // 2 rows x 2 data columns = 4 facts
        assert_eq!(facts.len(), 4);

        let all = facts_to_strings(&facts);
        assert!(all.contains("(age alice 30)"));
        assert!(all.contains("(city alice nyc)"));
        assert!(all.contains("(age bob 25)"));
        assert!(all.contains("(city bob sf)"));
    }

    #[test]
    fn test_csv_id_column() {
        let csv = "id,score,grade\n101,95,a\n102,87,b";
        let facts = from_csv(csv).unwrap();
        assert_eq!(facts.len(), 4);

        let all = facts_to_strings(&facts);
        assert!(all.contains("x-101")); // id 101 → x-101 (digit prefix)
    }

    #[test]
    fn test_csv_first_column_default() {
        let csv = "animal,legs,wings\nbird,2,2\ndog,4,0";
        let facts = from_csv(csv).unwrap();
        // "animal" column is entity (first col, no name/id)
        assert_eq!(facts.len(), 4);

        let all = facts_to_strings(&facts);
        assert!(all.contains("(legs bird 2)"));
        assert!(all.contains("(wings dog 0)"));
    }

    #[test]
    fn test_csv_empty_cells_skipped() {
        let csv = "name,age,city\nalice,30,\nbob,,sf";
        let facts = from_csv(csv).unwrap();
        // alice: age only (city empty). bob: city only (age empty)
        assert_eq!(facts.len(), 2);
    }

    #[test]
    fn test_csv_type_inference() {
        let csv = "name,score,active\nalice,95.5,true";
        let facts = from_csv(csv).unwrap();
        assert_eq!(facts.len(), 2);

        let all = facts_to_strings(&facts);
        assert!(all.contains("95.5")); // float
        assert!(all.contains("true")); // bool
    }

    #[test]
    fn test_csv_single_row() {
        let csv = "name,value\nalice,42";
        let facts = from_csv(csv).unwrap();
        assert_eq!(facts.len(), 1);
    }

    fn facts_to_strings(facts: &[Statement]) -> String {
        facts
            .iter()
            .map(|f| match f {
                Statement::Fact { neuron, .. } => neuron.to_string(),
                _ => String::new(),
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }
}
