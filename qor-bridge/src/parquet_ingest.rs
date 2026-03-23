use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;
use qor_core::neuron::{Neuron, QorValue, Statement};
use std::path::Path;

use crate::sanitize::{ingested_tv, sanitize_symbol};

/// Read a parquet file from disk and convert each row into QOR facts.
/// Each column becomes a predicate: (column-name row-id value)
pub fn from_parquet_file(path: &Path) -> Result<Vec<Statement>, String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("could not open '{}': {}", path.display(), e))?;

    let reader = SerializedFileReader::new(file)
        .map_err(|e| format!("invalid parquet file '{}': {}", path.display(), e))?;

    let metadata = reader.metadata();
    let schema = metadata.file_metadata().schema_descr();

    let columns: Vec<String> = schema
        .columns()
        .iter()
        .map(|c| sanitize_symbol(c.name()))
        .collect();

    let mut facts = Vec::new();
    let iter = reader.get_row_iter(None)
        .map_err(|e| format!("could not read rows: {}", e))?;

    for (row_idx, row_result) in iter.enumerate() {
        let row = row_result.map_err(|e| format!("row read error: {}", e))?;
        let row_id = format!("row-{}", row_idx);

        for (col_idx, (_name, field)) in row.get_column_iter().enumerate() {
            if col_idx >= columns.len() {
                break;
            }
            let predicate = &columns[col_idx];

            let value = match field {
                Field::Null => continue,
                Field::Bool(b) => Neuron::Value(QorValue::Bool(*b)),
                Field::Byte(n) => Neuron::Value(QorValue::Int(*n as i64)),
                Field::Short(n) => Neuron::Value(QorValue::Int(*n as i64)),
                Field::Int(n) => Neuron::Value(QorValue::Int(*n as i64)),
                Field::Long(n) => Neuron::Value(QorValue::Int(*n)),
                Field::UByte(n) => Neuron::Value(QorValue::Int(*n as i64)),
                Field::UShort(n) => Neuron::Value(QorValue::Int(*n as i64)),
                Field::UInt(n) => Neuron::Value(QorValue::Int(*n as i64)),
                Field::ULong(n) => Neuron::Value(QorValue::Int(*n as i64)),
                Field::Float(f) => Neuron::Value(QorValue::Float(*f as f64)),
                Field::Double(f) => Neuron::Value(QorValue::Float(*f)),
                Field::Str(s) => {
                    let sanitized = sanitize_symbol(s);
                    if sanitized.is_empty() || sanitized == "unknown" {
                        Neuron::Value(QorValue::Str(s.to_string()))
                    } else {
                        Neuron::Symbol(sanitized)
                    }
                }
                Field::Bytes(b) => {
                    Neuron::Value(QorValue::Str(String::from_utf8_lossy(b.data()).to_string()))
                }
                Field::TimestampMillis(ms) => Neuron::Value(QorValue::Int(*ms)),
                Field::TimestampMicros(us) => Neuron::Value(QorValue::Int(*us)),
                _ => continue,
            };

            facts.push(Statement::Fact {
                neuron: Neuron::Expression(vec![
                    Neuron::Symbol(predicate.clone()),
                    Neuron::Symbol(row_id.clone()),
                    value,
                ]),
                tv: Some(ingested_tv()),
                decay: None,
            });
        }

        // Cap at 10000 rows to avoid OOM on huge files
        if row_idx >= 9999 {
            break;
        }
    }

    Ok(facts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parquet_real_file() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../Ref/data/2025-03.parquet");
        if !path.exists() {
            return; // skip if file not present
        }

        let facts = from_parquet_file(&path).unwrap();
        assert!(!facts.is_empty());

        // Verify facts are valid statements
        for stmt in &facts {
            match stmt {
                Statement::Fact { neuron, tv, .. } => {
                    assert!(tv.is_some());
                    match neuron {
                        Neuron::Expression(parts) => assert_eq!(parts.len(), 3),
                        _ => panic!("expected expression"),
                    }
                }
                _ => panic!("expected fact"),
            }
        }
    }
}
