// ── KB Builder — Encode knowledge into binary triple format ────────────
//
// Shared infrastructure for ALL knowledge sources (OEIS, DBpedia, Wikidata, etc.)
// Provides:
//   - IdEncoder: maps strings ↔ integer IDs (one namespace for everything)
//   - write_triple(): writes 12-byte LE triples
//   - save_mappings(): writes entities.tsv + predicates.tsv
//
// Binary triple format (12 bytes):
//   u32 subject | u16 predicate | u32 object | u8 trust | u8 plausibility

use std::collections::HashMap;
use std::io::{self, BufWriter, Write};
use std::path::Path;

// ═══════════════════════════════════════════════════════════════════════
// IdEncoder — shared string ↔ ID mapping for all sources
// ═══════════════════════════════════════════════════════════════════════

/// Maps entity names → u32 IDs and predicate names → u16 IDs.
/// One encoder shared across ALL sources so everything connects in one graph.
pub struct IdEncoder {
    entity_map: HashMap<String, u32>,
    predicate_map: HashMap<String, u16>,
    next_entity: u32,
    next_predicate: u16,
}

impl IdEncoder {
    pub fn new() -> Self {
        IdEncoder {
            entity_map: HashMap::new(),
            predicate_map: HashMap::new(),
            next_entity: 1,    // 0 reserved for "null"
            next_predicate: 1, // 0 reserved for "null"
        }
    }

    /// Get or create a u32 ID for an entity name.
    pub fn entity_id(&mut self, name: &str) -> u32 {
        let lower = name.to_lowercase();
        if let Some(&id) = self.entity_map.get(&lower) {
            return id;
        }
        let id = self.next_entity;
        self.next_entity += 1;
        self.entity_map.insert(lower, id);
        id
    }

    /// Get or create a u16 ID for a predicate name.
    pub fn predicate_id(&mut self, name: &str) -> u16 {
        let lower = name.to_lowercase();
        if let Some(&id) = self.predicate_map.get(&lower) {
            return id;
        }
        let id = self.next_predicate;
        self.next_predicate += 1;
        self.predicate_map.insert(lower, id);
        id
    }

    /// Number of entities encoded so far.
    pub fn entity_count(&self) -> u32 {
        self.next_entity - 1
    }

    /// Number of predicates encoded so far.
    pub fn predicate_count(&self) -> u16 {
        self.next_predicate - 1
    }

    /// Load existing ID↔name mappings from TSV files.
    /// This allows adding new sources without losing existing IDs.
    pub fn load_mappings(dir: &Path) -> io::Result<Self> {
        let mut enc = IdEncoder::new();

        // Load entities.tsv
        let ent_path = dir.join("entities.tsv");
        if ent_path.exists() {
            let text = std::fs::read_to_string(&ent_path)?;
            for line in text.lines().skip(1) {
                // Format: id\tlabel
                let parts: Vec<&str> = line.splitn(2, '\t').collect();
                if parts.len() == 2 {
                    if let Ok(id) = parts[0].parse::<u32>() {
                        enc.entity_map.insert(parts[1].to_string(), id);
                        if id >= enc.next_entity {
                            enc.next_entity = id + 1;
                        }
                    }
                }
            }
        }

        // Load predicates.tsv
        let pred_path = dir.join("predicates.tsv");
        if pred_path.exists() {
            let text = std::fs::read_to_string(&pred_path)?;
            for line in text.lines().skip(1) {
                let parts: Vec<&str> = line.splitn(2, '\t').collect();
                if parts.len() == 2 {
                    if let Ok(id) = parts[0].parse::<u16>() {
                        enc.predicate_map.insert(parts[1].to_string(), id);
                        if id >= enc.next_predicate {
                            enc.next_predicate = id + 1;
                        }
                    }
                }
            }
        }

        Ok(enc)
    }

    /// Save ID↔name mappings to TSV files for decoding.
    /// Creates entities.tsv and predicates.tsv in the given directory.
    pub fn save_mappings(&self, dir: &Path) -> io::Result<()> {
        // entities.tsv: id\tlabel
        let mut ef = BufWriter::new(std::fs::File::create(dir.join("entities.tsv"))?);
        writeln!(ef, "id\tlabel")?;
        let mut entities: Vec<(&String, &u32)> = self.entity_map.iter().collect();
        entities.sort_by_key(|(_, id)| **id);
        for (name, id) in entities {
            writeln!(ef, "{}\t{}", id, name)?;
        }
        ef.flush()?;

        // predicates.tsv: id\tlabel
        let mut pf = BufWriter::new(std::fs::File::create(dir.join("predicates.tsv"))?);
        writeln!(pf, "id\tlabel")?;
        let mut preds: Vec<(&String, &u16)> = self.predicate_map.iter().collect();
        preds.sort_by_key(|(_, id)| **id);
        for (name, id) in preds {
            writeln!(pf, "{}\t{}", id, name)?;
        }
        pf.flush()?;

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Triple writer — 12 bytes per fact
// ═══════════════════════════════════════════════════════════════════════

/// Write a single 12-byte triple in little-endian format.
pub fn write_triple<W: Write>(
    w: &mut W,
    subject: u32,
    predicate: u16,
    object: u32,
    trust: f32,
    plausibility: f32,
) -> io::Result<()> {
    w.write_all(&subject.to_le_bytes())?;
    w.write_all(&predicate.to_le_bytes())?;
    w.write_all(&object.to_le_bytes())?;
    w.write_all(&[(trust.clamp(0.0, 1.0) * 255.0) as u8])?;
    w.write_all(&[(plausibility.clamp(0.0, 1.0) * 255.0) as u8])?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════
// Manifest — build metadata
// ═══════════════════════════════════════════════════════════════════════

/// Write a manifest.json with build metadata.
pub fn write_manifest(
    dir: &Path,
    sources: &[SourceStats],
) -> io::Result<()> {
    let mut f = BufWriter::new(std::fs::File::create(dir.join("manifest.json"))?);
    writeln!(f, "{{")?;
    writeln!(f, "  \"built\": \"{}\",", chrono_now())?;
    writeln!(f, "  \"sources\": [")?;
    for (i, src) in sources.iter().enumerate() {
        let comma = if i + 1 < sources.len() { "," } else { "" };
        writeln!(f, "    {{")?;
        writeln!(f, "      \"name\": \"{}\",", src.name)?;
        writeln!(f, "      \"domain\": \"{}\",", src.domain)?;
        writeln!(f, "      \"entities\": {},", src.entities)?;
        writeln!(f, "      \"triples\": {},", src.triples)?;
        writeln!(f, "      \"formulas\": {}", src.formulas)?;
        writeln!(f, "    }}{}", comma)?;
    }
    writeln!(f, "  ]")?;
    writeln!(f, "}}")?;
    f.flush()?;
    Ok(())
}

/// Stats for one knowledge source.
pub struct SourceStats {
    pub name: String,
    pub domain: String,
    pub entities: u64,
    pub triples: u64,
    pub formulas: u64,
}

/// Simple timestamp (no chrono dep — use manual formatting).
fn chrono_now() -> String {
    // We don't have a chrono dependency, so just return a placeholder.
    // The CLI can override this with the actual time.
    "unknown".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_encoder_entity_ids() {
        let mut enc = IdEncoder::new();
        let id1 = enc.entity_id("fibonacci");
        let id2 = enc.entity_id("euler");
        let id3 = enc.entity_id("fibonacci"); // same as id1
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 1); // same entity, same ID
        assert_eq!(enc.entity_count(), 2);
    }

    #[test]
    fn test_encoder_case_insensitive() {
        let mut enc = IdEncoder::new();
        let id1 = enc.entity_id("Fibonacci");
        let id2 = enc.entity_id("fibonacci");
        let id3 = enc.entity_id("FIBONACCI");
        assert_eq!(id1, id2);
        assert_eq!(id2, id3);
    }

    #[test]
    fn test_encoder_predicate_ids() {
        let mut enc = IdEncoder::new();
        let p1 = enc.predicate_id("instance_of");
        let p2 = enc.predicate_id("label");
        let p3 = enc.predicate_id("instance_of");
        assert_eq!(p1, 1);
        assert_eq!(p2, 2);
        assert_eq!(p3, 1);
        assert_eq!(enc.predicate_count(), 2);
    }

    #[test]
    fn test_write_triple() {
        let mut buf = Cursor::new(Vec::new());
        write_triple(&mut buf, 42, 7, 100, 0.95, 0.90).unwrap();
        let bytes = buf.into_inner();
        assert_eq!(bytes.len(), 12);
        // subject = 42 LE
        assert_eq!(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]), 42);
        // predicate = 7 LE
        assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), 7);
        // object = 100 LE
        assert_eq!(u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]), 100);
        // trust ≈ 0.95
        assert_eq!(bytes[10], 242); // (0.95 * 255) as u8 = 242
        // plaus ≈ 0.90
        assert_eq!(bytes[11], 229); // (0.90 * 255) as u8 = 229
    }

    #[test]
    fn test_save_mappings() {
        let dir = std::env::temp_dir().join("qor_kb_test_mappings");
        let _ = std::fs::create_dir_all(&dir);

        let mut enc = IdEncoder::new();
        enc.entity_id("fibonacci");
        enc.entity_id("euler");
        enc.predicate_id("instance_of");
        enc.predicate_id("label");

        enc.save_mappings(&dir).unwrap();

        let ent = std::fs::read_to_string(dir.join("entities.tsv")).unwrap();
        assert!(ent.contains("fibonacci"));
        assert!(ent.contains("euler"));

        let pred = std::fs::read_to_string(dir.join("predicates.tsv")).unwrap();
        assert!(pred.contains("instance_of"));
        assert!(pred.contains("label"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
