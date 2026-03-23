// ── Dharmic Data Ingestion — Sacred texts to binary triples ────────────
//
// Parses DharmicData JSON files (Vedas, Epics, Gita, etc.) into
// binary knowledge triples using the shared IdEncoder.
//
// Supported sources:
//   - Rigveda       (mandala → sukta → text)
//   - Atharvaveda   (kaanda → sukta → text)
//   - Yajurveda     (adhyaya → text)
//   - Bhagavad Gita (chapter → verse → text + commentaries)
//   - Mahabharata   (book → chapter → shloka → text)
//   - Valmiki Ramayana (kaanda → sarg → shloka → text)
//   - Ramcharitmanas   (kaand → verse_type → content)
//
// Rust is PLUMBING only — reads JSON, writes triples. No domain logic.

use std::io::BufWriter;
use std::path::Path;

use crate::kb_build::{IdEncoder, SourceStats, write_triple};

/// Run the full Dharmic data pipeline.
///
/// Scans `input_dir` for known subdirectories (Rigveda, Mahabharata, etc.),
/// parses each, and writes `dharmic_triples.bin` to `output_dir`.
pub fn run_dharmic_pipeline(
    input_dir: &Path,
    output_dir: &Path,
    encoder: &mut IdEncoder,
) -> Result<SourceStats, String> {
    let triple_path = output_dir.join("dharmic_triples.bin");
    let file = std::fs::File::create(&triple_path)
        .map_err(|e| format!("cannot create {}: {}", triple_path.display(), e))?;
    let mut writer = BufWriter::new(file);
    let mut triple_count: u64 = 0;
    let mut verse_count: u64 = 0;

    // Pre-register common predicates
    let p_instance_of = encoder.predicate_id("instance_of");
    let p_source = encoder.predicate_id("source");
    let p_division = encoder.predicate_id("division");
    let p_subdivision = encoder.predicate_id("subdivision");
    let p_verse_num = encoder.predicate_id("verse_num");
    let p_text = encoder.predicate_id("text");
    let p_label = encoder.predicate_id("label");
    let p_verse_type = encoder.predicate_id("verse_type");
    let p_commentary = encoder.predicate_id("commentary");
    let p_translation = encoder.predicate_id("translation");
    let p_commentator = encoder.predicate_id("commentator");
    let p_translator = encoder.predicate_id("translator");

    // Entity types
    let e_verse = encoder.entity_id("verse");
    let e_mantra = encoder.entity_id("mantra");
    let e_shloka = encoder.entity_id("shloka");
    let e_chaupai = encoder.entity_id("chaupai");
    let e_doha = encoder.entity_id("doha");
    let e_commentary_type = encoder.entity_id("commentary");
    let e_translation_type = encoder.entity_id("translation");

    // Source entities
    let src_rigveda = encoder.entity_id("rigveda");
    let src_atharvaveda = encoder.entity_id("atharvaveda");
    let src_yajurveda = encoder.entity_id("yajurveda");
    let src_gita = encoder.entity_id("bhagavad-gita");
    let src_mahabharata = encoder.entity_id("mahabharata");
    let src_ramayana = encoder.entity_id("valmiki-ramayana");
    let src_ramcharitmanas = encoder.entity_id("ramcharitmanas");

    // Helper closure to write a triple
    macro_rules! triple {
        ($s:expr, $p:expr, $o:expr) => {{
            write_triple(&mut writer, $s, $p, $o, 0.99, 0.99)
                .map_err(|e| format!("write error: {}", e))?;
            triple_count += 1;
        }};
    }

    // ── Rigveda ──────────────────────────────────────────────────────
    let rv_dir = find_subdir(input_dir, "Rigveda");
    if let Some(dir) = rv_dir {
        eprintln!("  Parsing Rigveda...");
        for entry in sorted_json_files(&dir) {
            let data = read_json_array(&entry)?;
            for obj in &data {
                let mandala = obj["mandala"].as_i64().unwrap_or(0);
                let sukta = obj["sukta"].as_i64().unwrap_or(0);
                let text = obj["text"].as_str().unwrap_or("");
                if text.is_empty() { continue; }

                let vid = format!("rv-{}-{}", mandala, sukta);
                let v = encoder.entity_id(&vid);
                let div = encoder.entity_id(&format!("mandala-{}", mandala));
                let sub = encoder.entity_id(&format!("sukta-{}", sukta));
                let txt = encoder.entity_id(text);

                triple!(v, p_instance_of, e_mantra);
                triple!(v, p_source, src_rigveda);
                triple!(v, p_division, div);
                triple!(v, p_subdivision, sub);
                triple!(v, p_text, txt);
                verse_count += 1;
            }
        }
        eprintln!("    {} Rigveda verses", verse_count);
    }

    // ── Atharvaveda ──────────────────────────────────────────────────
    let av_dir = find_subdir(input_dir, "AtharvaVeda");
    if let Some(dir) = av_dir {
        let before = verse_count;
        eprintln!("  Parsing Atharvaveda...");
        for entry in sorted_json_files(&dir) {
            let data = read_json_array(&entry)?;
            for obj in &data {
                let kaanda = obj["kaanda"].as_i64().unwrap_or(0);
                let sukta = obj["sukta"].as_i64().unwrap_or(0);
                let text = obj["text"].as_str().unwrap_or("");
                if text.is_empty() { continue; }

                let vid = format!("av-{}-{}", kaanda, sukta);
                let v = encoder.entity_id(&vid);
                let div = encoder.entity_id(&format!("kaanda-{}", kaanda));
                let sub = encoder.entity_id(&format!("sukta-{}", sukta));
                let txt = encoder.entity_id(text);

                triple!(v, p_instance_of, e_mantra);
                triple!(v, p_source, src_atharvaveda);
                triple!(v, p_division, div);
                triple!(v, p_subdivision, sub);
                triple!(v, p_text, txt);
                verse_count += 1;
            }
        }
        eprintln!("    {} Atharvaveda verses", verse_count - before);
    }

    // ── Yajurveda ────────────────────────────────────────────────────
    let yv_dir = find_subdir(input_dir, "Yajurveda");
    if let Some(dir) = yv_dir {
        let before = verse_count;
        eprintln!("  Parsing Yajurveda...");
        for entry in sorted_json_files(&dir) {
            let data = read_json_array(&entry)?;
            for obj in &data {
                let adhyaya = obj["adhyaya"].as_i64().unwrap_or(0);
                let samhita = obj["samhita"].as_str().unwrap_or("yajurveda");
                let text = obj["text"].as_str().unwrap_or("");
                if text.is_empty() { continue; }

                let vid = format!("yv-{}-{}", samhita, adhyaya);
                let v = encoder.entity_id(&vid);
                let div = encoder.entity_id(&format!("adhyaya-{}", adhyaya));
                let sam = encoder.entity_id(samhita);
                let txt = encoder.entity_id(text);

                triple!(v, p_instance_of, e_mantra);
                triple!(v, p_source, src_yajurveda);
                triple!(v, p_division, div);
                triple!(v, p_label, sam);
                triple!(v, p_text, txt);
                verse_count += 1;
            }
        }
        eprintln!("    {} Yajurveda verses", verse_count - before);
    }

    // ── Bhagavad Gita ────────────────────────────────────────────────
    let gita_dir = find_subdir(input_dir, "SrimadBhagvadGita");
    if let Some(dir) = gita_dir {
        let before = verse_count;
        eprintln!("  Parsing Bhagavad Gita...");
        for entry in sorted_json_files(&dir) {
            let raw = std::fs::read_to_string(&entry)
                .map_err(|e| format!("cannot read {}: {}", entry.display(), e))?;
            let parsed: serde_json::Value = serde_json::from_str(&raw)
                .map_err(|e| format!("JSON error in {}: {}", entry.display(), e))?;

            // Structure: { "BhagavadGitaChapter": [...] }
            let verses = parsed.get("BhagavadGitaChapter")
                .and_then(|v| v.as_array())
                .or_else(|| parsed.as_array());

            let Some(verses) = verses else { continue };

            for obj in verses {
                let chapter = obj["chapter"].as_i64().unwrap_or(0);
                let verse_n = obj["verse"].as_i64().unwrap_or(0);
                let text = obj["text"].as_str().unwrap_or("");
                if text.is_empty() { continue; }

                let vid = format!("gita-{}-{}", chapter, verse_n);
                let v = encoder.entity_id(&vid);
                let div = encoder.entity_id(&format!("chapter-{}", chapter));
                let vn = encoder.entity_id(&verse_n.to_string());
                let txt = encoder.entity_id(text);

                triple!(v, p_instance_of, e_shloka);
                triple!(v, p_source, src_gita);
                triple!(v, p_division, div);
                triple!(v, p_verse_num, vn);
                triple!(v, p_text, txt);
                verse_count += 1;

                // Commentaries
                if let Some(comms) = obj.get("commentaries").and_then(|c| c.as_object()) {
                    for (commentator, comm_text) in comms {
                        let ct = comm_text.as_str().unwrap_or("");
                        if ct.is_empty() || ct.len() < 5 { continue; }

                        let cid = format!("gita-{}-{}-comm-{}", chapter, verse_n,
                            &commentator.to_lowercase().replace(' ', "-")[..commentator.len().min(20)]);
                        let c = encoder.entity_id(&cid);
                        let commenter = encoder.entity_id(commentator);
                        let ct_ent = encoder.entity_id(ct);

                        triple!(c, p_instance_of, e_commentary_type);
                        triple!(c, p_commentator, commenter);
                        triple!(c, p_commentary, ct_ent);
                        triple!(c, p_source, v); // links commentary to verse
                        triple_count += 0; // already counted above
                    }
                }

                // Translations
                if let Some(trans) = obj.get("translations").and_then(|t| t.as_object()) {
                    for (translator, tr_text) in trans {
                        let tt = tr_text.as_str().unwrap_or("");
                        if tt.is_empty() || tt.len() < 5 { continue; }

                        let tid = format!("gita-{}-{}-tr-{}", chapter, verse_n,
                            &translator.to_lowercase().replace(' ', "-")[..translator.len().min(20)]);
                        let t = encoder.entity_id(&tid);
                        let tr_ent = encoder.entity_id(translator);
                        let tt_ent = encoder.entity_id(tt);

                        triple!(t, p_instance_of, e_translation_type);
                        triple!(t, p_translator, tr_ent);
                        triple!(t, p_translation, tt_ent);
                        triple!(t, p_source, v); // links translation to verse
                    }
                }
            }
        }
        eprintln!("    {} Bhagavad Gita verses", verse_count - before);
    }

    // ── Mahabharata ──────────────────────────────────────────────────
    let mb_dir = find_subdir(input_dir, "Mahabharata");
    if let Some(dir) = mb_dir {
        let before = verse_count;
        eprintln!("  Parsing Mahabharata...");
        // Only process root-level book files (mahabharata_book_*.json)
        // Skip Critical Edition subdirectory to avoid duplicates
        for entry in sorted_json_files(&dir) {
            let fname = entry.file_name()
                .and_then(|f| f.to_str())
                .unwrap_or("");
            if !fname.starts_with("mahabharata_book_") { continue; }

            let data = read_json_array(&entry)?;
            for obj in &data {
                let book = obj["book"].as_i64().unwrap_or(0);
                let chapter = obj["chapter"].as_i64().unwrap_or(0);
                let shloka_n = obj["shloka"].as_i64().unwrap_or(0);
                let text = obj["text"].as_str().unwrap_or("");
                if text.is_empty() { continue; }

                let vid = format!("mb-{}-{}-{}", book, chapter, shloka_n);
                let v = encoder.entity_id(&vid);
                let div = encoder.entity_id(&format!("book-{}", book));
                let sub = encoder.entity_id(&format!("chapter-{}", chapter));
                let sn = encoder.entity_id(&shloka_n.to_string());
                let txt = encoder.entity_id(text);

                triple!(v, p_instance_of, e_shloka);
                triple!(v, p_source, src_mahabharata);
                triple!(v, p_division, div);
                triple!(v, p_subdivision, sub);
                triple!(v, p_verse_num, sn);
                triple!(v, p_text, txt);
                verse_count += 1;
            }
        }
        eprintln!("    {} Mahabharata verses", verse_count - before);
    }

    // ── Valmiki Ramayana ─────────────────────────────────────────────
    let vr_dir = find_subdir(input_dir, "ValmikiRamayana");
    if let Some(dir) = vr_dir {
        let before = verse_count;
        eprintln!("  Parsing Valmiki Ramayana...");
        for entry in sorted_json_files(&dir) {
            let data = read_json_array(&entry)?;
            for obj in &data {
                let kaanda = obj["kaanda"].as_str().unwrap_or("unknown");
                let sarg = obj["sarg"].as_i64().unwrap_or(0);
                let shloka_n = obj["shloka"].as_i64().unwrap_or(0);
                let text = obj["text"].as_str().unwrap_or("");
                if text.is_empty() { continue; }

                let vid = format!("vr-{}-{}-{}", kaanda, sarg, shloka_n);
                let v = encoder.entity_id(&vid);
                let div = encoder.entity_id(kaanda);
                let sub = encoder.entity_id(&format!("sarg-{}", sarg));
                let sn = encoder.entity_id(&shloka_n.to_string());
                let txt = encoder.entity_id(text);

                triple!(v, p_instance_of, e_shloka);
                triple!(v, p_source, src_ramayana);
                triple!(v, p_division, div);
                triple!(v, p_subdivision, sub);
                triple!(v, p_verse_num, sn);
                triple!(v, p_text, txt);
                verse_count += 1;
            }
        }
        eprintln!("    {} Valmiki Ramayana verses", verse_count - before);
    }

    // ── Ramcharitmanas ───────────────────────────────────────────────
    let rc_dir = find_subdir(input_dir, "Ramcharitmanas");
    if let Some(dir) = rc_dir {
        let before = verse_count;
        eprintln!("  Parsing Ramcharitmanas...");
        let mut seq = 0u64;
        for entry in sorted_json_files(&dir) {
            let data = read_json_array(&entry)?;
            for obj in &data {
                let vtype = obj["type"].as_str().unwrap_or("verse");
                let content = obj["content"].as_str().unwrap_or("");
                let kaand = obj["kaand"].as_str().unwrap_or("unknown");
                if content.is_empty() { continue; }

                seq += 1;
                let vid = format!("rc-{}", seq);
                let v = encoder.entity_id(&vid);
                let div = encoder.entity_id(kaand);
                let txt = encoder.entity_id(content);

                // Map verse type to entity
                let vtype_ent = match vtype {
                    t if t.contains("श्लोक") => e_shloka,
                    t if t.contains("चौपाई") => e_chaupai,
                    t if t.contains("दोहा") || t.contains("सोरठा") => e_doha,
                    _ => e_verse,
                };

                triple!(v, p_instance_of, vtype_ent);
                triple!(v, p_source, src_ramcharitmanas);
                triple!(v, p_division, div);
                triple!(v, p_verse_type, encoder.entity_id(vtype));
                triple!(v, p_text, txt);
                verse_count += 1;
            }
        }
        eprintln!("    {} Ramcharitmanas verses", verse_count - before);
    }

    // Flush
    use std::io::Write;
    writer.flush().map_err(|e| format!("flush error: {}", e))?;

    let file_size = std::fs::metadata(&triple_path)
        .map(|m| m.len())
        .unwrap_or(0);

    eprintln!();
    eprintln!("  === Dharmic Data Results ===");
    eprintln!("  Verses:     {}", verse_count);
    eprintln!("  Triples:    {} ({:.1} MB)", triple_count, file_size as f64 / 1_048_576.0);

    Ok(SourceStats {
        name: "dharmic".to_string(),
        domain: "scripture".to_string(),
        entities: verse_count,
        triples: triple_count,
        formulas: 0,
    })
}

// ═══════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════

/// Find a subdirectory by name (case-insensitive).
fn find_subdir(base: &Path, name: &str) -> Option<std::path::PathBuf> {
    // Direct match
    let direct = base.join(name);
    if direct.is_dir() {
        return Some(direct);
    }
    // Case-insensitive scan
    if let Ok(entries) = std::fs::read_dir(base) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                if let Some(fname) = entry.file_name().to_str() {
                    if fname.eq_ignore_ascii_case(name) {
                        return Some(entry.path());
                    }
                }
            }
        }
    }
    None
}

/// Get sorted list of .json files in a directory (non-recursive).
fn sorted_json_files(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| {
            e.path().extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("json"))
                .unwrap_or(false)
                && e.path().is_file()
        })
        .map(|e| e.path())
        .collect();
    files.sort();
    files
}

/// Read a JSON file as an array of objects.
/// Handles both `[...]` (direct array) and `{"key": [...]}` (wrapped).
fn read_json_array(path: &Path) -> Result<Vec<serde_json::Value>, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
    let parsed: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| format!("JSON error in {}: {}", path.display(), e))?;

    match parsed {
        serde_json::Value::Array(arr) => Ok(arr),
        serde_json::Value::Object(map) => {
            // Try to find the first array value in the object
            for (_key, val) in map {
                if let serde_json::Value::Array(arr) = val {
                    return Ok(arr);
                }
            }
            Ok(Vec::new())
        }
        _ => Ok(Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_subdir_direct() {
        let tmp = std::env::temp_dir();
        // temp dir exists, but "Rigveda" probably doesn't
        assert!(find_subdir(&tmp, "nonexistent_dharmic_test_dir_12345").is_none());
    }

    #[test]
    fn test_sorted_json_files_empty() {
        let tmp = std::env::temp_dir().join("qor_dharmic_test_empty");
        let _ = std::fs::create_dir_all(&tmp);
        let files = sorted_json_files(&tmp);
        assert!(files.is_empty());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_read_json_array_direct() {
        let tmp = std::env::temp_dir().join("qor_dharmic_test.json");
        std::fs::write(&tmp, r#"[{"text": "hello"}, {"text": "world"}]"#).unwrap();
        let arr = read_json_array(&tmp).unwrap();
        assert_eq!(arr.len(), 2);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_read_json_array_wrapped() {
        let tmp = std::env::temp_dir().join("qor_dharmic_wrapped.json");
        std::fs::write(&tmp, r#"{"BhagavadGitaChapter": [{"chapter": 1, "verse": 1}]}"#).unwrap();
        let arr = read_json_array(&tmp).unwrap();
        assert_eq!(arr.len(), 1);
        let _ = std::fs::remove_file(&tmp);
    }
}
