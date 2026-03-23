// ── OEIS Parser — Integer Sequence Knowledge ──────────────────────────
//
// Parses OEIS data into binary triples + QOR formula rules.
//
// Supports two input formats:
//   1. oeisdata repo: seq/A000/A000045.seq files (internal format)
//   2. Downloadable: stripped.gz + names.gz from oeis.org
//
// Output:
//   - oeis_triples.bin: binary facts (names, terms, keywords, cross-refs)
//   - oeis_formulas.qor: mathematical formulas as QOR reasoning rules
//
// Binary format: same 12-byte triple as all other sources.

use regex::Regex;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::kb_build::{write_triple, IdEncoder, SourceStats};

// ═══════════════════════════════════════════════════════════════════════
// Parsed OEIS entry
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Default)]
struct OeisEntry {
    id: String,
    name: String,
    terms: Vec<i64>,
    keywords: Vec<String>,
    author: String,
    references: Vec<String>,
    formulas: Vec<String>,
    offset: i32,
}

// ═══════════════════════════════════════════════════════════════════════
// Parse OEIS internal format (%X AXXXXXX lines)
// ═══════════════════════════════════════════════════════════════════════

fn parse_internal_format(content: &str) -> OeisEntry {
    let mut entry = OeisEntry::default();
    let xref_re = Regex::new(r"A\d{6}").unwrap();

    for line in content.lines() {
        if line.len() < 4 {
            continue;
        }

        let field = &line[0..2];
        // Skip the "%X AXXXXXX " prefix (11 chars) to get content
        let data = if line.len() > 13 {
            line[11..].trim()
        } else {
            continue;
        };

        match field {
            "%I" => {
                if let Some(a) = xref_re.find(line) {
                    entry.id = a.as_str().to_string();
                }
            }
            "%N" => {
                entry.name = data.to_string();
            }
            "%S" | "%T" | "%U" => {
                for term in data.split(',') {
                    let t = term.trim();
                    if !t.is_empty() {
                        if let Ok(n) = t.parse::<i64>() {
                            entry.terms.push(n);
                        }
                    }
                }
            }
            "%F" => {
                if !data.is_empty() {
                    entry.formulas.push(data.to_string());
                }
            }
            "%Y" => {
                for m in xref_re.find_iter(data) {
                    entry.references.push(m.as_str().to_string());
                }
            }
            "%K" => {
                for kw in data.split(',') {
                    let k = kw.trim();
                    if !k.is_empty() {
                        entry.keywords.push(k.to_string());
                    }
                }
            }
            "%A" => {
                let author = data.trim_start_matches('_').trim_end_matches('_');
                entry.author = author.to_string();
            }
            "%O" => {
                if let Ok(o) = data.split(',').next().unwrap_or("0").parse::<i32>() {
                    entry.offset = o;
                }
            }
            _ => {}
        }
    }

    entry
}

// ═══════════════════════════════════════════════════════════════════════
// Encode one OEIS entry into binary triples
// ═══════════════════════════════════════════════════════════════════════

fn encode_entry(
    entry: &OeisEntry,
    encoder: &mut IdEncoder,
    bin_writer: &mut impl Write,
) -> io::Result<u64> {
    let mut triple_count = 0u64;

    if entry.id.is_empty() {
        return Ok(0);
    }

    let subj_id = encoder.entity_id(&entry.id);
    let seq_type_id = encoder.entity_id("integer_sequence");

    // Type assertion: this is an integer sequence
    let pred_instance_of = encoder.predicate_id("instance_of");
    write_triple(bin_writer, subj_id, pred_instance_of, seq_type_id, 0.99, 0.99)?;
    triple_count += 1;

    // Name/label
    if !entry.name.is_empty() {
        let pred_label = encoder.predicate_id("label");
        let name_id = encoder.entity_id(&entry.name);
        write_triple(bin_writer, subj_id, pred_label, name_id, 0.99, 0.99)?;
        triple_count += 1;
    }

    // Terms (first 20)
    for (i, term) in entry.terms.iter().take(20).enumerate() {
        let pred_n = encoder.predicate_id(&format!("term_{}", i));
        let val_id = encoder.entity_id(&term.to_string());
        write_triple(bin_writer, subj_id, pred_n, val_id, 0.99, 0.99)?;
        triple_count += 1;
    }

    // Keywords → domain links
    let pred_keyword = encoder.predicate_id("keyword");
    for kw in &entry.keywords {
        let kw_id = encoder.entity_id(kw);
        write_triple(bin_writer, subj_id, pred_keyword, kw_id, 0.95, 0.95)?;
        triple_count += 1;
    }

    // Cross-references → graph edges
    let pred_xref = encoder.predicate_id("related_to");
    for xref in &entry.references {
        let xref_id = encoder.entity_id(xref);
        write_triple(bin_writer, subj_id, pred_xref, xref_id, 0.95, 0.95)?;
        triple_count += 1;
    }

    // Author
    if !entry.author.is_empty() {
        let pred_author = encoder.predicate_id("author");
        let auth_id = encoder.entity_id(&entry.author);
        write_triple(bin_writer, subj_id, pred_author, auth_id, 0.97, 0.97)?;
        triple_count += 1;
    }

    // Formulas → categorized binary triples (all data lives in the graph)
    for formula in &entry.formulas {
        let pred_name = classify_formula(formula);
        let pred_id = encoder.predicate_id(pred_name);
        let f_id = encoder.entity_id(&sanitize_formula(formula));
        let confidence = formula_confidence(pred_name);
        write_triple(bin_writer, subj_id, pred_id, f_id, confidence, confidence)?;
        triple_count += 1;
    }

    Ok(triple_count)
}

// ═══════════════════════════════════════════════════════════════════════
// Formula classification — returns predicate name for binary graph
// ═══════════════════════════════════════════════════════════════════════

/// Classify a formula string into a predicate name for the binary graph.
fn classify_formula(formula: &str) -> &'static str {
    let formula = formula.trim();

    // Recurrence: a(n) = a(n-1) + a(n-2)
    if formula.contains("a(n)") && formula.contains("a(n-") {
        return "formula_recurrence";
    }
    // Closed form: a(n) = n^2 (no recursion, no Sum)
    if formula.contains("a(n) =") && !formula.contains("a(n-") && !formula.contains("Sum") {
        return "formula_closed";
    }
    // Generating function
    if formula.starts_with("G.f.") || formula.starts_with("E.g.f.") {
        return "formula_gf";
    }
    // Summation / product
    if formula.contains("Sum_") || formula.contains("sum_") || formula.contains("Product_") {
        return "formula_sum";
    }
    // Congruence / modular
    if formula.contains(" mod ") || formula.contains("equiv") {
        return "formula_mod";
    }
    // Bounds / asymptotics
    if formula.contains("<=") || formula.contains(">=") || formula.contains(" ~ ") {
        return "formula_bound";
    }
    // General formula
    "has_formula"
}

/// Return confidence for a formula category.
fn formula_confidence(pred_name: &str) -> f32 {
    match pred_name {
        "formula_closed" => 0.97,
        "formula_recurrence" | "formula_gf" => 0.95,
        "formula_sum" => 0.93,
        "formula_mod" => 0.90,
        "formula_bound" => 0.85,
        _ => 0.90,
    }
}

fn sanitize_formula(s: &str) -> String {
    s.replace('"', "'").replace('\\', "\\\\").trim().to_string()
}

// ═══════════════════════════════════════════════════════════════════════
// Parse oeisdata repo: seq/A000/A000045.seq files
// ═══════════════════════════════════════════════════════════════════════

fn parse_seq_dir(
    base_dir: &Path,
    encoder: &mut IdEncoder,
    bin_writer: &mut impl Write,
) -> io::Result<(u64, u64)> {
    let seq_dir = base_dir.join("seq");
    if !seq_dir.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("seq/ directory not found in {}", base_dir.display()),
        ));
    }

    let mut seq_count = 0u64;
    let mut triple_count = 0u64;

    // Walk seq/AXXX/ subdirectories
    let mut subdirs: Vec<_> = std::fs::read_dir(&seq_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();
    subdirs.sort_by_key(|e| e.path());

    for subdir in subdirs {
        let mut files: Vec<_> = std::fs::read_dir(subdir.path())?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map_or(false, |ext| ext == "seq")
            })
            .collect();
        files.sort_by_key(|e| e.path());

        for file_entry in files {
            let path = file_entry.path();
            match std::fs::read_to_string(&path) {
                Ok(content) => {
                    let entry = parse_internal_format(&content);
                    let t = encode_entry(&entry, encoder, bin_writer)?;
                    triple_count += t;
                    seq_count += 1;

                    if seq_count % 50000 == 0 {
                        eprintln!("  {} sequences processed...", seq_count);
                    }
                }
                Err(e) => {
                    eprintln!("  warning: could not read {}: {}", path.display(), e);
                }
            }
        }
    }

    Ok((seq_count, triple_count))
}

// ═══════════════════════════════════════════════════════════════════════
// Parse stripped.gz — sequence terms only
// ═══════════════════════════════════════════════════════════════════════

fn parse_stripped_gz(
    path: &Path,
    encoder: &mut IdEncoder,
    bin_writer: &mut impl Write,
) -> io::Result<(u64, u64)> {
    let file = std::fs::File::open(path)?;
    let reader: Box<dyn Read> = if path.extension().map_or(false, |e| e == "gz") {
        Box::new(flate2::read::GzDecoder::new(file))
    } else {
        Box::new(file)
    };

    let buf = BufReader::new(reader);
    let mut seq_count = 0u64;
    let mut triple_count = 0u64;

    for line in buf.lines() {
        let line = line?;
        if line.starts_with('#') || line.is_empty() {
            continue;
        }

        // Format: A000045 ,0,1,1,2,3,5,8,13,21,34,...
        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        if parts.len() != 2 {
            continue;
        }

        let a_number = parts[0].trim();
        if !a_number.starts_with('A') {
            continue;
        }

        let terms_str = parts[1].trim().trim_start_matches(',').trim_end_matches(',');
        let subj_id = encoder.entity_id(a_number);

        // Store individual terms (first 20)
        let terms: Vec<&str> = terms_str.split(',').collect();
        for (i, term) in terms.iter().take(20).enumerate() {
            let term = term.trim();
            if term.is_empty() {
                continue;
            }
            let pred_n = encoder.predicate_id(&format!("term_{}", i));
            let val_id = encoder.entity_id(term);
            write_triple(bin_writer, subj_id, pred_n, val_id, 0.99, 0.99)?;
            triple_count += 1;
        }

        // Type assertion
        let pred_instance_of = encoder.predicate_id("instance_of");
        let seq_type_id = encoder.entity_id("integer_sequence");
        write_triple(bin_writer, subj_id, pred_instance_of, seq_type_id, 0.99, 0.99)?;
        triple_count += 1;

        seq_count += 1;
    }

    Ok((seq_count, triple_count))
}

// ═══════════════════════════════════════════════════════════════════════
// Parse names.gz — sequence names
// ═══════════════════════════════════════════════════════════════════════

fn parse_names_gz(
    path: &Path,
    encoder: &mut IdEncoder,
    bin_writer: &mut impl Write,
) -> io::Result<u64> {
    let file = std::fs::File::open(path)?;
    let reader: Box<dyn Read> = if path.extension().map_or(false, |e| e == "gz") {
        Box::new(flate2::read::GzDecoder::new(file))
    } else {
        Box::new(file)
    };

    let buf = BufReader::new(reader);
    let pred_label = encoder.predicate_id("label");
    let mut count = 0u64;

    for line in buf.lines() {
        let line = line?;
        if line.starts_with('#') || line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        if parts.len() != 2 {
            continue;
        }

        let a_number = parts[0].trim();
        let name = parts[1].trim();

        if !a_number.starts_with('A') {
            continue;
        }

        let subj_id = encoder.entity_id(a_number);
        let name_id = encoder.entity_id(name);
        write_triple(bin_writer, subj_id, pred_label, name_id, 0.99, 0.99)?;
        count += 1;
    }

    Ok(count)
}

// ═══════════════════════════════════════════════════════════════════════
// Main pipeline — run OEIS ingestion
// ═══════════════════════════════════════════════════════════════════════

/// Run the OEIS knowledge pipeline.
///
/// `input_dir` can be either:
///   - An oeisdata git repo (has seq/ subdirectory)
///   - A directory with stripped.gz + names.gz files
///
/// All data goes into `oeis_triples.bin` (binary graph).
/// Reasoning rules that use this data live in `meta/math.qor`.
pub fn run_oeis_pipeline(
    input_dir: &Path,
    output_dir: &Path,
    encoder: &mut IdEncoder,
) -> io::Result<SourceStats> {
    std::fs::create_dir_all(output_dir)?;

    let triples_path = output_dir.join("oeis_triples.bin");

    let bin_file = std::fs::File::create(&triples_path)?;
    let mut bin_writer = BufWriter::with_capacity(4 * 1024 * 1024, bin_file);

    let mut total_seqs = 0u64;
    let mut total_triples = 0u64;

    // Strategy 1: oeisdata repo (has seq/ directory)
    if input_dir.join("seq").exists() {
        eprintln!("  Parsing oeisdata repo: {}", input_dir.display());
        let (s, t) = parse_seq_dir(input_dir, encoder, &mut bin_writer)?;
        total_seqs = s;
        total_triples = t;
    } else {
        // Strategy 2: stripped.gz + names.gz files
        let stripped = input_dir.join("stripped.gz");
        let stripped_plain = input_dir.join("stripped");
        if stripped.exists() {
            eprintln!("  Parsing stripped.gz...");
            let (s, t) = parse_stripped_gz(&stripped, encoder, &mut bin_writer)?;
            total_seqs += s;
            total_triples += t;
        } else if stripped_plain.exists() {
            eprintln!("  Parsing stripped...");
            let (s, t) = parse_stripped_gz(&stripped_plain, encoder, &mut bin_writer)?;
            total_seqs += s;
            total_triples += t;
        }

        let names = input_dir.join("names.gz");
        let names_plain = input_dir.join("names");
        if names.exists() {
            eprintln!("  Parsing names.gz...");
            let n = parse_names_gz(&names, encoder, &mut bin_writer)?;
            total_triples += n;
        } else if names_plain.exists() {
            eprintln!("  Parsing names...");
            let n = parse_names_gz(&names_plain, encoder, &mut bin_writer)?;
            total_triples += n;
        }
    }

    bin_writer.flush()?;

    let bin_size = std::fs::metadata(&triples_path)
        .map(|m| m.len())
        .unwrap_or(0);

    eprintln!();
    eprintln!("  === OEIS Results ===");
    eprintln!("  Sequences: {}", total_seqs);
    eprintln!(
        "  Triples:   {} ({:.1} MB)",
        total_triples,
        bin_size as f64 / (1024.0 * 1024.0)
    );

    Ok(SourceStats {
        name: "oeis".to_string(),
        domain: "math".to_string(),
        entities: total_seqs,
        triples: total_triples,
        formulas: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_parse_internal_format() {
        let content = r#"%I A000045 M0692 N0256
%S A000045 0,1,1,2,3,5,8,13,21,34,55,89,144,233
%N A000045 Fibonacci numbers: F(n) = F(n-1) + F(n-2) with F(0) = 0 and F(1) = 1.
%F A000045 a(n) = a(n-1) + a(n-2) with a(0) = 0, a(1) = 1.
%F A000045 G.f.: x/(1-x-x^2).
%Y A000045 Cf. A000032, A001622, A005478.
%K A000045 core,nonn,nice,easy,hear,changed
%A A000045 _N. J. A. Sloane_
%O A000045 0,4
"#;
        let entry = parse_internal_format(content);
        assert_eq!(entry.id, "A000045");
        assert_eq!(entry.terms.len(), 14);
        assert_eq!(entry.terms[0], 0);
        assert_eq!(entry.terms[5], 5);
        assert!(entry.name.contains("Fibonacci"));
        assert_eq!(entry.formulas.len(), 2);
        assert_eq!(entry.references.len(), 3);
        assert!(entry.keywords.contains(&"core".to_string()));
        assert!(entry.keywords.contains(&"nonn".to_string()));
        assert!(entry.author.contains("Sloane"));
        assert_eq!(entry.offset, 0);
    }

    #[test]
    fn test_encode_entry() {
        let entry = OeisEntry {
            id: "A000045".to_string(),
            name: "Fibonacci numbers".to_string(),
            terms: vec![0, 1, 1, 2, 3, 5],
            keywords: vec!["core".to_string(), "nonn".to_string()],
            author: "Sloane".to_string(),
            references: vec!["A000032".to_string()],
            formulas: vec!["a(n) = a(n-1) + a(n-2)".to_string()],
            offset: 0,
        };

        let mut encoder = IdEncoder::new();
        let mut bin_buf = Cursor::new(Vec::new());

        let triples = encode_entry(&entry, &mut encoder, &mut bin_buf).unwrap();

        // 1 instance_of + 1 label + 6 terms + 2 keywords + 1 xref + 1 author + 1 formula = 13
        assert_eq!(triples, 13);

        // Binary: 13 triples * 12 bytes = 156 bytes
        assert_eq!(bin_buf.into_inner().len(), 156);
    }

    #[test]
    fn test_classify_formula_recurrence() {
        assert_eq!(classify_formula("a(n) = a(n-1) + a(n-2)"), "formula_recurrence");
    }

    #[test]
    fn test_classify_formula_closed() {
        assert_eq!(classify_formula("a(n) = n^2"), "formula_closed");
    }

    #[test]
    fn test_classify_formula_gf() {
        assert_eq!(classify_formula("G.f.: x/(1-x-x^2)"), "formula_gf");
    }

    #[test]
    fn test_classify_formula_summation() {
        assert_eq!(classify_formula("a(n) = Sum_{k=0..n} binomial(2n,k)/(n+1)"), "formula_sum");
    }

    #[test]
    fn test_classify_formula_general() {
        assert_eq!(classify_formula("some general formula text here"), "has_formula");
    }

    #[test]
    fn test_parse_stripped_format() {
        // Create a temp file with stripped format
        let dir = std::env::temp_dir().join("qor_oeis_test_stripped");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("stripped");
        std::fs::write(
            &path,
            "# OEIS stripped\nA000045 ,0,1,1,2,3,5,8,13,21\nA000290 ,0,1,4,9,16,25\n",
        )
        .unwrap();

        let mut encoder = IdEncoder::new();
        let mut bin_buf = Cursor::new(Vec::new());

        let (seqs, triples) = parse_stripped_gz(&path, &mut encoder, &mut bin_buf).unwrap();
        assert_eq!(seqs, 2);
        // A000045: 9 terms (capped at 20) + 1 instance_of = 10
        // A000290: 6 terms + 1 instance_of = 7
        assert_eq!(triples, 17);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
