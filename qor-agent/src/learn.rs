//! LEARN — Structured memory persistence.
//!
//! 5-level memory system (from issues.md):
//!   Level 1: wrong-attempt    — "I failed this item N times"
//!   Level 2: item-feature     — "this item has feature X" (searchable via trie)
//!   Level 3: tried-and-failed — "I tried strategy X and it didn't work"
//!   Level 4: best-partial     — "my best score was X% using strategy Y"
//!   Level 5: solved           — "I solved it using strategy Y" (the gold)
//!
//! Data stored as QOR facts with concrete item IDs (trie-indexed, not $task).
//! Section-based file format — each item gets ONE section, replaced atomically.
//!
//! Two files:
//!   memory.qor        — structured memory facts (all 5 levels)
//!   rules_learned.qor — executable QOR rules discovered by search
//!
//! Domain-agnostic — features extracted from obs-*/detected-*/consistent-* facts.

use anyhow::Result;
use qor_core::neuron::Neuron;
use qor_runtime::eval::Session;
use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

// ═══════════════════════════════════════════════════════════════════════
// FEATURE EXTRACTION — Pull searchable attributes from session
// ═══════════════════════════════════════════════════════════════════════

/// Extract item features from session — what kind of item is this?
///
/// Looks for obs-*, detected-*, consistent-* facts (produced by DNA rules).
/// Returns deduplicated, sorted feature names.
pub fn extract_features(session: &Session) -> Vec<String> {
    let mut features = HashSet::new();
    for fact in session.all_facts() {
        if let Neuron::Expression(parts) = &fact.neuron {
            if let Some(Neuron::Symbol(pred)) = parts.first() {
                if pred.starts_with("obs-")
                    || pred.starts_with("detected-")
                    || pred.starts_with("consistent-")
                {
                    features.insert(pred.clone());
                }
            }
        }
    }
    let mut sorted: Vec<String> = features.into_iter().collect();
    sorted.sort();
    sorted
}

/// Extract strategies that were tried from session facts.
///
/// Looks for detected-* facts (detection implies the strategy was attempted).
/// Also checks save-known-transform facts for what the session recognized.
pub fn extract_tried(session: &Session) -> Vec<String> {
    let mut tried = HashSet::new();
    for fact in session.all_facts() {
        if let Neuron::Expression(parts) = &fact.neuron {
            if let Some(Neuron::Symbol(pred)) = parts.first() {
                // detected-identity, detected-reflect-h → identity, reflect-h
                if let Some(strategy) = pred.strip_prefix("detected-") {
                    tried.insert(strategy.to_string());
                }
                // save-known-transform $task same-size → same-size
                if pred == "save-known-transform" {
                    if let Some(Neuron::Symbol(transform)) = parts.get(2) {
                        tried.insert(transform.clone());
                    }
                }
            }
        }
    }
    let mut sorted: Vec<String> = tried.into_iter().collect();
    sorted.sort();
    sorted
}

/// Extract winning strategy from session facts.
///
/// Checks save-known-transform, detected-*, and obs-* facts.
/// Returns the most specific strategy found, or "unknown".
pub fn extract_winning_strategy(session: &Session) -> String {
    // Priority 1: save-known-transform gives the best signal
    for fact in session.all_facts() {
        if let Neuron::Expression(parts) = &fact.neuron {
            if let Some(Neuron::Symbol(pred)) = parts.first() {
                if pred == "save-known-transform" {
                    if let Some(Neuron::Symbol(transform)) = parts.get(2) {
                        // Return the most specific (non-generic) transform
                        if transform != "same-size" && transform != "output-smaller" {
                            return transform.clone();
                        }
                    }
                }
            }
        }
    }
    // Priority 2: detected-* facts
    for fact in session.all_facts() {
        if let Neuron::Expression(parts) = &fact.neuron {
            if let Some(Neuron::Symbol(pred)) = parts.first() {
                if let Some(strategy) = pred.strip_prefix("detected-") {
                    return strategy.to_string();
                }
            }
        }
    }
    "unknown".to_string()
}

// ═══════════════════════════════════════════════════════════════════════
// RECORDING — Write structured memory to disk
// ═══════════════════════════════════════════════════════════════════════

/// Record a failed attempt with rich context.
///
/// Writes/updates the item's section in memory.qor with:
///   - wrong-attempt (level 1)
///   - item-feature (level 2) — trie-indexed for similarity search
///   - tried-and-failed (level 3) — skip these strategies next time
///   - best-partial (level 4) — start from here next time
pub fn record_attempt(
    item_id: &str,
    attempt: usize,
    score: f64,
    best_source: &str,
    features: &[String],
    tried: &[String],
    memory_path: &Path,
) -> Result<usize> {
    let mut lines = Vec::new();

    // Level 1: attempt count
    lines.push(format!("(wrong-attempt {item_id} {attempt}) <0.99, 0.99>"));

    // Level 2: item features (trie-indexed: item-feature → item_id → feature)
    for feat in features {
        lines.push(format!("(item-feature {item_id} {feat}) <0.90, 0.90>"));
    }

    // Level 3: tried and failed strategies
    for strategy in tried {
        lines.push(format!(
            "(tried-and-failed {item_id} {strategy}) <0.95, 0.95>"
        ));
    }

    // Level 4: best partial result
    if score > 0.05 && score < 0.999 {
        lines.push(format!(
            "(best-partial {item_id} {best_source} {:.2}) <0.90, 0.90>",
            score
        ));
    }

    upsert_section(item_id, &lines, memory_path)
}

/// Record a solved item — compress to essentials.
///
/// Replaces any failure data for this item with:
///   - solved (level 5) — the gold
///   - item-feature (level 2) — kept for similarity search
pub fn record_solved(
    item_id: &str,
    winning_strategy: &str,
    features: &[String],
    memory_path: &Path,
) -> Result<usize> {
    let mut lines = Vec::new();

    // Level 5: solved — the gold
    lines.push(format!(
        "(solved {item_id} {winning_strategy}) <0.99, 0.99>"
    ));

    // Level 2: keep item features (useful for similarity matching)
    for feat in features {
        lines.push(format!("(item-feature {item_id} {feat}) <0.90, 0.90>"));
    }

    // Replaces entire section — wrong-attempt/tried-and-failed/best-partial gone
    upsert_section(item_id, &lines, memory_path)
}

/// Save winning rules to a separate rules file.
///
/// These are executable QOR rules (not memory facts).
/// Deduplicated — won't add a rule that already exists in the file.
pub fn save_winning_rules(
    rules: &[String],
    item_id: &str,
    rules_path: &Path,
) -> Result<usize> {
    if rules.is_empty() {
        return Ok(0);
    }

    // Read existing rules to deduplicate
    let existing = if rules_path.exists() {
        std::fs::read_to_string(rules_path).unwrap_or_default()
    } else {
        String::new()
    };

    let new_rules: Vec<&String> = rules
        .iter()
        .filter(|r| !existing.contains(r.as_str()))
        .collect();

    if new_rules.is_empty() {
        return Ok(0);
    }

    if let Some(parent) = rules_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(rules_path)?;

    writeln!(f, "\n;; discovered for item:{item_id}")?;
    for rule_text in &new_rules {
        writeln!(f, "{rule_text}")?;
    }

    Ok(new_rules.len())
}

// ═══════════════════════════════════════════════════════════════════════
// QUERYING — Read structured memory
// ═══════════════════════════════════════════════════════════════════════

/// Load solved IDs from memory.qor.
///
/// Looks for `(solved ITEM_ID ...)` facts in the file.
pub fn load_solved_ids(memory_path: &Path) -> HashSet<String> {
    let mut solved = HashSet::new();
    if let Ok(content) = std::fs::read_to_string(memory_path) {
        for line in content.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("(solved ") {
                // (solved 009d5c81 color-remap) <0.99, 0.99>
                if let Some(id) = rest.split_whitespace().next() {
                    solved.insert(id.to_string());
                }
            }
        }
    }
    solved
}

/// Get memory stats for display.
pub fn memory_stats(memory_path: &Path) -> (usize, usize, usize) {
    let mut solved = 0;
    let mut failed = 0;
    let mut total_facts = 0;
    if let Ok(content) = std::fs::read_to_string(memory_path) {
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("(solved ") {
                solved += 1;
            } else if trimmed.starts_with("(wrong-attempt ") {
                failed += 1;
            }
            if trimmed.starts_with('(') {
                total_facts += 1;
            }
        }
    }
    (solved, failed, total_facts)
}

// ═══════════════════════════════════════════════════════════════════════
// SECTION MANAGEMENT — Atomic per-item file updates
// ═══════════════════════════════════════════════════════════════════════

/// Section marker format: ";; ══ ITEM: <id> ══"
fn section_marker(item_id: &str) -> String {
    format!(";; ══ {item_id} ══")
}

/// Insert or replace a section for an item in the memory file.
///
/// Each item gets exactly ONE section. If the item already has a section,
/// it's replaced atomically (no duplication). If not, a new section is appended.
///
/// Returns the number of facts written.
fn upsert_section(item_id: &str, lines: &[String], path: &Path) -> Result<usize> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let marker = section_marker(item_id);

    // Read existing content
    let existing = if path.exists() {
        std::fs::read_to_string(path)?
    } else {
        String::new()
    };

    // Build new section
    let mut new_section = String::new();
    new_section.push_str(&marker);
    new_section.push('\n');
    for line in lines {
        new_section.push_str(line);
        new_section.push('\n');
    }

    // Find and replace existing section, or append
    if let Some(start) = existing.find(&marker) {
        // Find end of section (next marker or EOF)
        let after_marker = start + marker.len();
        let end = existing[after_marker..]
            .find(";; ══ ")
            .map(|pos| after_marker + pos)
            .unwrap_or(existing.len());

        let mut result = String::with_capacity(existing.len());
        result.push_str(&existing[..start]);
        result.push_str(&new_section);
        // Skip trailing blank line before next section
        let rest = existing[end..].trim_start_matches('\n');
        if !rest.is_empty() {
            result.push('\n');
            result.push_str(rest);
        }
        std::fs::write(path, result)?;
    } else {
        // Append new section
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        writeln!(f)?; // blank line separator
        write!(f, "{new_section}")?;
    }

    Ok(lines.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("qor_learn_test_{name}"))
    }

    #[test]
    fn test_record_attempt() {
        let path = temp_path("attempt");
        let _ = std::fs::remove_file(&path);

        let features = vec!["obs-same-size".to_string(), "detected-identity".to_string()];
        let tried = vec!["identity".to_string(), "reflect-h".to_string()];

        let n = record_attempt("abc123", 1, 0.75, "identity", &features, &tried, &path)
            .unwrap();

        assert!(n > 0);
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("(wrong-attempt abc123 1)"));
        assert!(content.contains("(item-feature abc123 obs-same-size)"));
        assert!(content.contains("(item-feature abc123 detected-identity)"));
        assert!(content.contains("(tried-and-failed abc123 identity)"));
        assert!(content.contains("(tried-and-failed abc123 reflect-h)"));
        assert!(content.contains("(best-partial abc123 identity 0.75)"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_record_solved_replaces_failure() {
        let path = temp_path("solved_replace");
        let _ = std::fs::remove_file(&path);

        // First: record a failure
        let features = vec!["obs-same-size".to_string()];
        let tried = vec!["identity".to_string()];
        record_attempt("abc123", 1, 0.50, "identity", &features, &tried, &path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("wrong-attempt"));
        assert!(content.contains("tried-and-failed"));

        // Then: record solved — should replace the failure section
        record_solved("abc123", "color-remap", &features, &path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("(solved abc123 color-remap)"));
        assert!(content.contains("(item-feature abc123 obs-same-size)"));
        // Failure data gone
        assert!(!content.contains("wrong-attempt"));
        assert!(!content.contains("tried-and-failed"));
        assert!(!content.contains("best-partial"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_upsert_no_duplication() {
        let path = temp_path("no_dup");
        let _ = std::fs::remove_file(&path);

        let features = vec!["obs-same-size".to_string()];
        let tried = vec!["identity".to_string()];

        // Record same item 3 times — should have exactly 1 section
        record_attempt("abc123", 1, 0.30, "identity", &features, &tried, &path).unwrap();
        record_attempt("abc123", 2, 0.50, "identity", &features, &tried, &path).unwrap();
        record_attempt("abc123", 3, 0.75, "reflect-h", &features, &tried, &path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let section_count = content.matches(";; ══ abc123 ══").count();
        assert_eq!(section_count, 1, "Should have exactly 1 section, got {section_count}");

        // Should have the latest attempt number
        assert!(content.contains("(wrong-attempt abc123 3)"));
        // Should NOT have old attempt numbers
        assert!(!content.contains("(wrong-attempt abc123 1)"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_multiple_items() {
        let path = temp_path("multi");
        let _ = std::fs::remove_file(&path);

        let f1 = vec!["obs-same-size".to_string()];
        let f2 = vec!["obs-reflect-h".to_string()];
        let tried = vec!["identity".to_string()];

        record_attempt("item_a", 1, 0.50, "identity", &f1, &tried, &path).unwrap();
        record_attempt("item_b", 1, 0.30, "identity", &f2, &tried, &path).unwrap();
        record_solved("item_a", "color-remap", &f1, &path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        // item_a: solved, no failure data
        assert!(content.contains("(solved item_a color-remap)"));
        assert!(!content.contains("(wrong-attempt item_a"));
        // item_b: still failed
        assert!(content.contains("(wrong-attempt item_b 1)"));
        assert!(content.contains("(item-feature item_b obs-reflect-h)"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_solved_ids() {
        let path = temp_path("solved_ids");
        let _ = std::fs::remove_file(&path);

        let features = vec!["obs-same-size".to_string()];
        record_solved("aaa", "identity", &features, &path).unwrap();
        record_solved("bbb", "reflect-h", &features, &path).unwrap();
        record_attempt("ccc", 1, 0.50, "identity", &features, &[], &path).unwrap();

        let solved = load_solved_ids(&path);
        assert!(solved.contains("aaa"));
        assert!(solved.contains("bbb"));
        assert!(!solved.contains("ccc"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_memory_stats() {
        let path = temp_path("stats");
        let _ = std::fs::remove_file(&path);

        let features = vec!["obs-same-size".to_string()];
        record_solved("aaa", "identity", &features, &path).unwrap();
        record_attempt("bbb", 2, 0.50, "identity", &features, &["identity".to_string()], &path).unwrap();

        let (solved, failed, total) = memory_stats(&path);
        assert_eq!(solved, 1);
        assert_eq!(failed, 1);
        assert!(total >= 4); // at least: solved + 2 features + wrong-attempt

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_winning_rules_dedup() {
        let path = temp_path("rules");
        let _ = std::fs::remove_file(&path);

        let rules = vec![
            "(predict-cell $r $c $v) if (grid-cell ti $r $c $v) <0.95, 0.95>".to_string(),
        ];

        let n1 = save_winning_rules(&rules, "aaa", &path).unwrap();
        assert_eq!(n1, 1);

        // Same rule again — should be deduped
        let n2 = save_winning_rules(&rules, "bbb", &path).unwrap();
        assert_eq!(n2, 0);

        let _ = std::fs::remove_file(&path);
    }
}
