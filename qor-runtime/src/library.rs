// ── Scored Rule Library ─────────────────────────────────────────────
//
// Phase 3D: Persistent library of learned rules with scoring.
// Rules come from: hand-written, induced, composed, mutated.
// Pruned by Occam's razor + usage tracking.

use std::path::{Path, PathBuf};
use qor_core::neuron::Statement;
use qor_core::parser;
use qor_core::truth_value::TruthValue;

use crate::chain::Rule;

/// Source of a rule in the library.
#[derive(Debug, Clone, PartialEq)]
pub enum RuleSource {
    HandWritten,
    Induced,
    Composed,
    Mutated,
    Template,
}

/// A rule with usage statistics.
#[derive(Debug, Clone)]
pub struct ScoredRule {
    pub rule_text: String,
    pub source: RuleSource,
    pub times_fired: usize,
    pub times_correct: usize,
    pub created_at: u64,
    pub last_used: u64,
}

impl ScoredRule {
    /// Accuracy = times_correct / times_fired (0.0 if never fired).
    pub fn accuracy(&self) -> f64 {
        if self.times_fired == 0 { 0.0 }
        else { self.times_correct as f64 / self.times_fired as f64 }
    }
}

/// Configuration for library pruning.
pub struct PruneConfig {
    pub max_rules: usize,
    pub min_accuracy: f64,
    pub unused_threshold: usize, // Remove if unused in this many cycles
}

impl Default for PruneConfig {
    fn default() -> Self {
        PruneConfig {
            max_rules: 500,
            min_accuracy: 0.1,
            unused_threshold: 100,
        }
    }
}

/// Persistent rule library.
pub struct RuleLibrary {
    rules: Vec<ScoredRule>,
    path: PathBuf,
    current_cycle: u64,
}

impl RuleLibrary {
    pub fn new(path: PathBuf) -> Self {
        RuleLibrary {
            rules: Vec::new(),
            path,
            current_cycle: 0,
        }
    }

    /// Load library from a directory of .qor files.
    pub fn load(dir: &Path) -> Self {
        let mut lib = RuleLibrary::new(dir.to_path_buf());
        if dir.is_dir() {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) == Some("qor") {
                        if let Ok(contents) = std::fs::read_to_string(&path) {
                            for line in contents.lines() {
                                let line = line.trim();
                                if line.is_empty() || line.starts_with(";;") {
                                    continue;
                                }
                                lib.rules.push(ScoredRule {
                                    rule_text: line.to_string(),
                                    source: RuleSource::HandWritten,
                                    times_fired: 0,
                                    times_correct: 0,
                                    created_at: 0,
                                    last_used: 0,
                                });
                            }
                        }
                    }
                }
            }
        }
        lib
    }

    /// Add a new rule to the library.
    pub fn add(&mut self, rule_text: String, source: RuleSource) {
        // Deduplicate
        if self.rules.iter().any(|r| r.rule_text == rule_text) {
            return;
        }
        self.rules.push(ScoredRule {
            rule_text,
            source,
            times_fired: 0,
            times_correct: 0,
            created_at: self.current_cycle,
            last_used: self.current_cycle,
        });
    }

    /// Mark a rule as fired (and whether it was correct).
    pub fn record_firing(&mut self, rule_text: &str, correct: bool) {
        for r in &mut self.rules {
            if r.rule_text == rule_text {
                r.times_fired += 1;
                if correct {
                    r.times_correct += 1;
                }
                r.last_used = self.current_cycle;
                return;
            }
        }
    }

    /// Get all rules as text.
    pub fn all_rules(&self) -> &[ScoredRule] {
        &self.rules
    }

    /// Number of rules in the library.
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Parse all rules into executable Rule objects.
    pub fn parse_rules(&self) -> Vec<Rule> {
        let mut rules = Vec::new();
        for sr in &self.rules {
            if let Ok(stmts) = parser::parse(&sr.rule_text) {
                for stmt in stmts {
                    if let Statement::Rule { head, body, tv } = stmt {
                        rules.push(Rule::new(
                            head,
                            body,
                            tv.unwrap_or(TruthValue::default_fact()),
                        ));
                    }
                }
            }
        }
        rules
    }

    /// Set current cycle (for tracking recency).
    pub fn set_cycle(&mut self, cycle: u64) {
        self.current_cycle = cycle;
    }

    /// Prune the library based on config.
    pub fn prune(&mut self, config: &PruneConfig) {
        // Remove rules that are too inaccurate (but keep hand-written)
        self.rules.retain(|r| {
            r.source == RuleSource::HandWritten
            || r.times_fired < 5 // Give new rules a chance
            || r.accuracy() >= config.min_accuracy
        });

        // Remove rules unused for too long (but keep hand-written)
        let threshold_cycle = self.current_cycle.saturating_sub(config.unused_threshold as u64);
        self.rules.retain(|r| {
            r.source == RuleSource::HandWritten
            || r.last_used >= threshold_cycle
            || r.times_correct > 0 // Keep any rule that ever worked
        });

        // Cap at max_rules — keep highest accuracy, then most recent
        if self.rules.len() > config.max_rules {
            self.rules.sort_by(|a, b| {
                // Hand-written first, then by accuracy, then by recency
                let a_hw = if a.source == RuleSource::HandWritten { 1.0 } else { 0.0 };
                let b_hw = if b.source == RuleSource::HandWritten { 1.0 } else { 0.0 };
                b_hw.partial_cmp(&a_hw).unwrap_or(std::cmp::Ordering::Equal)
                    .then(b.accuracy().partial_cmp(&a.accuracy()).unwrap_or(std::cmp::Ordering::Equal))
                    .then(b.last_used.cmp(&a.last_used))
            });
            self.rules.truncate(config.max_rules);
        }
    }

    /// Save library to the directory as a .qor file.
    pub fn save(&self) {
        if let Err(e) = std::fs::create_dir_all(&self.path) {
            eprintln!("[library] warning: could not create dir {:?}: {}", self.path, e);
        }
        let learned_path = self.path.join("learned.qor");

        let mut lines = Vec::new();
        lines.push(";; QOR Learned Rule Library".to_string());
        lines.push(format!(";; {} rules, cycle {}", self.rules.len(), self.current_cycle));
        lines.push(String::new());

        for r in &self.rules {
            if r.source == RuleSource::HandWritten {
                continue; // Don't re-save hand-written rules
            }
            let source_tag = match r.source {
                RuleSource::Induced => "induced",
                RuleSource::Composed => "composed",
                RuleSource::Mutated => "mutated",
                RuleSource::Template => "template",
                RuleSource::HandWritten => unreachable!(),
            };
            lines.push(format!(";; source={} fired={} correct={} accuracy={:.2}",
                source_tag, r.times_fired, r.times_correct, r.accuracy()));
            lines.push(r.rule_text.clone());
        }

        if let Err(e) = std::fs::write(&learned_path, lines.join("\n")) {
            eprintln!("[library] warning: could not save {:?}: {}", learned_path, e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_dedup() {
        let mut lib = RuleLibrary::new(PathBuf::from("/tmp/test_lib"));
        lib.add("(a) if (b) <0.9, 0.9>".into(), RuleSource::Induced);
        lib.add("(a) if (b) <0.9, 0.9>".into(), RuleSource::Induced); // dup
        lib.add("(c) if (d) <0.9, 0.9>".into(), RuleSource::Mutated);
        assert_eq!(lib.len(), 2);
    }

    #[test]
    fn test_record_firing() {
        let mut lib = RuleLibrary::new(PathBuf::from("/tmp/test_lib"));
        lib.add("(a) if (b) <0.9, 0.9>".into(), RuleSource::Induced);
        lib.record_firing("(a) if (b) <0.9, 0.9>", true);
        lib.record_firing("(a) if (b) <0.9, 0.9>", false);
        assert_eq!(lib.all_rules()[0].times_fired, 2);
        assert_eq!(lib.all_rules()[0].times_correct, 1);
        assert!((lib.all_rules()[0].accuracy() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_prune_low_accuracy() {
        let mut lib = RuleLibrary::new(PathBuf::from("/tmp/test_lib"));
        lib.add("(good) if (x) <0.9, 0.9>".into(), RuleSource::Induced);
        lib.add("(bad) if (y) <0.9, 0.9>".into(), RuleSource::Induced);

        // Make "good" accurate, "bad" inaccurate
        for _ in 0..10 {
            lib.record_firing("(good) if (x) <0.9, 0.9>", true);
            lib.record_firing("(bad) if (y) <0.9, 0.9>", false);
        }

        lib.prune(&PruneConfig {
            max_rules: 500,
            min_accuracy: 0.5,
            unused_threshold: 100,
        });

        assert_eq!(lib.len(), 1);
        assert!(lib.all_rules()[0].rule_text.contains("good"));
    }

    #[test]
    fn test_handwritten_never_pruned() {
        let mut lib = RuleLibrary::new(PathBuf::from("/tmp/test_lib"));
        lib.add("(hw) if (x) <0.9, 0.9>".into(), RuleSource::HandWritten);

        // Even with bad stats, hand-written stays
        for _ in 0..10 {
            lib.record_firing("(hw) if (x) <0.9, 0.9>", false);
        }

        lib.prune(&PruneConfig {
            max_rules: 500,
            min_accuracy: 0.9,
            unused_threshold: 1,
        });

        assert_eq!(lib.len(), 1);
    }

    #[test]
    fn test_parse_rules() {
        let mut lib = RuleLibrary::new(PathBuf::from("/tmp/test_lib"));
        lib.add("(flies $x) if (bird $x) <0.95, 0.90>".into(), RuleSource::Induced);
        let rules = lib.parse_rules();
        assert_eq!(rules.len(), 1);
    }
}
