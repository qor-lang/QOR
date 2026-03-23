// ── Run History / Experience Memory ──────────────────────────────────
//
// Records every task attempt — what was tried, what worked.
// Analyzed by sleep.rs for pattern extraction + library learning.
//
// DOMAIN AGNOSTIC — works with any task (puzzles, trading, medical, etc.)
// The engine doesn't know or care what domain it's solving.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use qor_core::neuron::{Neuron, QorValue, Statement};
use qor_core::truth_value::TruthValue;

/// Record of a single task attempt.
#[derive(Debug, Clone)]
pub struct TaskRun {
    pub task_id: String,
    pub detected_transforms: Vec<String>,
    pub selected_transform: String,
    pub rules_fired: Vec<String>,
    pub correct: bool,
    pub accuracy: f64,
    pub total_items: usize,
    pub wrong_items: usize,
}

/// Persistent run history across solving sessions.
pub struct RunHistory {
    runs: Vec<TaskRun>,
    solved_ids: std::collections::HashSet<String>,
    path: PathBuf,
    cycle_count: usize,
}

impl RunHistory {
    pub fn new(path: PathBuf) -> Self {
        RunHistory {
            runs: Vec::new(),
            solved_ids: std::collections::HashSet::new(),
            path,
            cycle_count: 0,
        }
    }

    /// Load history from a file (simple line-based format).
    pub fn load(path: &Path) -> Self {
        let mut history = RunHistory::new(path.to_path_buf());
        if let Ok(data) = std::fs::read_to_string(path) {
            for line in data.lines() {
                if let Some(run) = parse_run_line(line) {
                    if run.correct {
                        history.solved_ids.insert(run.task_id.clone());
                    }
                    history.runs.push(run);
                }
            }
        }
        history
    }

    /// Record a new task run.
    pub fn record(&mut self, run: TaskRun) {
        if run.correct {
            self.solved_ids.insert(run.task_id.clone());
        }
        self.runs.push(run);
    }

    /// Get all runs.
    pub fn all_runs(&self) -> &[TaskRun] {
        &self.runs
    }

    /// Get successful runs.
    pub fn successes(&self) -> Vec<&TaskRun> {
        self.runs.iter().filter(|r| r.correct).collect()
    }

    /// Get failed runs.
    pub fn failures(&self) -> Vec<&TaskRun> {
        self.runs.iter().filter(|r| !r.correct).collect()
    }

    /// Group failures by selected transform.
    pub fn failure_patterns(&self) -> HashMap<String, Vec<&TaskRun>> {
        let mut groups: HashMap<String, Vec<&TaskRun>> = HashMap::new();
        for run in &self.runs {
            if !run.correct {
                groups.entry(run.selected_transform.clone())
                    .or_default().push(run);
            }
        }
        groups
    }

    /// Check if a task has been solved.
    pub fn is_solved(&self, task_id: &str) -> bool {
        self.solved_ids.contains(task_id)
    }

    /// Mark a task as solved.
    pub fn mark_solved(&mut self, task_id: &str) {
        self.solved_ids.insert(task_id.to_string());
    }

    /// Total solved count.
    pub fn total_solved(&self) -> usize {
        self.solved_ids.len()
    }

    /// Increment cycle count.
    pub fn increment_cycle(&mut self) {
        self.cycle_count += 1;
    }

    /// Get cycle count.
    pub fn cycle_count(&self) -> usize {
        self.cycle_count
    }

    /// Save history to file.
    pub fn save(&self) {
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut lines = Vec::new();
        for run in &self.runs {
            lines.push(format!(
                "{}|{}|{}|{}|{:.4}|{}|{}",
                run.task_id,
                run.detected_transforms.join(","),
                run.selected_transform,
                if run.correct { "OK" } else { "FAIL" },
                run.accuracy,
                run.total_items,
                run.wrong_items,
            ));
        }
        let _ = std::fs::write(&self.path, lines.join("\n"));
    }
}

/// Parse a history line back into a TaskRun.
fn parse_run_line(line: &str) -> Option<TaskRun> {
    let parts: Vec<&str> = line.split('|').collect();
    if parts.len() < 7 {
        return None;
    }
    Some(TaskRun {
        task_id: parts[0].to_string(),
        detected_transforms: parts[1].split(',').map(|s| s.to_string()).collect(),
        selected_transform: parts[2].to_string(),
        correct: parts[3] == "OK",
        accuracy: parts[4].parse().unwrap_or(0.0),
        total_items: parts[5].parse().unwrap_or(0),
        wrong_items: parts[6].parse().unwrap_or(0),
        rules_fired: Vec::new(), // Not persisted in simple format
    })
}

// ── Failure Memory — "Never repeat mistakes" ────────────────────────
//
// Tracks failed approaches to prevent retrying. Converts failures to
// QOR facts so reasoning.qor can read them and adjust strategy.
//
// "Those who cannot remember the past are condemned to repeat it."
//   — George Santayana

/// Tracks failed rules and approaches to prevent retrying.
/// Injected into sessions as QOR facts so reasoning.qor can react.
#[derive(Default, Clone, Debug)]
pub struct FailureMemory {
    /// Failed rule texts (exact match)
    failed_rules: HashSet<String>,
    /// Score when each rule was tried (for reporting)
    failed_rule_scores: HashMap<String, f64>,
    /// Failed approach types ("spatial", "remap", "identity-fix", etc.)
    failed_approaches: HashSet<String>,
    /// Count of attempts per approach
    attempt_counts: HashMap<String, usize>,
    /// Best score achieved per approach (for partial credit tracking)
    best_scores: HashMap<String, f64>,
    /// Predicate patterns that led to overfitting (100% train, <95% test)
    overfit_signatures: HashSet<String>,
    /// Per-predicate overfit count
    overfit_pred_counts: HashMap<String, usize>,
    /// Total overfits detected
    overfit_count: usize,
}

impl FailureMemory {
    pub fn new() -> Self { Self::default() }

    /// Record a failed attempt with its score.
    pub fn record_failure(&mut self, rule_text: &str, approach: &str, score: f64) {
        self.failed_rules.insert(rule_text.to_string());
        self.failed_rule_scores.insert(rule_text.to_string(), score);
        *self.attempt_counts.entry(approach.to_string()).or_default() += 1;
        let best = self.best_scores.entry(approach.to_string()).or_insert(0.0);
        if score > *best {
            *best = score;
        }
        // Only mark approach as failed if all attempts scored poorly
        let attempts = self.attempt_counts.get(approach).copied().unwrap_or(0);
        if attempts >= 5 && *best < 0.5 {
            self.failed_approaches.insert(approach.to_string());
        }
    }

    /// Record an approach as explicitly failed (no score threshold check).
    pub fn mark_approach_failed(&mut self, approach: &str) {
        self.failed_approaches.insert(approach.to_string());
    }

    /// Check if this exact rule already failed.
    pub fn already_failed(&self, rule_text: &str) -> bool {
        self.failed_rules.contains(rule_text)
    }

    /// Check if this approach type is exhausted.
    pub fn approach_exhausted(&self, approach: &str, max_attempts: usize) -> bool {
        self.attempt_counts.get(approach).copied().unwrap_or(0) >= max_attempts
            || self.failed_approaches.contains(approach)
    }

    /// Record a rule that scored well on training but failed on test (overfitting).
    /// Tracks the predicate pattern to detect similar overfitting rules.
    pub fn record_overfit(&mut self, rule_text: &str, body_preds: &[String], train_score: f64, test_score: f64) {
        self.failed_rules.insert(rule_text.to_string());
        self.failed_rule_scores.insert(rule_text.to_string(), train_score);
        self.overfit_count += 1;
        for pred in body_preds {
            *self.overfit_pred_counts.entry(pred.clone()).or_default() += 1;
        }
        let mut sig = body_preds.to_vec();
        sig.sort();
        sig.dedup();
        self.overfit_signatures.insert(sig.join("+"));
        eprintln!("    [memory] overfit #{}: train={:.1}% test={:.1}% preds={:?} rule={}",
            self.overfit_count, train_score * 100.0, test_score * 100.0, body_preds,
            rule_text.chars().take(100).collect::<String>());
    }

    /// Total number of overfits detected.
    pub fn overfit_count(&self) -> usize { self.overfit_count }

    /// Check if a candidate rule's predicates overlap with known overfit patterns.
    /// Returns a penalty (0.0 = no penalty, up to 0.5 = heavy penalty).
    pub fn overfit_penalty(&self, body_preds: &[String]) -> f64 {
        if self.overfit_signatures.is_empty() { return 0.0; }
        let mut sig = body_preds.to_vec();
        sig.sort();
        sig.dedup();
        let sig_str = sig.join("+");
        if self.overfit_signatures.contains(&sig_str) {
            return 0.5;
        }
        if body_preds.is_empty() { return 0.0; }
        let overlap: usize = body_preds.iter()
            .filter(|p| self.overfit_pred_counts.contains_key(*p))
            .count();
        let ratio = overlap as f64 / body_preds.len() as f64;
        ratio * 0.3
    }

    /// Get all failed rules with their scores (for debugging).
    pub fn failed_rules_with_scores(&self) -> Vec<(&str, f64)> {
        self.failed_rules.iter()
            .map(|r| (r.as_str(), self.failed_rule_scores.get(r).copied().unwrap_or(0.0)))
            .collect()
    }

    /// Filter candidates: remove already-failed rules.
    pub fn filter_candidates<T: HasRuleText>(&self, candidates: Vec<T>) -> Vec<T> {
        let before = candidates.len();
        let result: Vec<T> = candidates.into_iter()
            .filter(|c| !self.failed_rules.contains(c.rule_text()))
            .collect();
        let filtered = before - result.len();
        if filtered > 0 {
            eprintln!("    [memory] filtered {} previously-failed rules ({} → {})",
                filtered, before, result.len());
        }
        result
    }

    /// Convert failures to QOR facts (for reasoning.qor to read).
    ///
    /// Produces:
    ///   (failed-approach "spatial") — this approach was tried and failed
    ///   (attempt-count "spatial" 5) — tried 5 times
    ///   (best-attempt-score "spatial" 42) — best score was 42% (scaled to int)
    pub fn to_statements(&self) -> Vec<Statement> {
        let mut stmts = Vec::new();
        for approach in &self.failed_approaches {
            stmts.push(Statement::Fact {
                neuron: Neuron::Expression(vec![
                    Neuron::Symbol("failed-approach".into()),
                    Neuron::Symbol(approach.clone()),
                ]),
                tv: Some(TruthValue::new(0.95, 0.95)),
                decay: None,
            });
        }
        for (approach, count) in &self.attempt_counts {
            stmts.push(Statement::Fact {
                neuron: Neuron::Expression(vec![
                    Neuron::Symbol("attempt-count".into()),
                    Neuron::Symbol(approach.clone()),
                    Neuron::Value(QorValue::Int(*count as i64)),
                ]),
                tv: Some(TruthValue::new(0.95, 0.95)),
                decay: None,
            });
        }
        for (approach, score) in &self.best_scores {
            stmts.push(Statement::Fact {
                neuron: Neuron::Expression(vec![
                    Neuron::Symbol("best-attempt-score".into()),
                    Neuron::Symbol(approach.clone()),
                    Neuron::Value(QorValue::Int((*score * 100.0) as i64)),
                ]),
                tv: Some(TruthValue::new(0.95, 0.95)),
                decay: None,
            });
        }
        stmts
    }

    /// Total number of unique failed rules.
    pub fn total_failures(&self) -> usize { self.failed_rules.len() }

    /// Access the set of failed rule texts (for passing to genesis to skip).
    pub fn failed_rule_set(&self) -> &HashSet<String> { &self.failed_rules }

    /// Total number of exhausted approaches.
    pub fn exhausted_approaches(&self) -> usize { self.failed_approaches.len() }
}

/// Trait for types that have a rule_text field (for filter_candidates).
pub trait HasRuleText {
    fn rule_text(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn sample_run(id: &str, correct: bool) -> TaskRun {
        TaskRun {
            task_id: id.to_string(),
            detected_transforms: vec!["reflect-h".into()],
            selected_transform: "reflect-h".into(),
            rules_fired: vec![],
            correct,
            accuracy: if correct { 1.0 } else { 0.5 },
            total_items: 9,
            wrong_items: if correct { 0 } else { 4 },
        }
    }

    #[test]
    fn test_record_and_query() {
        let mut hist = RunHistory::new(PathBuf::from("/tmp/test_history.txt"));
        hist.record(sample_run("task-1", true));
        hist.record(sample_run("task-2", false));
        hist.record(sample_run("task-3", false));

        assert_eq!(hist.successes().len(), 1);
        assert_eq!(hist.failures().len(), 2);
        assert!(hist.is_solved("task-1"));
        assert!(!hist.is_solved("task-2"));
        assert_eq!(hist.total_solved(), 1);
    }

    #[test]
    fn test_failure_patterns() {
        let mut hist = RunHistory::new(PathBuf::from("/tmp/test_history.txt"));
        hist.record(TaskRun {
            task_id: "t1".into(),
            detected_transforms: vec!["crop".into()],
            selected_transform: "crop".into(),
            rules_fired: vec![],
            correct: false, accuracy: 0.6, total_items: 10, wrong_items: 4,
        });
        hist.record(TaskRun {
            task_id: "t2".into(),
            detected_transforms: vec!["crop".into()],
            selected_transform: "crop".into(),
            rules_fired: vec![],
            correct: false, accuracy: 0.7, total_items: 10, wrong_items: 3,
        });
        hist.record(TaskRun {
            task_id: "t3".into(),
            detected_transforms: vec!["reflect-h".into()],
            selected_transform: "reflect-h".into(),
            rules_fired: vec![],
            correct: false, accuracy: 0.4, total_items: 10, wrong_items: 6,
        });

        let patterns = hist.failure_patterns();
        assert_eq!(patterns["crop"].len(), 2);
        assert_eq!(patterns["reflect-h"].len(), 1);
    }

    #[test]
    fn test_save_and_load() {
        let tmp_path = std::env::temp_dir().join("qor_test_history.txt");
        let mut hist = RunHistory::new(tmp_path.clone());
        hist.record(sample_run("task-1", true));
        hist.record(sample_run("task-2", false));
        hist.save();

        let loaded = RunHistory::load(&tmp_path);
        assert_eq!(loaded.all_runs().len(), 2);
        assert!(loaded.is_solved("task-1"));
        assert!(!loaded.is_solved("task-2"));

        // Cleanup
        let _ = std::fs::remove_file(&tmp_path);
    }

    // ── FailureMemory tests ──────────────────────────────────────────

    #[test]
    fn test_failure_memory_record_and_check() {
        let mut mem = FailureMemory::new();
        mem.record_failure("(a) if (b) <0.9, 0.9>", "remap", 0.3);
        assert!(mem.already_failed("(a) if (b) <0.9, 0.9>"));
        assert!(!mem.already_failed("(c) if (d) <0.9, 0.9>"));
        assert_eq!(mem.total_failures(), 1);
    }

    #[test]
    fn test_failure_memory_approach_exhaustion() {
        let mut mem = FailureMemory::new();
        // 5 low-score failures exhaust the approach
        for i in 0..5 {
            mem.record_failure(&format!("rule-{}", i), "spatial", 0.1);
        }
        assert!(mem.approach_exhausted("spatial", 5));
        assert!(!mem.approach_exhausted("remap", 5));
    }

    #[test]
    fn test_failure_memory_high_score_not_exhausted() {
        let mut mem = FailureMemory::new();
        // 5 attempts but one scored well — approach not marked as failed
        for i in 0..4 {
            mem.record_failure(&format!("rule-{}", i), "remap", 0.1);
        }
        mem.record_failure("rule-4", "remap", 0.8); // High score
        // approach_exhausted by count but not by failed_approaches
        assert!(mem.approach_exhausted("remap", 5)); // count-based
        assert_eq!(mem.exhausted_approaches(), 0); // not in failed_approaches
    }

    #[test]
    fn test_failure_memory_to_statements() {
        let mut mem = FailureMemory::new();
        mem.mark_approach_failed("spatial");
        mem.record_failure("rule-1", "remap", 0.45);
        let stmts = mem.to_statements();
        assert!(!stmts.is_empty());
        // Should have failed-approach, attempt-count, best-attempt-score facts
        let texts: Vec<String> = stmts.iter().map(|s| format!("{:?}", s)).collect();
        let all = texts.join(" ");
        assert!(all.contains("failed-approach"));
        assert!(all.contains("attempt-count"));
        assert!(all.contains("best-attempt-score"));
    }
}
