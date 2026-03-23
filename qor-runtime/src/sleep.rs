// ── Sleep Phase — Pattern Extraction + Compression ──────────────────
//
// Phase 3C: Analyze successful rule traces, find common patterns,
// compose primitives into reusable library rules.
//
// Inspired by: DreamCoder (abstraction sleep), Stitch (compression).

use std::collections::HashMap;

use crate::library::{PruneConfig, RuleLibrary, RuleSource};
use crate::memory::RunHistory;

/// Result of a sleep cycle.
#[derive(Debug)]
pub struct SleepResult {
    pub patterns_found: usize,
    pub rules_composed: usize,
    pub rules_pruned: usize,
    pub library_size_before: usize,
    pub library_size_after: usize,
}

/// Run a sleep cycle: analyze successes, extract patterns, compress library.
pub fn sleep_cycle(history: &RunHistory, library: &mut RuleLibrary) -> SleepResult {
    let before = library.len();

    // 1. Collect successful rule traces
    let successes = history.successes();

    // 2. Find common sub-sequences across successful traces
    let patterns = find_common_patterns(&successes);

    // 3. Build meta-rules from patterns (transform → conditions that predict success)
    let composed = compose_meta_rules(&patterns, history);

    // 4. Add high-value composed rules to library
    let mut rules_composed = 0;
    for (rule_text, confidence) in &composed {
        if *confidence > 0.6 {
            library.add(rule_text.clone(), RuleSource::Composed);
            rules_composed += 1;
        }
    }

    // 5. Prune library
    library.prune(&PruneConfig::default());

    let after = library.len();
    let pruned = if before + rules_composed > after {
        (before + rules_composed) - after
    } else {
        0
    };

    SleepResult {
        patterns_found: patterns.len(),
        rules_composed,
        rules_pruned: pruned,
        library_size_before: before,
        library_size_after: after,
    }
}

/// Find common patterns across successful runs.
///
/// Groups by selected_transform and counts how often each transform succeeds
/// when specific detection patterns are present.
fn find_common_patterns(successes: &[&crate::memory::TaskRun]) -> Vec<CommonPattern> {
    let mut patterns: HashMap<String, CommonPattern> = HashMap::new();

    for run in successes {
        let key = run.selected_transform.clone();
        let entry = patterns.entry(key.clone()).or_insert_with(|| CommonPattern {
            transform: key,
            count: 0,
            co_occurring_detections: HashMap::new(),
        });
        entry.count += 1;

        // Track which other transforms were also detected in successful runs
        for det in &run.detected_transforms {
            *entry.co_occurring_detections.entry(det.clone()).or_insert(0) += 1;
        }
    }

    patterns.into_values().collect()
}

/// A pattern found across successful puzzle runs.
#[derive(Debug)]
struct CommonPattern {
    transform: String,
    count: usize,
    co_occurring_detections: HashMap<String, usize>,
}

/// Compose meta-rules from patterns.
///
/// E.g., if reflect-h succeeds 80% of the time when "symmetric" is also detected,
/// emit: (rule-applies-when reflect-h obs-symmetric) <0.80, 0.90>
fn compose_meta_rules(
    patterns: &[CommonPattern],
    history: &RunHistory,
) -> Vec<(String, f64)> {
    let mut rules = Vec::new();
    let total_runs = history.all_runs().len().max(1);

    for pattern in patterns {
        if pattern.count < 2 {
            continue; // Need at least 2 successes to learn from
        }

        let success_rate = pattern.count as f64 / total_runs as f64;

        // Meta-rule: this transform tends to work
        let meta = format!(
            "(rule-confidence {} {:.2}) <{:.2}, {:.2}>",
            pattern.transform,
            success_rate,
            success_rate.min(0.99),
            confidence_from_count(pattern.count),
        );
        rules.push((meta, success_rate));

        // Co-occurrence meta-rules
        for (det, &co_count) in &pattern.co_occurring_detections {
            if co_count >= 2 && det != &pattern.transform {
                let co_rate = co_count as f64 / pattern.count as f64;
                if co_rate > 0.5 {
                    let co_meta = format!(
                        "(rule-applies-when {} {}) <{:.2}, {:.2}>",
                        pattern.transform, det,
                        co_rate.min(0.99),
                        confidence_from_count(co_count),
                    );
                    rules.push((co_meta, co_rate));
                }
            }
        }
    }

    rules
}

/// Convert count to PLN confidence using w2c formula.
fn confidence_from_count(count: usize) -> f64 {
    let k = 1.0; // Default PLN parameter
    let w = count as f64;
    (w / (w + k)).min(0.95)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::TaskRun;
    use std::path::PathBuf;

    fn make_run(id: &str, transform: &str, correct: bool) -> TaskRun {
        TaskRun {
            task_id: id.into(),
            detected_transforms: vec![transform.into(), "symmetric".into()],
            selected_transform: transform.into(),
            rules_fired: vec![],
            correct,
            accuracy: if correct { 1.0 } else { 0.3 },
            total_items: 9,
            wrong_items: if correct { 0 } else { 6 },
        }
    }

    #[test]
    fn test_find_common_patterns() {
        let runs: Vec<TaskRun> = vec![
            make_run("p1", "reflect-h", true),
            make_run("p2", "reflect-h", true),
            make_run("p3", "value-remap", true),
        ];
        let refs: Vec<&TaskRun> = runs.iter().collect();
        let patterns = find_common_patterns(&refs);

        assert_eq!(patterns.len(), 2); // reflect-h + color-remap
        let reflect = patterns.iter().find(|p| p.transform == "reflect-h").unwrap();
        assert_eq!(reflect.count, 2);
    }

    #[test]
    fn test_sleep_cycle() {
        let mut history = RunHistory::new(PathBuf::from("/tmp/test_sleep"));
        history.record(make_run("p1", "reflect-h", true));
        history.record(make_run("p2", "reflect-h", true));
        history.record(make_run("p3", "reflect-h", true));
        history.record(make_run("p4", "value-remap", false));

        let mut library = RuleLibrary::new(PathBuf::from("/tmp/test_lib"));
        let result = sleep_cycle(&history, &mut library);

        assert!(result.patterns_found >= 1);
        // Should have composed meta-rules from the 3 reflect-h successes
    }

    #[test]
    fn test_confidence_from_count() {
        assert!(confidence_from_count(1) > 0.4);
        assert!(confidence_from_count(10) > 0.9);
        assert!(confidence_from_count(100) <= 0.95);
    }
}
