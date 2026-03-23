// ── Heartbeat Engine — Continuous Learning ──────────────────────────
//
// The living system. Pulses periodically, learns from failures,
// discovers new rules, improves over time.
//
// DOMAIN AGNOSTIC — works with any facts and any target predicate.
// Domain-specific conversion (grids to facts, etc.) happens in the
// caller (CLI, agent) using bridge modules. The heartbeat only sees
// generic Statement facts.
//
// PULSE 1: Wake — analyze failures
// PULSE 2: Create — generate candidate rules (templates + mutations)
// PULSE 3: Test — verify candidates against unsolved tasks
// PULSE 4: Sleep — compress + prune (every Nth cycle)
// PULSE 5: Save — persist state

use std::path::{Path, PathBuf};
use std::time::Instant;

use qor_core::neuron::Statement;
use qor_core::truth_value::TruthValue;

use crate::chain::Rule;
use crate::eval::Session;
use crate::library::{RuleLibrary, RuleSource};
use crate::memory::{TaskRun, RunHistory};
use crate::search;
use crate::sleep;

/// Result of a single heartbeat pulse.
#[derive(Debug)]
pub struct PulseResult {
    pub new_solves: usize,
    pub library_size: usize,
    pub total_solved: usize,
    pub mutations_tried: usize,
    pub pulse_duration_ms: u64,
    pub web_facts_extracted: usize,
    pub web_pages_crawled: usize,
}

/// Generic task data for the heartbeat to work on.
/// All domain-specific conversion happens BEFORE creating this struct.
/// The heartbeat only sees Statement facts — no grids, no colors, no domain logic.
#[derive(Debug, Clone)]
pub struct TaskData {
    pub id: String,
    /// Training pairs: each entry is a set of input facts
    pub training_inputs: Vec<Vec<Statement>>,
    /// Training pairs: each entry is a set of expected output facts
    pub training_outputs: Vec<Vec<Statement>>,
    /// Which predicate the rules should produce (e.g., "predict-cell", "recommend", "diagnose")
    pub target_pred: String,
}

/// The heartbeat engine.
pub struct Heartbeat {
    pub library: RuleLibrary,
    pub history: RunHistory,
    sleep_interval: usize,
}

impl Heartbeat {
    pub fn new(brain_path: PathBuf) -> Self {
        let lib_path = brain_path.join("rules");
        let hist_path = brain_path.join("history.txt");

        Heartbeat {
            library: RuleLibrary::new(lib_path),
            history: RunHistory::new(hist_path),
            sleep_interval: 10,
        }
    }

    /// Load existing state from brain directory.
    pub fn load(brain_path: &Path) -> Self {
        let lib_path = brain_path.join("rules");
        let hist_path = brain_path.join("history.txt");

        let library = if lib_path.is_dir() {
            RuleLibrary::load(&lib_path)
        } else {
            RuleLibrary::new(lib_path)
        };

        let history = if hist_path.exists() {
            RunHistory::load(&hist_path)
        } else {
            RunHistory::new(hist_path)
        };

        Heartbeat {
            library,
            history,
            sleep_interval: 10,
        }
    }

    /// Run one heartbeat pulse.
    ///
    /// `tasks`: pre-converted task data (domain-specific conversion done by caller)
    /// `rules_qor`: hand-written QOR rules text
    /// `web_seeds`: optional QOR rule texts from web search (unverified candidates)
    pub fn pulse(
        &mut self,
        tasks: &[TaskData],
        rules_qor: &str,
        web_seeds: &[String],
    ) -> PulseResult {
        let start = Instant::now();
        let mut new_solves = 0;
        let mut mutations_tried = 0;

        self.history.increment_cycle();
        self.library.set_cycle(self.history.cycle_count() as u64);

        // Parse web-sourced candidate rules into executable Rules
        let web_rules = parse_qor_rules(web_seeds);

        // PULSE 1: Wake — analyze failure patterns
        let failure_patterns = self.history.failure_patterns();
        let _near_miss_transforms: Vec<String> = failure_patterns.iter()
            .filter(|(_, runs)| {
                runs.iter().any(|r| r.accuracy > 0.5)
            })
            .map(|(t, _)| t.clone())
            .collect();

        // PULSE 2+3: Create + Test — generate and evaluate candidate rules
        for task in tasks {
            if self.history.is_solved(&task.id) {
                continue;
            }

            if task.training_inputs.is_empty() {
                continue;
            }

            // Create a base session with hand-written rules
            let mut base_session = Session::new();
            let _ = base_session.exec(rules_qor);

            // Combine library rules with web-sourced candidates
            let mut all_seeds = self.library.parse_rules();
            all_seeds.extend(web_rules.iter().cloned());

            // If we have any seed rules, try refining them
            if !all_seeds.is_empty() {
                let result = search::refinement_search(
                    &all_seeds,
                    &task.training_inputs,
                    &task.training_outputs,
                    &task.target_pred,
                    &base_session,
                    1000, // 1 second per task
                );

                mutations_tried += result.mutations_tried;

                for solution in &result.solutions {
                    self.library.add(solution.qor_text.clone(), RuleSource::Mutated);
                    new_solves += 1;
                    self.history.mark_solved(&task.id);
                    self.history.record(TaskRun {
                        task_id: task.id.clone(),
                        detected_transforms: vec!["mutated".into()],
                        selected_transform: "mutated".into(),
                        rules_fired: vec![solution.qor_text.clone()],
                        correct: true,
                        accuracy: 1.0,
                        total_items: task.training_outputs.iter().map(|e| e.len()).sum(),
                        wrong_items: 0,
                    });
                    break;
                }
            }
        }

        // PULSE 4: Sleep — compress + prune (every Nth cycle)
        if self.history.cycle_count() % self.sleep_interval == 0 {
            sleep::sleep_cycle(&self.history, &mut self.library);
        }

        // PULSE 5: Save
        self.library.save();
        self.history.save();

        PulseResult {
            new_solves,
            library_size: self.library.len(),
            total_solved: self.history.total_solved(),
            mutations_tried,
            pulse_duration_ms: start.elapsed().as_millis() as u64,
            web_facts_extracted: web_rules.len(),
            web_pages_crawled: 0,
        }
    }

    /// Get total solved count.
    pub fn total_solved(&self) -> usize {
        self.history.total_solved()
    }

    /// Get library size.
    pub fn library_size(&self) -> usize {
        self.library.len()
    }
}

/// Parse QOR rule text strings into executable Rule objects.
fn parse_qor_rules(texts: &[String]) -> Vec<Rule> {
    let mut rules = Vec::new();
    for text in texts {
        if let Ok(stmts) = qor_core::parser::parse(text) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use qor_core::neuron::{Neuron, QorValue};

    fn make_fact(pred: &str, id: &str, a1: i64, a2: i64, val: i64) -> Statement {
        Statement::Fact {
            neuron: Neuron::Expression(vec![
                Neuron::Symbol(pred.into()),
                Neuron::Symbol(id.into()),
                Neuron::Value(QorValue::Int(a1)),
                Neuron::Value(QorValue::Int(a2)),
                Neuron::Value(QorValue::Int(val)),
            ]),
            tv: None,
            decay: None,
        }
    }

    fn simple_task() -> TaskData {
        TaskData {
            id: "test-task".into(),
            training_inputs: vec![vec![
                make_fact("source", "t0", 0, 0, 3),
                make_fact("source", "t0", 0, 1, 3),
                make_fact("source", "t0", 1, 0, 3),
                make_fact("source", "t0", 1, 1, 3),
            ]],
            training_outputs: vec![vec![
                make_fact("target", "t0", 0, 0, 7),
                make_fact("target", "t0", 0, 1, 7),
                make_fact("target", "t0", 1, 0, 7),
                make_fact("target", "t0", 1, 1, 7),
            ]],
            target_pred: "target".into(),
        }
    }

    #[test]
    fn test_heartbeat_new() {
        let tmp = std::env::temp_dir().join("qor_heartbeat_test");
        let hb = Heartbeat::new(tmp);
        assert_eq!(hb.total_solved(), 0);
        assert_eq!(hb.library_size(), 0);
    }

    #[test]
    fn test_heartbeat_single_pulse() {
        let tmp = std::env::temp_dir().join("qor_heartbeat_pulse_test");
        let _ = std::fs::create_dir_all(&tmp);
        let mut hb = Heartbeat::new(tmp.clone());

        let task = simple_task();
        let result = hb.pulse(&[task], "", &[]);
        // No library rules yet, so probably no solves
        assert_eq!(result.library_size, 0);

        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
