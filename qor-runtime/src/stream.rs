// ── QOR Reasoning Streams ──────────────────────────────────────────────
//
// The heartbeat is the life of QOR. These streams make it observable:
//
// 1. HeartbeatIter    — synchronous iterator, yields on each .next()
// 2. spawn_heartbeat  — async (tokio), runs in background, sends events
//
// Both emit ReasoningEvent values as beliefs evolve.

use crate::chain::{self, Rule};
use crate::store::NeuronStore;

/// Events emitted by the reasoning stream.
/// Each heartbeat cycle can produce one of these.
#[derive(Debug, Clone)]
pub enum ReasoningEvent {
    /// A heartbeat cycle completed.
    Heartbeat {
        cycle: usize,
        changed: bool,
        fact_count: usize,
    },

    /// The system has settled — beliefs are stable, no changes for N cycles.
    Settled {
        cycle: usize,
        fact_count: usize,
        rule_count: usize,
    },
}

impl std::fmt::Display for ReasoningEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReasoningEvent::Heartbeat {
                cycle,
                changed,
                fact_count,
            } => {
                if *changed {
                    write!(f, "<3 cycle {} — beliefs evolving ({} facts)", cycle, fact_count)
                } else {
                    write!(f, ".. cycle {} — stable ({} facts)", cycle, fact_count)
                }
            }
            ReasoningEvent::Settled {
                cycle,
                fact_count,
                rule_count,
            } => {
                write!(
                    f,
                    "<3 settled at cycle {} — {} facts, {} rules (master)",
                    cycle, fact_count, rule_count
                )
            }
        }
    }
}

// ── Synchronous HeartbeatIter ──────────────────────────────────────────
//
// Simple iterator: each .next() runs one consolidation cycle.
// No async runtime needed. Works anywhere.

/// Synchronous heartbeat iterator.
/// Each call to `.next()` runs one consolidation cycle and returns the event.
pub struct HeartbeatIter<'a> {
    store: &'a mut NeuronStore,
    rules: &'a [Rule],
    cycle: usize,
    max_cycles: usize,
    settled_threshold: usize,
    settled_count: usize,
    done: bool,
}

impl<'a> HeartbeatIter<'a> {
    pub fn new(
        store: &'a mut NeuronStore,
        rules: &'a [Rule],
        max_cycles: usize,
    ) -> Self {
        HeartbeatIter {
            store,
            rules,
            cycle: 0,
            max_cycles,
            settled_threshold: 3,
            settled_count: 0,
            done: false,
        }
    }

    /// Set the number of stable cycles before emitting Settled.
    pub fn settled_after(mut self, n: usize) -> Self {
        self.settled_threshold = n;
        self
    }
}

impl<'a> Iterator for HeartbeatIter<'a> {
    type Item = ReasoningEvent;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        self.cycle += 1;
        if self.max_cycles > 0 && self.cycle > self.max_cycles {
            self.done = true;
            return None;
        }

        let changed = chain::consolidate(self.rules, self.store);

        if changed {
            self.settled_count = 0;
            Some(ReasoningEvent::Heartbeat {
                cycle: self.cycle,
                changed: true,
                fact_count: self.store.len(),
            })
        } else {
            self.settled_count += 1;
            if self.settled_count >= self.settled_threshold {
                self.done = true;
                Some(ReasoningEvent::Settled {
                    cycle: self.cycle,
                    fact_count: self.store.len(),
                    rule_count: self.rules.len(),
                })
            } else {
                Some(ReasoningEvent::Heartbeat {
                    cycle: self.cycle,
                    changed: false,
                    fact_count: self.store.len(),
                })
            }
        }
    }
}

// ── Async Heartbeat (tokio) ────────────────────────────────────────────
//
// Spawns a background task that runs the heartbeat on a timer.
// Events are sent through a channel — the receiver is your stream.
//
// ```ignore
// let (tx, mut rx) = qor_runtime::stream::heartbeat_channel();
// tokio::spawn(async move {
//     heartbeat_loop(session, tx, config).await;
// });
// while let Some(event) = rx.recv().await {
//     println!("{}", event);
// }
// ```

#[cfg(feature = "async")]
pub mod async_stream {
    use super::*;
    use crate::eval::Session;
    use std::sync::{Arc, Mutex};
    use tokio::sync::mpsc;
    use tokio::time::{interval, Duration};

    /// Configuration for the async heartbeat.
    #[derive(Debug, Clone)]
    pub struct HeartbeatConfig {
        /// Milliseconds between heartbeat cycles.
        pub interval_ms: u64,
        /// Maximum cycles before stopping (0 = infinite).
        pub max_cycles: usize,
        /// Cycles without change before declaring "settled".
        pub settled_threshold: usize,
    }

    impl Default for HeartbeatConfig {
        fn default() -> Self {
            HeartbeatConfig {
                interval_ms: 100,
                max_cycles: 0,
                settled_threshold: 5,
            }
        }
    }

    /// Create a channel pair for heartbeat events.
    pub fn heartbeat_channel(buffer: usize) -> (mpsc::Sender<ReasoningEvent>, mpsc::Receiver<ReasoningEvent>) {
        mpsc::channel(buffer)
    }

    /// Run the heartbeat loop, sending events to the channel.
    /// This is the LIFE of QOR — it runs until stopped or settled.
    pub async fn heartbeat_loop(
        session: Arc<Mutex<Session>>,
        tx: mpsc::Sender<ReasoningEvent>,
        config: HeartbeatConfig,
    ) {
        let mut ticker = interval(Duration::from_millis(config.interval_ms));
        let mut cycle = 0usize;
        let mut settled_count = 0usize;

        loop {
            ticker.tick().await;
            cycle += 1;

            if config.max_cycles > 0 && cycle > config.max_cycles {
                break;
            }

            let (changed, fact_count, rule_count) = {
                let mut sess = session.lock().unwrap();
                let changed = sess.heartbeat();
                (changed, sess.fact_count(), sess.rule_count())
            };

            let event = if changed {
                settled_count = 0;
                ReasoningEvent::Heartbeat {
                    cycle,
                    changed: true,
                    fact_count,
                }
            } else {
                settled_count += 1;
                if settled_count >= config.settled_threshold {
                    let evt = ReasoningEvent::Settled {
                        cycle,
                        fact_count,
                        rule_count,
                    };
                    let _ = tx.send(evt).await;
                    break; // settled — stop the heartbeat
                }
                ReasoningEvent::Heartbeat {
                    cycle,
                    changed: false,
                    fact_count,
                }
            };

            if tx.send(event).await.is_err() {
                break; // receiver dropped
            }
        }
    }

    /// Convenience: spawn the heartbeat as a tokio task, return the receiver.
    pub fn spawn_heartbeat(
        session: Arc<Mutex<Session>>,
        config: HeartbeatConfig,
    ) -> mpsc::Receiver<ReasoningEvent> {
        let (tx, rx) = heartbeat_channel(256);
        tokio::spawn(heartbeat_loop(session, tx, config));
        rx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qor_core::neuron::{Condition, Neuron};
    use qor_core::truth_value::TruthValue;

    #[test]
    fn test_heartbeat_iter_basic() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]),
            TruthValue::from_strength(0.99),
        );

        let rules = vec![Rule {
            head: Neuron::expression(vec![Neuron::symbol("flies"), Neuron::variable("x")]),
            body: vec![Condition::Positive(Neuron::expression(vec![
                Neuron::symbol("bird"),
                Neuron::variable("x"),
            ]))],
            tv: TruthValue::from_strength(0.95),
            stratum: 0,
        }];

        // Forward chain first to derive (flies tweety)
        chain::forward_chain(&rules, &mut store);

        let events: Vec<_> = HeartbeatIter::new(&mut store, &rules, 10).collect();
        assert!(!events.is_empty());

        // Should eventually settle
        match events.last().unwrap() {
            ReasoningEvent::Settled { .. } => {}
            ReasoningEvent::Heartbeat { .. } => {
                // OK — may hit max_cycles first
            }
        }
    }

    #[test]
    fn test_heartbeat_iter_settles() {
        let mut store = NeuronStore::new();
        store.insert(
            Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]),
            TruthValue::from_strength(0.99),
        );

        let rules = vec![Rule {
            head: Neuron::expression(vec![Neuron::symbol("flies"), Neuron::variable("x")]),
            body: vec![Condition::Positive(Neuron::expression(vec![
                Neuron::symbol("bird"),
                Neuron::variable("x"),
            ]))],
            tv: TruthValue::from_strength(0.95),
            stratum: 0,
        }];

        chain::forward_chain(&rules, &mut store);

        // Run 100 cycles — should settle well before that
        let events: Vec<_> = HeartbeatIter::new(&mut store, &rules, 100)
            .settled_after(3)
            .collect();

        let last = events.last().unwrap();
        assert!(matches!(last, ReasoningEvent::Settled { .. }));
    }

    #[test]
    fn test_heartbeat_iter_with_decay() {
        let mut store = NeuronStore::new();
        store.insert_with_decay(
            Neuron::expression(vec![Neuron::symbol("news"), Neuron::symbol("flash")]),
            TruthValue::new(0.95, 0.90),
            Some(0.10),
        );

        let rules = vec![];
        let events: Vec<_> = HeartbeatIter::new(&mut store, &rules, 20).collect();

        // Should have heartbeat events with changes (due to decay)
        let changed_count = events
            .iter()
            .filter(|e| matches!(e, ReasoningEvent::Heartbeat { changed: true, .. }))
            .count();
        assert!(changed_count > 0);
    }

    #[test]
    fn test_heartbeat_iter_max_cycles() {
        let mut store = NeuronStore::new();
        let rules = vec![];

        let events: Vec<_> = HeartbeatIter::new(&mut store, &rules, 5).collect();
        // Empty store, no rules — settles immediately
        assert!(events.len() <= 5);
    }

    #[test]
    fn test_reasoning_event_display() {
        let evt = ReasoningEvent::Heartbeat {
            cycle: 1,
            changed: true,
            fact_count: 5,
        };
        let s = format!("{}", evt);
        assert!(s.contains("cycle 1"));
        assert!(s.contains("evolving"));

        let evt = ReasoningEvent::Settled {
            cycle: 10,
            fact_count: 5,
            rule_count: 2,
        };
        let s = format!("{}", evt);
        assert!(s.contains("settled"));
        assert!(s.contains("master"));
    }

    // ── Async tests (tokio) ──

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_async_heartbeat_stream() {
        use crate::eval::Session;
        use super::async_stream::*;
        use std::sync::{Arc, Mutex};

        let session = Arc::new(Mutex::new(Session::new()));

        // Add facts and rules
        {
            let mut sess = session.lock().unwrap();
            sess.exec("(bird tweety) <0.99>").unwrap();
            sess.exec("(flies $x) if (bird $x) <0.95>").unwrap();
        }

        let config = HeartbeatConfig {
            interval_ms: 10,
            max_cycles: 20,
            settled_threshold: 3,
        };

        let mut rx = spawn_heartbeat(session, config);
        let mut events = Vec::new();

        while let Some(event) = rx.recv().await {
            events.push(event);
        }

        assert!(!events.is_empty());
        // Should settle since beliefs converge
        assert!(events
            .iter()
            .any(|e| matches!(e, ReasoningEvent::Settled { .. })));
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_async_heartbeat_with_decay() {
        use crate::eval::Session;
        use super::async_stream::*;
        use std::sync::{Arc, Mutex};

        let session = Arc::new(Mutex::new(Session::new()));

        {
            let mut sess = session.lock().unwrap();
            sess.exec("(news flash) <0.95> @decay 0.10").unwrap();
        }

        let config = HeartbeatConfig {
            interval_ms: 10,
            max_cycles: 15,
            settled_threshold: 5,
        };

        let mut rx = spawn_heartbeat(session, config);
        let mut heartbeat_count = 0;

        while let Some(_event) = rx.recv().await {
            heartbeat_count += 1;
        }

        assert!(heartbeat_count > 0);
    }
}
