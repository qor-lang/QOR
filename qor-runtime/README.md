# qor-runtime

**NeuronStore, Forward/Backward Chaining, Session Evaluator, and Self-Improving Engine for QORlang**

Version: workspace-inherited | License: MIT

---

## 1. Crate Overview

`qor-runtime` is the execution engine for QORlang (Quantified Ontological Reasoning Language). It provides:

- **NeuronStore**: A MORK-inspired prefix-trie-indexed knowledge store with belief revision, temporal decay, and O(k) lookups.
- **Forward/Backward Chaining**: PLN-based continuous reasoning with stratified negation, guards, aggregates, and arithmetic.
- **Session Evaluator**: Persistent REPL state management, multi-perspective reasoning (6D AGI scoring), hypothesis testing, and explanation tracing.
- **Self-Improving Engine**: Heartbeat-driven learning loop with rule invention (genesis), evolutionary mutation, refinement search, sleep/compression, and a scored rule library.
- **Reasoning Streams**: Synchronous and async (tokio) heartbeat iterators for observable reasoning.
- **Binary Knowledge Base**: Triple-indexed fact store (12 bytes/fact) with CRUD operations and QOR conversion.
- **Persistent Store**: O(1) snapshot/clone via HAMT structural sharing (feature-gated: `persistent`).

### Dependencies

| Dependency | Purpose |
|---|---|
| `qor-core` | Neuron enum, TruthValue, Parser, Unify |
| `roaring` 0.10 | RoaringBitmap for O(1) membership in trie nodes |
| `rayon` 1.10 | Parallel scoring in genesis/invent |
| `ahash` 0.8 | Fast hashing for internal HashMaps |
| `im` 15 (optional, `persistent` feature) | Persistent HAMT data structures |
| `tokio` 1 (optional, `async` feature) | Async heartbeat streams |
| `datafrog` 2.0 (optional, `datafrog` feature) | Datalog join engine (stub) |

### Feature Flags

| Feature | Effect |
|---|---|
| `async` | Enables `tokio`-based async heartbeat streams in `stream.rs` |
| `persistent` | Enables `persistent_store.rs` with O(1) clone via `im` crate |
| `datafrog` | Enables `datafrog` dependency (not deeply integrated yet) |

---

## 2. File-by-File Breakdown

---

### `lib.rs`

Module declarations only. Re-exports all submodules.

```rust
pub mod store;
pub mod chain;
pub mod stratify;
pub mod eval;
pub mod stream;
pub mod mutate;
pub mod search;
pub mod memory;
pub mod library;
pub mod sleep;
pub mod heartbeat;
pub mod invent;
pub mod kb;
#[cfg(feature = "persistent")]
pub mod persistent_store;
```

---

### `store.rs` -- NeuronStore (In-Memory Knowledge Store)

The core knowledge store. Uses a MORK-inspired prefix trie for O(k) exact lookups and O(k + fan-out) pattern queries.

#### Public Structs

| Struct | Fields | Description |
|---|---|---|
| `NeuronStore` | `neurons: Vec<StoredNeuron>`, `trie: TrieNode` | In-memory knowledge store with prefix trie indexing. Implements `Clone`, `Default`, `IntoIterator`. |
| `StoreIter<'a>` | `inner: slice::Iter<'a, StoredNeuron>` | Flat iterator over all neurons. Implements `ExactSizeIterator`. |
| `PredicateIter<'a>` | `store`, `indices: Vec<u32>`, `pos` | Trie-accelerated iterator filtered by predicate. Implements `ExactSizeIterator`. |
| `TrieWalker<'a>` | `store`, `stack: Vec<TrieFrame>` | Depth-first trie walker yielding `(path, &StoredNeuron)` pairs. |

#### Public Functions/Methods

| Signature | Description |
|---|---|
| `NeuronStore::new() -> Self` | Create an empty store. |
| `insert(&mut self, neuron: Neuron, tv: TruthValue)` | Insert an asserted fact. Performs belief revision on duplicates. |
| `insert_with_decay(&mut self, neuron: Neuron, tv: TruthValue, decay: Option<f64>)` | Insert with temporal decay rate. |
| `insert_inferred(&mut self, neuron: Neuron, tv: TruthValue)` | Insert a derived fact (from forward chaining). |
| `insert_functional(&mut self, neuron: Neuron, tv: TruthValue)` | Insert for functional predicates (one value per subject). Keeps higher-confidence value on conflict; applies belief revision when new confidence is equal or lower (instead of silently discarding). |
| `find_contradictions(&self) -> Vec<(&StoredNeuron, &StoredNeuron)>` | Find pairs of neurons with same predicate+subject but different values. |
| `query(&self, pattern: &Neuron) -> Vec<&StoredNeuron>` | Pattern query with variables. Trie-accelerated candidate lookup + pattern matching. |
| `contains(&self, neuron: &Neuron) -> bool` | Check existence. O(k) via trie for expressions. |
| `get_exact(&self, neuron: &Neuron) -> Option<&StoredNeuron>` | O(1) exact lookup for ground neurons (no variables). |
| `all(&self) -> &[StoredNeuron]` | All neurons in insertion order. |
| `len(&self) -> usize` | Number of stored neurons. |
| `is_empty(&self) -> bool` | Whether the store is empty. |
| `apply_decay(&mut self) -> bool` | Apply temporal decay to all decaying neurons. Returns true if any confidence changed. |
| `remove_by_predicate(&mut self, predicates: &[&str]) -> usize` | Remove all neurons matching given predicate names. Rebuilds trie. Returns count removed. |
| `remove_inferred_by_predicate(&mut self, predicate: &str) -> usize` | Remove only inferred (non-asserted) neurons matching a predicate name. Keeps asserted facts intact. Rebuilds trie. Returns count removed. |
| `trie_node_count(&self) -> usize` | Number of trie nodes (diagnostics). |
| `iter(&self) -> StoreIter<'_>` | Flat iterator over all neurons. |
| `iter_predicate(&self, predicate: &str) -> PredicateIter<'_>` | Trie-accelerated iterator filtered by first element. |
| `walk_trie(&self) -> TrieWalker<'_>` | Depth-first trie walk yielding (path, neuron) pairs. |
| `count_predicate(&self, predicate: &str) -> usize` | Count neurons matching a predicate (trie-accelerated). |
| `predicates(&self) -> Vec<String>` | All distinct predicates (first elements) in the store. |
| `find_path(&self, from: &str, to: &str, max_hops: usize) -> Option<Vec<(String, String)>>` | BFS graph walk through fact triples to find a path between entities. |

#### Internal Functions

| Function | Description |
|---|---|
| `has_variables(neuron: &Neuron) -> bool` | Check if a neuron contains any variables. Used by `chain.rs`. |
| `neuron_to_insert_path(neuron: &Neuron) -> Option<Vec<String>>` | Extract concrete trie path from a ground neuron. |
| `neuron_to_query_path(neuron: &Neuron) -> Option<Vec<QueryKey>>` | Extract query path (may contain wildcards) from a pattern. |
| `matches_pattern(pattern: &Neuron, target: &Neuron) -> bool` | Pattern matching with variables. |

#### Connections
- **Called by**: `chain.rs` (forward_chain, backward_chain, consolidate, resolve_body_cb), `eval.rs` (Session), `search.rs`, `invent.rs`, `persistent_store.rs`
- **Calls**: `qor_core::neuron`, `qor_core::truth_value`

---

### `chain.rs` -- Forward/Backward Chaining Engine

The reasoning engine. Implements PLN-based forward chaining (continuous reasoning) and backward chaining (goal-directed reasoning).

#### Public Structs

| Struct | Fields | Description |
|---|---|---|
| `Rule` | `head: Neuron`, `body: Vec<Condition>`, `tv: TruthValue`, `stratum: u32` | A rule ready for execution. |
| `RuleFiring` | `rule_head: String`, `derived_count: usize` | Record of a single rule firing (for tracing). |
| `ChainTrace` | `firings: Vec<RuleFiring>`, `total_derived: usize` | Trace of a forward chain execution. |

#### Public Functions

| Signature | Description |
|---|---|
| `Rule::new(head: Neuron, body: Vec<Condition>, tv: TruthValue) -> Self` | Create a rule with default stratum 0. |
| `forward_chain(rules: &[Rule], store: &mut NeuronStore) -> usize` | Run forward chaining to fixed point. Auto-stratifies via Kahn's topological sort. Returns count of NEW facts derived. Max 10 iterations per stratum. Uses MORK-style rule indexing, delta tracking, callback body resolution, and ground-condition shortcuts. Dedup uses `BTreeMap`/`BTreeSet` for deterministic derivation order. |
| `forward_chain_traced(rules: &[Rule], store: &mut NeuronStore) -> (usize, ChainTrace)` | Same as `forward_chain` but records which rules fired and how many facts each derived. |
| `consolidate(rules: &[Rule], store: &mut NeuronStore) -> bool` | Re-derive existing facts to strengthen beliefs (the "heartbeat"). Applies temporal decay first. Returns true if any TV changed significantly (> 0.001). |
| `backward_chain(query: &Neuron, rules: &[Rule], store: &NeuronStore) -> Vec<StoredNeuron>` | Goal-directed reasoning: derive facts matching a query through rules. |
| `resolve_body_for_explain(conditions: &[Condition], bindings: &Bindings, store: &NeuronStore) -> Option<Vec<(Neuron, TruthValue, bool)>>` | Resolve a rule body and return supporting facts for each positive condition. Used by `Session::explain()`. |

#### Internal Functions

| Function | Description |
|---|---|
| `forward_chain_stratum(rules: &[Rule], store: &mut NeuronStore) -> usize` | Fixed-point chaining for a single stratum. |
| `reorder_body(body: &[Condition]) -> Vec<Condition>` | Greedy reorder of rule body conditions for optimal binding propagation. |
| `score_condition(cond, bound, orig_idx) -> (u32, usize, usize)` | Score a condition for reorder priority (lower = earlier). |
| `condition_provides(cond) -> HashSet<String>` | Variables a condition binds for downstream conditions. |
| `resolve_body_cb(conditions, bindings, store, tv, cb)` | MORK-style callback body resolver. Zero intermediate allocations. Handles Positive, Negated, NegatedPresent, Guard, Aggregate, Arithmetic, EndsWith, Lookup conditions. |
| `resolve_body(conditions, bindings, store) -> Vec<(Bindings, TruthValue)>` | Thin Vec wrapper around `resolve_body_cb` for backward_chain/explain. |
| `could_match(head, query) -> bool` | Quick structural check: could a rule head match a query? |
| `matches_query(query, derived) -> bool` | Check if a derived fact matches a query pattern. |

#### Connections
- **Called by**: `eval.rs` (Session::exec, heartbeat, explain, test_hypothesis), `stream.rs` (HeartbeatIter), `search.rs`, `invent.rs`
- **Calls**: `store.rs` (NeuronStore::query, contains, get_exact, insert_inferred, apply_decay), `stratify.rs` (auto_stratify), `qor_core::unify`, `qor_core::neuron`

---

### `eval.rs` -- Session Evaluator & Multi-Perspective Reasoning

The main user-facing API. Manages persistent session state, executes QOR programs, and provides multi-perspective reasoning with 6D AGI scoring.

#### Public Structs

| Struct | Fields | Description |
|---|---|---|
| `Session` | `store: NeuronStore`, `rules: Vec<Rule>`, `consolidation_cycles: usize`, `stored_queries: Vec<Neuron>`, `kb: Option<Arc<KnowledgeBase>>` | Persistent reasoning session. Implements `Clone`, `Default`. |
| `Explanation` | `fact: String`, `tv: TruthValue`, `reason: ExplanationReason` | Why a fact is believed. |
| `ExplanationReason` (enum) | `Asserted` / `Derived { rule: String, from: Vec<Explanation> }` | The reason a fact exists. |
| `HypothesisResult` | `hypothesis: String`, `new_facts: usize`, `confidence: f64`, `facts: Vec<String>` | Result of testing a hypothesis. |
| `QueryResult` | `pattern: String`, `results: Vec<String>` | Result of running a query. |
| `ExecResult` (enum) | `Stored { neuron, derived }` / `RuleAdded { derived }` / `Query(QueryResult)` | Result of executing a single statement. |
| `MetaDomain` | `name: String`, `statements: Vec<Statement>` | A domain of knowledge (one meta file). |
| `DomainInsight` | `domain`, `inferred_count`, `predicates`, `unique_facts`, `all_inferred`, `elapsed_ms` | What one domain perspective saw. |
| `StateVector6D` | `feasibility`, `novelty`, `simplicity`, `ethics`, `consistency`, `information` | 6D state vector from agi.qor. |
| `ScoredInsight` | `fact`, `state: StateVector6D`, `combined_score`, `source_domains`, `accepted` | An insight scored through the 6D framework. Acceptance gate: `ethics > 0.5 && score > 0.50` (score + ethics only; no predicate-name filter). |
| `ReasonResult` | `insights`, `scored_insights`, `consensus_facts`, `total_unique_inferred`, `elapsed_ms` | Combined result of multi-perspective reasoning. |

#### Public Functions/Methods

| Signature | Description |
|---|---|
| `Session::new() -> Self` | Create an empty session. |
| `set_kb(&mut self, kb: Arc<KnowledgeBase>)` | Attach a binary knowledge base (Arc-shared across clones). |
| `kb(&self) -> Option<&KnowledgeBase>` | Read-only KB accessor. |
| `load_entity(&mut self, name: &str)` | Pull all facts about an entity from KB into the NeuronStore. |
| `exec(&mut self, input: &str) -> Result<Vec<ExecResult>, String>` | Parse and execute QOR source text. |
| `exec_statements(&mut self, stmts: Vec<Statement>) -> Result<Vec<ExecResult>, String>` | Execute pre-parsed statements (avoids re-serialization). |
| `exec_statements_with_strata(&mut self, stmts, strata) -> Result<Vec<ExecResult>, String>` | Execute with explicit stratum annotations. |
| `heartbeat(&mut self) -> bool` | Run one consolidation cycle. Returns true if beliefs changed. |
| `fact_count(&self) -> usize` | Number of facts in the store. |
| `rule_count(&self) -> usize` | Number of rules in the session. |
| `consolidation_cycles(&self) -> usize` | How many heartbeat cycles have produced changes. |
| `all_facts(&self) -> &[StoredNeuron]` | All stored facts. |
| `explain(&self, pattern: &Neuron) -> Vec<Explanation>` | Trace WHY a fact is believed (recursive reasoning chain). |
| `rules(&self) -> &[Rule]` | Raw Rule objects (for search/mutation). |
| `rules_as_statements(&self) -> Vec<Statement>` | Convert rules back to Statement::Rule. |
| `clear_turn(&mut self)` | Clear ephemeral chat facts between turns (input, intent, response, etc.). |
| `response_facts(&self) -> Vec<Vec<String>>` | Get response facts as string vectors. |
| `store(&self) -> &NeuronStore` | Read-only store accessor. |
| `store_mut(&mut self) -> &mut NeuronStore` | Mutable store accessor. |
| `remove_by_predicate(&mut self, predicates: &[&str]) -> usize` | Remove facts by predicate names. |
| `test_hypothesis(&mut self, hypothesis: &str) -> Result<HypothesisResult, String>` | Snapshot store, inject hypothesis, chain, measure new derivations, restore. |
| `queries(&self) -> &[Neuron]` | Get stored query patterns. |
| `run_queries(&self) -> Vec<QueryResult>` | Execute all stored queries. |
| `reason(&self, domains: &[MetaDomain], problem: Vec<Statement>, heartbeats: usize) -> ReasonResult` | Multi-perspective reasoning: run problem through N domain lenses in parallel (threaded), combine with 6D AGI scoring. Snapshots base fact keys before spawning workers and collects all new facts (inferred or not in base) via `Arc<HashSet<String>>` for thread-safe sharing. |
| `Session::load_meta_domains(meta_dir: &Path) -> Vec<MetaDomain>` | Load meta domain .qor files from a directory (skips agi.qor). |
| `StateVector6D::combined_score(&self) -> f64` | Project 6D to 1D: `0.35*f + 0.20*c + 0.15*s + 0.15*n + 0.10*i + 0.05*e`. |
| `StateVector6D::chain(&self, other: &StateVector6D) -> StateVector6D` | Tensor product for chain propagation (dimension-specific operators). |
| `ReasonResult::print_report(&self)` | Print full reasoning report to stderr. |
| `run(source: &str) -> Result<Vec<QueryResult>, String>` | Batch mode: parse, execute, and return query results. |

#### Connections
- **Called by**: `heartbeat.rs`, `search.rs`, `invent.rs`, `qor-cli`, `qor-agent`
- **Calls**: `chain.rs` (forward_chain, consolidate, resolve_body_for_explain), `store.rs` (NeuronStore), `kb.rs` (KnowledgeBase), `qor_core::parser`

---

### `stratify.rs` -- Auto-Stratification via Kahn's Topological Sort

Assigns strata to rules so that negated derived predicates reach fixed-point before negation checks them.

#### Public Structs

| Struct | Fields | Description |
|---|---|---|
| `StratifyError` | `cycle_preds: Vec<String>` | Error when rules contain a negation cycle (unstratifiable). |

#### Public Functions

| Signature | Description |
|---|---|
| `auto_stratify(rules: &mut [Rule]) -> Result<u32, StratifyError>` | Compute strata using Kahn's topological sort on the negation dependency graph. Respects manually-set strata (non-zero). Propagates through positive dependencies. Returns max stratum or cycle error. |

#### Internal Functions

| Function | Description |
|---|---|
| `head_predicate(rule) -> Option<String>` | Extract head predicate name. |
| `positive_predicates(rule) -> Vec<String>` | Extract predicates from positive body conditions. |
| `negated_predicates(rule) -> Vec<String>` | Extract predicates from negated body conditions. |

#### Connections
- **Called by**: `chain.rs` (forward_chain calls auto_stratify before chaining)
- **Calls**: `chain.rs` (Rule struct)

---

### `heartbeat.rs` -- Continuous Learning Engine

The living system. Pulses periodically: wake (analyze failures), create (generate candidates), test (verify), sleep (compress), save (persist).

#### Public Structs

| Struct | Fields | Description |
|---|---|---|
| `PulseResult` | `new_solves`, `library_size`, `total_solved`, `mutations_tried`, `pulse_duration_ms`, `web_facts_extracted`, `web_pages_crawled` | Result of a single heartbeat pulse. |
| `TaskData` | `id: String`, `training_inputs: Vec<Vec<Statement>>`, `training_outputs: Vec<Vec<Statement>>`, `target_pred: String` | Generic task data (domain-specific conversion done by caller). |
| `Heartbeat` | `library: RuleLibrary`, `history: RunHistory`, `sleep_interval: usize` | The heartbeat engine. |

#### Public Functions/Methods

| Signature | Description |
|---|---|
| `Heartbeat::new(brain_path: PathBuf) -> Self` | Create a new heartbeat engine. |
| `Heartbeat::load(brain_path: &Path) -> Self` | Load existing state from brain directory. |
| `pulse(&mut self, tasks: &[TaskData], rules_qor: &str, web_seeds: &[String]) -> PulseResult` | Run one heartbeat pulse. Increments cycle, parses web seeds, analyzes failures, runs refinement search, sleep cycle (every Nth), saves state. |
| `total_solved(&self) -> usize` | Total solved task count. |
| `library_size(&self) -> usize` | Number of rules in library. |

#### Connections
- **Called by**: `qor-cli` (`qor heartbeat` command)
- **Calls**: `library.rs` (RuleLibrary), `memory.rs` (RunHistory), `search.rs` (refinement_search), `sleep.rs` (sleep_cycle), `eval.rs` (Session), `qor_core::parser`

---

### `invent.rs` -- Rule Invention Engine (Genesis)

Creates QOR rules from scratch using training examples. Implements a Creativity Formula: Decompose, Contradict, Propose, Evaluate, Combine, Evolve.

#### Public Structs

| Struct | Fields | Description |
|---|---|---|
| `Candidate` | `rule_text: String`, `score: f64`, `source: String` | A candidate rule with fitness score. |
| `WorkerResult` | `candidates: Vec<Candidate>`, `worker_id: usize` | Result from one parallel worker thread. |
| `WorkerStrategy` (enum) | `BasicTransforms`, `PositionalTransforms`, `DerivedFacts`, `MemoryRecall`, `RandomExplore(u64)`, `MultiJoinArithmetic` | Worker focus strategy for parallel search. |

#### Public Functions

| Signature | Description |
|---|---|
| `genesis(base_session: &Session, training_inputs, training_outputs, time_budget_ms, library: Option<&mut RuleLibrary>) -> Vec<Candidate>` | Invent rules from scratch. 7 steps: recall, decompose, contradict, propose (patterns, multi-join, arithmetic, aggregate, conditional-split, spatial, observation-driven), evaluate (parallel rayon + PrebuiltScorer), combine, evolve (guided mutation + SCAMPER), remember. Returns top 10 candidates. |
| `genesis_swarm(base_session, training_inputs, training_outputs, time_budget_ms, library, num_workers) -> Vec<Candidate>` | Parallel swarm: N workers explore different strategy slices. First worker scoring >= 0.95 cancels all others. Uses scoped threads. |
| `optimal_worker_count() -> usize` | Detect optimal worker count based on available CPU cores (1-5 workers). |

#### Internal Types & Functions (selected)

| Item | Description |
|---|---|
| `FactProfile` | Profile of any fact set: predicates, constants, symbols. |
| `GeneralContradiction` | Difference between two fact sets: added, removed, value maps, new predicates, count changes. |
| `ScoreDetail` | Detailed score with failure analysis (wrong keys, missing keys, pattern). |
| `PrebuiltScorer` | Pre-builds per-pair sessions once, clones store only for each candidate (22% speedup). Uses `remove_inferred_by_predicate()` to clear only inferred target-pred facts (keeping asserted inputs intact for same-predicate puzzles). Scoring uses precision-recall geometric mean (`sqrt(precision * recall)`), consistent with solve.rs. No simplicity bonus -- pure accuracy. |
| `profile_facts()` | Profile fact shapes (predicates, arities, value types). |
| `find_general_contradiction()` | Compare input vs output fact sets. |
| `propose_from_patterns()` | Generate candidates from discovered patterns. |
| `propose_multi_join()` | Multi-body join proposals. |
| `propose_arithmetic_combos()` | Arithmetic combination proposals. |
| `propose_aggregate_conditions()` | Aggregate condition proposals. |
| `propose_conditional_split()` | Conditional split proposals. |
| `propose_spatial()` | Spatial relationship proposals. |
| `propose_from_observations()` | Observation-driven proposals. |
| `guided_mutate()` | Failure-pattern-guided mutations. |
| `general_scamper_mutate()` | SCAMPER-style mutations. |
| `combine_top_rules_fast()` | Merge top partial rules for better coverage. |
| `worker_genesis()` | Individual worker thread for parallel swarm. |

#### Connections
- **Called by**: `qor-cli` (puzzle solving), `qor-agent`
- **Calls**: `eval.rs` (Session), `library.rs` (RuleLibrary), `mutate.rs` (generate_mutations, generate_context_mutations, rule_to_qor), `chain.rs` (Rule), `qor_core::parser`, `rayon`

---

### `mutate.rs` -- Rule Mutation Operators

Operates on QOR Rule AST to produce candidate variations. Domain agnostic.

#### Public Structs/Enums

| Type | Description |
|---|---|
| `Mutation` (enum) | `RemoveCondition(usize)`, `ChangeGuardValue { index, delta }`, `ChangeIntConst { index, old_val, new_val }`, `SwapGuardOp { index, new_op }`, `AddGuard { op, var, value }`, `AddPositiveCondition(Neuron)`, `AddNegatedCondition(Neuron)`, `SpecializeHeadVar { var_name, value }`, `AddArithmetic { op, lhs, rhs, result_var }`, `AddAggregate { op, pattern, bind_var, result_var }` |

#### Public Functions

| Signature | Description |
|---|---|
| `mutate_rule(rule: &Rule, mutation: &Mutation) -> Option<Rule>` | Apply a mutation to a rule, returning a new rule. Returns None if mutation is invalid. |
| `generate_mutations(rule: &Rule) -> Vec<Rule>` | Generate all reasonable single-step structural mutations: remove condition, change guard values (+-1/2/5), swap guard ops, change integer constants (to other constants in rule or +-1), mutate head integer constants. |
| `generate_context_mutations(rule: &Rule, known_predicates: &[(String, usize)], known_values: &[i64]) -> Vec<Rule>` | Context-aware mutations using known predicates and values: specialize head vars, add positive/negated conditions from known predicates, add guard conditions with known values, add arithmetic between variable pairs. |
| `extract_variables(rule: &Rule) -> Vec<String>` | Extract all variable names from a rule (sorted, deduped). |
| `rule_to_qor(rule: &Rule) -> String` | Format a Rule back to QOR text for parsing/display. |

#### Internal Functions

| Function | Description |
|---|---|
| `extract_int_consts(cond) -> Option<Vec<i64>>` | Extract integer constants from a condition. |
| `collect_ints(n, out)` | Collect ints into HashSet. |
| `collect_ints_vec(n, out)` | Collect ints into Vec. |
| `rewrite_int_in_condition(cond, old, new) -> Option<Condition>` | Rewrite an integer constant in a condition. |
| `rewrite_int_in_neuron(n, old, new) -> Neuron` | Rewrite an integer constant in a neuron. |
| `specialize_var(n, var, val) -> Neuron` | Replace a variable with a constant. |
| `specialize_var_in_condition(cond, var, val) -> Condition` | Replace a variable with a constant in a condition. |
| `collect_vars(n, out)` | Collect variable names from a neuron. |

#### Connections
- **Called by**: `search.rs` (refinement_search), `invent.rs` (genesis evolution)
- **Calls**: `chain.rs` (Rule)

---

### `search.rs` -- Refinement Search Loop

Time-budgeted evolutionary search for rules. Takes near-miss rules, mutates, scores against training data.

#### Public Structs

| Struct | Fields | Description |
|---|---|---|
| `SearchResult` | `solutions: Vec<ScoredRule>`, `near_misses: Vec<ScoredRule>`, `mutations_tried: usize`, `elapsed_ms: u64` | Result of a search run. |
| `ScoredRule` | `rule: Rule`, `score: f64`, `qor_text: String` | A rule with its training score. |

#### Public Functions

| Signature | Description |
|---|---|
| `score_rule_on_training(rule, training_inputs, expected_outputs, target_pred, base_session) -> f64` | Score a rule against training pairs. Clones session, injects rule + input, chains, compares predicted vs expected facts by key matching. Returns fraction of expected facts matched (averaged across pairs). |
| `refinement_search(seed_rules, training_inputs, expected_outputs, target_pred, base_session, time_budget_ms) -> SearchResult` | Evolutionary refinement: score seeds, iteratively mutate top-5, keep top-20 population, up to 50 generations or time budget. Stops on perfect solution (score >= 0.999). |

#### Internal Functions

| Function | Description |
|---|---|
| `extract_facts_by_pred(session, pred) -> HashSet<String>` | Extract fact keys from session by predicate. |
| `extract_facts_from_stmts(stmts, pred) -> HashSet<String>` | Extract fact keys from statements by predicate. |
| `fact_key(parts: &[Neuron]) -> String` | Convert fact parts to a comparable key string. |

#### Connections
- **Called by**: `heartbeat.rs` (pulse)
- **Calls**: `eval.rs` (Session), `mutate.rs` (generate_mutations, rule_to_qor), `chain.rs` (Rule)

---

### `memory.rs` -- Run History / Experience Memory

Records every task attempt. Domain agnostic.

#### Public Structs

| Struct | Fields | Description |
|---|---|---|
| `TaskRun` | `task_id`, `detected_transforms: Vec<String>`, `selected_transform`, `rules_fired: Vec<String>`, `correct: bool`, `accuracy: f64`, `total_items`, `wrong_items` | Record of a single task attempt. |
| `RunHistory` | `runs: Vec<TaskRun>`, `solved_ids: HashSet<String>`, `path: PathBuf`, `cycle_count: usize` | Persistent run history across sessions. |

#### Public Functions/Methods

| Signature | Description |
|---|---|
| `RunHistory::new(path: PathBuf) -> Self` | Create empty history. |
| `RunHistory::load(path: &Path) -> Self` | Load from pipe-delimited file. |
| `record(&mut self, run: TaskRun)` | Record a new task run. |
| `all_runs(&self) -> &[TaskRun]` | Get all runs. |
| `successes(&self) -> Vec<&TaskRun>` | Get successful runs. |
| `failures(&self) -> Vec<&TaskRun>` | Get failed runs. |
| `failure_patterns(&self) -> HashMap<String, Vec<&TaskRun>>` | Group failures by selected transform. |
| `is_solved(&self, task_id: &str) -> bool` | Check if a task has been solved. |
| `mark_solved(&mut self, task_id: &str)` | Mark a task as solved. |
| `total_solved(&self) -> usize` | Total solved count. |
| `increment_cycle(&mut self)` | Increment cycle count. |
| `cycle_count(&self) -> usize` | Get cycle count. |
| `save(&self)` | Save to pipe-delimited file. Format: `task_id|transforms|selected|OK/FAIL|accuracy|total|wrong`. |

#### Connections
- **Called by**: `heartbeat.rs`, `sleep.rs`
- **Calls**: None (standalone data structure)

---

### `library.rs` -- Scored Rule Library

Persistent library of learned rules with usage tracking and Occam's razor pruning.

#### Public Structs/Enums

| Type | Fields | Description |
|---|---|---|
| `RuleSource` (enum) | `HandWritten`, `Induced`, `Composed`, `Mutated`, `Template` | Origin of a rule. |
| `ScoredRule` | `rule_text`, `source: RuleSource`, `times_fired`, `times_correct`, `created_at: u64`, `last_used: u64` | A rule with usage statistics. |
| `PruneConfig` | `max_rules: usize` (500), `min_accuracy: f64` (0.1), `unused_threshold: usize` (100) | Configuration for library pruning. |
| `RuleLibrary` | `rules: Vec<ScoredRule>`, `path: PathBuf`, `current_cycle: u64` | Persistent rule library. |

#### Public Functions/Methods

| Signature | Description |
|---|---|
| `ScoredRule::accuracy(&self) -> f64` | `times_correct / times_fired` (0.0 if never fired). |
| `RuleLibrary::new(path: PathBuf) -> Self` | Create empty library. |
| `RuleLibrary::load(dir: &Path) -> Self` | Load from directory of .qor files. |
| `add(&mut self, rule_text: String, source: RuleSource)` | Add rule (deduplicates). |
| `record_firing(&mut self, rule_text: &str, correct: bool)` | Track rule usage. |
| `all_rules(&self) -> &[ScoredRule]` | All rules. |
| `len(&self) -> usize` | Rule count. |
| `is_empty(&self) -> bool` | Check if empty. |
| `parse_rules(&self) -> Vec<Rule>` | Parse all rules into executable Rule objects. |
| `set_cycle(&mut self, cycle: u64)` | Set current cycle for recency tracking. |
| `prune(&mut self, config: &PruneConfig)` | Prune: remove low-accuracy (except hand-written and new), remove unused (except hand-written and ever-correct), cap at max_rules sorted by source/accuracy/recency. |
| `save(&self)` | Save to `learned.qor` in the library directory. Skips hand-written rules. Logs filesystem errors to stderr instead of silently ignoring them. |

#### Connections
- **Called by**: `heartbeat.rs`, `sleep.rs`, `invent.rs`
- **Calls**: `chain.rs` (Rule), `qor_core::parser`

---

### `sleep.rs` -- Pattern Extraction & Compression (Sleep Phase)

Analyzes successful traces, finds common patterns, composes reusable library rules. Inspired by DreamCoder/Stitch.

#### Public Structs

| Struct | Fields | Description |
|---|---|---|
| `SleepResult` | `patterns_found`, `rules_composed`, `rules_pruned`, `library_size_before`, `library_size_after` | Result of a sleep cycle. |

#### Public Functions

| Signature | Description |
|---|---|
| `sleep_cycle(history: &RunHistory, library: &mut RuleLibrary) -> SleepResult` | Run a sleep cycle: collect successful traces, find common sub-sequences, compose meta-rules (transform-success correlations, co-occurrence rules), add high-confidence (>0.6) rules to library, prune library. |

#### Internal Functions

| Function | Description |
|---|---|
| `find_common_patterns(successes) -> Vec<CommonPattern>` | Group successes by transform, count co-occurring detections. |
| `compose_meta_rules(patterns, history) -> Vec<(String, f64)>` | Build meta-rules from patterns (e.g., `(rule-confidence reflect-h 0.80)`). |
| `confidence_from_count(count) -> f64` | PLN w2c formula: `w / (w + k)`, capped at 0.95. |

#### Connections
- **Called by**: `heartbeat.rs` (pulse, every Nth cycle)
- **Calls**: `library.rs` (RuleLibrary::add, prune), `memory.rs` (RunHistory)

---

### `stream.rs` -- QOR Reasoning Streams

Observable heartbeat: synchronous iterator and async (tokio) background task.

#### Public Structs/Enums

| Type | Fields | Description |
|---|---|---|
| `ReasoningEvent` (enum) | `Heartbeat { cycle, changed, fact_count }` / `Settled { cycle, fact_count, rule_count }` | Events emitted by reasoning stream. Implements `Display`. |
| `HeartbeatIter<'a>` | `store`, `rules`, `cycle`, `max_cycles`, `settled_threshold`, `settled_count`, `done` | Synchronous heartbeat iterator. |

#### Public Functions (sync)

| Signature | Description |
|---|---|
| `HeartbeatIter::new(store, rules, max_cycles) -> Self` | Create sync heartbeat iterator. |
| `HeartbeatIter::settled_after(self, n: usize) -> Self` | Set stable cycles before emitting Settled. |

#### Public Functions (async, feature `async`)

| Signature | Description |
|---|---|
| `async_stream::HeartbeatConfig` | Config: `interval_ms` (100), `max_cycles` (0=infinite), `settled_threshold` (5). |
| `heartbeat_channel(buffer) -> (Sender, Receiver)` | Create tokio mpsc channel pair. |
| `heartbeat_loop(session: Arc<Mutex<Session>>, tx, config)` | Async heartbeat loop. Sends events through channel. Stops on settled or receiver drop. |
| `spawn_heartbeat(session, config) -> Receiver` | Spawn heartbeat as tokio task, return receiver. |

#### Connections
- **Called by**: `qor-cli` (think --watch), user code
- **Calls**: `chain.rs` (consolidate), `store.rs` (NeuronStore), `eval.rs` (Session, for async)

---

### `kb.rs` -- Binary Knowledge Base

Triple-indexed fact store. Binary format: 12 bytes per fact (`u32 subject | u16 predicate | u32 object | u8 trust | u8 plausibility`).

#### Public Structs

| Struct | Fields | Description |
|---|---|---|
| `Fact` | `subject: u32`, `predicate: u16`, `object: u32`, `trust: u8`, `plausibility: u8` | A single binary triple. |
| `FactWithNames` | `subject: String`, `predicate: String`, `object: String`, `trust: f32`, `plausibility: f32` | Resolved fact for display. |
| `KnowledgeBase` | `facts: Vec<Fact>`, `by_predicate`, `by_subject`, `by_object` (all `HashMap`), `deleted: HashSet`, `next_entity: u32`, `next_predicate: u16`, `entity_names/ids`, `predicate_names/ids` (all `HashMap`) | Triple-indexed fact store with tombstone deletion. |

#### Public Functions/Methods

| Signature | Description |
|---|---|
| `Fact::trust_f32(&self) -> f32` | Trust as float (0.0-1.0). |
| `Fact::plausibility_f32(&self) -> f32` | Plausibility as float. |
| `Fact::confidence(&self) -> f32` | `trust * plausibility`. |
| `FactWithNames::confidence(&self) -> f32` | `trust * plausibility`. |
| `FactWithNames::to_qor_string(&self) -> String` | Format as QOR fact string. |
| `FactWithNames::to_statement(&self) -> Statement` | Convert to parsed QOR Statement. |
| `KnowledgeBase::new() -> Self` | Create empty KB. |
| `KnowledgeBase::load(dir: &Path) -> io::Result<Self>` | Load from directory (triples.bin + entities.tsv + predicates.tsv + *_triples.bin). |
| `len(&self) -> usize` | Total facts (including deleted). |
| `is_empty(&self) -> bool` | Whether empty. |
| `about(&self, name: &str) -> Vec<FactWithNames>` | All facts about an entity, sorted by confidence. |
| `get(&self, subject, predicate) -> Option<(String, f32, f32)>` | Lookup (subject, predicate) -> best (object, trust, plausibility). |
| `query(&self, predicate, object) -> Vec<(String, f32, f32)>` | Who has (predicate = object)? |
| `reverse(&self, object, predicate) -> Vec<(String, f32, f32)>` | What points to entity via predicate? |
| `query_min_confidence(&self, predicate, object, min_t, min_p) -> Vec<(String, f32, f32)>` | Query with minimum confidence threshold. |
| `neighbors(&self, name, depth) -> Vec<FactWithNames>` | Graph neighbors within N hops. |
| `to_qor_facts(&self, name: &str) -> Vec<Statement>` | Convert all facts about entity to QOR Statements. |
| `lookup_statement(&self, subject, predicate) -> Option<Statement>` | Lookup as QOR Statement. |
| `stats(&self) -> String` | Human-readable stats string. |
| `register_entity(&mut self, name: &str) -> u32` | Register entity name, return ID (case-insensitive dedup). |
| `register_predicate(&mut self, name: &str) -> u16` | Register predicate name, return ID. |
| `upsert(&mut self, subject, predicate, object, trust, plausibility) -> usize` | Upsert by IDs. |
| `upsert_named(&mut self, subject, predicate, object, trust, plausibility) -> usize` | Upsert by names (auto-registers). |
| `insert(&mut self, subject, predicate, object, trust, plausibility) -> usize` | Insert new fact (no dedup). |
| `remove(&mut self, subject, predicate, object) -> usize` | Remove exact match. |
| `remove_by_pred(&mut self, subject, predicate) -> usize` | Remove all (subject, predicate, *). |
| `remove_named(&mut self, subject, predicate, object) -> usize` | Remove by names. |
| `update_confidence(&mut self, subject, predicate, object, trust, plausibility) -> bool` | Update trust/plausibility. |
| `save(&self, dir: &Path) -> io::Result<()>` | Save to disk (merges source files into single triples.bin). |
| `live_count(&self) -> usize` | Non-deleted fact count. |

#### Connections
- **Called by**: `eval.rs` (Session::set_kb, load_entity)
- **Calls**: `qor_core::neuron`, `qor_core::truth_value`

---

### `persistent_store.rs` -- O(1) Snapshot Store (feature: `persistent`)

Drop-in replacement for NeuronStore using `im` crate's HAMT for O(1) clone/snapshot via structural sharing.

#### Public Structs

| Struct | Description |
|---|---|
| `PersistentStore` | `neurons: ImVector<StoredNeuron>`, `trie: PersistentTrieNode`. Implements `Clone` (O(1)), `Default`. |

#### Public Functions/Methods

| Signature | Description |
|---|---|
| `PersistentStore::new() -> Self` | Create empty store. |
| `snapshot(&self) -> Self` | O(1) snapshot (the whole point of this module). |
| `insert(&mut self, neuron, tv)` | Insert asserted fact with belief revision. |
| `insert_with_decay(&mut self, neuron, tv, decay)` | Insert with decay. |
| `insert_inferred(&mut self, neuron, tv)` | Insert derived fact. |
| `query(&self, pattern) -> Vec<&StoredNeuron>` | Pattern query. |
| `contains(&self, neuron) -> bool` | Existence check. |
| `get_exact(&self, neuron) -> Option<&StoredNeuron>` | Exact lookup. |
| `len(&self) -> usize` | Count. |
| `is_empty(&self) -> bool` | Empty check. |
| `from_regular(store: &NeuronStore) -> Self` | Convert from regular NeuronStore. |
| `to_regular(&self) -> NeuronStore` | Convert back to regular NeuronStore. |

#### Connections
- **Called by**: User code when O(1) snapshots are needed (hypothesis testing)
- **Calls**: `store.rs` (has_variables, NeuronStore for conversion)

---

## 3. Connection Diagram

```
                              qor-core
                    (Neuron, TruthValue, Parser, Unify)
                                 |
                    +============|=============+
                    |        qor-runtime       |
                    |                          |
    +----------+    |    +------------------+  |
    | lib.rs   |----+--->| store.rs         |  |
    | (mods)   |    |    | NeuronStore      |  |
    +----------+    |    +--------+---------+  |
                    |             |             |
                    |    +--------v---------+  |
                    |    | chain.rs         |  |
                    |    | forward_chain    |<-+---- stratify.rs (auto_stratify)
                    |    | backward_chain   |  |
                    |    | consolidate      |  |
                    |    +--------+---------+  |
                    |             |             |
                    |    +--------v---------+  |
                    |    | eval.rs          |  |
                    |    | Session          |  |
                    |    | reason()         |<-+---- kb.rs (KnowledgeBase)
                    |    +--+---------+----+   |
                    |       |         |        |
              +-----+------+    +----+------+  |
              |                  |           |  |
     +--------v------+  +-------v------+ +--v--v--------+
     | stream.rs     |  | heartbeat.rs | | invent.rs    |
     | HeartbeatIter |  | Heartbeat    | | genesis      |
     | async_stream  |  | pulse()      | | genesis_swarm|
     +---------------+  +------+-------+ +------+-------+
                               |                 |
                        +------v-------+  +------v-------+
                        | search.rs    |  | mutate.rs    |
                        | refinement   |  | mutations    |
                        | _search()    |  | rule_to_qor  |
                        +------+-------+  +--------------+
                               |
                    +----------+----------+
                    |                     |
              +-----v------+     +-------v------+
              | memory.rs  |     | library.rs   |
              | RunHistory |     | RuleLibrary  |
              +-----+------+     +------+-------+
                    |                    |
              +-----v--------------------v------+
              | sleep.rs                        |
              | sleep_cycle (pattern extraction)|
              +---------------------------------+

     +-------------------+
     | persistent_store.rs  (feature: persistent)
     | PersistentStore      mirrors NeuronStore API
     | O(1) clone via HAMT
     +-------------------+
```

### Key Call Chains

1. **`Session::exec()` -> `chain::forward_chain()` -> `stratify::auto_stratify()` -> `store::NeuronStore` methods**

2. **`Session::heartbeat()` -> `chain::consolidate()` -> `store::apply_decay()` + `resolve_body_cb()` + `insert_inferred()`**

3. **`Heartbeat::pulse()` -> `search::refinement_search()` -> `mutate::generate_mutations()` -> `search::score_rule_on_training()` -> `Session::exec()` / `Session::exec_statements()`**

4. **`genesis()` -> `propose_*()` functions -> `PrebuiltScorer::score()` -> `combine_top_rules_fast()` -> `guided_mutate()` / `general_scamper_mutate()` -> `library::RuleLibrary::add()`**

5. **`Session::reason()` -> spawns N threads, each clones Session + loads domain meta + heartbeats -> collects DomainInsights -> 6D AGI scoring -> ReasonResult**

6. **`Session::explain()` -> `chain::resolve_body_for_explain()` -> recursively traces supporting facts**

7. **`sleep_cycle()` -> `RunHistory::successes()` -> `find_common_patterns()` -> `compose_meta_rules()` -> `RuleLibrary::add()` + `prune()`**

---

## 4. Public API Summary

### Core Types
- `NeuronStore` -- trie-indexed knowledge store
- `Rule` -- head + body + truth value + stratum
- `Session` -- persistent REPL state with forward chaining
- `KnowledgeBase` -- binary triple store

### Reasoning
- `forward_chain(rules, store) -> usize` -- derive new facts
- `backward_chain(query, rules, store) -> Vec<StoredNeuron>` -- goal-directed
- `consolidate(rules, store) -> bool` -- strengthen beliefs
- `auto_stratify(rules) -> Result<u32, StratifyError>` -- negation safety

### Evaluation
- `Session::exec(input) -> Result<Vec<ExecResult>>` -- execute QOR
- `Session::heartbeat() -> bool` -- one consolidation cycle
- `Session::explain(pattern) -> Vec<Explanation>` -- trace reasoning
- `Session::test_hypothesis(hyp) -> Result<HypothesisResult>` -- what-if
- `Session::reason(domains, problem, heartbeats) -> ReasonResult` -- multi-perspective 6D
- `run(source) -> Result<Vec<QueryResult>>` -- batch mode

### Self-Improvement
- `Heartbeat::pulse(tasks, rules_qor, web_seeds) -> PulseResult` -- learn cycle
- `genesis(session, inputs, outputs, budget, library) -> Vec<Candidate>` -- invent rules
- `genesis_swarm(session, inputs, outputs, budget, library, workers) -> Vec<Candidate>` -- parallel invention
- `refinement_search(seeds, inputs, outputs, pred, session, budget) -> SearchResult` -- evolve rules
- `sleep_cycle(history, library) -> SleepResult` -- compress knowledge

### Mutation
- `mutate_rule(rule, mutation) -> Option<Rule>` -- apply one mutation
- `generate_mutations(rule) -> Vec<Rule>` -- all structural mutations
- `generate_context_mutations(rule, predicates, values) -> Vec<Rule>` -- context-aware mutations
- `rule_to_qor(rule) -> String` -- serialize rule to QOR text

### Streams
- `HeartbeatIter::new(store, rules, max_cycles)` -- sync iterator
- `spawn_heartbeat(session, config) -> Receiver` -- async tokio stream

### Memory
- `RunHistory` -- task attempt records (save/load pipe-delimited)
- `RuleLibrary` -- scored rules with pruning (save/load .qor)

### Persistent (feature-gated)
- `PersistentStore` -- O(1) snapshot NeuronStore via HAMT

### Important Constants
- `forward_chain` max iterations per stratum: **10**
- `consolidate` TV change threshold: **0.001**
- `PruneConfig` defaults: max_rules=**500**, min_accuracy=**0.1**, unused_threshold=**100**
- `sleep_cycle` confidence PLN parameter k: **1.0**, cap: **0.95**
- `HeartbeatConfig` defaults: interval_ms=**100**, settled_threshold=**5**
- `genesis` evolution max generations: **15**
- `refinement_search` max generations: **50**, max population: **20**
- `genesis_swarm` cancel threshold: score >= **0.95**
- `HeartbeatIter` default settled_threshold: **3**
