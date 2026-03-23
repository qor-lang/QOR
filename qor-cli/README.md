# qor-cli

## 1. Crate Overview

**Crate:** `qor-cli`
**Binary name:** `qor`
**File:** `D:\QOR-LANG\qor\qor-cli\src\main.rs` (2308 lines)
**Cargo.toml:** `D:\QOR-LANG\qor\qor-cli\Cargo.toml`

The `qor-cli` crate is the user-facing command-line interface and binary entry point for the entire QORlang system. It compiles to a single binary named `qor` and provides subcommands for file execution, data ingestion, continuous reasoning, conversational chat, hypothesis testing, knowledge base construction, real-time perception, and an interactive REPL. It has no public API of its own -- it is a binary crate that orchestrates all other crates (`qor-core`, `qor-runtime`, `qor-bridge`, `qor-inference`).

**Dependencies (from Cargo.toml):**
- `qor-core` (parser, neuron types, unification)
- `qor-runtime` (session, eval, heartbeat, knowledge base)
- `qor-bridge` (data ingestion, DNA, language, web, learning, OEIS, dharmic, perception, KB building)
- `qor-inference` (induction/abduction inference)

**Feature flags:**
- `web` -- enables `qor-bridge/web` for live web crawling via spider-rs

---

## 2. File-by-File Breakdown: `main.rs`

### Entry Point

| Function | Signature | Description |
|----------|-----------|-------------|
| `main()` | `fn main()` | Parses CLI args and dispatches to the appropriate subcommand handler. Matches on `args[1]` against all known commands; prints usage and exits on unknown commands. |

### Command Handlers

| Function | Signature | Description |
|----------|-----------|-------------|
| `run_file(path)` | `fn run_file(path: &str)` | Reads a `.qor` file, calls `qor_runtime::eval::run()`, prints all query results. Exit 1 on read or runtime errors. |
| `check_file(path)` | `fn check_file(path: &str)` | Reads a `.qor` file, calls `qor_core::parser::parse_with_warnings()`, reports statement count and any warnings. Exit 1 on syntax errors. |
| `test_file(path)` | `fn test_file(path: &str)` | Reads a `.qor` file, separates `Statement::Test` blocks from rules/facts. For each test: creates a fresh `Session`, loads rules/facts, injects `given` facts, checks `expect` (Present/Absent) using unification. Reports PASS/FAIL with colored output. Exit 1 if any test fails. |
| `feed_file(path)` | `fn feed_file(path: &str)` | Calls `qor_bridge::feed_file()` to auto-detect format (JSON/CSV/KV/text/parquet) and learn condensed knowledge. Performs incremental merge via `qor_bridge::learn::merge_learned()` if brain file already exists. Writes resulting `.qor` to `brain/` directory. |
| `think()` | `fn think()` | Loads all brain knowledge into a `Session`, runs up to 50 heartbeat cycles until beliefs stabilize (3 consecutive no-change cycles). Then runs `qor_inference::infer()` for induction/abduction to discover new rules, loads them, runs 10 more heartbeats, executes stored queries, prints summary, and saves high-confidence inferred facts to the KB via `save_insights_to_kb()`. |
| `explain(pattern_str)` | `fn explain(pattern_str: &str)` | Parses a query pattern string, loads brain, runs 20 heartbeat cycles, then calls `session.explain(&pattern)` to recursively trace why a fact is believed. Prints explanation tree with `print_explanation()`. |
| `think_watch()` | `fn think_watch()` | Watch mode for `qor think`. Polls `brain/` every 2 seconds for file modifications using `load_brain_timestamps()`. On change, reloads brain into a fresh session and re-runs a think cycle. Runs until Ctrl+C. |
| `summary()` | `fn summary()` | Collects all statements from `brain/*.qor` files, calls `qor_bridge::llguidance::summarize()` to produce a natural-language summary. |
| `solve_file(path)` | `fn solve_file(path: &str)` | Loads a `.qor` file, loads meta rules via `load_meta_into_session()`, extracts expected facts (predicates starting with "expected-" or "predict-"), auto-detects the target predicate ("predict-cell" or "answer"), collects observation facts (predicates starting with "obs-"), then delegates to `qor_bridge::solve::solve()` which runs the full 6-phase solve pipeline. Displays results: score, solved status, phase reached, candidates explored, mutations tried, elapsed time, and best rules. |
| `chat()` | `fn chat()` | Interactive conversational mode. Loads brain, language knowledge (from `language/<code>/` or built-in), and optionally a DNA profile (exact ID or keyword search). Main loop: reads user input, handles exit commands, teach patterns ("X means Y"), direct QOR syntax passthrough, or natural language processing via tokenization + forward chaining + response formatting. Clears ephemeral facts each turn. |
| `repl()` | `fn repl()` | Interactive REPL. Supports direct QOR statement entry, and special commands: `:load`, `:feed`, `:facts`, `:rules`, `:stats`, `:beat`, `:help`, `:quit`. Empty Enter triggers one heartbeat cycle. |
| `dna_command()` | `fn dna_command()` | Handles `qor dna` subcommands: `list` (shows available DNA profiles with .qor generation status), `convert` (converts single or all DNA profiles to .qor via `qor_bridge::dna`), `help` (prints DNA usage). |
| `heartbeat_command()` | `fn heartbeat_command()` | Handles `qor heartbeat` subcommands: `--status` (prints solve/library stats), `--once` (single learning pulse), `--web-once <url>` (crawl single URL), or continuous mode (repeated pulses with configurable interval). Loads puzzle solver rules from DNA and web seeds from brain config. |
| `perceive_command()` | `fn perceive_command()` | Handles `qor perceive`. Loads source configs from `brain/sources.qor`, supports `--sources` (list sources), `--once` (single cycle), or continuous mode. Fetches data from configured URLs, injects as facts, runs heartbeat, identifies significant findings. |
| `build_kb_command()` | `fn build_kb_command()` | Handles `qor build-kb`. Supports `--list` (available sources: oeis, dharmic, general), `--stats` (KB statistics), or `--source <name> --input <dir>` to build a binary knowledge base. Dispatches to OEIS, Dharmic, or General pipelines. Saves shared entity/predicate mappings. |

### Helper Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `brain_dir()` | `fn brain_dir() -> PathBuf` | Locates the `brain/` directory by walking up from the binary location, looking for workspace root. Falls back to `./brain/`. |
| `language_dir()` | `fn language_dir() -> PathBuf` | Locates the `language/` directory via same workspace-root walk. |
| `dna_dir()` | `fn dna_dir() -> PathBuf` | Locates the `dna/` directory via same workspace-root walk. |
| `meta_dir()` | `fn meta_dir() -> PathBuf` | Locates the `meta/` directory via same workspace-root walk. |
| `load_brain_into_session(session, brain)` | `fn load_brain_into_session(session: &mut Session, brain: &Path)` | Reads all `*.qor` files from `brain/` sorted by path, executes each into the session. Also calls `load_kb_into_session()` for binary KB. |
| `load_kb_into_session(session, brain)` | `fn load_kb_into_session(session: &mut Session, brain: &Path)` | Loads the binary knowledge graph from `brain/knowledge/` via `KnowledgeBase::load()`, attaches it to the session with `session.set_kb()`. Also loads `oeis_formulas.qor` if present. |
| `load_meta_into_session(session)` | `fn load_meta_into_session(session: &mut Session)` | Loads all `*.qor` files from the `meta/` directory into the session. |
| `save_insights_to_kb(session, brain)` | `fn save_insights_to_kb(session: &Session, brain: &Path)` | Saves high-confidence inferred facts (strength >= 0.80, confidence >= 0.50) to the binary KB as triples. Skips ephemeral predicates (input, compound, intent, etc.). |
| `load_brain_timestamps(brain)` | `fn load_brain_timestamps(brain: &Path) -> Vec<(String, SystemTime)>` | Returns sorted list of (filename, modified-time) for all `*.qor` files in `brain/`. Used by watch mode to detect changes. |
| `run_think_cycle(session, verbose)` | `fn run_think_cycle(session: &mut Session, verbose: bool)` | Runs up to 50 heartbeats, then induction/abduction, then 10 more heartbeats, then queries. Used by watch mode. |
| `print_explanation(exp, indent)` | `fn print_explanation(exp: &Explanation, indent: usize)` | Recursively prints an explanation tree (asserted or derived-via-rule with sub-explanations). |
| `display_exec_result(result)` | `fn display_exec_result(result: &ExecResult)` | Formats and prints a single `ExecResult` (Stored, RuleAdded, or Query) with colored output. |
| `show_facts(session)` | `fn show_facts(session: &Session)` | Prints all facts, highlighting inferred ones in yellow. Shows total and inferred counts. |
| `show_rules(session)` | `fn show_rules(session: &Session)` | Prints rule count. |
| `show_stats(session)` | `fn show_stats(session: &Session)` | Prints facts, rules, and consolidation cycle counts. |
| `print_repl_help()` | `fn print_repl_help()` | Prints REPL command reference. |
| `print_usage()` | `fn print_usage()` | Prints full CLI usage/help text for all commands. |
| `load_web_seeds(brain)` | `fn load_web_seeds(brain: &Path) -> Vec<String>` | Loads `brain/web_config.qor`, parses web config, and (if `web` feature enabled) searches for candidate rules via spider-rs. Returns empty vec without the feature. |
| `web_once_command(url, brain)` | `fn web_once_command(url: &str, brain: &Path)` | Crawls a single URL (or reads local file), extracts QOR facts via `extract_facts_from_pages()` and candidate rules via `extract_rules()`. Displays results. |
| `build_general_kb(meta_dir, output_dir, encoder)` | `fn build_general_kb(meta_dir: &Path, output_dir: &Path, encoder: &mut IdEncoder) -> Result<SourceStats, String>` | Scans `meta/*.qor` files, extracts entities and predicates, writes binary triples for domain membership and cross-domain relationships (shared entities). |

---

## 3. Connection Diagram

```
main()
  |
  +-- run_file()           --> qor_runtime::eval::run()
  +-- check_file()         --> qor_core::parser::parse_with_warnings()
  +-- test_file()          --> qor_core::parser::parse()
  |                        --> qor_runtime::eval::Session::new/exec_statements/all_facts
  |                        --> qor_core::unify::unify()
  +-- feed_file()          --> qor_bridge::feed_file()
  |                        --> qor_bridge::learn::merge_learned()
  +-- think()              --> load_brain_into_session() --> qor_runtime::eval::Session::exec()
  |                        --> load_meta_into_session()
  |                        --> session.heartbeat() [loop]
  |                        --> qor_inference::infer()
  |                        --> session.run_queries()
  |                        --> save_insights_to_kb() --> qor_runtime::kb::KnowledgeBase
  +-- explain()            --> session.explain()
  |                        --> print_explanation() [recursive]
  +-- think_watch()        --> load_brain_timestamps()
  |                        --> run_think_cycle() --> heartbeat + infer + queries
  +-- summary()            --> qor_bridge::llguidance::summarize()
  +-- solve_file()         --> load_meta_into_session()
  |                        --> qor_bridge::solve::solve() [6-phase pipeline]
  +-- chat()               --> qor_bridge::language::load_language_dir()
  |                        --> qor_bridge::dna::load_dna/find_dna_by_keywords()
  |                        --> qor_bridge::language::tokenize/format_response()
  |                        --> session.heartbeat/response_facts/clear_turn()
  +-- dna_command()        --> qor_bridge::dna::convert_all/save_dna_qor/available_dna()
  +-- heartbeat_command()  --> qor_runtime::heartbeat::Heartbeat::load/pulse()
  |                        --> load_web_seeds() --> qor_bridge::web_fetch/web_search
  |                        --> web_once_command()
  +-- perceive_command()   --> qor_bridge::perceive::parse_sources/PerceptionState::perceive()
  +-- build_kb_command()   --> qor_bridge::oeis::run_oeis_pipeline()
  |                        --> qor_bridge::dharmic::run_dharmic_pipeline()
  |                        --> build_general_kb() --> qor_bridge::kb_build
  +-- repl()               --> session.exec() [interactive loop]
  +-- print_usage()
```

---

## 4. Public API Summary

This is a **binary crate** -- it has no public API. All functions are private (`fn`, not `pub fn`). The crate's "API" is its CLI interface.

---

## 5. Full Command Tree

```
qor
 |
 +-- run <file.qor>                  Execute a .qor file, print query results
 +-- check <file.qor>                Parse-check syntax, report warnings
 +-- test <file.qor>                 Run @test blocks, report PASS/FAIL
 +-- feed <file>                     Ingest data (JSON/CSV/Parquet/text/KV), learn, save to brain/
 +-- think                           Load brain, heartbeat loop, induction/abduction, show insights
 |    +-- --watch / -w               Monitor brain/ for changes, re-reason on file modification
 +-- explain "<pattern>"             Trace WHY a fact is believed (recursive explanation tree)
 +-- summary                         Natural language summary of brain knowledge
 +-- chat                            Interactive conversational mode (NL + QOR passthrough)
 |    +-- --dna <id>                 Load a DNA profession profile
 |    +-- --dna list                 List available DNA profiles
 |    +-- --dna "<keyword>"          Keyword search for a DNA profile
 +-- dna                             DNA profile management
 |    +-- list                       List all available profiles (shows .qor generation status)
 |    +-- convert                    Convert ALL DNA profiles (JSON -> .qor)
 |    +-- convert <id>               Convert a single DNA profile
 |    +-- help                       Print DNA subcommand usage
 +-- solve <file.qor>                Load meta rules, extract expected/observation facts, run 6-phase solve pipeline
 +-- heartbeat                       Continuous self-improving learning loop
 |    +-- --once                     Single learning pulse
 |    +-- --status                   Show learning progress (solved count, library size)
 |    +-- --interval <N>             Set seconds between pulses (default: 30)
 |    +-- --web-once <url>           Crawl a single URL, extract facts + rules
 +-- perceive                        Continuous real-time data perception loop
 |    +-- --once                     Single perception cycle
 |    +-- --sources                  List configured data sources from brain/sources.qor
 |    +-- --interval <N>             Set seconds between cycles (default: 300)
 +-- build-kb                        Build binary knowledge base
 |    +-- --source oeis --input <dir>     Build OEIS math knowledge base
 |    +-- --source dharmic --input <dir>  Build Dharmic sacred texts knowledge base
 |    +-- --source general                Build cross-domain links from meta/*.qor
 |    +-- --stats                    Show knowledge graph statistics
 |    +-- --list                     List available knowledge sources
 +-- repl                            Interactive REPL session
 |    REPL commands:
 |    +-- :load <file.qor>           Load a .qor file into session
 |    +-- :feed <file>               Ingest any data format
 |    +-- :facts / :f                Show all facts (inferred highlighted)
 |    +-- :rules / :r                Show rule count
 |    +-- :stats / :s                Show facts/rules/cycles
 |    +-- :beat                      Run 10 consolidation cycles
 |    +-- :help / :h                 Show REPL help
 |    +-- :quit / :q / :exit         Exit REPL
 |    +-- <Enter on empty line>      One heartbeat cycle
 +-- version                         Print version
 +-- help / --help / -h              Print usage
```
