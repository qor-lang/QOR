# qor-agent

## Crate Overview

`qor-agent` is the universal execution agent for QORlang. It follows the principle **"QOR Decides, Rust Executes"** -- all domain logic lives in DNA (`.qor` files), while Rust provides pure plumbing: loading data, running the QOR reasoning engine, reading its decisions, and executing actions (browser interactions, HTTP fetches, file I/O).

The agent operates in two modes:

- **Batch mode**: Loads data items from files or URLs, feeds them into a QOR session loaded with DNA rules, compares QOR's answers against expected outputs, and runs search/genesis to discover better rules when answers are wrong. Multi-attempt loop with confidence tracking.
- **Browse mode**: Drives a headless Chrome browser via CDP (Chrome DevTools Protocol). QOR DNA rules produce `(browser-action ...)` facts; Rust reads them and executes click, fill, navigate, snapshot, etc. Perceive-reason-act loop.

### Dependencies (Cargo.toml)

| Crate | Version | Purpose |
|-------|---------|---------|
| `qor-runtime` | path: `../qor-runtime` | Session, forward chaining, search, genesis, library |
| `qor-core` | path: `../qor-core` | Neuron, Statement, Parser, TruthValue |
| `qor-bridge` | path: `../qor-bridge` | Data ingestion (JSON, CSV, text, parquet) |
| `agentchrome` | 1.3 | Chrome CDP client, Chrome process launcher |
| `tokio` | 1 (rt-multi-thread, macros, sync, time, process) | Async runtime |
| `serde` | 1 (derive) | Serialization |
| `serde_json` | 1 | JSON parsing |
| `ureq` | 2 | HTTP client (sync) |
| `anyhow` | 1 | Error handling |
| `zip` | 2 | Chromium download extraction |
| `dirs` | 6 | Home directory resolution (`~/.qor/chromium/`) |

### CLI Usage

```
qor-agent <dna-dir> <data-source> [--search-budget N] [--max-attempts N]
qor-agent <dna-dir> --browse <url> [--steps N]
```

---

## File-by-File Breakdown

---

### 1. `types.rs`

**Purpose**: Core data types shared across all agent modules.

#### Public Structs

```rust
pub struct DataItem {
    pub id: String,
    pub facts: Vec<Statement>,
    pub expected: Option<Vec<(String, Vec<String>)>>,
}
```
A generic data item to process. Could be a puzzle, a patient, a trade -- anything. The DNA rules determine what to do with the facts. `expected` holds expected output facts for scoring, or `None` if unknown.

```rust
#[derive(Debug, Default)]
pub struct AgentStats {
    pub attempted: usize,
    pub correct:   usize,
    pub wrong:     usize,
    pub no_answer: usize,
    pub learned:   usize,
    pub skipped:   usize,
}
```
Tracks agent performance across all items in a batch run. Implements `Display` for formatted output: `"3/5 correct | 1 wrong | 1 no-answer | 2 learned | 0 skipped"`.

#### Trait Implementations

- `impl std::fmt::Display for AgentStats` -- Formats stats as a single-line summary string.

#### Connections

- `DataItem` is constructed by `perceive::load_items()` and consumed by `main.rs`'s batch loop.
- `AgentStats` is created and updated exclusively in `main.rs`.

---

### 2. `perceive.rs`

**Purpose**: Generic data loader. Loads data items from files, directories, or URLs. Knows nothing about the domain -- uses `qor-bridge` to auto-detect format and convert to QOR facts.

#### Public Functions

```rust
pub fn load_items(source: &str) -> Result<Vec<DataItem>>
```
Entry point for data loading. Dispatches based on source type:
- If `source` is a directory path: calls `load_from_directory()`, loads all `.json`, `.csv`, `.txt`, `.qor` files sorted alphabetically.
- If `source` is a file path: calls `load_from_file()`, returns single-item vec.
- Otherwise: treats as URL or ID, calls `load_from_url()`.

Called by `main()` in batch mode.

#### Private Functions

```rust
fn load_from_directory(dir: &Path) -> Result<Vec<DataItem>>
```
Reads all files with extensions `.json`, `.csv`, `.txt`, `.qor` from the directory, sorted alphabetically. Calls `load_from_file()` for each. Skips files that fail to parse (with warning).

```rust
fn load_from_file(path: &Path) -> Result<DataItem>
```
Loads a single file into a `DataItem`. For `.qor` files, uses `qor_core::parser::parse()` directly. For other formats (JSON, CSV, text), delegates to `qor_bridge::feed_file()` for auto-detection and conversion. Extracts expected output facts via `extract_expected()`. Uses the file stem as the item ID.

```rust
fn load_from_url(source: &str) -> Result<Vec<DataItem>>
```
Fetches data from a URL via `ureq`. If `source` starts with `"http"`, uses it directly. Otherwise, assumes it is an ARC task ID and constructs a GitHub raw URL. Converts response body to facts via `qor_bridge::feed()`.

```rust
fn extract_expected(facts: &[Statement]) -> Option<Vec<(String, Vec<String>)>>
```
Scans facts for any predicate starting with `"expected-"`. Returns `None` if no expected facts found, otherwise returns `Some(vec)` of `(predicate_name, argument_strings)` pairs.

#### Connections

- Calls: `qor_core::parser::parse()`, `qor_bridge::feed_file()`, `qor_bridge::feed()`, `ureq::get()`
- Called by: `main()` in batch mode
- Produces: `Vec<DataItem>` consumed by the main loop

---

### 3. `browser.rs`

**Purpose**: CDP wrapper using agentchrome's public API. Pure plumbing. Launches Chrome, connects via CDP, provides high-level async methods for browser automation. Auto-downloads Chromium on first use if not already present.

#### Constants

```rust
const INTERACTIVE_ROLES: &[&str] = &[
    "link", "button", "textbox", "checkbox", "radio", "combobox",
    "menuitem", "tab", "switch", "slider", "spinbutton", "searchbox",
    "option", "treeitem",
];
```
Accessibility roles that get assigned UIDs for click/fill targeting during snapshot.

#### Public Structs

```rust
pub struct Browser {
    _client: CdpClient,
    session: ManagedSession,
    _chrome: Option<ChromeProcess>,
    pub uid_map: HashMap<String, i64>,
}
```
A browser session connected to Chrome via CDP. `uid_map` maps short UIDs (e.g., `"s1"`, `"s2"`) to backend DOM node IDs from the last accessibility snapshot.

```rust
pub struct ProbeResult {
    pub chrome_found: bool,
    pub executable: Option<std::path::PathBuf>,
}
```
Result of probing for Chrome availability. Used by `browse_mode()` to decide whether to launch Chrome or fall back to HTTP-only mode.

```rust
#[derive(Debug, Clone)]
pub struct PageElement {
    pub role: String,
    pub name: String,
    pub uid: Option<String>,
    pub properties: Option<HashMap<String, Value>>,
}
```
A page element from the accessibility snapshot. `uid` is assigned only for interactive elements (those whose role is in `INTERACTIVE_ROLES`). `properties` holds accessibility properties like `url`, `checked`, `value`.

#### Public Methods on `Browser`

```rust
pub fn probe() -> ProbeResult
```
Check if Chrome/Chromium is available without launching it. Checks QOR's local copy (`~/.qor/chromium/`) first, then system Chrome.
Called by `browse_mode()` in `main.rs`.

```rust
pub async fn launch() -> Result<Self, String>
```
Launch headless Chrome and connect via CDP. Search order: (1) QOR's bundled Chromium at `~/.qor/chromium/`, (2) System Chrome, (3) Auto-download Chromium if neither found. Finds a free port, launches Chrome via `agentchrome::chrome::launch_chrome()`, queries targets, connects CDP WebSocket, creates a managed session.
Called by `browse_mode()` and lazily in `execute_actions()` in `main.rs`.

```rust
pub async fn navigate(&mut self, url: &str) -> Result<(String, String), String>
```
Navigate to a URL. Enables Page and Runtime CDP domains, subscribes to `Page.loadEventFired`, sends `Page.navigate`, waits up to 30 seconds for load. Returns `(final_url, page_title)`.
Called by `act::execute()` for `AgentAction::Navigate`.

```rust
pub async fn snapshot(&mut self) -> Result<Vec<PageElement>, String>
```
Get full accessibility tree snapshot. Enables Accessibility CDP domain, calls `Accessibility.getFullAXTree`. Filters out ignored nodes, assigns UIDs to interactive elements, updates `uid_map`. Skips empty/generic nodes.
Called by `act::execute()` for `AgentAction::Snapshot`.

```rust
pub async fn click(&mut self, uid: &str) -> Result<(), String>
```
Click an element by UID. Looks up `uid` in `uid_map`, enables DOM domain, scrolls element into view via `DOM.scrollIntoViewIfNeeded`, gets element center via `DOM.getBoxModel`, dispatches mousePressed + mouseReleased events at center coordinates. 100ms delay after click for effects.
Called by `act::execute()` for `AgentAction::Click`, and internally by `fill()`.

```rust
pub async fn fill(&mut self, uid: &str, value: &str) -> Result<(), String>
```
Fill a form field by UID. Clicks element first to focus, sends Ctrl+A to select all existing text, then types each character via `Input.dispatchKeyEvent` with type `"char"`.
Called by `act::execute()` for `AgentAction::Fill`.

```rust
pub async fn page_text(&mut self) -> Result<String, String>
```
Get visible page text. Evaluates `document.body.innerText` via `Runtime.evaluate`.
Called by `act::execute()` for `AgentAction::PageText`.

```rust
pub async fn eval_js(&mut self, script: &str) -> Result<String, String>
```
Execute arbitrary JavaScript and return result as string. Uses `Runtime.evaluate` with `returnByValue: true`.
Called by `act::execute()` for `AgentAction::EvalJs`.

```rust
pub async fn screenshot(&mut self, full_page: bool) -> Result<String, String>
```
Take a screenshot, return base64-encoded PNG. Uses `Page.captureScreenshot` with optional `captureBeyondViewport` for full-page captures.
Called by `act::execute()` for `AgentAction::Screenshot`.

```rust
pub async fn network_list(&mut self, filter: Option<&str>) -> Result<Value, String>
```
Get network requests via `performance.getEntriesByType('resource')`. Returns JSON array of `{name, type, duration, size}` objects. Optionally filters entries by substring match on `name`.
Called by `act::execute()` for `AgentAction::NetworkList`.

#### Private Methods on `Browser`

```rust
async fn get_page_info(&mut self) -> Result<(String, String), String>
```
Get current page URL and title by evaluating `location.href` and `document.title`. Called internally by `navigate()` after page load completes.

#### Private Module Functions

```rust
fn chromium_dir() -> PathBuf
```
Returns `~/.qor/chromium/` -- where QOR stores its bundled Chromium.

```rust
fn chromium_exe_name() -> &'static str  // platform-specific
```
Platform-specific Chromium executable name: `"chrome.exe"` (Windows), `"Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"` (macOS), `"chrome"` (Linux).

```rust
fn platform_key() -> &'static str  // platform-specific
```
Platform key for Chrome for Testing downloads: `"win64"`, `"win32"`, `"mac-arm64"`, `"mac-x64"`, or `"linux64"`.

```rust
fn find_chromium() -> Option<PathBuf>
```
Find Chromium: checks QOR's local copy first, then system Chrome via `agentchrome::chrome::find_chrome_executable(Channel::Stable)`.

```rust
fn find_exe_in_dir(dir: &Path) -> Option<PathBuf>
```
Recursively find the chrome executable inside a directory. Matches by `chromium_exe_name()` suffix or direct filename match (`chrome`, `chrome.exe`, `chromium`).

```rust
fn walkdir(dir: &Path) -> Vec<PathBuf>
```
Simple recursive directory walker. Returns all file paths under `dir`.

```rust
fn download_chromium() -> Result<PathBuf, String>
```
Download Chrome for Testing to `~/.qor/chromium/`. Queries Google's official Chrome for Testing API for the latest stable version, downloads the platform-specific zip, extracts it, sets executable permissions on Unix. Returns path to the chrome executable.

#### Connections

- Called by: `act::execute()` (all browser methods), `main.rs` (`launch()`, `probe()`)
- Calls: `agentchrome::cdp::CdpClient`, `agentchrome::chrome::launch_chrome()`, `agentchrome::chrome::find_chrome_executable()`, `ureq::get()` (for Chromium download), `zip::ZipArchive`

---

### 4. `act.rs`

**Purpose**: Action executor. Reads QOR decisions (`(browser-action ...)` facts), maps them to browser operations, executes them, and returns result facts. Pure plumbing with zero domain logic.

#### Public Enums

```rust
#[derive(Debug, Clone)]
pub enum AgentAction {
    Navigate(String),          // Navigate to URL
    Click(String),             // Click element by UID
    Fill(String, String),      // Fill form field (UID, value)
    Snapshot,                  // Get accessibility snapshot
    PageText,                  // Get visible page text
    Screenshot(bool),          // Take screenshot (full_page flag)
    EvalJs(String),            // Execute JavaScript
    NetworkList(Option<String>), // List network requests (optional filter)
    Done,                      // Agent finished
}
```
All possible actions the agent can perform on the browser. Maps 1:1 to `(browser-action <verb> ...)` QOR facts.

#### Private Structs

```rust
struct ScoredAction {
    action: AgentAction,
    confidence: f64,
}
```
An action paired with its TruthValue strength from the QOR session. Used internally for selecting the highest-confidence action.

#### Public Functions

```rust
pub fn next_action(session: &Session) -> Option<AgentAction>
```
Read `(browser-action ...)` facts from the QOR session. Iterates all facts, filters for `browser-action` predicate, parses each into an `AgentAction` with its TruthValue strength. Returns the highest-confidence action, or `None` if no action facts exist.
Called by `execute_actions()` and `browse_mode()` in `main.rs`, and by `http_fallback()`.

```rust
pub async fn execute(
    action: &AgentAction,
    browser: &mut Browser,
) -> Result<Vec<Statement>, String>
```
Execute an action on the browser, return results as QOR `Statement` facts. Dispatches to the appropriate `Browser` method and converts the result using `page::*` conversion functions:
- `Navigate` -> `browser.navigate()` -> `page::nav_to_facts()`
- `Click` -> `browser.click()` -> `action_result_fact("clicked", uid)`
- `Fill` -> `browser.fill()` -> `action_result_fact("filled", uid)`
- `Snapshot` -> `browser.snapshot()` -> `page::snapshot_to_facts()`
- `PageText` -> `browser.page_text()` -> `page::text_to_facts()`
- `Screenshot` -> `browser.screenshot()` -> `action_result_fact("screenshot", "taken")`
- `EvalJs` -> `browser.eval_js()` -> `page::js_result_to_facts()`
- `NetworkList` -> `browser.network_list()` -> `page::network_to_facts()`
- `Done` -> `action_result_fact("done", "true")`

Called by `execute_actions()` and `browse_mode()` in `main.rs`.

#### Private Functions

```rust
fn parse_action(parts: &[Neuron]) -> Option<AgentAction>
```
Parse an `AgentAction` from neuron parts after `"browser-action"`. Matches verb symbol (`navigate`, `click`, `fill`, `snapshot`, `page-text`, `screenshot`, `js`, `network-list`, `done`) and extracts arguments.

```rust
fn action_result_fact(verb: &str, detail: &str) -> Statement
```
Create a simple `(action-result <verb> <detail>)` fact with TruthValue `<0.99, 0.95>`.

```rust
fn get_string(n: &Neuron) -> Option<String>
```
Extract string value from a Neuron -- accepts both `QorValue::Str` and `Symbol`.

```rust
fn get_symbol(n: &Neuron) -> Option<String>
```
Extract symbol/string value from a Neuron -- accepts both `Symbol` and `QorValue::Str`.

#### Connections

- Calls: `Browser::navigate()`, `Browser::click()`, `Browser::fill()`, `Browser::snapshot()`, `Browser::page_text()`, `Browser::eval_js()`, `Browser::screenshot()`, `Browser::network_list()`, `page::nav_to_facts()`, `page::snapshot_to_facts()`, `page::text_to_facts()`, `page::js_result_to_facts()`, `page::network_to_facts()`
- Called by: `main.rs` (`execute_actions()`, `browse_mode()`, `http_fallback()`)
- Reads from: `Session::all_facts()` (QOR session)

---

### 5. `page.rs`

**Purpose**: Page-to-QOR converter. Transforms browser output (elements, text, network data) into QOR `Statement` facts. Pure data conversion with zero domain logic. DNA rules fire on these facts to decide what the agent does next.

#### Public Functions

```rust
pub fn nav_to_facts(url: &str, title: &str) -> Vec<Statement>
```
Convert navigation result into QOR facts. Produces two facts: `(page-url "<url>")` and `(page-title "<title>")`.
Called by `act::execute()` for `Navigate` action.

```rust
pub fn text_to_facts(text: &str) -> Vec<Statement>
```
Convert page text into a QOR fact. Returns empty vec if text is empty. Truncates to 4000 characters. Produces `(page-text "<text>")`.
Called by `act::execute()` for `PageText` action.

```rust
pub fn snapshot_to_facts(elements: &[PageElement]) -> Vec<Statement>
```
Convert accessibility snapshot elements into QOR facts. For each interactive element (has UID): produces `(page-element <uid> <role> "<name>" "<href>")`. For non-interactive named elements with roles `heading`, `paragraph`, `banner`, `navigation`, `main`, `contentinfo`: produces `(page-content <role> "<name>")`. Also produces three summary count facts: `(page-link-count N)`, `(page-button-count N)`, `(page-form-count N)`.
Called by `act::execute()` for `Snapshot` action.

```rust
pub fn network_to_facts(entries: &Value) -> Vec<Statement>
```
Convert network entries JSON array into QOR facts. For each entry: produces `(network-request "<rN>" <type> "<url>" <duration>)` where N is 1-indexed.
Called by `act::execute()` for `NetworkList` action.

```rust
pub fn js_result_to_facts(script: &str, result: &str) -> Vec<Statement>
```
Convert JavaScript eval result into a QOR fact. Produces `(js-result "<script>" "<result>")`.
Called by `act::execute()` for `EvalJs` action.

#### Private Helper Functions

```rust
fn sym(s: &str) -> Neuron        // Creates Neuron::Symbol
fn str_val(s: &str) -> Neuron    // Creates Neuron::Value(QorValue::Str(...))
fn int_val(i: i64) -> Neuron     // Creates Neuron::Value(QorValue::Int(...))
fn fact(parts: Vec<Neuron>) -> Statement  // Creates Statement::Fact with TV <0.95, 0.90>
```

#### Default TruthValue

All facts created by this module use `TruthValue::new(0.95, 0.90)` -- high strength, moderate-high confidence.

#### Connections

- Called by: `act::execute()` (all conversion functions)
- Uses: `browser::PageElement` struct
- Produces: `Vec<Statement>` injected into the QOR session

---

### 6. `learn.rs`

**Purpose**: Structured memory persistence. Implements a 5-level memory system for the agent to remember what it has tried, what worked, and what failed. Domain-agnostic -- features extracted from `obs-*`/`detected-*`/`consistent-*` facts produced by DNA rules.

#### 5-Level Memory System

| Level | Predicate | Purpose |
|-------|-----------|---------|
| 1 | `wrong-attempt` | "I failed this item N times" |
| 2 | `item-feature` | "This item has feature X" (trie-indexed for similarity search) |
| 3 | `tried-and-failed` | "I tried strategy X and it didn't work" |
| 4 | `best-partial` | "My best score was X% using strategy Y" |
| 5 | `solved` | "I solved it using strategy Y" (the gold) |

#### Two Files

- `memory.qor` -- structured memory facts (all 5 levels), section-based per item
- `rules_learned.qor` -- executable QOR rules discovered by search/genesis

#### Public Functions

```rust
pub fn extract_features(session: &Session) -> Vec<String>
```
Extract item features from session. Scans all facts for predicates starting with `obs-*`, `detected-*`, or `consistent-*`. Returns deduplicated, sorted feature names.
Called by `main.rs` after scoring (both correct and wrong).

```rust
pub fn extract_tried(session: &Session) -> Vec<String>
```
Extract strategies that were tried. Collects `detected-*` predicates (stripping prefix) and `save-known-transform` third arguments. Returns deduplicated, sorted strategy names.
Called by `main.rs` when an answer is wrong.

```rust
pub fn extract_winning_strategy(session: &Session) -> String
```
Extract the winning strategy from session facts. Priority: (1) `save-known-transform` third arg (skipping generic `same-size`/`output-smaller`), (2) `detected-*` prefix-stripped, (3) `"unknown"` fallback.
Called by `main.rs` when an item is solved.

```rust
pub fn record_attempt(
    item_id: &str,
    attempt: usize,
    score: f64,
    best_source: &str,
    features: &[String],
    tried: &[String],
    memory_path: &Path,
) -> Result<usize>
```
Record a failed attempt with rich context. Writes/updates the item's section in `memory.qor` with levels 1-4: `wrong-attempt`, `item-feature` (for each feature), `tried-and-failed` (for each strategy), `best-partial` (if score between 5% and 99.9%). Uses `upsert_section()` for atomic section replacement. Returns number of facts written.
Called by `main.rs` when an answer is wrong.

```rust
pub fn record_solved(
    item_id: &str,
    winning_strategy: &str,
    features: &[String],
    memory_path: &Path,
) -> Result<usize>
```
Record a solved item. Replaces entire item section with level 5 `solved` fact + level 2 `item-feature` facts. All failure data (wrong-attempt, tried-and-failed, best-partial) is removed. Returns number of facts written.
Called by `main.rs` when an item is solved.

```rust
pub fn save_winning_rules(
    rules: &[String],
    item_id: &str,
    rules_path: &Path,
) -> Result<usize>
```
Save winning rules to `rules_learned.qor`. Appends rules with a comment header `";; discovered for item:<id>"`. Deduplicates -- won't add a rule already present in the file. Returns count of new rules saved.
Called by `main.rs` after successful genesis or search.

```rust
pub fn load_solved_ids(memory_path: &Path) -> HashSet<String>
```
Load solved item IDs from `memory.qor`. Parses lines starting with `(solved ` and extracts the item ID. Returns a `HashSet<String>` for O(1) lookup.
Called by `main.rs` at startup to skip already-solved items.

```rust
pub fn memory_stats(memory_path: &Path) -> (usize, usize, usize)
```
Get memory stats: `(solved_count, failed_count, total_facts)`. Counts lines starting with `(solved `, `(wrong-attempt `, and any line starting with `(`.
Called by `main.rs` at startup for display.

#### Private Functions

```rust
fn section_marker(item_id: &str) -> String
```
Generate section marker: `";; ══ <item_id> ══"`.

```rust
fn upsert_section(item_id: &str, lines: &[String], path: &Path) -> Result<usize>
```
Insert or replace a section for an item in the memory file. Each item gets exactly one section. If the item already has a section, it is replaced atomically (found by marker, next marker or EOF defines end). If not found, a new section is appended. Creates parent directories if needed. Returns number of facts written.

#### Connections

- Called by: `main.rs` (all public functions)
- Reads from: `Session::all_facts()` (feature/strategy extraction)
- Writes to: `memory.qor`, `rules_learned.qor` files on disk

---

### 7. `main.rs`

**Purpose**: Entry point and orchestrator. Parses CLI arguments, dispatches to batch mode or browse mode, drives the main perceive-reason-act-learn loop.

#### Async Entry Point

```rust
#[tokio::main]
async fn main() -> Result<()>
```
Parses CLI args: `dna_dir` (default: `D:\QOR-LANG\qor\dna\puzzle_solver`), `--browse <url>`, `--steps N` (default 20), `--search-budget N` (default 3000ms), `--max-attempts N` (default 5). Dispatches to `browse_mode()` if `--browse` is present, otherwise runs batch mode. In batch mode: loads items via `perceive::load_items()`, loads rule library, loads solved IDs, then iterates items with multi-attempt loop. Prints final stats and saves library.

#### Private Functions

```rust
fn load_dna(dna_dir: &PathBuf) -> Result<Session>
```
Load all `.qor` files from a DNA directory into a fresh `Session`. Reads all files with `.qor` extension, sorts them alphabetically, parses each with `qor_core::parser::parse()`, and feeds statements into the session via `session.exec_statements()`. Returns the session with all DNA rules loaded.
Called at the start of each attempt in batch mode, and once in browse mode.

```rust
fn collect_answer_facts(session: &Session) -> Vec<(String, Vec<String>)>
```
Collect answer facts from QOR session. First checks if DNA specified `(answer-predicate <name>)` facts. If none found, defaults to looking for predicates named `"answer"` or starting with `"answer-"`. Returns `(predicate_name, all_args_as_strings)` pairs.
Called after reasoning in both batch and browse modes.

```rust
fn score_answer(
    answer: &[(String, Vec<String>)],
    expected: &[(String, Vec<String>)],
) -> f64
```
Score answer facts against expected facts. Handles `expected-*` prefix stripping. Handles arity mismatch: if answer has extra prefix args (e.g., grid ID), compares the suffix. Returns fraction of expected facts that were matched (0.0 to 1.0). Score >= 0.999 means solved.
Called in the main batch loop when both answer and expected are available.

```rust
fn extract_training_data(
    facts: &[Statement],
    target_pred: &str,
    source_pred: &str,
) -> Option<(Vec<Vec<Statement>>, Vec<Vec<Statement>>)>
```
Extract training pair data from item facts for the search/genesis engine. Reads `(train-pair input_id output_id)` structural markers. For each pair, collects input facts (renaming ID to `"ti"`) and output facts (converting `source_pred` to `target_pred`, renaming ID to `"ti"`). Adds `(test-input ti)` marker to each input set. Returns `None` if no train-pair facts found.
Called by `try_search()` and `try_genesis()`.

```rust
fn try_search(
    session: &Session,
    item_facts: &[Statement],
    budget_ms: u64,
    target_pred: &str,
    source_pred: &str,
) -> Option<(Vec<String>, Vec<search::ScoredRule>, usize, u64)>
```
Run the refinement search engine to find better rules. Extracts training data, collects seed rules (existing rules producing the target predicate), builds a base session with DNA rules + observation facts (`obs-*`, `detected-*`, `consistent-*`), and calls `search::refinement_search()`. Returns `(solution_texts, near_misses, mutations_tried, elapsed_ms)` or `None`.
Called in the main batch loop as fallback when genesis produces nothing.

```rust
fn try_genesis(
    session: &Session,
    item_facts: &[Statement],
    budget_ms: u64,
    target_pred: &str,
    source_pred: &str,
    library: &mut RuleLibrary,
    num_workers: usize,
) -> Option<(Vec<invent::Candidate>, u64)>
```
Run parallel genesis swarm to invent rules from scratch. Builds a clean base session with DNA rules + observation/metadata facts only (no puzzle data to avoid ID collisions). Calls `invent::genesis_swarm()` with the library for cross-pollination. Returns `(candidates, elapsed_ms)` or `None`.
Called in the main batch loop as the primary search strategy.

```rust
fn find_dna_predicate(session: &Session, dna_pred: &str, default: &str) -> String
```
Read a predicate name from DNA facts, with a default fallback. E.g., if DNA has `(target-predicate predict-cell)`, returns `"predict-cell"`. If not found, returns `default`.
Called to determine target and source predicates for search/genesis.

```rust
async fn execute_actions(
    session: &mut Session,
    lazy_browser: &mut Option<Browser>,
) -> bool
```
Execute any actions QOR produced. Pure plumbing. Reads `(browser-action ...)` facts via `act::next_action()`, executes up to 20 browser actions (safety valve). Lazy-launches browser on first use. Clears stale page facts before injecting new ones. Also reads `(http-fetch <url>)` facts and fetches via `ureq`, converting response to facts via `qor_bridge::feed()`. Clears `http-fetch` facts after execution. Returns `true` if any actions were executed.
Called in the batch loop after initial reasoning.

```rust
async fn browse_mode(dna_dir: &PathBuf, start_url: &str, max_steps: usize) -> Result<()>
```
Browse mode entry point. Loads DNA, injects `(start-url "<url>")`, probes for Chrome, falls back to HTTP-only if unavailable. Runs perceive-reason-act loop for up to `max_steps`: reads next action from QOR, executes on browser, clears stale page facts, feeds results back into session. Prints any answer/finding/result/extracted facts at the end.
Called from `main()` when `--browse` flag is present.

```rust
async fn http_fallback(session: &mut Session, url: &str, max_steps: usize) -> Result<()>
```
HTTP-only fallback when Chrome is not available. Fetches URL via `ureq`, converts body to QOR facts via `qor_bridge::feed()`, injects `page-url` and truncated `page-text` facts. Supports `Navigate` actions (via HTTP) and reports unsupported actions. Prints answer facts at the end.
Called from `browse_mode()` when Chrome is unavailable.

#### Connections

- Calls: `perceive::load_items()`, `learn::*` (all functions), `act::next_action()`, `act::execute()`, `browser::Browser::launch()`, `browser::Browser::probe()`, `qor_bridge::feed()`, `qor_runtime::invent::genesis_swarm()`, `qor_runtime::invent::optimal_worker_count()`, `qor_runtime::search::refinement_search()`, `qor_runtime::library::RuleLibrary::load()`, `qor_core::parser::parse()`
- Uses types: `types::AgentStats`, `types::DataItem`, `browser::Browser`, `act::AgentAction`

---

## Connection Diagram

```
main.rs
  |
  +-- perceive::load_items()          -- loads DataItems from files/dirs/URLs
  |     |
  |     +-- qor_bridge::feed_file()   -- auto-detect format, convert to Statements
  |     +-- qor_bridge::feed()        -- convert string content to Statements
  |     +-- qor_core::parser::parse() -- parse .qor files directly
  |
  +-- load_dna()                      -- loads .qor DNA files into Session
  |     +-- qor_core::parser::parse()
  |     +-- Session::exec_statements()
  |
  +-- learn::memory_stats()           -- reads memory.qor for display
  +-- learn::load_solved_ids()        -- reads memory.qor for skip set
  |
  +-- [BATCH LOOP per item]
  |     |
  |     +-- Session::exec()           -- inject item facts, attempt tracking
  |     +-- execute_actions()         -- QOR-driven browser/HTTP actions
  |     |     |
  |     |     +-- act::next_action()  -- read (browser-action ...) facts from Session
  |     |     +-- Browser::launch()   -- lazy Chrome launch
  |     |     +-- act::execute()      -- dispatch action to browser
  |     |     |     |
  |     |     |     +-- Browser::navigate() --> page::nav_to_facts()
  |     |     |     +-- Browser::click()    --> action_result_fact()
  |     |     |     +-- Browser::fill()     --> action_result_fact()
  |     |     |     +-- Browser::snapshot() --> page::snapshot_to_facts()
  |     |     |     +-- Browser::page_text()--> page::text_to_facts()
  |     |     |     +-- Browser::eval_js()  --> page::js_result_to_facts()
  |     |     |     +-- Browser::screenshot()--> action_result_fact()
  |     |     |     +-- Browser::network_list() --> page::network_to_facts()
  |     |     |
  |     |     +-- ureq::get()         -- HTTP fetch for (http-fetch ...) facts
  |     |     +-- qor_bridge::feed()  -- convert HTTP response to Statements
  |     |
  |     +-- collect_answer_facts()    -- read answer/predict facts from Session
  |     +-- score_answer()            -- compare answer vs expected
  |     |
  |     +-- [IF CORRECT]
  |     |     +-- learn::extract_features()
  |     |     +-- learn::extract_winning_strategy()
  |     |     +-- learn::record_solved()  -- writes to memory.qor
  |     |
  |     +-- [IF WRONG]
  |     |     +-- learn::extract_features()
  |     |     +-- learn::extract_tried()
  |     |     +-- try_genesis()           -- parallel rule invention
  |     |     |     +-- extract_training_data()
  |     |     |     +-- invent::genesis_swarm()
  |     |     +-- try_search()            -- refinement search (fallback)
  |     |     |     +-- extract_training_data()
  |     |     |     +-- search::refinement_search()
  |     |     +-- learn::save_winning_rules()  -- writes to rules_learned.qor
  |     |     +-- learn::record_attempt()      -- writes to memory.qor
  |
  +-- RuleLibrary::save()             -- persist library for future runs
  |
  +-- [BROWSE MODE]
        |
        +-- Browser::probe()          -- check Chrome availability
        +-- Browser::launch()         -- launch headless Chrome
        +-- [LOOP up to max_steps]
        |     +-- act::next_action()  -- read next QOR decision
        |     +-- act::execute()      -- execute on browser
        |     +-- Session::exec_statements() -- feed results back
        |
        +-- http_fallback()           -- fallback if no Chrome
              +-- ureq::get()
              +-- qor_bridge::feed()
              +-- act::next_action()  -- limited to Navigate in HTTP mode
```

---

## Public API Summary

### Entry Points

| Function | Module | Signature | Description |
|----------|--------|-----------|-------------|
| `main` | `main.rs` | `async fn main() -> Result<()>` | CLI entry point, dispatches batch/browse |

### Data Loading (perceive.rs)

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_items` | `fn load_items(source: &str) -> Result<Vec<DataItem>>` | Load data from dir/file/URL |

### Action System (act.rs)

| Function | Signature | Description |
|----------|-----------|-------------|
| `next_action` | `fn next_action(session: &Session) -> Option<AgentAction>` | Read highest-confidence browser-action from QOR |
| `execute` | `async fn execute(action: &AgentAction, browser: &mut Browser) -> Result<Vec<Statement>, String>` | Execute action on browser, return facts |

### Browser (browser.rs)

| Method | Signature | Description |
|--------|-----------|-------------|
| `Browser::probe` | `fn probe() -> ProbeResult` | Check Chrome availability |
| `Browser::launch` | `async fn launch() -> Result<Self, String>` | Launch Chrome, connect CDP |
| `Browser::navigate` | `async fn navigate(&mut self, url: &str) -> Result<(String, String), String>` | Navigate to URL |
| `Browser::snapshot` | `async fn snapshot(&mut self) -> Result<Vec<PageElement>, String>` | Get accessibility tree |
| `Browser::click` | `async fn click(&mut self, uid: &str) -> Result<(), String>` | Click element by UID |
| `Browser::fill` | `async fn fill(&mut self, uid: &str, value: &str) -> Result<(), String>` | Fill form field |
| `Browser::page_text` | `async fn page_text(&mut self) -> Result<String, String>` | Get visible page text |
| `Browser::eval_js` | `async fn eval_js(&mut self, script: &str) -> Result<String, String>` | Execute JavaScript |
| `Browser::screenshot` | `async fn screenshot(&mut self, full_page: bool) -> Result<String, String>` | Capture screenshot (base64 PNG) |
| `Browser::network_list` | `async fn network_list(&mut self, filter: Option<&str>) -> Result<Value, String>` | List network requests |

### Page Conversion (page.rs)

| Function | Signature | Description |
|----------|-----------|-------------|
| `nav_to_facts` | `fn nav_to_facts(url: &str, title: &str) -> Vec<Statement>` | URL + title to facts |
| `text_to_facts` | `fn text_to_facts(text: &str) -> Vec<Statement>` | Page text to fact (truncated at 4000 chars) |
| `snapshot_to_facts` | `fn snapshot_to_facts(elements: &[PageElement]) -> Vec<Statement>` | Accessibility elements to facts |
| `network_to_facts` | `fn network_to_facts(entries: &Value) -> Vec<Statement>` | Network entries to facts |
| `js_result_to_facts` | `fn js_result_to_facts(script: &str, result: &str) -> Vec<Statement>` | JS eval result to fact |

### Learning (learn.rs)

| Function | Signature | Description |
|----------|-----------|-------------|
| `extract_features` | `fn extract_features(session: &Session) -> Vec<String>` | Extract obs-*/detected-*/consistent-* feature names |
| `extract_tried` | `fn extract_tried(session: &Session) -> Vec<String>` | Extract tried strategy names |
| `extract_winning_strategy` | `fn extract_winning_strategy(session: &Session) -> String` | Get most specific winning strategy |
| `record_attempt` | `fn record_attempt(item_id, attempt, score, best_source, features, tried, path) -> Result<usize>` | Record failed attempt (levels 1-4) |
| `record_solved` | `fn record_solved(item_id, winning_strategy, features, path) -> Result<usize>` | Record solved item (level 5, replaces failures) |
| `save_winning_rules` | `fn save_winning_rules(rules, item_id, rules_path) -> Result<usize>` | Save discovered rules (deduped) |
| `load_solved_ids` | `fn load_solved_ids(memory_path: &Path) -> HashSet<String>` | Load set of solved item IDs |
| `memory_stats` | `fn memory_stats(memory_path: &Path) -> (usize, usize, usize)` | Get (solved, failed, total) counts |

### Types (types.rs)

| Type | Kind | Description |
|------|------|-------------|
| `DataItem` | struct | Generic data item with id, facts, optional expected output |
| `AgentStats` | struct | Performance tracking (attempted, correct, wrong, no_answer, learned, skipped) |
| `AgentAction` | enum | Browser action variants (Navigate, Click, Fill, Snapshot, etc.) |
| `PageElement` | struct | Accessibility tree element (role, name, uid, properties) |
| `Browser` | struct | CDP-connected browser session |
| `ProbeResult` | struct | Chrome availability check result |

### QOR Fact Predicates Produced

| Predicate | Source | Example |
|-----------|--------|---------|
| `page-url` | `page::nav_to_facts` | `(page-url "https://example.com")` |
| `page-title` | `page::nav_to_facts` | `(page-title "Example")` |
| `page-text` | `page::text_to_facts` | `(page-text "Hello world...")` |
| `page-element` | `page::snapshot_to_facts` | `(page-element s1 link "About" "https://...")` |
| `page-content` | `page::snapshot_to_facts` | `(page-content heading "Welcome")` |
| `page-link-count` | `page::snapshot_to_facts` | `(page-link-count 5)` |
| `page-button-count` | `page::snapshot_to_facts` | `(page-button-count 3)` |
| `page-form-count` | `page::snapshot_to_facts` | `(page-form-count 1)` |
| `network-request` | `page::network_to_facts` | `(network-request "r1" fetch "https://api..." 150)` |
| `js-result` | `page::js_result_to_facts` | `(js-result "document.title" "Example")` |
| `action-result` | `act::action_result_fact` | `(action-result clicked "s5")` |
| `action-error` | `main.rs` | `(action-error "Navigation failed: ...")` |
| `browser-unavailable` | `main.rs` | `(browser-unavailable true)` |

### QOR Fact Predicates Consumed

| Predicate | Reader | Purpose |
|-----------|--------|---------|
| `browser-action` | `act::next_action` | QOR's decision: what action to take |
| `http-fetch` | `execute_actions` | QOR requests HTTP fetch of a URL |
| `answer-predicate` | `collect_answer_facts` | DNA configures which predicates are answers |
| `target-predicate` | `find_dna_predicate` | DNA configures the target predicate for search |
| `source-predicate` | `find_dna_predicate` | DNA configures the source predicate for search |
| `train-pair` | `extract_training_data` | Structural marker for training input/output pairs |
| `save-known-transform` | `learn::extract_winning_strategy` | Strategy signal from DNA reasoning |
| `detected-*` | `learn::extract_features/tried` | Feature/strategy detection from DNA |
| `obs-*` | `learn::extract_features` | Observation features from DNA |
| `consistent-*` | `learn::extract_features` | Consistency observations from DNA |

### Memory File Format (memory.qor)

```qor
;; ══ item_abc ══
(solved item_abc color-remap) <0.99, 0.99>
(item-feature item_abc obs-same-size) <0.90, 0.90>

;; ══ item_xyz ══
(wrong-attempt item_xyz 3) <0.99, 0.99>
(item-feature item_xyz obs-reflect-h) <0.90, 0.90>
(tried-and-failed item_xyz identity) <0.95, 0.95>
(tried-and-failed item_xyz reflect-h) <0.95, 0.95>
(best-partial item_xyz genesis-template 0.75) <0.90, 0.90>
```

### Test Coverage

- `act.rs`: 8 tests (parse actions, empty session)
- `browser.rs`: 2 tests (probe, interactive roles)
- `learn.rs`: 7 tests (record attempt, solved replaces failure, upsert no duplication, multiple items, load solved IDs, memory stats, rule dedup)
- `page.rs`: 5 tests (nav, empty text, normal text, snapshot, network, JS result)
- **Total: 22 tests**

---

## ARC-AGI-3 Arcade Agent (`arc3.rs`)

### Overview

`arc3.rs` is a standalone binary that plays ARC-AGI-3 arcade games via the competition API. Rust is pure plumbing (API calls, frame extraction, action execution). All game strategy lives in QOR DNA rules (`arcade.qor`).

### CLI Usage

```
cargo run --release --bin arc3 [--game PREFIX] [--scorecard ID] [--close-scorecard]
```

| Flag | Description |
|------|-------------|
| `--game PREFIX` | Only play games whose ID starts with PREFIX (e.g., `--game ft`) |
| `--scorecard ID` | Use a specific scorecard ID (saved for future runs) |
| `--close-scorecard` | Close the scorecard after this run (start fresh next time) |

```bash
cargo run --release --bin arc3                          # Play all games with known combos
cargo run --release --bin arc3 -- --game ft             # Play only ft* games
cargo run --release --bin arc3 -- --scorecard abc-123   # Use specific scorecard
cargo run --release --bin arc3 -- --close-scorecard     # Close scorecard when done
```

### Persistent Scorecard

The scorecard stays **open across runs** so the score accumulates. Already-won games are automatically skipped.

```
Run 1:  Opens new scorecard → plays games → wins some → scorecard stays OPEN
Run 2:  Reuses same scorecard → skips won games → plays remaining → score grows
Run 3:  Same → score keeps growing
```

- Scorecard ID saved to `dna/puzzle_solver/active_scorecard.txt`
- Normal exit or Ctrl+C keeps scorecard open
- `--close-scorecard` closes it and deletes the file (next run starts fresh)
- `--scorecard ID` overrides with a specific ID

### How It Works: Try → Learn → Replay

```
PERCEIVE → REASON → ACT → LEARN
   │          │        │       │
   │ extract  │ QOR    │ send  │ save winning combo
   │ grid     │ chains │ action│ on level-up
   │ from     │ rules  │ to    │
   │ API      │ to     │ API   │ save failed attempt
   │ frame    │ pick   │       │ on game-over
   │          │ action │       │
```

1. **Perceive**: Get frame from API, extract grid, detect objects and edges
2. **Reason**: QOR forward chains on facts to select an action
3. **Act**: Send action to API, get next frame
4. **Learn**: On level-up, save the winning action sequence. On failure, save the failed attempt.

**Replay on next run**: Solved levels are replayed instantly from saved combos (~1 second each), leaving maximum time for unsolved levels.

### Data Persistence

- `game_rules.json` — per-game state: winning combos, failed attempts, grid signatures, game type
- `active_scorecard.txt` — current scorecard ID
- `arcade.qor` — all game strategy rules (QOR DNA)
- `arcade_memory.qor` / `arcade_rules_learned.qor` — learned knowledge
