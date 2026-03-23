# qor-bridge -- Comprehensive API Reference

## 1. Crate Overview

`qor-bridge` is the universal data ingestion and knowledge pipeline for QORlang (Quantified Ontological Reasoning Language). It accepts raw data in any supported format -- JSON, CSV, key-value, plain text, or Parquet -- auto-detects the format, parses it into QOR `Statement` facts, and enriches those facts through a multi-stage learning pipeline that produces condensed knowledge rather than retaining raw data.

The crate also provides grid perception for ARC-AGI puzzles, DNA personality profiles, natural language understanding, web intelligence, real-time perception, template-based rule instantiation, binary knowledge base encoding, and a reasoning-first solve loop.

### Dependencies (from `D:\QOR-LANG\qor\qor-bridge\Cargo.toml`)

| Dependency | Version | Purpose |
|------------|---------|---------|
| `qor-core` | workspace | Neuron, TruthValue, Parser, Unify |
| `qor-runtime` | workspace | NeuronStore, Chain, Eval/Session |
| `serde` | 1 (derive) | Serialization (DNA profiles) |
| `serde_json` | 1 | JSON parsing |
| `csv` | 1 | CSV parsing |
| `regex` | 1 | Text pattern extraction |
| `flate2` | 1 | Gzip decompression (OEIS) |
| `nom` | 7 | Parser combinators (KV format) |
| `parquet` | 54 | Parquet file reading |
| `spider` | 2 (optional) | Web crawling (feature: `web`) |
| `tokio` | 1 (optional) | Async runtime (feature: `web`) |
| `ureq` | 2 | Synchronous HTTP client |

**Feature flags:** `web` = `spider` + `tokio` (enables `web_search` module and spider-rs crawler)

### Module Map

```
lib.rs                 -- Crate root: feed(), feed_file(), feed_as(), enrich()
detect.rs              -- Format auto-detection
sanitize.rs            -- Symbol sanitization + value inference
json.rs                -- JSON -> QOR facts
csv_ingest.rs          -- CSV -> QOR facts
kv.rs                  -- Key-value -> QOR facts (nom)
text.rs                -- Text -> QOR facts (regex)
parquet_ingest.rs      -- Parquet -> QOR facts
context.rs             -- Domain detection + auto-context rules + auto-analysis
learn.rs               -- Learning engine (stats, patterns, co-occurrences, rules)
language.rs            -- Language understanding (tokenize, grammar, responses)
llguidance.rs          -- LLM integration stubs + template-based summarize()
dna.rs                 -- DNA personality profile system
grid.rs                -- 2D grid perception (ARC-AGI)
template.rs            -- Rule template instantiation engine
text_hint.rs           -- NL -> semantic hint facts
web_rules.rs           -- Text -> candidate rule/fact extraction
web_fetch.rs           -- Web crawling + extraction + cache + domain policy
web_search.rs          -- Web search pipeline (feature-gated: web)
memory_graph.rs        -- Puzzle reasoning memory as QOR graph facts
kb_build.rs            -- Binary triple format encoder
oeis.rs                -- OEIS integer sequence parser
perceive.rs            -- Real-time perception engine
dharmic.rs             -- Sacred text ingestion pipeline
solve.rs               -- Reasoning-first solve loop (6 phases)
```

---

## 2. File-by-File Breakdown

---

### `lib.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\lib.rs`

The crate entry point. Provides three public functions for data ingestion and an internal `enrich()` pipeline.

**Re-exports:**
- `pub use detect::DataFormat`
- `pub use context::DataDomain`

#### Public Functions

```rust
pub fn feed(data: &str) -> Result<Vec<Statement>, String>
```
Primary entry point. Auto-detects format via `detect::detect_format()`, parses via `feed_raw()`, then enriches through the full learning pipeline via `enrich()`. Returns learned knowledge, not raw data.

```rust
pub fn feed_file(path: &std::path::Path) -> Result<Vec<Statement>, String>
```
File-based ingestion. Handles binary formats (parquet via extension check) and text formats (reads file, auto-detects, parses). Calls `enrich()` on results.

```rust
pub fn feed_as(data: &str, format: DataFormat) -> Result<Vec<Statement>, String>
```
Explicit format ingestion. Bypasses auto-detection, uses the specified `DataFormat`. Calls `enrich()` on results.

#### Private Functions

```rust
fn feed_raw(data: &str, format: DataFormat) -> Result<Vec<Statement>, String>
```
Raw parse only (no enrichment). Dispatches to `json::from_json`, `csv_ingest::from_csv`, `kv::from_kv`, or `text::from_text`.

```rust
fn enrich(facts: Vec<Statement>) -> Vec<Statement>
```
The full learning pipeline: partitions grid facts from data facts, then runs `context::auto_context()`, `context::auto_analysis()`, `learn::learn()`, `context::auto_queries()`, and `grid::deep_grid_perception()`. Returns learned knowledge + context + queries (raw data discarded).

**Calls:** `detect::detect_format`, `json::from_json`, `csv_ingest::from_csv`, `kv::from_kv`, `text::from_text`, `parquet_ingest::from_parquet_file`, `context::auto_context`, `context::auto_analysis`, `learn::learn`, `context::auto_queries`, `grid::deep_grid_perception`

**Called by:** CLI commands (`qor feed`, `qor think`), external crates

---

### `detect.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\detect.rs`

Format auto-detection with priority: JSON > CSV > KV > Text.

#### Public Types

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataFormat {
    Json,      // Starts with { or [
    Csv,       // 2+ lines, consistent comma counts (>70% match)
    KeyValue,  // >50% lines match key=value or key: value
    Text,      // Fallback
}
```
Implements `Display` (returns "JSON", "CSV", "Key-Value", "Text").

#### Public Functions

```rust
pub fn detect_format(data: &str) -> DataFormat
```
Auto-detects input format. JSON detected by leading `{` or `[`. CSV needs 2+ lines with 70%+ matching comma counts. KV needs 50%+ lines with key=value or key: value patterns. Empty input returns `Text`.

**Called by:** `lib::feed`, `lib::feed_file`

---

### `sanitize.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\sanitize.rs`

Shared helpers for all ingestion parsers. Converts raw external data into valid QOR symbols and typed neurons.

#### Public Constants

```rust
pub const INGESTED_STRENGTH: f64 = 0.90;
```
Default truth value strength for external data (lower than user-asserted facts at 0.99).

#### Public Functions

```rust
pub fn ingested_tv() -> TruthValue
```
Builds the default `TruthValue` for ingested facts using `TruthValue::from_strength(0.90)`.

```rust
pub fn sanitize_symbol(raw: &str) -> String
```
Sanitizes raw strings into valid QOR symbol names: lowercase, replace spaces/underscores/dots with hyphens, collapse consecutive hyphens, prefix digits with `x-`, return "unknown" if empty.

```rust
pub fn infer_value(raw: &str) -> Neuron
```
Type inference for raw string values. Tries in order: bool ("true"/"false"/"yes"/"no") -> int -> float -> null/nil -> symbol-like -> string fallback.

```rust
pub fn make_fact(parts: Vec<Neuron>) -> Statement
```
Helper to create a `Statement::Fact` with `Expression(parts)`, ingested truth value, no decay.

**Called by:** `json.rs`, `csv_ingest.rs`, `kv.rs`, `text.rs`, `context.rs`, `perceive.rs`

---

### `json.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\json.rs`

Converts JSON to QOR facts. Handles objects, arrays of objects, nested objects (flattened with hyphenated prefix), and grid-structured data (2D arrays of numbers).

#### Public Functions

```rust
pub fn from_json(json: &str) -> Result<Vec<Statement>, String>
```
Parses a JSON string into QOR facts. Objects: each key-value becomes `(predicate entity value)`. Arrays of objects: entity auto-detected from name/id/key/title/label fields. Grid data (2D number arrays): dispatches to `Grid::to_statements()`. Nested objects: flattened with hyphenated path prefix. Nulls skipped.

**Calls:** `sanitize::sanitize_symbol`, `sanitize::infer_value`, `sanitize::make_fact`, `grid::Grid::from_vecs`, `grid::Grid::to_statements`

**Called by:** `lib::feed_raw`, `perceive::parse_json_to_facts`

---

### `csv_ingest.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\csv_ingest.rs`

Converts CSV data with headers into QOR facts.

#### Public Functions

```rust
pub fn from_csv(csv_data: &str) -> Result<Vec<Statement>, String>
```
First row = headers (predicates). Entity column auto-detected: prefers "name", "id", "key"; defaults to column 0. Each cell becomes `(header entity value)` with type inference. Uses `csv::ReaderBuilder` with flexible/trim options. Empty cells skipped.

**Calls:** `sanitize::sanitize_symbol`, `sanitize::infer_value`, `sanitize::ingested_tv`

**Called by:** `lib::feed_raw`

---

### `kv.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\kv.rs`

Key-value parsing using nom parser combinators.

#### Public Functions

```rust
pub fn from_kv(data: &str) -> Result<Vec<Statement>, String>
```
Supports three separator styles: `key=value`, `key: value` (space after colon required), `key\tvalue`. Comments (`#`, `//`, `;`) and empty lines skipped. Each pair becomes `(key value)` with type inference. Returns error if no pairs found.

**Calls:** `sanitize::sanitize_symbol`, `sanitize::infer_value`, `sanitize::ingested_tv`

**Called by:** `lib::feed_raw`

---

### `text.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\text.rs`

Natural language to QOR facts via regex pattern extraction.

#### Public Functions

```rust
pub fn from_text(text: &str) -> Result<Vec<Statement>, String>
```
Extracts facts from plain text using ordered patterns (most specific first):
- "X is a Y" -> `(is-a X Y)`
- "X are Y" -> `(is X Y)`
- "X is Y" -> `(is X Y)`
- "X has a Y" -> `(has X Y)`
- "X has Y" -> `(has X Y)`
- "X can Y" -> `(can X Y)`
- "X contains Y" -> `(contains X Y)`
- Fallback: `(text "raw sentence")`

Case-insensitive, strips trailing periods. First matching pattern wins per line.

**Calls:** `sanitize::sanitize_symbol`, `sanitize::infer_value`, `sanitize::ingested_tv`

**Called by:** `lib::feed_raw`

---

### `parquet_ingest.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\parquet_ingest.rs`

Binary Parquet file reader.

#### Public Functions

```rust
pub fn from_parquet_file(path: &Path) -> Result<Vec<Statement>, String>
```
Reads a Parquet file and converts each row into QOR facts. Each column becomes a predicate: `(column-name row-N value)`. Maps all Parquet field types (Bool, Byte, Short, Int, Long, UByte, UShort, UInt, ULong, Float, Double, Str, Bytes, TimestampMillis, TimestampMicros) to QOR values. Caps at 10,000 rows.

**Calls:** `sanitize::ingested_tv`, `sanitize::sanitize_symbol`

**Called by:** `lib::feed_file`

---

### `context.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\context.rs`

Domain detection, auto-context rule generation, analysis, and query generation.

#### Public Types

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum DataDomain {
    Market,   // open, close, rsi, macd, volume, etc.
    Medical,  // symptom, diagnosis, temperature, etc.
    News,     // title, headline, author, content, etc.
    Social,   // user, post, comment, likes, etc.
    Science,  // experiment, measurement, sample, etc.
    Bio,      // species, genus, habitat, etc.
    Config,   // env, port, debug, cache, etc.
    General,  // fallback (<2 keyword matches)
}
```
Implements `Display`. Each domain has a keyword list (e.g., `MARKET_KEYWORDS` with 28 entries).

#### Public Functions

```rust
pub fn detect_domain(facts: &[Statement]) -> DataDomain
```
Scores fact predicates against domain keyword lists. Needs >= 2 matches to claim a domain. Falls back to `General`.

```rust
pub fn auto_context(facts: &[Statement]) -> Vec<Statement>
```
Generates domain-specific context rules: `(domain X)` fact, `(purpose X)` facts, and structural rules like `(price-bar $x) if (close $x $v)`. Each domain has its own rule set (market, medical, news, social, science, bio, config, general).

```rust
pub fn auto_analysis(facts: &[Statement]) -> Vec<Statement>
```
Analyzes raw numeric values to generate analysis facts. Market domain: RSI > 70 = overbought, < 30 = oversold; ADX > 25 = trending; MACD vs Signal = bullish/bearish cross; close vs MA-200 = above/below trend; volume > 2x mean = high-volume. Medical domain: temperature > 39.5 = high-fever, > 38.0 = fever-detected.

```rust
pub fn auto_queries(facts: &[Statement]) -> Vec<Statement>
```
Generates domain-specific queries that auto-execute. Always includes `(domain $x)` and `(purpose $x)`. Market adds buy-signal, sell-signal, anomaly, pattern-rate, co-occur, etc. Each domain adds relevant queries.

#### Crate-Internal Functions

```rust
pub(crate) fn extract_predicates(facts: &[Statement]) -> HashSet<String>
```
Extracts all unique first-symbol predicates from facts.

```rust
pub(crate) fn extract_float_values(facts: &[Statement], predicate: &str) -> Vec<(String, f64)>
```
Extracts `(entity_name, float_value)` pairs for a given predicate from facts.

**Called by:** `lib::enrich`, `learn::learn`

---

### `learn.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\learn.rs`

The learning engine. Transforms raw data + analysis into condensed knowledge.

#### Public Types

```rust
#[derive(Debug, Clone)]
pub struct Stats {
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
}
```

#### Public Functions

```rust
pub fn learn(facts: &[Statement], analysis: &[Statement]) -> Vec<Statement>
```
Core learning function. Pipeline:
1. Semantic knowledge: indicator definitions, thresholds, meanings (domain-specific)
2. Guard rules from thresholds: `(overbought $x) if (rsi $x $v) (> $v 70)`
3. Statistical summaries per numeric predicate (count, mean, min, max, std-dev)
4. Pattern frequency rates (e.g., overbought 4% of 10000 entities)
5. Co-occurrence analysis (which patterns appear together, >= 5 entities)
6. Learned rules from strong co-occurrences (>= 70% strength, >= 10 base)
7. Single-pattern learned rules for high-frequency patterns (>= 5%, >= 10 count)
8. Metadata: data-points count, learned-at date

```rust
pub fn threshold_rules(knowledge: &[Statement]) -> Vec<Statement>
```
Reads `(threshold indicator pattern value)` facts and generates guard-based rules like `(overbought $x) if (rsi $x $v) (> $v 70) <0.95, 0.90>`. Maps pattern names to comparison operators (overbought = Gt, oversold = Lt, etc.).

```rust
pub fn indicator_knowledge(domain: &DataDomain) -> Vec<Statement>
```
Generates semantic indicator definitions for detected domains. Market: RSI, ADX, MACD, ATR, MA-200, etc. with thresholds, ranges, and meanings. Medical: temperature with fever thresholds and meanings. Other domains: empty vec.

```rust
pub fn merge_learned(existing: &[Statement], new: &[Statement]) -> Vec<Statement>
```
Incremental learning: merges new knowledge with existing brain knowledge. Merge strategies: stat-count = sum, stat-mean = average, stat-min = take minimum, stat-max = take maximum, stat-std-dev = keep newer, pattern-rate = sum counts/totals, co-occur = sum counts, data-points = sum, rules = TruthValue revision (confidence grows), old-only facts persist, new-only facts added.

**Calls:** `context::detect_domain`, `context::extract_predicates`, `context::extract_float_values`

**Called by:** `lib::enrich`

---

### `language.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\language.rs`

Language understanding system. QOR learns language by reasoning about grammar rules and vocabulary as facts.

#### Public Types

```rust
#[derive(Debug, Default)]
pub struct BrainContext {
    pub pattern_rates: HashMap<String, (i64, i64)>,           // pattern -> (count, total)
    pub strong_cooccur: HashMap<String, Vec<(String, f64)>>,  // pattern -> [(co_pattern, ratio)]
    pub thresholds: HashMap<(String, String), i64>,           // (indicator, condition) -> value
    pub personality: HashMap<String, Vec<String>>,             // key -> [text options]
}
```

**Methods:**
```rust
impl BrainContext {
    pub fn personality_text(&self, key: &str, seed: usize, default: &str) -> String
}
```
Picks a personality text variant for a given key, rotating through variants using `seed`. Falls back to `default`.

#### Public Functions

```rust
pub fn extract_brain_context(facts: &[StoredNeuron]) -> BrainContext
```
Scans stored facts for pattern-rate, co-occur (>50% only), threshold, and personality facts. Builds a `BrainContext` for enriching chat responses.

```rust
pub fn load_language_dir(dir: &Path) -> Vec<Statement>
```
Loads all `.qor` files from a language directory (e.g., `language/en/`). Falls back to `language_knowledge_bootstrap()` if empty/missing.

```rust
pub fn available_languages(language_root: &Path) -> Vec<String>
```
Lists available language subfolder names (e.g., `["en", "es"]`) that contain at least one `.qor` file.

```rust
pub fn language_knowledge_bootstrap() -> Vec<Statement>
```
Built-in bootstrap language knowledge (vocabulary, grammar rules, response rules). Includes 16+ vocab facts, 6+ grammar rules, 7+ response rules.

```rust
pub fn save_to_dictionary(dir: &Path, fact_line: &str)
```
Appends a learned fact line to `dictionary.qor` in the language directory.

```rust
pub fn tokenize(input: &str) -> Vec<Statement>
```
Tokenizes user input into `(input N word)` facts. Lowercases, splits on whitespace/punctuation, strips quotes/question marks.

```rust
pub fn parse_teach_pattern(input: &str) -> Option<(String, String)>
```
Detects teaching patterns: "X means Y", "define X as Y", "remember X is Y". Returns `(topic, description)` with sanitized hyphenated values.

```rust
pub fn format_response(responses: &[Vec<String>], ctx: Option<&BrainContext>) -> String
```
Formats response facts into natural language. Handles: greet, help, define (with rarity/threshold/co-occurrence context), explain, list-pattern (with percentage), list-indicator, list-meaning, count, farewell, curious. Enriches with `BrainContext` when provided. Suppresses "curious" fallback when specific responses exist. Deduplicates output lines.

**Called by:** `qor-cli` chat command

---

### `llguidance.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\llguidance.rs`

LLM integration stubs (Phase 5) and a working template-based summarizer.

#### Public Types

```rust
#[derive(Debug, Clone)]
pub struct GuidanceConfig {
    pub max_tokens: usize,    // default: 256
    pub temperature: f64,     // default: 0.3
    pub min_confidence: f64,  // default: 0.5
}

#[derive(Debug, Clone)]
pub struct Hypothesis {
    pub statement: Statement,
    pub estimated_tv: TruthValue,
    pub rationale: String,
}

#[derive(Debug)]
pub struct GuidanceResult {
    pub hypotheses: Vec<Hypothesis>,
    pub summary: String,
}
```

#### Public Functions (Stubs)

```rust
pub fn nl_to_qor(_text: &str, _config: &GuidanceConfig) -> Result<Vec<Statement>, String>
pub fn qor_to_nl(_statements: &[Statement], _config: &GuidanceConfig) -> Result<String, String>
pub fn generate_hypotheses(_knowledge: &[Statement], _config: &GuidanceConfig) -> Result<GuidanceResult, String>
```
All return `Err("Phase 5: llguidance integration not yet implemented")`.

#### Public Functions (Working)

```rust
pub fn summarize(statements: &[Statement]) -> String
```
Template-based natural language summary of brain knowledge. Collects meanings, domain, data points, stats, pattern rates (with meanings), co-occurrences (>=50% strength), and rules. No LLM required.

**Called by:** `qor-cli` summary command

---

### `dna.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\dna.rs`

DNA personality profile system. Reads profession DNA profiles (JSON + knowledge.txt) and converts to QOR statements.

#### Public Types

```rust
#[derive(Debug, Deserialize)]
pub struct DnaProfile {
    pub id: String,
    pub name: String,
    pub category: String,
    pub archetype: String,
    pub tagline: String,
    pub voice: DnaVoice,
    pub origin: DnaOrigin,
    pub psychological_layers: DnaPsychLayers,
    pub personality: HashMap<String, DnaTrait>,       // v1 format
    pub personality_traits: Vec<DnaTraitV2>,           // v2 format
    pub triggers: DnaTriggers,
    pub speech: DnaSpeech,
    pub contradiction_core: String,
    pub edges: Vec<DnaEdge>,
    pub system_prompt: String,
    pub sources: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct DnaOrigin { pub summary: String, pub core_wound: String, pub formative_belief: String }

#[derive(Debug, Default, Deserialize)]
pub struct DnaPsychLayers { pub surface: Vec<String>, pub middle: Vec<String>, pub deep: Vec<String> }

#[derive(Debug, Default, Deserialize)]
pub struct DnaTriggers { pub best_self: Vec<String>, pub shadow_self: Vec<String>, pub conflict_response: String }

#[derive(Debug, Default, Deserialize)]
pub struct DnaVoice { pub texture: String, pub internal_motto: String }

#[derive(Debug, Deserialize)]
pub struct DnaTrait { pub level: String, pub confidence: f64, pub shadow: String }

#[derive(Debug, Deserialize)]
pub struct DnaTraitV2 { pub name: String, pub source: String, pub level: u32, pub description: String }

#[derive(Debug, Default, Deserialize)]
pub struct DnaSpeech { pub signature_phrases: Vec<String>, pub never_says: Vec<String> }

#[derive(Debug, Deserialize)]
pub struct DnaEdge { pub source: String, pub relation: String, pub target: String, pub weight: f64 }
```

#### Public Functions

```rust
pub fn find_dna_by_keywords(dna_dir: &Path, query: &str) -> Vec<(String, String, usize)>
```
Searches DNA profiles by keyword against `index.json`. Returns `(id, name, score)` sorted by relevance. Scores: exact id = 3, id/name contains = 2, keyword match = 1.

```rust
pub fn load_dna(dna_dir: &Path, id: &str) -> Result<DnaProfile, String>
```
Loads a DNA profile from `dna_dir/<id>/<id>.json` or `dna_dir/<id>/dna.json`.

```rust
pub fn load_knowledge(dna_dir: &Path, id: &str) -> Result<String, String>
```
Loads knowledge from `dna_dir/<id>/knowledge.txt` or `knowledge.md`.

```rust
pub fn load_meta_rules(workspace_root: &Path) -> Vec<Statement>
```
Loads shared meta-rules from `workspace_root/meta/*.qor` (universal reasoning rules: transitivity, symmetry, etc.). Returns empty vec if `meta/` does not exist.

```rust
pub fn load_templates_text(dna_dir: &Path, id: &str) -> Option<String>
```
Loads `templates.qor` raw text from `dna_dir/<id>/templates.qor`. Returns `None` if file does not exist.

```rust
pub fn load_rules(dna_dir: &Path, id: &str) -> Vec<Statement>
```
Loads and parses `rules.qor` from `dna_dir/<id>/rules.qor`. Returns parsed QOR statements. Returns empty vec if no file.

```rust
pub fn available_dna(dna_dir: &Path) -> Vec<(String, String)>
```
Lists available DNA profiles as `(id, display_name)` pairs. Looks for subdirectories with `<name>/<name>.json` or `<name>/dna.json`.

```rust
pub fn dna_to_statements(profile: &DnaProfile) -> Vec<Statement>
```
Converts a `DnaProfile` into QOR statements: identity (dna-id, dna-name, dna-archetype, dna-tagline, dna-category), voice (dna-voice, dna-motto), traits (dna-trait with confidence as TV strength), speech phrases (personality greet), never-says, edges (dna-edge with weight as TV strength), origin, psychological layers, triggers, contradiction, system prompt, and v2 sources/traits.

```rust
pub fn parse_knowledge(text: &str) -> Vec<Statement>
```
Parses knowledge.txt (markdown with `## sections` and `- bullets`) into `(meaning section-topic "bullet text") <0.90, 0.85>` facts. Truncates bullets to ~200 chars at sentence boundaries.

```rust
pub fn save_dna_qor(dna_dir: &Path, id: &str) -> Result<(PathBuf, usize), String>
```
Converts a DNA profile to `.qor` format and saves to `dna_dir/<id>/<id>.qor`. Includes profile statements + knowledge + rules. Returns `(path, statement_count)`.

```rust
pub fn convert_all(dna_dir: &Path) -> Vec<(String, PathBuf, usize)>
```
Converts ALL DNA profiles to `.qor` files. Returns `(id, path, count)` per profile.

**Called by:** `qor-cli` chat command, solve pipeline

---

### `grid.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\grid.rs`

2D grid perception for ARC-AGI puzzles. Very large file (~1200 lines).

#### Public Types

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    pub rows: usize,
    pub cols: usize,
    pub cells: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GridObject {
    pub id: usize,
    pub color: u8,
    pub cells: Vec<(usize, usize)>,
}
```

**GridObject methods:**
```rust
pub fn bbox(&self) -> (usize, usize, usize, usize) // (min_row, min_col, max_row, max_col)
```

#### Grid Methods

```rust
pub fn from_vecs(cells: Vec<Vec<u8>>) -> Result<Grid, String>
```
Creates a grid from nested vectors. Returns error if rows are ragged.

```rust
pub fn cell_at(&self, row: usize, col: usize) -> Option<u8>
```
Gets cell value at (row, col).

```rust
pub fn dimensions(&self) -> (usize, usize)
```
Returns (rows, cols).

```rust
pub fn objects(&self) -> Vec<GridObject>
```
Detects connected components via BFS flood fill. Color 0 = background (skipped). 4-connected neighbors.

```rust
pub fn to_statements(&self, grid_id: &str) -> Vec<Statement>
```
Converts grid to comprehensive QOR facts: grid-size, grid-cell (all cells), grid-neighbor (4-connected), grid-object, grid-obj-cell, grid-obj-bbox, bbox-width, bbox-height, spatial relationships (above, left-of, h-aligned, v-aligned, contains, same-color, bbox-overlap), color distribution (has-color, color-count, color-cell-count), content-bbox, object-count, separator-row/col, grid-region, region-cell, grid-diag-neighbor.

```rust
pub fn equals(&self, other: &Grid) -> bool
pub fn crop(&self, r1: usize, c1: usize, r2: usize, c2: usize) -> Grid
pub fn reflect_h(&self) -> Grid
pub fn reflect_v(&self) -> Grid
pub fn rotate_90(&self) -> Grid
pub fn rotate_180(&self) -> Grid
pub fn rotate_270(&self) -> Grid
pub fn transpose(&self) -> Grid
```
Grid transforms for comparison and manipulation.

#### Static Methods

```rust
pub fn compare_pair(input: &Grid, output: &Grid, in_id: &str, out_id: &str) -> Vec<Statement>
```
Compares input/output grid pairs and emits comprehensive comparison facts: cell-kept, cell-removed, cell-added, recolored, pair-identity, pair-reflect-h/v, pair-rotate-*, pair-shift, pair-crop, pair-scale-up, pair-symmetry-complete, pair-flood-fill, color-remap, pair-composed, pair-gravity-down, pair-row-fill, pair-col-fill, pair-cross-fill, pair-region-fill-seed.

#### Module-Level Public Functions

```rust
pub fn deep_grid_perception(grid_facts: Vec<Statement>) -> Vec<Statement>
```
Deep grid perception pipeline for ARC-like data: remaps IDs, adds train-pair/test-input facts, runs `compare_pair()` and `compute_observations()` for cross-pair analysis. `compute_observations()` caches `objects()` results (expensive flood-fill BFS) upfront before running observation checks, avoiding repeated flood-fill calls per pair.

**Called by:** `lib::enrich`, `json::from_json`, `solve.rs`

---

### `template.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\template.rs`

Rule template instantiation engine. Fills QOR rule templates from cross-pair observations.

#### Public Types

```rust
#[derive(Debug, Clone)]
pub struct Template {
    pub name: String,
    pub rule_text: String,       // QOR rule with $HOLE_* placeholders
    pub holes: Vec<Hole>,
    pub trigger: TemplateTrigger,
}

#[derive(Debug, Clone)]
pub struct Hole {
    pub placeholder: String,     // e.g. "$HOLE_COLOR"
    pub hole_type: HoleType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HoleType { Color, Int, Predicate, Direction, Transform }

#[derive(Debug, Clone)]
pub enum TemplateTrigger {
    ObsConsistent(Vec<String>),  // requires specific obs-consistent facts
    ObsParameter(String),        // requires specific obs-parameter facts
    Always,                      // always try
}

#[derive(Debug, Clone)]
pub struct FilledTemplate {
    pub name: String,
    pub rule_text: String,
    pub fill_values: HashMap<String, String>,
    pub score: f64,
}
```

#### Public Functions

```rust
pub fn load_templates_from_qor(text: &str) -> Vec<Template>
```
Parses `.qor` file text with `$HOLE_*` placeholders. Extracts template name from comments (`";; -- T1: ..."`) and triggers from `";; Triggered by: ..."` comments. Auto-detects hole types from naming conventions (COLOR, DIR, TRANS, PRED, else Int). Internally calls `extract_holes_from_rule()` which uses word-boundary matching: strips the `$HOLE_` prefix, then checks for exact match, `_` prefixed, or `_` suffixed patterns (e.g., `COLOR`, `COLOR_*`, `*_COLOR`). This prevents misclassification such as `$HOLE_DIRECT` being typed as Direction.

```rust
pub fn builtin_templates() -> Vec<Template>
```
Returns 5 built-in templates: color-remap, shift, extract-by-color, fill-enclosed, identity.

```rust
pub fn extract_fill_candidates(observations: &[Statement]) -> HashMap<String, Vec<HashMap<String, String>>>
```
Extracts candidate fill values from observation facts (obs-parameter, obs-consistent). Maps obs-key to lists of placeholder-value mappings. Handles color-remap, shift-dr, shift-dc, extract-color, fill-color, scale-factor, tile-factor, border-color.

```rust
pub fn instantiate_template(template: &Template, candidates: &HashMap<String, Vec<HashMap<String, String>>>) -> Vec<FilledTemplate>
```
Tries to instantiate a template by filling all holes from candidates. Returns filled rule texts if all holes can be filled.

```rust
pub fn instantiate_all(observations: &[Statement]) -> Vec<FilledTemplate>
```
Tries all builtin templates against observations, returns all filled candidates.

```rust
pub fn instantiate_all_with_extra(observations: &[Statement], extra_templates: &[Template]) -> Vec<FilledTemplate>
```
Same as `instantiate_all` but also includes extra templates (typically from `load_templates_from_qor()`).

**Called by:** `solve.rs` (Phase 1)

---

### `text_hint.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\text_hint.rs`

Natural language to semantic hint facts for ARC puzzle descriptions.

#### Public Functions

```rust
pub fn parse_text_hints(text: &str) -> Vec<Statement>
```
Returns `(text-hint <hint_name> <confidence>)` facts for each detected keyword pattern. 20 keyword-to-hint mappings: reflect, rotate, gravity, flood-fill, crop, scale, recolor, symmetry, shift, copy, tile, border, sort, count, select-largest, select-smallest, horizontal, vertical, diagonal, overlap. Also detects directional modifiers (dir-horizontal, dir-vertical, dir-clockwise). Case-insensitive, no duplicates.

**Called by:** `web_fetch::extract_facts_from_pages`

---

### `web_rules.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\web_rules.rs`

Extracts candidate QOR rules and facts from plain text. Pure text processing (no web feature needed).

#### Public Types

```rust
#[derive(Debug, Clone)]
pub struct ExtractedRule {
    pub rule_text: String,
    pub source_url: String,
    pub source_sentence: String,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct ExtractedFact {
    pub statement: Statement,
    pub source_url: String,
    pub source_sentence: String,
}
```

#### Public Functions

```rust
pub fn extract_rules(text: &str, source_url: &str) -> Vec<ExtractedRule>
```
Extracts candidate rules from text using causal/conditional patterns: "if X then Y" (0.75), "when X, Y" (0.70), "X causes Y" (0.65), "X leads to Y" (0.65), "X results in Y" (0.65), "X implies Y" (0.70), "X therefore Y" (0.65). Skips lines < 10 or > 300 chars, sides < 3 or > 100 chars.

```rust
pub fn extract_facts(text: &str, source_url: &str) -> Vec<ExtractedFact>
```
Extracts candidate facts using definitional/taxonomic patterns: "X is defined as Y" / "X means Y", "X is a Y" (is-a), "X has Y" / "X contains Y", numeric patterns, property patterns.

```rust
pub fn build_search_urls(topic: &str, source_bases: &[String]) -> Vec<String>
```
Builds search URLs by appending topic to each source base URL.

**Called by:** `web_fetch::extract_facts_from_pages`, `web_search.rs`

---

### `web_fetch.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\web_fetch.rs`

Web crawling, extraction, caching, and domain policy.

#### Public Types

```rust
#[derive(Debug, Clone)]
pub struct PageContent { pub url: String, pub text: String, pub title: String }

#[derive(Debug, Clone)]
pub struct CrawlConfig { pub max_pages: usize, pub max_depth: usize, pub timeout_secs: u64 }
// Default: 10 pages, depth 2, 30s timeout

#[derive(Debug)]
pub struct CrawlResult { pub pages: Vec<PageContent>, pub elapsed_ms: u64 }

pub struct WebCache { cache_dir: PathBuf }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainTier { Tier1, Tier2, Tier3 }

#[derive(Debug, Clone)]
pub struct DomainPolicy { pub whitelist: Vec<String>, pub blacklist: Vec<String> }
// Default blacklist: pinterest, instagram, twitter, tiktok, facebook, youtube

pub struct WebConfig { /* parsed from QOR source facts */ }
```

**WebCache methods:** `new(cache_dir)`, `is_cached(url, max_age_hours)`, `load(url)`, `save(page)`

**DomainPolicy methods:** `can_crawl(url)` -- blacklist wins over whitelist; empty whitelist = allow all.

#### Public Functions

```rust
pub fn extract_facts_from_pages(pages: &[PageContent]) -> Vec<Statement>
```
**CRITICAL:** Everything from the web becomes a CANDIDATE, never a direct fact. Generates: `web-source` (metadata), `web-domain` (metadata), `web-hint-candidate` (from text_hint), `web-rule-candidate` (from web_rules::extract_rules), `web-fact-candidate` (from web_rules::extract_facts). All candidates have very low TV (0.30, 0.20).

```rust
pub fn domain_tier(url: &str) -> DomainTier
```
Classifies URL by tier: Tier1 = arxiv, openreview, arcprize, semanticscholar. Tier2 = wikipedia, mathworld, investopedia, paperswithcode. Tier3 = everything else.

```rust
pub fn parse_domain_policy(text: &str, section: &str) -> DomainPolicy
```
Parses `[section]` blocks with `whitelist = ...` and `blacklist = ...` lines.

```rust
pub fn sort_urls_by_tier(urls: &mut Vec<String>)
pub fn filter_urls(urls: &[String], policy: &DomainPolicy) -> Vec<String>
pub fn parse_web_config(facts: &[Statement]) -> WebConfig
```

#### Feature-Gated Module (`web`)

```rust
pub mod crawler {
    pub async fn safe_crawl(url, config, policy) -> Option<CrawlResult>
    pub async fn crawl_url(url, config) -> CrawlResult
    pub async fn crawl_urls(urls, config) -> Vec<CrawlResult>
}
```

**Called by:** `web_search.rs`

---

### `web_search.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\web_search.rs`

Feature-gated (`#[cfg(feature = "web")]`) web search pipeline.

#### Public Types

```rust
#[derive(Debug)]
pub struct WebSearchResult {
    pub candidate_rules: Vec<String>,
    pub pages_fetched: usize,
    pub rules_extracted: usize,
}
```

#### Public Functions

```rust
pub fn search_web(topics: &[String], sources: &[String], cache: &WebCache, policy: &DomainPolicy, cache_hours: u64) -> WebSearchResult
```
Pipeline: topics -> `build_search_urls()` -> filter by policy -> sort by tier -> cache check / HTTP fetch -> strip HTML -> `extract_rules()` -> candidate rule texts.

```rust
pub fn search_from_config(config: &WebConfig, cache_dir: &Path, policy: &DomainPolicy) -> WebSearchResult
```
Convenience wrapper that reads config and runs `search_web()`.

**Calls:** `web_rules::build_search_urls`, `web_rules::extract_rules`, `web_fetch::filter_urls`, `web_fetch::sort_urls_by_tier`

---

### `memory_graph.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\memory_graph.rs`

Puzzle reasoning memory as QOR graph facts. The agent reasons about its own experience.

#### Public Functions

```rust
pub fn run_to_statements(puzzle_id: &str, strategy: &str, correct: bool, accuracy: f64, features: &[String]) -> Vec<Statement>
```
Converts a single puzzle run into graph facts: `(puzzle id)`, `(puzzle-feature id feature)`, `(attempt id strategy win|fail accuracy)`.

```rust
pub fn aggregate_strategies(runs: &[(String, String, bool, f64, Vec<String>)]) -> Vec<Statement>
```
Aggregates all runs into `(strategy-for feature strategy) <win_rate, confidence>` facts. Groups by (feature, strategy), uses win-rate as strength, sample-count-based confidence.

```rust
pub fn load_memory(memory_dir: &Path) -> Result<Vec<Statement>, String>
```
Loads memory from `brain/memory/*.qor` files.

```rust
pub fn save_memory(puzzles: &[Statement], attempts: &[Statement], strategies: &[Statement], memory_dir: &Path) -> Result<(), String>
```
Saves memory facts to `puzzles.qor`, `attempts.qor`, `strategies.qor` in memory directory.

```rust
pub fn partition_memory(stmts: &[Statement]) -> (Vec<Statement>, Vec<Statement>, Vec<Statement>)
```
Partitions memory statements into (puzzles, attempts, strategies) by predicate.

```rust
pub fn extract_features_from_facts(facts: &[Statement]) -> Vec<String>
```
Extracts feature names from `obs-consistent-*` predicates (same-size, separator-v/h, reflect-h/v, rotate-90, color-count-same, obj-count-same).

```rust
pub fn memory_dir(brain_dir: &Path) -> PathBuf
```
Returns `brain_dir.join("memory")`.

**Called by:** solve pipeline, heartbeat engine

---

### `kb_build.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\kb_build.rs`

Binary triple format infrastructure shared across all knowledge sources.

**Binary format:** 12 bytes per triple: `u32 subject | u16 predicate | u32 object | u8 trust | u8 plausibility`

#### Public Types

```rust
pub struct IdEncoder {
    entity_map: HashMap<String, u32>,
    predicate_map: HashMap<String, u16>,
    // next_entity starts at 1 (0 = null)
    // next_predicate starts at 1 (0 = null)
}

pub struct SourceStats {
    pub name: String,
    pub domain: String,
    pub entities: u32,
    pub triples: u64,
    pub formulas: usize,
}
```

#### IdEncoder Methods

```rust
pub fn new() -> Self
pub fn entity_id(&mut self, name: &str) -> u32     // get or create (lowercased)
pub fn predicate_id(&mut self, name: &str) -> u16   // get or create (lowercased)
pub fn entity_count(&self) -> u32
pub fn predicate_count(&self) -> u16
pub fn load_mappings(dir: &Path) -> io::Result<Self> // from entities.tsv + predicates.tsv
pub fn save_mappings(&self, dir: &Path) -> io::Result<()> // to entities.tsv + predicates.tsv
```

#### Module Functions

```rust
pub fn write_triple(w: &mut impl Write, subject: u32, predicate: u16, object: u32, trust: f32, plausibility: f32) -> io::Result<()>
```
Writes one 12-byte LE triple.

```rust
pub fn write_manifest(dir: &Path, sources: &[SourceStats]) -> io::Result<()>
```
Writes `manifest.json` with source metadata.

**Called by:** `oeis.rs`, `dharmic.rs`

---

### `oeis.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\oeis.rs`

OEIS (Online Encyclopedia of Integer Sequences) parser.

#### Public Functions

```rust
pub fn run_oeis_pipeline(input_dir: &Path, output_dir: &Path, encoder: &mut IdEncoder) -> io::Result<SourceStats>
```
Parses OEIS data into binary triples + QOR formula rules. Supports two input formats: (1) oeisdata repo format (`seq/A000/A000045.seq`), (2) downloadable `stripped.gz` + `names.gz`. Outputs `oeis_triples.bin` (names, terms, keywords, cross-refs) and `oeis_formulas.qor` (recurrence, closed, generating function, summation, congruence, bounds as QOR rules).

**Calls:** `kb_build::write_triple`, `kb_build::IdEncoder`

---

### `dharmic.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\dharmic.rs`

Sacred text ingestion pipeline.

#### Public Functions

```rust
pub fn run_dharmic_pipeline(input_dir: &Path, output_dir: &Path, encoder: &mut IdEncoder) -> Result<SourceStats, String>
```
Scans `input_dir` for known subdirectories and writes `dharmic_triples.bin`. Supported sources: Rigveda (mandala -> sukta -> text), Atharvaveda (kaanda -> sukta -> text), Yajurveda (adhyaya -> text), Bhagavad Gita (chapter -> verse -> text + commentaries/translations), Mahabharata (book -> chapter -> shloka -> text), Valmiki Ramayana (kaanda -> sarg -> shloka -> text), Ramcharitmanas (kaand -> verse_type -> content).

**Calls:** `kb_build::write_triple`, `kb_build::IdEncoder`

---

### `perceive.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\perceive.rs`

Real-time perception engine. Fetches live data from configured HTTP sources.

#### Public Constants

```rust
pub const SHORT_TERM_DECAY: f64 = 0.15;
// At 0.15/heartbeat, facts drop below 0.10 confidence in ~6 heartbeats

pub const PERCEPTION_TV: TruthValue = TruthValue { strength: 0.90, confidence: 0.85 };
```

#### Public Types

```rust
#[derive(Debug, Clone)]
pub struct SourceConfig {
    pub id: String,
    pub url: String,
    pub format: String,       // "json", "csv", "text"
    pub interval: Duration,
}

#[derive(Debug, Default)]
pub struct PerceptionState {
    last_fetch: HashMap<String, Instant>,
}

#[derive(Debug)]
pub struct PerceptionResult {
    pub source_id: String,
    pub facts: Vec<Statement>,
    pub raw_size: usize,
    pub error: Option<String>,
}
```

#### Public Functions

```rust
pub fn parse_sources(statements: &[Statement]) -> Vec<SourceConfig>
```
Parses `(source <id> <url> <format> <interval-secs>)` facts into `SourceConfig` objects.

#### PerceptionState Methods

```rust
pub fn new() -> Self
pub fn sources_due<'a>(&self, sources: &'a [SourceConfig]) -> Vec<&'a SourceConfig>
```
Returns sources whose fetch interval has elapsed (or never fetched).

```rust
pub fn fetch_source(&mut self, source: &SourceConfig) -> PerceptionResult
```
HTTP GET via ureq (10s timeout), parses by format (json/csv/text), tags facts with `SHORT_TERM_DECAY` and `PERCEPTION_TV`, adds `(perceived source_id timestamp)` metadata.

```rust
pub fn perceive(&mut self, sources: &[SourceConfig]) -> Vec<PerceptionResult>
```
Runs a full perception cycle: fetches all due sources.

---

### `solve.rs`

**Path:** `D:\QOR-LANG\qor\qor-bridge\src\solve.rs`

Reasoning-first solve pipeline with 6 phases.

#### Public Types

```rust
#[derive(Debug)]
pub struct SolveResult {
    pub best_rules: Vec<String>,        // winning QOR rule texts
    pub score: f64,                     // best training accuracy (0.0..1.0)
    pub solved: bool,                   // score >= 0.999
    pub candidates_explored: usize,
    pub mutations_tried: usize,
    pub elapsed_ms: u64,
    pub solved_in_phase: Option<&'static str>,
    pub baseline_score: f64,            // DNA-only score
}
```

#### Public Functions

```rust
pub fn solve(
    puzzle_session: &Session,
    all_expected: &[Statement],
    target_pred: &str,
    observations: &[Statement],
    time_budget_ms: u64,
    library: Option<&mut RuleLibrary>,
) -> SolveResult
```
6-phase pipeline:
- **Phase 0 (REASON):** Score DNA rules baseline (MATCH + CHAIN)
- **Phase 1 (TEMPLATES):** Fill template holes from observations (UNIFY). Uses a `seen_rules: HashSet<String>` to deduplicate filled templates before scoring, avoiding redundant scoring of identical rules.
- **Phase 2 (REFINE):** Mutate near-misses (REDUCE + CONTRADICT)
- **Phase 3 (GENESIS):** Invent new rules from scratch (MATCH + CHAIN + REDUCE)
- **Phase 3.5:** Post-genesis refinement (budget capped at 3s to leave budget for Phase 4 Swarm and Phase 5 Combine)
- **Phase 4 (SWARM):** Parallel multi-strategy search (all 5 ops)
- **Phase 5 (COMBINE):** Combine partial rules (UNIFY)

Uses `precision_recall_score()` (geometric mean of precision and recall) for scoring. Shares findings between phases via `all_near_misses`.

**Calls:** `template::instantiate_all_with_extra`, `qor_runtime::mutate`, `qor_runtime::invent`, `qor_runtime::search`, `qor_runtime::library`

**Called by:** `qor-cli` solve command

---

## 3. Connection Diagram

### Data Flow: Ingestion Pipeline

```
User Input (string or file path)
    |
    v
lib::feed() / feed_file() / feed_as()
    |
    +---> detect::detect_format()          -- determines DataFormat
    |
    +---> lib::feed_raw()                  -- dispatches to format parser
    |       |
    |       +---> json::from_json()        -- uses sanitize::*, grid::Grid
    |       +---> csv_ingest::from_csv()   -- uses sanitize::*
    |       +---> kv::from_kv()            -- uses sanitize::*
    |       +---> text::from_text()        -- uses sanitize::*
    |       +---> parquet_ingest::from_parquet_file()  -- uses sanitize::*
    |
    +---> lib::enrich()                    -- enrichment pipeline
            |
            +---> context::auto_context()  -- domain rules
            |       +---> context::detect_domain()
            |       +---> context::extract_predicates()
            |
            +---> context::auto_analysis() -- pattern detection
            |       +---> context::extract_float_values()
            |
            +---> learn::learn()           -- statistical learning
            |       +---> learn::indicator_knowledge()
            |       +---> learn::threshold_rules()
            |       +---> context::extract_predicates()
            |       +---> context::extract_float_values()
            |
            +---> context::auto_queries()  -- query generation
            |
            +---> grid::deep_grid_perception()  -- grid analysis
                    +---> Grid::compare_pair()
                    +---> compute_observations()
```

### Data Flow: Chat System

```
User text
    |
    v
language::tokenize()          -- (input N word) facts
    |
    v
language::load_language_dir() -- vocabulary + grammar rules
    |
    v
Session::heartbeat()          -- forward chaining fires grammar rules
    |
    v
Session::response_facts()     -- extracts (response ...) facts
    |
    v
language::format_response()   -- natural language output
    |
    +---> language::extract_brain_context()  -- enriches with stats
    |
    +---> BrainContext::personality_text()   -- personality rotation
```

### Data Flow: Solve Pipeline

```
solve::solve()
    |
    +---> Phase 0: Score DNA baseline (Session predictions)
    |
    +---> Phase 1: template::instantiate_all_with_extra()
    |       +---> template::extract_fill_candidates()
    |       +---> template::instantiate_template()
    |       +---> template::load_templates_from_qor()
    |
    +---> Phase 2: qor_runtime::mutate::generate_mutations()
    |              qor_runtime::mutate::generate_context_mutations()
    |
    +---> Phase 3: qor_runtime::invent::*
    |
    +---> Phase 4: qor_runtime::search::* (time-budgeted)
    |
    +---> Phase 5: Combine partial rules
```

### Data Flow: Web Intelligence

```
web_search::search_web()
    |
    +---> web_rules::build_search_urls()
    +---> web_fetch::filter_urls()
    +---> web_fetch::sort_urls_by_tier()
    +---> HTTP fetch (ureq) / web_fetch::WebCache
    +---> web_fetch::extract_facts_from_pages()
            |
            +---> text_hint::parse_text_hints()  -- semantic hints
            +---> web_rules::extract_rules()      -- candidate rules
            +---> web_rules::extract_facts()      -- candidate facts
```

### Data Flow: DNA System

```
dna::load_dna()            -- reads JSON profile
dna::load_knowledge()      -- reads knowledge.txt
dna::load_rules()          -- reads rules.qor
dna::load_templates_text() -- reads templates.qor
dna::load_meta_rules()     -- reads meta/*.qor
    |
    v
dna::dna_to_statements()   -- profile -> QOR facts
dna::parse_knowledge()     -- knowledge -> (meaning) facts
    |
    v
Session (loaded into runtime for chat or solve)
    |
    +---> language::extract_brain_context()  -- personality integration
```

### Cross-File Dependencies (who calls whom)

| Caller | Calls |
|--------|-------|
| `lib.rs` | `detect`, `json`, `csv_ingest`, `kv`, `text`, `parquet_ingest`, `context`, `learn`, `grid` |
| `json.rs` | `sanitize`, `grid` |
| `csv_ingest.rs` | `sanitize` |
| `kv.rs` | `sanitize` |
| `text.rs` | `sanitize` |
| `parquet_ingest.rs` | `sanitize` |
| `context.rs` | `qor_core::neuron`, `qor_core::truth_value` |
| `learn.rs` | `context` |
| `language.rs` | `qor_core::parser`, `qor_core::neuron`, `qor_runtime::eval` |
| `llguidance.rs` | `qor_core::neuron`, `qor_core::truth_value` |
| `dna.rs` | `qor_core::parser`, `qor_core::neuron`, `qor_core::truth_value` |
| `grid.rs` | `qor_core::neuron`, `qor_core::truth_value` |
| `template.rs` | `qor_core::neuron` |
| `text_hint.rs` | `qor_core::neuron`, `qor_core::truth_value` |
| `web_fetch.rs` | `text_hint`, `web_rules` |
| `web_rules.rs` | `sanitize` |
| `web_search.rs` | `web_fetch`, `web_rules` |
| `solve.rs` | `template`, `qor_runtime::chain`, `qor_runtime::eval`, `qor_runtime::invent`, `qor_runtime::library`, `qor_runtime::mutate`, `qor_runtime::search` |
| `perceive.rs` | `qor_core::neuron`, `qor_core::truth_value` (ureq for HTTP) |
| `memory_graph.rs` | `qor_core::neuron`, `qor_core::parser`, `qor_core::truth_value` |
| `kb_build.rs` | (standalone I/O) |
| `oeis.rs` | `kb_build` |
| `dharmic.rs` | `kb_build` |

---

## 4. Public API Summary

### Crate-Level Entry Points

| Function | Signature | Purpose |
|----------|-----------|---------|
| `feed` | `(data: &str) -> Result<Vec<Statement>, String>` | Auto-detect format, ingest, enrich |
| `feed_file` | `(path: &Path) -> Result<Vec<Statement>, String>` | File-based ingestion (incl. parquet) |
| `feed_as` | `(data: &str, format: DataFormat) -> Result<Vec<Statement>, String>` | Explicit format ingestion |

### Format Parsers

| Function | Module | Signature |
|----------|--------|-----------|
| `from_json` | `json` | `(json: &str) -> Result<Vec<Statement>, String>` |
| `from_csv` | `csv_ingest` | `(csv_data: &str) -> Result<Vec<Statement>, String>` |
| `from_kv` | `kv` | `(data: &str) -> Result<Vec<Statement>, String>` |
| `from_text` | `text` | `(text: &str) -> Result<Vec<Statement>, String>` |
| `from_parquet_file` | `parquet_ingest` | `(path: &Path) -> Result<Vec<Statement>, String>` |

### Context and Learning

| Function | Module | Signature |
|----------|--------|-----------|
| `detect_format` | `detect` | `(data: &str) -> DataFormat` |
| `detect_domain` | `context` | `(facts: &[Statement]) -> DataDomain` |
| `auto_context` | `context` | `(facts: &[Statement]) -> Vec<Statement>` |
| `auto_analysis` | `context` | `(facts: &[Statement]) -> Vec<Statement>` |
| `auto_queries` | `context` | `(facts: &[Statement]) -> Vec<Statement>` |
| `learn` | `learn` | `(facts: &[Statement], analysis: &[Statement]) -> Vec<Statement>` |
| `threshold_rules` | `learn` | `(knowledge: &[Statement]) -> Vec<Statement>` |
| `indicator_knowledge` | `learn` | `(domain: &DataDomain) -> Vec<Statement>` |
| `merge_learned` | `learn` | `(existing: &[Statement], new: &[Statement]) -> Vec<Statement>` |

### Language Understanding

| Function | Module | Signature |
|----------|--------|-----------|
| `tokenize` | `language` | `(input: &str) -> Vec<Statement>` |
| `format_response` | `language` | `(responses: &[Vec<String>], ctx: Option<&BrainContext>) -> String` |
| `extract_brain_context` | `language` | `(facts: &[StoredNeuron]) -> BrainContext` |
| `load_language_dir` | `language` | `(dir: &Path) -> Vec<Statement>` |
| `parse_teach_pattern` | `language` | `(input: &str) -> Option<(String, String)>` |
| `save_to_dictionary` | `language` | `(dir: &Path, fact_line: &str)` |

### DNA System

| Function | Module | Signature |
|----------|--------|-----------|
| `load_dna` | `dna` | `(dna_dir: &Path, id: &str) -> Result<DnaProfile, String>` |
| `load_knowledge` | `dna` | `(dna_dir: &Path, id: &str) -> Result<String, String>` |
| `load_rules` | `dna` | `(dna_dir: &Path, id: &str) -> Vec<Statement>` |
| `load_meta_rules` | `dna` | `(workspace_root: &Path) -> Vec<Statement>` |
| `dna_to_statements` | `dna` | `(profile: &DnaProfile) -> Vec<Statement>` |
| `parse_knowledge` | `dna` | `(text: &str) -> Vec<Statement>` |
| `available_dna` | `dna` | `(dna_dir: &Path) -> Vec<(String, String)>` |
| `find_dna_by_keywords` | `dna` | `(dna_dir: &Path, query: &str) -> Vec<(String, String, usize)>` |
| `save_dna_qor` | `dna` | `(dna_dir: &Path, id: &str) -> Result<(PathBuf, usize), String>` |
| `convert_all` | `dna` | `(dna_dir: &Path) -> Vec<(String, PathBuf, usize)>` |

### Grid Perception

| Function/Method | Module | Signature |
|-----------------|--------|-----------|
| `Grid::from_vecs` | `grid` | `(cells: Vec<Vec<u8>>) -> Result<Grid, String>` |
| `Grid::objects` | `grid` | `(&self) -> Vec<GridObject>` |
| `Grid::to_statements` | `grid` | `(&self, grid_id: &str) -> Vec<Statement>` |
| `Grid::compare_pair` | `grid` | `(input: &Grid, output: &Grid, in_id: &str, out_id: &str) -> Vec<Statement>` |
| `deep_grid_perception` | `grid` | `(grid_facts: Vec<Statement>) -> Vec<Statement>` |

### Template Engine

| Function | Module | Signature |
|----------|--------|-----------|
| `load_templates_from_qor` | `template` | `(text: &str) -> Vec<Template>` |
| `builtin_templates` | `template` | `() -> Vec<Template>` |
| `extract_fill_candidates` | `template` | `(observations: &[Statement]) -> HashMap<String, Vec<HashMap<String, String>>>` |
| `instantiate_all` | `template` | `(observations: &[Statement]) -> Vec<FilledTemplate>` |
| `instantiate_all_with_extra` | `template` | `(observations: &[Statement], extra: &[Template]) -> Vec<FilledTemplate>` |

### Web Intelligence

| Function | Module | Signature |
|----------|--------|-----------|
| `parse_text_hints` | `text_hint` | `(text: &str) -> Vec<Statement>` |
| `extract_rules` | `web_rules` | `(text: &str, source_url: &str) -> Vec<ExtractedRule>` |
| `extract_facts` | `web_rules` | `(text: &str, source_url: &str) -> Vec<ExtractedFact>` |
| `extract_facts_from_pages` | `web_fetch` | `(pages: &[PageContent]) -> Vec<Statement>` |
| `domain_tier` | `web_fetch` | `(url: &str) -> DomainTier` |
| `parse_domain_policy` | `web_fetch` | `(text: &str, section: &str) -> DomainPolicy` |
| `search_web` | `web_search` | `(topics, sources, cache, policy, cache_hours) -> WebSearchResult` |

### Solve Pipeline

| Function | Module | Signature |
|----------|--------|-----------|
| `solve` | `solve` | `(session, expected, pred, observations, budget_ms, library) -> SolveResult` |

### Perception

| Function | Module | Signature |
|----------|--------|-----------|
| `parse_sources` | `perceive` | `(statements: &[Statement]) -> Vec<SourceConfig>` |
| `PerceptionState::perceive` | `perceive` | `(&mut self, sources: &[SourceConfig]) -> Vec<PerceptionResult>` |
| `PerceptionState::fetch_source` | `perceive` | `(&mut self, source: &SourceConfig) -> PerceptionResult` |

### Knowledge Base Building

| Function | Module | Signature |
|----------|--------|-----------|
| `write_triple` | `kb_build` | `(w, subject: u32, predicate: u16, object: u32, trust: f32, plausibility: f32) -> io::Result<()>` |
| `IdEncoder::entity_id` | `kb_build` | `(&mut self, name: &str) -> u32` |
| `IdEncoder::predicate_id` | `kb_build` | `(&mut self, name: &str) -> u16` |
| `run_oeis_pipeline` | `oeis` | `(input_dir, output_dir, encoder) -> io::Result<SourceStats>` |
| `run_dharmic_pipeline` | `dharmic` | `(input_dir, output_dir, encoder) -> Result<SourceStats, String>` |

### Memory Graph

| Function | Module | Signature |
|----------|--------|-----------|
| `run_to_statements` | `memory_graph` | `(puzzle_id, strategy, correct, accuracy, features) -> Vec<Statement>` |
| `aggregate_strategies` | `memory_graph` | `(runs) -> Vec<Statement>` |
| `load_memory` | `memory_graph` | `(memory_dir: &Path) -> Result<Vec<Statement>, String>` |
| `save_memory` | `memory_graph` | `(puzzles, attempts, strategies, memory_dir) -> Result<(), String>` |

### Utilities

| Function | Module | Signature |
|----------|--------|-----------|
| `sanitize_symbol` | `sanitize` | `(raw: &str) -> String` |
| `infer_value` | `sanitize` | `(raw: &str) -> Neuron` |
| `ingested_tv` | `sanitize` | `() -> TruthValue` |
| `make_fact` | `sanitize` | `(parts: Vec<Neuron>) -> Statement` |
| `summarize` | `llguidance` | `(statements: &[Statement]) -> String` |
