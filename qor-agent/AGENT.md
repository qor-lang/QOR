# ARC-AGI Rust Agent — Full Function Reference

> **Architecture**: Python is replaced by Rust. QOR is the brain. Rust is the agent.
> Every function below has one job. QOR owns all reasoning. Rust owns all action.

---

## How It All Connects

```
arcprize.org/play?task=XXXXXXXX
          │
          ▼
    ┌─────────────┐
    │  perceive   │  reads DOM / fetches GitHub JSON
    └──────┬──────┘
           │  Task { train: Vec<Pair>, test: Vec<TestPair> }
           ▼
    ┌─────────────┐
    │    reason   │  ← QOR BRAIN — asserts facts, fires rules, reads answers
    └──────┬──────┘
           │  Reasoning { detected, color_map, rule_count, fact_count }
           ▼
    ┌─────────────┐
    │    plan     │  pure math — converts QOR answer into output Grid
    └──────┬──────┘
           │  Grid (Vec<Vec<u8>>)
           ▼
    ┌─────────────┐
    │     act     │  controls browser — fills cells, submits, navigates
    └──────┬──────┘
           │  bool (correct / wrong)
           ▼
    ┌─────────────┐
    │    learn    │  if correct → append rule to learned.qor
    └─────────────┘
           │
           ▼
        NEXT TASK  (queue grows from Next button)
```

---

## Types — `src/types.rs`

All shared data structures used across every module.

| Type | Description |
|---|---|
| `Grid` | `Vec<Vec<u8>>` — a 2D grid of color values 0–9 |
| `Pair` | `{ input: Grid, output: Grid }` — one training example |
| `TestPair` | `{ input: Grid }` — test input (output unknown, agent must produce it) |
| `Task` | `{ id, train: Vec<Pair>, test: Vec<TestPair> }` — full ARC task |
| `Reasoning` | `{ detected, color_map, rule_count, fact_count }` — QOR output |
| `AgentStats` | `{ attempts, correct, wrong, learned }` — running score |

---

## Module: `perceive` — Eyes

**File**: `src/perceive.rs`  
**Job**: Get the task data from the outside world into a `Task` struct.

---

### `read_task(browser, url) -> Result<Task>`

Main entry point. Opens the URL, waits for page to render, tries two strategies.

```rust
pub fn read_task(browser: &Browser, url: &str) -> Result<Task>
```

**Flow:**
1. Opens new browser tab
2. Navigates to `url`
3. Waits 2 seconds for JavaScript to render
4. Calls `read_from_js()` first (fastest)
5. Falls back to `fetch_from_github()` if JS fails

---

### `read_from_js(tab, task_id) -> Result<Task>`

Extracts task data directly from the page's JavaScript state.

```rust
fn read_from_js(tab: &Tab, task_id: &str) -> Result<Task>
```

**What it does:**
- Evaluates `window.__task || window.taskData || null` in the browser
- Parses the JSON string into a `Task`
- Returns error if window state is null (page hasn't loaded task data yet)

---

### `fetch_from_github(task_id) -> Result<Task>`

Fallback: fetches raw task JSON from the ARC-AGI GitHub repository.

```rust
fn fetch_from_github(task_id: &str) -> Result<Task>
```

**Tries in order:**
1. `data/evaluation/{task_id}.json`
2. `data/training/{task_id}.json`

**Returns:** Parsed `Task` with all train pairs and test input.

---

### `parse_pairs(value) -> Result<Vec<Pair>>`
### `parse_test(value) -> Result<Vec<TestPair>>`
### `parse_grid(value) -> Result<Grid>`

Internal JSON → Rust type converters. Each row is parsed as `Vec<u8>` values 0–9.

---

## Module: `reason` — QOR Brain

**File**: `src/reason.rs`  
**Job**: Feed all task data to QOR as facts. Fire all rules. Read back what QOR inferred.

> This is the only module with intelligence. Everything else is mechanical.

---

### `qor_reason(task, rules_path, learned_path) -> Result<Reasoning>`

The brain call. All reasoning happens inside here via QOR.

```rust
pub fn qor_reason(
    task:         &Task,
    rules_path:   &str,
    learned_path: &str,
) -> Result<Reasoning>
```

**Step by step:**

**1. Load rules into fresh QOR session**
```
rules.qor    ← core rules (written by you)
learned.qor  ← rules QOR learned from past correct answers
```

**2. Assert training pair facts**
```prolog
(train-pair task 0 inp0 out0)
(grid-height inp0 5)
(grid-width  inp0 5)
(same-size   inp0 out0)       ← only if same dimensions
(cell inp0 0 0 3)             ← every cell value
(cell inp0 0 1 0)
...
(cell out0 0 0 7)
...
```

**3. Assert test input facts**
```prolog
(grid-height test 5)
(grid-width  test 5)
(cell test 0 0 3)
...
```

**4. QOR forward-chains automatically** — fires all matching rules

**5. Read back what QOR inferred:**

| Query | Meaning |
|---|---|
| `(is-copy task)` | Output = input unchanged |
| `(is-recolor task)` | Same grid, different colors |
| `(is-mirror-h task)` | Each row reversed |
| `(is-mirror-v task)` | Rows in reverse order |
| `(is-crop task)` | Output is cropped sub-region |
| `(is-rotate-90 task)` | Output is 90° rotation |
| `(is-fill task)` | Fill background cells |
| `(is-tile task)` | Tiling pattern |
| `(color-map task $src $dst)` | Color replacement mapping |

**Returns:** `Reasoning` with all detected transforms + color map + stats.

---

## Module: `plan` — Math

**File**: `src/plan.rs`  
**Job**: Take QOR's answer and produce the actual output grid. No intelligence here — just execution.

---

### `apply_transform(reasoning, task) -> Grid`

Dispatcher. Reads `reasoning.detected[0]` and calls the right transform function.

```rust
pub fn apply_transform(reasoning: &Reasoning, task: &Task) -> Grid
```

| `detected[0]` | Function called |
|---|---|
| `"copy"` | `test_input.clone()` |
| `"mirror-h"` | `mirror_h()` |
| `"mirror-v"` | `mirror_v()` |
| `"rotate-90"` | `rotate_90()` |
| `"crop"` | `crop_nonzero()` |
| `"recolor"` | `recolor()` + `infer_color_map()` |
| unknown | `guess_from_examples()` |

---

### `rotate_90(grid) -> Grid`

Rotates grid 90 degrees clockwise.

```rust
pub fn rotate_90(g: &Grid) -> Grid
```

Math: output cell `[c][h-1-r]` ← input cell `[r][c]`

---

### `crop_nonzero(grid) -> Grid`

Finds bounding box of all non-zero cells. Returns cropped sub-grid.

```rust
pub fn crop_nonzero(g: &Grid) -> Grid
```

**Steps:**
1. Find all rows containing at least one non-zero value
2. Find all cols containing at least one non-zero value in those rows
3. Return slice of grid at those row/col ranges

---

### `recolor(grid, map) -> Grid`

Applies a color-to-color mapping to every cell.

```rust
pub fn recolor(g: &Grid, map: &[(u8, u8)]) -> Grid
```

Each cell: if `cell_value == src` → replace with `dst`. If no match, keep original.

---

### `infer_color_map(train) -> Vec<(u8, u8)>`

Derives the color mapping from training pairs by majority vote.

```rust
pub fn infer_color_map(train: &[Pair]) -> Vec<(u8, u8)>
```

**How:**
- Counts every `(src, dst)` pair across all aligned cells
- For each source color → picks the most common destination
- Only works when input/output have same dimensions

---

### `guess_from_examples(train, test_input) -> Grid`

Fallback when QOR has no rule for this task yet.

```rust
pub fn guess_from_examples(train: &[Pair], test_input: &Grid) -> Grid
```

- If all pairs have same-size input/output → try recolor
- Otherwise → try crop

---

## Module: `act` — Hands

**File**: `src/act.rs`  
**Job**: Control the browser. Fill the output grid on the website. Submit. Read result. Navigate.

---

### `launch_browser() -> Result<Browser>`

Starts a Chromium browser via `headless_chrome`.

```rust
pub fn launch_browser() -> Result<Browser>
```

- `headless: false` → you can watch the agent work
- Change to `true` for silent background operation

---

### `fill_and_submit(browser, output) -> Result<bool>`

Main action function. Takes the computed output grid and performs all browser interactions.

```rust
pub fn fill_and_submit(browser: &Browser, output: &Grid) -> Result<bool>
```

**Steps:**

**1. Resize output grid**
```javascript
// Sets the resize input to "WxH" and clicks Resize button
input.value = '5x5';
resizeButton.click();
```

**2. For each color 0–9:**
- Finds cells in output grid that should be this color
- Clicks the color selector button
- Clicks each target cell

**3. Submit**
```javascript
[...buttons].find(b => b.textContent.includes('Submit')).click()
```

**4. Read result**
- Scans `document.body.innerText` for "correct" / "incorrect"
- Checks for CSS classes `.correct` / `.incorrect`
- Returns `true` (correct) or `false` (wrong)

---

### `get_next_task_id(browser) -> Result<Option<String>>`

Clicks the Next button and reads the new task ID from the URL.

```rust
pub fn get_next_task_id(browser: &Browser) -> Result<Option<String>>
```

- Finds button/link with text "Next"
- Clicks it, waits 1.5s for navigation
- Reads `tab.get_url()` and extracts `task=XXXXXXXX` with regex

---

## Module: `learn` — Memory

**File**: `src/learn.rs`  
**Job**: When an answer is correct, write a QOR rule to `learned.qor`. This is how QOR accumulates knowledge over time.

---

### `ensure_rules(rules_path, learned_path) -> Result<()>`

Creates rules files on first run if they don't exist.

```rust
pub fn ensure_rules(rules_path: &str, learned_path: &str) -> Result<()>
```

- Creates parent directories
- Writes bootstrap rules to `rules.qor` if missing
- Creates empty `learned.qor` if missing

---

### `save_rule(task_id, detected, task, learned_path) -> Result<()>`

Appends a learned rule to `learned.qor`.

```rust
pub fn save_rule(
    task_id:      &str,
    detected:     &[String],
    task:         &Task,
    learned_path: &str,
) -> Result<()>
```

**What gets written per transform:**

| Transform | Rule written to learned.qor |
|---|---|
| `recolor` | `(known-recolor {task_id} {src} {dst})` per color pair |
| `mirror-h` | `(known-mirror-h {task_id})` |
| `mirror-v` | `(known-mirror-v {task_id})` |
| `crop` | `(known-crop {task_id})` |
| `copy` | `(known-copy {task_id})` |

**Example output in `learned.qor`:**
```prolog
;; task:00576224 transform:recolor
(known-recolor 00576224 0 3)
(known-recolor 00576224 7 1)

;; task:007bbfb7 transform:mirror-h
(known-mirror-h 007bbfb7)
```

---

## Main Loop — `src/main.rs`

```
INIT
  ensure_rules()          create rules.qor + learned.qor if missing
  launch_browser()        open Chromium

LOOP (queue of task IDs)
  ├── perceive            read_task(browser, url)
  ├── reason              qor_reason(task, rules, learned)   ← QOR BRAIN
  ├── plan                apply_transform(reasoning, task)
  ├── act                 fill_and_submit(browser, output)
  ├── if correct
  │     learn             save_rule(task_id, ...)
  └── navigate            get_next_task_id(browser) → push to queue
```

---

## Build & Run

```bash
# Build
cd D:\Systum\qor-agent
cargo build --release

# Run
cargo run --release

# Silent (no browser window)
# Set headless: true in act.rs then rebuild
cargo run --release
```

---

## QOR Rules Reference

**Core rules file**: `D:\QOR-LANG\qor\dna\arc\rules.qor`  
**Learned rules file**: `D:\QOR-LANG\qor\dna\arc\learned.qor`

### QOR Syntax
```prolog
(fact arg1 arg2)                    ← assert a fact
(head $x $y) if (body $x) (body2 $y)  ← define a rule
$x                                  ← variable (must start with $)
;;                                  ← comment (double semicolon ONLY)
```

### Facts the agent asserts
```prolog
(train-pair task 0 inp0 out0)       ← pair index + grid IDs
(grid-height inp0 5)                ← grid dimensions
(grid-width  inp0 5)
(same-size   inp0 out0)             ← only if dimensions match
(cell inp0 0 0 3)                   ← every cell: grid row col value
(cell test 0 0 3)                   ← test input cells
```

### Queries the agent reads back
```prolog
(is-copy task)                      ← copy transform detected
(is-recolor task)                   ← recolor transform detected
(is-mirror-h task)                  ← horizontal mirror detected
(is-mirror-v task)                  ← vertical mirror detected
(is-crop task)                      ← crop transform detected
(is-rotate-90 task)                 ← 90° rotation detected
(color-map task $src $dst)          ← color replacement pairs
```

---

## File Layout

```
qor-agent/
├── Cargo.toml
├── AGENT.md               ← this file
└── src/
    ├── main.rs            ← agent loop
    ├── types.rs           ← Grid, Task, Pair, Reasoning, AgentStats
    ├── perceive.rs        ← read task from browser/GitHub
    ├── reason.rs          ← QOR brain — all reasoning here
    ├── plan.rs            ← apply QOR answer → output grid
    ├── act.rs             ← browser control — fill, submit, navigate
    └── learn.rs           ← save correct answers as QOR rules
```

---

## The Big Idea

```
Day 1:   QOR has 5 bootstrap rules.   Solves ~5% of puzzles.
Day 7:   QOR has 200 learned rules.   Solves ~20% of puzzles.
Day 30:  QOR has 1000 learned rules.  Solves ~60%+ of puzzles.
```

Every correct answer makes QOR permanently smarter.  
The rules are readable — you can open `learned.qor` and see exactly what QOR knows.  
This is the opposite of a neural network: **explicit, inspectable, growing intelligence.**
