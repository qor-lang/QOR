# qor-py

## 1. Crate Overview

**Crate:** `qor-py`
**Library name:** `qor_py`
**Crate type:** `cdylib` (compiled as a Python extension module via PyO3)
**File:** `D:\QOR-LANG\qor\qor-py\src\lib.rs` (149 lines)
**Cargo.toml:** `D:\QOR-LANG\qor\qor-py\Cargo.toml`

The `qor-py` crate provides Python bindings for the QOR reasoning engine using PyO3. It exposes a `QorSession` class that wraps the full `qor_runtime::eval::Session`, allowing Python code to load `.qor` rule files, assert facts, run forward chaining, and query the resulting knowledge store. This enables Python programs (e.g., ARC-AGI solvers, data science scripts) to use QOR as a reasoning backend without spawning a subprocess.

**Dependencies:**
- `pyo3` 0.24 with `extension-module` feature (Python C-API bindings)
- `qor-core` (Neuron types, TruthValue)
- `qor-runtime` (Session, eval engine)

---

## 2. File-by-File Breakdown: `lib.rs`

### Python Module

| Item | Signature | Description |
|------|-----------|-------------|
| `qor_py` (module) | `#[pymodule] fn qor_py(m: &Bound<'_, PyModule>) -> PyResult<()>` | PyO3 module entry point. Registers the `QorSession` class as the sole export of the `qor_py` Python module. |

### `QorSession` Struct (pyclass)

| Field | Type | Description |
|-------|------|-------------|
| `session` | `qor_runtime::eval::Session` | The wrapped QOR reasoning session that stores facts, rules, and performs forward chaining. |

### `QorSession` Methods (pymethods)

| Method | Python Signature | Return Type | Description |
|--------|-----------------|-------------|-------------|
| `new()` | `QorSession()` | `QorSession` | Constructor. Creates a new empty QOR session with no facts or rules loaded. |
| `load_rules(path)` | `session.load_rules("rules.qor")` | `int` | Reads a `.qor` file from disk and executes all statements (facts, rules, queries) into the session. Returns the number of results (statements processed). Raises `RuntimeError` on file read or parse errors. |
| `exec(source)` | `session.exec('(tile 0 0 1)')` | `list[str]` | Executes QOR source text directly (can contain facts, rules, queries). Returns a list of debug-formatted result strings. Raises `RuntimeError` on parse errors. |
| `assert_fact(predicate, args)` | `session.assert_fact("tile", [0, 0, 1])` | `None` | Asserts a fact with a string predicate name and integer arguments. Creates `(predicate arg1 arg2 ...)` with truth value `<0.99>`. Directly inserts into the store (no parsing). |
| `assert_fact_mixed(predicate, parts)` | `session.assert_fact_mixed("cnbr", ["0", "0", "same", "2"])` | `None` | Asserts a fact with mixed-type arguments (strings that are auto-parsed as int, float, or symbol). Creates `(predicate part1 part2 ...)` with truth value `<0.99>`. |
| `run()` | `session.run()` | `bool` | Runs one forward chaining heartbeat cycle. Returns `True` if any new facts were derived, `False` if beliefs are stable. |
| `query(predicate, arity)` | `session.query("tile", 3)` | `list[list[int]]` | Queries the store for facts matching `(predicate $V0 $V1 ... $Vn)`. Returns a list of integer argument tuples. Only returns results where all arguments are numeric and the arity matches exactly. |
| `query_str(predicate, arity)` | `session.query_str("color", 2)` | `list[list[str]]` | Same as `query()` but returns all arguments as strings (regardless of type). Useful for mixed int/symbol results. |
| `fact_count()` | `session.fact_count()` | `int` | Returns the total number of facts currently in the store. |
| `rule_count()` | `session.rule_count()` | `int` | Returns the total number of rules loaded in the session. |
| `all_facts()` | `session.all_facts()` | `list[str]` | Returns all facts as formatted strings: `"(predicate args) <strength, confidence>"`. |

---

## 3. Connection Diagram

```
Python code
  |
  +-- import qor_py
  |     +-- QorSession()          --> qor_runtime::eval::Session::new()
  |     +-- .load_rules(path)     --> std::fs::read_to_string() + session.exec()
  |     +-- .exec(source)         --> session.exec()
  |     +-- .assert_fact(p, args) --> Neuron::symbol/int_val + session.store_mut().insert()
  |     +-- .assert_fact_mixed()  --> Neuron::symbol/int_val/float_val + store_mut().insert()
  |     +-- .run()                --> session.heartbeat()
  |     +-- .query(pred, arity)   --> Neuron::variable + session.store().query()
  |     +-- .query_str(pred, ar.) --> Neuron::variable + session.store().query()
  |     +-- .fact_count()         --> session.fact_count()
  |     +-- .rule_count()         --> session.rule_count()
  |     +-- .all_facts()          --> session.all_facts()

qor-py depends on:
  +-- qor_core::neuron::Neuron       (Symbol, Expression, Variable, Value/QorValue)
  +-- qor_core::truth_value::TruthValue (from_strength, new)
  +-- qor_runtime::eval::Session     (new, exec, heartbeat, store, store_mut,
  |                                   fact_count, rule_count, all_facts)
```

---

## 4. Public API Summary

The crate's public API is the **Python module** `qor_py` which exports a single class:

### `class QorSession`

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `__init__()` | -- | `QorSession` | Create empty session |
| `load_rules(path: str)` | file path | `int` | Load .qor file, return statement count |
| `exec(source: str)` | QOR source text | `list[str]` | Execute QOR text directly |
| `assert_fact(predicate: str, args: list[int])` | predicate + int args | `None` | Assert integer-arg fact |
| `assert_fact_mixed(predicate: str, parts: list[str])` | predicate + mixed args | `None` | Assert mixed-type fact |
| `run()` | -- | `bool` | One heartbeat cycle |
| `query(predicate: str, arity: int)` | predicate + arity | `list[list[int]]` | Query for int results |
| `query_str(predicate: str, arity: int)` | predicate + arity | `list[list[str]]` | Query for string results |
| `fact_count()` | -- | `int` | Total fact count |
| `rule_count()` | -- | `int` | Total rule count |
| `all_facts()` | -- | `list[str]` | All facts as formatted strings |

**Example Python usage:**
```python
import qor_py

session = qor_py.QorSession()
session.load_rules("rules.qor")
session.assert_fact("tile", [0, 0, 1])
session.run()
results = session.query("tile", 3)  # [[0, 0, 1]]
```
