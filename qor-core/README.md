# qor-core

Core types, parser, truth values, and unification engine for QORlang (Quantified Ontological Reasoning Language).

This crate defines the foundational data structures and algorithms that every other QOR crate depends on. It has **zero external dependencies** -- pure Rust only.

---

## Cargo.toml

```toml
[package]
name = "qor-core"
version.workspace = true
edition.workspace = true
license.workspace = true
description = "Core types and parser for QORlang"

[dependencies]
# None -- zero external dependencies
```

---

## Module Structure

```
qor-core/src/
  lib.rs          -- Crate root; re-exports all modules
  neuron.rs       -- Core data types: Neuron, Condition, Statement, enums
  parser.rs       -- Recursive-descent parser for .qor source files
  truth_value.rs  -- PLN-inspired two-component truth values + formulas
  unify.rs        -- Pattern matching (unification) + variable substitution
  macros.rs       -- Ergonomic Rust macros for building QOR structures
```

---

## File-by-File Breakdown

---

### 1. `lib.rs`

**Path:** `D:\QOR-LANG\qor\qor-core\src\lib.rs`

Crate root. Declares and re-exports all public modules. The `#[macro_use]` attribute on `macros` makes all macros available crate-wide.

```rust
#[macro_use]
pub mod macros;
pub mod neuron;
pub mod truth_value;
pub mod parser;
pub mod unify;
```

No functions, no types -- purely structural.

---

### 2. `neuron.rs`

**Path:** `D:\QOR-LANG\qor\qor-core\src\neuron.rs`

The heart of QOR's type system. Defines the core data unit ("Neuron") and all related types for rules, conditions, statements, and operators.

#### Public Enums

| Enum | Variants | Description |
|------|----------|-------------|
| `Neuron` | `Symbol(String)`, `Variable(String)`, `Expression(Vec<Neuron>)`, `Value(QorValue)` | The fundamental data unit. Everything in QOR is a Neuron -- facts, rules, queries, values. Symbols are names, Variables are `$x`-style placeholders, Expressions are compound S-expressions, Values are literals. |
| `QorValue` | `Int(i64)`, `Float(f64)`, `Str(String)`, `Bool(bool)` | Literal values that Neurons can hold. |
| `ComparisonOp` | `Gt`, `Lt`, `Ge`, `Le`, `Eq`, `Ne` | Comparison operators for guard conditions in rule bodies (`>`, `<`, `>=`, `<=`, `==`, `!=`). |
| `ArithmeticOp` | `Add`, `Sub`, `Mul`, `Div`, `Mod`, `Min`, `Max`, `Sqrt`, `Abs`, `Power`, `DigitSum` | Arithmetic operators for computing values in rule bodies. Includes both binary (`+`, `-`, `*`, `/`, `%`, `min`, `max`, `power`) and unary (`sqrt`, `abs`, `digit-sum`) ops. |
| `AggregateOp` | `Count`, `Sum`, `Min`, `Max` | Aggregate operations for counting/summing over matched facts in rule bodies. |
| `Condition` | `Positive(Neuron)`, `Negated(Neuron)`, `NegatedPresent(Neuron)`, `Guard(...)`, `Aggregate{...}`, `Arithmetic{...}`, `Lookup{...}`, `EndsWith(Neuron, Neuron)` | A condition in a rule body. See details below. |
| `TestExpect` | `Present(Neuron)`, `Absent(Neuron)` | What a `@test` block expects: a fact present or absent after chaining. |
| `Statement` | `Fact{...}`, `Query{...}`, `Rule{...}`, `Test{...}` | A parsed QOR statement -- one line/block in a `.qor` file. |

#### Condition Variants (Detail)

| Variant | Fields | Description |
|---------|--------|-------------|
| `Positive(Neuron)` | -- | Must be true: `(bird $x)` |
| `Negated(Neuron)` | -- | Must NOT be true (negation as failure): `not (penguin $x)` |
| `NegatedPresent(Neuron)` | -- | Must NOT be present as a base (non-inferred) fact: `not-present (grid-cell ...)`. Unlike `Negated`, only checks asserted facts, ignoring derived ones. |
| `Guard(ComparisonOp, Neuron, Neuron)` | op, lhs, rhs | Numeric comparison guard: `(> $v 70)`, `(<= $x $y)` |
| `Aggregate` | `op: AggregateOp`, `pattern: Neuron`, `bind_var: String`, `result_var: String` | Aggregate over matched facts: `(count (bird $x) $x -> $n)` |
| `Arithmetic` | `op: ArithmeticOp`, `lhs: Neuron`, `rhs: Neuron`, `result_var: String` | Compute `lhs op rhs`, bind to `$result_var`: `(+ $a $b $result)` |
| `Lookup` | `predicate: String`, `subject: Neuron`, `result_var: String` | Binary KB lookup: `(lookup "predicate" $subject $result)` |
| `EndsWith(Neuron, Neuron)` | num, digit | Digit check: `(ends-with $n $d)` -- true if last digit of `$n` equals `$d` |

#### Statement Variants (Detail)

| Variant | Fields | Description |
|---------|--------|-------------|
| `Fact` | `neuron: Neuron`, `tv: Option<TruthValue>`, `decay: Option<f64>` | A fact: `(bird tweety) <0.99>` or `(news headline) <0.95> @decay 0.01` |
| `Query` | `pattern: Neuron` | A query: `? (bird tweety)` or `? (bird $x)` |
| `Rule` | `head: Neuron`, `body: Vec<Condition>`, `tv: Option<TruthValue>` | A rule: `(flies $x) if (bird $x) not (penguin $x) <0.95>` |
| `Test` | `name: String`, `given: Vec<Neuron>`, `expect: Vec<TestExpect>` | A test block: `@test name given (...) expect (...)` |

#### Public Struct

| Struct | Fields | Description |
|--------|--------|-------------|
| `StoredNeuron` | `neuron: Neuron`, `tv: TruthValue`, `timestamp: Option<i64>`, `decay_rate: Option<f64>`, `inferred: bool` | A neuron stored in the NeuronStore with its truth value, optional temporal metadata, and whether it was derived by forward chaining. |

#### Public Functions

| Signature | Description |
|-----------|-------------|
| `pub fn is_unary_arith(op: &ArithmeticOp) -> bool` | Returns true if the arithmetic op is unary (1 operand + result). Matches `Sqrt`, `Abs`, `DigitSum`. Called by the parser when deciding how many operands to expect, and by `Condition::Display` for formatting. |

#### Neuron Methods (impl Neuron)

| Signature | Description |
|-----------|-------------|
| `pub fn symbol(name: &str) -> Self` | Constructor: creates a `Neuron::Symbol` from a string slice. Used throughout tests and by other crates to build neurons programmatically. |
| `pub fn variable(name: &str) -> Self` | Constructor: creates a `Neuron::Variable`. |
| `pub fn expression(children: Vec<Neuron>) -> Self` | Constructor: creates a `Neuron::Expression` wrapping child neurons. |
| `pub fn str_val(s: &str) -> Self` | Constructor: creates a `Neuron::Value(QorValue::Str(...))`. |
| `pub fn int_val(n: i64) -> Self` | Constructor: creates a `Neuron::Value(QorValue::Int(...))`. |
| `pub fn float_val(f: f64) -> Self` | Constructor: creates a `Neuron::Value(QorValue::Float(...))`. |
| `pub fn as_f64(&self) -> Option<f64>` | Extracts a numeric value as f64. Works for both `Int` and `Float` variants. Returns `None` for non-numeric neurons. Used by the chain resolver for arithmetic/guard evaluation. |

#### Display Implementations

- `Neuron`: S-expression format -- `(bird tweety)`, `$x`, `"hello"`, `42`
- `QorValue`: Raw value -- `42`, `3.14`, `"text"`, `true`
- `StoredNeuron`: Neuron + TV + optional `(inferred)` tag
- `Condition`: QOR syntax -- `(> $v 70)`, `not (penguin $x)`, `(count (bird $x) $x -> $n)`, etc.
- `ArithmeticOp`: Operator symbol -- `+`, `-`, `*`, `/`, `%`, `min`, `max`, `sqrt`, `abs`, `power`, `digit-sum`
- `AggregateOp`: Keyword -- `count`, `sum`, `min`, `max`

#### Derive Traits

- `Neuron`: `Debug, Clone, PartialEq`
- `QorValue`: `Debug, Clone, PartialEq`
- `ComparisonOp`: `Debug, Clone, PartialEq`
- `ArithmeticOp`: `Debug, Clone, PartialEq`
- `AggregateOp`: `Debug, Clone, PartialEq`
- `Condition`: `Debug, Clone, PartialEq`
- `TestExpect`: `Debug, Clone, PartialEq`
- `Statement`: `Debug, Clone, PartialEq`
- `StoredNeuron`: `Debug, Clone, PartialEq`

---

### 3. `parser.rs`

**Path:** `D:\QOR-LANG\qor\qor-core\src\parser.rs`

A hand-written recursive-descent parser for the QOR language. Parses `.qor` source text into `Vec<Statement>`. Supports facts, rules (with `if` keyword), queries (`?`), test blocks (`@test`), truth values (`<s, c>`), decay annotations (`@decay`), guard conditions, aggregates, arithmetic, lookups, `ends-with`, negation (`not`, `not-present`), named rules (`(rule name body -> head)`), module system (`@module`, `@end`, `@import`), and stratum annotations (`@stratum`).

#### Public Structs

| Struct | Fields | Description |
|--------|--------|-------------|
| `ParseError` | `message: String`, `line: usize`, `col: usize` | Parse error with location information. Implements `Display` and `Error`. |
| `ParseWarning` | `message: String`, `line: usize`, `col: usize` | Non-fatal parse warning (e.g., unbound variables). Implements `Display`. |

#### Public Functions

| Signature | Description | Calls | Called By |
|-----------|-------------|-------|-----------|
| `pub fn parse(input: &str) -> Result<Vec<Statement>, ParseError>` | Parse a QOR source string into a list of statements. Main entry point for most callers. | `Parser::new()`, `Parser::parse_program()` | `qor!` macro, `qor-runtime` session/eval, `qor-bridge` context/template, `qor-cli` run/check, `qor-agent` DNA loading |
| `pub fn parse_with_warnings(input: &str) -> Result<(Vec<Statement>, Vec<ParseWarning>), ParseError>` | Parse and also return any warnings (unbound variables, single-use variables). | `Parser::new()`, `Parser::parse_program()` | `qor-cli` check command (displays warnings to user) |
| `pub fn parse_with_strata(input: &str) -> Result<(Vec<Statement>, Vec<ParseWarning>, Vec<Option<u32>>), ParseError>` | Parse and also return strata annotations (`@stratum(N)` before rules) plus warnings. Returns a parallel `Vec<Option<u32>>` aligned with statements. | `Parser::new()`, `Parser::parse_program()` | `qor-runtime` stratified chaining |
| `pub fn parse_neuron(input: &str) -> Result<Neuron, ParseError>` | Parse a single neuron expression from a string (not a full statement). | `Parser::new()`, `Parser::parse_neuron()` | `qor-runtime` chain (parsing hypothesis strings), `qor-bridge` template instantiation |

#### Internal Parser State (private struct)

```rust
struct Parser<'a> {
    input: &'a str,        // Source text
    pos: usize,            // Current byte position
    line: usize,           // Current line number (1-based)
    col: usize,            // Current column number (1-based)
    warnings: Vec<ParseWarning>,
    pending_stratum: Option<u32>,      // @stratum annotation for next rule
    strata: Vec<Option<u32>>,          // Parallel to output statements
    current_module: Option<String>,    // @module prefix
    imports: HashMap<String, String>,  // @import short -> full name
    pending: Vec<Statement>,           // Multi-head named rule overflow
}
```

#### Internal Parser Methods (private, key logic)

| Method | Description |
|--------|-------------|
| `parse_program()` | Top-level loop: skip whitespace/comments, dispatch to `parse_statement()`, `parse_test_block()`, handle `@stratum`, `@module`, `@end`, `@import` annotations. Drains `pending` buffer for multi-head named rules. |
| `parse_statement()` | Dispatches on first character: `?` -> Query, `(` -> fact-or-rule. For `(`: peeks for `(rule ...)` named rule syntax, otherwise parses expression then checks for `if` keyword to decide Rule vs Fact. |
| `parse_expression()` | Parses `(symbol ...)` S-expressions recursively. Rejects empty `()`. |
| `parse_neuron()` | Parses a single atom: `(...)` expression, `$var` variable, `"string"` literal, number, `true`/`false` boolean, or symbol name. |
| `parse_named_rule()` | Parses `(rule name body-conditions -> head <tv>)` syntax. Supports multiple heads (emits one `Statement::Rule` per head). |
| `parse_test_block()` | Parses `@test name given (...) expect (...) expect not (...)` blocks. |
| `parse_truth_value()` | Parses `<strength>` or `<strength, confidence>` or `<auto>` (auto maps to `<1.0, 1.0>`). |
| `try_parse_comparison_op()` | Tries to match `>`, `<`, `>=`, `<=`, `==`, `!=` without consuming input on failure. |
| `try_parse_aggregate_op()` | Tries to match `count`, `sum`, `min`, `max`. For `min`/`max`, disambiguates from arithmetic by peeking: aggregate expects `(` next (for pattern), arithmetic expects `$` or number. |
| `try_parse_arithmetic_op()` | Tries to match `+`, `-`, `*`, `/`, `%`, `min`, `max`, `sqrt`, `abs`, `power`, `digit-sum`. Only matches if followed by whitespace (so `-3` is not confused with subtraction). |
| `validate_rule()` | Post-parse validation: warns if head variables are not bound by positive conditions, if negation variables are unbound, if variables are used only once (suggests `$_name` convention). |
| `prefix_neuron()` / `prefix_condition()` / `prefix_statement()` | Module system: resolves `@import` names and prefixes predicates with `@module` name for qualified naming (`module.predicate`). |
| `skip_whitespace_and_comments()` | Skips whitespace and `;;` line comments (QOR uses double-semicolon comments). |

#### Helper Functions (private, module-level)

| Function | Description |
|----------|-------------|
| `fn collect_variables(neuron: &Neuron) -> HashSet<String>` | Recursively collects all variable names from a Neuron tree. Used by `validate_rule()`. |
| `fn collect_variables_inner(neuron: &Neuron, vars: &mut HashSet<String>)` | Recursive implementation for `collect_variables`. |

---

### 4. `truth_value.rs`

**Path:** `D:\QOR-LANG\qor\qor-core\src\truth_value.rs`

PLN-inspired two-component truth values. Every fact and rule in QOR carries a `TruthValue` with **strength** (how likely true) and **confidence** (how much evidence). This module implements all the PLN formulas for combining truth values during reasoning.

#### Public Constants

| Constant | Type | Value | Description |
|----------|------|-------|-------------|
| `DEFAULT_CONFIDENCE` | `f64` | `0.9` | Default confidence when only strength is provided in a truth value literal. |

#### Public Struct

| Struct | Fields | Derives | Description |
|--------|--------|---------|-------------|
| `TruthValue` | `pub strength: f64`, `pub confidence: f64` | `Debug, Clone, Copy, PartialEq` | A PLN-inspired truth value. Strength is clamped to [0.0, 1.0], confidence is clamped to [0.0, 1.0]. Implements `Default` (returns `default_fact()`). |

#### TruthValue Methods

| Signature | Description | Called By |
|-----------|-------------|-----------|
| `pub fn new(strength: f64, confidence: f64) -> Self` | Constructor. Clamps both components to [0.0, 1.0]. | Everywhere -- parser, chain, store, bridge, inference |
| `pub fn from_strength(strength: f64) -> Self` | Creates a TV with default confidence (0.9). | `tv!` macro, bridge ingestion |
| `pub fn default_fact() -> Self` | Returns `<1.0, 0.9>` -- the default for facts with no explicit truth value. | `Default` impl, store insertion |
| `pub fn and(&self, other: &TruthValue) -> TruthValue` | AND formula: `s = s1 * s2`, `c = min(c1, c2)`. Both must be true. | `qor-runtime` chain (combining multiple body conditions) |
| `pub fn or(&self, other: &TruthValue) -> TruthValue` | OR formula: `s = s1 + s2 - s1*s2`, `c = min(c1, c2)`. At least one true. | Available for disjunctive reasoning |
| `pub fn deduction(&self, other: &TruthValue) -> TruthValue` | Deduction (modus ponens through a rule): `s = s1 * s2`, `c = c1 * c2 * 0.9`. PLN Book 1.4. | `qor-runtime` forward chain (applying a rule to derive new facts) |
| `pub fn negation(&self) -> TruthValue` | Negation: `s = 1 - s`, `c = c` (unchanged). | `qor-runtime` chain (negated conditions) |
| `pub fn revision(&self, other: &TruthValue) -> TruthValue` | Revision (merge evidence): `s_new = (s1*c1 + s2*c2)/(c1+c2)`, `c_new = c1 + c2 - c1*c2`. Confidence always increases. PLN Book 5.10.2. This is how QOR learns -- re-observing strengthens beliefs. | `qor-runtime` store (inserting duplicate facts), consolidation heartbeat |
| `pub fn inversion(&self, target_tv: &TruthValue) -> TruthValue` | Inversion (A->B implies B->A): `s = s_AB`, `c = c_B * c_AB * 0.6`. Derived from Bayes' rule. PLN Book p.11. | `qor-inference` inversion engine |
| `pub fn induction(&self, other: &TruthValue) -> TruthValue` | Induction (B->A + B->C => A<->C): `s = s1*s2`, `c = w2c(min(c2w(c1), c2w(c2)))`. PLN Book 5.1. | `qor-inference` induction engine |
| `pub fn abduction(&self, other: &TruthValue) -> TruthValue` | Abduction (A->B + C->B => A<->C): Same formula as induction (difference is semantic). PLN Book 5.1. | `qor-inference` abduction engine |
| `pub fn c2w(c: f64) -> f64` | Confidence to weight (evidence count): `w = c / (1 - c)`. Returns `f64::MAX` for `c >= 1.0`. Used internally by induction/abduction formulas. | `induction()`, `abduction()`, `qor-bridge` learn |
| `pub fn w2c(w: f64) -> f64` | Weight to confidence: `c = w / (w + 1)`. Inverse of `c2w`. | `induction()`, `abduction()`, `qor-bridge` learn (capped at 0.95) |

#### Display Implementation

Format: `<0.99, 0.90>` (two decimal places for both components).

#### Default Implementation

`TruthValue::default()` returns `TruthValue::default_fact()` which is `<1.00, 0.90>`.

---

### 5. `unify.rs`

**Path:** `D:\QOR-LANG\qor\qor-core\src\unify.rs`

Pattern matching (unification) engine and variable substitution. This is what makes QOR rules work -- it matches patterns with variables against concrete facts.

#### Public Types

| Type | Definition | Description |
|------|------------|-------------|
| `Bindings` | `HashMap<String, Neuron>` | Variable bindings: maps variable names to the concrete neurons they matched. |

#### Public Functions

| Signature | Description | Called By |
|-----------|-------------|-----------|
| `pub fn unify(pattern: &Neuron, target: &Neuron) -> Option<Bindings>` | Unify a pattern (with variables) against a concrete neuron. Returns `Some(bindings)` if the pattern matches, `None` otherwise. Variables in `pattern` bind to whatever they match in `target`. Repeated variables must match the same value (consistency check). Target variables act as wildcards (match anything). | `qor-runtime` chain (`resolve_body` for matching rule conditions against store), `qor-runtime` store (`query`), `qor-inference` engine |
| `pub fn apply_bindings(neuron: &Neuron, bindings: &Bindings) -> Neuron` | Substitute all bound variables in a neuron tree. Unbound variables remain as-is. | `qor-runtime` chain (building derived facts from rule heads after body matches), `qor-bridge` template instantiation |
| `pub fn extract_variables(neuron: &Neuron) -> Vec<String>` | Extract variable names from a neuron, in first-appearance order (no duplicates). | `qor-runtime` chain (determining which variables a rule introduces), `qor-runtime` search/mutate |

#### Unification Rules (internal logic)

| Pattern | Target | Result |
|---------|--------|--------|
| `Variable(name)` | anything | Bind `name` -> target (or check consistency if already bound) |
| anything | `Variable(_)` | Match (wildcard) |
| `Symbol(a)` | `Symbol(b)` | Match iff `a == b` |
| `Value(a)` | `Value(b)` | Match iff `a == b` |
| `Expression(a)` | `Expression(b)` | Match iff same length and all children unify recursively |
| mismatched types | -- | No match |

---

### 6. `macros.rs`

**Path:** `D:\QOR-LANG\qor\qor-core\src\macros.rs`

Ergonomic `macro_rules!` macros for embedding QOR in Rust code. All stable -- no nightly features needed. Convention: Rust underscores become QOR hyphens (`stiff_neck` -> `stiff-neck`).

#### Public Macros

| Macro | Syntax | Expands To | Description |
|-------|--------|------------|-------------|
| `qor!($source)` | `qor!("(bird tweety) <0.99>")` | `parser::parse($source)` | Parse a QOR source string into `Result<Vec<Statement>, ParseError>`. |
| `neuron!($($sym)+)` | `neuron!(bird tweety)` | `Neuron::Expression(vec![Neuron::Symbol("bird"), ...])` | Build a `Neuron::Expression` from identifier tokens. Underscores converted to hyphens. |
| `neuron!(@sym $sym)` | `neuron!(@sym bird)` | `Neuron::Symbol("bird")` | Build a single `Neuron::Symbol`. |
| `neuron!(@var $name)` | `neuron!(@var x)` | `Neuron::Variable("x")` | Build a single `Neuron::Variable`. |
| `tv!($s)` | `tv!(0.99)` | `TruthValue::from_strength(0.99)` | Build a `TruthValue` with default confidence. |
| `tv!($s, $c)` | `tv!(0.99, 0.90)` | `TruthValue::new(0.99, 0.90)` | Build a `TruthValue` with explicit strength and confidence. |
| `assert_fact!($store, $($sym)+ => $s)` | `assert_fact!(store, bird tweety => 0.99)` | `$store.insert(neuron!(...), tv!(...))` | Insert a fact into a `NeuronStore` with strength only. Requires `qor-runtime`. |
| `assert_fact!($store, $($sym)+ => $s, $c)` | `assert_fact!(store, bird eagle => 0.95, 0.90)` | `$store.insert(neuron!(...), tv!(...))` | Insert a fact into a `NeuronStore` with strength + confidence. |
| `qor_run!($source)` | `qor_run!("(bird tweety) <0.99> ? (bird $x)")` | `parser::parse($source)` | Alias for `qor!` -- parses source into statements. (Does not actually execute; runtime needed for that.) |

---

## Connection Diagram

### Cross-File Dependencies Within qor-core

```
macros.rs
  |-- calls --> parser::parse()       (qor! and qor_run! macros)
  |-- uses  --> neuron::Neuron         (neuron! macro)
  |-- uses  --> truth_value::TruthValue (tv! macro)

parser.rs
  |-- uses  --> neuron::*             (Neuron, QorValue, Statement, Condition, ComparisonOp,
  |                                    ArithmeticOp, AggregateOp, TestExpect, is_unary_arith)
  |-- uses  --> truth_value::TruthValue, DEFAULT_CONFIDENCE

neuron.rs
  |-- uses  --> truth_value::TruthValue  (StoredNeuron.tv field)

unify.rs
  |-- uses  --> neuron::Neuron         (pattern matching on Neuron variants)

truth_value.rs
  |-- standalone, no internal dependencies
```

### How Other Crates Use qor-core

```
qor-runtime (NeuronStore, Chain, Session, Eval)
  |-- neuron.rs:   Neuron, Statement, Condition, StoredNeuron, ComparisonOp,
  |                ArithmeticOp, AggregateOp, QorValue, is_unary_arith
  |-- parser.rs:   parse(), parse_neuron(), parse_with_strata()
  |-- truth_value: TruthValue (deduction, revision, and, negation in chain)
  |-- unify.rs:    unify(), apply_bindings(), extract_variables()
  |-- macros.rs:   neuron!, tv!, assert_fact!

qor-bridge (data ingestion, DNA, learning, grid, template)
  |-- neuron.rs:   Neuron, Statement, Condition, QorValue
  |-- parser.rs:   parse(), parse_neuron()
  |-- truth_value: TruthValue (from_strength, w2c for learned confidence)
  |-- unify.rs:    apply_bindings() (template instantiation)

qor-inference (induction, abduction engine)
  |-- neuron.rs:   Neuron, Statement, Condition
  |-- truth_value: TruthValue (induction, abduction, inversion formulas)
  |-- unify.rs:    unify() (finding shared body/head terms between rules)

qor-cli (command-line interface)
  |-- parser.rs:   parse(), parse_with_warnings() (run, check commands)
  |-- neuron.rs:   Statement (iterating parsed results)
  |-- truth_value: TruthValue (display)

qor-agent (web agent)
  |-- parser.rs:   parse() (loading DNA .qor files)
  |-- neuron.rs:   Neuron, Statement (building page facts, reading actions)
```

### Key Call Chains

```
[.qor file text]
    |
    v
parser::parse()
    |-- Parser::parse_program()
    |     |-- Parser::parse_statement()
    |     |     |-- Parser::parse_expression() --> Neuron::Expression
    |     |     |-- Parser::parse_neuron()     --> Neuron (any variant)
    |     |     |-- Parser::try_parse_comparison_op() --> ComparisonOp
    |     |     |-- Parser::try_parse_aggregate_op()  --> AggregateOp
    |     |     |-- Parser::try_parse_arithmetic_op() --> ArithmeticOp
    |     |     |-- Parser::parse_truth_value()       --> TruthValue
    |     |     `-- Parser::validate_rule()           --> warnings
    |     |-- Parser::parse_test_block()  --> Statement::Test
    |     |-- Parser::parse_named_rule()  --> Vec<Statement::Rule>
    |     `-- Parser::prefix_statement()  --> module-qualified Statement
    |
    v
Vec<Statement>  -->  qor-runtime::Session::exec_statements()
                          |
                          v
                     forward_chain() uses:
                       unify::unify(pattern, fact)  --> Option<Bindings>
                       unify::apply_bindings(head, bindings) --> derived Neuron
                       TruthValue::deduction(fact_tv, rule_tv)
                       TruthValue::and(tv1, tv2) [multiple body conditions]
                       TruthValue::revision(old, new) [duplicate facts]
```

---

## Public API Summary

### Types (re-exported via modules)

| Type | Module | Kind | Primary Use |
|------|--------|------|-------------|
| `Neuron` | `neuron` | enum | Core data unit for all QOR values |
| `QorValue` | `neuron` | enum | Literal values (int, float, string, bool) |
| `ComparisonOp` | `neuron` | enum | Guard condition operators |
| `ArithmeticOp` | `neuron` | enum | Rule body arithmetic |
| `AggregateOp` | `neuron` | enum | Rule body aggregation |
| `Condition` | `neuron` | enum | Rule body conditions (8 variants) |
| `TestExpect` | `neuron` | enum | Test expectations |
| `Statement` | `neuron` | enum | Parsed QOR statements (4 variants) |
| `StoredNeuron` | `neuron` | struct | Neuron + TruthValue + metadata in store |
| `TruthValue` | `truth_value` | struct | Two-component truth value (strength, confidence) |
| `ParseError` | `parser` | struct | Error with line/col location |
| `ParseWarning` | `parser` | struct | Non-fatal warning with location |
| `Bindings` | `unify` | type alias | `HashMap<String, Neuron>` |

### Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `parse(input)` | `parser` | Parse QOR source -> statements |
| `parse_with_warnings(input)` | `parser` | Parse -> statements + warnings |
| `parse_with_strata(input)` | `parser` | Parse -> statements + warnings + strata |
| `parse_neuron(input)` | `parser` | Parse single neuron expression |
| `unify(pattern, target)` | `unify` | Pattern match with variable binding |
| `apply_bindings(neuron, bindings)` | `unify` | Substitute variables in neuron |
| `extract_variables(neuron)` | `unify` | Get ordered list of variable names |
| `is_unary_arith(op)` | `neuron` | Check if arithmetic op is unary |

### Constants

| Constant | Module | Value | Purpose |
|----------|--------|-------|---------|
| `DEFAULT_CONFIDENCE` | `truth_value` | `0.9` | Default confidence for `<strength>` syntax |

### Macros

| Macro | Purpose |
|-------|---------|
| `qor!(source)` | Parse QOR source string |
| `neuron!(sym1 sym2 ...)` | Build Neuron::Expression |
| `neuron!(@sym name)` | Build Neuron::Symbol |
| `neuron!(@var name)` | Build Neuron::Variable |
| `tv!(s)` / `tv!(s, c)` | Build TruthValue |
| `assert_fact!(store, syms => s)` | Insert fact into NeuronStore |
| `qor_run!(source)` | Parse QOR source (alias for qor!) |

---

## Test Coverage

The crate contains comprehensive unit tests in each module:

- **neuron.rs**: 4 tests -- display formatting for expressions, variables, string values, numbers, and stored neurons
- **parser.rs**: 28 tests -- facts (with/without TV, with decay), queries, rules (single/multi condition, negation, guards for all operators, aggregates, arithmetic, mixed conditions), comments, error cases (unterminated, empty expression), keyword-in-name disambiguation, program-level parsing
- **truth_value.rs**: 14 tests -- constructors, AND, OR, deduction, negation, revision (merge, confidence growth, convergence), display, clamping, inversion, w2c/c2w roundtrip, induction, abduction, conservative induction
- **unify.rs**: 7 tests -- concrete match, variable binding, no-match, repeated variables (consistency), apply_bindings (bound and unbound), extract_variables (with dedup)
- **macros.rs**: 9 tests -- neuron macro (simple, 3-part, underscore-to-hyphen, multi-hyphen), tv macro (strength-only, two-component), qor macro (single, multi-statement), sym/var helpers
