# qor-inference

## 1. Crate Overview

**Crate:** `qor-inference`
**File:** `D:\QOR-LANG\qor\qor-inference\src\lib.rs` (302 lines)
**Cargo.toml:** `D:\QOR-LANG\qor\qor-inference\Cargo.toml`

The `qor-inference` crate implements **induction** and **abduction** -- the "creative" inference steps that discover new rules from existing ones. Unlike deduction (which applies existing rules), induction and abduction generate novel relationships between concepts by finding shared patterns in rule structure.

- **Induction**: If `B->A` and `B->C` (shared body term B), infer `A->C` and `C->A`
- **Abduction**: If `A->B` and `C->B` (shared head term B), infer relationships between bodies A and C

Truth values for inferred rules use PLN formulas: `s = s1 * s2`, `c = w2c(min(c2w(c1), c2w(c2)))`, ensuring inferred rules have appropriately conservative confidence.

**Dependencies:**
- `qor-core` (Neuron, Statement, Condition, TruthValue types)
- `qor-runtime` (not directly used in lib.rs, but declared as dependency)

---

## 2. File-by-File Breakdown: `lib.rs`

### Public Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `infer(rules)` | `pub fn infer(rules: &[Statement]) -> Vec<Statement>` | Main entry point. Takes existing rules, runs induction (shared body) and abduction (shared head) to discover new rules. Deduplicates against existing rules using signature tracking. Returns only newly inferred rules. |

### Internal Structs

| Struct | Fields | Description |
|--------|--------|-------------|
| `ParsedRule` | `head_pred: String`, `body_preds: Vec<String>`, `tv: TruthValue` | Internal representation of a rule decomposed into its head predicate name, body predicate names, and truth value. Only rules with at least one positive body condition are parsed. |

### Internal Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `ParsedRule::from_stmt(stmt)` | `fn from_stmt(stmt: &Statement) -> Option<Self>` | Converts a `Statement::Rule` into a `ParsedRule` by extracting first symbols from head and positive body conditions. Returns `None` for non-rules, rules with empty bodies, or rules where symbols cannot be extracted. |
| `first_symbol(neuron)` | `fn first_symbol(neuron: &Neuron) -> Option<String>` | Extracts the first symbol from a Neuron. For `Symbol(s)`, returns `s`. For `Expression([Symbol(s), ...])`, returns `s`. Returns `None` otherwise. |
| `shared_terms(a, b)` | `fn shared_terms(a: &[String], b: &[String]) -> Vec<String>` | Returns the intersection of two string slices (terms appearing in both). |
| `rule_signature(head, body)` | `fn rule_signature(head: &str, body: &[String]) -> String` | Creates a canonical string signature for deduplication: `"head:if:body1:body2"` with sorted body terms. |
| `make_rule(head_pred, body_preds, tv)` | `fn make_rule(head_pred: &str, body_preds: &[String], tv: TruthValue) -> Statement` | Constructs a `Statement::Rule` from predicate names and a truth value. Creates expressions with a variable `$x` for generality. |

### Test Functions (9 tests)

| Test | Description |
|------|-------------|
| `test_induction_shared_body` | Two rules sharing body "bird" should produce 2+ new rules connecting "flies" and "has-feathers". |
| `test_abduction_shared_head` | Two rules with shared head "dangerous" should produce rules connecting "poisonous" and "venomous". |
| `test_no_duplicates` | Verifies no duplicate rules appear in output. |
| `test_empty_rules` | Empty input produces empty output. |
| `test_single_rule_no_inference` | Single rule cannot produce any inferences. |
| `test_conservative_truth_values` | Inferred rules must have confidence <= min input confidence. |
| `test_does_not_duplicate_existing` | If a rule already exists in input, it is not re-inferred. |
| `test_induction_tv_formula` | Verifies `s = s1*s2` and `c = w2c(min(c2w(c1), c2w(c2)))` for induction. |
| `test_abduction_tv_formula` | Verifies the same PLN formula for abduction. |

---

## 3. Connection Diagram

```
qor-inference::infer()
  |
  +-- reads: qor_core::neuron::Statement (input rules)
  |          qor_core::neuron::Condition::Positive
  |          qor_core::neuron::Neuron::Symbol/Expression/Variable
  |
  +-- uses:  qor_core::truth_value::TruthValue::induction()
  |          qor_core::truth_value::TruthValue::abduction()
  |          qor_core::truth_value::TruthValue::default_fact()
  |          qor_core::truth_value::TruthValue::new()
  |
  +-- produces: Vec<Statement::Rule> (newly inferred rules)

Called by:
  +-- qor_cli::think()           -- discovers new rules after heartbeat stabilization
  +-- qor_cli::run_think_cycle() -- discovers new rules in watch mode
```

---

## 4. Public API Summary

The crate exposes exactly **one public function**:

```rust
pub fn infer(rules: &[Statement]) -> Vec<Statement>
```

**Input:** A slice of `Statement` values (typically `Statement::Rule` variants with heads, bodies, and truth values).

**Output:** A `Vec<Statement>` containing only newly discovered rules (not duplicates of existing ones). Each new rule has a truth value computed using PLN induction/abduction formulas.

**Algorithm:**
1. Parse all input rules into `ParsedRule` (extract predicate names)
2. Register existing rule signatures in a `HashSet` for dedup
3. **Induction pass**: For every pair of rules, check if they share body predicates. If so, create rules linking their heads (both directions).
4. **Abduction pass**: For every pair of rules with the same head predicate, create rules linking their body predicates.
5. Return all new rules not already present in the input.
