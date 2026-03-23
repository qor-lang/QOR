// ── Rule Mutation Operators ──────────────────────────────────────────
//
// Operate on QOR Rule AST to produce candidate variations.
// Used by search.rs for evolutionary refinement.
//
// DOMAIN AGNOSTIC — works with any QOR rules regardless of domain.
// Mutations are small, local changes that might fix a near-miss rule.

use rand::Rng;

use qor_core::neuron::{Condition, ComparisonOp, ArithmeticOp, AggregateOp, Neuron, QorValue};
use crate::chain::Rule;

/// A single mutation operation on a rule.
#[derive(Debug, Clone)]
pub enum Mutation {
    /// Remove a body condition by index.
    RemoveCondition(usize),
    /// Change a guard comparison value by delta.
    ChangeGuardValue { index: usize, delta: i64 },
    /// Change an integer constant in a condition.
    ChangeIntConst { index: usize, old_val: i64, new_val: i64 },
    /// Swap the comparison operator in a guard.
    SwapGuardOp { index: usize, new_op: ComparisonOp },
    /// Add a guard condition.
    AddGuard { op: ComparisonOp, var: String, value: i64 },
    /// Add a positive body condition (from known predicates).
    AddPositiveCondition(Neuron),
    /// Add a negated body condition (`not (pattern)`).
    AddNegatedCondition(Neuron),
    /// Replace a variable in the head with a constant.
    SpecializeHeadVar { var_name: String, value: i64 },
    /// Add an arithmetic condition: `(op $lhs $rhs $result_var)`
    AddArithmetic { op: ArithmeticOp, lhs: String, rhs: String, result_var: String },
    /// Add an aggregate condition: `(op (pattern) $bind_var -> $result_var)`
    AddAggregate { op: AggregateOp, pattern: Neuron, bind_var: String, result_var: String },
}

/// Apply a mutation to a rule, returning a new rule.
pub fn mutate_rule(rule: &Rule, mutation: &Mutation) -> Option<Rule> {
    let mut new_body = rule.body.clone();

    match mutation {
        Mutation::RemoveCondition(idx) => {
            if *idx >= new_body.len() || new_body.len() <= 1 {
                return None;
            }
            new_body.remove(*idx);
        }

        Mutation::ChangeGuardValue { index, delta } => {
            if *index >= new_body.len() {
                return None;
            }
            match &new_body[*index] {
                Condition::Guard(op, lhs, rhs) => {
                    let new_rhs = match rhs {
                        Neuron::Value(QorValue::Int(v)) => {
                            Neuron::Value(QorValue::Int(v + delta))
                        }
                        Neuron::Value(QorValue::Float(v)) => {
                            Neuron::Value(QorValue::Float(v + *delta as f64))
                        }
                        _ => return None,
                    };
                    new_body[*index] = Condition::Guard(op.clone(), lhs.clone(), new_rhs);
                }
                _ => return None,
            }
        }

        Mutation::ChangeIntConst { index, old_val, new_val } => {
            if *index >= new_body.len() {
                return None;
            }
            let modified = rewrite_int_in_condition(&new_body[*index], *old_val, *new_val);
            if let Some(c) = modified {
                new_body[*index] = c;
            } else {
                return None;
            }
        }

        Mutation::SwapGuardOp { index, new_op } => {
            if *index >= new_body.len() {
                return None;
            }
            match &new_body[*index] {
                Condition::Guard(_, lhs, rhs) => {
                    new_body[*index] = Condition::Guard(new_op.clone(), lhs.clone(), rhs.clone());
                }
                _ => return None,
            }
        }

        Mutation::AddGuard { op, var, value } => {
            new_body.push(Condition::Guard(
                op.clone(),
                Neuron::Variable(var.clone()),
                Neuron::Value(QorValue::Int(*value)),
            ));
        }

        Mutation::AddPositiveCondition(pattern) => {
            new_body.push(Condition::Positive(pattern.clone()));
        }

        Mutation::AddNegatedCondition(pattern) => {
            new_body.push(Condition::Negated(pattern.clone()));
        }

        Mutation::SpecializeHeadVar { var_name, value } => {
            let new_head = specialize_var(&rule.head, var_name, *value);
            let new_body_spec: Vec<Condition> = new_body.iter()
                .map(|c| specialize_var_in_condition(c, var_name, *value))
                .collect();
            return Some(Rule {
                head: new_head,
                body: new_body_spec,
                tv: rule.tv,
                stratum: rule.stratum,
            });
        }

        Mutation::AddArithmetic { op, lhs, rhs, result_var } => {
            new_body.push(Condition::Arithmetic {
                op: op.clone(),
                lhs: Neuron::Variable(lhs.clone()),
                rhs: Neuron::Variable(rhs.clone()),
                result_var: result_var.clone(),
            });
        }

        Mutation::AddAggregate { op, pattern, bind_var, result_var } => {
            new_body.push(Condition::Aggregate {
                op: op.clone(),
                pattern: pattern.clone(),
                bind_var: bind_var.clone(),
                result_var: result_var.clone(),
            });
        }
    }

    Some(Rule {
        head: rule.head.clone(),
        body: new_body,
        tv: rule.tv,
        stratum: rule.stratum,
    })
}

/// Generate all reasonable single-step mutations of a rule.
pub fn generate_mutations(rule: &Rule) -> Vec<Rule> {
    let mut results = Vec::new();

    // Collect all integer constants in the rule (body + head) for substitution
    let mut all_ints = std::collections::HashSet::new();
    collect_ints(&rule.head, &mut all_ints);
    for cond in &rule.body {
        match cond {
            Condition::Positive(n) | Condition::Negated(n) | Condition::NegatedPresent(n) => {
                collect_ints(n, &mut all_ints);
            }
            Condition::Guard(_, lhs, rhs) => {
                collect_ints(lhs, &mut all_ints);
                collect_ints(rhs, &mut all_ints);
            }
            _ => {}
        }
    }

    for i in 0..rule.body.len() {
        // Try removing each condition
        if rule.body.len() > 1 {
            if let Some(r) = mutate_rule(rule, &Mutation::RemoveCondition(i)) {
                results.push(r);
            }
        }

        // Try changing guard values
        if let Condition::Guard(_, _, _) = &rule.body[i] {
            for delta in &[-1, 1, -2, 2, -5, 5] {
                if let Some(r) = mutate_rule(rule, &Mutation::ChangeGuardValue { index: i, delta: *delta }) {
                    results.push(r);
                }
            }
            // Try swapping guard operators
            for op in &[ComparisonOp::Gt, ComparisonOp::Lt, ComparisonOp::Ge, ComparisonOp::Le, ComparisonOp::Eq, ComparisonOp::Ne] {
                if let Condition::Guard(current_op, _, _) = &rule.body[i] {
                    if current_op != op {
                        if let Some(r) = mutate_rule(rule, &Mutation::SwapGuardOp { index: i, new_op: op.clone() }) {
                            results.push(r);
                        }
                    }
                }
            }
        }

        // Try changing integer constants — use values discovered in the rule
        if let Some(consts) = extract_int_consts(&rule.body[i]) {
            for &old_v in &consts {
                // Try each other constant found in the rule
                for &new_v in &all_ints {
                    if new_v != old_v {
                        if let Some(r) = mutate_rule(rule, &Mutation::ChangeIntConst {
                            index: i, old_val: old_v, new_val: new_v,
                        }) {
                            results.push(r);
                        }
                    }
                }
                // Also try ±1 from the original value
                for delta in [-1i64, 1] {
                    let new_v = old_v + delta;
                    if !all_ints.contains(&new_v) {
                        if let Some(r) = mutate_rule(rule, &Mutation::ChangeIntConst {
                            index: i, old_val: old_v, new_val: new_v,
                        }) {
                            results.push(r);
                        }
                    }
                }
            }
        }
    }

    // Also mutate integer constants in the rule HEAD
    let mut head_ints = Vec::new();
    collect_ints_vec(&rule.head, &mut head_ints);
    for &old_v in &head_ints {
        for &new_v in &all_ints {
            if new_v != old_v {
                let new_head = rewrite_int_in_neuron(&rule.head, old_v, new_v);
                results.push(Rule {
                    head: new_head,
                    body: rule.body.clone(),
                    tv: rule.tv,
                    stratum: rule.stratum,
                });
            }
        }
        // Also try ±1 from the head value
        for delta in [-1i64, 1] {
            let new_v = old_v + delta;
            if !all_ints.contains(&new_v) {
                let new_head = rewrite_int_in_neuron(&rule.head, old_v, new_v);
                results.push(Rule {
                    head: new_head,
                    body: rule.body.clone(),
                    tv: rule.tv,
                    stratum: rule.stratum,
                });
            }
        }
    }

    results
}

/// Extract integer constants from a condition.
fn extract_int_consts(cond: &Condition) -> Option<Vec<i64>> {
    let mut vals = Vec::new();
    match cond {
        Condition::Positive(n) | Condition::Negated(n) | Condition::NegatedPresent(n) => {
            collect_ints_vec(n, &mut vals);
        }
        Condition::Guard(_, lhs, rhs) => {
            collect_ints_vec(lhs, &mut vals);
            collect_ints_vec(rhs, &mut vals);
        }
        _ => {}
    }
    if vals.is_empty() { None } else { Some(vals) }
}

/// Collect all integer constants in a Neuron into a HashSet.
fn collect_ints(n: &Neuron, out: &mut std::collections::HashSet<i64>) {
    match n {
        Neuron::Value(QorValue::Int(v)) => { out.insert(*v); }
        Neuron::Expression(parts) => {
            for p in parts { collect_ints(p, out); }
        }
        _ => {}
    }
}

/// Collect all integer constants in a Neuron into a Vec.
fn collect_ints_vec(n: &Neuron, out: &mut Vec<i64>) {
    match n {
        Neuron::Value(QorValue::Int(v)) => { out.push(*v); }
        Neuron::Expression(parts) => {
            for p in parts { collect_ints_vec(p, out); }
        }
        _ => {}
    }
}

/// Rewrite a specific integer constant in a condition.
fn rewrite_int_in_condition(cond: &Condition, old: i64, new: i64) -> Option<Condition> {
    match cond {
        Condition::Positive(n) => {
            Some(Condition::Positive(rewrite_int_in_neuron(n, old, new)))
        }
        Condition::Negated(n) => {
            Some(Condition::Negated(rewrite_int_in_neuron(n, old, new)))
        }
        Condition::NegatedPresent(n) => {
            Some(Condition::NegatedPresent(rewrite_int_in_neuron(n, old, new)))
        }
        Condition::Guard(op, lhs, rhs) => {
            Some(Condition::Guard(
                op.clone(),
                rewrite_int_in_neuron(lhs, old, new),
                rewrite_int_in_neuron(rhs, old, new),
            ))
        }
        _ => None,
    }
}

fn rewrite_int_in_neuron(n: &Neuron, old: i64, new: i64) -> Neuron {
    match n {
        Neuron::Value(QorValue::Int(v)) if *v == old => {
            Neuron::Value(QorValue::Int(new))
        }
        Neuron::Expression(parts) => {
            Neuron::Expression(parts.iter().map(|p| rewrite_int_in_neuron(p, old, new)).collect())
        }
        other => other.clone(),
    }
}

/// Replace a variable with a constant in a Neuron.
fn specialize_var(n: &Neuron, var: &str, val: i64) -> Neuron {
    match n {
        Neuron::Variable(v) if v == var => Neuron::Value(QorValue::Int(val)),
        Neuron::Expression(parts) => {
            Neuron::Expression(parts.iter().map(|p| specialize_var(p, var, val)).collect())
        }
        other => other.clone(),
    }
}

/// Replace a variable with a constant in a Condition.
fn specialize_var_in_condition(cond: &Condition, var: &str, val: i64) -> Condition {
    match cond {
        Condition::Positive(n) => Condition::Positive(specialize_var(n, var, val)),
        Condition::Negated(n) => Condition::Negated(specialize_var(n, var, val)),
        Condition::NegatedPresent(n) => Condition::NegatedPresent(specialize_var(n, var, val)),
        Condition::Guard(op, lhs, rhs) => {
            Condition::Guard(op.clone(), specialize_var(lhs, var, val), specialize_var(rhs, var, val))
        }
        other => other.clone(),
    }
}

/// Extract all variable names from a rule.
pub fn extract_variables(rule: &Rule) -> Vec<String> {
    let mut vars = Vec::new();
    collect_vars(&rule.head, &mut vars);
    for cond in &rule.body {
        match cond {
            Condition::Positive(n) | Condition::Negated(n) | Condition::NegatedPresent(n) => {
                collect_vars(n, &mut vars);
            }
            Condition::Guard(_, lhs, rhs) => {
                collect_vars(lhs, &mut vars);
                collect_vars(rhs, &mut vars);
            }
            _ => {}
        }
    }
    vars.sort();
    vars.dedup();
    vars
}

fn collect_vars(n: &Neuron, out: &mut Vec<String>) {
    match n {
        Neuron::Variable(v) => out.push(v.clone()),
        Neuron::Expression(parts) => {
            for p in parts { collect_vars(p, out); }
        }
        _ => {}
    }
}

/// Generate context-aware mutations using known predicates and values.
///
/// Unlike generate_mutations (which only does structural changes),
/// this uses knowledge about available predicates to ADD meaningful
/// conditions, specialize variables, and add negations.
pub fn generate_context_mutations(
    rule: &Rule,
    known_predicates: &[(String, usize)],  // (pred_name, arity)
    known_values: &[i64],                   // values seen in data
) -> Vec<Rule> {
    let mut results = Vec::new();
    let vars = extract_variables(rule);

    // 1. Specialize head variables with known values
    //    e.g., (predict-cell $a0 $a1 $a2 $a3) → (predict-cell $a0 $a1 $a2 7)
    //    This creates rules that predict specific colors.
    for var in &vars {
        for &val in known_values {
            if let Some(r) = mutate_rule(rule, &Mutation::SpecializeHeadVar {
                var_name: var.clone(), value: val,
            }) {
                results.push(r);
            }
        }
    }

    // 2. Add positive conditions from known predicates
    //    e.g., add (detected-identity) or (obs-consistent same-size)
    for (pred, arity) in known_predicates {
        if *arity == 0 { continue; }
        // For nullary-style facts (arity 1 = just the predicate symbol)
        if *arity == 1 {
            let pattern = Neuron::Expression(vec![Neuron::Symbol(pred.clone())]);
            if let Some(r) = mutate_rule(rule, &Mutation::AddPositiveCondition(pattern.clone())) {
                results.push(r);
            }
            // Also try negated
            if let Some(r) = mutate_rule(rule, &Mutation::AddNegatedCondition(pattern)) {
                results.push(r);
            }
        }
        // For predicates with matching arity to source, try joining on shared variables
        if *arity <= vars.len() + 1 {
            let mut args: Vec<Neuron> = vec![Neuron::Symbol(pred.clone())];
            for i in 0..(*arity - 1).min(vars.len()) {
                args.push(Neuron::Variable(vars[i].clone()));
            }
            // Pad remaining with fresh vars
            for i in vars.len()..(*arity - 1) {
                args.push(Neuron::Variable(format!("$ctx{}", i)));
            }
            let pattern = Neuron::Expression(args);
            if let Some(r) = mutate_rule(rule, &Mutation::AddPositiveCondition(pattern)) {
                results.push(r);
            }
        }
    }

    // 3. Add guard conditions with known values
    for var in &vars {
        for &val in known_values.iter().take(10) {
            for op in &[ComparisonOp::Eq, ComparisonOp::Ne, ComparisonOp::Gt, ComparisonOp::Lt] {
                if let Some(r) = mutate_rule(rule, &Mutation::AddGuard {
                    op: op.clone(), var: var.clone(), value: val,
                }) {
                    results.push(r);
                }
            }
        }
    }

    // 4. Add arithmetic conditions between sequential variable pairs
    //    e.g., (- $a1 $a2 $diff) or (+ $a1 1 $shifted)
    if vars.len() >= 2 {
        let mut arith_count = 0;
        for i in 0..vars.len() {
            for j in (i+1)..vars.len() {
                if arith_count >= 20 { break; }
                // Difference
                let diff_var = format!("$diff_{}_{}", i, j);
                if let Some(r) = mutate_rule(rule, &Mutation::AddArithmetic {
                    op: ArithmeticOp::Sub,
                    lhs: vars[i].clone(), rhs: vars[j].clone(),
                    result_var: diff_var,
                }) {
                    results.push(r);
                    arith_count += 1;
                }
                // Sum
                let sum_var = format!("$sum_{}_{}", i, j);
                if let Some(r) = mutate_rule(rule, &Mutation::AddArithmetic {
                    op: ArithmeticOp::Add,
                    lhs: vars[i].clone(), rhs: vars[j].clone(),
                    result_var: sum_var,
                }) {
                    results.push(r);
                    arith_count += 1;
                }
            }
        }
    }

    results
}

/// Format a Rule back to QOR text for parsing / display.
pub fn rule_to_qor(rule: &Rule) -> String {
    let body_strs: Vec<String> = rule.body.iter().map(|c| format!("{}", c)).collect();
    format!("{} if {} <{:.2}, {:.2}>",
        rule.head, body_strs.join(" "),
        rule.tv.strength, rule.tv.confidence)
}

/// Single-point crossover on rule bodies.
/// Splits each parent's condition list at a random point and swaps tails.
pub fn crossover_rules(r1: &Rule, r2: &Rule, rng: &mut impl Rng) -> (Rule, Rule) {
    let cut1 = if r1.body.is_empty() { 0 } else { rng.gen_range(0..=r1.body.len()) };
    let cut2 = if r2.body.is_empty() { 0 } else { rng.gen_range(0..=r2.body.len()) };

    let mut child1_body = r1.body[..cut1].to_vec();
    child1_body.extend_from_slice(&r2.body[cut2..]);

    let mut child2_body = r2.body[..cut2].to_vec();
    child2_body.extend_from_slice(&r1.body[cut1..]);

    // Keep at least 1 condition per child
    if child1_body.is_empty() && !r1.body.is_empty() {
        child1_body.push(r1.body[0].clone());
    }
    if child2_body.is_empty() && !r2.body.is_empty() {
        child2_body.push(r2.body[0].clone());
    }

    (
        Rule { head: r1.head.clone(), body: child1_body, tv: r1.tv, stratum: 0 },
        Rule { head: r2.head.clone(), body: child2_body, tv: r2.tv, stratum: 0 },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use qor_core::truth_value::TruthValue;

    fn sample_rule() -> Rule {
        Rule {
            head: Neuron::Expression(vec![
                Neuron::Symbol("target".into()),
                Neuron::Variable("id".into()),
                Neuron::Variable("r".into()),
                Neuron::Variable("c".into()),
                Neuron::Value(QorValue::Int(7)),
            ]),
            body: vec![
                Condition::Positive(Neuron::Expression(vec![
                    Neuron::Symbol("source".into()),
                    Neuron::Variable("id".into()),
                    Neuron::Variable("r".into()),
                    Neuron::Variable("c".into()),
                    Neuron::Value(QorValue::Int(3)),
                ])),
                Condition::Guard(
                    ComparisonOp::Gt,
                    Neuron::Variable("r".into()),
                    Neuron::Value(QorValue::Int(2)),
                ),
            ],
            tv: TruthValue::new(0.95, 0.95),
            stratum: 0,
        }
    }

    #[test]
    fn test_remove_condition() {
        let rule = sample_rule();
        let mutated = mutate_rule(&rule, &Mutation::RemoveCondition(1)).unwrap();
        assert_eq!(mutated.body.len(), 1);
    }

    #[test]
    fn test_change_guard_value() {
        let rule = sample_rule();
        let mutated = mutate_rule(&rule, &Mutation::ChangeGuardValue { index: 1, delta: 3 }).unwrap();
        if let Condition::Guard(_, _, Neuron::Value(QorValue::Int(v))) = &mutated.body[1] {
            assert_eq!(*v, 5); // 2 + 3
        } else {
            panic!("Expected guard with int");
        }
    }

    #[test]
    fn test_change_int_const() {
        let rule = sample_rule();
        let mutated = mutate_rule(&rule, &Mutation::ChangeIntConst {
            index: 0, old_val: 3, new_val: 5,
        }).unwrap();
        let body_str = format!("{}", mutated.body[0]);
        assert!(body_str.contains("5"));
        assert!(!body_str.contains("3"));
    }

    #[test]
    fn test_generate_mutations_nonempty() {
        let rule = sample_rule();
        let mutations = generate_mutations(&rule);
        assert!(!mutations.is_empty());
        assert!(mutations.len() >= 15);
    }

    #[test]
    fn test_rule_to_qor() {
        let rule = sample_rule();
        let text = rule_to_qor(&rule);
        assert!(text.contains("target"));
        assert!(text.contains("source"));
        assert!(text.contains("if"));
    }

    #[test]
    fn test_swap_guard_op() {
        let rule = sample_rule();
        let mutated = mutate_rule(&rule, &Mutation::SwapGuardOp {
            index: 1, new_op: ComparisonOp::Le,
        }).unwrap();
        if let Condition::Guard(op, _, _) = &mutated.body[1] {
            assert_eq!(*op, ComparisonOp::Le);
        } else {
            panic!("Expected guard");
        }
    }
}
