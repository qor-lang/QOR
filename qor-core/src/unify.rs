use std::collections::HashMap;

use crate::neuron::Neuron;

/// Variable bindings: maps variable names to concrete neurons.
pub type Bindings = HashMap<String, Neuron>;

/// Unify a pattern (with variables) against a concrete neuron.
/// Returns bindings if successful, None if no match.
///
/// Variables in `pattern` bind to whatever they match in `target`.
/// If a variable appears twice, both occurrences must match the same thing.
pub fn unify(pattern: &Neuron, target: &Neuron) -> Option<Bindings> {
    let mut bindings = Bindings::new();
    if unify_inner(pattern, target, &mut bindings) {
        Some(bindings)
    } else {
        None
    }
}

fn unify_inner(pattern: &Neuron, target: &Neuron, bindings: &mut Bindings) -> bool {
    match (pattern, target) {
        // Pattern variable → bind or check consistency
        (Neuron::Variable(name), _) => {
            if let Some(existing) = bindings.get(name) {
                *existing == *target
            } else {
                bindings.insert(name.clone(), target.clone());
                true
            }
        }

        // Target variable → matches anything (wildcards in store)
        (_, Neuron::Variable(_)) => true,

        // Symbols must be equal
        (Neuron::Symbol(a), Neuron::Symbol(b)) => a == b,

        // Values must be equal
        (Neuron::Value(a), Neuron::Value(b)) => a == b,

        // Expressions: same length, all children unify
        (Neuron::Expression(a), Neuron::Expression(b)) => {
            a.len() == b.len()
                && a.iter()
                    .zip(b.iter())
                    .all(|(pa, pb)| unify_inner(pa, pb, bindings))
        }

        _ => false,
    }
}

/// Substitute all bound variables in a neuron.
/// Unbound variables remain as-is.
pub fn apply_bindings(neuron: &Neuron, bindings: &Bindings) -> Neuron {
    match neuron {
        Neuron::Variable(name) => bindings
            .get(name)
            .cloned()
            .unwrap_or_else(|| neuron.clone()),
        Neuron::Expression(children) => {
            Neuron::Expression(children.iter().map(|c| apply_bindings(c, bindings)).collect())
        }
        other => other.clone(),
    }
}

/// Extract variable names from a neuron, in the order they first appear.
pub fn extract_variables(neuron: &Neuron) -> Vec<String> {
    let mut vars = Vec::new();
    extract_vars_inner(neuron, &mut vars);
    vars
}

fn extract_vars_inner(neuron: &Neuron, vars: &mut Vec<String>) {
    match neuron {
        Neuron::Variable(name) => {
            if !vars.contains(name) {
                vars.push(name.clone());
            }
        }
        Neuron::Expression(children) => {
            for child in children {
                extract_vars_inner(child, vars);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::Neuron;

    #[test]
    fn test_unify_concrete() {
        let a = Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]);
        let b = Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]);
        let bindings = unify(&a, &b).unwrap();
        assert!(bindings.is_empty());
    }

    #[test]
    fn test_unify_variable() {
        let pattern = Neuron::expression(vec![Neuron::symbol("bird"), Neuron::variable("x")]);
        let target = Neuron::expression(vec![Neuron::symbol("bird"), Neuron::symbol("tweety")]);
        let bindings = unify(&pattern, &target).unwrap();
        assert_eq!(bindings.get("x").unwrap(), &Neuron::symbol("tweety"));
    }

    #[test]
    fn test_unify_no_match() {
        let pattern = Neuron::expression(vec![Neuron::symbol("bird"), Neuron::variable("x")]);
        let target = Neuron::expression(vec![Neuron::symbol("fish"), Neuron::symbol("salmon")]);
        assert!(unify(&pattern, &target).is_none());
    }

    #[test]
    fn test_unify_repeated_variable() {
        // (same $x $x) should match (same a a) but not (same a b)
        let pattern = Neuron::expression(vec![
            Neuron::symbol("same"),
            Neuron::variable("x"),
            Neuron::variable("x"),
        ]);

        let good = Neuron::expression(vec![
            Neuron::symbol("same"),
            Neuron::symbol("a"),
            Neuron::symbol("a"),
        ]);
        assert!(unify(&pattern, &good).is_some());

        let bad = Neuron::expression(vec![
            Neuron::symbol("same"),
            Neuron::symbol("a"),
            Neuron::symbol("b"),
        ]);
        assert!(unify(&pattern, &bad).is_none());
    }

    #[test]
    fn test_apply_bindings() {
        let neuron = Neuron::expression(vec![Neuron::symbol("flies"), Neuron::variable("x")]);
        let mut bindings = Bindings::new();
        bindings.insert("x".to_string(), Neuron::symbol("tweety"));
        let result = apply_bindings(&neuron, &bindings);
        assert_eq!(
            result,
            Neuron::expression(vec![Neuron::symbol("flies"), Neuron::symbol("tweety")])
        );
    }

    #[test]
    fn test_apply_bindings_unbound_stays() {
        let neuron = Neuron::expression(vec![Neuron::symbol("flies"), Neuron::variable("y")]);
        let bindings = Bindings::new();
        let result = apply_bindings(&neuron, &bindings);
        assert_eq!(result, neuron);
    }

    #[test]
    fn test_extract_variables() {
        let n = Neuron::expression(vec![
            Neuron::symbol("claim"),
            Neuron::variable("what"),
            Neuron::variable("source"),
        ]);
        let vars = extract_variables(&n);
        assert_eq!(vars, vec!["what", "source"]);
    }

    #[test]
    fn test_extract_variables_no_duplicates() {
        let n = Neuron::expression(vec![
            Neuron::symbol("same"),
            Neuron::variable("x"),
            Neuron::variable("x"),
        ]);
        let vars = extract_variables(&n);
        assert_eq!(vars, vec!["x"]);
    }
}
