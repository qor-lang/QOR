// ── QOR Macros ─────────────────────────────────────────────────────────
//
// Ergonomic macros for embedding QOR in Rust code.
// All stable `macro_rules!` — no nightly features needed.
//
// Convention: Rust underscores become QOR hyphens.
//   neuron!(stiff_neck carol) → (stiff-neck carol)

/// Parse a QOR source string into statements.
///
/// # Examples
/// ```
/// let stmts = qor_core::qor!("(bird tweety) <0.99>").unwrap();
/// assert_eq!(stmts.len(), 1);
/// ```
#[macro_export]
macro_rules! qor {
    ($source:expr) => {
        $crate::parser::parse($source)
    };
}

/// Build a Neuron::Expression from symbol identifiers.
/// Underscores in identifiers are converted to hyphens (Rust → QOR convention).
///
/// # Examples
/// ```
/// use qor_core::neuron;
/// let n = neuron!(bird tweety);
/// assert_eq!(n.to_string(), "(bird tweety)");
///
/// let n = neuron!(stiff_neck carol);
/// assert_eq!(n.to_string(), "(stiff-neck carol)");
/// ```
#[macro_export]
macro_rules! neuron {
    // Single symbol (not an expression)
    (@sym $sym:ident) => {
        $crate::neuron::Neuron::Symbol(stringify!($sym).replace('_', "-"))
    };
    // Variable marker: var(name)
    (@var $name:ident) => {
        $crate::neuron::Neuron::Variable(stringify!($name).replace('_', "-"))
    };
    // Expression: neuron!(bird tweety) → (bird tweety)
    ($($sym:ident)+) => {
        $crate::neuron::Neuron::Expression(vec![
            $($crate::neuron::Neuron::Symbol(stringify!($sym).replace('_', "-"))),+
        ])
    };
}

/// Build a TruthValue.
///
/// # Examples
/// ```
/// use qor_core::tv;
/// let t = tv!(0.99);         // strength only, default confidence
/// let t = tv!(0.99, 0.90);   // strength + confidence
/// ```
#[macro_export]
macro_rules! tv {
    ($s:expr) => {
        $crate::truth_value::TruthValue::from_strength($s)
    };
    ($s:expr, $c:expr) => {
        $crate::truth_value::TruthValue::new($s, $c)
    };
}

/// Assert a fact into a NeuronStore.
///
/// # Examples
/// ```ignore
/// use qor_core::{neuron, tv};
/// let mut store = qor_runtime::store::NeuronStore::new();
/// assert_fact!(store, bird tweety => 0.99);
/// assert_fact!(store, bird eagle => 0.95, 0.90);
/// ```
#[macro_export]
macro_rules! assert_fact {
    ($store:expr, $($sym:ident)+ => $s:expr) => {
        $store.insert(
            $crate::neuron!($($sym)+),
            $crate::tv!($s),
        )
    };
    ($store:expr, $($sym:ident)+ => $s:expr, $c:expr) => {
        $store.insert(
            $crate::neuron!($($sym)+),
            $crate::tv!($s, $c),
        )
    };
}

/// Run a QOR program and collect query results.
///
/// # Examples
/// ```
/// let results = qor_core::qor_run!("
///     (bird tweety) <0.99>
///     ? (bird $x)
/// ");
/// ```
#[macro_export]
macro_rules! qor_run {
    ($source:expr) => {{
        let stmts = $crate::parser::parse($source);
        stmts
    }};
}

#[cfg(test)]
mod tests {
    use crate::neuron::Neuron;

    #[test]
    fn test_neuron_macro_simple() {
        let n = neuron!(bird tweety);
        assert_eq!(n.to_string(), "(bird tweety)");
    }

    #[test]
    fn test_neuron_macro_three_parts() {
        let n = neuron!(claim topic source);
        assert_eq!(n.to_string(), "(claim topic source)");
    }

    #[test]
    fn test_neuron_macro_underscore_to_hyphen() {
        let n = neuron!(stiff_neck carol);
        assert_eq!(n.to_string(), "(stiff-neck carol)");
    }

    #[test]
    fn test_neuron_macro_multi_hyphen() {
        let n = neuron!(very_long_predicate alice);
        assert_eq!(n.to_string(), "(very-long-predicate alice)");
    }

    #[test]
    fn test_tv_macro_strength_only() {
        let t = tv!(0.99);
        assert!((t.strength - 0.99).abs() < 0.001);
        assert!((t.confidence - 0.90).abs() < 0.001); // default confidence
    }

    #[test]
    fn test_tv_macro_two_component() {
        let t = tv!(0.85, 0.70);
        assert!((t.strength - 0.85).abs() < 0.001);
        assert!((t.confidence - 0.70).abs() < 0.001);
    }

    #[test]
    fn test_qor_macro_parse() {
        let stmts = qor!("(bird tweety) <0.99>").unwrap();
        assert_eq!(stmts.len(), 1);
    }

    #[test]
    fn test_qor_macro_multi_statement() {
        let stmts = qor!("
            (bird tweety) <0.99>
            (bird eagle) <0.95>
            ? (bird $x)
        ")
        .unwrap();
        assert_eq!(stmts.len(), 3);
    }

    #[test]
    fn test_neuron_macro_sym() {
        let s = neuron!(@sym bird);
        assert_eq!(s, Neuron::Symbol("bird".to_string()));
    }

    #[test]
    fn test_neuron_macro_var() {
        let v = neuron!(@var x);
        assert_eq!(v, Neuron::Variable("x".to_string()));
    }
}
