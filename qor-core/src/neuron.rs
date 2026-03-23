use crate::truth_value::TruthValue;
use std::fmt;

/// The core data unit in QORlang.
/// Everything in QOR is a Neuron — facts, rules, queries, values.
#[derive(Debug, Clone, PartialEq)]
pub enum Neuron {
    /// A named thing: `bird`, `tweety`, `wikipedia`
    Symbol(String),

    /// A placeholder for pattern matching: `$x`, `$claim`
    Variable(String),

    /// A compound structure: `(bird tweety)`, `(flies $x)`
    Expression(Vec<Neuron>),

    /// A literal value
    Value(QorValue),
}

/// Literal values that neurons can hold.
#[derive(Debug, Clone, PartialEq)]
pub enum QorValue {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

/// Comparison operators for guard conditions in rules.
#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonOp {
    Gt,  // >
    Lt,  // <
    Ge,  // >=
    Le,  // <=
    Eq,  // ==
    Ne,  // !=
}

/// Arithmetic operators for computing values in rule bodies.
/// Syntax: `(+ $a $b $result)` — evaluates a op b, binds to $result.
#[derive(Debug, Clone, PartialEq)]
pub enum ArithmeticOp {
    Add,      // +
    Sub,      // -
    Mul,      // *
    Div,      // /
    Mod,      // %
    Min,      // min
    Max,      // max
    Sqrt,     // sqrt  (unary)
    Abs,      // abs   (unary)
    Power,    // power (binary)
    DigitSum, // digit-sum (unary)
}

/// Returns true if this arithmetic op is unary (1 operand + result).
pub fn is_unary_arith(op: &ArithmeticOp) -> bool {
    matches!(op, ArithmeticOp::Sqrt | ArithmeticOp::Abs | ArithmeticOp::DigitSum)
}

/// A neuron stored in the NeuronStore with its truth value and metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredNeuron {
    pub neuron: Neuron,
    pub tv: TruthValue,
    pub timestamp: Option<i64>,
    pub decay_rate: Option<f64>,
    /// True if this fact was derived by forward chaining (not directly asserted).
    pub inferred: bool,
}

/// Aggregate operations for counting/summing over matched facts.
#[derive(Debug, Clone, PartialEq)]
pub enum AggregateOp {
    Count,
    Sum,
    Min,
    Max,
}

/// A condition in a rule body — positive, negated, guard, aggregate, or arithmetic.
#[derive(Debug, Clone, PartialEq)]
pub enum Condition {
    /// Must be true: `(bird $x)`
    Positive(Neuron),
    /// Must NOT be true (negation as failure): `not (penguin $x)`
    Negated(Neuron),
    /// Must NOT be present as a base (non-inferred) fact: `not-present (grid-cell ...)`
    /// Unlike `Negated`, this only checks asserted facts, ignoring derived ones.
    NegatedPresent(Neuron),
    /// Numeric comparison guard: `(> $v 70)`, `(<= $x $y)`
    Guard(ComparisonOp, Neuron, Neuron),
    /// Aggregate: `(count (bird $x) $x -> $n)`
    Aggregate {
        op: AggregateOp,
        pattern: Neuron,
        bind_var: String,
        result_var: String,
    },
    /// Arithmetic: `(+ $a $b $result)` — compute a op b, bind to result
    Arithmetic {
        op: ArithmeticOp,
        lhs: Neuron,
        rhs: Neuron,
        result_var: String,
    },
    /// Binary KB lookup: `(lookup "predicate" $subject $result)`
    /// Queries the binary KnowledgeBase for matching facts.
    Lookup {
        predicate: String,
        subject: Neuron,
        result_var: String,
    },
    /// Digit check: `(ends-with $n $d)` — true if last digit of $n equals $d
    EndsWith(Neuron, Neuron),
}

impl fmt::Display for Condition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Condition::Positive(n) => write!(f, "{}", n),
            Condition::Negated(n) => write!(f, "not {}", n),
            Condition::NegatedPresent(n) => write!(f, "not-present {}", n),
            Condition::Guard(op, lhs, rhs) => {
                let op_str = match op {
                    ComparisonOp::Gt => ">",
                    ComparisonOp::Lt => "<",
                    ComparisonOp::Ge => ">=",
                    ComparisonOp::Le => "<=",
                    ComparisonOp::Eq => "==",
                    ComparisonOp::Ne => "!=",
                };
                write!(f, "({} {} {})", op_str, lhs, rhs)
            }
            Condition::Aggregate { op, pattern, bind_var, result_var } => {
                write!(f, "({} {} ${} -> ${})", op, pattern, bind_var, result_var)
            }
            Condition::Arithmetic { op, lhs, rhs, result_var } => {
                if is_unary_arith(op) {
                    write!(f, "({} {} ${})", op, lhs, result_var)
                } else {
                    write!(f, "({} {} {} ${})", op, lhs, rhs, result_var)
                }
            }
            Condition::Lookup { predicate, subject, result_var } => {
                write!(f, "(lookup \"{}\" {} ${})", predicate, subject, result_var)
            }
            Condition::EndsWith(n, d) => {
                write!(f, "(ends-with {} {})", n, d)
            }
        }
    }
}

impl fmt::Display for ArithmeticOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArithmeticOp::Add => write!(f, "+"),
            ArithmeticOp::Sub => write!(f, "-"),
            ArithmeticOp::Mul => write!(f, "*"),
            ArithmeticOp::Div => write!(f, "/"),
            ArithmeticOp::Mod => write!(f, "%"),
            ArithmeticOp::Min => write!(f, "min"),
            ArithmeticOp::Max => write!(f, "max"),
            ArithmeticOp::Sqrt => write!(f, "sqrt"),
            ArithmeticOp::Abs => write!(f, "abs"),
            ArithmeticOp::Power => write!(f, "power"),
            ArithmeticOp::DigitSum => write!(f, "digit-sum"),
        }
    }
}

impl fmt::Display for AggregateOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AggregateOp::Count => write!(f, "count"),
            AggregateOp::Sum => write!(f, "sum"),
            AggregateOp::Min => write!(f, "min"),
            AggregateOp::Max => write!(f, "max"),
        }
    }
}

/// What a @test block expects: a fact present or absent.
#[derive(Debug, Clone, PartialEq)]
pub enum TestExpect {
    /// `expect (overbought btc)` — fact should exist after chaining
    Present(Neuron),
    /// `expect not (overbought btc)` — fact should NOT exist
    Absent(Neuron),
}

/// A parsed QOR statement — one line/block in a .qor file.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// A fact: `(bird tweety) <0.99>` or `(news headline) <0.95> @decay 0.01`
    Fact {
        neuron: Neuron,
        tv: Option<TruthValue>,
        decay: Option<f64>,
    },

    /// A query: `? (bird tweety)` or `? (bird $x)`
    Query {
        pattern: Neuron,
    },

    /// A rule: `(flies $x) if (bird $x) not (penguin $x) <0.95>`
    Rule {
        head: Neuron,
        body: Vec<Condition>,
        tv: Option<TruthValue>,
    },

    /// A test block: `@test name given (...) expect (...)`
    Test {
        name: String,
        given: Vec<Neuron>,
        expect: Vec<TestExpect>,
    },
}

impl Statement {
    /// Create a fact from an expression with default truth value (0.99, 0.99).
    pub fn simple_fact(parts: Vec<Neuron>) -> Self {
        Statement::Fact {
            neuron: Neuron::Expression(parts),
            tv: Some(TruthValue::new(0.99, 0.99)),
            decay: None,
        }
    }
}

impl Neuron {
    pub fn symbol(name: &str) -> Self {
        Neuron::Symbol(name.to_string())
    }

    pub fn variable(name: &str) -> Self {
        Neuron::Variable(name.to_string())
    }

    pub fn expression(children: Vec<Neuron>) -> Self {
        Neuron::Expression(children)
    }

    pub fn str_val(s: &str) -> Self {
        Neuron::Value(QorValue::Str(s.to_string()))
    }

    pub fn int_val(n: i64) -> Self {
        Neuron::Value(QorValue::Int(n))
    }

    pub fn float_val(f: f64) -> Self {
        Neuron::Value(QorValue::Float(f))
    }

    /// Extract a numeric value as f64 (works for Int and Float).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Neuron::Value(QorValue::Float(f)) => Some(*f),
            Neuron::Value(QorValue::Int(i)) => Some(*i as f64),
            _ => None,
        }
    }

    /// Extract a numeric value as usize (works for non-negative Int and Float).
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Neuron::Value(QorValue::Int(i)) if *i >= 0 => Some(*i as usize),
            Neuron::Value(QorValue::Float(f)) if *f >= 0.0 => Some(*f as usize),
            _ => None,
        }
    }
}

impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Neuron::Symbol(name) => write!(f, "{}", name),
            Neuron::Variable(name) => write!(f, "${}", name),
            Neuron::Expression(children) => {
                write!(f, "(")?;
                for (i, child) in children.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", child)?;
                }
                write!(f, ")")
            }
            Neuron::Value(val) => write!(f, "{}", val),
        }
    }
}

impl fmt::Display for QorValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QorValue::Int(n) => write!(f, "{}", n),
            QorValue::Float(n) => write!(f, "{}", n),
            QorValue::Str(s) => write!(f, "\"{}\"", s),
            QorValue::Bool(b) => write!(f, "{}", b),
        }
    }
}

impl fmt::Display for StoredNeuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.inferred {
            write!(f, "{} {} (inferred)", self.neuron, self.tv)
        } else {
            write!(f, "{} {}", self.neuron, self.tv)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_display() {
        let n = Neuron::expression(vec![
            Neuron::symbol("bird"),
            Neuron::symbol("tweety"),
        ]);
        assert_eq!(n.to_string(), "(bird tweety)");
    }

    #[test]
    fn test_neuron_with_variable() {
        let n = Neuron::expression(vec![
            Neuron::symbol("flies"),
            Neuron::variable("x"),
        ]);
        assert_eq!(n.to_string(), "(flies $x)");
    }

    #[test]
    fn test_neuron_with_string_value() {
        let n = Neuron::expression(vec![
            Neuron::symbol("claim"),
            Neuron::str_val("eagles can fly"),
        ]);
        assert_eq!(n.to_string(), "(claim \"eagles can fly\")");
    }

    #[test]
    fn test_neuron_with_number() {
        let n = Neuron::expression(vec![
            Neuron::symbol("price"),
            Neuron::symbol("bitcoin"),
            Neuron::int_val(95432),
        ]);
        assert_eq!(n.to_string(), "(price bitcoin 95432)");
    }

    #[test]
    fn test_stored_neuron_display() {
        let sn = StoredNeuron {
            neuron: Neuron::expression(vec![
                Neuron::symbol("bird"),
                Neuron::symbol("tweety"),
            ]),
            tv: TruthValue::new(0.99, 0.90),
            timestamp: None,
            decay_rate: None,
            inferred: false,
        };
        assert_eq!(sn.to_string(), "(bird tweety) <0.99, 0.90>");

        let inferred = StoredNeuron {
            neuron: Neuron::expression(vec![
                Neuron::symbol("flies"),
                Neuron::symbol("tweety"),
            ]),
            tv: TruthValue::new(0.94, 0.73),
            timestamp: None,
            decay_rate: None,
            inferred: true,
        };
        assert_eq!(inferred.to_string(), "(flies tweety) <0.94, 0.73> (inferred)");
    }
}
