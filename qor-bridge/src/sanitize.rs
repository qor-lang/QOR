// ── Sanitize + Value Inference ──────────────────────────────────────────
//
// Shared helpers for all ingestion parsers.
// Converts raw external data into valid QOR symbols and typed neurons.

use qor_core::neuron::{Neuron, QorValue};
use qor_core::truth_value::TruthValue;

/// Default truth value strength for ingested (external) data.
/// Lower than user-asserted facts (0.99) — external data is less trusted.
pub const INGESTED_STRENGTH: f64 = 0.90;

/// Build the default truth value for ingested facts.
pub fn ingested_tv() -> TruthValue {
    TruthValue::from_strength(INGESTED_STRENGTH)
}

/// Sanitize a raw string into a valid QOR symbol name.
///
/// Rules:
/// 1. Lowercase everything
/// 2. Replace spaces, underscores, dots, and special chars with hyphens
/// 3. Collapse consecutive hyphens into one
/// 4. Strip leading/trailing hyphens
/// 5. Prefix with "x-" if starts with a digit
/// 6. Return "unknown" if empty after sanitization
pub fn sanitize_symbol(raw: &str) -> String {
    let lowered = raw.trim().to_lowercase();
    let mut result = String::with_capacity(lowered.len());
    let mut prev_hyphen = false;

    for ch in lowered.chars() {
        if ch.is_ascii_alphanumeric() {
            prev_hyphen = false;
            result.push(ch);
        } else if ch == '-' || ch == '_' || ch == ' ' || ch == '.' {
            if !prev_hyphen && !result.is_empty() {
                result.push('-');
                prev_hyphen = true;
            }
        }
        // All other characters are dropped
    }

    // Trim trailing hyphen
    while result.ends_with('-') {
        result.pop();
    }

    if result.is_empty() {
        return "unknown".to_string();
    }

    // QOR symbols must start with a letter
    if result.chars().next().map_or(false, |c| c.is_ascii_digit()) {
        result = format!("x-{}", result);
    }

    result
}

/// Infer a Neuron from a raw string value.
///
/// Try in order: bool → int → float → symbol → string
pub fn infer_value(raw: &str) -> Neuron {
    let trimmed = raw.trim();

    // Boolean
    match trimmed.to_lowercase().as_str() {
        "true" | "yes" => return Neuron::Value(QorValue::Bool(true)),
        "false" | "no" => return Neuron::Value(QorValue::Bool(false)),
        _ => {}
    }

    // Integer
    if let Ok(n) = trimmed.parse::<i64>() {
        return Neuron::Value(QorValue::Int(n));
    }

    // Float
    if trimmed.contains('.') {
        if let Ok(f) = trimmed.parse::<f64>() {
            return Neuron::Value(QorValue::Float(f));
        }
    }

    // Null/empty
    if trimmed.is_empty() || trimmed == "null" || trimmed == "nil" {
        return Neuron::Symbol("null".to_string());
    }

    // Symbol-like (no spaces, starts with letter, alphanumeric + hyphens)
    if is_symbol_like(trimmed) {
        return Neuron::Symbol(sanitize_symbol(trimmed));
    }

    // Fallback: string value
    Neuron::Value(QorValue::Str(trimmed.to_string()))
}

fn is_symbol_like(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let first = s.chars().next().unwrap();
    if !first.is_ascii_alphabetic() {
        return false;
    }
    s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Helper: create a Statement::Fact with the default ingested truth value.
pub fn make_fact(parts: Vec<Neuron>) -> qor_core::neuron::Statement {
    qor_core::neuron::Statement::Fact {
        neuron: Neuron::Expression(parts),
        tv: Some(ingested_tv()),
        decay: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_basic() {
        assert_eq!(sanitize_symbol("hello"), "hello");
    }

    #[test]
    fn test_sanitize_spaces() {
        assert_eq!(sanitize_symbol("First Name"), "first-name");
    }

    #[test]
    fn test_sanitize_underscores() {
        assert_eq!(sanitize_symbol("user_id"), "user-id");
    }

    #[test]
    fn test_sanitize_special_chars() {
        assert_eq!(sanitize_symbol("Price ($)"), "price");
    }

    #[test]
    fn test_sanitize_leading_digit() {
        assert_eq!(sanitize_symbol("123abc"), "x-123abc");
    }

    #[test]
    fn test_sanitize_empty() {
        assert_eq!(sanitize_symbol(""), "unknown");
        assert_eq!(sanitize_symbol("   "), "unknown");
        assert_eq!(sanitize_symbol("$$$"), "unknown");
    }

    #[test]
    fn test_sanitize_consecutive_hyphens() {
        assert_eq!(sanitize_symbol("a__b"), "a-b");
        assert_eq!(sanitize_symbol("a - - b"), "a-b");
    }

    #[test]
    fn test_sanitize_dots() {
        assert_eq!(sanitize_symbol("address.city"), "address-city");
    }

    #[test]
    fn test_infer_int() {
        assert_eq!(infer_value("42"), Neuron::Value(QorValue::Int(42)));
        assert_eq!(infer_value("-7"), Neuron::Value(QorValue::Int(-7)));
    }

    #[test]
    fn test_infer_float() {
        assert_eq!(infer_value("3.14"), Neuron::Value(QorValue::Float(3.14)));
    }

    #[test]
    fn test_infer_bool() {
        assert_eq!(infer_value("true"), Neuron::Value(QorValue::Bool(true)));
        assert_eq!(infer_value("false"), Neuron::Value(QorValue::Bool(false)));
        assert_eq!(infer_value("yes"), Neuron::Value(QorValue::Bool(true)));
    }

    #[test]
    fn test_infer_symbol() {
        assert_eq!(infer_value("alice"), Neuron::Symbol("alice".to_string()));
        assert_eq!(infer_value("new-york"), Neuron::Symbol("new-york".to_string()));
    }

    #[test]
    fn test_infer_string_fallback() {
        match infer_value("hello world") {
            Neuron::Value(QorValue::Str(s)) => assert_eq!(s, "hello world"),
            other => panic!("expected string, got {:?}", other),
        }
    }

    #[test]
    fn test_infer_null() {
        assert_eq!(infer_value("null"), Neuron::Symbol("null".to_string()));
    }
}
