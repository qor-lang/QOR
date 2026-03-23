// ── Key-Value → QOR Facts (nom parsing) ────────────────────────────────
//
// Converts key-value formatted data into QOR facts.
// Uses nom parser combinators for extensible structured parsing.
// Supports: key=value, key: value, key\tvalue

use qor_core::neuron::{Neuron, Statement};
use crate::sanitize::{sanitize_symbol, infer_value, ingested_tv};

use nom::{
    IResult,
    branch::alt,
    bytes::complete::{take_while1, take_till1},
    character::complete::{char as nom_char, space0, space1},
    sequence::tuple,
};

/// A parsed key-value pair.
struct KvPair {
    key: String,
    value: String,
}

/// Convert key-value formatted data into QOR facts.
///
/// Each pair becomes: (key value) <0.90>
/// Comments (# or //) and empty lines are skipped.
pub fn from_kv(data: &str) -> Result<Vec<Statement>, String> {
    let mut facts = Vec::new();

    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") || line.starts_with(';') {
            continue;
        }

        if let Ok((_, pair)) = parse_kv_line(line) {
            let key = sanitize_symbol(&pair.key);
            let value = infer_value(&pair.value);

            facts.push(Statement::Fact {
                neuron: Neuron::expression(vec![Neuron::symbol(&key), value]),
                tv: Some(ingested_tv()),
                decay: None,
            });
        }
        // Lines that don't parse as KV are silently skipped
    }

    if facts.is_empty() {
        return Err("No key-value pairs found".to_string());
    }

    Ok(facts)
}

/// Parse a single KV line using nom combinators.
fn parse_kv_line(input: &str) -> IResult<&str, KvPair> {
    alt((parse_equals, parse_colon, parse_tab))(input)
}

/// Parse `key = value` or `key=value`
fn parse_equals(input: &str) -> IResult<&str, KvPair> {
    let (rest, (key, _, _, _, value)) = tuple((
        take_while1(|c: char| c != '=' && c != '\n' && c != '\r'),
        space0,
        nom_char('='),
        space0,
        take_while1(|c: char| c != '\n' && c != '\r'),
    ))(input)?;

    Ok((
        rest,
        KvPair {
            key: key.trim().to_string(),
            value: value.trim().to_string(),
        },
    ))
}

/// Parse `key: value` (requires space after colon to avoid URL confusion)
fn parse_colon(input: &str) -> IResult<&str, KvPair> {
    let (rest, (key, _, _, _, value)) = tuple((
        take_while1(|c: char| c != ':' && c != '\n' && c != '\r'),
        space0,
        nom_char(':'),
        space1, // require space after colon
        take_while1(|c: char| c != '\n' && c != '\r'),
    ))(input)?;

    Ok((
        rest,
        KvPair {
            key: key.trim().to_string(),
            value: value.trim().to_string(),
        },
    ))
}

/// Parse `key\tvalue`
fn parse_tab(input: &str) -> IResult<&str, KvPair> {
    let (rest, (key, _, value)) = tuple((
        take_till1(|c: char| c == '\t'),
        nom_char('\t'),
        take_while1(|c: char| c != '\n' && c != '\r'),
    ))(input)?;

    Ok((
        rest,
        KvPair {
            key: key.trim().to_string(),
            value: value.trim().to_string(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_equals() {
        let facts = from_kv("name=alice\nage=30").unwrap();
        assert_eq!(facts.len(), 2);
        let all = facts_to_strings(&facts);
        assert!(all.contains("(name alice)"));
        assert!(all.contains("(age 30)"));
    }

    #[test]
    fn test_kv_equals_spaces() {
        let facts = from_kv("name = alice\nage = 30").unwrap();
        assert_eq!(facts.len(), 2);
    }

    #[test]
    fn test_kv_colon() {
        let facts = from_kv("name: alice\nage: 30").unwrap();
        assert_eq!(facts.len(), 2);
        let all = facts_to_strings(&facts);
        assert!(all.contains("(name alice)"));
    }

    #[test]
    fn test_kv_tab() {
        let facts = from_kv("name\talice\nage\t30").unwrap();
        assert_eq!(facts.len(), 2);
    }

    #[test]
    fn test_kv_comments_skipped() {
        let facts = from_kv("# this is a comment\nname=alice\n// another comment\nage=30").unwrap();
        assert_eq!(facts.len(), 2);
    }

    #[test]
    fn test_kv_semicolon_comments() {
        let facts = from_kv("; comment\nname=alice").unwrap();
        assert_eq!(facts.len(), 1);
    }

    #[test]
    fn test_kv_type_inference() {
        let facts = from_kv("count=42\nscore=95.5\nactive=true").unwrap();
        assert_eq!(facts.len(), 3);
    }

    #[test]
    fn test_kv_empty_returns_error() {
        assert!(from_kv("").is_err());
        assert!(from_kv("# only comments\n// more comments").is_err());
    }

    #[test]
    fn test_kv_mixed_formats() {
        let facts = from_kv("name=alice\nage: 30\ncity\tnyc").unwrap();
        assert_eq!(facts.len(), 3);
    }

    fn facts_to_strings(facts: &[Statement]) -> String {
        facts
            .iter()
            .map(|f| match f {
                Statement::Fact { neuron, .. } => neuron.to_string(),
                _ => String::new(),
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }
}
