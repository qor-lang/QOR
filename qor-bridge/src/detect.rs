// ── Format Auto-Detection ──────────────────────────────────────────────
//
// Detects the format of raw input data.
// Priority: JSON > CSV > KeyValue > Text (most specific to least).

/// Detected data format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataFormat {
    Json,
    Csv,
    KeyValue,
    Text,
}

impl std::fmt::Display for DataFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataFormat::Json => write!(f, "JSON"),
            DataFormat::Csv => write!(f, "CSV"),
            DataFormat::KeyValue => write!(f, "Key-Value"),
            DataFormat::Text => write!(f, "Text"),
        }
    }
}

/// Auto-detect the format of raw input data.
pub fn detect_format(data: &str) -> DataFormat {
    let trimmed = data.trim();

    if trimmed.is_empty() {
        return DataFormat::Text;
    }

    // 1. JSON: starts with { or [
    let first = trimmed.chars().next().unwrap();
    if first == '{' || first == '[' {
        return DataFormat::Json;
    }

    // 2. CSV: multiple lines with consistent comma counts
    if looks_like_csv(trimmed) {
        return DataFormat::Csv;
    }

    // 3. KV: most non-empty lines match key=value or key: value
    if looks_like_kv(trimmed) {
        return DataFormat::KeyValue;
    }

    // 4. Fallback: text
    DataFormat::Text
}

fn looks_like_csv(data: &str) -> bool {
    let lines: Vec<&str> = data
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();

    if lines.len() < 2 {
        return false;
    }

    let header_commas = lines[0].matches(',').count();
    if header_commas == 0 {
        return false;
    }

    // At least 70% of data lines must have the same comma count
    let matching = lines[1..]
        .iter()
        .filter(|l| l.matches(',').count() == header_commas)
        .count();

    matching as f64 / (lines.len() - 1) as f64 > 0.7
}

fn looks_like_kv(data: &str) -> bool {
    let lines: Vec<&str> = data
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("//"))
        .collect();

    if lines.is_empty() {
        return false;
    }

    // Count lines that match key=value or key: value patterns
    let kv_count = lines
        .iter()
        .filter(|l| {
            // key=value or key = value
            if let Some(eq_pos) = l.find('=') {
                let key = l[..eq_pos].trim();
                return !key.is_empty()
                    && key
                        .chars()
                        .next()
                        .map_or(false, |c| c.is_ascii_alphabetic());
            }
            // key: value (with space after colon)
            if let Some(col_pos) = l.find(": ") {
                let key = l[..col_pos].trim();
                return !key.is_empty()
                    && key
                        .chars()
                        .next()
                        .map_or(false, |c| c.is_ascii_alphabetic());
            }
            false
        })
        .count();

    kv_count as f64 / lines.len() as f64 > 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_json_object() {
        assert_eq!(detect_format(r#"{"name": "alice"}"#), DataFormat::Json);
    }

    #[test]
    fn test_detect_json_array() {
        assert_eq!(detect_format("[1, 2, 3]"), DataFormat::Json);
    }

    #[test]
    fn test_detect_json_whitespace() {
        assert_eq!(detect_format("  \n  {\"a\": 1}  "), DataFormat::Json);
    }

    #[test]
    fn test_detect_csv() {
        assert_eq!(
            detect_format("name,age,city\nalice,30,nyc\nbob,25,sf"),
            DataFormat::Csv
        );
    }

    #[test]
    fn test_detect_csv_needs_two_lines() {
        // Single line with commas is NOT csv
        assert_ne!(detect_format("a,b,c"), DataFormat::Csv);
    }

    #[test]
    fn test_detect_kv_equals() {
        assert_eq!(
            detect_format("name=alice\nage=30\ncity=nyc"),
            DataFormat::KeyValue
        );
    }

    #[test]
    fn test_detect_kv_colon() {
        assert_eq!(
            detect_format("name: alice\nage: 30\ncity: nyc"),
            DataFormat::KeyValue
        );
    }

    #[test]
    fn test_detect_text_fallback() {
        assert_eq!(detect_format("birds can fly"), DataFormat::Text);
    }

    #[test]
    fn test_detect_empty() {
        assert_eq!(detect_format(""), DataFormat::Text);
    }

    #[test]
    fn test_detect_multiline_text() {
        assert_eq!(
            detect_format("Alice is a doctor.\nBob has a cat."),
            DataFormat::Text
        );
    }
}
