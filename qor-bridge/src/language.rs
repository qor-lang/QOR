// ── QOR Language Understanding ───────────────────────────────────────────
//
// QOR learns language by reasoning about grammar rules and vocabulary
// as facts — no LLM needed. Forward chaining detects intent from input
// tokens and composes responses from brain knowledge.
//
// Flow:
//   User text → tokenize → (input N word) facts
//   Grammar rules fire → (intent ...) facts
//   Response rules fire → (response ...) facts
//   format_response → natural language output

use std::collections::HashMap;
use std::path::Path;

use qor_core::neuron::{Neuron, QorValue, Statement, StoredNeuron};
use qor_core::parser;
use qor_core::truth_value::TruthValue;

// ── Brain Context ────────────────────────────────────────────────────────
//
// Extracted from session facts to enrich chat responses with statistical
// context, co-occurrences, and personality. Scanned once per turn.

/// Brain knowledge context for enriching chat responses.
#[derive(Debug, Default)]
pub struct BrainContext {
    /// pattern-rate: name -> (count, total)
    pub pattern_rates: HashMap<String, (i64, i64)>,
    /// Strong co-occurrences (>50%): pattern -> [(co_pattern, ratio)]
    pub strong_cooccur: HashMap<String, Vec<(String, f64)>>,
    /// threshold: (indicator, condition) -> value
    pub thresholds: HashMap<(String, String), i64>,
    /// Personality text variants: key -> [text options]
    pub personality: HashMap<String, Vec<String>>,
}

impl BrainContext {
    /// Pick a personality text variant for a given key.
    /// Rotates through variants using `seed` (e.g. fact_count).
    /// Falls back to `default` if no personality facts exist.
    pub fn personality_text(&self, key: &str, seed: usize, default: &str) -> String {
        match self.personality.get(key) {
            Some(variants) if !variants.is_empty() => {
                variants[seed % variants.len()].clone()
            }
            _ => default.to_string(),
        }
    }
}

/// Extract brain context from stored facts for response formatting.
pub fn extract_brain_context(facts: &[StoredNeuron]) -> BrainContext {
    let mut ctx = BrainContext::default();

    for sn in facts {
        if let Neuron::Expression(parts) = &sn.neuron {
            if parts.is_empty() {
                continue;
            }
            if let Neuron::Symbol(pred) = &parts[0] {
                match pred.as_str() {
                    "pattern-rate" if parts.len() == 4 => {
                        let name = parts[1].to_string();
                        let count = parts[2].as_f64().unwrap_or(0.0) as i64;
                        let total = parts[3].as_f64().unwrap_or(0.0) as i64;
                        ctx.pattern_rates.insert(name, (count, total));
                    }
                    "co-occur" if parts.len() == 5 => {
                        let a = parts[1].to_string();
                        let b = parts[2].to_string();
                        let count = parts[3].as_f64().unwrap_or(0.0);
                        let base = parts[4].as_f64().unwrap_or(1.0);
                        let ratio = if base > 0.0 { count / base } else { 0.0 };
                        if ratio > 0.50 {
                            ctx.strong_cooccur
                                .entry(a)
                                .or_default()
                                .push((b, ratio));
                        }
                    }
                    "threshold" if parts.len() == 4 => {
                        let indicator = parts[1].to_string();
                        let condition = parts[2].to_string();
                        let value = parts[3].as_f64().unwrap_or(0.0) as i64;
                        ctx.thresholds.insert((indicator, condition), value);
                    }
                    "personality" if parts.len() >= 3 => {
                        let key = parts[1].to_string();
                        let text = match &parts[2] {
                            Neuron::Value(QorValue::Str(s)) => s.clone(),
                            Neuron::Symbol(s) => s.replace('-', " "),
                            other => other.to_string(),
                        };
                        ctx.personality.entry(key).or_default().push(text);
                    }
                    _ => {}
                }
            }
        }
    }

    ctx
}

/// Load language knowledge from a language subfolder (e.g. `language/en/`).
///
/// Reads all `.qor` files from the given directory (vocabulary, grammar,
/// responses, dictionary). Falls back to built-in bootstrap if missing.
pub fn load_language_dir(dir: &Path) -> Vec<Statement> {
    let mut stmts = Vec::new();

    if dir.exists() {
        stmts = load_qor_files(dir);
    }

    // If folder was empty or missing, use built-in bootstrap
    if stmts.is_empty() {
        stmts = language_knowledge_bootstrap();
    }

    stmts
}

/// List available languages in the language/ root folder.
///
/// Returns subfolder names like `["en", "es"]`.
pub fn available_languages(language_root: &Path) -> Vec<String> {
    let mut langs = Vec::new();
    if let Ok(entries) = std::fs::read_dir(language_root) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_dir() {
                // Check it has at least one .qor file
                let has_qor = std::fs::read_dir(&path)
                    .map(|rd| rd.filter_map(|e| e.ok())
                        .any(|e| e.path().extension().map_or(false, |ext| ext == "qor")))
                    .unwrap_or(false);
                if has_qor {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        langs.push(name.to_string());
                    }
                }
            }
        }
    }
    langs.sort();
    langs
}

/// Load all .qor files from a directory.
fn load_qor_files(dir: &Path) -> Vec<Statement> {
    let mut stmts = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "qor"))
            .collect();
        files.sort_by_key(|e| e.path());

        for entry in files {
            if let Ok(source) = std::fs::read_to_string(entry.path()) {
                match parser::parse(&source) {
                    Ok(parsed) => stmts.extend(parsed),
                    Err(e) => {
                        eprintln!("  warning: {}: {}", entry.path().display(), e);
                    }
                }
            }
        }
    }
    stmts
}

/// Built-in bootstrap language knowledge (fallback when no language/ folder).
pub fn language_knowledge_bootstrap() -> Vec<Statement> {
    let source = r#"
;; ── Bootstrap Vocabulary ─────────────────────────────────────
(vocab hello greeting) <0.99, 0.99>
(vocab hi greeting) <0.99, 0.99>
(vocab bye farewell) <0.99, 0.99>
(vocab what question-word) <0.99, 0.99>
(vocab why question-word) <0.99, 0.99>
(vocab how question-word) <0.99, 0.99>
(vocab is copula) <0.99, 0.99>
(vocab are copula) <0.99, 0.99>
(vocab show command-word) <0.99, 0.99>
(vocab list command-word) <0.99, 0.99>
(vocab help command-word) <0.99, 0.99>
(vocab tell command-word) <0.99, 0.99>
(vocab about preposition) <0.99, 0.99>
(vocab the article) <0.99, 0.99>
(vocab a article) <0.99, 0.99>
(vocab me pronoun) <0.99, 0.99>
(vocab many quantifier) <0.99, 0.99>

;; ── Grammar Rules ────────────────────────────────────────────
(intent greet) if (input $i $w) (vocab $w greeting) <0.95, 0.90>
(intent farewell) if (input $i $w) (vocab $w farewell) <0.95, 0.90>
(intent help) if (input $i help) <0.95, 0.90>
(intent-define $topic) if (input $i what) (input $j $cop) (vocab $cop copula) (input $k $topic) (> $k $j) not (vocab $topic article) not (vocab $topic preposition) not (vocab $topic question-word) not (vocab $topic copula) <0.90, 0.85>
(intent-list $topic) if (input $i $cmd) (vocab $cmd command-word) (input $j $topic) (> $j $i) not (vocab $topic article) not (vocab $topic preposition) not (vocab $topic pronoun) not (vocab $topic command-word) <0.90, 0.85>
(intent curious) if (input $i $w) (vocab $w question-word) <0.70, 0.60>

;; ── Response Rules ───────────────────────────────────────────
(response greet hello) if (intent greet) <0.95, 0.90>
(response help commands) if (intent help) <0.95, 0.90>
(response define $topic $desc) if (intent-define $topic) (meaning $topic $desc) <0.90, 0.85>
(response farewell goodbye) if (intent farewell) <0.95, 0.90>
(response list-pattern $name $count) if (intent-list patterns) (pattern-rate $name $count $total) <0.90, 0.85>
(response list-indicator $name $desc) if (intent-list indicators) (indicator $name $desc) <0.90, 0.85>
(response curious unknown) if (intent curious) not (intent-define $a1) not (intent-explain $a2 $a3) not (intent-count $a4) not (intent-list $a5) not (intent greet) not (intent farewell) not (intent help) <0.70, 0.60>
"#;

    match parser::parse(source) {
        Ok(stmts) => stmts,
        Err(e) => {
            eprintln!("bootstrap parse error: {}", e);
            Vec::new()
        }
    }
}

/// Save a learned fact to the dictionary file in language/.
pub fn save_to_dictionary(dir: &Path, fact_line: &str) {
    let dict_path = dir.join("dictionary.qor");
    let line = format!("{}\n", fact_line);
    if let Err(e) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&dict_path)
        .and_then(|mut f| {
            use std::io::Write;
            f.write_all(line.as_bytes())
        })
    {
        eprintln!("  warning: could not save to dictionary: {}", e);
    }
}

/// Tokenize user input into QOR (input N word) facts.
///
/// Lowercases, splits on whitespace/punctuation, generates positional facts.
pub fn tokenize(input: &str) -> Vec<Statement> {
    let lower = input.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| c.is_whitespace() || c == ',' || c == '!' || c == '.' || c == ';')
        .map(|w| w.trim_matches(|c: char| c == '?' || c == '\'' || c == '"'))
        .filter(|w| !w.is_empty())
        .collect();

    words
        .iter()
        .enumerate()
        .map(|(i, word)| Statement::Fact {
            neuron: Neuron::Expression(vec![
                Neuron::Symbol("input".into()),
                Neuron::Value(QorValue::Int(i as i64)),
                Neuron::Symbol(word.to_string()),
            ]),
            tv: Some(TruthValue::new(0.99, 0.99)),
            decay: None,
        })
        .collect()
}

/// Detect teaching patterns in user input.
///
/// Returns `Some((topic, description))` for patterns like:
/// - "X means Y"
/// - "X is Y" (when not a question)
/// - "remember X is Y"
/// - "define X as Y"
pub fn parse_teach_pattern(input: &str) -> Option<(String, String)> {
    let lower = input.to_lowercase();
    let lower = lower.trim();

    // "X means Y"
    if let Some(idx) = lower.find(" means ") {
        let topic = sanitize_topic(&lower[..idx]);
        let desc = sanitize_desc(&lower[idx + 7..]);
        if !topic.is_empty() && !desc.is_empty() {
            return Some((topic, desc));
        }
    }

    // "define X as Y"
    if lower.starts_with("define ") {
        let rest = &lower[7..];
        if let Some(idx) = rest.find(" as ") {
            let topic = sanitize_topic(&rest[..idx]);
            let desc = sanitize_desc(&rest[idx + 4..]);
            if !topic.is_empty() && !desc.is_empty() {
                return Some((topic, desc));
            }
        }
    }

    // "remember X is Y" / "remember that X is Y"
    if lower.starts_with("remember ") {
        let rest = lower[9..].trim_start_matches("that ");
        if let Some(idx) = rest.find(" is ") {
            let topic = sanitize_topic(&rest[..idx]);
            let desc = sanitize_desc(&rest[idx + 4..]);
            if !topic.is_empty() && !desc.is_empty() {
                return Some((topic, desc));
            }
        }
    }

    None
}

fn sanitize_topic(s: &str) -> String {
    s.trim()
        .trim_matches(|c: char| !c.is_alphanumeric() && c != '-')
        .replace(' ', "-")
}

fn sanitize_desc(s: &str) -> String {
    s.trim()
        .trim_end_matches(|c: char| c == '.' || c == '!' || c == '?')
        .trim()
        .replace(' ', "-")
}

/// Format response facts into natural language output.
///
/// When `ctx` is `Some`, enriches responses with brain knowledge
/// (pattern rarity, co-occurrences, thresholds) and personality variants.
/// When `ctx` is `None`, produces the same output as the original implementation.
pub fn format_response(responses: &[Vec<String>], ctx: Option<&BrainContext>) -> String {
    if responses.is_empty() {
        return "I don't have enough knowledge to answer that. Try 'help' to see what I can do.".into();
    }

    // "curious" is a low-confidence fallback — suppress it when any specific
    // response was produced (define, explain, list-*, greet, help, etc.)
    let has_specific = responses.iter().any(|p| {
        !p.is_empty() && !matches!(p[0].as_str(), "curious" | "affirm")
    });

    let mut lines = Vec::new();
    for parts in responses {
        if parts.is_empty() {
            continue;
        }
        match parts[0].as_str() {
            "greet" => {
                let text = ctx
                    .map(|c| c.personality_text("greet", lines.len(), ""))
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "Hello! I am QOR. Ask me about what I know.".into());
                lines.push(text);
            }
            "help" => {
                let intro = ctx
                    .map(|c| c.personality_text("help-intro", lines.len(), ""))
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "I can answer questions about my knowledge. Try:".into());
                lines.push(intro);
                lines.push("  what is <topic>      — look up what I know about something".into());
                lines.push("  why is <X> <Y>       — explain a relationship".into());
                lines.push("  show <topic>         — list what I know".into());
                lines.push("  <X> means <Y>        — teach me new vocabulary".into());
                lines.push("  define <X> as <Y>    — teach me a definition".into());
                lines.push("  help                 — show this help".into());
                lines.push("  quit                 — exit chat".into());
                lines.push("You can also type QOR directly: (fact) or ? (query $x)".into());
            }
            "define" if parts.len() >= 3 => {
                let topic = &parts[1];
                let desc = parts[2].replace('-', " ");
                let mut line = format!("{}: {}", topic, desc);

                if let Some(c) = ctx {
                    // Rarity/frequency context
                    if let Some(&(count, total)) = c.pattern_rates.get(topic.as_str()) {
                        if total > 0 {
                            let pct = (count as f64 / total as f64) * 100.0;
                            let rarity = if pct < 5.0 {
                                "rare"
                            } else if pct > 40.0 {
                                "common"
                            } else {
                                "seen"
                            };
                            line.push_str(&format!(
                                " ({} — {:.1}% of {} data points)", rarity, pct, total
                            ));
                        }
                    }

                    // Threshold context
                    for ((ind, cond), val) in &c.thresholds {
                        if cond == topic {
                            line.push_str(&format!(". Triggered when {} > {}", ind, val));
                        }
                    }

                    // Strong co-occurrences
                    if let Some(cooccurs) = c.strong_cooccur.get(topic.as_str()) {
                        if !cooccurs.is_empty() {
                            let names: Vec<String> = cooccurs
                                .iter()
                                .take(3)
                                .map(|(name, ratio)| format!("{} ({:.0}%)", name, ratio * 100.0))
                                .collect();
                            line.push_str(&format!(". Often appears with: {}", names.join(", ")));
                        }
                    }
                }

                lines.push(line);
            }
            "explain" if parts.len() >= 3 => {
                let topic = &parts[1];
                let state = &parts[2];
                lines.push(format!(
                    "{} is {} — this was derived from the rules I've learned.", topic, state
                ));
            }
            "list-pattern" if parts.len() >= 3 => {
                let name = &parts[1];
                let count = &parts[2];
                if let Some(c) = ctx {
                    if let Some(&(cnt, total)) = c.pattern_rates.get(name.as_str()) {
                        if total > 0 {
                            let pct = (cnt as f64 / total as f64) * 100.0;
                            lines.push(format!("  {}: {} / {} ({:.1}%)", name, count, total, pct));
                        } else {
                            lines.push(format!("  {}: {} occurrences", name, count));
                        }
                    } else {
                        lines.push(format!("  {}: {} occurrences", name, count));
                    }
                } else {
                    lines.push(format!("  {}: {} occurrences", name, count));
                }
            }
            "list-indicator" if parts.len() >= 3 => {
                let name = &parts[1];
                let desc = parts[2].replace('-', " ");
                lines.push(format!("  {}: {}", name, desc));
            }
            "list-meaning" if parts.len() >= 3 => {
                let name = &parts[1];
                let desc = parts[2].replace('-', " ");
                lines.push(format!("  {}: {}", name, desc));
            }
            "count" if parts.len() >= 3 => {
                let topic = &parts[1];
                let n = &parts[2];
                lines.push(format!("There are {} {} facts.", n, topic));
            }
            "farewell" => {
                let text = ctx
                    .map(|c| c.personality_text("farewell", lines.len(), ""))
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "Goodbye! Keep reasoning.".into());
                lines.push(text);
            }
            "curious" if !has_specific => {
                let text = ctx
                    .map(|c| c.personality_text("curious", lines.len(), ""))
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| {
                        "Hmm, I'm not sure about that. Try 'what is <topic>' or 'help'.".into()
                    });
                lines.push(text);
            }
            "curious" => {} // suppressed — a better response exists
            _ => {
                let text = parts.join(" ");
                lines.push(text);
            }
        }
    }

    // Deduplicate (multiple rules may fire the same response type)
    lines.dedup();
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("What is overbought?");
        assert_eq!(tokens.len(), 3);

        // Check first token: (input 0 what)
        if let Statement::Fact { neuron, .. } = &tokens[0] {
            if let Neuron::Expression(parts) = neuron {
                assert_eq!(parts[0], Neuron::Symbol("input".into()));
                assert_eq!(parts[1], Neuron::Value(QorValue::Int(0)));
                assert_eq!(parts[2], Neuron::Symbol("what".into()));
            } else {
                panic!("expected Expression");
            }
        }

        // Check third token is lowercased
        if let Statement::Fact { neuron, .. } = &tokens[2] {
            if let Neuron::Expression(parts) = neuron {
                assert_eq!(parts[2], Neuron::Symbol("overbought".into()));
            }
        }
    }

    #[test]
    fn test_tokenize_strips_punctuation() {
        let tokens = tokenize("Hello, world! How are you?");
        assert_eq!(tokens.len(), 5);

        // "Hello," should become "hello"
        if let Statement::Fact { neuron, .. } = &tokens[0] {
            if let Neuron::Expression(parts) = neuron {
                assert_eq!(parts[2], Neuron::Symbol("hello".into()));
            }
        }
    }

    #[test]
    fn test_bootstrap_parses() {
        let stmts = language_knowledge_bootstrap();
        assert!(!stmts.is_empty(), "should produce statements");

        // Should have vocab facts
        let vocab_count = stmts.iter().filter(|s| {
            if let Statement::Fact { neuron, .. } = s {
                if let Neuron::Expression(parts) = neuron {
                    return parts.first() == Some(&Neuron::Symbol("vocab".into()));
                }
            }
            false
        }).count();
        assert!(vocab_count >= 10, "should have at least 10 vocab facts, got {}", vocab_count);

        // Should have rules
        let rule_count = stmts.iter().filter(|s| matches!(s, Statement::Rule { .. })).count();
        assert!(rule_count >= 5, "should have at least 5 rules, got {}", rule_count);
    }

    #[test]
    fn test_load_language_dir() {
        // Test loading from the actual language/ folder
        let lang_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .join("language")
            .join("en");
        let stmts = load_language_dir(&lang_dir);
        assert!(!stmts.is_empty(), "should load from language/ folder");

        // Should have more vocab than bootstrap (language/ has richer files)
        let vocab_count = stmts.iter().filter(|s| {
            if let Statement::Fact { neuron, .. } = s {
                if let Neuron::Expression(parts) = neuron {
                    return parts.first() == Some(&Neuron::Symbol("vocab".into()));
                }
            }
            false
        }).count();
        assert!(vocab_count >= 40, "language/ should have at least 40 vocab, got {}", vocab_count);

        // Should have language identity
        let has_lang = stmts.iter().any(|s| {
            if let Statement::Fact { neuron, .. } = s {
                if let Neuron::Expression(parts) = neuron {
                    return parts.len() == 2
                        && parts[0] == Neuron::Symbol("language".into())
                        && parts[1] == Neuron::Symbol("english".into());
                }
            }
            false
        });
        assert!(has_lang, "should have (language english) fact");
    }

    #[test]
    fn test_format_response_types() {
        assert_eq!(
            format_response(&[vec!["greet".into(), "hello".into()]], None),
            "Hello! I am QOR. Ask me about what I know."
        );

        assert!(format_response(&[vec!["help".into(), "commands".into()]], None).contains("what is <topic>"));

        assert_eq!(
            format_response(&[vec!["define".into(), "overbought".into(), "buying-pressure-extreme".into()]], None),
            "overbought: buying pressure extreme"
        );

        assert!(format_response(&[], None).contains("don't have enough"));

        assert_eq!(
            format_response(&[vec!["farewell".into(), "goodbye".into()]], None),
            "Goodbye! Keep reasoning."
        );
    }

    #[test]
    fn test_extract_brain_context() {
        let facts = vec![
            StoredNeuron {
                neuron: Neuron::expression(vec![
                    Neuron::symbol("pattern-rate"),
                    Neuron::symbol("overbought"),
                    Neuron::int_val(395),
                    Neuron::int_val(10000),
                ]),
                tv: TruthValue::new(0.04, 0.95),
                timestamp: None, decay_rate: None, inferred: false,
            },
            StoredNeuron {
                neuron: Neuron::expression(vec![
                    Neuron::symbol("co-occur"),
                    Neuron::symbol("overbought"),
                    Neuron::symbol("trending"),
                    Neuron::int_val(316),
                    Neuron::int_val(395),
                ]),
                tv: TruthValue::new(0.80, 0.95),
                timestamp: None, decay_rate: None, inferred: false,
            },
            StoredNeuron {
                neuron: Neuron::expression(vec![
                    Neuron::symbol("co-occur"),
                    Neuron::symbol("overbought"),
                    Neuron::symbol("range-bound"),
                    Neuron::int_val(30),
                    Neuron::int_val(395),
                ]),
                tv: TruthValue::new(0.08, 0.95),
                timestamp: None, decay_rate: None, inferred: false,
            },
            StoredNeuron {
                neuron: Neuron::expression(vec![
                    Neuron::symbol("threshold"),
                    Neuron::symbol("rsi"),
                    Neuron::symbol("overbought"),
                    Neuron::int_val(70),
                ]),
                tv: TruthValue::new(0.99, 0.99),
                timestamp: None, decay_rate: None, inferred: false,
            },
        ];

        let ctx = extract_brain_context(&facts);
        assert_eq!(ctx.pattern_rates.get("overbought"), Some(&(395, 10000)));
        // Only strong co-occur (>50%) should be included
        let cooccurs = ctx.strong_cooccur.get("overbought").unwrap();
        assert_eq!(cooccurs.len(), 1);
        assert_eq!(cooccurs[0].0, "trending");
        assert!((cooccurs[0].1 - 0.80).abs() < 0.01);
        // Threshold
        assert_eq!(ctx.thresholds.get(&("rsi".into(), "overbought".into())), Some(&70));
    }

    #[test]
    fn test_personality_text_rotation() {
        let mut ctx = BrainContext::default();
        ctx.personality.insert("greet".into(), vec![
            "Hello A".into(), "Hello B".into(), "Hello C".into(),
        ]);
        assert_eq!(ctx.personality_text("greet", 0, "default"), "Hello A");
        assert_eq!(ctx.personality_text("greet", 1, "default"), "Hello B");
        assert_eq!(ctx.personality_text("greet", 2, "default"), "Hello C");
        assert_eq!(ctx.personality_text("greet", 3, "default"), "Hello A"); // wraps
        assert_eq!(ctx.personality_text("missing", 0, "fallback"), "fallback");
    }

    #[test]
    fn test_format_response_with_context() {
        let mut ctx = BrainContext::default();
        ctx.pattern_rates.insert("overbought".into(), (395, 10000));
        ctx.strong_cooccur.insert("overbought".into(), vec![("trending".into(), 0.80)]);
        ctx.thresholds.insert(("rsi".into(), "overbought".into()), 70);

        let result = format_response(
            &[vec!["define".into(), "overbought".into(), "buying-pressure-extreme".into()]],
            Some(&ctx),
        );
        assert!(result.contains("buying pressure extreme"), "should have description");
        assert!(result.contains("4.0%") || result.contains("3.9%"), "should have percentage: {}", result);
        assert!(result.contains("trending"), "should mention co-occur: {}", result);
        assert!(result.contains("rsi"), "should mention threshold: {}", result);
    }

    #[test]
    fn test_format_response_none_context_unchanged() {
        // Passing None should produce identical output to old behavior
        assert_eq!(
            format_response(&[vec!["greet".into(), "hello".into()]], None),
            "Hello! I am QOR. Ask me about what I know."
        );
        assert_eq!(
            format_response(&[vec!["farewell".into(), "goodbye".into()]], None),
            "Goodbye! Keep reasoning."
        );
    }

    #[test]
    fn test_list_pattern_with_context() {
        let mut ctx = BrainContext::default();
        ctx.pattern_rates.insert("trending".into(), (3671, 10000));

        let result = format_response(
            &[vec!["list-pattern".into(), "trending".into(), "3671".into()]],
            Some(&ctx),
        );
        assert!(result.contains("36.7%"), "should have percentage: {}", result);
        assert!(result.contains("10000"), "should have total: {}", result);
    }

    #[test]
    fn test_personality_from_qor_facts() {
        let mut session = qor_runtime::eval::Session::new();
        session.exec(r#"(personality greet "Hey there friend!") <0.99, 0.99>"#).unwrap();

        let ctx = extract_brain_context(session.all_facts());
        assert_eq!(ctx.personality.get("greet").unwrap().len(), 1);
        assert_eq!(ctx.personality.get("greet").unwrap()[0], "Hey there friend!");
    }

    #[test]
    fn test_parse_teach_pattern_means() {
        let result = parse_teach_pattern("overbought means extreme buying pressure");
        assert_eq!(result, Some(("overbought".into(), "extreme-buying-pressure".into())));
    }

    #[test]
    fn test_parse_teach_pattern_define() {
        let result = parse_teach_pattern("define trending as upward price momentum");
        assert_eq!(result, Some(("trending".into(), "upward-price-momentum".into())));
    }

    #[test]
    fn test_parse_teach_pattern_remember() {
        let result = parse_teach_pattern("remember that RSI is relative strength index");
        assert_eq!(result, Some(("rsi".into(), "relative-strength-index".into())));
    }

    #[test]
    fn test_parse_teach_pattern_none() {
        assert!(parse_teach_pattern("what is overbought").is_none());
        assert!(parse_teach_pattern("hello").is_none());
        assert!(parse_teach_pattern("show patterns").is_none());
    }

    #[test]
    fn test_grammar_rule_fires() {
        // End-to-end: tokenize "hello" → load language + tokens → heartbeat → get response
        let mut session = qor_runtime::eval::Session::new();

        // Load language knowledge (bootstrap)
        let lang = language_knowledge_bootstrap();
        session.exec_statements(lang).unwrap();

        // Tokenize user input
        let tokens = tokenize("hello");
        session.exec_statements(tokens).unwrap();

        // Forward chaining should fire greeting rule
        for _ in 0..5 {
            session.heartbeat();
        }

        let responses = session.response_facts();
        assert!(!responses.is_empty(), "should have response facts");
        assert!(responses.iter().any(|r| r.first().map(|s| s.as_str()) == Some("greet")),
            "should detect greeting intent, got: {:?}", responses);
    }
}
