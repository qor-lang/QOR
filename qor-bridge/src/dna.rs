// ── QOR DNA — Profession Profile Converter ──────────────────────────────
//
// Reads profession DNA profiles (JSON + knowledge.txt) and converts them
// into QOR statements. Each profession becomes a loadable personality:
//
//   qor chat --dna doctor_gp
//
// DNA facts integrate with the existing BrainContext + personality system.
// Signature phrases become (personality greet "...") facts that mix into
// the rotation. Knowledge becomes (meaning topic "...") facts queryable
// via "what is <topic>".

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use qor_core::neuron::{Neuron, QorValue, Statement};
use qor_core::truth_value::TruthValue;

// ── Flexible Deserializers ─────────────────────────────────────────────
// DNA profiles come in two formats: the original (v1) and user-authored (v2).
// v2 uses single strings where v1 uses Vec<String>, and different field names.

/// Accept either a single string or an array of strings.
fn string_or_vec<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct StringOrVec;

    impl<'de> de::Visitor<'de> for StringOrVec {
        type Value = Vec<String>;

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("string or array of strings")
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            Ok(vec![v.to_string()])
        }

        fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
            Ok(vec![v])
        }

        fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut v = Vec::new();
            while let Some(s) = seq.next_element::<String>()? {
                v.push(s);
            }
            Ok(v)
        }
    }

    deserializer.deserialize_any(StringOrVec)
}

fn default_vec() -> Vec<String> {
    Vec::new()
}

// ── Serde Structs ────────────────────────────────────────────────────────
// Only the fields we convert — serde ignores unknown fields by default.
// Supports both v1 (HashMap personality) and v2 (array personality_traits).

#[derive(Debug, Deserialize)]
pub struct DnaProfile {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub category: String,
    pub archetype: String,
    pub tagline: String,
    #[serde(default)]
    pub voice: DnaVoice,
    #[serde(default)]
    pub origin: DnaOrigin,
    #[serde(default)]
    pub psychological_layers: DnaPsychLayers,
    #[serde(default)]
    pub personality: HashMap<String, DnaTrait>,
    /// v2 format: personality as array of {name, level, source, description}
    #[serde(default)]
    pub personality_traits: Vec<DnaTraitV2>,
    #[serde(default, alias = "trigger_conditions")]
    pub triggers: DnaTriggers,
    #[serde(default)]
    pub speech: DnaSpeech,
    #[serde(default)]
    pub contradiction_core: String,
    #[serde(default)]
    pub edges: Vec<DnaEdge>,
    #[serde(default)]
    pub system_prompt: String,
    /// v2 format: sources list
    #[serde(default)]
    pub sources: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct DnaOrigin {
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub core_wound: String,
    #[serde(default)]
    pub formative_belief: String,
}

#[derive(Debug, Default, Deserialize)]
pub struct DnaPsychLayers {
    #[serde(default = "default_vec", deserialize_with = "string_or_vec")]
    pub surface: Vec<String>,
    #[serde(default = "default_vec", deserialize_with = "string_or_vec")]
    pub middle: Vec<String>,
    #[serde(default = "default_vec", deserialize_with = "string_or_vec")]
    pub deep: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct DnaTriggers {
    #[serde(default = "default_vec", deserialize_with = "string_or_vec")]
    pub best_self: Vec<String>,
    #[serde(default = "default_vec", deserialize_with = "string_or_vec")]
    pub shadow_self: Vec<String>,
    #[serde(default)]
    pub conflict_response: String,
}

#[derive(Debug, Default, Deserialize)]
pub struct DnaVoice {
    #[serde(default)]
    pub texture: String,
    #[serde(default, alias = "motto")]
    pub internal_motto: String,
}

#[derive(Debug, Deserialize)]
pub struct DnaTrait {
    pub level: String,
    pub confidence: f64,
    #[serde(default)]
    pub shadow: String,
}

/// v2 format: personality trait with numeric level and source attribution
#[derive(Debug, Deserialize)]
pub struct DnaTraitV2 {
    pub name: String,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub level: u32,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Default, Deserialize)]
pub struct DnaSpeech {
    #[serde(default)]
    pub signature_phrases: Vec<String>,
    #[serde(default)]
    pub never_says: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct DnaEdge {
    pub source: String,
    pub relation: String,
    pub target: String,
    pub weight: f64,
}

// ── Index — keyword search ──────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct DnaIndex {
    #[serde(default)]
    professions: Vec<DnaIndexEntry>,
}

#[derive(Debug, Deserialize)]
struct DnaIndexEntry {
    id: String,
    name: String,
    #[serde(default)]
    keywords: Vec<String>,
}

/// Search DNA profiles by keyword. Returns matching profile IDs sorted by
/// relevance (most keyword hits first). Input is split on whitespace and
/// matched case-insensitively against each profile's keyword list, name, and id.
pub fn find_dna_by_keywords(dna_dir: &Path, query: &str) -> Vec<(String, String, usize)> {
    let index_path = dna_dir.join("index.json");
    let index: DnaIndex = match std::fs::read_to_string(&index_path) {
        Ok(data) => match serde_json::from_str(&data) {
            Ok(idx) => idx,
            Err(_) => return vec![],
        },
        Err(_) => return vec![],
    };

    let query_words: Vec<String> = query
        .to_lowercase()
        .split_whitespace()
        .map(|w| w.to_string())
        .collect();

    if query_words.is_empty() {
        return vec![];
    }

    let mut scored: Vec<(String, String, usize)> = Vec::new();

    for entry in &index.professions {
        let mut hits = 0usize;
        let id_lower = entry.id.to_lowercase();
        let name_lower = entry.name.to_lowercase();

        for qw in &query_words {
            // Exact id match = strong signal
            if id_lower == *qw {
                hits += 3;
                continue;
            }
            // Id contains the word
            if id_lower.contains(qw.as_str()) {
                hits += 2;
                continue;
            }
            // Name contains the word
            if name_lower.contains(qw.as_str()) {
                hits += 2;
                continue;
            }
            // Keyword match
            for kw in &entry.keywords {
                let kw_lower = kw.to_lowercase();
                if kw_lower == *qw || kw_lower.contains(qw.as_str()) {
                    hits += 1;
                    break;
                }
            }
        }

        if hits > 0 {
            scored.push((entry.id.clone(), entry.name.clone(), hits));
        }
    }

    scored.sort_by(|a, b| b.2.cmp(&a.2));
    scored
}

// ── Loading ──────────────────────────────────────────────────────────────

/// Load a DNA profile from `dna_dir/<id>/<id>.json` or `dna_dir/<id>/dna.json`.
pub fn load_dna(dna_dir: &Path, id: &str) -> Result<DnaProfile, String> {
    let dir = dna_dir.join(id);
    // Try <id>.json first, then dna.json
    let json_path = {
        let id_path = dir.join(format!("{}.json", id));
        if id_path.exists() {
            id_path
        } else {
            let dna_path = dir.join("dna.json");
            if dna_path.exists() {
                dna_path
            } else {
                return Err(format!("no DNA JSON found in '{}'", dir.display()));
            }
        }
    };
    let data = std::fs::read_to_string(&json_path)
        .map_err(|e| format!("could not read '{}': {}", json_path.display(), e))?;
    serde_json::from_str(&data)
        .map_err(|e| format!("could not parse '{}': {}", json_path.display(), e))
}

/// Load knowledge from `dna_dir/<id>/knowledge.txt` or `knowledge.md`.
pub fn load_knowledge(dna_dir: &Path, id: &str) -> Result<String, String> {
    let dir = dna_dir.join(id);
    // Try knowledge.txt first, then knowledge.md
    let path = {
        let txt = dir.join("knowledge.txt");
        if txt.exists() {
            txt
        } else {
            let md = dir.join("knowledge.md");
            if md.exists() {
                md
            } else {
                return Err(format!("no knowledge file found in '{}'", dir.display()));
            }
        }
    };
    std::fs::read_to_string(&path)
        .map_err(|e| format!("could not read '{}': {}", path.display(), e))
}

/// Load shared meta-rules from `workspace_root/meta/*.qor`.
///
/// Meta-rules (agi.qor, general.qor, biology.qor, physics.qor) apply to ALL
/// DNA profiles. They provide universal reasoning rules — transitivity,
/// symmetry, classification, 6D state vectors, etc.
///
/// `workspace_root` is the QOR workspace root (where `meta/`, `dna/`, `brain/` live).
/// Returns parsed QOR statements. Returns empty vec if meta/ doesn't exist.
pub fn load_meta_rules(workspace_root: &Path) -> Vec<Statement> {
    let meta_dir = workspace_root.join("meta");
    if !meta_dir.exists() {
        return vec![];
    }
    let mut stmts = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&meta_dir) {
        let mut files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "qor"))
            .collect();
        files.sort_by_key(|e| e.path());
        for entry in files {
            let path = entry.path();
            match std::fs::read_to_string(&path) {
                Ok(source) => match qor_core::parser::parse(&source) {
                    Ok(parsed) => {
                        stmts.extend(parsed);
                    }
                    Err(e) => {
                        eprintln!(
                            "warning: could not parse meta-rule '{}': {}",
                            path.display(),
                            e
                        );
                    }
                },
                Err(e) => {
                    eprintln!(
                        "warning: could not read meta-rule '{}': {}",
                        path.display(),
                        e
                    );
                }
            }
        }
    }
    stmts
}

/// Load templates.qor raw text from `dna_dir/<id>/templates.qor`.
/// Returns None if no file exists.
pub fn load_templates_text(dna_dir: &Path, id: &str) -> Option<String> {
    let path = dna_dir.join(id).join("templates.qor");
    std::fs::read_to_string(&path).ok()
}

/// Load rules.qor from `dna_dir/<id>/rules.qor`.
/// Returns parsed QOR statements (facts + rules). Returns empty vec if no file.
pub fn load_rules(dna_dir: &Path, id: &str) -> Vec<Statement> {
    let path = dna_dir.join(id).join("rules.qor");
    match std::fs::read_to_string(&path) {
        Ok(source) => {
            match qor_core::parser::parse(&source) {
                Ok(stmts) => stmts,
                Err(e) => {
                    eprintln!("warning: could not parse '{}': {}", path.display(), e);
                    vec![]
                }
            }
        }
        Err(_) => vec![], // No rules.qor — that's fine
    }
}

/// List available DNA profile IDs in the dna/ folder.
///
/// Looks for subdirectories containing `<name>/<name>.json` or `<name>/dna.json`.
pub fn available_dna(dna_dir: &Path) -> Vec<(String, String)> {
    let mut profiles = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dna_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    let id_json = path.join(format!("{}.json", name));
                    let dna_json = path.join("dna.json");
                    let json_path = if id_json.exists() {
                        Some(id_json)
                    } else if dna_json.exists() {
                        Some(dna_json)
                    } else {
                        None
                    };
                    if let Some(jp) = json_path {
                        let display_name = std::fs::read_to_string(&jp)
                            .ok()
                            .and_then(|data| serde_json::from_str::<DnaProfile>(&data).ok())
                            .map(|p| p.name)
                            .unwrap_or_else(|| name.to_string());
                        profiles.push((name.to_string(), display_name));
                    }
                }
            }
        }
    }
    profiles.sort_by(|a, b| a.0.cmp(&b.0));
    profiles
}

// ── Conversion ───────────────────────────────────────────────────────────

/// Convert a DNA profile into QOR statements.
///
/// Generates:
/// - Identity facts: (dna-id X), (dna-name "X"), (dna-archetype "X"), (dna-tagline "X")
/// - Voice: (dna-voice "texture text")
/// - Traits: (dna-trait name level) <confidence, 0.90>
/// - Speech phrases: (personality greet "phrase") <0.99, 0.99>
/// - Never-says: (dna-never-says "phrase") <0.99, 0.99>
/// - Edges: (dna-edge source relation target) <weight, 0.90>
pub fn dna_to_statements(profile: &DnaProfile) -> Vec<Statement> {
    let mut stmts = Vec::new();
    let high_tv = Some(TruthValue::new(0.99, 0.99));

    // Identity facts
    stmts.push(make_fact(
        vec![sym("dna-id"), sym(&profile.id)],
        high_tv,
    ));
    stmts.push(make_fact(
        vec![sym("dna-name"), str_val(&profile.name)],
        high_tv,
    ));
    stmts.push(make_fact(
        vec![sym("dna-archetype"), str_val(&profile.archetype)],
        high_tv,
    ));
    stmts.push(make_fact(
        vec![sym("dna-tagline"), str_val(&profile.tagline)],
        high_tv,
    ));
    if !profile.category.is_empty() {
        stmts.push(make_fact(
            vec![sym("dna-category"), sym(&profile.category.to_lowercase())],
            high_tv,
        ));
    }

    // Voice
    if !profile.voice.texture.is_empty() {
        stmts.push(make_fact(
            vec![sym("dna-voice"), str_val(&profile.voice.texture)],
            high_tv,
        ));
    }
    if !profile.voice.internal_motto.is_empty() {
        stmts.push(make_fact(
            vec![sym("dna-motto"), str_val(&profile.voice.internal_motto)],
            high_tv,
        ));
    }

    // Personality traits — v1 format (HashMap)
    let mut traits: Vec<_> = profile.personality.iter().collect();
    traits.sort_by_key(|(k, _)| (*k).clone());
    for (name, trait_info) in traits {
        let tv = Some(TruthValue::new(trait_info.confidence, 0.90));
        stmts.push(make_fact(
            vec![
                sym("dna-trait"),
                sym(&name.replace('_', "-")),
                sym(&trait_info.level.replace('_', "-")),
            ],
            tv,
        ));
    }

    // Personality traits — v2 format (array with numeric level)
    for t in &profile.personality_traits {
        let confidence = (t.level as f64) / 100.0;
        let tv = Some(TruthValue::new(confidence.min(0.99), 0.90));
        let trait_name = t.name.to_lowercase().replace(' ', "-").replace('_', "-");
        stmts.push(make_fact(
            vec![sym("dna-trait"), sym(&trait_name), sym("master")],
            tv,
        ));
        if !t.source.is_empty() {
            stmts.push(make_fact(
                vec![sym("dna-trait-source"), sym(&trait_name), str_val(&t.source)],
                Some(TruthValue::new(0.95, 0.90)),
            ));
        }
    }

    // Sources (v2)
    for source in &profile.sources {
        stmts.push(make_fact(
            vec![sym("dna-source"), str_val(source)],
            high_tv,
        ));
    }

    // Speech: signature phrases become personality greet facts
    for phrase in &profile.speech.signature_phrases {
        stmts.push(make_fact(
            vec![sym("personality"), sym("greet"), str_val(phrase)],
            high_tv,
        ));
    }

    // Never-says
    for phrase in &profile.speech.never_says {
        stmts.push(make_fact(
            vec![sym("dna-never-says"), str_val(phrase)],
            high_tv,
        ));
    }

    // Edges
    for edge in &profile.edges {
        let tv = Some(TruthValue::new(edge.weight, 0.90));
        stmts.push(make_fact(
            vec![
                sym("dna-edge"),
                sym(&edge.source.replace('_', "-")),
                sym(&edge.relation),
                sym(&edge.target.replace('_', "-")),
            ],
            tv,
        ));
    }

    // Origin
    if !profile.origin.summary.is_empty() {
        stmts.push(make_fact(
            vec![sym("dna-origin"), str_val(&profile.origin.summary)],
            high_tv,
        ));
    }
    if !profile.origin.core_wound.is_empty() {
        stmts.push(make_fact(
            vec![sym("dna-core-wound"), str_val(&profile.origin.core_wound)],
            high_tv,
        ));
    }
    if !profile.origin.formative_belief.is_empty() {
        stmts.push(make_fact(
            vec![sym("dna-formative-belief"), str_val(&profile.origin.formative_belief)],
            high_tv,
        ));
    }

    // Psychological layers
    for layer_text in &profile.psychological_layers.surface {
        stmts.push(make_fact(
            vec![sym("dna-psych"), sym("surface"), str_val(layer_text)],
            high_tv,
        ));
    }
    for layer_text in &profile.psychological_layers.middle {
        stmts.push(make_fact(
            vec![sym("dna-psych"), sym("middle"), str_val(layer_text)],
            high_tv,
        ));
    }
    for layer_text in &profile.psychological_layers.deep {
        stmts.push(make_fact(
            vec![sym("dna-psych"), sym("deep"), str_val(layer_text)],
            high_tv,
        ));
    }

    // Triggers
    for trigger in &profile.triggers.best_self {
        stmts.push(make_fact(
            vec![sym("dna-trigger"), sym("best-self"), str_val(trigger)],
            high_tv,
        ));
    }
    for trigger in &profile.triggers.shadow_self {
        stmts.push(make_fact(
            vec![sym("dna-trigger"), sym("shadow-self"), str_val(trigger)],
            high_tv,
        ));
    }
    if !profile.triggers.conflict_response.is_empty() {
        stmts.push(make_fact(
            vec![sym("dna-conflict-response"), str_val(&profile.triggers.conflict_response)],
            high_tv,
        ));
    }

    // Contradiction
    if !profile.contradiction_core.is_empty() {
        stmts.push(make_fact(
            vec![sym("dna-contradiction"), str_val(&profile.contradiction_core)],
            high_tv,
        ));
    }

    // System prompt (complete persona prompt)
    if !profile.system_prompt.is_empty() {
        stmts.push(make_fact(
            vec![sym("dna-system-prompt"), str_val(&profile.system_prompt)],
            high_tv,
        ));
    }

    stmts
}

/// Parse knowledge.txt (markdown with ## sections and - bullets) into meaning facts.
///
/// Each bullet becomes `(meaning section-topic "bullet text") <0.90, 0.85>`.
pub fn parse_knowledge(text: &str) -> Vec<Statement> {
    let mut stmts = Vec::new();
    let tv = Some(TruthValue::new(0.90, 0.85));
    let mut current_section = String::new();

    for line in text.lines() {
        let trimmed = line.trim();

        // Section header: ## Cardiovascular
        if let Some(header) = trimmed.strip_prefix("## ") {
            current_section = sanitize_topic(header.trim());
            continue;
        }

        // Skip top-level # header
        if trimmed.starts_with("# ") {
            continue;
        }

        // Bullet point: - text here
        if let Some(bullet) = trimmed.strip_prefix("- ") {
            if current_section.is_empty() {
                continue;
            }
            let text = sanitize_knowledge_text(bullet.trim());
            if text.is_empty() {
                continue;
            }
            stmts.push(make_fact(
                vec![
                    sym("meaning"),
                    sym(&current_section),
                    str_val(&text),
                ],
                tv,
            ));
        }
    }

    stmts
}

// ── Saving ───────────────────────────────────────────────────────────────

/// Serialize QOR statements to `.qor` text format.
fn statements_to_qor_text(stmts: &[Statement], header: &str) -> String {
    let mut lines = Vec::new();
    lines.push(format!(";; {}", header));
    lines.push(";; Auto-generated from DNA JSON — edit freely.".to_string());
    lines.push(String::new());

    let mut current_section = String::new();

    for stmt in stmts {
        if let Statement::Fact { neuron, tv, decay } = stmt {
            if let Neuron::Expression(parts) = neuron {
                // Detect section changes for readability
                let pred = parts.first()
                    .map(|p| p.to_string())
                    .unwrap_or_default();
                let section = match pred.as_str() {
                    "dna-id" | "dna-name" | "dna-archetype"
                    | "dna-tagline" | "dna-category" => "Identity",
                    "dna-voice" | "dna-motto" => "Voice",
                    "dna-trait" => "Personality Traits",
                    "personality" => "Personality Phrases",
                    "dna-never-says" => "Never Says",
                    "dna-edge" => "Trait Edges",
                    "dna-origin" | "dna-core-wound"
                    | "dna-formative-belief" => "Origin",
                    "dna-psych" => "Psychological Layers",
                    "dna-trigger" => "Triggers",
                    "dna-conflict-response" => "Conflict Response",
                    "dna-contradiction" => "Contradiction",
                    "dna-system-prompt" => "System Prompt",
                    "meaning" => "Knowledge",
                    _ => "",
                };
                if section != current_section {
                    if !current_section.is_empty() {
                        lines.push(String::new());
                    }
                    if !section.is_empty() {
                        lines.push(format!(";; ── {} ──", section));
                    }
                    current_section = section.to_string();
                }

                // Format the fact
                let neuron_str = format_neuron(neuron);
                let tv_str = tv
                    .map(|t| format!(" <{:.2}, {:.2}>", t.strength, t.confidence))
                    .unwrap_or_default();
                let decay_str = decay
                    .map(|d| format!(" @decay {:.2}", d))
                    .unwrap_or_default();
                lines.push(format!("{}{}{}", neuron_str, tv_str, decay_str));
            }
        }
    }

    lines.push(String::new());
    lines.join("\n")
}

/// Format a Neuron as QOR syntax.
fn format_neuron(neuron: &Neuron) -> String {
    match neuron {
        Neuron::Expression(parts) => {
            let inner: Vec<String> = parts.iter().map(|p| format_neuron(p)).collect();
            format!("({})", inner.join(" "))
        }
        Neuron::Symbol(s) => s.clone(),
        Neuron::Variable(v) => format!("${}", v),
        Neuron::Value(QorValue::Int(n)) => n.to_string(),
        Neuron::Value(QorValue::Float(f)) => format!("{:.2}", f),
        Neuron::Value(QorValue::Str(s)) => format!("\"{}\"", s.replace('"', "\\\"")),
        Neuron::Value(QorValue::Bool(b)) => b.to_string(),
    }
}

/// Convert a DNA profile to .qor and save to `dna_dir/<id>/<id>.qor`.
///
/// Returns the path of the saved file and statement count.
pub fn save_dna_qor(dna_dir: &Path, id: &str) -> Result<(std::path::PathBuf, usize), String> {
    let profile = load_dna(dna_dir, id)?;
    let mut stmts = dna_to_statements(&profile);

    // Also include knowledge
    if let Ok(knowledge_text) = load_knowledge(dna_dir, id) {
        stmts.extend(parse_knowledge(&knowledge_text));
    }

    // Also include rules
    stmts.extend(load_rules(dna_dir, id));

    let header = format!(
        "DNA: {} — {} — \"{}\"",
        profile.id, profile.name, profile.archetype
    );
    let text = statements_to_qor_text(&stmts, &header);
    let out_path = dna_dir.join(id).join(format!("{}.qor", id));
    let count = stmts.len();

    std::fs::write(&out_path, &text)
        .map_err(|e| format!("could not write '{}': {}", out_path.display(), e))?;

    Ok((out_path, count))
}

/// Convert ALL DNA profiles to .qor files.
///
/// Returns a list of (id, path, count) for each converted profile.
pub fn convert_all(dna_dir: &Path) -> Vec<(String, std::path::PathBuf, usize)> {
    let profiles = available_dna(dna_dir);
    let mut results = Vec::new();

    for (id, _name) in &profiles {
        match save_dna_qor(dna_dir, id) {
            Ok((path, count)) => results.push((id.clone(), path, count)),
            Err(e) => eprintln!("  warning: {}: {}", id, e),
        }
    }

    results
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn sym(s: &str) -> Neuron {
    Neuron::Symbol(s.to_string())
}

fn str_val(s: &str) -> Neuron {
    Neuron::Value(QorValue::Str(s.to_string()))
}

fn make_fact(parts: Vec<Neuron>, tv: Option<TruthValue>) -> Statement {
    Statement::Fact {
        neuron: Neuron::Expression(parts),
        tv,
        decay: None,
    }
}

fn sanitize_topic(s: &str) -> String {
    s.to_lowercase()
        .replace(' ', "-")
        .replace(|c: char| !c.is_alphanumeric() && c != '-', "")
}

fn sanitize_knowledge_text(s: &str) -> String {
    // Truncate to ~200 chars at a sentence boundary (respecting UTF-8)
    if s.len() <= 200 {
        return s.to_string();
    }
    // Find a safe char boundary at or before 200
    let mut end = 200;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    // Try to cut at a sentence boundary
    if let Some(pos) = s[..end].rfind(". ") {
        s[..pos + 1].to_string()
    } else {
        s[..end].to_string()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_dna_doctor() {
        let dna_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("dna");
        let profile = load_dna(&dna_dir, "doctor_gp").unwrap();
        assert_eq!(profile.id, "doctor_gp");
        assert_eq!(profile.archetype, "The Healer");
        assert!(!profile.personality.is_empty());
        assert!(profile.personality.contains_key("empathy"));
        assert!((profile.personality["empathy"].confidence - 0.88).abs() < 0.01);
        assert!(!profile.speech.signature_phrases.is_empty());
        assert!(!profile.edges.is_empty());
    }

    #[test]
    fn test_dna_to_statements() {
        let dna_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("dna");
        let profile = load_dna(&dna_dir, "doctor_gp").unwrap();
        let stmts = dna_to_statements(&profile);

        // Should have identity facts
        let has_id = stmts.iter().any(|s| {
            if let Statement::Fact { neuron, .. } = s {
                if let Neuron::Expression(parts) = neuron {
                    return parts.len() == 2
                        && parts[0] == Neuron::Symbol("dna-id".into())
                        && parts[1] == Neuron::Symbol("doctor_gp".into());
                }
            }
            false
        });
        assert!(has_id, "should have (dna-id doctor_gp)");

        // Should have trait facts with confidence as TruthValue strength
        let has_empathy = stmts.iter().any(|s| {
            if let Statement::Fact { neuron, tv, .. } = s {
                if let Neuron::Expression(parts) = neuron {
                    if parts.len() == 3
                        && parts[0] == Neuron::Symbol("dna-trait".into())
                        && parts[1] == Neuron::Symbol("empathy".into())
                    {
                        if let Some(t) = tv {
                            return (t.strength - 0.88).abs() < 0.01;
                        }
                    }
                }
            }
            false
        });
        assert!(has_empathy, "should have (dna-trait empathy high) <0.88, 0.90>");

        // Should have personality greet facts from signature phrases
        let greet_count = stmts.iter().filter(|s| {
            if let Statement::Fact { neuron, .. } = s {
                if let Neuron::Expression(parts) = neuron {
                    return parts.len() == 3
                        && parts[0] == Neuron::Symbol("personality".into())
                        && parts[1] == Neuron::Symbol("greet".into());
                }
            }
            false
        }).count();
        assert!(greet_count >= 3, "should have at least 3 personality greet facts, got {}", greet_count);

        // Should have edge facts
        let has_edge = stmts.iter().any(|s| {
            if let Statement::Fact { neuron, .. } = s {
                if let Neuron::Expression(parts) = neuron {
                    return parts.len() == 4
                        && parts[0] == Neuron::Symbol("dna-edge".into());
                }
            }
            false
        });
        assert!(has_edge, "should have dna-edge facts");
    }

    #[test]
    fn test_parse_knowledge() {
        let text = "\
# MEDICAL KNOWLEDGE CORE — General Practitioner

## Diagnostic Framework
- HISTORY: Chief complaint, HPC, Past Medical History
- RED FLAGS: Chest pain + radiation = ACS

## Cardiovascular
- Hypertension: Target <130/80. First-line: ACEi or ARB.
- Heart Failure: HFrEF — ACEi + beta-blocker + MRA.
";
        let stmts = parse_knowledge(text);
        assert_eq!(stmts.len(), 4);

        // Check first fact: (meaning diagnostic-framework "HISTORY: ...")
        if let Statement::Fact { neuron, tv, .. } = &stmts[0] {
            if let Neuron::Expression(parts) = neuron {
                assert_eq!(parts[0], Neuron::Symbol("meaning".into()));
                assert_eq!(parts[1], Neuron::Symbol("diagnostic-framework".into()));
                if let Neuron::Value(QorValue::Str(s)) = &parts[2] {
                    assert!(s.contains("HISTORY"), "should contain bullet text: {}", s);
                } else {
                    panic!("expected string value");
                }
            }
            assert!(tv.is_some());
        }

        // Check cardiovascular section
        let cardio_count = stmts.iter().filter(|s| {
            if let Statement::Fact { neuron, .. } = s {
                if let Neuron::Expression(parts) = neuron {
                    return parts[1] == Neuron::Symbol("cardiovascular".into());
                }
            }
            false
        }).count();
        assert_eq!(cardio_count, 2);
    }

    #[test]
    fn test_available_dna() {
        let dna_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("dna");
        let profiles = available_dna(&dna_dir);
        assert!(profiles.len() >= 10, "should have at least 10 DNA profiles, got {}", profiles.len());

        // Should include doctor_gp
        assert!(profiles.iter().any(|(id, _)| id == "doctor_gp"),
            "should include doctor_gp: {:?}", profiles);
    }

    #[test]
    fn test_find_dna_by_keywords() {
        let dna_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("dna");

        // "doctor" should find doctor_gp
        let results = find_dna_by_keywords(&dna_dir, "doctor");
        assert!(!results.is_empty(), "should find something for 'doctor'");
        assert_eq!(results[0].0, "doctor_gp");

        // "circuit voltage" should find electrical_engineer
        let results = find_dna_by_keywords(&dna_dir, "circuit voltage");
        assert!(!results.is_empty(), "should find something for 'circuit voltage'");
        assert_eq!(results[0].0, "electrical_engineer");

        // "chess puzzle" should find puzzle_solver
        let results = find_dna_by_keywords(&dna_dir, "chess puzzle");
        assert!(!results.is_empty(), "should find something for 'chess puzzle'");
        assert_eq!(results[0].0, "puzzle_solver");

        // "finance budget" should find cfo
        let results = find_dna_by_keywords(&dna_dir, "finance budget");
        assert!(!results.is_empty(), "should find something for 'finance budget'");
        assert_eq!(results[0].0, "cfo");

        // Exact id match should rank highest
        let results = find_dna_by_keywords(&dna_dir, "surgeon");
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "surgeon");

        // Empty query returns nothing
        let results = find_dna_by_keywords(&dna_dir, "");
        assert!(results.is_empty());
    }

    #[test]
    fn test_trader_rules_parse() {
        let dna_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("dna");
        let stmts = load_rules(&dna_dir, "trader");
        // Should have a large number of rules (20 sections, ~100+ rules)
        assert!(stmts.len() > 80, "expected 80+ statements, got {}", stmts.len());

        // Check that buy/sell/hold rules exist
        let has_buy = stmts.iter().any(|s| {
            if let Statement::Rule { head, .. } = s {
                if let Neuron::Expression(parts) = head {
                    return parts.len() == 2
                        && parts[0] == Neuron::Symbol("buy".into());
                }
            }
            false
        });
        assert!(has_buy, "should have (buy $sym) rules");

        let has_sell = stmts.iter().any(|s| {
            if let Statement::Rule { head, .. } = s {
                if let Neuron::Expression(parts) = head {
                    return parts.len() == 2
                        && parts[0] == Neuron::Symbol("sell".into());
                }
            }
            false
        });
        assert!(has_sell, "should have (sell $sym) rules");

        let has_hold = stmts.iter().any(|s| {
            if let Statement::Rule { head, .. } = s {
                if let Neuron::Expression(parts) = head {
                    return parts.len() == 2
                        && parts[0] == Neuron::Symbol("hold".into());
                }
            }
            false
        });
        assert!(has_hold, "should have (hold $sym) rules");

        // Check regime detection rules
        let has_regime = stmts.iter().any(|s| {
            if let Statement::Rule { head, .. } = s {
                if let Neuron::Expression(parts) = head {
                    if let Neuron::Symbol(name) = &parts[0] {
                        return name.starts_with("regime-");
                    }
                }
            }
            false
        });
        assert!(has_regime, "should have regime detection rules");

        // Check confluence rules
        let has_confluence = stmts.iter().any(|s| {
            if let Statement::Rule { head, .. } = s {
                if let Neuron::Expression(parts) = head {
                    if let Neuron::Symbol(name) = &parts[0] {
                        return name == "multi-confirm-buy" || name == "multi-confirm-sell";
                    }
                }
            }
            false
        });
        assert!(has_confluence, "should have multi-confirm confluence rules");
    }

    #[test]
    fn test_trader_rules_forward_chain() {
        // Feed indicator facts, run forward chaining, verify buy/sell/hold reasoning
        let dna_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("dna");
        let rules = load_rules(&dna_dir, "trader");
        assert!(!rules.is_empty(), "trader rules should parse");

        let mut session = qor_runtime::eval::Session::new();
        session.exec_statements(rules).unwrap();

        // Inject indicator facts for a bullish VWAP mean-reversion setup:
        // RSI oversold + bullish divergence + lower band touched + below VWAP + R:R > 1.5
        let facts_src = r#"
            (current-price btc 84000.0)
            (rsi btc 28.5)
            (rsi-div btc bullish)
            (lower-band-touched btc)
            (vwap btc 86000.0)
            (vwap-upper btc 88000.0)
            (vwap-lower btc 83500.0)
            (risk-reward btc 2.3)
            (adx btc 18.0)
            (ema-21 btc 85000.0)
            (ema-50 btc 85500.0)
            (ema-200 btc 84500.0)
            (bb-upper btc 88000.0)
            (bb-mid btc 85000.0)
            (bb-lower btc 83800.0)
            (macd-line btc -50.0)
            (macd-signal btc -30.0)
            (macd-histogram btc -20.0)
            (pivot-pp btc 85200.0)
            (pivot-s1 btc 84000.0)
            (pivot-s2 btc 83000.0)
            (pivot-r1 btc 86500.0)
            (pivot-r2 btc 87500.0)
            (atr btc 1200.0)
        "#;
        let facts = qor_core::parser::parse(facts_src).unwrap();
        session.exec_statements(facts).unwrap();

        let all = session.all_facts();

        // Helper: check if a fact with given predicate exists
        let has_pred = |pred: &str| -> bool {
            all.iter().any(|sn| {
                if let Neuron::Expression(p) = &sn.neuron {
                    p.len() >= 2 && p[0] == Neuron::Symbol(pred.into())
                } else { false }
            })
        };
        let has_pred_sym = |pred: &str, sym: &str| -> bool {
            all.iter().any(|sn| {
                if let Neuron::Expression(p) = &sn.neuron {
                    p.len() == 2
                        && p[0] == Neuron::Symbol(pred.into())
                        && p[1] == Neuron::Symbol(sym.into())
                } else { false }
            })
        };

        // Should derive (oversold btc)
        assert!(has_pred("oversold"), "should derive (oversold btc)");

        // Should derive (signal-long btc) from lower-band-touched
        assert!(has_pred("signal-long"), "should derive (signal-long btc)");

        // Should derive (rsi-confirmed-long btc) from bullish divergence
        assert!(has_pred("rsi-confirmed-long"), "should derive (rsi-confirmed-long btc)");

        // Should derive (buy btc) — VWAP buy with R:R gate
        assert!(has_pred_sym("buy", "btc"),
            "should derive (buy btc) from VWAP mean-reversion setup.\nDerived facts: {:?}",
            all.iter().map(|sn| format!("{:?}", sn.neuron)).collect::<Vec<_>>());

        // Should NOT derive (sell btc) in this setup
        assert!(!has_pred_sym("sell", "btc"), "should NOT derive (sell btc) in bullish setup");
    }

    #[test]
    fn test_dna_personality_overrides() {
        // DNA signature phrases should become personality greet facts
        // that BrainContext picks up alongside default personality
        let dna_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("dna");
        let profile = load_dna(&dna_dir, "doctor_gp").unwrap();
        let stmts = dna_to_statements(&profile);

        // Load into a session and extract brain context
        let mut session = qor_runtime::eval::Session::new();
        session.exec_statements(stmts).unwrap();

        let ctx = crate::language::extract_brain_context(session.all_facts());

        // Should have greet personality from DNA signature phrases
        let greets = ctx.personality.get("greet").expect("should have greet personality");
        assert!(greets.len() >= 3, "should have at least 3 greet variants, got {}", greets.len());
        assert!(greets.iter().any(|g| g.contains("start from the beginning")),
            "should include doctor signature phrase: {:?}", greets);
    }
}
