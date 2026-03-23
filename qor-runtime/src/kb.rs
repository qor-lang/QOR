// ── Binary Knowledge Base ────────────────────────────────────────────
//
// Triple-indexed fact store for fast lookups. Data stays in binary format
// (12 bytes per fact). Rules stay in .qor files. This module bridges them.
//
// Binary format: u32 subject | u16 predicate | u32 object | u8 trust | u8 plausibility
//
// Ported from Ref/qor-data/query.rs — stripped external deps, std only + byteorder.

use qor_core::neuron::{Neuron, Statement};
use qor_core::truth_value::TruthValue;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write as IoWrite};
use std::path::Path;

// ═══════════════════════════════════════════════════════════════
// Fact — a single triple with confidence
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy)]
pub struct Fact {
    pub subject: u32,
    pub predicate: u16,
    pub object: u32,
    pub trust: u8,
    pub plausibility: u8,
}

impl Fact {
    pub fn trust_f32(&self) -> f32 {
        self.trust as f32 / 255.0
    }
    pub fn plausibility_f32(&self) -> f32 {
        self.plausibility as f32 / 255.0
    }
    pub fn confidence(&self) -> f32 {
        self.trust_f32() * self.plausibility_f32()
    }

    fn read_from<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut buf = [0u8; 12];
        r.read_exact(&mut buf)?;
        Ok(Fact {
            subject: u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            predicate: u16::from_le_bytes([buf[4], buf[5]]),
            object: u32::from_le_bytes([buf[6], buf[7], buf[8], buf[9]]),
            trust: buf[10],
            plausibility: buf[11],
        })
    }
}

// ═══════════════════════════════════════════════════════════════
// FactWithNames — resolved fact for display and QOR conversion
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct FactWithNames {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub trust: f32,
    pub plausibility: f32,
}

impl FactWithNames {
    pub fn confidence(&self) -> f32 {
        self.trust * self.plausibility
    }

    /// Format as QOR fact string
    pub fn to_qor_string(&self) -> String {
        format!(
            "({} {} {}) <{:.2}, {:.2}>",
            self.predicate,
            sanitize_qor(&self.subject),
            sanitize_qor(&self.object),
            self.trust,
            self.plausibility
        )
    }

    /// Convert to parsed QOR Statement
    pub fn to_statement(&self) -> Statement {
        let tv = TruthValue::new(self.trust as f64, self.plausibility as f64);
        Statement::Fact {
            neuron: Neuron::Expression(vec![
                Neuron::Symbol(self.predicate.clone()),
                Neuron::Symbol(sanitize_qor(&self.subject)),
                Neuron::Symbol(sanitize_qor(&self.object)),
            ]),
            tv: Some(tv),
            decay: None,
        }
    }
}

/// Make a string safe for QOR (no spaces, lowercase, replace special chars)
fn sanitize_qor(s: &str) -> String {
    s.to_lowercase()
        .replace(' ', "_")
        .replace('\'', "")
        .replace('"', "")
        .replace('(', "")
        .replace(')', "")
        .replace(',', "")
        .replace('.', "")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
        .collect()
}

// ═══════════════════════════════════════════════════════════════
// KnowledgeBase — triple-indexed fact store
// ═══════════════════════════════════════════════════════════════

pub struct KnowledgeBase {
    facts: Vec<Fact>,

    /// Index: predicate → fact indices
    by_predicate: HashMap<u16, Vec<usize>>,

    /// Index: subject → fact indices
    by_subject: HashMap<u32, Vec<usize>>,

    /// Index: object → fact indices
    by_object: HashMap<u32, Vec<usize>>,

    /// Tombstoned fact indices (removed but slot kept to preserve indices)
    deleted: HashSet<usize>,

    /// Next available entity ID (for runtime registration)
    next_entity: u32,

    /// Next available predicate ID (for runtime registration)
    next_predicate: u16,

    pub entity_names: HashMap<u32, String>,
    pub entity_ids: HashMap<String, u32>,
    pub predicate_names: HashMap<u16, String>,
    pub predicate_ids: HashMap<String, u16>,
}

impl KnowledgeBase {
    /// Create an empty knowledge base
    pub fn new() -> Self {
        KnowledgeBase {
            facts: Vec::new(),
            by_predicate: HashMap::new(),
            by_subject: HashMap::new(),
            by_object: HashMap::new(),
            deleted: HashSet::new(),
            next_entity: 1,
            next_predicate: 1,
            entity_names: HashMap::new(),
            entity_ids: HashMap::new(),
            predicate_names: HashMap::new(),
            predicate_ids: HashMap::new(),
        }
    }

    /// Load knowledge base from a directory containing triples.bin + *.tsv mappings.
    ///
    /// Expected files:
    /// - `triples.bin` — binary facts (12 bytes each)
    /// - `entities.tsv` — id\tlabel mapping
    /// - `predicates.tsv` — id\tlabel mapping
    ///
    /// Also loads any `*_triples.bin` files (oeis_triples.bin, dbpedia_triples.bin, etc.)
    pub fn load(dir: &Path) -> io::Result<Self> {
        let mut kb = KnowledgeBase::new();

        // Load entity mappings
        let entities_path = dir.join("entities.tsv");
        if entities_path.exists() {
            let f = File::open(&entities_path)?;
            for line in BufReader::new(f).lines().skip(1) {
                let line = line?;
                let parts: Vec<&str> = line.splitn(2, '\t').collect();
                if parts.len() == 2 {
                    if let Ok(id) = parts[0].parse::<u32>() {
                        let label = parts[1].to_string();
                        kb.entity_names.insert(id, label.clone());
                        kb.entity_ids.insert(label.to_lowercase(), id);
                        if id >= kb.next_entity {
                            kb.next_entity = id + 1;
                        }
                    }
                }
            }
        }

        // Load predicate mappings
        let preds_path = dir.join("predicates.tsv");
        if preds_path.exists() {
            let f = File::open(&preds_path)?;
            for line in BufReader::new(f).lines().skip(1) {
                let line = line?;
                let parts: Vec<&str> = line.splitn(2, '\t').collect();
                if parts.len() == 2 {
                    if let Ok(id) = parts[0].parse::<u16>() {
                        let label = parts[1].to_string();
                        kb.predicate_names.insert(id, label.clone());
                        kb.predicate_ids.insert(label.to_lowercase(), id);
                        if id >= kb.next_predicate {
                            kb.next_predicate = id + 1;
                        }
                    }
                }
            }
        }

        // Load all binary triple files
        let mut triple_files = Vec::new();
        let main_triples = dir.join("triples.bin");
        if main_triples.exists() {
            triple_files.push(main_triples);
        }
        // Also load oeis_triples.bin, dbpedia_triples.bin, etc.
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.ends_with("_triples.bin") {
                        triple_files.push(path);
                    }
                }
            }
        }

        for triple_path in &triple_files {
            let f = File::open(triple_path)?;
            let mut reader = BufReader::with_capacity(4 * 1024 * 1024, f);
            loop {
                match Fact::read_from(&mut reader) {
                    Ok(fact) => {
                        let idx = kb.facts.len();
                        kb.by_predicate.entry(fact.predicate).or_default().push(idx);
                        kb.by_subject.entry(fact.subject).or_default().push(idx);
                        kb.by_object.entry(fact.object).or_default().push(idx);
                        kb.facts.push(fact);
                    }
                    Err(_) => break,
                }
            }
        }

        Ok(kb)
    }

    /// Number of facts in the knowledge base
    pub fn len(&self) -> usize {
        self.facts.len()
    }

    /// Whether the knowledge base is empty
    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }

    // ═══════════════════════════════════════════════════════════
    // Queries — all return confidence scores
    // ═══════════════════════════════════════════════════════════

    /// Everything about an entity, sorted by confidence
    pub fn about(&self, name: &str) -> Vec<FactWithNames> {
        let id = match self.entity_ids.get(&name.to_lowercase()) {
            Some(id) => *id,
            None => return vec![],
        };

        let mut out: Vec<FactWithNames> = self
            .by_subject
            .get(&id)
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|&idx| {
                let f = &self.facts[idx];
                Some(FactWithNames {
                    subject: self.entity_names.get(&f.subject)?.clone(),
                    predicate: self.predicate_names.get(&f.predicate)?.clone(),
                    object: self.entity_names.get(&f.object)?.clone(),
                    trust: f.trust_f32(),
                    plausibility: f.plausibility_f32(),
                })
            })
            .collect();
        out.sort_by(|a, b| b.confidence().partial_cmp(&a.confidence()).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Lookup (subject, predicate) → best (object, trust, plausibility)
    pub fn get(&self, subject: &str, predicate: &str) -> Option<(String, f32, f32)> {
        let sid = self.entity_ids.get(&subject.to_lowercase())?;
        let pid = self.predicate_ids.get(&predicate.to_lowercase())?;

        self.by_subject
            .get(sid)?
            .iter()
            .filter_map(|&idx| {
                let f = &self.facts[idx];
                if f.predicate == *pid {
                    Some((idx, f.trust_f32() * f.plausibility_f32()))
                } else {
                    None
                }
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .and_then(|(idx, _)| {
                let f = &self.facts[idx];
                Some((
                    self.entity_names.get(&f.object)?.clone(),
                    f.trust_f32(),
                    f.plausibility_f32(),
                ))
            })
    }

    /// Who has (predicate = object)? → [(entity, trust, plausibility)]
    pub fn query(&self, predicate: &str, object: &str) -> Vec<(String, f32, f32)> {
        let pid = match self.predicate_ids.get(&predicate.to_lowercase()) {
            Some(id) => *id,
            None => return vec![],
        };
        let oid = match self.entity_ids.get(&object.to_lowercase()) {
            Some(id) => *id,
            None => return vec![],
        };

        let mut out: Vec<_> = self
            .by_predicate
            .get(&pid)
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|&idx| {
                let f = &self.facts[idx];
                if f.object == oid {
                    Some((
                        self.entity_names.get(&f.subject)?.clone(),
                        f.trust_f32(),
                        f.plausibility_f32(),
                    ))
                } else {
                    None
                }
            })
            .collect();
        out.sort_by(|a, b| (b.1 * b.2).partial_cmp(&(a.1 * a.2)).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Reverse: what points to entity via predicate?
    pub fn reverse(&self, object: &str, predicate: &str) -> Vec<(String, f32, f32)> {
        let oid = match self.entity_ids.get(&object.to_lowercase()) {
            Some(id) => *id,
            None => return vec![],
        };
        let pid = match self.predicate_ids.get(&predicate.to_lowercase()) {
            Some(id) => *id,
            None => return vec![],
        };

        let mut out: Vec<_> = self
            .by_object
            .get(&oid)
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|&idx| {
                let f = &self.facts[idx];
                if f.predicate == pid {
                    Some((
                        self.entity_names.get(&f.subject)?.clone(),
                        f.trust_f32(),
                        f.plausibility_f32(),
                    ))
                } else {
                    None
                }
            })
            .collect();
        out.sort_by(|a, b| (b.1 * b.2).partial_cmp(&(a.1 * a.2)).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Query with minimum confidence threshold
    pub fn query_min_confidence(
        &self,
        predicate: &str,
        object: &str,
        min_t: f32,
        min_p: f32,
    ) -> Vec<(String, f32, f32)> {
        self.query(predicate, object)
            .into_iter()
            .filter(|(_, t, p)| *t >= min_t && *p >= min_p)
            .collect()
    }

    /// Graph neighbors within N hops
    pub fn neighbors(&self, name: &str, depth: u8) -> Vec<FactWithNames> {
        let start = match self.entity_ids.get(&name.to_lowercase()) {
            Some(id) => *id,
            None => return vec![],
        };

        let mut visited = std::collections::HashSet::new();
        let mut frontier = vec![start];
        let mut out = Vec::new();

        for _ in 0..depth {
            let mut next = Vec::new();
            for &eid in &frontier {
                if !visited.insert(eid) {
                    continue;
                }
                if let Some(indices) = self.by_subject.get(&eid) {
                    for &idx in indices {
                        let f = &self.facts[idx];
                        if let (Some(sn), Some(pn), Some(on)) = (
                            self.entity_names.get(&f.subject),
                            self.predicate_names.get(&f.predicate),
                            self.entity_names.get(&f.object),
                        ) {
                            out.push(FactWithNames {
                                subject: sn.clone(),
                                predicate: pn.clone(),
                                object: on.clone(),
                                trust: f.trust_f32(),
                                plausibility: f.plausibility_f32(),
                            });
                            next.push(f.object);
                        }
                    }
                }
            }
            frontier = next;
        }
        out
    }

    /// Convert all facts about an entity into QOR Statements
    pub fn to_qor_facts(&self, name: &str) -> Vec<Statement> {
        self.about(name)
            .into_iter()
            .map(|f| f.to_statement())
            .collect()
    }

    /// Lookup a specific (subject, predicate) and return as QOR Statement
    pub fn lookup_statement(
        &self,
        subject: &str,
        predicate: &str,
    ) -> Option<Statement> {
        let (object, trust, plausibility) = self.get(subject, predicate)?;
        Some(FactWithNames {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object,
            trust,
            plausibility,
        }
        .to_statement())
    }

    /// Human-readable stats
    pub fn stats(&self) -> String {
        let live = self.facts.len() - self.deleted.len();
        format!(
            "{} facts ({:.1} MB) | {} entities | {} predicates",
            live,
            (live * 12) as f64 / (1024.0 * 1024.0),
            self.entity_names.len(),
            self.predicate_names.len(),
        )
    }

    // ═══════════════════════════════════════════════════════════
    // Mutations — CRUD for a living graph
    // ═══════════════════════════════════════════════════════════

    /// Register a new entity name, returning its ID.
    /// If already registered, returns existing ID.
    pub fn register_entity(&mut self, name: &str) -> u32 {
        let lower = name.to_lowercase();
        if let Some(&id) = self.entity_ids.get(&lower) {
            return id;
        }
        let id = self.next_entity;
        self.next_entity += 1;
        self.entity_names.insert(id, name.to_string());
        self.entity_ids.insert(lower, id);
        id
    }

    /// Register a new predicate name, returning its ID.
    /// If already registered, returns existing ID.
    pub fn register_predicate(&mut self, name: &str) -> u16 {
        let lower = name.to_lowercase();
        if let Some(&id) = self.predicate_ids.get(&lower) {
            return id;
        }
        let id = self.next_predicate;
        self.next_predicate += 1;
        self.predicate_names.insert(id, name.to_string());
        self.predicate_ids.insert(lower, id);
        id
    }

    /// Upsert by IDs: if (subject, predicate) exists, update object + trust.
    /// If not, insert new fact. Returns the fact index.
    pub fn upsert(&mut self, subject: u32, predicate: u16, object: u32, trust: u8, plausibility: u8) -> usize {
        // Find existing fact with same (subject, predicate)
        if let Some(indices) = self.by_subject.get(&subject) {
            for &idx in indices {
                if self.deleted.contains(&idx) {
                    continue;
                }
                let f = &self.facts[idx];
                if f.predicate == predicate {
                    // Found existing — update in place
                    let old_object = self.facts[idx].object;
                    self.facts[idx].object = object;
                    self.facts[idx].trust = trust;
                    self.facts[idx].plausibility = plausibility;
                    // Update object index if object changed
                    if old_object != object {
                        if let Some(obj_list) = self.by_object.get_mut(&old_object) {
                            obj_list.retain(|&i| i != idx);
                        }
                        self.by_object.entry(object).or_default().push(idx);
                    }
                    return idx;
                }
            }
        }
        // Not found — insert new
        self.insert(subject, predicate, object, trust, plausibility)
    }

    /// Upsert by names: register entities/predicates as needed, then upsert.
    pub fn upsert_named(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
        trust: f32,
        plausibility: f32,
    ) -> usize {
        let sid = self.register_entity(subject);
        let pid = self.register_predicate(predicate);
        let oid = self.register_entity(object);
        self.upsert(sid, pid, oid, (trust.clamp(0.0, 1.0) * 255.0) as u8, (plausibility.clamp(0.0, 1.0) * 255.0) as u8)
    }

    /// Insert a new fact (always creates, no dedup). Returns fact index.
    pub fn insert(&mut self, subject: u32, predicate: u16, object: u32, trust: u8, plausibility: u8) -> usize {
        let idx = self.facts.len();
        self.facts.push(Fact { subject, predicate, object, trust, plausibility });
        self.by_predicate.entry(predicate).or_default().push(idx);
        self.by_subject.entry(subject).or_default().push(idx);
        self.by_object.entry(object).or_default().push(idx);
        idx
    }

    /// Remove all facts matching exact (subject, predicate, object). Returns count removed.
    pub fn remove(&mut self, subject: u32, predicate: u16, object: u32) -> usize {
        let indices: Vec<usize> = self.by_subject
            .get(&subject)
            .unwrap_or(&vec![])
            .iter()
            .filter(|&&idx| {
                !self.deleted.contains(&idx)
                    && self.facts[idx].predicate == predicate
                    && self.facts[idx].object == object
            })
            .copied()
            .collect();

        for &idx in &indices {
            self.tombstone(idx);
        }
        indices.len()
    }

    /// Remove all facts matching (subject, predicate) regardless of object. Returns count removed.
    pub fn remove_by_pred(&mut self, subject: u32, predicate: u16) -> usize {
        let indices: Vec<usize> = self.by_subject
            .get(&subject)
            .unwrap_or(&vec![])
            .iter()
            .filter(|&&idx| {
                !self.deleted.contains(&idx) && self.facts[idx].predicate == predicate
            })
            .copied()
            .collect();

        for &idx in &indices {
            self.tombstone(idx);
        }
        indices.len()
    }

    /// Remove by names. Returns count removed.
    pub fn remove_named(&mut self, subject: &str, predicate: &str, object: &str) -> usize {
        let sid = match self.entity_ids.get(&subject.to_lowercase()) {
            Some(&id) => id,
            None => return 0,
        };
        let pid = match self.predicate_ids.get(&predicate.to_lowercase()) {
            Some(&id) => id,
            None => return 0,
        };
        let oid = match self.entity_ids.get(&object.to_lowercase()) {
            Some(&id) => id,
            None => return 0,
        };
        self.remove(sid, pid, oid)
    }

    /// Update trust/plausibility on an existing (subject, predicate, object). Returns true if found.
    pub fn update_confidence(
        &mut self,
        subject: u32,
        predicate: u16,
        object: u32,
        trust: u8,
        plausibility: u8,
    ) -> bool {
        if let Some(indices) = self.by_subject.get(&subject) {
            for &idx in indices {
                if self.deleted.contains(&idx) {
                    continue;
                }
                let f = &self.facts[idx];
                if f.predicate == predicate && f.object == object {
                    self.facts[idx].trust = trust;
                    self.facts[idx].plausibility = plausibility;
                    return true;
                }
            }
        }
        false
    }

    /// Mark a fact index as deleted, remove from all indexes.
    fn tombstone(&mut self, idx: usize) {
        if !self.deleted.insert(idx) {
            return; // already deleted
        }
        let f = &self.facts[idx];
        let subj = f.subject;
        let pred = f.predicate;
        let obj = f.object;
        if let Some(list) = self.by_subject.get_mut(&subj) {
            list.retain(|&i| i != idx);
        }
        if let Some(list) = self.by_predicate.get_mut(&pred) {
            list.retain(|&i| i != idx);
        }
        if let Some(list) = self.by_object.get_mut(&obj) {
            list.retain(|&i| i != idx);
        }
    }

    /// Save the full graph to disk. Writes:
    /// - `triples.bin` (all live facts, merged from all sources)
    /// - `entities.tsv` + `predicates.tsv` (updated mappings)
    /// Old source-specific `*_triples.bin` files are removed to prevent
    /// duplicate loading on next `load()`.
    pub fn save(&self, dir: &Path) -> io::Result<()> {
        std::fs::create_dir_all(dir)?;

        // Write all live facts to triples.bin
        let mut f = BufWriter::new(File::create(dir.join("triples.bin"))?);
        for (idx, fact) in self.facts.iter().enumerate() {
            if self.deleted.contains(&idx) {
                continue;
            }
            f.write_all(&fact.subject.to_le_bytes())?;
            f.write_all(&fact.predicate.to_le_bytes())?;
            f.write_all(&fact.object.to_le_bytes())?;
            f.write_all(&[fact.trust])?;
            f.write_all(&[fact.plausibility])?;
        }
        f.flush()?;

        // Remove old source-specific triple files (now merged into triples.bin)
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.ends_with("_triples.bin") {
                        let _ = std::fs::remove_file(entry.path());
                    }
                }
            }
        }

        // Write entities.tsv
        let mut ef = BufWriter::new(File::create(dir.join("entities.tsv"))?);
        writeln!(ef, "id\tlabel")?;
        let mut entities: Vec<(&u32, &String)> = self.entity_names.iter().collect();
        entities.sort_by_key(|(&id, _)| id);
        for (&id, name) in &entities {
            writeln!(ef, "{}\t{}", id, name)?;
        }
        ef.flush()?;

        // Write predicates.tsv
        let mut pf = BufWriter::new(File::create(dir.join("predicates.tsv"))?);
        writeln!(pf, "id\tlabel")?;
        let mut preds: Vec<(&u16, &String)> = self.predicate_names.iter().collect();
        preds.sort_by_key(|(&id, _)| id);
        for (&id, name) in &preds {
            writeln!(pf, "{}\t{}", id, name)?;
        }
        pf.flush()?;

        Ok(())
    }

    /// Number of live (non-deleted) facts
    pub fn live_count(&self) -> usize {
        self.facts.len() - self.deleted.len()
    }
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Helper: write a minimal binary triple file + tsv mappings for testing
    fn create_test_kb(dir: &Path) {
        // entities.tsv
        let mut f = File::create(dir.join("entities.tsv")).unwrap();
        writeln!(f, "id\tlabel").unwrap();
        writeln!(f, "1\tAlbert Einstein").unwrap();
        writeln!(f, "2\tphysicist").unwrap();
        writeln!(f, "3\tGermany").unwrap();
        writeln!(f, "4\tE=mc²").unwrap();
        writeln!(f, "5\tMarie Curie").unwrap();
        writeln!(f, "6\tPoland").unwrap();
        writeln!(f, "7\tNobel Prize").unwrap();

        // predicates.tsv
        let mut f = File::create(dir.join("predicates.tsv")).unwrap();
        writeln!(f, "id\tlabel").unwrap();
        writeln!(f, "1\toccupation").unwrap();
        writeln!(f, "2\tcountry").unwrap();
        writeln!(f, "3\tnotable_work").unwrap();
        writeln!(f, "4\taward").unwrap();

        // triples.bin — 4 facts
        let mut f = File::create(dir.join("triples.bin")).unwrap();
        // Einstein occupation physicist  <0.95, 0.95>
        write_triple(&mut f, 1, 1, 2, 0.95, 0.95);
        // Einstein country Germany  <0.95, 0.95>
        write_triple(&mut f, 1, 2, 3, 0.95, 0.95);
        // Einstein notable_work E=mc²  <0.98, 0.98>
        write_triple(&mut f, 1, 3, 4, 0.98, 0.98);
        // Marie Curie country Poland <0.95, 0.95>
        write_triple(&mut f, 5, 2, 6, 0.95, 0.95);
        // Marie Curie occupation physicist <0.95, 0.95>
        write_triple(&mut f, 5, 1, 2, 0.95, 0.95);
        // Marie Curie award Nobel Prize <0.98, 0.98>
        write_triple(&mut f, 5, 4, 7, 0.98, 0.98);
    }

    fn write_triple(f: &mut File, subj: u32, pred: u16, obj: u32, trust: f32, plaus: f32) {
        f.write_all(&subj.to_le_bytes()).unwrap();
        f.write_all(&pred.to_le_bytes()).unwrap();
        f.write_all(&obj.to_le_bytes()).unwrap();
        f.write_all(&[(trust * 255.0) as u8]).unwrap();
        f.write_all(&[(plaus * 255.0) as u8]).unwrap();
    }

    #[test]
    fn test_kb_load_and_stats() {
        let dir = std::env::temp_dir().join("qor_kb_test_load");
        let _ = std::fs::create_dir_all(&dir);
        create_test_kb(&dir);

        let kb = KnowledgeBase::load(&dir).unwrap();
        assert_eq!(kb.len(), 6);
        assert_eq!(kb.entity_names.len(), 7);
        assert_eq!(kb.predicate_names.len(), 4);

        let stats = kb.stats();
        assert!(stats.contains("6 facts"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_kb_about() {
        let dir = std::env::temp_dir().join("qor_kb_test_about");
        let _ = std::fs::create_dir_all(&dir);
        create_test_kb(&dir);

        let kb = KnowledgeBase::load(&dir).unwrap();
        let facts = kb.about("Albert Einstein");
        assert_eq!(facts.len(), 3);
        // Should be sorted by confidence (highest first)
        assert!(facts[0].confidence() >= facts[1].confidence());

        // Check specific facts exist
        let preds: Vec<&str> = facts.iter().map(|f| f.predicate.as_str()).collect();
        assert!(preds.contains(&"occupation"));
        assert!(preds.contains(&"country"));
        assert!(preds.contains(&"notable_work"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_kb_get() {
        let dir = std::env::temp_dir().join("qor_kb_test_get");
        let _ = std::fs::create_dir_all(&dir);
        create_test_kb(&dir);

        let kb = KnowledgeBase::load(&dir).unwrap();

        // Exact lookup
        let result = kb.get("albert einstein", "occupation");
        assert!(result.is_some());
        let (obj, t, p) = result.unwrap();
        assert_eq!(obj, "physicist");
        assert!(t > 0.9);
        assert!(p > 0.9);

        // Missing lookup
        let result = kb.get("nobody", "occupation");
        assert!(result.is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_kb_query() {
        let dir = std::env::temp_dir().join("qor_kb_test_query");
        let _ = std::fs::create_dir_all(&dir);
        create_test_kb(&dir);

        let kb = KnowledgeBase::load(&dir).unwrap();

        // Who is a physicist?
        let results = kb.query("occupation", "physicist");
        assert_eq!(results.len(), 2); // Einstein + Curie

        // Who is from Germany?
        let results = kb.query("country", "Germany");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "Albert Einstein");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_kb_reverse() {
        let dir = std::env::temp_dir().join("qor_kb_test_reverse");
        let _ = std::fs::create_dir_all(&dir);
        create_test_kb(&dir);

        let kb = KnowledgeBase::load(&dir).unwrap();

        // Who won the Nobel Prize?
        let results = kb.reverse("Nobel Prize", "award");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "Marie Curie");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_kb_to_qor_facts() {
        let dir = std::env::temp_dir().join("qor_kb_test_qor");
        let _ = std::fs::create_dir_all(&dir);
        create_test_kb(&dir);

        let kb = KnowledgeBase::load(&dir).unwrap();
        let stmts = kb.to_qor_facts("Albert Einstein");
        assert_eq!(stmts.len(), 3);

        // Each should be a Statement::Fact
        for stmt in &stmts {
            match stmt {
                Statement::Fact { tv, .. } => {
                    assert!(tv.is_some());
                }
                _ => panic!("expected Statement::Fact"),
            }
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_kb_empty() {
        let kb = KnowledgeBase::new();
        assert!(kb.is_empty());
        assert_eq!(kb.len(), 0);
        assert!(kb.about("anything").is_empty());
        assert!(kb.get("a", "b").is_none());
        assert!(kb.query("a", "b").is_empty());
    }

    #[test]
    fn test_kb_case_insensitive() {
        let dir = std::env::temp_dir().join("qor_kb_test_case");
        let _ = std::fs::create_dir_all(&dir);
        create_test_kb(&dir);

        let kb = KnowledgeBase::load(&dir).unwrap();

        // Case insensitive lookups
        assert!(!kb.about("albert einstein").is_empty());
        assert!(!kb.about("ALBERT EINSTEIN").is_empty());
        assert!(kb.get("Albert Einstein", "OCCUPATION").is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_kb_neighbors() {
        let dir = std::env::temp_dir().join("qor_kb_test_neighbors");
        let _ = std::fs::create_dir_all(&dir);
        create_test_kb(&dir);

        let kb = KnowledgeBase::load(&dir).unwrap();
        let neighbors = kb.neighbors("Albert Einstein", 1);
        assert_eq!(neighbors.len(), 3); // 3 direct connections

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ═══════════════════════════════════════════════════════════
    // Mutation tests
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_kb_register_entity() {
        let mut kb = KnowledgeBase::new();
        let id1 = kb.register_entity("Bitcoin");
        let id2 = kb.register_entity("Ethereum");
        let id3 = kb.register_entity("bitcoin"); // same as id1 (case insensitive)
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 1);
        assert_eq!(kb.entity_names.len(), 2);
    }

    #[test]
    fn test_kb_register_predicate() {
        let mut kb = KnowledgeBase::new();
        let p1 = kb.register_predicate("price");
        let p2 = kb.register_predicate("volume");
        let p3 = kb.register_predicate("Price"); // same
        assert_eq!(p1, 1);
        assert_eq!(p2, 2);
        assert_eq!(p3, 1);
    }

    #[test]
    fn test_kb_upsert_insert() {
        let mut kb = KnowledgeBase::new();
        let btc = kb.register_entity("btc");
        let price = kb.register_predicate("price");
        let v40k = kb.register_entity("40000");

        let idx = kb.upsert(btc, price, v40k, 242, 242);
        assert_eq!(kb.live_count(), 1);
        assert_eq!(idx, 0);

        // Verify via query
        let result = kb.get("btc", "price");
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, "40000");
    }

    #[test]
    fn test_kb_upsert_update() {
        let mut kb = KnowledgeBase::new();
        let btc = kb.register_entity("btc");
        let price = kb.register_predicate("price");
        let v40k = kb.register_entity("40000");
        let v42k = kb.register_entity("42000");

        // Insert initial price
        kb.upsert(btc, price, v40k, 242, 242);
        assert_eq!(kb.live_count(), 1);

        // Update — same (subject, predicate), new object
        let idx = kb.upsert(btc, price, v42k, 250, 250);
        assert_eq!(kb.live_count(), 1); // still 1 fact, not 2
        assert_eq!(idx, 0); // same slot

        // Verify updated value
        let result = kb.get("btc", "price");
        assert_eq!(result.unwrap().0, "42000");
    }

    #[test]
    fn test_kb_upsert_named() {
        let mut kb = KnowledgeBase::new();
        kb.upsert_named("btc", "price", "40000", 0.95, 0.95);
        assert_eq!(kb.live_count(), 1);

        // Update via named
        kb.upsert_named("btc", "price", "42000", 0.98, 0.98);
        assert_eq!(kb.live_count(), 1);
        assert_eq!(kb.get("btc", "price").unwrap().0, "42000");
    }

    #[test]
    fn test_kb_remove() {
        let dir = std::env::temp_dir().join("qor_kb_test_remove");
        let _ = std::fs::create_dir_all(&dir);
        create_test_kb(&dir);

        let mut kb = KnowledgeBase::load(&dir).unwrap();
        assert_eq!(kb.live_count(), 6);

        // Remove Einstein's occupation
        let sid = *kb.entity_ids.get("albert einstein").unwrap();
        let pid = *kb.predicate_ids.get("occupation").unwrap();
        let oid = *kb.entity_ids.get("physicist").unwrap();
        let removed = kb.remove(sid, pid, oid);
        assert_eq!(removed, 1);
        assert_eq!(kb.live_count(), 5);

        // Verify it's gone from queries
        let facts = kb.about("Albert Einstein");
        assert_eq!(facts.len(), 2); // country + notable_work remain

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_kb_remove_named() {
        let mut kb = KnowledgeBase::new();
        kb.upsert_named("btc", "price", "40000", 0.95, 0.95);
        kb.upsert_named("btc", "volume", "1000000", 0.90, 0.90);
        assert_eq!(kb.live_count(), 2);

        let removed = kb.remove_named("btc", "price", "40000");
        assert_eq!(removed, 1);
        assert_eq!(kb.live_count(), 1);

        // volume still there
        assert!(kb.get("btc", "volume").is_some());
        assert!(kb.get("btc", "price").is_none());
    }

    #[test]
    fn test_kb_remove_by_pred() {
        let mut kb = KnowledgeBase::new();
        let btc = kb.register_entity("btc");
        let price = kb.register_predicate("price");
        let v1 = kb.register_entity("40000");
        let v2 = kb.register_entity("42000");

        // Insert two price facts (via raw insert, not upsert)
        kb.insert(btc, price, v1, 200, 200);
        kb.insert(btc, price, v2, 250, 250);
        assert_eq!(kb.live_count(), 2);

        // Remove all (btc, price, *) facts
        let removed = kb.remove_by_pred(btc, price);
        assert_eq!(removed, 2);
        assert_eq!(kb.live_count(), 0);
    }

    #[test]
    fn test_kb_update_confidence() {
        let mut kb = KnowledgeBase::new();
        let a = kb.register_entity("a");
        let r = kb.register_predicate("r");
        let b = kb.register_entity("b");

        kb.insert(a, r, b, 200, 200);
        assert!(kb.update_confidence(a, r, b, 250, 250));

        // Check updated values
        let (_, t, p) = kb.get("a", "r").unwrap();
        assert!(t > 0.95);
        assert!(p > 0.95);
    }

    #[test]
    fn test_kb_save_and_reload() {
        let dir = std::env::temp_dir().join("qor_kb_test_save");
        let _ = std::fs::create_dir_all(&dir);

        // Build a KB, mutate it, save
        let mut kb = KnowledgeBase::new();
        kb.upsert_named("btc", "price", "40000", 0.95, 0.95);
        kb.upsert_named("eth", "price", "3000", 0.90, 0.90);
        kb.upsert_named("btc", "volume", "1000000", 0.85, 0.85);

        // Update BTC price
        kb.upsert_named("btc", "price", "42000", 0.98, 0.98);

        // Remove ETH price
        kb.remove_named("eth", "price", "3000");

        assert_eq!(kb.live_count(), 2); // btc price + btc volume

        // Save
        kb.save(&dir).unwrap();

        // Reload and verify
        let kb2 = KnowledgeBase::load(&dir).unwrap();
        assert_eq!(kb2.live_count(), 2);
        assert_eq!(kb2.get("btc", "price").unwrap().0, "42000");
        assert!(kb2.get("btc", "volume").is_some());
        assert!(kb2.get("eth", "price").is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_kb_save_merges_source_files() {
        let dir = std::env::temp_dir().join("qor_kb_test_merge");
        let _ = std::fs::create_dir_all(&dir);

        // Create a source-specific triple file (like oeis_triples.bin)
        create_test_kb(&dir);
        // Rename triples.bin to oeis_triples.bin to simulate source file
        std::fs::rename(dir.join("triples.bin"), dir.join("oeis_triples.bin")).unwrap();

        // Load, mutate, save
        let mut kb = KnowledgeBase::load(&dir).unwrap();
        assert_eq!(kb.live_count(), 6);
        kb.upsert_named("btc", "price", "40000", 0.95, 0.95);
        assert_eq!(kb.live_count(), 7);

        kb.save(&dir).unwrap();

        // oeis_triples.bin should be gone (merged into triples.bin)
        assert!(!dir.join("oeis_triples.bin").exists());
        assert!(dir.join("triples.bin").exists());

        // Reload — should have all 7 facts
        let kb2 = KnowledgeBase::load(&dir).unwrap();
        assert_eq!(kb2.live_count(), 7);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
