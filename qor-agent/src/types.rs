use qor_core::neuron::Statement;

/// A generic data item to process. Could be a puzzle, a patient, a trade — anything.
/// The DNA rules determine what to do with the facts.
pub struct DataItem {
    pub id: String,
    pub facts: Vec<Statement>,
    /// Expected output facts (for scoring). None if unknown.
    pub expected: Option<Vec<(String, Vec<String>)>>,
}

#[derive(Debug, Default)]
pub struct AgentStats {
    pub attempted: usize,
    pub correct:   usize,
    pub wrong:     usize,
    pub no_answer: usize,
    pub learned:   usize,
    pub skipped:   usize,
}

impl std::fmt::Display for AgentStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{} correct | {} wrong | {} no-answer | {} learned | {} skipped",
            self.correct, self.attempted, self.wrong, self.no_answer, self.learned, self.skipped)
    }
}
