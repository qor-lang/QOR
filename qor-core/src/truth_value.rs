use std::fmt;

/// A PLN-inspired truth value with two components.
///
/// - **strength**: how likely this is true (0.0 to 1.0)
/// - **confidence**: how much evidence supports this strength (0.0 to 1.0)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TruthValue {
    pub strength: f64,
    pub confidence: f64,
}

/// Default confidence when only strength is provided.
pub const DEFAULT_CONFIDENCE: f64 = 0.9;

impl TruthValue {
    pub fn new(strength: f64, confidence: f64) -> Self {
        TruthValue {
            strength: strength.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Create a truth value with default confidence (0.9).
    pub fn from_strength(strength: f64) -> Self {
        TruthValue::new(strength, DEFAULT_CONFIDENCE)
    }

    /// Default truth value for facts with no explicit truth value: <1.0, 0.9>
    pub fn default_fact() -> Self {
        TruthValue::new(1.0, DEFAULT_CONFIDENCE)
    }

    // -- PLN-inspired formulas --

    /// AND: both must be true.
    /// s = s1 * s2, c = min(c1, c2)
    pub fn and(&self, other: &TruthValue) -> TruthValue {
        TruthValue::new(
            self.strength * other.strength,
            self.confidence.min(other.confidence),
        )
    }

    /// OR: at least one is true.
    /// s = s1 + s2 - s1*s2, c = min(c1, c2)
    pub fn or(&self, other: &TruthValue) -> TruthValue {
        TruthValue::new(
            self.strength + other.strength - self.strength * other.strength,
            self.confidence.min(other.confidence),
        )
    }

    /// Deduction (implication through a rule).
    /// s = s1 * s2, c = c1 * c2 * 0.9
    /// PLN Book §1.4, p.15
    pub fn deduction(&self, other: &TruthValue) -> TruthValue {
        TruthValue::new(
            self.strength * other.strength,
            self.confidence * other.confidence * 0.9,
        )
    }

    /// Negation.
    /// s = 1 - s, c = c (unchanged)
    pub fn negation(&self) -> TruthValue {
        TruthValue::new(1.0 - self.strength, self.confidence)
    }

    /// Revision: merge two truth values for the same fact (new evidence).
    /// This is how QOR learns — confidence ALWAYS increases.
    /// PLN Book §5.10.2, p.116
    ///
    /// s_new = (s1*c1 + s2*c2) / (c1 + c2)
    /// c_new = c1 + c2 - c1*c2
    pub fn revision(&self, other: &TruthValue) -> TruthValue {
        let c_sum = self.confidence + other.confidence;
        if c_sum == 0.0 {
            return TruthValue::new(0.0, 0.0);
        }
        let s_new = (self.strength * self.confidence + other.strength * other.confidence) / c_sum;
        let c_new = self.confidence + other.confidence - self.confidence * other.confidence;
        TruthValue::new(s_new, c_new)
    }

    /// Inversion: if we know A→B, infer B→A.
    /// PLN Book p.11 — derives from Bayes' rule.
    /// Ref: PLN-main/lib_pln.metta lines 208-211
    ///
    /// s_result = s_AB  (unchanged)
    /// c_result = c_B * c_AB * 0.6  (penalty factor)
    ///
    /// `other` is the truth value of node B (the target).
    pub fn inversion(&self, target_tv: &TruthValue) -> TruthValue {
        TruthValue::new(
            self.strength,
            target_tv.confidence * self.confidence * 0.6,
        )
    }

    /// Induction: two rules share a body term → infer relationship.
    /// B→A + B→C ⇒ A↔C
    /// s = s1 * s2, c = w2c(min(c2w(c1), c2w(c2)))
    /// PLN Book §5.1
    pub fn induction(&self, other: &TruthValue) -> TruthValue {
        let s = self.strength * other.strength;
        let w = TruthValue::c2w(self.confidence).min(TruthValue::c2w(other.confidence));
        TruthValue::new(s, TruthValue::w2c(w))
    }

    /// Abduction: two rules share a head term → infer relationship.
    /// A→B + C→B ⇒ A↔C
    /// Same formula as induction (the difference is semantic, not mathematical
    /// in simplified PLN).
    pub fn abduction(&self, other: &TruthValue) -> TruthValue {
        let s = self.strength * other.strength;
        let w = TruthValue::c2w(self.confidence).min(TruthValue::c2w(other.confidence));
        TruthValue::new(s, TruthValue::w2c(w))
    }

    // -- Helper conversions (used by inference engine) --

    /// Confidence → weight (evidence count).
    /// w = c / (1 - c)
    /// Used internally by PLN formulas.
    pub fn c2w(c: f64) -> f64 {
        if c >= 1.0 {
            return f64::MAX;
        }
        c / (1.0 - c)
    }

    /// Weight → confidence.
    /// c = w / (w + 1)
    pub fn w2c(w: f64) -> f64 {
        w / (w + 1.0)
    }
}

impl Default for TruthValue {
    fn default() -> Self {
        TruthValue::default_fact()
    }
}

impl fmt::Display for TruthValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{:.2}, {:.2}>", self.strength, self.confidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 0.01
    }

    #[test]
    fn test_from_strength_uses_default_confidence() {
        let tv = TruthValue::from_strength(0.99);
        assert!(approx(tv.strength, 0.99));
        assert!(approx(tv.confidence, 0.9));
    }

    #[test]
    fn test_and() {
        let a = TruthValue::new(0.8, 0.9);
        let b = TruthValue::new(0.6, 0.7);
        let r = a.and(&b);
        assert!(approx(r.strength, 0.48)); // 0.8 * 0.6
        assert!(approx(r.confidence, 0.7)); // min(0.9, 0.7)
    }

    #[test]
    fn test_or() {
        let a = TruthValue::new(0.8, 0.9);
        let b = TruthValue::new(0.6, 0.7);
        let r = a.or(&b);
        assert!(approx(r.strength, 0.92)); // 0.8 + 0.6 - 0.48
        assert!(approx(r.confidence, 0.7));
    }

    #[test]
    fn test_deduction() {
        let a = TruthValue::new(0.92, 0.90);
        let b = TruthValue::new(0.85, 0.90);
        let r = a.deduction(&b);
        assert!(approx(r.strength, 0.782)); // 0.92 * 0.85
        assert!(approx(r.confidence, 0.729)); // 0.9 * 0.9 * 0.9
    }

    #[test]
    fn test_negation() {
        let a = TruthValue::new(0.8, 0.9);
        let r = a.negation();
        assert!(approx(r.strength, 0.2));
        assert!(approx(r.confidence, 0.9));
    }

    #[test]
    fn test_revision_merges_evidence() {
        // First observation: weak evidence
        let old = TruthValue::new(0.70, 0.40);
        // Second observation: strong evidence
        let new = TruthValue::new(0.95, 0.80);
        let r = old.revision(&new);
        // s = (0.70*0.40 + 0.95*0.80) / (0.40+0.80) = 0.867
        assert!(approx(r.strength, 0.867));
        // c = 0.40 + 0.80 - 0.40*0.80 = 0.88
        assert!(approx(r.confidence, 0.88));
    }

    #[test]
    fn test_revision_confidence_always_grows() {
        let a = TruthValue::new(0.5, 0.3);
        let b = TruthValue::new(0.5, 0.3);
        let r = a.revision(&b);
        assert!(r.confidence > a.confidence);
    }

    #[test]
    fn test_revision_many_times_converges() {
        let mut tv = TruthValue::new(0.8, 0.1);
        let evidence = TruthValue::new(0.8, 0.1);
        for _ in 0..100 {
            tv = tv.revision(&evidence);
        }
        // After 100 revisions, confidence should be near 1.0
        assert!(tv.confidence > 0.99);
    }

    #[test]
    fn test_display() {
        let tv = TruthValue::new(0.99, 0.90);
        assert_eq!(tv.to_string(), "<0.99, 0.90>");
    }

    #[test]
    fn test_clamp() {
        let tv = TruthValue::new(1.5, -0.3);
        assert!(approx(tv.strength, 1.0));
        assert!(approx(tv.confidence, 0.0));
    }

    #[test]
    fn test_inversion() {
        // A→B with strength 0.8, confidence 0.9
        let ab = TruthValue::new(0.8, 0.9);
        // Target node B with confidence 0.85
        let b = TruthValue::new(0.7, 0.85);
        let r = ab.inversion(&b);
        // s stays the same
        assert!(approx(r.strength, 0.8));
        // c = 0.85 * 0.9 * 0.6 = 0.459
        assert!(approx(r.confidence, 0.459));
    }

    #[test]
    fn test_w2c_and_c2w_roundtrip() {
        let c = 0.8;
        let w = TruthValue::c2w(c);
        let c_back = TruthValue::w2c(w);
        assert!(approx(c, c_back));
    }

    #[test]
    fn test_w2c_values() {
        // w=1 → c=0.5
        assert!(approx(TruthValue::w2c(1.0), 0.5));
        // w=9 → c=0.9
        assert!(approx(TruthValue::w2c(9.0), 0.9));
        // w=0 → c=0
        assert!(approx(TruthValue::w2c(0.0), 0.0));
    }

    #[test]
    fn test_induction() {
        // B→A: s=0.8, c=0.9
        // B→C: s=0.7, c=0.8
        let ba = TruthValue::new(0.8, 0.9);
        let bc = TruthValue::new(0.7, 0.8);
        let r = ba.induction(&bc);
        // s = 0.8 * 0.7 = 0.56
        assert!(approx(r.strength, 0.56));
        // c = w2c(min(c2w(0.9), c2w(0.8)))
        // c2w(0.9)=9, c2w(0.8)=4, min=4, w2c(4)=0.8
        assert!(approx(r.confidence, 0.8));
    }

    #[test]
    fn test_abduction() {
        let ab = TruthValue::new(0.9, 0.85);
        let cb = TruthValue::new(0.8, 0.80);
        let r = ab.abduction(&cb);
        // s = 0.9 * 0.8 = 0.72
        assert!(approx(r.strength, 0.72));
        // c = w2c(min(c2w(0.85), c2w(0.80))) = w2c(4) = 0.8
        assert!(approx(r.confidence, 0.8));
    }

    #[test]
    fn test_induction_conservative() {
        // Weak evidence should produce weak conclusions
        let a = TruthValue::new(0.9, 0.3);
        let b = TruthValue::new(0.9, 0.3);
        let r = a.induction(&b);
        assert!(r.confidence < 0.5, "induction with weak evidence should be low confidence");
    }
}
