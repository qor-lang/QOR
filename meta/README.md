# Meta Domain Files

QOR's meta-cognitive layer. Each `.qor` file is a **lens** — a different way of seeing the same problem. The swarm loads each file into a separate worker, and they race to solve the problem from their unique perspective.

## Architecture

Every strategy file has two sections:

1. **Knowledge facts** — domain formulas, laws, constants (read-only reference library)
2. **Bridge rules** — active reasoning rules that match observation predicates and produce domain-specific insight facts

Same observation, different lens:

```
(consistent-remap $from $to)  fires in:
  math.qor        → (math-insight bijection)
  physics.qor     → (physics-insight phase-transition)
  chemistry.qor   → (chem-insight transmutation)
  biology.qor     → (bio-insight mutation)
  vedic_science.qor → (vedic-insight karma-transform)
```

## Roles

### Strategy Workers (11 files — one per swarm worker)

| File | Domain | Knowledge | Bridge Rules Produce |
|------|--------|-----------|---------------------|
| `genesis.qor` | Creation | Creativity hints (tiling, color-remap, reflect, extraction), fallbacks (identity, shift) | `(genesis-active)`, `(genesis-hint ...)` |
| `math.qor` | Mathematics | ~250 formulas: arithmetic, algebra, number theory, combinatorics, geometry, trig, calculus, linear algebra, probability, logic, set theory | `(math-insight ...)`: symmetry, bijection, scaling, translation |
| `physics.qor` | Physics | ~250 laws: mechanics, thermodynamics, EM, waves/optics, quantum, relativity, nuclear | `(physics-insight ...)`: conservation, displacement, phase-transition |
| `chemistry.qor` | Chemistry | ~180 facts: atomic structure, bonding, reactions, thermochem, acids/bases, equilibrium, electrochemistry, kinetics, organic, nuclear | `(chem-insight ...)`: transmutation, catalysis, synthesis, elimination |
| `biology.qor` | Biology | ~170 facts: cell biology, DNA, genetics, evolution, ecology, biochemistry, human body | `(bio-insight ...)`: growth, apoptosis, homeostasis, mutation, differentiation |
| `computer_science.qor` | CS | ~120 facts: complexity/Big-O, data structures, sorting, search, algorithm paradigms, information theory, boolean algebra, AI/ML | `(cs-insight ...)`: compression, filter, map, template-match, interpolation |
| `astronomy_earth.qor` | Astro/Geo | ~130 facts: solar system, stellar evolution, cosmology, geology, atmosphere | `(astro-insight ...)`: orbital motion, expansion, erosion, accretion |
| `general.qor` | General | History, transitivity, symmetry, classification, geographic reasoning, knowledge gaps, KB graph reasoning (formula type hierarchy, computability, core-sequence) | `(general-insight ...)`: transform-detected, size-change |
| `perception.qor` | Attention | Object permanence, figure-ground, gestalt grouping, size constancy, motion, color constancy | `(perception-insight ...)`: salient pattern, perceptual link, categorization |
| `sanskrit.qor` | Language | 49 phonemes, 50+ sandhi rules, noun declension, verb conjugation (10 lakaras, 10 ganas, 25 dhatus), 6 compound types, karaka parsing | `(sanskrit-insight ...)`: sandhi-transform, samasa-compound, vibhakti-transform |
| `vedic_science.qor` | Vedic | 16 sutras (Ekadhikena, Nikhilam, Paravartya...), Pingala binary/Fibonacci, Aryabhata sine table, Madhava infinite series, music theory (22 srutis), Ayurveda, Vastu, Katapayadi encoding | `(vedic-insight ...)`: ekadhikena, nikhilam, paravartya, pingala-binary, dharma-conservation |

### Advisor (applied AFTER the race)

| File | Role | What It Does |
|------|------|-------------|
| `reasoning.qor` | **Advisor** | 25+ strategies from 15+ mathematicians/scientists (Gauss, Euler, Shannon, Ramanujan, Noether, Godel, Turing, Poincare, Grothendieck, Mandelbrot, Kolmogorov, Dijkstra, Penrose, Wittgenstein, Panini). Quantum-like multi-stage search compression. Produces `(genesis-hint)`, `(genesis-skip)`, `(problem-class)`, `(strategy-order)`. |

### Judge (scores candidates)

| File | Role | What It Does |
|------|------|-------------|
| `agi.qor` | **Judge** | 6D state vector: R = (feasibility, novelty, simplicity, ethics, consistency, information). Dempster-Shafer evidence theory. Quantum-inspired operators (superposition, collapse, entanglement). Adaptive threshold. Decoherence for pruning. |

## How the Swarm Race Works

```
Round 1:
  Base session = DNA rules + problem facts (no meta files)

  Worker 1  (genesis.qor)        ──┐
  Worker 2  (math.qor)           ──┤
  Worker 3  (physics.qor)        ──┤
  Worker 4  (chemistry.qor)      ──┤
  Worker 5  (biology.qor)        ──┤── RACE ──→ Best score wins
  Worker 6  (computer_science.qor)─┤
  Worker 7  (astronomy_earth.qor)──┤
  Worker 8  (general.qor)        ──┤
  Worker 9  (perception.qor)     ──┤
  Worker 10 (sanskrit.qor)       ──┤
  Worker 11 (vedic_science.qor)  ──┘

  Winner → reasoning.qor (advisor) → refined hints → further mutation

Rounds 2-5:
  Inject failure memory → skip exhausted strategies → focused search
```

## Observation Predicates (inputs to bridge rules)

These predicates are computed from problem data and trigger bridge rules:

- `obs-consistent` — consistent pattern across examples
- `detected-reflect-h` / `detected-reflect-v` — reflection symmetry detected
- `consistent-remap` — color/value remapping detected
- `selective-removal` — selective deletion pattern
- `detected-shift` — translation/shift pattern
- `marker-driven` — marker-based transformation
- `objects-preserved` — objects maintained across transform
- `color-introduced` — new colors appear in output
- `pattern-rate` — frequency of a pattern
- `co-occur` — co-occurrence of patterns
- `threshold` — threshold-based classification
- `is-a` / `has` / `can` — ontological relations
- `domain` / `meaning` — semantic context
- `dna-edge` / `dna-trait` — DNA graph edges and traits
- `intent` — detected user intent
