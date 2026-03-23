use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

use qor_core::neuron::Neuron;
use qor_core::truth_value::TruthValue;
use qor_runtime::eval::Session;
use qor_bridge::grid::Grid;

/// Python-facing QOR session — wraps the full QOR reasoning engine.
/// Rules are loaded from .qor files, NOT hardcoded in Rust.
#[pyclass]
struct QorSession {
    session: Session,
}

#[pymethods]
impl QorSession {
    #[new]
    fn new() -> Self {
        QorSession {
            session: Session::new(),
        }
    }

    /// Load rules and facts from a .qor file.
    /// e.g. session.load_rules("rules.qor")
    fn load_rules(&mut self, path: &str) -> PyResult<usize> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| PyRuntimeError::new_err(format!("cannot read {}: {}", path, e)))?;
        let results = self.session.exec(&content)
            .map_err(|e| PyRuntimeError::new_err(format!("parse error: {}", e)))?;
        Ok(results.len())
    }

    /// Execute QOR source text directly (facts, rules, queries).
    /// e.g. session.exec('(tile 0 0 1)\n(tile 0 1 2)')
    fn exec(&mut self, source: &str) -> PyResult<Vec<String>> {
        let results = self.session.exec(source)
            .map_err(|e| PyRuntimeError::new_err(format!("parse error: {}", e)))?;
        Ok(results.iter().map(|r| format!("{:?}", r)).collect())
    }

    /// Assert a fact with integer arguments.
    /// e.g. session.assert_fact("tile", [0, 0, 1])
    fn assert_fact(&mut self, predicate: &str, args: Vec<i64>) {
        let mut parts = vec![Neuron::symbol(predicate)];
        for a in args {
            parts.push(Neuron::int_val(a));
        }
        self.session.store_mut().insert(
            Neuron::expression(parts),
            TruthValue::from_strength(0.99),
        );
    }

    /// Assert a fact with int args + a trailing string tag.
    /// e.g. session.assert_fact_s("cnbr", [0, 0, 0, 1], "same", [2])
    /// produces: (cnbr 0 0 0 1 same 2)
    fn assert_fact_mixed(&mut self, predicate: &str, parts: Vec<String>) {
        let mut neurons = vec![Neuron::symbol(predicate)];
        for p in &parts {
            if let Ok(n) = p.parse::<i64>() {
                neurons.push(Neuron::int_val(n));
            } else if let Ok(f) = p.parse::<f64>() {
                neurons.push(Neuron::float_val(f));
            } else {
                neurons.push(Neuron::symbol(p));
            }
        }
        self.session.store_mut().insert(
            Neuron::expression(neurons),
            TruthValue::from_strength(0.99),
        );
    }

    /// Run forward chaining, return count of new facts derived.
    fn run(&mut self) -> bool {
        self.session.heartbeat()
    }

    /// Query: returns list of int-arg tuples matching predicate.
    fn query(&self, predicate: &str, arity: usize) -> Vec<Vec<i64>> {
        let mut pattern = vec![Neuron::symbol(predicate)];
        for i in 0..arity {
            pattern.push(Neuron::variable(&format!("V{}", i)));
        }
        let q = Neuron::expression(pattern);
        let hits = self.session.store().query(&q);
        let mut results = Vec::new();
        for sn in hits {
            if let Neuron::Expression(parts) = &sn.neuron {
                let vals: Vec<i64> = parts[1..]
                    .iter()
                    .filter_map(|n| n.as_f64().map(|f| f as i64))
                    .collect();
                if vals.len() == arity {
                    results.push(vals);
                }
            }
        }
        results
    }

    /// Query with mixed int/string results as strings.
    fn query_str(&self, predicate: &str, arity: usize) -> Vec<Vec<String>> {
        let mut pattern = vec![Neuron::symbol(predicate)];
        for i in 0..arity {
            pattern.push(Neuron::variable(&format!("V{}", i)));
        }
        let q = Neuron::expression(pattern);
        let hits = self.session.store().query(&q);
        let mut results = Vec::new();
        for sn in hits {
            if let Neuron::Expression(parts) = &sn.neuron {
                let vals: Vec<String> = parts[1..]
                    .iter()
                    .map(|n| format!("{}", n))
                    .collect();
                if vals.len() == arity {
                    results.push(vals);
                }
            }
        }
        results
    }

    /// Return total number of facts in the store.
    fn fact_count(&self) -> usize {
        self.session.fact_count()
    }

    /// Return total number of rules loaded.
    fn rule_count(&self) -> usize {
        self.session.rule_count()
    }

    /// Get all facts as strings.
    fn all_facts(&self) -> Vec<String> {
        self.session.all_facts().iter()
            .map(|sn| format!("{} <{:.2}, {:.2}>", sn.neuron, sn.tv.strength, sn.tv.confidence))
            .collect()
    }

    /// Remove all facts matching the given predicate names.
    /// Returns the number of facts removed.
    /// e.g. session.remove_by_predicate(["frame-number", "region", "cell-changed"])
    fn remove_by_predicate(&mut self, predicates: Vec<String>) -> usize {
        let refs: Vec<&str> = predicates.iter().map(|s| s.as_str()).collect();
        self.session.remove_by_predicate(&refs)
    }

    /// Feed a 2D grid into QOR using the existing Grid perception engine.
    /// Uses Grid::from_vecs + to_statements — same as qor solve / qor-agent.
    /// Returns the number of facts inserted.
    /// e.g. session.feed_grid("frame", [[5,5,9,8],[5,5,8,9]])
    fn feed_grid(&mut self, grid_id: &str, cells: Vec<Vec<u8>>) -> PyResult<usize> {
        let grid = Grid::from_vecs(cells)
            .map_err(|e| PyRuntimeError::new_err(format!("grid error: {}", e)))?;
        let stmts = grid.to_statements(grid_id);
        let count = stmts.len();
        self.session.exec_statements(stmts)
            .map_err(|e| PyRuntimeError::new_err(format!("exec error: {}", e)))?;
        Ok(count)
    }
}

#[pymodule]
fn qor_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<QorSession>()?;
    Ok(())
}
