//! Debug test for puzzle 017c7c7b — periodic tiling
//! Run: cargo test -p qor-agent debug_017 -- --nocapture

use qor_runtime::eval::Session;

#[test]
fn debug_017() {
    // TEST A: Without modulo — works correctly
    eprintln!("\n=== TEST A: No modulo (baseline) ===");
    {
        let mut s = Session::new();
        let stmts = qor_core::parser::parse(r#"
            (remap 1 2)
            (cell 0 0 1) (cell 0 1 0) (cell 1 0 0) (cell 1 1 1)
            (test $r $c $to) if (remap $fv $to) (cell $r $c $fv) (!= $fv 0)
        "#).unwrap();
        let _ = s.exec_statements(stmts);
        for f in s.all_facts().iter() {
            if format!("{}", f.neuron).starts_with("(test") {
                eprintln!("  {}", f.neuron);
            }
        }
    }

    // TEST B: With modulo — does it break?
    eprintln!("\n=== TEST B: With modulo ===");
    {
        let mut s = Session::new();
        let stmts = qor_core::parser::parse(r#"
            (remap 1 2)
            (idx 0) (idx 1) (idx 2) (idx 3)
            (per 2)
            (cell 0 0 1) (cell 0 1 0) (cell 1 0 0) (cell 1 1 1)
            (test $r $c $to) if (remap $fv $to) (idx $r) (per $p) (% $r $p $sr) (cell $sr $c $fv) (!= $fv 0)
        "#).unwrap();
        let _ = s.exec_statements(stmts);
        for f in s.all_facts().iter() {
            if format!("{}", f.neuron).starts_with("(test") {
                eprintln!("  {}", f.neuron);
            }
        }
    }

    // TEST C: Move modulo AFTER the cell match to confirm it's the modulo
    eprintln!("\n=== TEST C: Modulo after cell (should fail - sr unbound) ===");
    {
        let mut s = Session::new();
        let stmts = qor_core::parser::parse(r#"
            (remap 1 2)
            (idx 0) (idx 1)
            (per 2)
            (cell 0 0 1) (cell 0 1 0) (cell 1 0 0) (cell 1 1 1)
            (test $r $c $to) if (remap $fv $to) (idx $r) (per $p) (cell $r $c $fv) (!= $fv 0)
        "#).unwrap();
        let _ = s.exec_statements(stmts);
        for f in s.all_facts().iter() {
            if format!("{}", f.neuron).starts_with("(test") {
                eprintln!("  {}", f.neuron);
            }
        }
    }

    // TEST D: Arithmetic with + instead of % — does + also break?
    eprintln!("\n=== TEST D: Addition instead of modulo ===");
    {
        let mut s = Session::new();
        let stmts = qor_core::parser::parse(r#"
            (remap 1 2)
            (row 3)
            (cell 3 0 1) (cell 3 1 0)
            (test $c $to) if (remap $fv $to) (row $r) (+ $r 0 $sr) (cell $sr $c $fv) (!= $fv 0)
        "#).unwrap();
        let _ = s.exec_statements(stmts);
        for f in s.all_facts().iter() {
            if format!("{}", f.neuron).starts_with("(test") {
                eprintln!("  {}", f.neuron);
            }
        }
    }

    // TEST E: Subtraction
    eprintln!("\n=== TEST E: Subtraction instead of modulo ===");
    {
        let mut s = Session::new();
        let stmts = qor_core::parser::parse(r#"
            (remap 1 2)
            (row 3)
            (cell 3 0 1) (cell 3 1 0)
            (test $c $to) if (remap $fv $to) (row $r) (- $r 0 $sr) (cell $sr $c $fv) (!= $fv 0)
        "#).unwrap();
        let _ = s.exec_statements(stmts);
        for f in s.all_facts().iter() {
            if format!("{}", f.neuron).starts_with("(test") {
                eprintln!("  {}", f.neuron);
            }
        }
    }

    // TEST F: No arithmetic, use pre-computed lookup
    eprintln!("\n=== TEST F: Pre-computed lookup (no arithmetic) ===");
    {
        let mut s = Session::new();
        let stmts = qor_core::parser::parse(r#"
            (remap 1 2)
            (map 0 0) (map 1 1) (map 2 0) (map 3 1)
            (cell 0 0 1) (cell 0 1 0) (cell 1 0 0) (cell 1 1 1)
            (test $r $c $to) if (remap $fv $to) (map $r $sr) (cell $sr $c $fv) (!= $fv 0)
        "#).unwrap();
        let _ = s.exec_statements(stmts);
        for f in s.all_facts().iter() {
            if format!("{}", f.neuron).starts_with("(test") {
                eprintln!("  {}", f.neuron);
            }
        }
    }
}
