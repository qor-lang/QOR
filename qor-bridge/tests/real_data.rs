use qor_bridge;
use qor_core::neuron::Statement;
use qor_runtime::eval::{ExecResult, Session};

fn ref_data_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Ref/data")
}

// ── Test learning from real market parquet data ─────────────────────────

#[test]
fn test_parquet_market_learned_knowledge() {
    let path = ref_data_dir().join("2025-03.parquet");
    let stmts = qor_bridge::feed_file(&path).unwrap();

    // Should have condensed knowledge, NOT 230k raw facts
    // Learned facts + context rules + queries = ~100-300 statements
    assert!(stmts.len() < 1000,
        "brain should be condensed knowledge, not raw data — got {} statements", stmts.len());
    assert!(stmts.len() > 30,
        "should have meaningful learned knowledge, got only {}", stmts.len());

    // ── Verify statistical summaries ──
    let has_stat_close = stmts.iter().any(|s| {
        if let Statement::Fact { neuron, .. } = s {
            format!("{}", neuron).contains("stat-close")
        } else { false }
    });
    assert!(has_stat_close, "should have stat-close summaries");

    let has_stat_rsi = stmts.iter().any(|s| {
        if let Statement::Fact { neuron, .. } = s {
            format!("{}", neuron).contains("stat-rsi")
        } else { false }
    });
    assert!(has_stat_rsi, "should have stat-rsi summaries");

    // ── Verify pattern rates ──
    let pattern_rates: Vec<_> = stmts.iter().filter(|s| {
        if let Statement::Fact { neuron, .. } = s {
            format!("{}", neuron).contains("pattern-rate")
        } else { false }
    }).collect();
    assert!(pattern_rates.len() >= 3,
        "should have multiple pattern rates, got {}", pattern_rates.len());

    // ── Verify co-occurrences ──
    let co_occurs: Vec<_> = stmts.iter().filter(|s| {
        if let Statement::Fact { neuron, .. } = s {
            format!("{}", neuron).contains("co-occur")
        } else { false }
    }).collect();
    assert!(!co_occurs.is_empty(), "should have co-occurrence facts");

    // ── Verify domain + purpose still present ──
    let has_market_domain = stmts.iter().any(|s| {
        if let Statement::Fact { neuron, .. } = s {
            let s = format!("{}", neuron);
            s.contains("domain") && s.contains("market")
        } else { false }
    });
    assert!(has_market_domain, "should detect market domain");

    let has_prediction = stmts.iter().any(|s| {
        if let Statement::Fact { neuron, .. } = s {
            let s = format!("{}", neuron);
            s.contains("purpose") && s.contains("prediction")
        } else { false }
    });
    assert!(has_prediction, "should have purpose=prediction");

    // ── Verify structural rules still present ──
    let rules: Vec<&Statement> = stmts.iter()
        .filter(|s| matches!(s, Statement::Rule { .. }))
        .collect();
    assert!(rules.len() >= 8, "expected 8+ rules, got {}", rules.len());

    // ── Verify metadata ──
    let has_data_points = stmts.iter().any(|s| {
        if let Statement::Fact { neuron, .. } = s {
            format!("{}", neuron).contains("data-points")
        } else { false }
    });
    assert!(has_data_points, "should have data-points metadata");

    let has_learned_at = stmts.iter().any(|s| {
        if let Statement::Fact { neuron, .. } = s {
            format!("{}", neuron).contains("learned-at")
        } else { false }
    });
    assert!(has_learned_at, "should have learned-at metadata");

    // ── Verify queries exist ──
    let queries: Vec<_> = stmts.iter()
        .filter(|s| matches!(s, Statement::Query { .. }))
        .collect();
    assert!(queries.len() >= 4, "should have auto-queries, got {}", queries.len());
}

// ── Test learned knowledge loads and queries correctly ───────────────────

#[test]
fn test_parquet_learned_knowledge_queries() {
    let path = ref_data_dir().join("2025-03.parquet");
    let stmts = qor_bridge::feed_file(&path).unwrap();

    let mut session = Session::new();
    session.exec_statements(stmts).unwrap();

    // ── Query domain ──
    let results = session.exec("? (domain $x)").unwrap();
    match &results[0] {
        ExecResult::Query(qr) => {
            assert_eq!(qr.results.len(), 1);
            assert!(qr.results[0].contains("market"));
        }
        _ => panic!("expected query result"),
    }

    // ── Query purpose ──
    let results = session.exec("? (purpose $x)").unwrap();
    match &results[0] {
        ExecResult::Query(qr) => {
            assert_eq!(qr.results.len(), 2);
            let all = qr.results.join(" ");
            assert!(all.contains("analysis"));
            assert!(all.contains("prediction"));
        }
        _ => panic!("expected query result"),
    }

    // ── Query stat summaries ──
    let results = session.exec("? (stat-close $x $y)").unwrap();
    match &results[0] {
        ExecResult::Query(qr) => {
            assert!(qr.results.len() >= 4,
                "expected count/mean/min/max for close, got {}", qr.results.len());
        }
        _ => panic!("expected query result"),
    }

    // ── Query pattern rates ──
    let results = session.exec("? (pattern-rate $x $y $z)").unwrap();
    match &results[0] {
        ExecResult::Query(qr) => {
            assert!(qr.results.len() >= 3,
                "expected multiple pattern rates, got {}", qr.results.len());
        }
        _ => panic!("expected query result"),
    }

    // ── Query data-points metadata ──
    let results = session.exec("? (data-points $x)").unwrap();
    match &results[0] {
        ExecResult::Query(qr) => {
            assert_eq!(qr.results.len(), 1);
            // Should be 10000 (the number of unique entities)
            assert!(qr.results[0].contains("10"), "data-points should show entity count");
        }
        _ => panic!("expected query result"),
    }
}
