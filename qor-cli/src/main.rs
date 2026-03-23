use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    match args[1].as_str() {
        "run" => {
            if args.len() < 3 {
                eprintln!("error: qor run requires a file path");
                eprintln!("usage: qor run <file.qor>");
                process::exit(1);
            }
            run_file(&args[2]);
        }
        "check" => {
            if args.len() < 3 {
                eprintln!("error: qor check requires a file path");
                eprintln!("usage: qor check <file.qor>");
                process::exit(1);
            }
            check_file(&args[2]);
        }
        "feed" => {
            if args.len() < 3 {
                eprintln!("error: qor feed requires a file path");
                eprintln!("usage: qor feed <file>");
                process::exit(1);
            }
            feed_file(&args[2]);
        }
        "think" => {
            let watch = args.iter().any(|a| a == "--watch" || a == "-w");
            if watch {
                think_watch();
            } else {
                think();
            }
        }
        "explain" => {
            if args.len() < 3 {
                eprintln!("error: qor explain requires a pattern");
                eprintln!("usage: qor explain \"(overbought $x)\"");
                process::exit(1);
            }
            explain(&args[2]);
        }
        "summary" => {
            summary();
        }
        "chat" => {
            chat();
        }
        "test" => {
            if args.len() < 3 {
                eprintln!("error: qor test requires a file path");
                eprintln!("usage: qor test <file.qor>");
                process::exit(1);
            }
            test_file(&args[2]);
        }
        "solve" => {
            if args.len() < 3 {
                eprintln!("error: qor solve requires a file path");
                eprintln!("usage: qor solve <file.qor>");
                process::exit(1);
            }
            solve_file(&args[2]);
        }
        "dna" => {
            dna_command();
        }
        "heartbeat" => {
            heartbeat_command();
        }
        "build-kb" => {
            build_kb_command();
        }
        "perceive" => {
            perceive_command();
        }
        "repl" => {
            repl();
        }
        "version" => {
            println!("qor {}", env!("CARGO_PKG_VERSION"));
        }
        "help" | "--help" | "-h" => {
            print_usage();
        }
        other => {
            eprintln!("error: unknown command '{}'", other);
            print_usage();
            process::exit(1);
        }
    }
}

fn run_file(path: &str) {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read '{}': {}", path, e);
            process::exit(1);
        }
    };

    match qor_runtime::eval::run(&source) {
        Ok(results) => {
            for qr in results {
                println!("? {}", qr.pattern);
                for r in &qr.results {
                    println!("  -> {}", r);
                }
                println!();
            }
        }
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    }
}

fn check_file(path: &str) {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read '{}': {}", path, e);
            process::exit(1);
        }
    };

    match qor_core::parser::parse_with_warnings(&source) {
        Ok((stmts, warnings)) => {
            println!("OK: {} statements parsed", stmts.len());
            if !warnings.is_empty() {
                println!();
                println!("Warnings ({}):", warnings.len());
                for w in &warnings {
                    println!("  WARNING: {}", w);
                }
            }
        }
        Err(e) => {
            eprintln!("SYNTAX ERROR: {}", e);
            process::exit(1);
        }
    }
}

fn test_file(path: &str) {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read '{}': {}", path, e);
            process::exit(1);
        }
    };

    let stmts = match qor_core::parser::parse(&source) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SYNTAX ERROR: {}", e);
            process::exit(1);
        }
    };

    // Separate rules/facts from tests
    let mut rules_and_facts = Vec::new();
    let mut tests = Vec::new();
    for stmt in stmts {
        if matches!(stmt, qor_core::neuron::Statement::Test { .. }) {
            tests.push(stmt);
        } else {
            rules_and_facts.push(stmt);
        }
    }

    if tests.is_empty() {
        println!("No @test blocks found in {}", path);
        return;
    }

    println!();
    println!("  \x1b[1;36m~~~ QOR Test ~~~\x1b[0m");
    println!("  \x1b[2m{}\x1b[0m", path);
    println!();

    let mut passed = 0;
    let mut failed = 0;

    for test_stmt in &tests {
        if let qor_core::neuron::Statement::Test { name, given, expect } = test_stmt {
            // Fresh session per test
            let mut session = qor_runtime::eval::Session::new();
            if let Err(e) = session.exec_statements(rules_and_facts.clone()) {
                eprintln!("  \x1b[31mERROR\x1b[0m {}: {}", name, e);
                failed += 1;
                continue;
            }

            // Inject given facts
            let given_stmts: Vec<qor_core::neuron::Statement> = given.iter().map(|n| {
                qor_core::neuron::Statement::Fact {
                    neuron: n.clone(),
                    tv: None,
                    decay: None,
                }
            }).collect();
            if let Err(e) = session.exec_statements(given_stmts) {
                eprintln!("  \x1b[31mERROR\x1b[0m {}: {}", name, e);
                failed += 1;
                continue;
            }

            // Check expectations
            let all_facts = session.all_facts();
            let mut test_passed = true;
            let mut failure_reason = String::new();

            for exp in expect {
                match exp {
                    qor_core::neuron::TestExpect::Present(n) => {
                        let found = all_facts.iter().any(|sn|
                            qor_core::unify::unify(n, &sn.neuron).is_some());
                        if !found {
                            test_passed = false;
                            failure_reason = format!("expected {} but not found", n);
                            break;
                        }
                    }
                    qor_core::neuron::TestExpect::Absent(n) => {
                        let found = all_facts.iter().any(|sn|
                            qor_core::unify::unify(n, &sn.neuron).is_some());
                        if found {
                            test_passed = false;
                            failure_reason = format!("expected NOT {} but it was derived", n);
                            break;
                        }
                    }
                }
            }

            if test_passed {
                println!("  \x1b[32mPASS\x1b[0m {}", name);
                passed += 1;
            } else {
                println!("  \x1b[31mFAIL\x1b[0m {} — {}", name, failure_reason);
                failed += 1;
            }
        }
    }

    println!();
    if failed == 0 {
        println!("  \x1b[32mAll {} tests passed\x1b[0m", passed);
    } else {
        println!("  \x1b[31m{} passed, {} failed\x1b[0m", passed, failed);
    }
    println!();

    if failed > 0 {
        process::exit(1);
    }
}

fn brain_dir() -> std::path::PathBuf {
    // brain/ lives next to the qor binary's project root
    let mut dir = env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."));

    // Walk up to find the workspace root (has Cargo.toml with [workspace])
    for _ in 0..5 {
        if dir.join("brain").exists() {
            return dir.join("brain");
        }
        if dir.join("Cargo.toml").exists() {
            let brain = dir.join("brain");
            let _ = fs::create_dir_all(&brain);
            return brain;
        }
        if let Some(parent) = dir.parent() {
            dir = parent.to_path_buf();
        } else {
            break;
        }
    }

    // Fallback: brain/ in current directory
    let brain = std::path::PathBuf::from("brain");
    let _ = fs::create_dir_all(&brain);
    brain
}

fn feed_file(path: &str) {
    let file_path = std::path::Path::new(path);
    match qor_bridge::feed_file(file_path) {
        Ok(new_statements) => {
            // Auto-save .qor into brain/
            let brain = brain_dir();
            let file_name = file_path.file_stem()
                .unwrap_or_else(|| std::ffi::OsStr::new("data"));
            let out_path = brain.join(file_name).with_extension("qor");

            // Incremental learning: merge with existing brain if present
            let statements = if out_path.exists() {
                match fs::read_to_string(&out_path) {
                    Ok(existing_source) => {
                        match qor_core::parser::parse(&existing_source) {
                            Ok(existing) => {
                                let merged = qor_bridge::learn::merge_learned(&existing, &new_statements);
                                println!("merged:   {} existing + {} new", existing.len(), new_statements.len());
                                merged
                            }
                            Err(_) => new_statements, // Can't parse old brain, overwrite
                        }
                    }
                    Err(_) => new_statements,
                }
            } else {
                new_statements
            };

            let mut lines = Vec::new();
            let mut learned_count = 0;
            let mut rule_count = 0;
            let mut query_count = 0;

            for stmt in &statements {
                match stmt {
                    qor_core::neuron::Statement::Fact { neuron, tv, decay } => {
                        let tv_str = tv.map(|t| format!(" <{:.2}, {:.2}>", t.strength, t.confidence))
                            .unwrap_or_default();
                        let decay_str = decay.map(|d| format!(" @decay {:.2}", d))
                            .unwrap_or_default();
                        lines.push(format!("{}{}{}", neuron, tv_str, decay_str));
                        learned_count += 1;
                    }
                    qor_core::neuron::Statement::Rule { head, body, tv } => {
                        let body_str: Vec<String> = body.iter().map(|c| format!("{}", c)).collect();
                        let tv_str = tv.map(|t| format!(" <{:.2}, {:.2}>", t.strength, t.confidence))
                            .unwrap_or_default();
                        lines.push(format!("{} if {}{}", head, body_str.join(" "), tv_str));
                        rule_count += 1;
                    }
                    qor_core::neuron::Statement::Query { pattern } => {
                        lines.push(format!("? {}", pattern));
                        query_count += 1;
                    }
                    qor_core::neuron::Statement::Test { .. } => {} // skip tests in feed output
                }
            }
            match fs::write(&out_path, lines.join("\n") + "\n") {
                Ok(_) => {
                    println!("learned:  {}", learned_count);
                    println!("rules:    {}", rule_count);
                    if query_count > 0 {
                        println!("queries:  {}", query_count);
                    }
                    println!("brain:    {}", out_path.display());
                }
                Err(e) => {
                    eprintln!("error: could not write '{}': {}", out_path.display(), e);
                    process::exit(1);
                }
            }
        }
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    }
}

// ── Think ───────────────────────────────────────────────────────────────

fn think() {
    let brain = brain_dir();
    if !brain.exists() {
        eprintln!("error: no brain/ directory found — run `qor feed <file>` first");
        process::exit(1);
    }

    let mut session = qor_runtime::eval::Session::new();
    load_brain_into_session(&mut session, &brain);
    load_meta_into_session(&mut session);

    if session.fact_count() == 0 && session.rule_count() == 0 {
        eprintln!("error: brain/ is empty — run `qor feed <file>` first");
        process::exit(1);
    }

    println!();
    println!("  \x1b[1;36m~~~ QOR Think ~~~\x1b[0m");
    println!("  \x1b[2mLoaded {} facts, {} rules\x1b[0m",
        session.fact_count(), session.rule_count());
    println!();

    // Run heartbeat loop — reason continuously
    let max_cycles = 50;
    let mut stable_count = 0;
    let mut total_cycles = 0;

    for i in 0..max_cycles {
        let changed = session.heartbeat();
        if changed {
            stable_count = 0;
            total_cycles += 1;
            if i < 5 || i % 10 == 0 {
                println!("  \x1b[31m<3\x1b[0m cycle {} — beliefs strengthening...", i + 1);
            }
        } else {
            stable_count += 1;
            if stable_count >= 3 {
                println!("  \x1b[32m<3\x1b[0m settled after {} cycles — beliefs are stable", total_cycles);
                break;
            }
        }
    }

    if stable_count < 3 {
        println!("  \x1b[33m<3\x1b[0m {} cycles — still evolving", total_cycles);
    }

    println!();

    // Run induction/abduction to discover new rules
    let rule_stmts = session.rules_as_statements();
    if !rule_stmts.is_empty() {
        let discovered = qor_inference::infer(&rule_stmts);
        if !discovered.is_empty() {
            println!("  \x1b[1;35mDiscovered Rules:\x1b[0m");
            for rule in &discovered {
                if let qor_core::neuron::Statement::Rule { head, body, tv } = rule {
                    let body_str: Vec<String> = body.iter().map(|c| format!("{}", c)).collect();
                    let tv_str = tv.map(|t| format!(" <{:.2}, {:.2}>", t.strength, t.confidence))
                        .unwrap_or_default();
                    println!("    \x1b[35m*\x1b[0m {} if {}{}", head, body_str.join(" "), tv_str);
                }
            }
            println!();

            // Load discovered rules into session and run more heartbeats
            if session.exec_statements(discovered).is_ok() {
                for _ in 0..10 {
                    session.heartbeat();
                }
            }
        }
    }

    // Run stored queries
    let query_results = session.run_queries();
    if !query_results.is_empty() {
        println!("  \x1b[1mInsights:\x1b[0m");
        println!();
        for qr in &query_results {
            println!("  \x1b[36m?\x1b[0m {}", qr.pattern);
            for r in &qr.results {
                if r.contains("(inferred)") {
                    println!("    \x1b[33m->\x1b[0m {}", r);
                } else if r == "no results" {
                    println!("    \x1b[2m-> no results\x1b[0m");
                } else {
                    println!("    -> {}", r);
                }
            }
        }
        println!();
    }

    // Summary
    println!("  \x1b[1mSummary:\x1b[0m");
    println!("    facts:    {}", session.fact_count());
    println!("    rules:    {}", session.rule_count());
    println!("    queries:  {}", session.queries().len());
    println!("    cycles:   {}", session.consolidation_cycles());
    let inferred = session.all_facts().iter().filter(|f| f.inferred).count();
    if inferred > 0 {
        println!("    derived:  {}", inferred);
    }
    println!();

    // Save inferred knowledge back to the mutable KB
    save_insights_to_kb(&session, &brain);
}

// ── Explain ──────────────────────────────────────────────────────────────

fn explain(pattern_str: &str) {
    // Parse the pattern
    let source = format!("? {}", pattern_str);
    let stmts = match qor_core::parser::parse(&source) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not parse pattern: {}", e);
            process::exit(1);
        }
    };

    let pattern = match stmts.into_iter().next() {
        Some(qor_core::neuron::Statement::Query { pattern }) => pattern,
        _ => {
            eprintln!("error: expected a query pattern like \"(overbought $x)\"");
            process::exit(1);
        }
    };

    // Load brain
    let brain = brain_dir();
    if !brain.exists() {
        eprintln!("error: no brain/ directory found — run `qor feed <file>` first");
        process::exit(1);
    }

    let mut session = qor_runtime::eval::Session::new();
    load_brain_into_session(&mut session, &brain);
    load_meta_into_session(&mut session);

    // Run heartbeat to derive facts
    for _ in 0..20 {
        if !session.heartbeat() { break; }
    }

    // Explain
    let explanations = session.explain(&pattern);
    println!();
    if explanations.is_empty() {
        println!("  \x1b[2mno matching facts found for {}\x1b[0m", pattern);
    } else {
        for exp in &explanations {
            print_explanation(exp, 0);
        }
    }
    println!();
}

fn print_explanation(exp: &qor_runtime::eval::Explanation, indent: usize) {
    let pad = "  ".repeat(indent + 1);
    match &exp.reason {
        qor_runtime::eval::ExplanationReason::Asserted => {
            println!("{}{} {} \x1b[2m-- asserted\x1b[0m", pad, exp.fact, exp.tv);
        }
        qor_runtime::eval::ExplanationReason::Derived { rule, from } => {
            println!("{}{} {}", pad, exp.fact, exp.tv);
            println!("{}  \x1b[2mvia:\x1b[0m {}", pad, rule);
            for sub in from {
                print_explanation(sub, indent + 2);
            }
        }
    }
}

// ── Watch Mode ───────────────────────────────────────────────────────────

fn think_watch() {
    let brain = brain_dir();
    if !brain.exists() {
        eprintln!("error: no brain/ directory found — run `qor feed <file>` first");
        process::exit(1);
    }

    println!();
    println!("  \x1b[1;36m~~~ QOR Think (Watch Mode) ~~~\x1b[0m");
    println!("  \x1b[2mMonitoring brain/ for changes... Ctrl+C to stop\x1b[0m");
    println!();

    let mut session = qor_runtime::eval::Session::new();
    let mut timestamps = load_brain_timestamps(&brain);
    load_brain_into_session(&mut session, &brain);
    load_meta_into_session(&mut session);

    // Initial think cycle
    run_think_cycle(&mut session, true);

    // Poll loop
    loop {
        std::thread::sleep(std::time::Duration::from_secs(2));

        let new_timestamps = load_brain_timestamps(&brain);
        if new_timestamps != timestamps {
            // Files changed — reload
            println!("  \x1b[33m~\x1b[0m brain/ changed — re-reasoning...");
            session = qor_runtime::eval::Session::new();
            load_brain_into_session(&mut session, &brain);
            load_meta_into_session(&mut session);
            run_think_cycle(&mut session, false);
            timestamps = new_timestamps;
        }
    }
}

fn load_brain_timestamps(brain: &std::path::Path) -> Vec<(String, std::time::SystemTime)> {
    let mut timestamps = Vec::new();
    if let Ok(entries) = fs::read_dir(brain) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "qor") {
                if let Ok(meta) = path.metadata() {
                    if let Ok(modified) = meta.modified() {
                        timestamps.push((path.display().to_string(), modified));
                    }
                }
            }
        }
    }
    timestamps.sort_by_key(|(name, _)| name.clone());
    timestamps
}

fn load_brain_into_session(session: &mut qor_runtime::eval::Session, brain: &std::path::Path) {
    if let Ok(entries) = fs::read_dir(brain) {
        let mut files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "qor"))
            .collect();
        files.sort_by_key(|e| e.path());

        for entry in files {
            if let Ok(source) = fs::read_to_string(entry.path()) {
                if let Err(e) = session.exec(&source) {
                    eprintln!("  \x1b[31mwarning:\x1b[0m {}: {}", entry.path().display(), e);
                }
            }
        }
    }

    // Auto-load binary KB + OEIS formulas if brain/knowledge/ exists
    load_kb_into_session(session, brain);
}

/// Load the binary knowledge graph into the session.
/// Attaches the KB for entity lookups and loads OEIS formula rules.
fn load_kb_into_session(session: &mut qor_runtime::eval::Session, brain: &std::path::Path) {
    let kb_dir = brain.join("knowledge");
    if !kb_dir.exists() {
        return;
    }

    match qor_runtime::kb::KnowledgeBase::load(&kb_dir) {
        Ok(kb) => {
            let stats = kb.stats();
            let kb = std::sync::Arc::new(kb);
            session.set_kb(kb);
            eprintln!("  \x1b[2mkb: {}\x1b[0m", stats);
        }
        Err(e) => {
            eprintln!("  \x1b[33mwarning:\x1b[0m KB load: {}", e);
        }
    }

    // All OEIS data (including categorized formulas) is in oeis_triples.bin.
    // Reasoning rules that USE the graph data live in meta/math.qor.
}

/// Load meta-rules from meta/ directory into session.
fn load_meta_into_session(session: &mut qor_runtime::eval::Session) {
    let meta = meta_dir();
    if !meta.exists() {
        return;
    }

    if let Ok(entries) = fs::read_dir(&meta) {
        let mut files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "qor"))
            .collect();
        files.sort_by_key(|e| e.path());

        let mut loaded = 0;
        for entry in files {
            if let Ok(source) = fs::read_to_string(entry.path()) {
                if let Err(e) = session.exec(&source) {
                    eprintln!("  \x1b[33mwarning:\x1b[0m {}: {}", entry.path().display(), e);
                } else {
                    loaded += 1;
                }
            }
        }
        if loaded > 0 {
            eprintln!("  \x1b[2mmeta: loaded {} rule files\x1b[0m", loaded);
        }
    }
}

/// Save significant inferred facts back to the mutable KB.
fn save_insights_to_kb(session: &qor_runtime::eval::Session, brain: &std::path::Path) {
    let kb_dir = brain.join("knowledge");
    if !kb_dir.exists() {
        return;
    }

    // Load KB mutably
    let mut kb = match qor_runtime::kb::KnowledgeBase::load(&kb_dir) {
        Ok(kb) => kb,
        Err(_) => return,
    };

    // Find high-confidence inferred facts and upsert them into the KB
    let mut saved = 0;
    for sn in session.all_facts() {
        if !sn.inferred {
            continue;
        }
        // Only save high-confidence inferences (strength >= 0.80)
        if sn.tv.strength < 0.80 || sn.tv.confidence < 0.50 {
            continue;
        }
        // Skip ephemeral predicates
        if let qor_core::neuron::Neuron::Expression(parts) = &sn.neuron {
            if let Some(qor_core::neuron::Neuron::Symbol(pred)) = parts.first() {
                // Skip chat/perception ephemeral predicates
                if matches!(pred.as_str(),
                    "input" | "compound" | "intent" | "response" | "perceived"
                    | "significant-move" | "market-sentiment" | "promote"
                ) {
                    continue;
                }
                // Convert to KB triple: (predicate, arg1, arg2)
                if parts.len() == 3 {
                    let arg1 = parts[1].to_string();
                    let arg2 = parts[2].to_string();
                    kb.upsert_named(
                        &arg1,
                        pred,
                        &arg2,
                        sn.tv.strength as f32,
                        sn.tv.confidence as f32,
                    );
                    saved += 1;
                }
            }
        }
    }

    if saved > 0 {
        if let Err(e) = kb.save(&kb_dir) {
            eprintln!("  \x1b[33mwarning:\x1b[0m KB save: {}", e);
        } else {
            eprintln!("  \x1b[2mkb: saved {} inferred facts to graph\x1b[0m", saved);
        }
    }
}

fn run_think_cycle(session: &mut qor_runtime::eval::Session, verbose: bool) {
    // Heartbeat
    let mut total_cycles = 0;
    for _ in 0..50 {
        if session.heartbeat() {
            total_cycles += 1;
        } else {
            break;
        }
    }

    if verbose {
        println!("  \x1b[32m<3\x1b[0m {} heartbeat cycles", total_cycles);
    }

    // Inference
    let rule_stmts = session.rules_as_statements();
    if !rule_stmts.is_empty() {
        let discovered = qor_inference::infer(&rule_stmts);
        if !discovered.is_empty() && verbose {
            println!("  \x1b[35m*\x1b[0m {} new rules discovered", discovered.len());
        }
        let _ = session.exec_statements(discovered);
        for _ in 0..10 {
            session.heartbeat();
        }
    }

    // Queries
    let query_results = session.run_queries();
    if !query_results.is_empty() {
        for qr in &query_results {
            println!("  \x1b[36m?\x1b[0m {}", qr.pattern);
            for r in &qr.results {
                if r == "no results" {
                    println!("    \x1b[2m-> no results\x1b[0m");
                } else {
                    println!("    -> {}", r);
                }
            }
        }
    }

    // Status line
    println!("  \x1b[2m[{}f/{}r]\x1b[0m", session.fact_count(), session.rule_count());
    println!();
}

// ── Summary ──────────────────────────────────────────────────────────────

fn summary() {
    let brain = brain_dir();
    if !brain.exists() {
        eprintln!("error: no brain/ directory found — run `qor feed <file>` first");
        process::exit(1);
    }

    // Collect all brain statements
    let mut all_statements = Vec::new();
    if let Ok(entries) = fs::read_dir(&brain) {
        let mut files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "qor"))
            .collect();
        files.sort_by_key(|e| e.path());

        for entry in files {
            if let Ok(source) = fs::read_to_string(entry.path()) {
                if let Ok(stmts) = qor_core::parser::parse(&source) {
                    all_statements.extend(stmts);
                }
            }
        }
    }

    if all_statements.is_empty() {
        eprintln!("error: brain/ is empty — run `qor feed <file>` first");
        process::exit(1);
    }

    let summary = qor_bridge::llguidance::summarize(&all_statements);
    println!();
    println!("{}", summary);
    println!();
}

// ── Solve ────────────────────────────────────────────────────────────────

fn solve_file(path: &str) {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read '{}': {}", path, e);
            process::exit(1);
        }
    };

    let mut session = qor_runtime::eval::Session::new();

    // Load meta rules (universal reasoning)
    load_meta_into_session(&mut session);

    if let Err(e) = session.exec(&source) {
        eprintln!("error: {}", e);
        process::exit(1);
    }

    // Extract expected facts (any predicate starting with "expected-" or "predict-")
    let all_facts = session.all_facts();
    let expected: Vec<qor_core::neuron::Statement> = all_facts.iter()
        .filter(|sn| {
            if let qor_core::neuron::Neuron::Expression(parts) = &sn.neuron {
                if let Some(qor_core::neuron::Neuron::Symbol(pred)) = parts.first() {
                    return pred.starts_with("expected-") || pred.starts_with("predict-");
                }
            }
            false
        })
        .map(|sn| qor_core::neuron::Statement::Fact {
            neuron: sn.neuron.clone(),
            tv: Some(sn.tv),
            decay: None,
        })
        .collect();

    // Detect target predicate
    let target_pred = if expected.iter().any(|s| {
        if let qor_core::neuron::Statement::Fact { neuron: qor_core::neuron::Neuron::Expression(p), .. } = s {
            p.first().map(|n| n.to_string()).unwrap_or_default().starts_with("predict-")
        } else { false }
    }) { "predict-cell" } else { "answer" };

    // Collect observations
    let observations: Vec<qor_core::neuron::Statement> = all_facts.iter()
        .filter(|sn| {
            if let qor_core::neuron::Neuron::Expression(parts) = &sn.neuron {
                if let Some(qor_core::neuron::Neuron::Symbol(pred)) = parts.first() {
                    return pred.starts_with("obs-");
                }
            }
            false
        })
        .map(|sn| qor_core::neuron::Statement::Fact {
            neuron: sn.neuron.clone(), tv: Some(sn.tv), decay: None,
        })
        .collect();

    println!("~~~ QOR Solve (6-phase pipeline) ~~~");
    let meta = meta_dir();
    let result = qor_bridge::solve::solve(
        &session, &expected, target_pred, &observations, 15000, None,
        if meta.is_dir() { Some(meta.as_path()) } else { None },
        None,
    );

    println!("Score: {:.1}%", result.score * 100.0);
    println!("Solved: {}", result.solved);
    if let Some(phase) = result.solved_in_phase {
        println!("Solved in: {}", phase);
    }
    println!("Rounds: {}", result.rounds);
    println!("Candidates explored: {}", result.candidates_explored);
    println!("Mutations tried: {}", result.mutations_tried);
    println!("Failures tracked: {}", result.failures_tracked);
    println!("Overfits detected: {}", result.overfit_count);
    println!("Elapsed: {}ms", result.elapsed_ms);
    if !result.best_rules.is_empty() {
        println!("\nBest rules:");
        for r in &result.best_rules {
            println!("  {}", r);
        }
    }
}

// ── DNA ──────────────────────────────────────────────────────────────────

fn dna_command() {
    let args: Vec<String> = env::args().collect();
    let sub = args.get(2).map(|s| s.as_str()).unwrap_or("help");
    let dna_root = dna_dir();

    match sub {
        "convert" => {
            let target = args.get(3).map(|s| s.as_str());
            match target {
                Some("all") | None => {
                    // Convert all DNA profiles
                    println!();
                    println!("  \x1b[1;35mConverting all DNA profiles to .qor\x1b[0m");
                    println!();
                    let results = qor_bridge::dna::convert_all(&dna_root);
                    if results.is_empty() {
                        eprintln!("error: no DNA profiles found in {}", dna_root.display());
                        process::exit(1);
                    }
                    for (id, path, count) in &results {
                        println!("  \x1b[32m✓\x1b[0m {:<20} {} facts → {}",
                            id, count, path.display());
                    }
                    println!();
                    println!("  \x1b[2mConverted {} profiles\x1b[0m", results.len());
                    println!();
                }
                Some(id) => {
                    // Convert single DNA profile
                    match qor_bridge::dna::save_dna_qor(&dna_root, id) {
                        Ok((path, count)) => {
                            println!();
                            println!("  \x1b[32m✓\x1b[0m {} — {} facts → {}",
                                id, count, path.display());
                            println!();
                        }
                        Err(e) => {
                            eprintln!("error: {}", e);
                            process::exit(1);
                        }
                    }
                }
            }
        }
        "list" => {
            let profiles = qor_bridge::dna::available_dna(&dna_root);
            if profiles.is_empty() {
                eprintln!("error: no DNA profiles found in {}", dna_root.display());
                process::exit(1);
            }
            println!();
            println!("  \x1b[1mAvailable DNA profiles:\x1b[0m");
            println!();
            for (id, name) in &profiles {
                let has_qor = dna_root.join(id).join(format!("{}.qor", id)).exists();
                let marker = if has_qor { "\x1b[32m●\x1b[0m" } else { "\x1b[2m○\x1b[0m" };
                println!("    {} {:<20} — {}", marker, id, name);
            }
            println!();
            println!("  \x1b[2m● = .qor generated   ○ = JSON only\x1b[0m");
            println!("  \x1b[2mRun 'qor dna convert' to generate all .qor files\x1b[0m");
            println!();
        }
        _ => {
            println!("QOR DNA — Profession Profile Manager");
            println!();
            println!("USAGE:");
            println!("  qor dna list                List available DNA profiles");
            println!("  qor dna convert             Convert ALL DNA profiles to .qor");
            println!("  qor dna convert <id>        Convert a single DNA profile to .qor");
            println!();
            println!("CHAT:");
            println!("  qor chat --dna <id>         Chat with a DNA loaded");
            println!("  qor chat --dna list         List available profiles");
        }
    }
}

// ── Chat ─────────────────────────────────────────────────────────────────

fn language_dir() -> std::path::PathBuf {
    // language/ lives at workspace root, same as brain/
    let mut dir = env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."));

    for _ in 0..5 {
        if dir.join("language").exists() {
            return dir.join("language");
        }
        if dir.join("Cargo.toml").exists() {
            return dir.join("language");
        }
        if let Some(parent) = dir.parent() {
            dir = parent.to_path_buf();
        } else {
            break;
        }
    }

    std::path::PathBuf::from("language")
}

fn dna_dir() -> std::path::PathBuf {
    // dna/ lives at workspace root, same as brain/ and language/
    let mut dir = env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."));

    for _ in 0..5 {
        if dir.join("dna").exists() {
            return dir.join("dna");
        }
        if dir.join("Cargo.toml").exists() {
            return dir.join("dna");
        }
        if let Some(parent) = dir.parent() {
            dir = parent.to_path_buf();
        } else {
            break;
        }
    }

    std::path::PathBuf::from("dna")
}

fn chat() {
    let brain = brain_dir();
    let lang_root = language_dir();
    let dna_root = dna_dir();

    // Parse --dna <id> flag from args
    let args: Vec<String> = env::args().collect();
    let mut dna_id: Option<String> = None;
    let mut lang_code = "en".to_string();
    {
        let mut i = 2; // skip "qor" and "chat"
        while i < args.len() {
            if args[i] == "--dna" && i + 1 < args.len() {
                dna_id = Some(args[i + 1].clone());
                i += 2;
            } else {
                lang_code = args[i].clone();
                i += 1;
            }
        }
    }

    // Handle --dna list
    if dna_id.as_deref() == Some("list") {
        let profiles = qor_bridge::dna::available_dna(&dna_root);
        if profiles.is_empty() {
            eprintln!("error: no DNA profiles found in {}", dna_root.display());
            process::exit(1);
        }
        println!();
        println!("  \x1b[1mAvailable DNA profiles:\x1b[0m");
        println!();
        for (id, name) in &profiles {
            println!("    {:<20} — {}", id, name);
        }
        println!();
        println!("  \x1b[2mUsage: qor chat --dna <id>         Load by exact id\x1b[0m");
        println!("  \x1b[2m       qor chat --dna \"keyword\"    Search by keyword\x1b[0m");
        println!();
        return;
    }

    let lang_dir = lang_root.join(&lang_code);

    let mut session = qor_runtime::eval::Session::new();

    // Load brain knowledge + KB if available
    if brain.exists() {
        load_brain_into_session(&mut session, &brain);
    }
    load_meta_into_session(&mut session);

    // Load language knowledge from language/<code>/ folder (fallback to built-in)
    let lang = qor_bridge::language::load_language_dir(&lang_dir);
    let lang_source = if lang_dir.exists() {
        format!("language/{}/", lang_code)
    } else {
        "built-in".to_string()
    };
    if let Err(e) = session.exec_statements(lang) {
        eprintln!("error loading language knowledge: {}", e);
        process::exit(1);
    }

    // Load DNA profile if requested — supports exact ID or keyword search
    let mut dna_label = String::new();
    if let Some(ref raw_id) = dna_id {
        // Resolve: try exact ID first, then keyword search
        let resolved_id = if dna_root.join(raw_id).join(format!("{}.json", raw_id)).exists() {
            raw_id.clone()
        } else {
            // Keyword search
            let matches = qor_bridge::dna::find_dna_by_keywords(&dna_root, raw_id);
            if matches.is_empty() {
                eprintln!("error: no DNA profile matches '{}'", raw_id);
                eprintln!("hint: use 'qor chat --dna list' to see available profiles");
                process::exit(1);
            } else if matches.len() == 1 || matches[0].2 > matches[1].2 {
                // Clear winner
                let (ref best_id, ref best_name, _) = matches[0];
                println!("  \x1b[2mMatched '{}' → {} ({})\x1b[0m", raw_id, best_name, best_id);
                best_id.clone()
            } else {
                // Multiple equally strong matches — show them
                eprintln!("  Multiple DNA profiles match '{}':", raw_id);
                eprintln!();
                for (mid, mname, score) in matches.iter().take(5) {
                    eprintln!("    {:<20} — {} (score: {})", mid, mname, score);
                }
                eprintln!();
                eprintln!("  Use an exact id: qor chat --dna <id>");
                process::exit(1);
            }
        };

        match qor_bridge::dna::load_dna(&dna_root, &resolved_id) {
            Ok(profile) => {
                dna_label = format!("{} — \"{}\"", resolved_id, profile.archetype);
                let dna_stmts = qor_bridge::dna::dna_to_statements(&profile);
                if let Err(e) = session.exec_statements(dna_stmts) {
                    eprintln!("error loading DNA profile: {}", e);
                    process::exit(1);
                }

                // Load knowledge.txt
                if let Ok(knowledge) = qor_bridge::dna::load_knowledge(&dna_root, &resolved_id) {
                    let knowledge_stmts = qor_bridge::dna::parse_knowledge(&knowledge);
                    if let Err(e) = session.exec_statements(knowledge_stmts) {
                        eprintln!("warning: could not load DNA knowledge: {}", e);
                    }
                }

                // Load rules.qor — domain reasoning rules
                let rule_stmts = qor_bridge::dna::load_rules(&dna_root, &resolved_id);
                if !rule_stmts.is_empty() {
                    if let Err(e) = session.exec_statements(rule_stmts) {
                        eprintln!("warning: could not load DNA rules: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("error: {}", e);
                eprintln!("hint: use 'qor chat --dna list' to see available profiles");
                process::exit(1);
            }
        }
    }

    // Initial heartbeat to internalize rules
    for _ in 0..10 {
        session.heartbeat();
    }

    println!();
    println!("  \x1b[1;36m~~~ QOR Chat ~~~\x1b[0m");
    if !dna_label.is_empty() {
        println!("  \x1b[1;35mDNA: {}\x1b[0m", dna_label);
        // Show tagline from session facts
        let ctx = qor_bridge::language::extract_brain_context(session.all_facts());
        // Find tagline from dna-tagline fact
        for sn in session.all_facts() {
            if let qor_core::neuron::Neuron::Expression(parts) = &sn.neuron {
                if parts.len() == 2
                    && parts[0] == qor_core::neuron::Neuron::Symbol("dna-tagline".into())
                {
                    if let qor_core::neuron::Neuron::Value(
                        qor_core::neuron::QorValue::Str(tagline),
                    ) = &parts[1]
                    {
                        println!("  \x1b[2;3m\"{}\"\x1b[0m", tagline);
                    }
                }
            }
        }
        let _ = ctx; // suppress unused warning
    }
    let source_label = if dna_id.is_some() {
        format!("{} + dna/{}", lang_source, dna_id.as_deref().unwrap_or(""))
    } else {
        lang_source
    };
    println!("  \x1b[2mLoaded {} facts, {} rules from {}\x1b[0m",
        session.fact_count(), session.rule_count(), source_label);
    println!("  \x1b[2mAsk me anything. Type 'quit' to exit, 'help' for commands.\x1b[0m");
    println!();

    let stdin = io::stdin();

    loop {
        print!("  \x1b[31m<3\x1b[0m \x1b[1myou>\x1b[0m ");
        io::stdout().flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 {
            break; // EOF
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Exit commands
        match line.to_lowercase().as_str() {
            "quit" | "exit" | "bye" | ":quit" | ":q" => {
                let ctx = qor_bridge::language::extract_brain_context(session.all_facts());
                let farewell = ctx.personality_text(
                    "farewell", session.fact_count(), "Goodbye! Keep reasoning.",
                );
                println!("  \x1b[2m{}\x1b[0m\n", farewell);
                break;
            }
            _ => {}
        }

        // Teach pattern: "X means Y", "define X as Y", "remember X is Y"
        if let Some((topic, desc)) = qor_bridge::language::parse_teach_pattern(line) {
            let fact = format!("(meaning {} {}) <0.99, 0.90>", topic, desc);
            match session.exec(&fact) {
                Ok(_) => {
                    // Save to language/dictionary.qor for persistence
                    qor_bridge::language::save_to_dictionary(&lang_dir, &fact);
                    let ctx = qor_bridge::language::extract_brain_context(session.all_facts());
                    let ack = ctx.personality_text(
                        "teach-ack",
                        session.fact_count(),
                        &format!("Learned: {} means {}", topic, desc.replace('-', " ")),
                    );
                    println!();
                    println!("  \x1b[36mqor>\x1b[0m {}", ack);
                    println!("  \x1b[2m     (saved to dictionary)\x1b[0m");
                    println!();
                }
                Err(e) => eprintln!("  \x1b[31merror:\x1b[0m {}", e),
            }
            continue;
        }

        // Direct QOR syntax passthrough
        if line.starts_with('(') || line.starts_with('?') {
            match session.exec(line) {
                Ok(results) => {
                    for result in results {
                        display_exec_result(&result);
                    }
                }
                Err(e) => eprintln!("  \x1b[31merror:\x1b[0m {}", e),
            }
            continue;
        }

        // Natural language processing via reasoning
        let tokens = qor_bridge::language::tokenize(line);
        if let Err(e) = session.exec_statements(tokens) {
            eprintln!("  \x1b[31merror:\x1b[0m {}", e);
            session.clear_turn();
            continue;
        }

        // Forward chaining: grammar rules fire → intent → response
        for _ in 0..5 {
            session.heartbeat();
        }

        // Read response facts and format with brain context
        let responses = session.response_facts();
        let ctx = qor_bridge::language::extract_brain_context(session.all_facts());
        let reply = qor_bridge::language::format_response(&responses, Some(&ctx));

        println!();
        for reply_line in reply.lines() {
            println!("  \x1b[36mqor>\x1b[0m {}", reply_line);
        }
        println!();

        // Clear ephemeral facts for next turn
        session.clear_turn();
    }
}

// ── REPL ────────────────────────────────────────────────────────────────

fn repl() {
    println!();
    println!("  \x1b[1;36m~~~ QOR v{} ~~~\x1b[0m", env!("CARGO_PKG_VERSION"));
    println!("  \x1b[2mContinuous Reasoning Engine\x1b[0m");
    println!("  \x1b[2mType :help for commands, :quit to exit\x1b[0m");
    println!();

    let stdin = io::stdin();
    let mut session = qor_runtime::eval::Session::new();

    loop {
        // Prompt with heartbeat indicator
        let heart = if session.fact_count() > 0 { "\x1b[31m<3\x1b[0m" } else { "\x1b[2m..\x1b[0m" };
        print!("{} \x1b[1mqor>\x1b[0m ", heart);
        io::stdout().flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 {
            break; // EOF
        }
        let line = line.trim();
        if line.is_empty() {
            // Empty enter = one heartbeat cycle
            if session.heartbeat() {
                println!("  \x1b[2m... reasoning ...\x1b[0m");
            }
            continue;
        }

        match line {
            ":quit" | ":q" | ":exit" => break,
            ":help" | ":h" => print_repl_help(),
            ":facts" | ":f" => show_facts(&session),
            ":rules" | ":r" => show_rules(&session),
            ":stats" | ":s" => show_stats(&session),
            _ if line.starts_with(":load") => {
                let path = line.strip_prefix(":load").unwrap().trim();
                if path.is_empty() {
                    eprintln!("  usage: :load <file.qor>");
                } else {
                    match fs::read_to_string(path) {
                        Ok(source) => {
                            match session.exec(&source) {
                                Ok(results) => {
                                    let mut facts = 0;
                                    let mut rules = 0;
                                    let mut _queries = 0;
                                    for r in &results {
                                        match r {
                                            qor_runtime::eval::ExecResult::Stored { .. } => facts += 1,
                                            qor_runtime::eval::ExecResult::RuleAdded { .. } => rules += 1,
                                            qor_runtime::eval::ExecResult::Query(_) => {
                                                _queries += 1;
                                                display_exec_result(r);
                                            }
                                        }
                                    }
                                    println!("  \x1b[32mloaded\x1b[0m {} facts, {} rules from {}", facts, rules, path);
                                }
                                Err(e) => eprintln!("  \x1b[31merror:\x1b[0m {}", e),
                            }
                        }
                        Err(e) => eprintln!("  \x1b[31merror:\x1b[0m could not read '{}': {}", path, e),
                    }
                }
            }
            _ if line.starts_with(":feed") => {
                let path = line.strip_prefix(":feed").unwrap().trim();
                if path.is_empty() {
                    eprintln!("  usage: :feed <file>  (JSON, CSV, Parquet, text, KV — auto-detected)");
                } else {
                    let file_path = std::path::Path::new(path);
                    match qor_bridge::feed_file(file_path) {
                        Ok(statements) => {
                            let count = statements.len();
                            match session.exec_statements(statements) {
                                Ok(results) => {
                                    let derived: usize = results.iter().map(|r| match r {
                                        qor_runtime::eval::ExecResult::Stored { derived, .. } => *derived,
                                        qor_runtime::eval::ExecResult::RuleAdded { derived } => *derived,
                                        _ => 0,
                                    }).sum();
                                    print!("  \x1b[32mfed\x1b[0m {} facts from {}", count, path);
                                    if derived > 0 {
                                        print!(" \x1b[2m— {} derived\x1b[0m", derived);
                                    }
                                    println!();
                                }
                                Err(e) => eprintln!("  \x1b[31merror:\x1b[0m {}", e),
                            }
                        }
                        Err(e) => eprintln!("  \x1b[31merror:\x1b[0m {}", e),
                    }
                }
            }
            ":beat" => {
                // Run 10 consolidation cycles
                let mut changes = 0;
                for _ in 0..10 {
                    if session.heartbeat() {
                        changes += 1;
                    }
                }
                if changes > 0 {
                    println!("  \x1b[31m<3\x1b[0m {} cycles, beliefs strengthened", changes);
                } else {
                    println!("  \x1b[2m<3\x1b[0m settled — beliefs are stable");
                }
            }
            _ => {
                match session.exec(line) {
                    Ok(results) => {
                        for result in results {
                            display_exec_result(&result);
                        }
                    }
                    Err(e) => {
                        eprintln!("  \x1b[31merror:\x1b[0m {}", e);
                    }
                }
            }
        }
    }

    println!("\n  Goodbye.\n");
}

fn display_exec_result(result: &qor_runtime::eval::ExecResult) {
    use qor_runtime::eval::ExecResult;

    match result {
        ExecResult::Stored { neuron, derived } => {
            print!("  \x1b[32mstored\x1b[0m {}", neuron);
            if *derived > 0 {
                print!(" \x1b[2m— {} new fact{} derived\x1b[0m", derived, if *derived == 1 { "" } else { "s" });
            }
            println!();
        }
        ExecResult::RuleAdded { derived } => {
            print!("  \x1b[33mrule added\x1b[0m");
            if *derived > 0 {
                print!(" \x1b[2m— {} new fact{} derived\x1b[0m", derived, if *derived == 1 { "" } else { "s" });
            }
            println!();
        }
        ExecResult::Query(qr) => {
            println!("  \x1b[36m?\x1b[0m {}", qr.pattern);
            for r in &qr.results {
                if r.contains("(inferred)") {
                    println!("    \x1b[33m->\x1b[0m {}", r);
                } else if r == "no results" {
                    println!("    \x1b[2m-> no results\x1b[0m");
                } else {
                    println!("    -> {}", r);
                }
            }
        }
    }
}

fn show_facts(session: &qor_runtime::eval::Session) {
    let facts = session.all_facts();
    if facts.is_empty() {
        println!("  \x1b[2mno facts\x1b[0m");
    } else {
        for f in facts {
            if f.inferred {
                println!("  \x1b[33m{}\x1b[0m", f);
            } else {
                println!("  {}", f);
            }
        }
        println!("  \x1b[2m{} facts total ({} inferred)\x1b[0m",
            facts.len(),
            facts.iter().filter(|f| f.inferred).count());
    }
}

fn show_rules(session: &qor_runtime::eval::Session) {
    if session.rule_count() == 0 {
        println!("  \x1b[2mno rules\x1b[0m");
    } else {
        println!("  \x1b[2m{} rule{}\x1b[0m", session.rule_count(),
            if session.rule_count() == 1 { "" } else { "s" });
    }
}

fn show_stats(session: &qor_runtime::eval::Session) {
    println!("  facts:  {}", session.fact_count());
    println!("  rules:  {}", session.rule_count());
    println!("  cycles: {}", session.consolidation_cycles());
}

fn print_repl_help() {
    println!("  \x1b[1mQOR REPL Commands:\x1b[0m");
    println!();
    println!("  \x1b[1mStatements:\x1b[0m");
    println!("    (bird tweety) <0.99>            Assert a fact");
    println!("    (flies $x) if (bird $x) <0.95>  Add a rule");
    println!("    ? (flies $x)                     Query");
    println!();
    println!("  \x1b[1mNegation:\x1b[0m");
    println!("    (can-fly $x) if (bird $x) not (penguin $x) <0.95>");
    println!();
    println!("  \x1b[1mCommands:\x1b[0m");
    println!("    :load <file.qor>   Load a .qor file into this session");
    println!("    :feed <file>       Ingest any data (JSON, CSV, text, KV)");
    println!("    :facts             Show all stored facts");
    println!("    :rules             Show rule count");
    println!("    :stats             Show stats");
    println!("    :beat              Run 10 consolidation cycles");
    println!("    :help              Show this help");
    println!("    :quit              Exit REPL");
    println!();
    println!("  \x1b[2mPress Enter with empty line = one heartbeat cycle\x1b[0m");
}

fn heartbeat_command() {
    let args: Vec<String> = env::args().collect();
    let once = args.iter().any(|a| a == "--once");
    let status = args.iter().any(|a| a == "--status");
    let web_once = args.iter().position(|a| a == "--web-once");

    let brain = brain_dir();
    let _ = fs::create_dir_all(&brain);

    let mut hb = qor_runtime::heartbeat::Heartbeat::load(&brain);

    if status {
        println!();
        println!("  \x1b[1;35mQOR Heartbeat Status\x1b[0m");
        println!();
        println!("  Total solved:  {}", hb.total_solved());
        println!("  Library rules: {}", hb.library_size());
        println!("  Brain dir:     {}", brain.display());
        println!();
        return;
    }

    // Handle --web-once <url>: crawl a single URL, extract facts, display
    if let Some(idx) = web_once {
        let url = match args.get(idx + 1) {
            Some(u) => u.clone(),
            None => {
                eprintln!("error: --web-once requires a URL");
                eprintln!("usage: qor heartbeat --web-once <url>");
                process::exit(1);
            }
        };
        web_once_command(&url, &brain);
        return;
    }

    // Load hand-written puzzle rules if available
    let rules_path = dna_dir().join("puzzle_solver").join("rules.qor");
    let rules_qor = fs::read_to_string(&rules_path).unwrap_or_default();

    // Tasks would be loaded and converted to Statement facts by the caller.
    // For now, heartbeat works with whatever the library has learned.
    let tasks: Vec<qor_runtime::heartbeat::TaskData> = Vec::new();

    // Web search: load config, search for candidate rules
    let web_seeds = load_web_seeds(&brain);

    if once {
        println!();
        println!("  \x1b[1;35mQOR Heartbeat — Single Pulse\x1b[0m");
        println!();
        let result = hb.pulse(&tasks, &rules_qor, &web_seeds);
        println!("  New solves:    {}", result.new_solves);
        println!("  Total solved:  {}", result.total_solved);
        println!("  Library rules: {}", result.library_size);
        println!("  Mutations:     {}", result.mutations_tried);
        println!("  Duration:      {}ms", result.pulse_duration_ms);
        if result.web_facts_extracted > 0 {
            println!("  Web seeds:     {}", result.web_facts_extracted);
        }
        println!();
    } else {
        println!();
        println!("  \x1b[1;35mQOR Heartbeat — Continuous Learning\x1b[0m");
        println!("  \x1b[2mPress Ctrl+C to stop\x1b[0m");
        println!();

        let interval = args.iter()
            .position(|a| a == "--interval")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(30);

        loop {
            let result = hb.pulse(&tasks, &rules_qor, &web_seeds);
            eprintln!(
                "  \x1b[35mpulse:\x1b[0m +{} solved, {} total, {} rules, {}ms{}",
                result.new_solves, result.total_solved,
                result.library_size, result.pulse_duration_ms,
                if result.web_facts_extracted > 0 {
                    format!(" (+{} web seeds)", result.web_facts_extracted)
                } else {
                    String::new()
                },
            );

            if result.new_solves == 0 {
                std::thread::sleep(std::time::Duration::from_secs(interval * 2));
            } else {
                std::thread::sleep(std::time::Duration::from_secs(interval));
            }
        }
    }
}

/// Load web-sourced candidate rule texts.
///
/// Reads brain/web_config.qor for topics + sources, then searches the web
/// (if web feature enabled) for candidate rules. Falls back to empty if
/// no config or web feature disabled.
fn load_web_seeds(brain: &std::path::Path) -> Vec<String> {
    // Load web config from brain/web_config.qor
    let config_path = brain.join("web_config.qor");
    let config = if config_path.exists() {
        let text = fs::read_to_string(&config_path).unwrap_or_default();
        let stmts = qor_core::parser::parse(&text).unwrap_or_default();
        qor_bridge::web_fetch::parse_web_config(&stmts)
    } else {
        qor_bridge::web_fetch::WebConfig::default()
    };

    if !config.enabled || config.topics.is_empty() {
        return Vec::new();
    }

    // Web search requires the web feature
    #[cfg(feature = "web")]
    {
        let policy_path = brain.join("domain_policy.conf");
        let policy = if policy_path.exists() {
            let text = fs::read_to_string(&policy_path).unwrap_or_default();
            qor_bridge::web_fetch::parse_domain_policy(&text, "general")
        } else {
            qor_bridge::web_fetch::DomainPolicy::default()
        };

        let cache_dir = brain.join("web");
        let result = qor_bridge::web_search::search_from_config(
            &config, &cache_dir, &policy,
        );

        if result.pages_fetched > 0 {
            eprintln!(
                "  \x1b[36mweb:\x1b[0m fetched {} pages, extracted {} candidate rules",
                result.pages_fetched, result.rules_extracted,
            );
        }

        result.candidate_rules
    }

    #[cfg(not(feature = "web"))]
    {
        Vec::new()
    }
}

/// Crawl a single URL, extract QOR facts, display results.
/// Works without web feature by reading from cache or local file.
fn web_once_command(url: &str, brain: &std::path::Path) {
    println!();
    println!("  \x1b[1;35mQOR Web Intelligence — Single URL\x1b[0m");
    println!("  URL: {}", url);
    println!();

    // Check if it's a local file path (for testing without network)
    let page = if std::path::Path::new(url).exists() {
        let text = fs::read_to_string(url).unwrap_or_default();
        qor_bridge::web_fetch::PageContent {
            url: url.to_string(),
            text,
            title: url.to_string(),
        }
    } else {
        // Try loading from cache
        let cache = qor_bridge::web_fetch::WebCache::new(brain.join("web"));
        if let Some(cached) = cache.load(url) {
            println!("  \x1b[2m(loaded from cache)\x1b[0m");
            cached
        } else {
            eprintln!("  \x1b[31merror:\x1b[0m URL not in cache and web feature not enabled");
            eprintln!("  Build with: cargo build --features web");
            eprintln!("  Or provide a local file path to test extraction");
            process::exit(1);
        }
    };

    // Extract facts
    let facts = qor_bridge::web_fetch::extract_facts_from_pages(&[page]);

    println!("  \x1b[32mExtracted {} facts:\x1b[0m", facts.len());
    println!();

    for fact in &facts {
        println!("    {:?}", fact);
    }

    // Also extract rules separately for display
    let text = if std::path::Path::new(url).exists() {
        fs::read_to_string(url).unwrap_or_default()
    } else {
        String::new()
    };
    let rules = qor_bridge::web_rules::extract_rules(&text, url);
    if !rules.is_empty() {
        println!();
        println!("  \x1b[32mCandidate rules ({}):\x1b[0m", rules.len());
        for rule in &rules {
            println!("    {} \x1b[2m(from: {})\x1b[0m", rule.rule_text,
                &rule.source_sentence[..rule.source_sentence.len().min(60)]);
        }
    }
    println!();
}

fn meta_dir() -> std::path::PathBuf {
    // meta/ lives at workspace root, same as brain/ and dna/
    let mut dir = env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."));

    for _ in 0..5 {
        if dir.join("meta").exists() {
            return dir.join("meta");
        }
        if dir.join("Cargo.toml").exists() {
            return dir.join("meta");
        }
        if let Some(parent) = dir.parent() {
            dir = parent.to_path_buf();
        } else {
            break;
        }
    }

    std::path::PathBuf::from("meta")
}

fn perceive_command() {
    let args: Vec<String> = env::args().collect();
    let once = args.iter().any(|a| a == "--once");
    let sources_flag = args.iter().any(|a| a == "--sources");

    let brain = brain_dir();

    // Load source configuration from brain/sources.qor
    let sources_path = brain.join("sources.qor");
    if !sources_path.exists() {
        eprintln!("error: no brain/sources.qor found");
        eprintln!("Create brain/sources.qor with (source <id> <url> <format> <interval>) entries");
        process::exit(1);
    }

    let sources_text = fs::read_to_string(&sources_path).unwrap_or_default();
    let source_stmts = qor_core::parser::parse(&sources_text).unwrap_or_default();
    let sources = qor_bridge::perceive::parse_sources(&source_stmts);

    if sources.is_empty() {
        eprintln!("error: no sources configured in brain/sources.qor");
        process::exit(1);
    }

    // --sources flag: just list configured sources
    if sources_flag {
        println!();
        println!("  \x1b[1;36mConfigured Perception Sources\x1b[0m");
        println!();
        for s in &sources {
            println!("  \x1b[32m*\x1b[0m {:<16} {} \x1b[2m(every {}s, {})\x1b[0m",
                s.id, s.url, s.interval.as_secs(), s.format);
        }
        println!();
        return;
    }

    println!();
    println!("  \x1b[1;36m~~~ QOR Perceive ~~~\x1b[0m");
    println!("  \x1b[2m{} sources configured\x1b[0m", sources.len());
    println!();

    // Create session with all knowledge: brain + KB + meta rules
    let mut session = qor_runtime::eval::Session::new();

    // Load brain knowledge + KB
    if brain.exists() {
        load_brain_into_session(&mut session, &brain);
    }
    // Load ALL meta rules from meta/ folder
    load_meta_into_session(&mut session);

    let mut state = qor_bridge::perceive::PerceptionState::new();

    if once {
        // Single perception cycle
        run_perception_cycle(&mut state, &sources, &mut session);
    } else {
        // Continuous loop with 5-minute default interval
        let interval = args.iter()
            .position(|a| a == "--interval")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(300);

        println!("  \x1b[2mContinuous mode — fetching every {}s. Ctrl+C to stop.\x1b[0m", interval);
        println!();

        loop {
            run_perception_cycle(&mut state, &sources, &mut session);
            std::thread::sleep(std::time::Duration::from_secs(interval));
        }
    }
}

fn run_perception_cycle(
    state: &mut qor_bridge::perceive::PerceptionState,
    sources: &[qor_bridge::perceive::SourceConfig],
    session: &mut qor_runtime::eval::Session,
) {
    let results = state.perceive(sources);

    if results.is_empty() {
        println!("  \x1b[2mno sources due yet\x1b[0m");
        return;
    }

    let mut total_facts = 0;
    for result in &results {
        if let Some(ref err) = result.error {
            println!("  \x1b[31m!\x1b[0m {}: {}", result.source_id, err);
        } else {
            println!("  \x1b[32m*\x1b[0m {}: {} facts ({} bytes)",
                result.source_id, result.facts.len(), result.raw_size);
            total_facts += result.facts.len();
        }
    }

    // Inject perceived facts into session
    let mut all_facts: Vec<qor_core::neuron::Statement> = Vec::new();
    for result in results {
        all_facts.extend(result.facts);
    }

    if let Err(e) = session.exec_statements(all_facts) {
        eprintln!("  \x1b[31merror:\x1b[0m injecting facts: {}", e);
        return;
    }

    // Run heartbeat to let perception rules reason
    let mut derived = 0;
    for _ in 0..10 {
        if session.heartbeat() {
            derived += 1;
        } else {
            break;
        }
    }

    // Check for promoted facts (significant findings)
    let all = session.all_facts();
    let significant: Vec<_> = all.iter().filter(|f| {
        if let qor_core::neuron::Neuron::Expression(parts) = &f.neuron {
            if let Some(qor_core::neuron::Neuron::Symbol(pred)) = parts.first() {
                return pred == "significant-move"
                    || pred == "market-sentiment"
                    || pred == "weather-alert"
                    || pred == "promote";
            }
        }
        false
    }).collect();

    if !significant.is_empty() {
        println!();
        println!("  \x1b[1;33mSignificant:\x1b[0m");
        for s in &significant {
            println!("    \x1b[33m!\x1b[0m {}", s);
        }
    }

    println!();
    println!("  \x1b[2m[{}f perceived, {} reasoning cycles, {}f total in session]\x1b[0m",
        total_facts, derived, session.fact_count());
    println!();
}

fn build_kb_command() {
    let args: Vec<String> = env::args().collect();

    // qor build-kb --list
    if args.iter().any(|a| a == "--list") {
        println!("Available knowledge sources:");
        println!("  oeis        OEIS integer sequences (math)");
        println!("              Input: oeisdata git repo or dir with stripped.gz + names.gz");
        println!("  dharmic     Vedas, Epics, Gita — sacred texts (scripture)");
        println!("              Input: DharmicData repo with Rigveda/, Mahabharata/ etc.");
        println!("  general     Cross-domain links from meta/ rules (no --input needed)");
        println!("              Scans meta/*.qor and creates domain/instance_of/related_to links");
        println!();
        println!("Coming soon:");
        println!("  dbpedia     DBpedia encyclopedic knowledge (general)");
        println!("  wikidata    Wikidata structured facts (general)");
        return;
    }

    // qor build-kb --stats
    if args.iter().any(|a| a == "--stats") {
        let kb_dir = std::path::Path::new("brain/knowledge");
        if !kb_dir.exists() {
            eprintln!("No knowledge base found at brain/knowledge/");
            eprintln!("Run: qor build-kb --source oeis --input <dir>");
            process::exit(1);
        }
        match qor_runtime::kb::KnowledgeBase::load(kb_dir) {
            Ok(kb) => println!("{}", kb.stats()),
            Err(e) => {
                eprintln!("error loading KB: {}", e);
                process::exit(1);
            }
        }
        return;
    }

    // qor build-kb --source <name> --input <dir>
    let source = args.iter().position(|a| a == "--source")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

    let input = args.iter().position(|a| a == "--input")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

    let source = match source {
        Some(s) => s,
        None => {
            eprintln!("error: --source required");
            eprintln!("usage: qor build-kb --source oeis --input <dir>");
            eprintln!("       qor build-kb --list");
            eprintln!("       qor build-kb --stats");
            process::exit(1);
        }
    };

    // "general" source doesn't need --input (reads from meta/)
    let input_dir = if source == "general" {
        meta_dir() // dummy, not used
    } else {
        match input {
            Some(d) => {
                let p = std::path::PathBuf::from(d);
                if !p.exists() {
                    eprintln!("error: input directory '{}' not found", p.display());
                    process::exit(1);
                }
                p
            }
            None => {
                eprintln!("error: --input required (path to source data)");
                process::exit(1);
            }
        }
    };

    let output_dir = std::path::Path::new("brain/knowledge");
    if let Err(e) = fs::create_dir_all(output_dir) {
        eprintln!("error: could not create {}: {}", output_dir.display(), e);
        process::exit(1);
    }

    // Load existing mappings so new sources don't overwrite old IDs
    let mut encoder = qor_bridge::kb_build::IdEncoder::load_mappings(output_dir)
        .unwrap_or_else(|_| qor_bridge::kb_build::IdEncoder::new());
    let existing_entities = encoder.entity_count();
    let existing_predicates = encoder.predicate_count();
    if existing_entities > 0 {
        println!("  Loaded existing mappings: {} entities, {} predicates",
            existing_entities, existing_predicates);
        println!();
    }
    let mut all_stats = Vec::new();

    match source {
        "oeis" => {
            println!("Building OEIS knowledge base...");
            println!("  Input:  {}", input_dir.display());
            println!("  Output: {}", output_dir.display());
            println!();

            match qor_bridge::oeis::run_oeis_pipeline(&input_dir, output_dir, &mut encoder) {
                Ok(stats) => {
                    all_stats.push(stats);
                }
                Err(e) => {
                    eprintln!("error: OEIS pipeline failed: {}", e);
                    process::exit(1);
                }
            }
        }
        "dharmic" => {
            println!("Building Dharmic knowledge base...");
            println!("  Input:  {}", input_dir.display());
            println!("  Output: {}", output_dir.display());
            println!();

            match qor_bridge::dharmic::run_dharmic_pipeline(&input_dir, output_dir, &mut encoder) {
                Ok(stats) => {
                    all_stats.push(stats);
                }
                Err(e) => {
                    eprintln!("error: Dharmic pipeline failed: {}", e);
                    process::exit(1);
                }
            }
        }
        "general" => {
            let meta = meta_dir();
            println!("Building cross-domain knowledge links...");
            println!("  Meta:   {}", meta.display());
            println!("  Output: {}", output_dir.display());
            println!();

            match build_general_kb(&meta, output_dir, &mut encoder) {
                Ok(stats) => {
                    all_stats.push(stats);
                }
                Err(e) => {
                    eprintln!("error: General KB failed: {}", e);
                    process::exit(1);
                }
            }
        }
        other => {
            eprintln!("error: unknown source '{}'. Run: qor build-kb --list", other);
            process::exit(1);
        }
    }

    // Save shared mappings
    if let Err(e) = encoder.save_mappings(output_dir) {
        eprintln!("error saving mappings: {}", e);
        process::exit(1);
    }

    // Save manifest
    if let Err(e) = qor_bridge::kb_build::write_manifest(output_dir, &all_stats) {
        eprintln!("error saving manifest: {}", e);
        process::exit(1);
    }

    println!();
    println!("Knowledge base built successfully!");
    println!("  Entities:   {}", encoder.entity_count());
    println!("  Predicates: {}", encoder.predicate_count());
    println!("  Output:     {}/", output_dir.display());
    println!();
    println!("Run 'qor build-kb --stats' to see graph statistics.");
}

/// Build cross-domain knowledge links from meta/*.qor files.
/// Scans all meta-rule files, extracts entities and predicates mentioned,
/// creates instance_of, domain, and related_to triples connecting concepts
/// across domains (math ↔ physics ↔ chemistry ↔ vedic science ↔ ...).
fn build_general_kb(
    meta_dir: &std::path::Path,
    output_dir: &std::path::Path,
    encoder: &mut qor_bridge::kb_build::IdEncoder,
) -> Result<qor_bridge::kb_build::SourceStats, String> {
    if !meta_dir.exists() {
        return Err(format!("meta/ directory not found: {}", meta_dir.display()));
    }

    let entries = fs::read_dir(meta_dir)
        .map_err(|e| format!("cannot read {}: {}", meta_dir.display(), e))?;

    let mut files: Vec<_> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "qor"))
        .collect();
    files.sort_by_key(|e| e.path());

    if files.is_empty() {
        return Err("no .qor files found in meta/".into());
    }

    let out_path = output_dir.join("general_triples.bin");
    let mut writer = std::io::BufWriter::new(
        fs::File::create(&out_path)
            .map_err(|e| format!("cannot create {}: {}", out_path.display(), e))?,
    );

    let mut triples: u64 = 0;
    let mut entities: u64 = 0;

    // Predicate IDs for cross-domain links
    let domain_pid = encoder.predicate_id("domain");
    let instance_of_pid = encoder.predicate_id("instance_of");
    let related_to_pid = encoder.predicate_id("related_to");

    // Track entities per domain for cross-domain linking
    let mut domain_entities: std::collections::HashMap<String, Vec<u32>> =
        std::collections::HashMap::new();

    for entry in &files {
        let path = entry.path();
        let domain = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let source = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let stmts = match qor_core::parser::parse(&source) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let domain_eid = encoder.entity_id(&domain);
        let domain_concept = encoder.entity_id("knowledge-domain");

        // domain is-a knowledge-domain
        qor_bridge::kb_build::write_triple(
            &mut writer, domain_eid, instance_of_pid, domain_concept, 1.0, 0.95,
        ).map_err(|e| e.to_string())?;
        triples += 1;

        // Extract all unique predicates and entities from the facts
        let mut domain_ents = Vec::new();
        for stmt in &stmts {
            if let qor_core::neuron::Statement::Fact { neuron, .. } = stmt {
                if let qor_core::neuron::Neuron::Expression(parts) = neuron {
                    // First element = predicate
                    if let Some(qor_core::neuron::Neuron::Symbol(pred)) = parts.first() {
                        let pred_eid = encoder.entity_id(pred);
                        // predicate belongs-to domain
                        qor_bridge::kb_build::write_triple(
                            &mut writer, pred_eid, domain_pid, domain_eid, 0.95, 0.90,
                        ).map_err(|e| e.to_string())?;
                        triples += 1;
                        entities += 1;

                        // Other elements = entities
                        for part in parts.iter().skip(1) {
                            if let qor_core::neuron::Neuron::Symbol(s) = part {
                                if !s.starts_with('$') && s.len() > 1 {
                                    let eid = encoder.entity_id(s);
                                    qor_bridge::kb_build::write_triple(
                                        &mut writer, eid, domain_pid, domain_eid, 0.90, 0.85,
                                    ).map_err(|e| e.to_string())?;
                                    triples += 1;
                                    entities += 1;
                                    domain_ents.push(eid);
                                }
                            }
                        }
                    }
                }
            }
        }

        domain_ents.sort();
        domain_ents.dedup();
        domain_entities.insert(domain.clone(), domain_ents);

        println!("  {} — {} statements", domain, stmts.len());
    }

    // Cross-domain links: find entities that appear in multiple domains
    let all_domains: Vec<String> = domain_entities.keys().cloned().collect();
    for i in 0..all_domains.len() {
        for j in (i + 1)..all_domains.len() {
            let ents_a = &domain_entities[&all_domains[i]];
            let ents_b = &domain_entities[&all_domains[j]];
            // Find shared entities
            let mut shared = 0;
            for &ea in ents_a {
                if ents_b.contains(&ea) {
                    shared += 1;
                }
            }
            if shared > 0 {
                let da = encoder.entity_id(&all_domains[i]);
                let db = encoder.entity_id(&all_domains[j]);
                qor_bridge::kb_build::write_triple(
                    &mut writer, da, related_to_pid, db, 0.85, 0.80,
                ).map_err(|e| e.to_string())?;
                qor_bridge::kb_build::write_triple(
                    &mut writer, db, related_to_pid, da, 0.85, 0.80,
                ).map_err(|e| e.to_string())?;
                triples += 2;
                println!("  {} <-> {} ({} shared concepts)", all_domains[i], all_domains[j], shared);
            }
        }
    }

    use std::io::Write;
    writer.flush().map_err(|e| e.to_string())?;

    Ok(qor_bridge::kb_build::SourceStats {
        name: "general".into(),
        domain: "cross-domain".into(),
        entities,
        triples,
        formulas: 0,
    })
}

fn print_usage() {
    println!("QORlang - Quantified Ontological Reasoning Language");
    println!();
    println!("USAGE:");
    println!("  qor run <file.qor>       Run a .qor file");
    println!("  qor check <file.qor>     Check syntax + show warnings");
    println!("  qor test <file.qor>      Run @test blocks in a .qor file");
    println!("  qor feed <file>          Ingest any data (JSON, CSV, Parquet, text, KV)");
    println!("  qor think                Reason on brain knowledge, show insights");
    println!("  qor think --watch        Monitor brain/ and re-reason on changes");
    println!("  qor explain \"(pattern)\"  Trace WHY a fact is believed");
    println!("  qor summary              Natural language summary of brain knowledge");
    println!("  qor chat                 Talk to QOR — ask questions about its knowledge");
    println!("  qor chat --dna <id>      Chat with a profession DNA loaded");
    println!("  qor dna list             List available DNA profiles");
    println!("  qor dna convert          Convert all DNA profiles to .qor");
    println!("  qor dna convert <id>     Convert a single DNA profile to .qor");
    println!("  qor solve <file.qor>     Test hypotheses to resolve unanswered queries");
    println!("  qor heartbeat              Start continuous learning (self-improving)");
    println!("  qor heartbeat --once       Run a single learning pulse");
    println!("  qor heartbeat --status     Show learning progress");
    println!("  qor heartbeat --web-once <url>  Crawl URL, extract facts + rules");
    println!("  qor perceive               Perceive real-time data (continuous)");
    println!("  qor perceive --once        Single perception cycle");
    println!("  qor perceive --sources     List configured data sources");
    println!("  qor build-kb --source <name> --input <dir>  Build binary knowledge base");
    println!("  qor build-kb --stats       Show knowledge graph statistics");
    println!("  qor build-kb --list        List available sources");
    println!("  qor repl                 Interactive reasoning session");
    println!("  qor version              Show version");
    println!("  qor help                 Show this help");
}
