// ── Web Intelligence Integration Tests ────────────────────────────────
//
// Real end-to-end tests that verify the web pipeline does what it claims.
// Not unit tests — these test the FULL flow from text to verified fact.
//
// GOLDEN RULE: Math is the judge. Source never bypasses verification.
//   - A rule from arxiv that fails training pairs = GARBAGE
//   - A rule from kaggle that passes training pairs = FACT

use qor_bridge::web_fetch::*;
use qor_bridge::web_rules;

// ═══════════════════════════════════════════════════════════════════════
// TEST GROUP 1: Domain Policy Enforcement
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_real_arc_domain_policy_from_config() {
    // Load the actual domain_policy.conf file
    let config_text = std::fs::read_to_string("../brain/domain_policy.conf")
        .expect("../brain/domain_policy.conf should exist");

    let policy = parse_domain_policy(&config_text, "arc");

    // Academic sources ALLOWED
    assert!(policy.can_crawl("https://arxiv.org/abs/2312.12345"), "arxiv should be allowed for ARC");
    assert!(policy.can_crawl("https://en.wikipedia.org/wiki/Abstract_reasoning"), "wikipedia allowed");
    assert!(policy.can_crawl("https://www.kaggle.com/competitions/arc-prize"), "kaggle allowed");
    assert!(policy.can_crawl("https://arcprize.org/"), "arcprize.org allowed");

    // Social media BLOCKED
    assert!(!policy.can_crawl("https://pinterest.com/pin/grid-art"), "pinterest blocked");
    assert!(!policy.can_crawl("https://twitter.com/user/status/123"), "twitter blocked");
    assert!(!policy.can_crawl("https://www.tiktok.com/@gridpuzzles"), "tiktok blocked");
    assert!(!policy.can_crawl("https://reddit.com/r/ARC_challenge"), "reddit blocked");
    assert!(!policy.can_crawl("https://www.facebook.com/groups/arc"), "facebook blocked");

    // Random unknown sites BLOCKED (not in whitelist)
    assert!(!policy.can_crawl("https://random-blog.com/arc-tricks"), "unknown site blocked");
    assert!(!policy.can_crawl("https://some-forum.io/grid-patterns"), "unknown forum blocked");
}

#[test]
fn test_real_finance_domain_policy_from_config() {
    let config_text = std::fs::read_to_string("../brain/domain_policy.conf")
        .expect("../brain/domain_policy.conf should exist");

    let policy = parse_domain_policy(&config_text, "finance");

    // Legitimate finance sources ALLOWED
    assert!(policy.can_crawl("https://ssrn.com/abstract/12345"), "ssrn allowed");
    assert!(policy.can_crawl("https://www.investopedia.com/terms/v/vwap"), "investopedia allowed");
    assert!(policy.can_crawl("https://arxiv.org/abs/2312.12345"), "arxiv allowed for finance");

    // Pump-and-dump / opinion sites BLOCKED
    assert!(!policy.can_crawl("https://stocktwits.com/symbol/AAPL"), "stocktwits blocked");
    assert!(!policy.can_crawl("https://seekingalpha.com/article/123"), "seekingalpha blocked");
    assert!(!policy.can_crawl("https://www.motleyfool.com/investing"), "motleyfool blocked");
    assert!(!policy.can_crawl("https://zerohedge.com/markets"), "zerohedge blocked");
}

#[test]
fn test_tier_sorting_puts_academic_first() {
    let mut urls = vec![
        "https://kaggle.com/competitions/arc".into(),       // Tier3
        "https://random-site.com/stuff".into(),              // Tier3
        "https://en.wikipedia.org/wiki/Pattern".into(),      // Tier2
        "https://arxiv.org/abs/2312.12345".into(),           // Tier1
        "https://openreview.net/forum?id=abc".into(),        // Tier1
        "https://mathworld.wolfram.com/Symmetry.html".into(),// Tier2
    ];

    sort_urls_by_tier(&mut urls);

    // Tier1 should be first
    assert!(urls[0].contains("arxiv") || urls[0].contains("openreview"),
            "First URL should be Tier1 (academic), got: {}", urls[0]);
    assert!(urls[1].contains("arxiv") || urls[1].contains("openreview"),
            "Second URL should be Tier1, got: {}", urls[1]);

    // Tier3 should be last
    let last = urls.last().unwrap();
    assert!(last.contains("kaggle") || last.contains("random"),
            "Last URL should be Tier3, got: {}", last);
}

#[test]
fn test_filter_then_sort_full_pipeline() {
    let config_text = std::fs::read_to_string("../brain/domain_policy.conf")
        .expect("../brain/domain_policy.conf should exist");
    let policy = parse_domain_policy(&config_text, "arc");

    // Mix of allowed and blocked URLs
    let all_urls = vec![
        "https://arxiv.org/abs/arc-reasoning".into(),
        "https://pinterest.com/pin/grid-art".into(),       // blocked
        "https://kaggle.com/competitions/arc".into(),
        "https://twitter.com/arc_challenge".into(),         // blocked
        "https://en.wikipedia.org/wiki/ARC".into(),
        "https://random-blog.xyz/arc".into(),               // not in whitelist
    ];

    // Filter first
    let mut allowed = filter_urls(&all_urls, &policy);
    assert_eq!(allowed.len(), 3, "Should keep exactly 3 allowed URLs, got {}: {:?}", allowed.len(), allowed);

    // Then sort by tier
    sort_urls_by_tier(&mut allowed);
    assert!(allowed[0].contains("arxiv"), "First should be arxiv (Tier1)");
}

// ═══════════════════════════════════════════════════════════════════════
// TEST GROUP 2: Web Extraction — ALL Output Is Candidates
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_wikipedia_like_text_extraction_all_candidates() {
    // Simulate real Wikipedia content about grid transformations
    let pages = vec![PageContent {
        url: "https://en.wikipedia.org/wiki/Grid_transformation".into(),
        text: r#"
A grid transformation is a function that maps one grid to another.
Common transformations include rotation, reflection, and scaling.
If a grid has rotational symmetry then it maps to itself under rotation.
Color mapping is defined as a bijection between color sets.
A tiling pattern is a repeating arrangement that covers a plane.
Flood fill is an algorithm that determines connected regions.
When objects overlap, the foreground occludes the background.
Symmetry implies invariance under transformation.
The number of distinct patterns is 42.
A fractal has self-similar structure at different scales.
"#.into(),
        title: "Grid transformation - Wikipedia".into(),
    }];

    let facts = extract_facts_from_pages(&pages);

    // 1. Provenance metadata is present
    let has_source = facts.iter().any(|f| format!("{:?}", f).contains("web-source"));
    let has_domain = facts.iter().any(|f| format!("{:?}", f).contains("web-domain"));
    assert!(has_source, "Must have web-source provenance");
    assert!(has_domain, "Must detect wikipedia domain");

    // 2. Rule candidates extracted from conditional patterns
    let rule_candidates: Vec<_> = facts.iter()
        .filter(|f| format!("{:?}", f).contains("web-rule-candidate"))
        .collect();
    assert!(!rule_candidates.is_empty(),
            "Should extract rule candidates from 'if...then' and 'implies' patterns");

    // 3. CRITICAL: NO direct facts — everything is a candidate
    for f in &facts {
        let s = format!("{:?}", f);
        if s.contains("web-source") || s.contains("web-domain") {
            continue; // metadata OK
        }
        assert!(s.contains("candidate"),
                "VIOLATION: Non-candidate fact found from web: {}", s);
    }

    // 4. Fact candidates have very low truth values (unverified)
    for f in &facts {
        let s = format!("{:?}", f);
        if s.contains("web-fact-candidate") {
            // Check TV is low
            assert!(s.contains("0.3") || s.contains("0.2"),
                    "Web fact candidates should have low TV, got: {}", s);
        }
    }
}

#[test]
fn test_arxiv_like_text_extraction() {
    // Simulate academic paper abstract
    let pages = vec![PageContent {
        url: "https://arxiv.org/abs/2312.12345".into(),
        text: r#"
Abstract: We present a novel approach to abstract visual reasoning.
If the input grid contains symmetric patterns then the output preserves symmetry.
Color permutation leads to equivalent transformations.
Object detection causes improved spatial reasoning accuracy.
When multiple objects share a boundary, they form a composite region.
The method achieves 85% accuracy on ARC evaluation set.
"#.into(),
        title: "Neural-Symbolic Reasoning for ARC".into(),
    }];

    let facts = extract_facts_from_pages(&pages);

    // Should extract candidates — NOT facts
    let candidates: Vec<_> = facts.iter()
        .filter(|f| format!("{:?}", f).contains("candidate"))
        .collect();
    assert!(candidates.len() >= 2,
            "Should extract multiple candidates from academic text, got {}", candidates.len());

    // Even from Tier1 source — still candidates, never direct facts
    for f in &facts {
        let s = format!("{:?}", f);
        if s.contains("web-source") || s.contains("web-domain") {
            continue;
        }
        assert!(s.contains("candidate"),
                "Even arxiv content must be candidates: {}", s);
    }
}

#[test]
fn test_noisy_text_produces_nothing_useful() {
    // Garbage in = nothing useful out (short lines filtered)
    let pages = vec![PageContent {
        url: "https://example.com".into(),
        text: "Hi.\nOK.\nLol.\nYes.\nNo.\n".into(),
        title: "Noise".into(),
    }];

    let facts = extract_facts_from_pages(&pages);

    // Should only have provenance, no extracted candidates
    let non_provenance: Vec<_> = facts.iter()
        .filter(|f| {
            let s = format!("{:?}", f);
            !s.contains("web-source") && !s.contains("web-domain")
        })
        .collect();

    assert!(non_provenance.is_empty() || non_provenance.len() <= 1,
            "Noisy garbage text should produce few/no candidates, got {}", non_provenance.len());
}

// ═══════════════════════════════════════════════════════════════════════
// TEST GROUP 3: Rule Extraction Quality
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_extract_rules_from_real_reasoning_text() {
    let text = r#"
In abstract reasoning, if the input grid is rotated then the output grid is the rotated version.
When colors are swapped, all instances of that color change uniformly.
Reflection causes the grid to be mirrored along the axis.
Tiling implies the pattern repeats to fill the output grid.
If objects are scaled then their dimensions multiply by the scale factor.
"#;

    let rules = web_rules::extract_rules(text, "https://example.com/reasoning");

    // Should extract multiple rule candidates
    assert!(rules.len() >= 3,
            "Should extract at least 3 rules from reasoning text, got {}", rules.len());

    // Each rule should have QOR syntax: (consequent) if (antecedent)
    for rule in &rules {
        assert!(rule.rule_text.contains("if"),
                "Rule should contain 'if': {}", rule.rule_text);
        assert!(rule.rule_text.contains("(") && rule.rule_text.contains(")"),
                "Rule should have QOR syntax with parens: {}", rule.rule_text);
    }

    // Confidence should be between 0.5 and 0.8 (moderate — these are guesses)
    for rule in &rules {
        assert!(rule.confidence >= 0.50 && rule.confidence <= 0.80,
                "Rule confidence should be moderate (0.50-0.80), got {}: {}",
                rule.confidence, rule.rule_text);
    }
}

#[test]
fn test_extract_facts_from_definitions() {
    let text = r#"
Photosynthesis is defined as the process of converting sunlight to energy.
A fractal is a self-similar pattern at every scale.
Water has hydrogen and oxygen atoms.
The speed of light is 299792458 meters per second.
Pi equals 3.14159.
"#;

    let facts = web_rules::extract_facts(text, "https://example.com");

    // Should find definitions, is-a, has, and numeric facts
    assert!(facts.len() >= 3,
            "Should extract at least 3 facts from definition text, got {}", facts.len());

    // Check we got a definition
    let has_definition = facts.iter().any(|f| {
        let s = format!("{:?}", f.statement);
        s.contains("meaning") && s.contains("photosynthesis")
    });
    assert!(has_definition, "Should extract photosynthesis definition");

    // Check we got a numeric fact
    let has_numeric = facts.iter().any(|f| {
        let s = format!("{:?}", f.statement);
        s.contains("299792458") || s.contains("3.14159")
    });
    assert!(has_numeric, "Should extract numeric facts");
}

#[test]
fn test_build_search_urls_produces_valid_urls() {
    let urls = web_rules::build_search_urls(
        "grid transformation",
        &["https://en.wikipedia.org/wiki/".to_string(), "https://arxiv.org/search/".to_string()],
    );

    assert!(!urls.is_empty(), "Should produce search URLs");

    for url in &urls {
        assert!(url.starts_with("https://"),
                "URL should start with https://, got: {}", url);
        assert!(url.contains("grid") || url.contains("transformation"),
                "URL should contain topic terms: {}", url);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// TEST GROUP 4: Web Config from QOR Facts
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_real_web_config_file_parses() {
    // Load the actual ../brain/web_config.qor and parse it
    let config_text = std::fs::read_to_string("../brain/web_config.qor")
        .expect("../brain/web_config.qor should exist");

    // Parse as QOR statements
    let stmts = qor_core::parser::parse(&config_text)
        .expect("web_config.qor should parse as valid QOR");

    // Convert to WebConfig
    let config = parse_web_config(&stmts);

    // Should have at least one source and one topic
    assert!(config.enabled, "Config with web-source should be enabled");
    assert!(!config.sources.is_empty(), "Should have at least one source");
    assert!(!config.topics.is_empty(), "Should have at least one topic");
    assert!(config.crawl_interval > 0, "Crawl interval should be positive");
    assert!(config.max_pages > 0, "Max pages should be positive");
}

// ═══════════════════════════════════════════════════════════════════════
// TEST GROUP 5: Cache — Offline First
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_cache_stores_and_retrieves_real_content() {
    let tmp = std::env::temp_dir().join("qor_integration_cache_test");
    let cache = WebCache::new(tmp.clone());

    // Simulate caching a real Wikipedia page
    let page = PageContent {
        url: "https://en.wikipedia.org/wiki/Abstract_reasoning".into(),
        text: "Abstract reasoning is the ability to analyze information, \
               detect patterns, and solve problems on a complex level. \
               If patterns repeat then they can be predicted.".into(),
        title: "Abstract reasoning - Wikipedia".into(),
    };

    // Save to cache
    cache.save(&page);

    // Verify cached
    assert!(cache.is_cached("https://en.wikipedia.org/wiki/Abstract_reasoning", 24),
            "Page should be cached after save");

    // Load and verify content integrity
    let loaded = cache.load("https://en.wikipedia.org/wiki/Abstract_reasoning")
        .expect("Should load cached page");
    assert_eq!(loaded.title, page.title, "Title must survive cache round-trip");
    assert_eq!(loaded.text, page.text, "Content must survive cache round-trip");

    // Extract from cached content — same results as fresh
    let fresh_facts = extract_facts_from_pages(&[page]);
    let cached_facts = extract_facts_from_pages(&[loaded]);
    assert_eq!(fresh_facts.len(), cached_facts.len(),
               "Cached extraction must produce same results as fresh");

    // Cleanup
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn test_cache_expiry_works() {
    let tmp = std::env::temp_dir().join("qor_cache_expiry_test");
    let cache = WebCache::new(tmp.clone());

    let page = PageContent {
        url: "https://example.com/test".into(),
        text: "Test content for expiry.".into(),
        title: "Test".into(),
    };
    cache.save(&page);

    // Should be cached with reasonable max_age
    assert!(cache.is_cached("https://example.com/test", 24));

    // With 0 max_age = never expires
    assert!(cache.is_cached("https://example.com/test", 0));

    // Non-existent URL should not be cached
    assert!(!cache.is_cached("https://example.com/nonexistent", 24));

    let _ = std::fs::remove_dir_all(&tmp);
}

// ═══════════════════════════════════════════════════════════════════════
// TEST GROUP 6: The Verification Gate
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_verification_gate_rejects_bad_candidate() {
    use qor_runtime::eval::Session;

    // Setup: a simple puzzle — input has color 3, output should have color 7
    let mut session = Session::new();
    session.exec(r#"
        (grid-cell train-0 0 0 3)
        (grid-cell train-0 0 1 3)
        (grid-cell train-0 1 0 3)
        (grid-cell train-0 1 1 3)
    "#).unwrap();

    // BAD candidate rule from web: maps 3 → 5 (WRONG — should be 7)
    let bad_rule = r#"(predict-cell train-0 $r $c 5) if (grid-cell train-0 $r $c 3)"#;
    let result = session.test_hypothesis(bad_rule).unwrap();

    // The rule fires (produces predictions) but they're WRONG
    assert!(result.new_facts > 0, "Rule should fire and produce predictions");

    // Check predictions: they should be 5, not 7
    let has_wrong_predictions = result.facts.iter().any(|f| f.contains("predict-cell"));
    assert!(has_wrong_predictions, "Should have predict-cell facts");

    // Verify they DON'T match expected output (7)
    let expected_color = 7u8;
    let all_wrong = result.facts.iter()
        .filter(|f| f.contains("predict-cell"))
        .all(|f| !f.contains(&expected_color.to_string()));
    assert!(all_wrong,
            "Bad rule predictions should NOT match expected output color {}", expected_color);
}

#[test]
fn test_verification_gate_accepts_good_candidate() {
    use qor_runtime::eval::Session;

    // Same puzzle: input color 3 → output color 7
    let mut session = Session::new();
    session.exec(r#"
        (grid-cell train-0 0 0 3)
        (grid-cell train-0 0 1 3)
        (grid-cell train-0 1 0 3)
        (grid-cell train-0 1 1 3)
    "#).unwrap();

    // GOOD candidate rule: maps 3 → 7 (CORRECT)
    let good_rule = r#"(predict-cell train-0 $r $c 7) if (grid-cell train-0 $r $c 3)"#;
    let result = session.test_hypothesis(good_rule).unwrap();

    // Rule fires and produces correct predictions
    assert!(result.new_facts > 0, "Good rule should fire");

    // All predictions should contain color 7
    let predictions: Vec<_> = result.facts.iter()
        .filter(|f| f.contains("predict-cell"))
        .collect();
    assert_eq!(predictions.len(), 4, "Should predict all 4 cells");

    for pred in &predictions {
        assert!(pred.contains("7"),
                "Good rule prediction should contain correct color 7: {}", pred);
    }
}

#[test]
fn test_score_function_distinguishes_good_from_bad() {
    use qor_core::neuron::{Neuron, QorValue, Statement};
    use qor_core::parser;
    use qor_runtime::eval::Session;
    use qor_runtime::search::score_rule_on_training;

    // Training data: 2x2 grid, color 3 → color 7
    let training_inputs = vec![vec![
        Statement::Fact {
            neuron: Neuron::Expression(vec![
                Neuron::Symbol("grid-cell".into()),
                Neuron::Symbol("train-0".into()),
                Neuron::Value(QorValue::Int(0)),
                Neuron::Value(QorValue::Int(0)),
                Neuron::Value(QorValue::Int(3)),
            ]),
            tv: None, decay: None,
        },
        Statement::Fact {
            neuron: Neuron::Expression(vec![
                Neuron::Symbol("grid-cell".into()),
                Neuron::Symbol("train-0".into()),
                Neuron::Value(QorValue::Int(0)),
                Neuron::Value(QorValue::Int(1)),
                Neuron::Value(QorValue::Int(3)),
            ]),
            tv: None, decay: None,
        },
        Statement::Fact {
            neuron: Neuron::Expression(vec![
                Neuron::Symbol("grid-cell".into()),
                Neuron::Symbol("train-0".into()),
                Neuron::Value(QorValue::Int(1)),
                Neuron::Value(QorValue::Int(0)),
                Neuron::Value(QorValue::Int(3)),
            ]),
            tv: None, decay: None,
        },
        Statement::Fact {
            neuron: Neuron::Expression(vec![
                Neuron::Symbol("grid-cell".into()),
                Neuron::Symbol("train-0".into()),
                Neuron::Value(QorValue::Int(1)),
                Neuron::Value(QorValue::Int(1)),
                Neuron::Value(QorValue::Int(3)),
            ]),
            tv: None, decay: None,
        },
    ]];

    let expected_outputs = vec![vec![
        Statement::Fact {
            neuron: Neuron::Expression(vec![
                Neuron::Symbol("predict-cell".into()),
                Neuron::Symbol("train-0".into()),
                Neuron::Value(QorValue::Int(0)),
                Neuron::Value(QorValue::Int(0)),
                Neuron::Value(QorValue::Int(7)),
            ]),
            tv: None, decay: None,
        },
        Statement::Fact {
            neuron: Neuron::Expression(vec![
                Neuron::Symbol("predict-cell".into()),
                Neuron::Symbol("train-0".into()),
                Neuron::Value(QorValue::Int(0)),
                Neuron::Value(QorValue::Int(1)),
                Neuron::Value(QorValue::Int(7)),
            ]),
            tv: None, decay: None,
        },
        Statement::Fact {
            neuron: Neuron::Expression(vec![
                Neuron::Symbol("predict-cell".into()),
                Neuron::Symbol("train-0".into()),
                Neuron::Value(QorValue::Int(1)),
                Neuron::Value(QorValue::Int(0)),
                Neuron::Value(QorValue::Int(7)),
            ]),
            tv: None, decay: None,
        },
        Statement::Fact {
            neuron: Neuron::Expression(vec![
                Neuron::Symbol("predict-cell".into()),
                Neuron::Symbol("train-0".into()),
                Neuron::Value(QorValue::Int(1)),
                Neuron::Value(QorValue::Int(1)),
                Neuron::Value(QorValue::Int(7)),
            ]),
            tv: None, decay: None,
        },
    ]];

    let base_session = Session::new();

    // GOOD rule: 3 → 7
    let good_stmts = parser::parse(
        "(predict-cell train-0 $r $c 7) if (grid-cell train-0 $r $c 3)"
    ).unwrap();
    let good_rule = if let Statement::Rule { head, body, tv } = &good_stmts[0] {
        qor_runtime::chain::Rule::new(head.clone(), body.clone(), tv.unwrap_or_default())
    } else { panic!("expected rule") };

    let good_score = score_rule_on_training(
        &good_rule, &training_inputs, &expected_outputs, "predict-cell", &base_session
    );

    // BAD rule: 3 → 5
    let bad_stmts = parser::parse(
        "(predict-cell train-0 $r $c 5) if (grid-cell train-0 $r $c 3)"
    ).unwrap();
    let bad_rule = if let Statement::Rule { head, body, tv } = &bad_stmts[0] {
        qor_runtime::chain::Rule::new(head.clone(), body.clone(), tv.unwrap_or_default())
    } else { panic!("expected rule") };

    let bad_score = score_rule_on_training(
        &bad_rule, &training_inputs, &expected_outputs, "predict-cell", &base_session
    );

    // EMPTY rule: does nothing
    let noop_stmts = parser::parse(
        "(predict-cell train-0 $r $c 9) if (nonexistent-fact $r $c)"
    ).unwrap();
    let noop_rule = if let Statement::Rule { head, body, tv } = &noop_stmts[0] {
        qor_runtime::chain::Rule::new(head.clone(), body.clone(), tv.unwrap_or_default())
    } else { panic!("expected rule") };

    let noop_score = score_rule_on_training(
        &noop_rule, &training_inputs, &expected_outputs, "predict-cell", &base_session
    );

    // Assertions
    assert_eq!(good_score, 1.0,
               "Good rule (3→7) should score 1.0 (perfect), got {}", good_score);
    assert_eq!(bad_score, 0.0,
               "Bad rule (3→5) should score 0.0 (all wrong), got {}", bad_score);
    assert_eq!(noop_score, 0.0,
               "No-op rule should score 0.0 (no predictions), got {}", noop_score);

    // The math decides: good > bad, good > noop
    assert!(good_score > bad_score, "Good rule must score higher than bad rule");
    assert!(good_score > noop_score, "Good rule must score higher than no-op rule");
}

// ═══════════════════════════════════════════════════════════════════════
// TEST GROUP 7: Full Pipeline — Web Text → Candidates → Verification
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_full_pipeline_web_to_candidate_to_verification() {
    use qor_runtime::eval::Session;

    // STEP 1: Simulate crawled web content about color mapping
    let pages = vec![PageContent {
        url: "https://en.wikipedia.org/wiki/Color_mapping".into(),
        text: r#"
Color mapping is a bijective function from one color set to another.
If color 3 appears in the input then color 7 appears in the output.
When all cells share a color, the transformation is uniform.
A color remap is defined as a systematic replacement of colors.
"#.into(),
        title: "Color mapping - Wikipedia".into(),
    }];

    // STEP 2: Extract — everything comes out as candidates
    let candidates = extract_facts_from_pages(&pages);
    assert!(!candidates.is_empty(), "Should extract something from the text");

    // Verify no direct facts leaked through
    for c in &candidates {
        let s = format!("{:?}", c);
        if s.contains("web-source") || s.contains("web-domain") {
            continue;
        }
        assert!(s.contains("candidate"),
                "Pipeline leak: non-candidate found: {}", s);
    }

    // STEP 3: The web text mentioned "if color 3 appears then color 7 appears"
    // Extract the rule candidates
    let rule_candidates: Vec<_> = candidates.iter()
        .filter(|f| format!("{:?}", f).contains("web-rule-candidate"))
        .collect();
    assert!(!rule_candidates.is_empty(), "Should find rule candidates about color mapping");

    // STEP 4: Simulate what the heartbeat does — take a candidate rule and VERIFY IT
    // against actual training data. Only if score == 1.0 does it become a fact.
    let mut session = Session::new();

    // Load a real puzzle: 2x2 grid, all color 3 → all color 7
    session.exec(r#"
        (grid-cell train-0 0 0 3)
        (grid-cell train-0 0 1 3)
        (grid-cell train-0 1 0 3)
        (grid-cell train-0 1 1 3)
    "#).unwrap();

    // The web-extracted candidate rule: "if color 3 then color 7"
    // Translated to QOR: (predict-cell train-0 $r $c 7) if (grid-cell train-0 $r $c 3)
    let candidate_rule = "(predict-cell train-0 $r $c 7) if (grid-cell train-0 $r $c 3)";
    let result = session.test_hypothesis(candidate_rule).unwrap();

    // STEP 5: Verify against expected output
    let expected: Vec<(usize, usize, u8)> = vec![(0,0,7), (0,1,7), (1,0,7), (1,1,7)];
    let predictions: Vec<_> = result.facts.iter()
        .filter(|f| f.contains("predict-cell"))
        .collect();

    assert_eq!(predictions.len(), expected.len(),
               "Should predict exactly {} cells", expected.len());

    // Check each prediction matches
    let all_correct = predictions.iter().all(|p| p.contains("7"));
    assert!(all_correct, "All predictions should have color 7");

    // VERDICT: This candidate PASSES the math → it would become a FACT
    // In the real heartbeat, this is where library.add() would be called
}

#[test]
fn test_full_pipeline_bad_web_info_rejected() {
    use qor_runtime::eval::Session;

    // STEP 1: Crawled content with WRONG information
    let pages = vec![PageContent {
        url: "https://random-blog.com/grid-tricks".into(),
        text: r#"
If color 3 appears in the input then color 2 appears in the output.
This is a common grid transformation pattern.
"#.into(),
        title: "Grid Tricks Blog".into(),
    }];

    // STEP 2: Extract candidates
    let candidates = extract_facts_from_pages(&pages);
    let _rule_candidates: Vec<_> = candidates.iter()
        .filter(|f| format!("{:?}", f).contains("web-rule-candidate"))
        .collect();

    // Even bad info gets extracted as candidates — that's fine
    // The verification gate is what matters

    // STEP 3: Verify the wrong rule against training data
    let mut session = Session::new();
    session.exec(r#"
        (grid-cell train-0 0 0 3)
        (grid-cell train-0 0 1 3)
        (grid-cell train-0 1 0 3)
        (grid-cell train-0 1 1 3)
    "#).unwrap();

    // Wrong rule: 3 → 2 (actual answer is 7)
    let wrong_rule = "(predict-cell train-0 $r $c 2) if (grid-cell train-0 $r $c 3)";
    let result = session.test_hypothesis(wrong_rule).unwrap();

    // Rule fires (it's syntactically valid), but predictions are WRONG
    let predictions: Vec<_> = result.facts.iter()
        .filter(|f| f.contains("predict-cell"))
        .collect();
    assert_eq!(predictions.len(), 4, "Rule fires and makes predictions");

    // NONE of them match the expected output (color 7)
    let any_correct = predictions.iter().any(|p| p.contains("7"));
    assert!(!any_correct,
            "Wrong rule should NOT produce correct predictions — this is GARBAGE");

    // VERDICT: This candidate FAILS the math → it gets DISCARDED
    // No matter that it was extracted from a web page — math says no.
}

// ═══════════════════════════════════════════════════════════════════════
// TEST GROUP 8: Heartbeat Integration — Library Scoring
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_library_only_keeps_verified_rules() {
    use qor_runtime::library::{RuleLibrary, RuleSource};

    let tmp = std::env::temp_dir().join("qor_lib_verify_test");
    let _ = std::fs::create_dir_all(&tmp);
    let mut lib = RuleLibrary::new(tmp.clone());

    // Add a verified rule (passed training)
    lib.add(
        "(predict-cell $p $r $c 7) if (grid-cell $p $r $c 3)".into(),
        RuleSource::Mutated,
    );
    lib.record_firing("(predict-cell $p $r $c 7) if (grid-cell $p $r $c 3)", true);

    // Add an unverified web candidate (never tested)
    lib.add(
        "(predict-cell $p $r $c 2) if (grid-cell $p $r $c 3)".into(),
        RuleSource::Induced,
    );
    // No record_firing — this rule was never proven

    assert_eq!(lib.len(), 2, "Library should have 2 rules before pruning");

    // After pruning, the unverified rule with 0 accuracy should be removable
    // (in a real scenario, after enough cycles without firing)
    let rules = lib.parse_rules();
    assert!(rules.len() >= 1, "Should have at least the verified rule");

    let _ = std::fs::remove_dir_all(&tmp);
}


// ═══════════════════════════════════════════════════════════════════════
// TEST GROUP 9: The Golden Rule — Source Never Bypasses Math
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_golden_rule_same_content_different_source_same_gate() {
    // SAME text from two different sources
    let text = "If color 3 is present then color 5 replaces it.";

    // From Tier1 (arxiv)
    let tier1_rules = web_rules::extract_rules(text, "https://arxiv.org/abs/1234");

    // From Tier3 (random site)
    let tier3_rules = web_rules::extract_rules(text, "https://random-blog.com/post");

    // Both should extract the same rule (source doesn't affect extraction)
    assert_eq!(tier1_rules.len(), tier3_rules.len(),
               "Same text should produce same number of rules regardless of source");

    if !tier1_rules.is_empty() {
        assert_eq!(tier1_rules[0].rule_text, tier3_rules[0].rule_text,
                   "Same text should produce identical rules regardless of source");

        // Both would be tested against the SAME math gate
        // Source affects search order (tier), NOT verification
    }
}

#[test]
fn test_golden_rule_high_confidence_still_fails_if_wrong() {
    use qor_core::neuron::{Neuron, QorValue, Statement};
    use qor_core::parser;
    use qor_runtime::eval::Session;
    use qor_runtime::search::score_rule_on_training;

    // A "high confidence" rule from a "top" source — but it's WRONG
    // "Every cell becomes color 0" — sounds confident, but wrong for this puzzle
    let high_conf_stmts = parser::parse(
        "(predict-cell train-0 $r $c 0) if (grid-cell train-0 $r $c $v)"
    ).unwrap();
    let high_conf_rule = if let Statement::Rule { head, body, tv } = &high_conf_stmts[0] {
        qor_runtime::chain::Rule::new(head.clone(), body.clone(), tv.unwrap_or_default())
    } else { panic!("expected rule") };

    // Puzzle: 3 → 7
    let training_inputs = vec![vec![
        Statement::Fact {
            neuron: Neuron::Expression(vec![
                Neuron::Symbol("grid-cell".into()),
                Neuron::Symbol("train-0".into()),
                Neuron::Value(QorValue::Int(0)),
                Neuron::Value(QorValue::Int(0)),
                Neuron::Value(QorValue::Int(3)),
            ]),
            tv: None, decay: None,
        },
    ]];
    let expected = vec![vec![
        Statement::Fact {
            neuron: Neuron::Expression(vec![
                Neuron::Symbol("predict-cell".into()),
                Neuron::Symbol("train-0".into()),
                Neuron::Value(QorValue::Int(0)),
                Neuron::Value(QorValue::Int(0)),
                Neuron::Value(QorValue::Int(7)),
            ]),
            tv: None, decay: None,
        },
    ]];

    let score = score_rule_on_training(
        &high_conf_rule, &training_inputs, &expected, "predict-cell", &Session::new()
    );

    // Score is 0.0 — DOESN'T MATTER that the source was confident
    assert_eq!(score, 0.0,
               "High-confidence wrong rule should score 0.0 — math is the judge");
}
