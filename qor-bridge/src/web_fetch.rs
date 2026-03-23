// ── Web Fetch — Crawl + Extract ─────────────────────────────────────
//
// Web intelligence layer for QOR.
// - Crawler module (feature-gated: web) uses spider-rs for fast crawling
// - Extraction module (always available) converts text → QOR facts
// - Cache module stores crawled content for offline use
//
// The crawler is pure plumbing. Domain reasoning stays in QOR rules.

use std::path::PathBuf;

use qor_core::neuron::{Neuron, QorValue, Statement};
use qor_core::truth_value::TruthValue;

use crate::text_hint;
use crate::web_rules;

// ── Page Content (shared type) ──────────────────────────────────────

/// Content extracted from a web page.
#[derive(Debug, Clone)]
pub struct PageContent {
    pub url: String,
    pub text: String,
    pub title: String,
}

/// Configuration for web crawling.
#[derive(Debug, Clone)]
pub struct CrawlConfig {
    pub max_pages: usize,
    pub max_depth: usize,
    pub timeout_secs: u64,
}

impl Default for CrawlConfig {
    fn default() -> Self {
        CrawlConfig {
            max_pages: 10,
            max_depth: 2,
            timeout_secs: 30,
        }
    }
}

/// Result of a crawl operation.
#[derive(Debug)]
pub struct CrawlResult {
    pub pages: Vec<PageContent>,
    pub elapsed_ms: u64,
}

// ── Spider-rs Crawler (feature-gated) ───────────────────────────────

#[cfg(feature = "web")]
pub mod crawler {
    use super::*;
    use spider::website::Website;
    use std::time::Instant;

    /// Safe crawl — checks domain policy BEFORE crawling.
    /// Returns None if the URL is blocked by policy.
    pub async fn safe_crawl(
        url: &str,
        config: &CrawlConfig,
        policy: &DomainPolicy,
    ) -> Option<CrawlResult> {
        if !policy.can_crawl(url) {
            return None;
        }
        Some(crawl_url(url, config).await)
    }

    /// Crawl a URL using spider-rs.
    /// Returns clean text content from each page.
    pub async fn crawl_url(url: &str, config: &CrawlConfig) -> CrawlResult {
        let start = Instant::now();
        let mut pages = Vec::new();

        let mut website = Website::new(url);
        website.with_limit(config.max_pages as u32);
        website.with_depth(config.max_depth);

        website.crawl().await;

        // Extract text from crawled pages
        let empty = Vec::new();
        let site_pages = website.get_pages().unwrap_or(&empty);
        for page in site_pages.iter() {
            let page_url = page.get_url().to_string();
            let html = page.get_html();

            // Simple HTML → text: strip tags
            let text = strip_html_tags(&html);
            let title = extract_title(&html);

            if !text.is_empty() {
                pages.push(PageContent {
                    url: page_url,
                    text,
                    title,
                });
            }
        }

        CrawlResult {
            pages,
            elapsed_ms: start.elapsed().as_millis() as u64,
        }
    }

    /// Crawl multiple URLs.
    pub async fn crawl_urls(urls: &[String], config: &CrawlConfig) -> Vec<CrawlResult> {
        let mut results = Vec::new();
        for url in urls {
            results.push(crawl_url(url, config).await);
        }
        results
    }

    /// Simple HTML tag stripper (no external dependency).
    fn strip_html_tags(html: &str) -> String {
        let mut result = String::new();
        let mut in_tag = false;
        let mut in_script = false;
        let mut in_style = false;

        let lower = html.to_lowercase();
        let chars: Vec<char> = html.chars().collect();
        let lower_chars: Vec<char> = lower.chars().collect();

        let mut i = 0;
        while i < chars.len() {
            if !in_tag && i + 7 < lower_chars.len() {
                let ahead: String = lower_chars[i..i+7].iter().collect();
                if ahead == "<script" {
                    in_script = true;
                }
                if ahead == "<style " || (i + 6 < lower_chars.len() && lower_chars[i..i+6].iter().collect::<String>() == "<style") {
                    in_style = true;
                }
            }

            if chars[i] == '<' {
                // Check for end of script/style
                if in_script && i + 9 < lower_chars.len() {
                    let ahead: String = lower_chars[i..i+9].iter().collect();
                    if ahead == "</script>" {
                        in_script = false;
                        i += 9;
                        continue;
                    }
                }
                if in_style && i + 8 < lower_chars.len() {
                    let ahead: String = lower_chars[i..i+8].iter().collect();
                    if ahead == "</style>" {
                        in_style = false;
                        i += 8;
                        continue;
                    }
                }
                in_tag = true;
            } else if chars[i] == '>' {
                in_tag = false;
                // Add space after block elements
                result.push(' ');
            } else if !in_tag && !in_script && !in_style {
                result.push(chars[i]);
            }
            i += 1;
        }

        // Clean up: collapse whitespace, trim
        let cleaned: String = result.split_whitespace()
            .collect::<Vec<&str>>()
            .join(" ");
        cleaned
    }

    /// Extract <title> from HTML.
    fn extract_title(html: &str) -> String {
        let lower = html.to_lowercase();
        if let Some(start) = lower.find("<title>") {
            let after = start + 7;
            if let Some(end) = lower[after..].find("</title>") {
                return html[after..after+end].trim().to_string();
            }
        }
        String::new()
    }
}

// ── Fact Extraction (always available) ──────────────────────────────

/// Extract CANDIDATE facts from crawled web pages.
///
/// CRITICAL: Nothing from the web becomes a real fact directly.
/// Everything is tagged as a CANDIDATE. Only the heartbeat's
/// verification loop (test against training pairs) promotes
/// candidates to real facts. The math is the judge.
///
/// Extraction strategies:
/// 1. Provenance metadata (web-source, web-domain) — kept as-is (metadata, not claims)
/// 2. text_hint.rs semantic hints — tagged as web-hint-candidate
/// 3. web_rules.rs rule extraction — tagged as web-rule-candidate
/// 4. web_rules.rs fact extraction — tagged as web-fact-candidate
pub fn extract_facts_from_pages(pages: &[PageContent]) -> Vec<Statement> {
    let mut all_facts = Vec::new();

    for page in pages {
        // 1. Source provenance — metadata about WHERE we crawled (not a claim)
        all_facts.push(Statement::Fact {
            neuron: Neuron::Expression(vec![
                Neuron::Symbol("web-source".into()),
                Neuron::str_val(&page.url),
                Neuron::str_val(&page.title),
            ]),
            tv: Some(TruthValue::new(0.99, 0.99)),
            decay: None,
        });

        // 2. Domain detection — metadata about source category
        let domain = detect_domain(&page.url);
        if !domain.is_empty() {
            all_facts.push(Statement::Fact {
                neuron: Neuron::Expression(vec![
                    Neuron::Symbol("web-domain".into()),
                    Neuron::str_val(&page.url),
                    Neuron::Symbol(domain.clone()),
                ]),
                tv: Some(TruthValue::new(0.95, 0.95)),
                decay: None,
            });
        }

        // 3. Text hints — tagged as CANDIDATES, not real hints yet
        let hints = text_hint::parse_text_hints(&page.text);
        for hint in hints {
            if let Statement::Fact { neuron, tv, decay } = hint {
                // Wrap: (text-hint name conf) → (web-hint-candidate name conf)
                if let Neuron::Expression(ref parts) = neuron {
                    let mut candidate_parts = vec![Neuron::Symbol("web-hint-candidate".into())];
                    candidate_parts.extend(parts.iter().skip(1).cloned());
                    all_facts.push(Statement::Fact {
                        neuron: Neuron::Expression(candidate_parts),
                        tv,
                        decay,
                    });
                }
            }
        }

        // 4. Rule extraction — stored as CANDIDATES, never promoted without verification
        let rules = web_rules::extract_rules(&page.text, &page.url);
        for rule in &rules {
            all_facts.push(Statement::Fact {
                neuron: Neuron::Expression(vec![
                    Neuron::Symbol("web-rule-candidate".into()),
                    Neuron::str_val(&rule.rule_text),
                    Neuron::Value(QorValue::Float(rule.confidence)),
                ]),
                tv: Some(TruthValue::new(rule.confidence * 0.5, 0.30)),
                decay: None,
            });
        }

        // 5. Fact extraction — ALL tagged as CANDIDATES
        //    Nothing becomes a fact until verified against training pairs.
        //    The math is the judge, not the source.
        let facts = web_rules::extract_facts(&page.text, &page.url);
        for fact in facts {
            if let Statement::Fact { neuron, .. } = &fact.statement {
                // Wrap original fact inside web-fact-candidate envelope
                // (meaning X "Y") → (web-fact-candidate "meaning" X "Y" source_url)
                all_facts.push(Statement::Fact {
                    neuron: Neuron::Expression(vec![
                        Neuron::Symbol("web-fact-candidate".into()),
                        Neuron::str_val(&format!("{:?}", neuron)),
                        Neuron::str_val(&page.url),
                    ]),
                    tv: Some(TruthValue::new(0.30, 0.20)), // Very low — unverified
                    decay: None,
                });
            }
        }
    }

    all_facts
}

/// Detect domain category from URL.
fn detect_domain(url: &str) -> String {
    let lower = url.to_lowercase();
    if lower.contains("arxiv.org") { return "academic".into(); }
    if lower.contains("wikipedia.org") { return "encyclopedia".into(); }
    if lower.contains("stackoverflow.com") { return "programming".into(); }
    if lower.contains("github.com") { return "code".into(); }
    if lower.contains("kaggle.com") { return "data-science".into(); }
    if lower.contains("mathworld") || lower.contains("math.") { return "mathematics".into(); }
    if lower.contains("pubmed") || lower.contains("nih.gov") { return "medical".into(); }
    if lower.contains("investopedia") || lower.contains("finance") { return "finance".into(); }
    String::new()
}

// ── Offline Cache ───────────────────────────────────────────────────

/// Cache for storing crawled web content offline.
pub struct WebCache {
    cache_dir: PathBuf,
}

impl WebCache {
    pub fn new(cache_dir: PathBuf) -> Self {
        let _ = std::fs::create_dir_all(&cache_dir);
        WebCache { cache_dir }
    }

    /// Check if a URL is cached (and not expired).
    pub fn is_cached(&self, url: &str, max_age_hours: u64) -> bool {
        let path = self.url_to_path(url);
        if !path.exists() {
            return false;
        }
        if max_age_hours == 0 {
            return true; // No expiry
        }
        // Check file age
        if let Ok(metadata) = std::fs::metadata(&path) {
            if let Ok(modified) = metadata.modified() {
                if let Ok(elapsed) = modified.elapsed() {
                    return elapsed.as_secs() < max_age_hours * 3600;
                }
            }
        }
        true // Default to cached if we can't check age
    }

    /// Load a cached page.
    pub fn load(&self, url: &str) -> Option<PageContent> {
        let path = self.url_to_path(url);
        let text = std::fs::read_to_string(&path).ok()?;

        // First line is the title, rest is content
        let mut lines = text.lines();
        let title = lines.next().unwrap_or("").to_string();
        let content: String = lines.collect::<Vec<&str>>().join("\n");

        Some(PageContent {
            url: url.to_string(),
            text: content,
            title,
        })
    }

    /// Save a page to cache.
    pub fn save(&self, page: &PageContent) {
        let path = self.url_to_path(&page.url);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        // First line = title, rest = content
        let content = format!("{}\n{}", page.title, page.text);
        let _ = std::fs::write(&path, content);
    }

    /// Convert URL to a cache file path.
    fn url_to_path(&self, url: &str) -> PathBuf {
        // Simple hash: use URL bytes to create a filename
        let hash = simple_hash(url);
        let domain = url.split("//")
            .nth(1)
            .and_then(|s| s.split('/').next())
            .unwrap_or("unknown");
        self.cache_dir.join(domain).join(format!("{:016x}.txt", hash))
    }
}

/// Simple non-crypto hash for cache filenames.
fn simple_hash(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for byte in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
    }
    hash
}

// ── Domain Policy — Whitelist + Blacklist + Tiers ───────────────────
//
// ARCHITECTURE: Two separate concerns, never confuse them:
//   Domain policy → controls WHERE to search (saves heartbeat cycles)
//   Verification gate → controls WHAT to keep (math decides, not source)
//
// Tier affects search ORDER. Verification gate is IDENTICAL for all tiers.
// A rule from arxiv that fails training pairs = garbage.
// A rule from kaggle that passes training pairs = fact.

/// Domain quality tier — affects search ORDER, NOT verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainTier {
    /// Academic, peer-reviewed — searched first
    Tier1,
    /// Professional reference — standard trust
    Tier2,
    /// Community, competition — searched last
    Tier3,
}

/// Domain-level access control for web crawling.
#[derive(Debug, Clone)]
pub struct DomainPolicy {
    pub whitelist: Vec<String>,
    pub blacklist: Vec<String>,
}

impl DomainPolicy {
    /// Check if a URL is allowed to be crawled.
    /// Blacklist wins over whitelist. Must be in whitelist.
    pub fn can_crawl(&self, url: &str) -> bool {
        let domain = extract_domain(url);

        // Blacklist wins — check first
        if self.blacklist.iter().any(|b| domain.contains(b.as_str())) {
            return false;
        }

        // Must be in whitelist (empty whitelist = allow all)
        if self.whitelist.is_empty() {
            return true;
        }
        self.whitelist.iter().any(|w| domain.contains(w.as_str()))
    }
}

impl Default for DomainPolicy {
    fn default() -> Self {
        DomainPolicy {
            whitelist: Vec::new(),
            blacklist: vec![
                "pinterest.com".into(),
                "instagram.com".into(),
                "twitter.com".into(),
                "tiktok.com".into(),
                "facebook.com".into(),
                "youtube.com".into(),
            ],
        }
    }
}

/// Get the tier for a domain — controls search order, NOT verification.
pub fn domain_tier(url: &str) -> DomainTier {
    let domain = extract_domain(url);
    if domain.contains("arxiv.org")
        || domain.contains("openreview.net")
        || domain.contains("arcprize.org")
        || domain.contains("semanticscholar.org")
    {
        return DomainTier::Tier1;
    }
    if domain.contains("wikipedia.org")
        || domain.contains("mathworld.wolfram.com")
        || domain.contains("investopedia.com")
        || domain.contains("corporatefinanceinstitute.com")
        || domain.contains("paperswithcode.com")
    {
        return DomainTier::Tier2;
    }
    DomainTier::Tier3
}

/// Extract domain from a URL (e.g. "https://arxiv.org/abs/123" → "arxiv.org").
fn extract_domain(url: &str) -> String {
    url.split("//")
        .nth(1)
        .and_then(|s| s.split('/').next())
        .unwrap_or("")
        .to_lowercase()
}

/// Parse domain policy from a config file.
///
/// Expected format (simple, no TOML dependency):
/// ```text
/// [arc]
/// whitelist = arxiv.org, arcprize.org, kaggle.com, wikipedia.org
/// blacklist = pinterest.com, twitter.com, tiktok.com
///
/// [finance]
/// whitelist = ssrn.com, investopedia.com, arxiv.org
/// blacklist = reddit.com, stocktwits.com
/// ```
pub fn parse_domain_policy(text: &str, section: &str) -> DomainPolicy {
    let mut policy = DomainPolicy::default();
    let mut in_section = false;

    for line in text.lines() {
        let line = line.trim();

        // Section header
        if line.starts_with('[') && line.ends_with(']') {
            let name = &line[1..line.len()-1];
            in_section = name == section;
            continue;
        }

        if !in_section || line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse "key = value1, value2, value3"
        if let Some(eq_pos) = line.find('=') {
            let key = line[..eq_pos].trim();
            let vals = line[eq_pos+1..].trim();

            let items: Vec<String> = vals.split(',')
                .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string())
                .filter(|s| !s.is_empty())
                .collect();

            match key {
                "whitelist" => policy.whitelist = items,
                "blacklist" => {
                    // Merge with defaults
                    for item in items {
                        if !policy.blacklist.contains(&item) {
                            policy.blacklist.push(item);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    policy
}

/// Sort URLs by domain tier — Tier1 first, then Tier2, then Tier3.
pub fn sort_urls_by_tier(urls: &mut Vec<String>) {
    urls.sort_by_key(|url| match domain_tier(url) {
        DomainTier::Tier1 => 0,
        DomainTier::Tier2 => 1,
        DomainTier::Tier3 => 2,
    });
}

/// Filter URLs through domain policy, returning only allowed ones.
pub fn filter_urls(urls: &[String], policy: &DomainPolicy) -> Vec<String> {
    urls.iter()
        .filter(|url| policy.can_crawl(url))
        .cloned()
        .collect()
}

// ── Web Config from QOR facts ───────────────────────────────────────

/// Configuration parsed from QOR facts in web_config.qor.
#[derive(Debug, Clone)]
pub struct WebConfig {
    pub enabled: bool,
    pub sources: Vec<String>,
    pub topics: Vec<String>,
    pub crawl_interval: usize,
    pub max_pages: usize,
    pub cache_hours: u64,
}

impl Default for WebConfig {
    fn default() -> Self {
        WebConfig {
            enabled: false,
            sources: vec![
                "https://en.wikipedia.org/".into(),
            ],
            topics: Vec::new(),
            crawl_interval: 10,
            max_pages: 10,
            cache_hours: 24,
        }
    }
}

/// Parse web configuration from QOR statements.
pub fn parse_web_config(facts: &[Statement]) -> WebConfig {
    let mut config = WebConfig::default();

    for stmt in facts {
        if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = stmt {
            let pred = match parts.first() {
                Some(Neuron::Symbol(s)) => s.as_str(),
                _ => continue,
            };

            match pred {
                "web-source" => {
                    if let Some(Neuron::Value(QorValue::Str(url))) = parts.get(1) {
                        config.sources.push(url.clone());
                        config.enabled = true;
                    }
                }
                "web-topic" => {
                    if let Some(Neuron::Value(QorValue::Str(topic))) = parts.get(1) {
                        config.topics.push(topic.clone());
                    }
                }
                "web-crawl-interval" => {
                    if let Some(Neuron::Value(QorValue::Int(n))) = parts.get(1) {
                        config.crawl_interval = *n as usize;
                    }
                }
                "web-max-pages" => {
                    if let Some(Neuron::Value(QorValue::Int(n))) = parts.get(1) {
                        config.max_pages = *n as usize;
                    }
                }
                "web-cache-hours" => {
                    if let Some(Neuron::Value(QorValue::Int(n))) = parts.get(1) {
                        config.cache_hours = *n as u64;
                    }
                }
                _ => {}
            }
        }
    }

    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_facts_from_pages() {
        let pages = vec![PageContent {
            url: "https://en.wikipedia.org/wiki/Water".into(),
            text: "Water is a liquid. It has hydrogen atoms. The boiling point is 100.".into(),
            title: "Water - Wikipedia".into(),
        }];
        let facts = extract_facts_from_pages(&pages);
        // Should have: web-source + web-domain + extracted facts
        assert!(facts.len() >= 3, "Should extract multiple facts, got {}", facts.len());

        // Check provenance
        let has_source = facts.iter().any(|f| {
            format!("{:?}", f).contains("web-source")
        });
        assert!(has_source, "Should have web-source provenance fact");

        let has_domain = facts.iter().any(|f| {
            format!("{:?}", f).contains("web-domain")
        });
        assert!(has_domain, "Should detect wikipedia domain");
    }

    #[test]
    fn test_detect_domain() {
        assert_eq!(detect_domain("https://arxiv.org/abs/1234"), "academic");
        assert_eq!(detect_domain("https://en.wikipedia.org/wiki/Test"), "encyclopedia");
        assert_eq!(detect_domain("https://stackoverflow.com/q/123"), "programming");
        assert_eq!(detect_domain("https://github.com/user/repo"), "code");
        assert_eq!(detect_domain("https://random-site.com"), "");
    }

    #[test]
    fn test_parse_web_config() {
        let facts = vec![
            Statement::Fact {
                neuron: Neuron::Expression(vec![
                    Neuron::Symbol("web-source".into()),
                    Neuron::str_val("https://example.com/"),
                ]),
                tv: None, decay: None,
            },
            Statement::Fact {
                neuron: Neuron::Expression(vec![
                    Neuron::Symbol("web-topic".into()),
                    Neuron::str_val("grid transformation"),
                ]),
                tv: None, decay: None,
            },
            Statement::Fact {
                neuron: Neuron::Expression(vec![
                    Neuron::Symbol("web-crawl-interval".into()),
                    Neuron::Value(QorValue::Int(5)),
                ]),
                tv: None, decay: None,
            },
            Statement::Fact {
                neuron: Neuron::Expression(vec![
                    Neuron::Symbol("web-max-pages".into()),
                    Neuron::Value(QorValue::Int(20)),
                ]),
                tv: None, decay: None,
            },
        ];

        let config = parse_web_config(&facts);
        assert!(config.enabled);
        assert!(config.sources.iter().any(|s| s.contains("example.com")));
        assert_eq!(config.topics, vec!["grid transformation"]);
        assert_eq!(config.crawl_interval, 5);
        assert_eq!(config.max_pages, 20);
    }

    #[test]
    fn test_default_web_config() {
        let config = WebConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.crawl_interval, 10);
        assert_eq!(config.max_pages, 10);
    }

    #[test]
    fn test_web_cache_path() {
        let cache = WebCache::new(PathBuf::from("/tmp/qor_web_cache"));
        let path = cache.url_to_path("https://example.com/page");
        assert!(path.to_str().unwrap().contains("example.com"));
    }

    #[test]
    fn test_simple_hash() {
        let h1 = simple_hash("hello");
        let h2 = simple_hash("world");
        assert_ne!(h1, h2);
        // Same input = same hash
        assert_eq!(simple_hash("test"), simple_hash("test"));
    }

    #[test]
    fn test_web_cache_save_load() {
        let tmp = std::env::temp_dir().join("qor_web_cache_test");
        let cache = WebCache::new(tmp.clone());

        let page = PageContent {
            url: "https://example.com/test".into(),
            text: "This is test content.".into(),
            title: "Test Page".into(),
        };

        cache.save(&page);
        assert!(cache.is_cached("https://example.com/test", 24));

        let loaded = cache.load("https://example.com/test").unwrap();
        assert_eq!(loaded.title, "Test Page");
        assert_eq!(loaded.text, "This is test content.");

        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_rule_candidates_in_extraction() {
        let pages = vec![PageContent {
            url: "https://example.com".into(),
            text: "If temperature rises then pressure increases. Water boils at 100 degrees.".into(),
            title: "Science".into(),
        }];
        let facts = extract_facts_from_pages(&pages);

        let has_rule_candidate = facts.iter().any(|f| {
            format!("{:?}", f).contains("web-rule-candidate")
        });
        assert!(has_rule_candidate, "Should extract rule candidates from conditional text");
    }

    #[test]
    fn test_domain_policy_whitelist() {
        let policy = DomainPolicy {
            whitelist: vec!["arxiv.org".into(), "wikipedia.org".into()],
            blacklist: vec!["pinterest.com".into()],
        };
        assert!(policy.can_crawl("https://arxiv.org/abs/1234"));
        assert!(policy.can_crawl("https://en.wikipedia.org/wiki/Test"));
        assert!(!policy.can_crawl("https://pinterest.com/pin/123"));
        assert!(!policy.can_crawl("https://random-site.com"));
    }

    #[test]
    fn test_domain_policy_blacklist_wins() {
        // Even if in whitelist, blacklist wins
        let policy = DomainPolicy {
            whitelist: vec!["example.com".into()],
            blacklist: vec!["example.com".into()],
        };
        assert!(!policy.can_crawl("https://example.com/page"));
    }

    #[test]
    fn test_domain_policy_empty_whitelist_allows_all() {
        let policy = DomainPolicy {
            whitelist: vec![],
            blacklist: vec!["bad.com".into()],
        };
        assert!(policy.can_crawl("https://anything.com"));
        assert!(!policy.can_crawl("https://bad.com"));
    }

    #[test]
    fn test_domain_tiers() {
        assert_eq!(domain_tier("https://arxiv.org/abs/123"), DomainTier::Tier1);
        assert_eq!(domain_tier("https://openreview.net"), DomainTier::Tier1);
        assert_eq!(domain_tier("https://wikipedia.org/wiki/X"), DomainTier::Tier2);
        assert_eq!(domain_tier("https://kaggle.com/comp"), DomainTier::Tier3);
        assert_eq!(domain_tier("https://random.com"), DomainTier::Tier3);
    }

    #[test]
    fn test_sort_urls_by_tier() {
        let mut urls = vec![
            "https://kaggle.com/comp".into(),
            "https://arxiv.org/abs/1".into(),
            "https://wikipedia.org/wiki/X".into(),
        ];
        sort_urls_by_tier(&mut urls);
        assert!(urls[0].contains("arxiv"));      // Tier1 first
        assert!(urls[1].contains("wikipedia"));   // Tier2 second
        assert!(urls[2].contains("kaggle"));      // Tier3 last
    }

    #[test]
    fn test_parse_domain_policy() {
        let text = r#"
[arc]
whitelist = arxiv.org, kaggle.com
blacklist = reddit.com

[finance]
whitelist = ssrn.com
"#;
        let arc = parse_domain_policy(text, "arc");
        assert_eq!(arc.whitelist, vec!["arxiv.org", "kaggle.com"]);
        assert!(arc.blacklist.contains(&"reddit.com".to_string()));
        assert!(arc.can_crawl("https://arxiv.org/abs/1"));
        assert!(!arc.can_crawl("https://reddit.com/r/arc"));
        assert!(!arc.can_crawl("https://random.com"));

        let fin = parse_domain_policy(text, "finance");
        assert_eq!(fin.whitelist, vec!["ssrn.com"]);
    }

    #[test]
    fn test_filter_urls() {
        let policy = DomainPolicy {
            whitelist: vec!["arxiv.org".into(), "wikipedia.org".into()],
            blacklist: vec![],
        };
        let urls = vec![
            "https://arxiv.org/abs/1".into(),
            "https://random.com".into(),
            "https://wikipedia.org/wiki/X".into(),
        ];
        let filtered = filter_urls(&urls, &policy);
        assert_eq!(filtered.len(), 2);
        assert!(filtered[0].contains("arxiv"));
        assert!(filtered[1].contains("wikipedia"));
    }

    #[test]
    fn test_web_facts_are_candidates_not_real() {
        // CRITICAL: web-extracted info must NEVER become direct facts.
        // Everything is a candidate until verified by the math (training pairs).
        let pages = vec![PageContent {
            url: "https://example.com".into(),
            text: "A penguin is a bird. Water has hydrogen.".into(),
            title: "Facts".into(),
        }];
        let facts = extract_facts_from_pages(&pages);

        // No direct (is-a ...) or (has ...) facts — all wrapped as candidates
        for f in &facts {
            let s = format!("{:?}", f);
            // Provenance metadata (web-source, web-domain) are OK
            if s.contains("web-source") || s.contains("web-domain") {
                continue;
            }
            // Everything else MUST be a candidate
            assert!(
                s.contains("candidate"),
                "Non-provenance web fact should be a candidate, got: {}",
                s
            );
        }
    }
}
