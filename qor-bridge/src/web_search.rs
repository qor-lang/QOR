// ── Web Search — Fetch + Extract Candidate Rules ─────────────────────
//
// High-level pipeline for web intelligence:
//   1. Build search URLs from topics + configured sources
//   2. Check cache → fetch via HTTP (ureq) if needed
//   3. Strip HTML → extract candidate rules via web_rules
//   4. Return candidate QOR rule texts for the search engine
//
// Feature-gated behind `--features web`.
// Uses ureq for simple synchronous HTTP GET (no async runtime needed).

#![cfg(feature = "web")]

use std::path::Path;

use crate::web_fetch::{PageContent, WebCache, DomainPolicy, WebConfig};
use crate::web_rules;

/// Result of a web search operation.
#[derive(Debug)]
pub struct WebSearchResult {
    /// Candidate QOR rule texts (ready to parse + feed as seeds).
    pub candidate_rules: Vec<String>,
    /// Number of pages successfully fetched.
    pub pages_fetched: usize,
    /// Number of candidate rules extracted.
    pub rules_extracted: usize,
}

/// Search the web for candidate QOR rules on given topics.
///
/// Pipeline:
///   topics → build_search_urls → cache check → HTTP fetch → strip HTML
///   → web_rules::extract_rules → candidate rule texts
///
/// Returns candidate rule texts that can be fed as seeds to
/// the refinement search engine. The math is the judge — these
/// candidates are unverified until they pass training pair scoring.
pub fn search_web(
    topics: &[String],
    sources: &[String],
    cache: &WebCache,
    policy: &DomainPolicy,
    cache_hours: u64,
) -> WebSearchResult {
    let mut all_rules = Vec::new();
    let mut pages_fetched = 0;

    // Build search URLs from topics × sources
    let urls: Vec<String> = topics.iter()
        .flat_map(|topic| web_rules::build_search_urls(topic, sources))
        .collect();

    // Filter by domain policy, sort by tier (academic first)
    let mut urls = crate::web_fetch::filter_urls(&urls, policy);
    crate::web_fetch::sort_urls_by_tier(&mut urls);

    for url in &urls {
        // Cache-first: check before hitting network
        let page = if cache.is_cached(url, cache_hours) {
            cache.load(url)
        } else {
            match fetch_page(url) {
                Some(page) => {
                    cache.save(&page);
                    Some(page)
                }
                None => None,
            }
        };

        if let Some(page) = page {
            pages_fetched += 1;

            // Extract candidate rules from page text
            let rules = web_rules::extract_rules(&page.text, &page.url);
            for rule in rules {
                all_rules.push(rule.rule_text);
            }
        }
    }

    let rules_extracted = all_rules.len();
    WebSearchResult {
        candidate_rules: all_rules,
        pages_fetched,
        rules_extracted,
    }
}

/// Convenience: search using a WebConfig (parsed from brain/web_config.qor).
pub fn search_from_config(
    config: &WebConfig,
    cache_dir: &Path,
    policy: &DomainPolicy,
) -> WebSearchResult {
    if !config.enabled || config.topics.is_empty() {
        return WebSearchResult {
            candidate_rules: Vec::new(),
            pages_fetched: 0,
            rules_extracted: 0,
        };
    }

    let cache = WebCache::new(cache_dir.to_path_buf());
    search_web(
        &config.topics,
        &config.sources,
        &cache,
        policy,
        config.cache_hours,
    )
}

/// Fetch a single page via HTTP GET using ureq.
/// Returns None on any error (network, timeout, non-200, etc).
fn fetch_page(url: &str) -> Option<PageContent> {
    let resp = ureq::get(url)
        .timeout(std::time::Duration::from_secs(15))
        .call()
        .ok()?;

    // Only accept text/html responses
    let content_type = resp.content_type().to_string();
    if !content_type.contains("text") && !content_type.contains("html") {
        return None;
    }

    let html = resp.into_string().ok()?;
    let text = strip_html(&html);
    let title = extract_title(&html);

    if text.is_empty() {
        return None;
    }

    Some(PageContent {
        url: url.to_string(),
        text,
        title,
    })
}

/// Simple HTML tag stripper — converts HTML to plain text.
/// Removes script/style blocks, collapses whitespace.
fn strip_html(html: &str) -> String {
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
            let ahead: String = lower_chars[i..i + 7].iter().collect();
            if ahead == "<script" {
                in_script = true;
            }
            if ahead == "<style " || (i + 6 < lower_chars.len()
                && lower_chars[i..i + 6].iter().collect::<String>() == "<style")
            {
                in_style = true;
            }
        }

        if chars[i] == '<' {
            if in_script && i + 9 < lower_chars.len() {
                let ahead: String = lower_chars[i..i + 9].iter().collect();
                if ahead == "</script>" {
                    in_script = false;
                    i += 9;
                    continue;
                }
            }
            if in_style && i + 8 < lower_chars.len() {
                let ahead: String = lower_chars[i..i + 8].iter().collect();
                if ahead == "</style>" {
                    in_style = false;
                    i += 8;
                    continue;
                }
            }
            in_tag = true;
        } else if chars[i] == '>' {
            in_tag = false;
            result.push(' ');
        } else if !in_tag && !in_script && !in_style {
            result.push(chars[i]);
        }
        i += 1;
    }

    // Collapse whitespace
    result.split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ")
}

/// Extract <title> from HTML.
fn extract_title(html: &str) -> String {
    let lower = html.to_lowercase();
    if let Some(start) = lower.find("<title>") {
        let after = start + 7;
        if let Some(end) = lower[after..].find("</title>") {
            return html[after..after + end].trim().to_string();
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_html_basic() {
        let html = "<html><body><p>Hello world</p></body></html>";
        let text = strip_html(html);
        assert!(text.contains("Hello world"));
        assert!(!text.contains("<p>"));
    }

    #[test]
    fn test_strip_html_script_removal() {
        let html = "<p>Before</p><script>var x = 1;</script><p>After</p>";
        let text = strip_html(html);
        assert!(text.contains("Before"));
        assert!(text.contains("After"));
        assert!(!text.contains("var x"));
    }

    #[test]
    fn test_extract_title() {
        let html = "<html><head><title>Test Page</title></head></html>";
        assert_eq!(extract_title(html), "Test Page");
    }

    #[test]
    fn test_extract_title_missing() {
        let html = "<html><body>No title here</body></html>";
        assert_eq!(extract_title(html), "");
    }

    #[test]
    fn test_search_web_empty_topics() {
        let cache = WebCache::new(std::env::temp_dir().join("qor_web_search_test"));
        let policy = DomainPolicy::default();
        let result = search_web(&[], &[], &cache, &policy, 24);
        assert_eq!(result.pages_fetched, 0);
        assert_eq!(result.rules_extracted, 0);
        assert!(result.candidate_rules.is_empty());
    }

    #[test]
    fn test_search_from_config_disabled() {
        let config = WebConfig::default(); // enabled = false
        let policy = DomainPolicy::default();
        let result = search_from_config(
            &config,
            &std::env::temp_dir().join("qor_web_search_test2"),
            &policy,
        );
        assert!(result.candidate_rules.is_empty());
    }

    #[test]
    fn test_fetch_page_invalid_url() {
        // Invalid URL should return None, not panic
        let result = fetch_page("not-a-url");
        assert!(result.is_none());
    }
}
