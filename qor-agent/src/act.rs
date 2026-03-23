//! Action Executor — Reads QOR decisions, executes on browser.
//!
//! QOR DNA rules produce `(browser-action <verb> <args...>)` facts.
//! This module reads those facts and maps them to browser operations.
//! Pure plumbing. Zero domain logic.

use qor_core::neuron::{Neuron, QorValue};
use qor_core::truth_value::TruthValue;
use qor_runtime::eval::Session;

use crate::browser::Browser;
#[allow(unused_imports)]
use crate::page;

/// Actions the agent can perform on the browser.
#[derive(Debug, Clone)]
pub enum AgentAction {
    Navigate(String),
    Click(String),
    Fill(String, String),
    Snapshot,
    PageText,
    Screenshot(bool),         // full_page
    EvalJs(String),
    NetworkList(Option<String>),
    Done,
}

/// An action with its confidence from QOR.
struct ScoredAction {
    action: AgentAction,
    confidence: f64,
}

/// Read `(browser-action ...)` facts from the QOR session.
/// Returns the highest-confidence action, or None if no action facts.
pub fn next_action(session: &Session) -> Option<AgentAction> {
    let mut candidates: Vec<ScoredAction> = Vec::new();

    for fact in session.all_facts() {
        if let Neuron::Expression(parts) = &fact.neuron {
            if parts.len() < 2 { continue; }
            if let Neuron::Symbol(pred) = &parts[0] {
                if pred != "browser-action" { continue; }

                if let Some(action) = parse_action(&parts[1..]) {
                    candidates.push(ScoredAction {
                        action,
                        confidence: fact.tv.strength,
                    });
                }
            }
        }
    }

    // Return highest confidence action
    candidates.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
    candidates.into_iter().next().map(|s| s.action)
}

/// Parse action from neuron parts after "browser-action".
fn parse_action(parts: &[Neuron]) -> Option<AgentAction> {
    let verb = match &parts[0] {
        Neuron::Symbol(s) => s.as_str(),
        _ => return None,
    };

    match verb {
        "navigate" => {
            let url = get_string(parts.get(1)?)?;
            Some(AgentAction::Navigate(url))
        }
        "click" => {
            let uid = get_symbol(parts.get(1)?)?;
            Some(AgentAction::Click(uid))
        }
        "fill" => {
            let uid = get_symbol(parts.get(1)?)?;
            let value = get_string(parts.get(2)?)?;
            Some(AgentAction::Fill(uid, value))
        }
        "snapshot" => Some(AgentAction::Snapshot),
        "page-text" => Some(AgentAction::PageText),
        "screenshot" => {
            let full = get_symbol(parts.get(1).unwrap_or(&Neuron::Symbol("viewport".into())))
                .map(|s| s == "full-page")
                .unwrap_or(false);
            Some(AgentAction::Screenshot(full))
        }
        "js" => {
            let script = get_string(parts.get(1)?)?;
            Some(AgentAction::EvalJs(script))
        }
        "network-list" => {
            let filter = parts.get(1).and_then(|n| get_string(n));
            Some(AgentAction::NetworkList(filter))
        }
        "done" => Some(AgentAction::Done),
        _ => None,
    }
}

/// Execute an action on the browser, return result as QOR facts.
pub async fn execute(
    action: &AgentAction,
    browser: &mut Browser,
) -> Result<Vec<qor_core::neuron::Statement>, String> {
    match action {
        AgentAction::Navigate(url) => {
            let (page_url, title) = browser.navigate(url).await?;
            Ok(page::nav_to_facts(&page_url, &title))
        }
        AgentAction::Click(uid) => {
            browser.click(uid).await?;
            Ok(vec![action_result_fact("clicked", uid)])
        }
        AgentAction::Fill(uid, value) => {
            browser.fill(uid, value).await?;
            Ok(vec![action_result_fact("filled", uid)])
        }
        AgentAction::Snapshot => {
            let elements = browser.snapshot().await?;
            Ok(page::snapshot_to_facts(&elements))
        }
        AgentAction::PageText => {
            let text = browser.page_text().await?;
            Ok(page::text_to_facts(&text))
        }
        AgentAction::Screenshot(full_page) => {
            let _b64 = browser.screenshot(*full_page).await?;
            Ok(vec![action_result_fact("screenshot", "taken")])
        }
        AgentAction::EvalJs(script) => {
            let result = browser.eval_js(script).await?;
            Ok(page::js_result_to_facts(script, &result))
        }
        AgentAction::NetworkList(filter) => {
            let entries = browser.network_list(filter.as_deref()).await?;
            Ok(page::network_to_facts(&entries))
        }
        AgentAction::Done => {
            Ok(vec![action_result_fact("done", "true")])
        }
    }
}

/// Create a simple (action-result <verb> <detail>) fact.
fn action_result_fact(verb: &str, detail: &str) -> qor_core::neuron::Statement {
    qor_core::neuron::Statement::Fact {
        neuron: Neuron::Expression(vec![
            Neuron::Symbol("action-result".into()),
            Neuron::Symbol(verb.into()),
            Neuron::Value(QorValue::Str(detail.into())),
        ]),
        tv: Some(TruthValue::new(0.99, 0.95)),
        decay: None,
    }
}

fn get_string(n: &Neuron) -> Option<String> {
    match n {
        Neuron::Value(QorValue::Str(s)) => Some(s.clone()),
        Neuron::Symbol(s) => Some(s.clone()),
        _ => None,
    }
}

fn get_symbol(n: &Neuron) -> Option<String> {
    match n {
        Neuron::Symbol(s) => Some(s.clone()),
        Neuron::Value(QorValue::Str(s)) => Some(s.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_navigate_action() {
        let parts = vec![
            Neuron::Symbol("navigate".into()),
            Neuron::Value(QorValue::Str("https://example.com".into())),
        ];
        let action = parse_action(&parts);
        assert!(matches!(action, Some(AgentAction::Navigate(url)) if url == "https://example.com"));
    }

    #[test]
    fn parse_click_action() {
        let parts = vec![
            Neuron::Symbol("click".into()),
            Neuron::Symbol("s5".into()),
        ];
        let action = parse_action(&parts);
        assert!(matches!(action, Some(AgentAction::Click(uid)) if uid == "s5"));
    }

    #[test]
    fn parse_fill_action() {
        let parts = vec![
            Neuron::Symbol("fill".into()),
            Neuron::Symbol("s12".into()),
            Neuron::Value(QorValue::Str("hello@test.com".into())),
        ];
        let action = parse_action(&parts);
        assert!(matches!(action, Some(AgentAction::Fill(uid, val)) if uid == "s12" && val == "hello@test.com"));
    }

    #[test]
    fn parse_snapshot_action() {
        let parts = vec![Neuron::Symbol("snapshot".into())];
        assert!(matches!(parse_action(&parts), Some(AgentAction::Snapshot)));
    }

    #[test]
    fn parse_done_action() {
        let parts = vec![Neuron::Symbol("done".into())];
        assert!(matches!(parse_action(&parts), Some(AgentAction::Done)));
    }

    #[test]
    fn parse_js_action() {
        let parts = vec![
            Neuron::Symbol("js".into()),
            Neuron::Value(QorValue::Str("document.title".into())),
        ];
        let action = parse_action(&parts);
        assert!(matches!(action, Some(AgentAction::EvalJs(s)) if s == "document.title"));
    }

    #[test]
    fn parse_network_list_with_filter() {
        let parts = vec![
            Neuron::Symbol("network-list".into()),
            Neuron::Value(QorValue::Str("api".into())),
        ];
        let action = parse_action(&parts);
        assert!(matches!(action, Some(AgentAction::NetworkList(Some(f))) if f == "api"));
    }

    #[test]
    fn parse_unknown_verb() {
        let parts = vec![Neuron::Symbol("unknown-action".into())];
        assert!(parse_action(&parts).is_none());
    }

    #[test]
    fn next_action_empty_session() {
        let session = Session::new();
        assert!(next_action(&session).is_none());
    }
}
