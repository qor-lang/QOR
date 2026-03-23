//! Browser — CDP wrapper using agentchrome's public API.
//!
//! Pure plumbing. Zero domain logic.
//! Launches Chrome, connects via CDP, provides high-level async methods.
//! Auto-downloads Chromium on first use if not already present.

use std::collections::HashMap;
use std::io::Read as _;
use std::path::PathBuf;
use std::time::Duration;

use agentchrome::cdp::{CdpClient, CdpConfig};
use agentchrome::chrome::{self, Channel, ChromeProcess, LaunchConfig};
use agentchrome::connection::ManagedSession;
use serde_json::Value;

/// Interactive roles that get assigned UIDs for click/fill targeting.
const INTERACTIVE_ROLES: &[&str] = &[
    "link", "button", "textbox", "checkbox", "radio", "combobox",
    "menuitem", "tab", "switch", "slider", "spinbutton", "searchbox",
    "option", "treeitem",
];

/// A browser session connected to Chrome via CDP.
pub struct Browser {
    _client: CdpClient,
    session: ManagedSession,
    _chrome: Option<ChromeProcess>,
    /// UID → backend DOM node ID (from last snapshot).
    pub uid_map: HashMap<String, i64>,
}

/// Result of probing for Chrome availability.
pub struct ProbeResult {
    pub chrome_found: bool,
    pub executable: Option<std::path::PathBuf>,
}

impl Browser {
    /// Check if Chrome/Chromium is available (local QOR copy or system).
    pub fn probe() -> ProbeResult {
        match find_chromium() {
            Some(path) => ProbeResult { chrome_found: true, executable: Some(path) },
            None => ProbeResult { chrome_found: false, executable: None },
        }
    }

    /// Launch headless Chrome and connect via CDP.
    /// Tries: 1) QOR's bundled Chromium (~/.qor/chromium/)
    ///        2) System Chrome
    ///        3) Auto-download Chromium if neither found
    pub async fn launch() -> Result<Self, String> {
        let executable = match find_chromium() {
            Some(path) => path,
            None => {
                // Auto-download on first use
                eprintln!("  [BROWSER] No Chromium found — downloading...");
                download_chromium().map_err(|e| format!("Chromium download failed: {e}"))?
            }
        };
        let port = chrome::find_available_port()
            .map_err(|e| format!("No free port: {e}"))?;

        let config = LaunchConfig {
            executable,
            port,
            headless: true,
            extra_args: vec![],
            user_data_dir: None,
        };

        let chrome_proc = chrome::launch_chrome(config, Duration::from_secs(15))
            .await
            .map_err(|e| format!("Failed to launch Chrome: {e}"))?;

        // Connect via CDP
        let targets = chrome::query_targets("localhost", port)
            .await
            .map_err(|e| format!("Failed to query targets: {e}"))?;

        let target = targets.iter()
            .find(|t| t.target_type == "page")
            .ok_or("No page target found")?;

        let cdp_config = CdpConfig::default();
        let ws_url = format!("ws://localhost:{}/devtools/browser", port);
        let client = CdpClient::connect(&ws_url, cdp_config)
            .await
            .map_err(|e| format!("CDP connect failed: {e}"))?;

        let cdp_session = client.create_session(&target.id)
            .await
            .map_err(|e| format!("Session creation failed: {e}"))?;

        let session = ManagedSession::new(cdp_session);

        Ok(Browser {
            _client: client,
            session,
            _chrome: Some(chrome_proc),
            uid_map: HashMap::new(),
        })
    }

    /// Navigate to a URL. Returns (url, title).
    pub async fn navigate(&mut self, url: &str) -> Result<(String, String), String> {
        self.session.ensure_domain("Page").await.map_err(|e| e.to_string())?;
        self.session.ensure_domain("Runtime").await.map_err(|e| e.to_string())?;

        // Subscribe to load event before navigating
        let mut load_rx = self.session.subscribe("Page.loadEventFired")
            .await.map_err(|e| e.to_string())?;

        let params = serde_json::json!({ "url": url });
        let result = self.session.send_command("Page.navigate", Some(params))
            .await.map_err(|e| e.to_string())?;

        // Check for navigation error
        if let Some(err) = result["errorText"].as_str() {
            if !err.is_empty() {
                return Err(format!("Navigation failed: {err}"));
            }
        }

        // Wait for load (up to 30s)
        tokio::select! {
            _ = load_rx.recv() => {},
            _ = tokio::time::sleep(Duration::from_secs(30)) => {
                return Err("Navigation timeout".into());
            }
        }

        self.get_page_info().await
    }

    /// Get accessibility snapshot. Returns flat list of elements.
    /// Updates uid_map for subsequent click/fill operations.
    pub async fn snapshot(&mut self) -> Result<Vec<PageElement>, String> {
        self.session.ensure_domain("Accessibility").await.map_err(|e| e.to_string())?;

        let response = self.session.send_command("Accessibility.getFullAXTree", None)
            .await.map_err(|e| e.to_string())?;

        let nodes = response["nodes"].as_array()
            .ok_or("Missing nodes array in snapshot")?;

        let mut elements = Vec::new();
        let mut uid_counter = 0u32;
        self.uid_map.clear();

        for node in nodes {
            let ignored = node["ignored"].as_bool().unwrap_or(false);
            if ignored { continue; }

            let role = node["role"]["value"].as_str().unwrap_or_default().to_string();
            let name = node["name"]["value"].as_str().unwrap_or_default().to_string();
            let backend_id = node["backendDOMNodeId"].as_i64();

            // Extract properties (url, checked, value, etc.)
            let mut props = HashMap::new();
            if let Some(prop_arr) = node["properties"].as_array() {
                for p in prop_arr {
                    if let Some(k) = p["name"].as_str() {
                        let v = &p["value"]["value"];
                        if !v.is_null() {
                            props.insert(k.to_string(), v.clone());
                        }
                    }
                }
            }

            // Assign UID to interactive elements
            let uid = if INTERACTIVE_ROLES.contains(&role.as_str()) {
                if let Some(bid) = backend_id {
                    uid_counter += 1;
                    let uid = format!("s{uid_counter}");
                    self.uid_map.insert(uid.clone(), bid);
                    Some(uid)
                } else {
                    None
                }
            } else {
                None
            };

            // Skip empty/generic nodes
            if role.is_empty() || (role == "generic" && name.is_empty()) {
                continue;
            }

            elements.push(PageElement {
                role,
                name,
                uid,
                properties: if props.is_empty() { None } else { Some(props) },
            });
        }

        Ok(elements)
    }

    /// Click element by UID.
    pub async fn click(&mut self, uid: &str) -> Result<(), String> {
        let backend_id = self.uid_map.get(uid).copied()
            .ok_or_else(|| format!("UID '{uid}' not found — run snapshot first"))?;

        self.session.ensure_domain("DOM").await.map_err(|e| e.to_string())?;

        // Scroll into view
        let params = serde_json::json!({ "backendNodeId": backend_id });
        self.session.send_command("DOM.scrollIntoViewIfNeeded", Some(params))
            .await.map_err(|e| format!("Scroll failed: {e}"))?;

        // Get element center
        let params = serde_json::json!({ "backendNodeId": backend_id });
        let box_model = self.session.send_command("DOM.getBoxModel", Some(params))
            .await.map_err(|e| format!("getBoxModel failed: {e}"))?;

        let content = box_model["model"]["content"].as_array()
            .ok_or("No content quad in box model")?;

        if content.len() < 8 {
            return Err("Element has zero size".into());
        }

        let x = (content[0].as_f64().unwrap_or(0.0) + content[4].as_f64().unwrap_or(0.0)) / 2.0;
        let y = (content[1].as_f64().unwrap_or(0.0) + content[5].as_f64().unwrap_or(0.0)) / 2.0;

        // Mouse press + release
        let press = serde_json::json!({
            "type": "mousePressed", "x": x, "y": y,
            "button": "left", "clickCount": 1
        });
        self.session.send_command("Input.dispatchMouseEvent", Some(press))
            .await.map_err(|e| format!("Click press failed: {e}"))?;

        let release = serde_json::json!({
            "type": "mouseReleased", "x": x, "y": y,
            "button": "left", "clickCount": 1
        });
        self.session.send_command("Input.dispatchMouseEvent", Some(release))
            .await.map_err(|e| format!("Click release failed: {e}"))?;

        // Brief wait for effects
        tokio::time::sleep(Duration::from_millis(100)).await;
        Ok(())
    }

    /// Fill a form field by UID.
    pub async fn fill(&mut self, uid: &str, value: &str) -> Result<(), String> {
        // Click the element first to focus it
        self.click(uid).await?;

        // Select all existing text (Ctrl+A) then type new value
        let select_all = serde_json::json!({
            "type": "keyDown", "key": "a", "code": "KeyA", "modifiers": 2
        });
        self.session.send_command("Input.dispatchKeyEvent", Some(select_all))
            .await.map_err(|e| format!("Select all failed: {e}"))?;

        let select_all_up = serde_json::json!({
            "type": "keyUp", "key": "a", "code": "KeyA", "modifiers": 0
        });
        self.session.send_command("Input.dispatchKeyEvent", Some(select_all_up))
            .await.map_err(|e| format!("Select all up failed: {e}"))?;

        // Type each character
        for ch in value.chars() {
            let params = serde_json::json!({
                "type": "char", "text": ch.to_string()
            });
            self.session.send_command("Input.dispatchKeyEvent", Some(params))
                .await.map_err(|e| format!("Typing failed: {e}"))?;
        }

        Ok(())
    }

    /// Get visible page text.
    pub async fn page_text(&mut self) -> Result<String, String> {
        self.session.ensure_domain("Runtime").await.map_err(|e| e.to_string())?;

        let result = self.session.send_command(
            "Runtime.evaluate",
            Some(serde_json::json!({ "expression": "document.body.innerText" })),
        ).await.map_err(|e| e.to_string())?;

        Ok(result["result"]["value"].as_str().unwrap_or_default().to_string())
    }

    /// Execute JavaScript, return result as string.
    pub async fn eval_js(&mut self, script: &str) -> Result<String, String> {
        self.session.ensure_domain("Runtime").await.map_err(|e| e.to_string())?;

        let result = self.session.send_command(
            "Runtime.evaluate",
            Some(serde_json::json!({ "expression": script, "returnByValue": true })),
        ).await.map_err(|e| e.to_string())?;

        // Return the value as a string
        let val = &result["result"]["value"];
        if val.is_string() {
            Ok(val.as_str().unwrap_or_default().to_string())
        } else {
            Ok(val.to_string())
        }
    }

    /// Take a screenshot, return base64-encoded PNG.
    pub async fn screenshot(&mut self, full_page: bool) -> Result<String, String> {
        self.session.ensure_domain("Page").await.map_err(|e| e.to_string())?;

        let params = serde_json::json!({
            "format": "png",
            "captureBeyondViewport": full_page,
        });

        let result = self.session.send_command("Page.captureScreenshot", Some(params))
            .await.map_err(|e| e.to_string())?;

        Ok(result["data"].as_str().unwrap_or_default().to_string())
    }

    /// Get network requests (enable Network domain + evaluate).
    pub async fn network_list(&mut self, filter: Option<&str>) -> Result<Value, String> {
        self.session.ensure_domain("Network").await.map_err(|e| e.to_string())?;
        self.session.ensure_domain("Runtime").await.map_err(|e| e.to_string())?;

        // Use Performance.getEntries for a list of loaded resources
        let script = r#"JSON.stringify(
            performance.getEntriesByType('resource').map(e => ({
                name: e.name,
                type: e.initiatorType,
                duration: Math.round(e.duration),
                size: e.transferSize || 0
            }))
        )"#;

        let result = self.session.send_command(
            "Runtime.evaluate",
            Some(serde_json::json!({ "expression": script, "returnByValue": true })),
        ).await.map_err(|e| e.to_string())?;

        let json_str = result["result"]["value"].as_str().unwrap_or("[]");
        let entries: Value = serde_json::from_str(json_str)
            .unwrap_or(Value::Array(vec![]));

        // Filter if requested
        if let Some(f) = filter {
            if let Value::Array(arr) = entries {
                let filtered: Vec<Value> = arr.into_iter()
                    .filter(|e| {
                        e["name"].as_str().unwrap_or_default().contains(f)
                    })
                    .collect();
                Ok(Value::Array(filtered))
            } else {
                Ok(entries)
            }
        } else {
            Ok(entries)
        }
    }

    /// Get current page URL and title.
    async fn get_page_info(&mut self) -> Result<(String, String), String> {
        self.session.ensure_domain("Runtime").await.map_err(|e| e.to_string())?;

        let url_result = self.session.send_command(
            "Runtime.evaluate",
            Some(serde_json::json!({ "expression": "location.href" })),
        ).await.map_err(|e| e.to_string())?;

        let title_result = self.session.send_command(
            "Runtime.evaluate",
            Some(serde_json::json!({ "expression": "document.title" })),
        ).await.map_err(|e| e.to_string())?;

        Ok((
            url_result["result"]["value"].as_str().unwrap_or_default().to_string(),
            title_result["result"]["value"].as_str().unwrap_or_default().to_string(),
        ))
    }
}

/// A page element from the accessibility snapshot.
#[derive(Debug, Clone)]
pub struct PageElement {
    pub role: String,
    pub name: String,
    pub uid: Option<String>,
    pub properties: Option<HashMap<String, Value>>,
}

// ═══════════════════════════════════════════════════════════════════════
// Chromium auto-download — ships with QOR
// ═══════════════════════════════════════════════════════════════════════

/// Where QOR stores its bundled Chromium.
fn chromium_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".qor")
        .join("chromium")
}

/// Platform-specific Chromium executable name inside the download.
#[cfg(target_os = "windows")]
fn chromium_exe_name() -> &'static str { "chrome.exe" }
#[cfg(target_os = "macos")]
fn chromium_exe_name() -> &'static str {
    "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
}
#[cfg(target_os = "linux")]
fn chromium_exe_name() -> &'static str { "chrome" }

/// Platform key for Chrome for Testing downloads.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
fn platform_key() -> &'static str { "win64" }
#[cfg(all(target_os = "windows", target_arch = "x86"))]
fn platform_key() -> &'static str { "win32" }
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn platform_key() -> &'static str { "mac-arm64" }
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn platform_key() -> &'static str { "mac-x64" }
#[cfg(target_os = "linux")]
fn platform_key() -> &'static str { "linux64" }

/// Find Chromium: check QOR's local copy first, then system Chrome.
fn find_chromium() -> Option<PathBuf> {
    // 1. Check QOR's bundled copy
    let local = chromium_dir();
    if local.exists() {
        // Find the executable inside the download directory
        if let Some(exe) = find_exe_in_dir(&local) {
            return Some(exe);
        }
    }

    // 2. Fall back to system Chrome
    chrome::find_chrome_executable(Channel::Stable).ok()
}

/// Recursively find the chrome executable inside a directory.
fn find_exe_in_dir(dir: &std::path::Path) -> Option<PathBuf> {
    let exe_name = chromium_exe_name();
    // Walk directory looking for the executable
    for entry in walkdir(dir) {
        if entry.file_name().map(|n| n.to_string_lossy().ends_with(exe_name)).unwrap_or(false) {
            if entry.is_file() {
                return Some(entry);
            }
        }
        // Also check direct match on filename
        let fname = entry.file_name().unwrap_or_default().to_string_lossy();
        if fname == "chrome" || fname == "chrome.exe" || fname == "chromium" {
            if entry.is_file() {
                return Some(entry);
            }
        }
    }
    None
}

/// Simple recursive directory walker (no external crate needed).
fn walkdir(dir: &std::path::Path) -> Vec<PathBuf> {
    let mut results = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                results.extend(walkdir(&path));
            } else {
                results.push(path);
            }
        }
    }
    results
}

/// Download Chrome for Testing to ~/.qor/chromium/.
/// Uses Google's official Chrome for Testing API.
fn download_chromium() -> Result<PathBuf, String> {
    let platform = platform_key();
    let dest = chromium_dir();
    std::fs::create_dir_all(&dest)
        .map_err(|e| format!("Cannot create {}: {e}", dest.display()))?;

    // Get latest stable version info
    eprintln!("  [BROWSER] Querying Chrome for Testing versions...");
    let api_url = "https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json";
    let resp = ureq::get(api_url).call()
        .map_err(|e| format!("Version API failed: {e}"))?;
    let body = resp.into_string()
        .map_err(|e| format!("Read error: {e}"))?;
    let json: Value = serde_json::from_str(&body)
        .map_err(|e| format!("JSON parse error: {e}"))?;

    // Extract download URL for our platform
    let downloads = &json["channels"]["Stable"]["downloads"]["chrome"];
    let url = downloads.as_array()
        .and_then(|arr| {
            arr.iter().find(|entry| {
                entry["platform"].as_str() == Some(platform)
            })
        })
        .and_then(|entry| entry["url"].as_str())
        .ok_or_else(|| format!("No Chrome for Testing download for platform '{platform}'"))?;

    let version = json["channels"]["Stable"]["version"].as_str().unwrap_or("unknown");
    eprintln!("  [BROWSER] Downloading Chrome {version} for {platform}...");
    eprintln!("  [BROWSER] URL: {url}");

    // Download the zip
    let resp = ureq::get(url).call()
        .map_err(|e| format!("Download failed: {e}"))?;

    let mut zip_data = Vec::new();
    resp.into_reader().read_to_end(&mut zip_data)
        .map_err(|e| format!("Download read failed: {e}"))?;

    eprintln!("  [BROWSER] Downloaded {} MB — extracting...",
        zip_data.len() / 1_048_576);

    // Extract zip
    let cursor = std::io::Cursor::new(&zip_data);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| format!("Zip open failed: {e}"))?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)
            .map_err(|e| format!("Zip entry error: {e}"))?;

        let out_path = dest.join(file.mangled_name());

        if file.is_dir() {
            std::fs::create_dir_all(&out_path).ok();
        } else {
            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            let mut outfile = std::fs::File::create(&out_path)
                .map_err(|e| format!("Extract error: {e}"))?;
            std::io::copy(&mut file, &mut outfile)
                .map_err(|e| format!("Write error: {e}"))?;

            // Set executable permission on Unix
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if let Some(mode) = file.unix_mode() {
                    std::fs::set_permissions(&out_path,
                        std::fs::Permissions::from_mode(mode)).ok();
                }
            }
        }
    }

    eprintln!("  [BROWSER] Chrome {version} installed to {}", dest.display());

    // Find the executable in the extracted files
    find_exe_in_dir(&dest)
        .ok_or_else(|| "Extracted but cannot find chrome executable".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_returns_result() {
        let result = Browser::probe();
        // Just verify it doesn't panic — Chrome may or may not be installed
        let _ = result.chrome_found;
    }

    #[test]
    fn interactive_roles_are_non_empty() {
        assert!(!INTERACTIVE_ROLES.is_empty());
        assert!(INTERACTIVE_ROLES.contains(&"button"));
        assert!(INTERACTIVE_ROLES.contains(&"link"));
        assert!(INTERACTIVE_ROLES.contains(&"textbox"));
    }
}
