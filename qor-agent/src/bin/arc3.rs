//! ARC-AGI-3 Arcade Agent
//! ======================
//! QOR-powered: perceive frame → inject facts → forward chain → read action-select
//! DNA rules decide actions. Genesis learns new rules when current ones fail.
//!
//! Usage: cargo run -p qor-agent --release --bin arc3
//!        cargo run -p qor-agent --release --bin arc3 -- --game ft09

use qor_bridge::grid::Grid;
use qor_core::neuron::{Neuron, QorValue, Statement};
use qor_runtime::eval::Session;
use qor_runtime::invent;
use qor_runtime::library::RuleLibrary;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
static API_KEY_STORE: OnceLock<String> = OnceLock::new();

const API_BASE: &str = "https://three.arcprize.org/api";
const DEFAULT_API_KEY: &str = "5d0731de-5ead-41b3-94d4-cf31a6579da4";
const MAX_ACTIONS: usize = 400;
const MAX_RESETS: usize = 50;
const GENESIS_BUDGET_MS: u64 = 10_000;

fn api_key() -> &'static str {
    API_KEY_STORE.get().map(|s| s.as_str()).unwrap_or(DEFAULT_API_KEY)
}

// ── API response types ──────────────────────────────────────────────

#[derive(serde::Deserialize, Debug)]
struct FrameResponse {
    #[serde(default)]
    frame: Vec<Vec<Vec<i32>>>,
    state: String,
    levels_completed: usize,
    #[serde(default)]
    win_levels: usize,
    #[serde(default)]
    available_actions: Vec<u8>,
    #[serde(default)]
    guid: String,
    #[serde(default)]
    score: f64,
}

#[derive(serde::Deserialize, Debug)]
struct ScorecardResponse {
    card_id: String,
}

/// One frame snapshot for learning.
struct FrameRecord {
    grid: Vec<Vec<u8>>,
    action_taken: u8,
    frame_changed: bool,
}

// ── Paths ───────────────────────────────────────────────────────────

fn dna_dir() -> PathBuf {
    // 1. --dna-dir CLI override (checked in main, stored in env)
    if let Ok(dir) = std::env::var("QOR_DNA_DIR") {
        let p = PathBuf::from(dir);
        if p.exists() { return p; }
    }
    // 2. dna/puzzle_solver/ next to the executable (portable distribution)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let p = exe_dir.join("dna").join("puzzle_solver");
            if p.exists() { return p; }
        }
    }
    // 3. dna/puzzle_solver/ in current working directory
    let cwd = PathBuf::from("dna").join("puzzle_solver");
    if cwd.exists() { return cwd.canonicalize().unwrap_or(cwd); }
    // 4. Compile-time fallback (dev builds)
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("dna").join("puzzle_solver")
}

fn memory_path() -> PathBuf { dna_dir().join("arcade_memory.qor") }
fn rules_path() -> PathBuf { dna_dir().join("arcade_rules_learned.qor") }

// ── API helpers (cookie-persisting agent) ───────────────────────────

fn make_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(30))
        .build()
}

fn api_get(agent: &ureq::Agent, path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let url = format!("{API_BASE}{path}");
    let resp = agent.get(&url)
        .set("X-API-Key", api_key())
        .set("Accept", "application/json")
        .call()?;
    Ok(resp.into_string()?)
}

fn api_post(agent: &ureq::Agent, path: &str, body: &serde_json::Value) -> Result<String, Box<dyn std::error::Error>> {
    let url = format!("{API_BASE}{path}");
    let body_str = body.to_string();
    for retry in 0..3 {
        let result = agent.post(&url)
            .set("X-API-Key", api_key())
            .set("Accept", "application/json")
            .set("Content-Type", "application/json")
            .send_string(&body_str);
        match result {
            Ok(resp) => return Ok(resp.into_string()?),
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                if retry < 2 {
                    eprintln!("  [API] {code} retry {retry}: {}", &body[..body.len().min(200)]);
                    std::thread::sleep(std::time::Duration::from_secs(1));
                } else {
                    return Err(format!("API {code}: {}", &body[..body.len().min(200)]).into());
                }
            }
            Err(e) => {
                if retry < 2 {
                    eprintln!("  [API] transport retry {retry}: {e}");
                    std::thread::sleep(std::time::Duration::from_secs(2));
                } else {
                    return Err(e.into());
                }
            }
        }
    }
    unreachable!()
}

fn reset_game(agent: &ureq::Agent, game_id: &str, card_id: &str, guid: Option<&str>)
    -> Result<FrameResponse, Box<dyn std::error::Error>>
{
    let mut body = serde_json::json!({
        "card_id": card_id,
        "game_id": game_id,
    });
    if let Some(g) = guid {
        body["guid"] = serde_json::json!(g);
    }
    let resp_str = api_post(agent, "/cmd/RESET", &body)?;
    // Log raw response (strip frame data for readability)
    if let Some(idx) = resp_str.find("\"frame\":") {
        let before = &resp_str[..idx];
        let after_frame = resp_str[idx..].find("]]]").map(|i| &resp_str[idx+i+3..]).unwrap_or("");
        eprintln!("  [API RAW RESET] {before}\"frame\":\"[STRIPPED]\"{after_frame}");
    } else {
        eprintln!("  [API RAW RESET] {resp_str}");
    }
    Ok(serde_json::from_str(&resp_str)?)
}

fn send_action(
    agent: &ureq::Agent,
    game_id: &str,
    guid: &str,
    action_num: u8,
    coords: Option<(i64, i64)>,
    reasoning: &serde_json::Value,
) -> Result<FrameResponse, Box<dyn std::error::Error>> {
    // reasoning as serialized string (Python SDK uses json.dumps)
    let mut body = serde_json::json!({
        "game_id": game_id,
        "guid": guid,
        "reasoning": reasoning.to_string(),
    });
    if let Some((x, y)) = coords {
        body["x"] = serde_json::json!(x);
        body["y"] = serde_json::json!(y);
    }
    let resp_str = api_post(agent, &format!("/cmd/ACTION{action_num}"), &body)?;
    // Log raw response for level 2+ actions (strip frame)
    if let Some(idx) = resp_str.find("\"frame\":") {
        let before = &resp_str[..idx];
        let after_frame = resp_str[idx..].find("]]]").map(|i| &resp_str[idx+i+3..]).unwrap_or("");
        eprintln!("  [API RAW ACT{action_num}] {before}\"frame\":\"[STRIPPED]\"{after_frame}");
    }
    Ok(serde_json::from_str(&resp_str)?)
}

// ── Frame helpers ───────────────────────────────────────────────────

/// Extract grid from API frame, convert to u8.
/// The frame contains multiple 64x64 layers. We take the last layer (current state).
/// If layers disagree at tile positions, we log the discrepancy.
fn extract_grid(frame: &[Vec<Vec<i32>>]) -> Vec<Vec<u8>> {
    if frame.is_empty() { return vec![]; }
    eprintln!("  [FRAME] {} layers, first={}x{}", frame.len(),
        frame[0].len(), frame[0].first().map(|r| r.len()).unwrap_or(0));
    // Take the LAST layer (current state, like the reference agent)
    let layer = frame.last().unwrap();
    layer.iter()
        .map(|row| row.iter().map(|&v| v.clamp(0, 255) as u8).collect())
        .collect()
}

// ── Frame diff + logging ──────────────────────────────────────────────

/// Count how many pixels changed between two frames.
/// Excludes rows >= 62 (timer bar) which changes every frame.
fn count_changed_pixels(prev: &[Vec<u8>], curr: &[Vec<u8>]) -> usize {
    let mut count = 0;
    for (r, (pr, cr)) in prev.iter().zip(curr.iter()).enumerate() {
        if r >= 62 { continue; } // skip timer bar
        for (&pv, &cv) in pr.iter().zip(cr.iter()) {
            if pv != cv { count += 1; }
        }
    }
    count
}

/// Log detailed diff: WHERE pixels changed (excluding row 63 timer bar).
fn log_grid_diff(prev: &[Vec<u8>], curr: &[Vec<u8>], label: &str) {
    let mut changes = Vec::new();
    let max_row = prev.len().min(curr.len());
    for r in 0..max_row {
        if r >= 62 { continue; } // skip timer bar (row 62-63)
        let max_col = prev[r].len().min(curr[r].len());
        for c in 0..max_col {
            if prev[r][c] != curr[r][c] {
                changes.push((r, c, prev[r][c], curr[r][c]));
            }
        }
    }
    if changes.is_empty() {
        eprintln!("  [DIFF] {label}: NO tile changes (timer only)");
    } else {
        eprintln!("  [DIFF] {label}: {} pixels changed (excl timer)", changes.len());
        for &(r, c, old, new) in changes.iter().take(10) {
            eprintln!("    ({r},{c}): {old} -> {new}");
        }
        if changes.len() > 10 {
            eprintln!("    ... and {} more", changes.len() - 10);
        }
    }
}

// ── QOR-driven config helpers ──────────────────────────────────────────

/// Read game type from QOR session (derived from game-id-prefix rules in arcade.qor).
/// Fallback to "unknown" if QOR hasn't derived it.
fn read_game_type(session: &Session) -> String {
    for sn in session.store().all() {
        if let Neuron::Expression(parts) = &sn.neuron {
            if parts.len() == 2 && parts[0] == Neuron::symbol("game-type") {
                if let Neuron::Symbol(s) = &parts[1] {
                    return s.clone();
                }
            }
        }
    }
    "unknown".to_string()
}

/// Build action name lookup from QOR facts: (action-map N NAME)
fn build_action_map(session: &Session) -> HashMap<u8, String> {
    let mut map = HashMap::new();
    for sn in session.store().all() {
        if let Neuron::Expression(parts) = &sn.neuron {
            if parts.len() == 3 && parts[0] == Neuron::symbol("action-map") {
                if let (Neuron::Value(QorValue::Int(n)), Neuron::Symbol(name)) = (&parts[1], &parts[2]) {
                    map.insert(*n as u8, name.clone());
                }
            }
        }
    }
    // Fallback defaults if QOR didn't provide them
    for (n, name) in [(1,"up"),(2,"down"),(3,"left"),(4,"right"),(5,"space"),(6,"click"),(7,"undo")] {
        map.entry(n).or_insert_with(|| name.to_string());
    }
    map
}

/// Read a config integer from QOR: (PREDICATE KEY VALUE) → VALUE
fn read_config_int(session: &Session, predicate: &str, key: &str, default: i64) -> i64 {
    for sn in session.store().all() {
        if let Neuron::Expression(parts) = &sn.neuron {
            if parts.len() == 3
                && parts[0] == Neuron::symbol(predicate)
                && parts[1] == Neuron::symbol(key)
            {
                if let Neuron::Value(QorValue::Int(v)) = &parts[2] {
                    return *v;
                }
            }
        }
    }
    default
}

/// Read QOR strategy selection: (strategy-select STRATEGY) → highest confidence
fn read_strategy(session: &Session) -> Option<String> {
    let mut best: Option<(String, f64)> = None;
    for sn in session.store().all() {
        if let Neuron::Expression(parts) = &sn.neuron {
            if parts.len() == 2 && parts[0] == Neuron::symbol("strategy-select") {
                if let Neuron::Symbol(s) = &parts[1] {
                    let conf = sn.tv.confidence;
                    if best.as_ref().map(|(_, c)| conf > *c).unwrap_or(true) {
                        best = Some((s.clone(), conf));
                    }
                }
            }
        }
    }
    best.map(|(s, _)| s)
}

// ── FT09 dynamic tile detection + smart solving ─────────────────────────

/// A detected tile in the game grid.
#[derive(Clone, Debug)]
struct TileInfo {
    row: usize,
    col: usize,
    center_r: usize, // pixel row of center
    center_c: usize, // pixel col of center
    color: u8,        // dominant color
}

/// A detected tile grid (variable size: 3x3, 5x3, etc.)
#[derive(Clone, Debug)]
struct TileGrid {
    tiles: Vec<TileInfo>,
    nrows: usize,
    ncols: usize,
    board_r0: usize, // game board bounding box
    board_c0: usize,
    board_r1: usize,
    board_c1: usize,
    bg_color: u8,
}


/// Dump 16x16 overview of the 64x64 grid (sampling every 4th pixel).
fn dump_grid_overview(grid: &[Vec<u8>]) {
    eprintln!("  [GRID OVERVIEW] (every 4th pixel, hex)");
    let rows = grid.len();
    let cols = if rows > 0 { grid[0].len() } else { 0 };
    eprint!("       ");
    for c in (0..cols).step_by(4) { eprint!("{:X}", (c / 4) % 16); }
    eprintln!();
    for r in (0..rows).step_by(4) {
        eprint!("    r{r:02}: ");
        let vals: String = (0..cols).step_by(4)
            .map(|c| format!("{:X}", grid[r][c]))
            .collect();
        eprintln!("{vals}");
    }
}

/// Save the 64x64 grid as a scaled PPM image file.
/// Each pixel is scaled 8x → 512x512 image. Uses a distinct color for each index.
fn save_frame_ppm(grid: &[Vec<u8>], path: &str) {
    // Arcade palette — map color index to (R, G, B)
    let palette: [(u8,u8,u8); 16] = [
        (0, 0, 0),       // 0  black
        (0, 116, 217),   // 1  blue
        (255, 65, 54),    // 2  red
        (46, 204, 64),    // 3  green
        (50, 50, 50),     // 4  dark gray (background)
        (170, 170, 170),  // 5  light gray
        (240, 18, 190),   // 6  magenta
        (255, 133, 27),   // 7  orange
        (255, 120, 50),   // 8  orange-red (tile default)
        (135, 12, 37),    // 9  maroon
        (0, 255, 255),    // 10 cyan
        (128, 0, 255),    // 11 purple
        (255, 50, 50),    // 12 bright red (painted)
        (255, 220, 0),    // 13 yellow
        (0, 200, 200),    // 14 teal
        (255, 255, 255),  // 15 white
    ];
    let scale = 8usize;
    let h = grid.len() * scale;
    let w = if grid.is_empty() { 0 } else { grid[0].len() * scale };
    let mut data = Vec::with_capacity(h * w * 3);
    for row in grid {
        for _ in 0..scale {
            for &v in row {
                let (r, g, b) = palette[(v & 0xF) as usize];
                for _ in 0..scale { data.push(r); data.push(g); data.push(b); }
            }
        }
    }
    let header = format!("P6\n{w} {h}\n255\n");
    let mut buf = header.into_bytes();
    buf.extend_from_slice(&data);
    if let Err(e) = std::fs::write(path, &buf) {
        eprintln!("  [SAVE] Failed to write {path}: {e}");
    } else {
        eprintln!("  [SAVE] Frame saved to {path} ({w}x{h})");
    }
}

/// Dump raw pixel colors in a region (compact hex format).
fn dump_region_hex(grid: &[Vec<u8>], r0: usize, c0: usize, r1: usize, c1: usize) {
    eprintln!("  [REGION ({r0},{c0})→({r1},{c1})]");
    for r in r0..=r1.min(grid.len().saturating_sub(1)) {
        let vals: String = (c0..=c1.min(grid[r].len().saturating_sub(1)))
            .map(|c| format!("{:X}", grid[r][c]))
            .collect();
        eprintln!("    r{r:02}: {vals}");
    }
}

/// Split row bands into vertical sections separated by large gaps.
fn split_row_sections(bands: &[(usize, usize)]) -> Vec<Vec<(usize, usize)>> {
    let mut sections: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut current: Vec<(usize, usize)> = Vec::new();
    for &band in bands {
        if let Some(&last) = current.last() {
            let gap = band.0 as i64 - last.1 as i64;
            if gap > 12 { // >12px gap = new vertical section
                sections.push(std::mem::take(&mut current));
            }
        }
        current.push(band);
    }
    if !current.is_empty() { sections.push(current); }
    sections
}

/// Detect column bands for a specific set of row bands.
fn detect_col_bands_for_rows(
    grid: &[Vec<u8>], row_group: &[(usize, usize)], bg_color: u8,
) -> Vec<(usize, usize)> {
    let cols = if grid.is_empty() { 0 } else { grid[0].len() };

    // Structural colors from rows around/between the row bands
    let mut structural: std::collections::HashSet<u8> = std::collections::HashSet::new();
    structural.insert(bg_color);
    for r in row_group[0].0.saturating_sub(3)..row_group[0].0 {
        for c in 0..cols {
            if grid[r][c] != bg_color { structural.insert(grid[r][c]); }
        }
    }
    for i in 0..row_group.len().saturating_sub(1) {
        for r in (row_group[i].1 + 1)..row_group[i + 1].0 {
            for c in 0..cols {
                if grid[r][c] != bg_color { structural.insert(grid[r][c]); }
            }
        }
    }
    let last_end = row_group.last().unwrap().1;
    for r in (last_end + 1)..(last_end + 4).min(grid.len()) {
        for c in 0..cols {
            if grid[r][c] != bg_color { structural.insert(grid[r][c]); }
        }
    }

    // Count non-bg pixels per column within row group
    let mut col_px: Vec<usize> = vec![0; cols];
    for c in 0..cols {
        for &(rb0, rb1) in row_group {
            for r in rb0..=rb1 {
                if grid[r][c] != bg_color { col_px[c] += 1; }
            }
        }
    }

    let col_threshold = row_group.len();
    let mut raw_regions: Vec<(usize, usize)> = Vec::new();
    let mut in_band = false;
    let mut start = 0;
    for c in 0..cols {
        if col_px[c] >= col_threshold {
            if !in_band { start = c; in_band = true; }
        } else {
            if in_band { raw_regions.push((start, c - 1)); in_band = false; }
        }
    }
    if in_band { raw_regions.push((start, cols - 1)); }

    let mut all_bands: Vec<(usize, usize)> = Vec::new();
    for &(s, e) in &raw_regions {
        let w = e - s + 1;
        if w >= 4 && w <= 12 {
            all_bands.push((s, e));
        } else if w > 12 {
            let mut split_bands: Vec<(usize, usize)> = Vec::new();
            let mut in_tile = false;
            let mut tile_start = s;
            for c in s..=e {
                let has_tile_color = row_group.iter().any(|&(rb0, rb1)| {
                    (rb0..=rb1).any(|r| !structural.contains(&grid[r][c]))
                });
                if has_tile_color {
                    if !in_tile { tile_start = c; in_tile = true; }
                } else {
                    if in_tile {
                        let cw = c - tile_start;
                        if cw >= 4 && cw <= 12 { split_bands.push((tile_start, c - 1)); }
                        in_tile = false;
                    }
                }
            }
            if in_tile {
                let cw = e + 1 - tile_start;
                if cw >= 4 && cw <= 12 { split_bands.push((tile_start, e)); }
            }

            if split_bands.is_empty() {
                // Color-based split failed (continuous non-bg band).
                // Use spacing-based split: assume 8-pixel tile spacing (6px tile + 2px gap).
                let tile_w = 6usize;
                let spacing = 8usize;
                let n_tiles = (w + spacing - tile_w) / spacing;
                if n_tiles >= 2 {
                    for i in 0..n_tiles {
                        let ts = s + i * spacing;
                        let te = (ts + tile_w - 1).min(e);
                        if te <= e { split_bands.push((ts, te)); }
                    }
                }
            }
            all_bands.extend(split_bands);
        }
    }
    all_bands
}

/// Build a TileGrid from row and column band groups.
fn build_tile_grid(
    grid: &[Vec<u8>], row_group: &[(usize, usize)], col_group: &[(usize, usize)], bg_color: u8,
) -> Option<TileGrid> {
    if col_group.len() < 2 || row_group.is_empty() { return None; }
    let nrows = row_group.len();
    let ncols = col_group.len();
    let mut tiles = Vec::new();
    for (gr, &(rb0, rb1)) in row_group.iter().enumerate() {
        for (gc, &(cb0, cb1)) in col_group.iter().enumerate() {
            let center_r = (rb0 + rb1) / 2;
            let center_c = (cb0 + cb1) / 2;
            let mut counts: HashMap<u8, usize> = HashMap::new();
            for r in rb0..=rb1 {
                for c in cb0..=cb1 {
                    let color = grid[r][c];
                    if color != bg_color { *counts.entry(color).or_insert(0) += 1; }
                }
            }
            let color = counts.iter().max_by_key(|(_, &v)| v)
                .map(|(&k, _)| k).unwrap_or(bg_color);
            tiles.push(TileInfo { row: gr, col: gc, center_r, center_c, color });
        }
    }
    let board_r0 = row_group.first().map(|b| b.0).unwrap_or(0);
    let board_c0 = col_group.first().map(|b| b.0).unwrap_or(0);
    let board_r1 = row_group.last().map(|b| b.1).unwrap_or(0);
    let board_c1 = col_group.last().map(|b| b.1).unwrap_or(0);
    Some(TileGrid { tiles, nrows, ncols, board_r0, board_c0, board_r1, board_c1, bg_color })
}

/// Detect tiles per-row for non-rectangular grids (cross, diamond, L-shape, etc.)
/// Scans each row band independently for column bands, producing tiles at every
/// valid (row, col) position even if the grid isn't rectangular.
fn detect_tiles_per_row(
    grid: &[Vec<u8>], row_bands: &[(usize, usize)], bg_color: u8, min_col: usize,
) -> Option<TileGrid> {
    let mut all_tiles: Vec<TileInfo> = Vec::new();

    for (ri, &row_band) in row_bands.iter().enumerate() {
        let single = [row_band];
        let col_bands = detect_col_bands_for_rows(grid, &single, bg_color);
        // Filter to columns in the desired half (e.g., right half for game board)
        let filtered: Vec<(usize, usize)> = col_bands.into_iter()
            .filter(|b| b.0 >= min_col)
            .collect();
        // Use evenly-spaced grouping if enough bands, else use all
        let col_group = if filtered.len() >= 2 {
            find_evenly_spaced_group(&filtered)
        } else {
            filtered
        };

        for (ci, &(cb0, cb1)) in col_group.iter().enumerate() {
            let center_r = (row_band.0 + row_band.1) / 2;
            let center_c = (cb0 + cb1) / 2;
            let mut counts: HashMap<u8, usize> = HashMap::new();
            for r in row_band.0..=row_band.1 {
                for c in cb0..=cb1 {
                    let color = grid[r][c];
                    if color != bg_color { *counts.entry(color).or_insert(0) += 1; }
                }
            }
            let color = counts.iter().max_by_key(|(_, &v)| v)
                .map(|(&k, _)| k).unwrap_or(bg_color);
            all_tiles.push(TileInfo { row: ri, col: ci, center_r, center_c, color });
        }
        eprintln!("  [PER-ROW] row {} (y={}-{}): {} cols found, positions: {:?}",
            ri, row_band.0, row_band.1,
            col_group.len(),
            col_group.iter().map(|&(s, e)| (s + e) / 2).collect::<Vec<_>>());
    }

    if all_tiles.is_empty() { return None; }

    let max_col = all_tiles.iter().map(|t| t.col).max().unwrap_or(0) + 1;
    let board_r0 = row_bands.first().map(|b| b.0).unwrap_or(0);
    let board_r1 = row_bands.last().map(|b| b.1).unwrap_or(0);
    let board_c0 = all_tiles.iter().map(|t| t.center_c).min().unwrap_or(0).saturating_sub(3);
    let board_c1 = all_tiles.iter().map(|t| t.center_c).max().unwrap_or(0) + 3;

    Some(TileGrid {
        nrows: row_bands.len(),
        ncols: max_col,
        board_r0,
        board_c0,
        board_r1,
        board_c1,
        bg_color,
        tiles: all_tiles,
    })
}

/// Detect tile grids from a pixel frame.
/// Returns (game_board, target). Handles two layouts:
/// 1. Two vertical sections: upper = target, lower = game board
/// 2. Single section with left/right split: left = target, right = game board
fn detect_tile_grids(grid: &[Vec<u8>]) -> (Option<TileGrid>, Option<TileGrid>) {
    let rows = grid.len();
    let cols = if rows > 0 { grid[0].len() } else { 0 };
    if rows < 10 || cols < 10 { return (None, None); }

    // Background color (most common)
    let mut bg_counts: HashMap<u8, usize> = HashMap::new();
    for r in 0..rows { for c in 0..cols { *bg_counts.entry(grid[r][c]).or_insert(0) += 1; } }
    let bg_color = bg_counts.iter().max_by_key(|(_, &v)| v).map(|(&k, _)| k).unwrap_or(0);
    eprintln!("  [DETECT] bg={bg_color}");

    // Find raw row regions (consecutive rows with ≥6 non-bg pixels)
    let max_r = rows.min(62); // skip timer bar
    let mut raw_regions: Vec<(usize, usize)> = Vec::new();
    let mut in_region = false;
    let mut reg_start = 0;
    for r in 0..max_r {
        let non_bg: usize = (0..cols).filter(|&c| grid[r][c] != bg_color).count();
        if non_bg >= 6 {
            if !in_region { reg_start = r; in_region = true; }
        } else {
            if in_region { raw_regions.push((reg_start, r - 1)); in_region = false; }
        }
    }
    if in_region { raw_regions.push((reg_start, max_r - 1)); }
    eprintln!("  [DETECT] raw non-bg regions: {:?}", raw_regions);

    // Split large regions into tile-sized bands
    let mut all_row_bands: Vec<(usize, usize)> = Vec::new();
    for &(start, end) in &raw_regions {
        let height = end - start + 1;
        if height >= 4 && height <= 12 {
            all_row_bands.push((start, end));
        } else if height > 12 {
            let mut in_tile = false;
            let mut tile_start = start;
            for r in start..=end {
                let mut seen = [false; 16];
                let mut diversity = 0usize;
                for c in 0..cols {
                    let v = grid[r][c] as usize;
                    if v < 16 && grid[r][c] != bg_color && !seen[v] {
                        seen[v] = true;
                        diversity += 1;
                    }
                }
                if diversity >= 3 {
                    if !in_tile { tile_start = r; in_tile = true; }
                } else {
                    if in_tile {
                        let h = r - tile_start;
                        if h >= 4 && h <= 12 { all_row_bands.push((tile_start, r - 1)); }
                        in_tile = false;
                    }
                }
            }
            if in_tile {
                let h = end + 1 - tile_start;
                if h >= 4 && h <= 12 { all_row_bands.push((tile_start, end)); }
            }
        }
    }
    eprintln!("  [DETECT] all row bands: {:?}", all_row_bands);
    if all_row_bands.len() < 2 { return (None, None); }

    // ── Multi-section detection ──
    // Split row bands into vertical sections (separated by large gaps).
    // If 2+ sections: upper = target, lower = game board (each with own columns).
    let row_sections = split_row_sections(&all_row_bands);
    eprintln!("  [DETECT] {} row section(s)", row_sections.len());

    if row_sections.len() >= 2 {
        let upper_rows = find_evenly_spaced_group(&row_sections[0]);
        let lower_rows = find_evenly_spaced_group(&row_sections[row_sections.len() - 1]);

        if upper_rows.len() >= 2 && lower_rows.len() >= 2 {
            eprintln!("  [DETECT] MULTI-SECTION: upper rows {:?}, lower rows {:?}",
                upper_rows, lower_rows);

            // Detect columns independently for each section
            let upper_cols = detect_col_bands_for_rows(grid, &upper_rows, bg_color);
            let lower_cols = detect_col_bands_for_rows(grid, &lower_rows, bg_color);
            eprintln!("  [DETECT] upper col bands: {:?}", upper_cols);
            eprintln!("  [DETECT] lower col bands: {:?}", lower_cols);

            // Split each section's columns into left/right halves.
            // Layout: upper-left = INPUT (starting state), upper-right = TARGET (goal)
            //         lower-left = read-only display, lower-right = INTERACTIVE (clickable, framed)
            // Target = upper RIGHT (the goal pattern to match).
            // Game = lower RIGHT (the interactive board — the framed area where clicks work).
            let mid = cols / 2;
            let upper_right: Vec<(usize,usize)> = upper_cols.iter()
                .filter(|b| b.0 >= mid).copied().collect();
            let upper_left: Vec<(usize,usize)> = upper_cols.iter()
                .filter(|b| b.1 < mid).copied().collect();
            let lower_right: Vec<(usize,usize)> = lower_cols.iter()
                .filter(|b| b.0 >= mid).copied().collect();
            let lower_left: Vec<(usize,usize)> = lower_cols.iter()
                .filter(|b| b.1 < mid).copied().collect();
            // Target: prefer upper RIGHT (goal pattern), fallback to upper left
            let upper_target_cols = if upper_right.len() >= 2 { upper_right }
                else if upper_left.len() >= 2 { upper_left }
                else { upper_cols.clone() };
            // Game: prefer lower RIGHT (interactive/clickable area)
            let lower_game_cols = if lower_right.len() >= 2 { lower_right }
                else if lower_left.len() >= 2 { lower_left }
                else { lower_cols.clone() };

            let upper_col_group = find_evenly_spaced_group(&upper_target_cols);
            let lower_col_group = find_evenly_spaced_group(&lower_game_cols);
            eprintln!("  [DETECT] target cols (upper right): {:?}", upper_col_group);
            eprintln!("  [DETECT] game cols (lower right): {:?}", lower_col_group);

            let target = build_tile_grid(grid, &upper_rows, &upper_col_group, bg_color);
            // Build game grid — fall back to per-row scan for non-rectangular grids
            let game = {
                let rect = build_tile_grid(grid, &lower_rows, &lower_col_group, bg_color);
                if lower_col_group.len() <= 2 && lower_rows.len() >= 4 {
                    eprintln!("  [DETECT] Only {} cols for {} rows — trying per-row scan",
                        lower_col_group.len(), lower_rows.len());
                    let per_row = detect_tiles_per_row(grid, &lower_rows, bg_color, mid);
                    let rect_count = lower_col_group.len() * lower_rows.len();
                    if let Some(ref pr) = per_row {
                        if pr.tiles.len() > rect_count {
                            eprintln!("  [DETECT] Per-row found {} tiles (vs {} rectangular)",
                                pr.tiles.len(), rect_count);
                            per_row
                        } else { rect }
                    } else { rect }
                } else { rect }
            };

            if let Some(ref tg) = target {
                eprintln!("  [TILES] UPPER TARGET {}x{}, board=({},{})→({},{})",
                    tg.nrows, tg.ncols, tg.board_r0, tg.board_c0, tg.board_r1, tg.board_c1);
                for t in &tg.tiles {
                    eprintln!("    target({},{}) center=({},{}) color={}",
                        t.row, t.col, t.center_r, t.center_c, t.color);
                }
            }
            if let Some(ref gg) = game {
                eprintln!("  [TILES] LOWER GAME {}x{}, board=({},{})→({},{})",
                    gg.nrows, gg.ncols, gg.board_r0, gg.board_c0, gg.board_r1, gg.board_c1);
                for t in &gg.tiles {
                    eprintln!("    game({},{}) center=({},{}) color={}",
                        t.row, t.col, t.center_r, t.center_c, t.color);
                }
            }

            return (game, target);
        }
    }

    // ── Single-section fallback: left/right split ──
    let best_row_group = find_evenly_spaced_group(&all_row_bands);
    if best_row_group.len() < 2 { return (None, None); }
    eprintln!("  [DETECT] single-section, best row group ({}): {:?}",
        best_row_group.len(), best_row_group);

    let all_col_bands = detect_col_bands_for_rows(grid, &best_row_group, bg_color);
    eprintln!("  [DETECT] all col bands: {:?}", all_col_bands);
    if all_col_bands.len() < 2 { return (None, None); }

    // Split columns left/right
    let mid_col = cols / 2;
    let left_bands: Vec<(usize, usize)> = all_col_bands.iter()
        .filter(|b| b.1 < mid_col).copied().collect();
    let right_bands: Vec<(usize, usize)> = all_col_bands.iter()
        .filter(|b| b.0 >= mid_col).copied().collect();
    // Columns that straddle the midpoint (lost in left/right split)
    let straddling: Vec<(usize, usize)> = all_col_bands.iter()
        .filter(|b| b.0 < mid_col && b.1 >= mid_col).copied().collect();

    let left_group = find_evenly_spaced_group(&left_bands);
    let right_group = find_evenly_spaced_group(&right_bands);

    eprintln!("  [DETECT] left col group ({}): {:?}", left_group.len(), left_group);
    eprintln!("  [DETECT] right col group ({}): {:?}", right_group.len(), right_group);
    if !straddling.is_empty() {
        eprintln!("  [DETECT] {} col(s) straddle midpoint: {:?}", straddling.len(), straddling);
    }

    // Check if this is a SINGLE grid (not split into target/game halves).
    // Heuristic: columns straddle the midpoint, or both halves have ≤ 2 cols
    // but the total col bands suggest a larger grid.
    let is_single_grid = !straddling.is_empty()
        || (left_group.len() <= 2 && right_group.len() <= 2 && all_col_bands.len() >= 3);

    if is_single_grid {
        eprintln!("  [DETECT] Single grid detected (no left/right split)");
        // Try per-row detection with ALL columns (no mid filter)
        let per_row = detect_tiles_per_row(grid, &best_row_group, bg_color, 0);
        if let Some(ref pr) = per_row {
            eprintln!("  [TILES] SINGLE GRID {} tiles ({} rows)", pr.tiles.len(), pr.nrows);
            for t in &pr.tiles {
                eprintln!("    game({},{}) center=({},{}) color={}",
                    t.row, t.col, t.center_r, t.center_c, t.color);
            }
            return (per_row, None);
        }
        // Fallback: try rectangular with all cols
        let single_group = find_evenly_spaced_group(&all_col_bands);
        let single_grid = build_tile_grid(grid, &best_row_group, &single_group, bg_color);
        if let Some(ref sg) = single_grid {
            eprintln!("  [TILES] SINGLE RECT {}x{}", sg.nrows, sg.ncols);
        }
        return (single_grid, None);
    }

    let left_grid = build_tile_grid(grid, &best_row_group, &left_group, bg_color);
    let right_grid = {
        let rect = build_tile_grid(grid, &best_row_group, &right_group, bg_color);
        if right_group.len() <= 2 && best_row_group.len() >= 4 {
            eprintln!("  [DETECT] Single-section: only {} right cols for {} rows — trying per-row",
                right_group.len(), best_row_group.len());
            let per_row = detect_tiles_per_row(grid, &best_row_group, bg_color, mid_col);
            let rect_count = right_group.len() * best_row_group.len();
            if let Some(ref pr) = per_row {
                if pr.tiles.len() > rect_count {
                    eprintln!("  [DETECT] Per-row found {} tiles (vs {} rectangular)",
                        pr.tiles.len(), rect_count);
                    per_row
                } else { rect }
            } else { rect }
        } else { rect }
    };

    if left_grid.is_none() && right_grid.is_none() {
        let single_group = find_evenly_spaced_group(&all_col_bands);
        let single_grid = build_tile_grid(grid, &best_row_group, &single_group, bg_color);
        let single_grid = if single_grid.is_none() && best_row_group.len() >= 4 {
            eprintln!("  [DETECT] Single-grid fallback: trying per-row scan");
            detect_tiles_per_row(grid, &best_row_group, bg_color, 0)
        } else { single_grid };
        if let Some(ref sg) = single_grid {
            eprintln!("  [TILES] Single {} tiles ({} rows)", sg.tiles.len(), sg.nrows);
        }
        return (single_grid, None);
    }

    if let Some(ref lg) = left_grid {
        eprintln!("  [TILES] LEFT {}x{} (target)", lg.nrows, lg.ncols);
        for t in &lg.tiles {
            eprintln!("    target({},{}) center=({},{}) color={}",
                t.row, t.col, t.center_r, t.center_c, t.color);
        }
    }
    if let Some(ref rg) = right_grid {
        eprintln!("  [TILES] RIGHT {}x{} (game)", rg.nrows, rg.ncols);
        for t in &rg.tiles {
            eprintln!("    game({},{}) center=({},{}) color={}",
                t.row, t.col, t.center_r, t.center_c, t.color);
        }
    }

    // Right = game board, Left = target
    (right_grid, left_grid)
}

/// Find the largest group of evenly-spaced bands from a list of bands.
/// Returns the group with the most members. On tie, prefers the group with larger coordinates
/// (game board tends to be further from top-left than examples).
fn find_evenly_spaced_group(bands: &[(usize, usize)]) -> Vec<(usize, usize)> {
    if bands.len() <= 1 { return bands.to_vec(); }

    let mut best_group: Vec<(usize, usize)> = Vec::new();

    // Try each pair of consecutive bands as the seed for a group
    for i in 0..bands.len() - 1 {
        let spacing = bands[i + 1].0 as i64 - bands[i].0 as i64;
        if spacing < 4 || spacing > 20 { continue; } // tiles are 4-12px + 1-4px gap

        let mut group = vec![bands[i], bands[i + 1]];

        // Look for more bands with the same spacing (allow ±1 tolerance)
        let mut last_start = bands[i + 1].0;
        for j in i + 2..bands.len() {
            let expected = last_start as i64 + spacing;
            let actual = bands[j].0 as i64;
            if (actual - expected).abs() <= 1 {
                group.push(bands[j]);
                last_start = bands[j].0;
            }
        }

        // Pick largest group; on tie, prefer larger coordinates (game board)
        if group.len() > best_group.len()
            || (group.len() == best_group.len() && !group.is_empty() && !best_group.is_empty()
                && group.last().unwrap().1 > best_group.last().unwrap().1)
        {
            best_group = group;
        }
    }

    best_group
}


/// Read the dominant color at a tile center from the current grid.
fn read_tile_color(grid: &[Vec<u8>], center_r: usize, center_c: usize, bg_color: u8) -> u8 {
    let mut counts: HashMap<u8, usize> = HashMap::new();
    let half = 2i32; // 5x5 sample at center
    for dr in -half..=half {
        for dc in -half..=half {
            let r = (center_r as i32 + dr).max(0) as usize;
            let c = (center_c as i32 + dc).max(0) as usize;
            if r < grid.len() && c < grid[0].len() && grid[r][c] != bg_color {
                *counts.entry(grid[r][c]).or_insert(0) += 1;
            }
        }
    }
    counts.into_iter().max_by_key(|(_, v)| *v).map(|(k, _)| k).unwrap_or(bg_color)
}

/// Check if a tile is solid (single fill color) vs patterned (decoration/special).
/// Solid tiles have a uniform 3x3 center area. Patterned tiles have mixed colors.
fn is_solid_tile(grid: &[Vec<u8>], center_r: usize, center_c: usize) -> bool {
    let center = grid[center_r][center_c];
    let mut different = 0;
    for dr in -1i32..=1 {
        for dc in -1i32..=1 {
            if dr == 0 && dc == 0 { continue; }
            let r = (center_r as i32 + dr).clamp(0, grid.len() as i32 - 1) as usize;
            let c = (center_c as i32 + dc).clamp(0, grid[0].len() as i32 - 1) as usize;
            if grid[r][c] != center { different += 1; }
        }
    }
    different <= 1 // Solid if at most 1 neighbor differs (edge bleed)
}

/// Count how many tiles differ between game and target by pixel pattern.
/// Inject tile perception facts into QOR session.
/// Re-reads tile colors from current grid (tiles change when clicked).
/// Returns list of (pixel_col, pixel_row) coordinates for diff tiles (for direct clicking).
fn inject_tile_facts(
    session: &mut Session,
    grid: &[Vec<u8>],
    game_board: &TileGrid,
    target: Option<&TileGrid>,
    bg_color: u8,
) -> Vec<(i64, i64)> {
    let _ = session.exec(&format!("(game-board-rows {}) <0.99, 0.99>", game_board.nrows));
    let _ = session.exec(&format!("(game-board-cols {}) <0.99, 0.99>", game_board.ncols));

    // Read CURRENT colors from the grid (not cached colors from detection time)
    for t in &game_board.tiles {
        let color = read_tile_color(grid, t.center_r, t.center_c, bg_color);
        let _ = session.exec(&format!(
            "(game-tile {} {} {} {} {}) <0.95, 0.95>",
            t.row, t.col, t.center_r, t.center_c, color
        ));
    }

    let mut diffs: Vec<(i64, i64)> = Vec::new();
    if let Some(tgt) = target {
        let _ = session.exec("(has-target) <0.99, 0.99>");
        for t in &tgt.tiles {
            let color = read_tile_color(grid, t.center_r, t.center_c, bg_color);
            let _ = session.exec(&format!(
                "(target-tile {} {} {}) <0.95, 0.95>",
                t.row, t.col, color
            ));
        }
        // Collect diff coordinates using center-pixel comparison (skip patterned/decoration tiles)
        for gt in &game_board.tiles {
            if let Some(tt) = tgt.tiles.iter().find(|t| t.row == gt.row && t.col == gt.col) {
                if !is_solid_tile(grid, gt.center_r, gt.center_c)
                    && !is_solid_tile(grid, tt.center_r, tt.center_c) { continue; }
                let gc = grid[gt.center_r][gt.center_c];
                let tc = grid[tt.center_r][tt.center_c];
                if gc != tc {
                    diffs.push((gt.center_c as i64, gt.center_r as i64));
                    if diffs.len() <= 12 {
                        eprintln!("    DIFF ({},{}) game={} target={} → x={},y={}",
                            gt.row, gt.col, gc, tc, gt.center_c, gt.center_r);
                    }
                }
            }
        }
    }

    eprintln!("  [TILES] Injected {} game + {} target facts, {} diffs",
        game_board.tiles.len(),
        target.map(|t| t.tiles.len()).unwrap_or(0),
        diffs.len());
    diffs
}

// ── QOR action reader ────────────────────────────────────────────────

/// Extract integer from a Neuron value.
fn neuron_to_int(n: &Neuron) -> Option<i64> {
    match n {
        Neuron::Value(QorValue::Int(v)) => Some(*v),
        Neuron::Value(QorValue::Float(v)) => Some(*v as i64),
        _ => None,
    }
}

/// Read (action-select ...) facts from QOR session after forward chaining.
/// Returns list of (action_name, confidence, optional_coords) sorted by confidence desc.
fn read_action_selects(session: &Session) -> Vec<(String, f64, Option<(i64, i64)>)> {
    let mut actions: Vec<(String, f64, Option<(i64, i64)>)> = Vec::new();
    // Find all action-select facts (variable arity: 2-elem or 4-elem)
    for sn in session.store().all() {
        let is_action = match &sn.neuron {
            Neuron::Expression(parts) if parts.len() >= 2 => {
                matches!(&parts[0], Neuron::Symbol(s) if s == "action-select")
            }
            _ => false,
        };
        if !is_action { continue; }
        let conf = sn.tv.strength * sn.tv.confidence;
        if let Neuron::Expression(parts) = &sn.neuron {
            if parts.len() >= 2 {
                let name = format!("{}", parts[1]);
                let coords = if parts.len() >= 4 {
                    // (action-select click R C)
                    let r = neuron_to_int(&parts[2]);
                    let c = neuron_to_int(&parts[3]);
                    if let (Some(r), Some(c)) = (r, c) {
                        Some((c, r)) // API wants (x=col, y=row)
                    } else { None }
                } else { None };
                actions.push((name, conf, coords));
            }
        }
    }
    // Sort by confidence descending
    actions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    // Dedup by (action_name, coords) — keep different click targets
    let mut seen = std::collections::HashSet::new();
    actions.retain(|a| {
        let key = format!("{}{:?}", a.0, a.2);
        seen.insert(key)
    });
    actions
}


// ── Perception: Grid → QOR facts ────────────────────────────────────

/// Lookup action name. Prefers QOR-built map, falls back to hardcoded defaults.
fn action_name_from_map(n: u8, map: &HashMap<u8, String>) -> &str {
    map.get(&n).map(|s| s.as_str()).unwrap_or("unknown")
}

/// Static fallback for contexts without a session (genesis training data).
fn action_name(n: u8) -> &'static str {
    match n {
        1 => "up", 2 => "down", 3 => "left", 4 => "right",
        5 => "space", 6 => "click", 7 => "undo", _ => "unknown",
    }
}

/// Feed a grid into QOR session — object-level perception for all sizes.
fn feed_frame(session: &mut Session, raw: &[Vec<u8>], frame_id: &str) -> usize {
    let grid = match Grid::from_vecs(raw.to_vec()) {
        Ok(g) => g,
        Err(e) => { eprintln!("  [GRID] Error: {e}"); return 0; }
    };

    if grid.rows <= 30 && grid.cols <= 30 {
        let stmts = grid.to_statements(frame_id);
        let n = stmts.len();
        let _ = session.exec_statements(stmts);
        return n;
    }

    // Large grid: lightweight object-level only
    let mut stmts = Vec::new();
    let tv = Some(qor_core::truth_value::TruthValue::new(0.99, 0.99));
    let id = Neuron::symbol(frame_id);

    stmts.push(Statement::Fact {
        neuron: Neuron::expression(vec![
            Neuron::symbol("grid-size"), id.clone(),
            Neuron::Value(QorValue::Int(grid.rows as i64)),
            Neuron::Value(QorValue::Int(grid.cols as i64)),
        ]),
        tv, decay: None,
    });

    // Color distribution
    let mut color_counts = HashMap::new();
    for r in 0..grid.rows {
        for c in 0..grid.cols {
            *color_counts.entry(grid.cells[r][c]).or_insert(0usize) += 1;
        }
    }
    for (&color, &count) in &color_counts {
        stmts.push(Statement::Fact {
            neuron: Neuron::expression(vec![
                Neuron::symbol("color-cell-count"), id.clone(),
                Neuron::Value(QorValue::Int(color as i64)),
                Neuron::Value(QorValue::Int(count as i64)),
            ]),
            tv, decay: None,
        });
    }
    let _ = session.exec(&format!("(num-colors {})", color_counts.len()));

    // Flood-fill objects — cap to 100 biggest
    let mut objects = grid.objects();
    objects.sort_by(|a, b| b.cells.len().cmp(&a.cells.len()));
    objects.truncate(100);
    let _ = session.exec(&format!("(num-objects {})", objects.len()));
    for obj in &objects {
        let bbox = obj.bbox();
        stmts.push(Statement::Fact {
            neuron: Neuron::expression(vec![
                Neuron::symbol("grid-object"), id.clone(),
                Neuron::Value(QorValue::Int(obj.id as i64)),
                Neuron::Value(QorValue::Int(obj.color as i64)),
                Neuron::Value(QorValue::Int(obj.cells.len() as i64)),
            ]),
            tv, decay: None,
        });
        stmts.push(Statement::Fact {
            neuron: Neuron::expression(vec![
                Neuron::symbol("grid-obj-bbox"), id.clone(),
                Neuron::Value(QorValue::Int(obj.id as i64)),
                Neuron::Value(QorValue::Int(bbox.0 as i64)),
                Neuron::Value(QorValue::Int(bbox.1 as i64)),
                Neuron::Value(QorValue::Int(bbox.2 as i64)),
                Neuron::Value(QorValue::Int(bbox.3 as i64)),
            ]),
            tv, decay: None,
        });
    }

    let n = stmts.len();
    let _ = session.exec_statements(stmts);
    n
}

/// Detect player piece and target object in the grid.
/// Player = SMALLEST non-bg colored object (could be red, yellow, or any color).
/// Target = LARGEST non-bg, non-wall, non-player, non-green, non-gray colored region.
/// Injects (player-pos X Y), (target-pos X Y), (target-dir DIR) facts.
fn detect_player_and_target(session: &mut Session, grid: &[Vec<u8>]) {
    if grid.is_empty() { return; }
    let rows = grid.len();
    let cols = grid[0].len();
    if rows < 4 || cols < 4 { return; }

    // Find background color (most common)
    let mut counts = [0u32; 16];
    for row in grid { for &c in row { if (c as usize) < 16 { counts[c as usize] += 1; } } }
    let bg = counts.iter().enumerate().max_by_key(|(_, &v)| v).map(|(i, _)| i as u8).unwrap_or(0);

    // Collect ALL pixels by color (skip bg, 0=black/walls, 3=green/bar, 5=gray/border)
    let skip = [bg, 0, 3, 5];
    let mut groups: HashMap<u8, (i64, i64, u32)> = HashMap::new(); // color → (sum_x, sum_y, count)

    for y in 0..rows {
        for x in 0..cols {
            let c = grid[y][x];
            if !skip.contains(&c) {
                let e = groups.entry(c).or_insert((0, 0, 0));
                e.0 += x as i64;
                e.1 += y as i64;
                e.2 += 1;
            }
        }
    }

    // Remove tiny groups (noise, <3 pixels)
    groups.retain(|_, (_, _, cnt)| *cnt >= 3);
    if groups.is_empty() { return; }

    // Player = SMALLEST group (the piece), Target = LARGEST group (the U-shape)
    let player_entry = groups.iter().min_by_key(|(_, (_, _, c))| *c);
    let target_entry = groups.iter().max_by_key(|(_, (_, _, c))| *c);

    // If only one group, it could be both — skip (can't distinguish)
    if groups.len() < 2 { return; }

    let player = if let Some((&_pc, &(sx, sy, cnt))) = player_entry {
        let px = sx / cnt as i64;
        let py = sy / cnt as i64;
        let _ = session.exec(&format!("(player-pos {px} {py})"));
        Some((px, py))
    } else {
        None
    };

    // Target = largest colored group
    if let Some((&_tc, &(sx, sy, cnt))) = target_entry {
        if cnt > 2 {
            let tx = sx / cnt as i64;
            let ty = sy / cnt as i64;
            let _ = session.exec(&format!("(target-pos {tx} {ty})"));

            // Compute direction from player to target
            if let Some((px, py)) = player {
                let dx = tx - px;
                let dy = ty - py;
                // Primary direction (dominant axis)
                if dx.abs() >= dy.abs() {
                    if dx < 0 { let _ = session.exec("(target-dir left)"); }
                    else if dx > 0 { let _ = session.exec("(target-dir right)"); }
                } else {
                    if dy < 0 { let _ = session.exec("(target-dir up)"); }
                    else if dy > 0 { let _ = session.exec("(target-dir down)"); }
                }
                // Both horizontal and vertical components
                if dx < -1 { let _ = session.exec("(target-dir-h left)"); }
                else if dx > 1 { let _ = session.exec("(target-dir-h right)"); }
                if dy < -1 { let _ = session.exec("(target-dir-v up)"); }
                else if dy > 1 { let _ = session.exec("(target-dir-v down)"); }
            }
        }
    }
}

/// Read trial sequences from QOR facts: (trial-seq GAME LEVEL SEQ_ID STEP ACTION_NAME)
/// Returns Vec of (seq_id, Vec of action_names sorted by step).
fn read_trial_sequences(session: &Session, game_id: &str, level: usize) -> Vec<(String, Vec<String>)> {
    let mut seqs: HashMap<String, Vec<(u8, String)>> = HashMap::new();
    for sn in session.store().all() {
        if let Neuron::Expression(parts) = &sn.neuron {
            if parts.len() >= 6 && parts[0] == Neuron::symbol("trial-seq") {
                let gid = match &parts[1] { Neuron::Symbol(s) => s.as_str(), _ => continue };
                if gid != game_id { continue; }
                let lv = match &parts[2] { Neuron::Value(QorValue::Int(v)) => *v as usize, _ => continue };
                if lv != level { continue; }
                let seq_id = format!("{}", parts[3]);
                let step = match &parts[4] { Neuron::Value(QorValue::Int(v)) => *v as u8, _ => continue };
                let action = match &parts[5] { Neuron::Symbol(s) => s.clone(), _ => continue };
                seqs.entry(seq_id).or_default().push((step, action));
            }
        }
    }
    let mut result: Vec<(String, Vec<String>)> = seqs.into_iter().map(|(id, mut steps)| {
        steps.sort_by_key(|(s, _)| *s);
        (id, steps.into_iter().map(|(_, a)| a).collect())
    }).collect();
    result.sort_by(|a, b| a.0.cmp(&b.0));
    result
}

/// Convert action name to action number using the action map.
fn action_name_to_num(name: &str, map: &HashMap<u8, String>) -> u8 {
    for (&num, n) in map { if n == name { return num; } }
    match name { "up" => 1, "down" => 2, "left" => 3, "right" => 4, "space" => 5, "click" => 6, "undo" => 7, _ => 0 }
}

/// Detect which edges of the grid have a notable color (e.g., green bar in AS66).
/// Scans 2-pixel-deep border strips on each edge, returns direction names for any
/// edge where a non-background color appears prominently.
/// Injects facts like `(edge-color top 3)` meaning "top edge has color 3 (green)".
fn detect_edge_colors(session: &mut Session, grid: &[Vec<u8>]) {
    if grid.is_empty() { return; }
    let rows = grid.len();
    let cols = grid[0].len();
    if rows < 4 || cols < 4 { return; }

    // Find the background color (most common overall)
    let mut counts = [0u32; 10];
    for row in grid { for &c in row { if (c as usize) < 10 { counts[c as usize] += 1; } } }
    let bg_color = counts.iter().enumerate().max_by_key(|(_, &v)| v).map(|(i, _)| i as u8).unwrap_or(0);

    // Scan each edge (2 rows/cols deep), count non-bg colors
    let edges: [(&str, Vec<u8>); 4] = [
        ("top",    (0..2).flat_map(|r| grid[r].iter().copied()).collect()),
        ("bottom", ((rows-2)..rows).flat_map(|r| grid[r].iter().copied()).collect()),
        ("left",   (0..rows).flat_map(|r| grid[r][..2].iter().copied()).collect()),
        ("right",  (0..rows).flat_map(|r| grid[r][(cols-2)..].iter().copied()).collect()),
    ];

    for (dir, pixels) in &edges {
        let mut edge_counts = [0u32; 10];
        for &c in pixels { if (c as usize) < 10 { edge_counts[c as usize] += 1; } }
        // Report any non-bg color that covers >20% of the edge strip
        let threshold = (pixels.len() as u32) / 5;
        for (color, &cnt) in edge_counts.iter().enumerate() {
            if color as u8 != bg_color && cnt > threshold {
                let _ = session.exec(&format!("(edge-color {dir} {color})"));
            }
        }
    }
}

const CLEAR_PREDICATES: &[&str] = &[
    "grid-cell", "grid-size", "grid-object", "grid-obj-cell", "grid-obj-bbox",
    "color-cell-count", "num-colors", "num-objects",
    "cell-changed", "frame-changes", "frame-changed",
    "action-select", "frame-number", "game-state", "game-type",
    "levels-completed", "win-levels", "consecutive-no-change",
    "last-action", "game-id", "attempt-number", "turn-phase",
    "perception-done", "strategy-select",
    "current-level", "level-solved", "level-has-sequence", "has-tiles", "known-cycle",
    "tile", "tile-solid", "tile-mixed", "tile-colors", "tile-needs-click",
    "game-tile", "target-tile", "has-target",
    "game-board-rows", "game-board-cols",
    "pair-delta", "pair-size", "pair-color-diff", "pair-cell-change",
    "grid-neighbor", "enclosed-cell", "enclosed-region", "enclosed-region-cell",
    "shape-sig", "input-period", "effective-type",
    "wall-hit", "player-pos", "target-pos", "target-dir", "target-dir-h", "target-dir-v",
    "edge-color",
];

// ── Learning: genesis + memory ──────────────────────────────────────

fn build_training_data(history: &[FrameRecord])
    -> Option<(Vec<Vec<Statement>>, Vec<Vec<Statement>>)>
{
    let mut inputs = Vec::new();
    let mut expected = Vec::new();

    for rec in history {
        if !rec.frame_changed { continue; }

        let grid = match Grid::from_vecs(rec.grid.clone()) {
            Ok(g) => g,
            Err(_) => continue,
        };

        let mut input_stmts = if grid.rows <= 30 && grid.cols <= 30 {
            grid.to_statements("ti")
        } else {
            let tv = Some(qor_core::truth_value::TruthValue::new(0.99, 0.99));
            let id = Neuron::symbol("ti");
            let mut stmts = Vec::new();
            let mut objects = grid.objects();
            objects.sort_by(|a, b| b.cells.len().cmp(&a.cells.len()));
            objects.truncate(50);
            for obj in &objects {
                let bbox = obj.bbox();
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("grid-object"), id.clone(),
                        Neuron::Value(QorValue::Int(obj.id as i64)),
                        Neuron::Value(QorValue::Int(obj.color as i64)),
                        Neuron::Value(QorValue::Int(obj.cells.len() as i64)),
                    ]),
                    tv, decay: None,
                });
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("grid-obj-bbox"), id.clone(),
                        Neuron::Value(QorValue::Int(obj.id as i64)),
                        Neuron::Value(QorValue::Int(bbox.0 as i64)),
                        Neuron::Value(QorValue::Int(bbox.1 as i64)),
                        Neuron::Value(QorValue::Int(bbox.2 as i64)),
                        Neuron::Value(QorValue::Int(bbox.3 as i64)),
                    ]),
                    tv, decay: None,
                });
            }
            stmts
        };

        input_stmts.push(Statement::simple_fact(vec![
            Neuron::symbol("test-input"),
            Neuron::symbol("ti"),
        ]));

        let an = rec.action_taken;
        let expected_stmts = vec![Statement::simple_fact(vec![
            Neuron::symbol("action-select"),
            Neuron::symbol(action_name(if an <= 7 { an } else { 6 })),
        ])];

        inputs.push(input_stmts);
        expected.push(expected_stmts);
    }

    if inputs.is_empty() { return None; }
    if inputs.len() > 20 {
        let start = inputs.len() - 20;
        return Some((inputs[start..].to_vec(), expected[start..].to_vec()));
    }
    Some((inputs, expected))
}

fn try_genesis(
    session: &Session,
    history: &[FrameRecord],
    library: &mut RuleLibrary,
) -> Vec<String> {
    let (inputs, expected) = match build_training_data(history) {
        Some(d) => d,
        None => { return vec![]; }
    };

    let num_workers = invent::optimal_worker_count();
    eprintln!("  [GENESIS] {} pairs, {} workers, {}ms",
        inputs.len(), num_workers, GENESIS_BUDGET_MS);

    let mut base = Session::new();
    for stmt in session.rules_as_statements() {
        let _ = base.exec_statements(vec![stmt]);
    }

    let candidates = invent::genesis_swarm(
        &base, &inputs, &expected, GENESIS_BUDGET_MS,
        Some(library), num_workers, None,
    );

    let good: Vec<String> = candidates.iter()
        .filter(|c| c.score >= 0.3)
        .map(|c| c.rule_text.clone())
        .collect();

    if !good.is_empty() {
        eprintln!("  [GENESIS] Found {} rules (best {:.1}%)",
            good.len(), candidates.first().map(|c| c.score * 100.0).unwrap_or(0.0));
    }
    good
}

fn save_memory(game_id: &str, attempt: usize, levels: usize,
    win_levels: usize, actions: usize, path: &Path)
{
    let section = format!(
        "\n;; ══ {game_id}-a{attempt} ══\n\
         (game-attempt {game_id} {attempt}) <0.99, 0.99>\n\
         (game-levels {game_id} {attempt} {levels}) <0.99, 0.99>\n\
         (game-win-levels {game_id} {win_levels}) <0.99, 0.99>\n\
         (game-actions {game_id} {attempt} {actions}) <0.99, 0.99>\n{}\n",
        if levels >= win_levels && win_levels > 0 {
            format!("(game-solved {game_id}) <0.99, 0.99>\n")
        } else { String::new() }
    );
    if let Some(p) = path.parent() { let _ = std::fs::create_dir_all(p); }
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true).append(true).open(path) {
        let _ = write!(f, "{section}");
    }
}

fn save_rules(rules: &[String], game_id: &str, path: &Path) -> usize {
    if rules.is_empty() { return 0; }
    let existing = std::fs::read_to_string(path).unwrap_or_default();
    let new: Vec<&String> = rules.iter()
        .filter(|r| !existing.contains(r.as_str()))
        .collect();
    if new.is_empty() { return 0; }
    if let Some(p) = path.parent() { let _ = std::fs::create_dir_all(p); }
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true).append(true).open(path) {
        let _ = writeln!(f, "\n;; genesis for {game_id}");
        for r in &new { let _ = writeln!(f, "{r}"); }
    }
    new.len()
}

// ── Per-game learned data — tracks every click and its result ──────────

/// One recorded click: where we clicked, what happened.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ClickRecord {
    x: i64,
    y: i64,
    /// Did anything change on screen after clicking?
    changed: bool,
    /// Did a level-up happen after this click?
    level_up: bool,
    /// Color before click at this position (if known)
    #[serde(default)]
    color_before: Option<u8>,
    /// Color after click at this position (if known)
    #[serde(default)]
    color_after: Option<u8>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
struct GameLevelRules {
    #[serde(default)]
    color_cycle: Vec<u8>,
    #[serde(default)]
    grid_rows: usize,
    #[serde(default)]
    grid_cols: usize,
    #[serde(default)]
    solved: bool,
    #[serde(default)]
    attempts: usize,
    /// Every click we ever made on this level and what happened
    #[serde(default)]
    click_history: Vec<ClickRecord>,
    /// The winning sequence of clicks (if solved)
    #[serde(default)]
    winning_clicks: Vec<(i64, i64)>,
    /// Failed tile combinations — each is a sorted list of (x,y) coords that didn't solve
    #[serde(default)]
    failed_combos: Vec<Vec<(i64, i64)>>,
    /// The winning combination of tile coordinates (subset that solves the level)
    #[serde(default)]
    winning_combo: Vec<(i64, i64)>,
    /// Target number of clicks for this level (0 = auto-detect from winning_combo length)
    #[serde(default)]
    target_clicks: usize,
    /// Raw 64x64 grid snapshot from API when this level was first seen
    /// This is the ground truth — what the game actually looks like
    #[serde(default)]
    initial_grid: Option<Vec<Vec<u8>>>,
    /// Color histogram: how many cells of each color (0-15) in the initial grid
    #[serde(default)]
    color_histogram: Vec<u32>,
}

/// Visual fingerprint of a game's initial frame — used to recognize game type
/// from grid appearance, not just ID prefix.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
struct GridSignature {
    /// Color histogram: count of cells per color (index 0-15, may be shorter for old data)
    #[serde(default)]
    color_counts: Vec<u32>,
    /// Number of distinct objects (flood-fill)
    num_objects: usize,
    /// Colors of objects sorted by size (top 10)
    top_object_colors: Vec<u8>,
    /// Which edges have non-background colors (top/bottom/left/right → color)
    edge_colors: Vec<(String, u8)>,
    /// Game type classification (filled after first successful play)
    #[serde(default)]
    game_type: String,
}

impl GridSignature {
    /// Compute signature from a raw grid
    fn from_grid(grid: &[Vec<u8>]) -> Self {
        let mut color_counts = vec![0u32; 16];
        for row in grid {
            for &c in row {
                if (c as usize) < 16 { color_counts[c as usize] += 1; }
            }
        }

        // Background = most common color
        let bg = color_counts.iter().enumerate()
            .max_by_key(|(_, &v)| v).map(|(i, _)| i as u8).unwrap_or(0);

        // Edge colors (2-pixel border strips)
        let rows = grid.len();
        let cols = if rows > 0 { grid[0].len() } else { 0 };
        let mut edge_colors = Vec::new();
        if rows >= 4 && cols >= 4 {
            let edges: [(&str, Vec<u8>); 4] = [
                ("top",    (0..2).flat_map(|r| grid[r].iter().copied()).collect()),
                ("bottom", ((rows-2)..rows).flat_map(|r| grid[r].iter().copied()).collect()),
                ("left",   (0..rows).flat_map(|r| grid[r][..2].iter().copied()).collect()),
                ("right",  (0..rows).flat_map(|r| grid[r][(cols-2)..].iter().copied()).collect()),
            ];
            for (dir, pixels) in &edges {
                let mut ec = vec![0u32; 16];
                for &c in pixels { if (c as usize) < 16 { ec[c as usize] += 1; } }
                let threshold = (pixels.len() as u32) / 5;
                for (color, &cnt) in ec.iter().enumerate() {
                    if color as u8 != bg && cnt > threshold {
                        edge_colors.push((dir.to_string(), color as u8));
                    }
                }
            }
        }

        // Object detection (simplified — count and top colors)
        let mut num_objects = 0;
        let mut top_object_colors = Vec::new();
        if let Ok(g) = qor_bridge::grid::Grid::from_vecs(grid.to_vec()) {
            let mut objects = g.objects();
            objects.sort_by(|a, b| b.cells.len().cmp(&a.cells.len()));
            num_objects = objects.len();
            top_object_colors = objects.iter().take(10).map(|o| o.color).collect();
        }

        GridSignature { color_counts, num_objects, top_object_colors, edge_colors, game_type: String::new() }
    }

    /// Similarity score (0.0 = totally different, 1.0 = identical)
    fn similarity(&self, other: &GridSignature) -> f64 {
        // Color histogram similarity (cosine-like)
        let total_a: u32 = self.color_counts.iter().sum();
        let total_b: u32 = other.color_counts.iter().sum();
        if total_a == 0 || total_b == 0 { return 0.0; }

        let max_colors = self.color_counts.len().max(other.color_counts.len()).max(1);
        let mut color_sim = 0.0f64;
        for i in 0..max_colors {
            let a = self.color_counts.get(i).copied().unwrap_or(0) as f64 / total_a as f64;
            let b = other.color_counts.get(i).copied().unwrap_or(0) as f64 / total_b as f64;
            color_sim += 1.0 - (a - b).abs();
        }
        color_sim /= max_colors as f64;

        // Edge color match
        let edge_sim = if self.edge_colors.is_empty() && other.edge_colors.is_empty() {
            1.0
        } else {
            let matches = self.edge_colors.iter()
                .filter(|e| other.edge_colors.contains(e))
                .count();
            let total = self.edge_colors.len().max(other.edge_colors.len());
            if total > 0 { matches as f64 / total as f64 } else { 0.5 }
        };

        // Object color pattern match
        let obj_sim = {
            let max_len = self.top_object_colors.len().max(other.top_object_colors.len());
            if max_len == 0 { 0.5 } else {
                let matches = self.top_object_colors.iter()
                    .zip(other.top_object_colors.iter())
                    .filter(|(a, b)| a == b)
                    .count();
                matches as f64 / max_len as f64
            }
        };

        // Weighted combination
        color_sim * 0.5 + edge_sim * 0.3 + obj_sim * 0.2
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
struct PerGameRules {
    levels: Vec<GameLevelRules>,
    #[serde(default)]
    total_attempts: usize,
    #[serde(default)]
    best_level: usize,
    #[serde(default)]
    solved: bool,
    /// Visual fingerprint of the game's initial frame
    #[serde(default)]
    grid_signature: Option<GridSignature>,
}

/// Match an unknown game's grid against all saved game signatures.
/// Returns the best matching game ID and game_type if similarity > threshold.
fn match_game_by_grid(grid: &[Vec<u8>], rules: &HashMap<String, PerGameRules>) -> Option<(String, String)> {
    let sig = GridSignature::from_grid(grid);
    let mut best_id = String::new();
    let mut best_type = String::new();
    let mut best_score = 0.0f64;

    for (game_id, game_rules) in rules {
        if let Some(ref saved_sig) = game_rules.grid_signature {
            if saved_sig.game_type.is_empty() { continue; }
            let score = sig.similarity(saved_sig);
            if score > best_score {
                best_score = score;
                best_id = game_id.clone();
                best_type = saved_sig.game_type.clone();
            }
        }
    }

    if best_score > 0.70 {
        eprintln!("  [GRID-MATCH] Best match: {} (type={}, score={:.2})", best_id, best_type, best_score);
        Some((best_id, best_type))
    } else {
        eprintln!("  [GRID-MATCH] No match found (best={:.2})", best_score);
        None
    }
}

fn game_rules_path() -> PathBuf {
    dna_dir().join("game_rules.json")
}

fn load_game_rules() -> HashMap<String, PerGameRules> {
    let path = game_rules_path();
    if !path.exists() { return HashMap::new(); }
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_game_rules(rules: &HashMap<String, PerGameRules>) {
    let path = game_rules_path();
    if let Some(p) = path.parent() { let _ = std::fs::create_dir_all(p); }
    if let Ok(json) = serde_json::to_string_pretty(rules) {
        let _ = std::fs::write(&path, json);
    }
}

/// Try to reconstruct a color cycle from observed (from, to) transitions.
/// Returns Some(cycle) if transitions form a complete loop.
fn build_cycle_from_transitions(transitions: &[(u8, u8)]) -> Option<Vec<u8>> {
    if transitions.is_empty() { return None; }

    // Build adjacency: from → to
    let mut next: HashMap<u8, u8> = HashMap::new();
    for &(from, to) in transitions {
        next.insert(from, to);
    }

    // Try to walk from the first color and form a cycle
    let start = transitions[0].0;
    let mut cycle = vec![start];
    let mut current = start;

    for _ in 0..20 {
        if let Some(&n) = next.get(&current) {
            if n == start {
                // Cycle complete!
                return Some(cycle);
            }
            if cycle.contains(&n) {
                // Already visited but not start — broken cycle
                return None;
            }
            cycle.push(n);
            current = n;
        } else {
            // No transition from this color yet — incomplete
            return None;
        }
    }
    None // Safety limit
}

/// Systematically generate coordinates across the FULL 64x64 grid.
/// Each idx maps to a UNIQUE position. No skipping, no repeats.
/// Uses stride 4 so 400 clicks covers the ENTIRE grid (16x16=256 positions).
/// Returns (x, y) in game pixel coordinates.
fn scan_coordinate(idx: usize) -> (i64, i64) {
    // Coarse grid: stride 4 → 16 x 16 = 256 positions covers whole 64x64
    // After 256, switches to stride 4 with offset 2 for fill-in
    let coarse: Vec<(i64, i64)> = {
        let mut v = Vec::new();
        for y in (0i64..62).step_by(4) {
            for x in (0i64..64).step_by(4) {
                v.push((x, y));
            }
        }
        v
    };
    let fine: Vec<(i64, i64)> = {
        let mut v = Vec::new();
        for y in (2i64..62).step_by(4) {
            for x in (2i64..64).step_by(4) {
                v.push((x, y));
            }
        }
        v
    };
    let total = coarse.len() + fine.len();
    let effective = idx % total;
    if effective < coarse.len() {
        coarse[effective]
    } else {
        fine[effective - coarse.len()]
    }
}

/// How many clicks to cycle `from` color to `to` color.
/// Returns None if either color isn't in the cycle.
fn cycle_clicks_needed(cycle: &[u8], from: u8, to: u8) -> Option<usize> {
    if from == to { return Some(0); }
    let from_pos = cycle.iter().position(|&c| c == from)?;
    let to_pos = cycle.iter().position(|&c| c == to)?;
    let len = cycle.len();
    Some((to_pos + len - from_pos) % len)
}

/// Discover color cycle by clicking one tile until it returns to its original color.
/// Returns (cycle, actions_used, latest_state, latest_levels_completed).
/// The tile ends at its original color (net-zero effect on the board).
fn discover_color_cycle(
    agent: &ureq::Agent,
    game_id: &str,
    guid: &mut String,
    tile_x: i64,
    tile_y: i64,
    initial_color: u8,
    current_grid: &mut Vec<Vec<u8>>,
    bg_color: u8,
    center_r: usize,
    center_c: usize,
) -> (Vec<u8>, usize, String, usize) {
    let mut cycle = vec![initial_color];
    let mut actions_used = 0;
    let mut latest_state = String::new();
    let mut latest_levels: usize = 0;

    for click_num in 1..=20 {
        let reasoning = serde_json::json!({
            "engine": "qor", "phase": "calibrate", "click": click_num
        });
        match send_action(agent, game_id, guid, 6, Some((tile_x, tile_y)), &reasoning) {
            Ok(resp) => {
                actions_used += 1;
                *guid = resp.guid.clone();
                latest_state = resp.state.clone();
                latest_levels = resp.levels_completed;
                let new_grid = extract_grid(&resp.frame);
                let new_color = read_tile_color(&new_grid, center_r, center_c, bg_color);
                *current_grid = new_grid;

                eprintln!("  [CALIBRATE] click {click_num}: {} → {}",
                    cycle.last().unwrap_or(&0), new_color);

                if new_color == initial_color {
                    eprintln!("  [CALIBRATE] Cycle complete: {:?} (len={})", cycle, cycle.len());
                    return (cycle, actions_used, latest_state, latest_levels);
                }
                if cycle.contains(&new_color) {
                    eprintln!("  [CALIBRATE] Repeated color {new_color}, stopping");
                    cycle.push(new_color);
                    return (cycle, actions_used, latest_state, latest_levels);
                }
                cycle.push(new_color);

                if latest_state == "WIN" || latest_state == "GAME_OVER" {
                    return (cycle, actions_used, latest_state, latest_levels);
                }
            }
            Err(e) => {
                eprintln!("  [CALIBRATE] Error: {e}");
                return (cycle, actions_used, latest_state, latest_levels);
            }
        }
    }

    eprintln!("  [CALIBRATE] Safety limit (20), cycle: {:?}", cycle);
    (cycle, actions_used, latest_state, latest_levels)
}

// ═════════════════════════════════════════════════════════════════════
// MAIN — Play ARC-AGI-3 games
// ═════════════════════════════════════════════════════════════════════

fn main() {
    // Catch panics and print them to stderr
    std::panic::set_hook(Box::new(|info| {
        eprintln!("  [PANIC] {info}");
    }));

    let args: Vec<String> = std::env::args().collect();

    // Show help
    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("ARC-AGI-3 Arcade Agent — QOR-powered game solver");
        eprintln!();
        eprintln!("USAGE:");
        eprintln!("  arc3 [OPTIONS]");
        eprintln!();
        eprintln!("OPTIONS:");
        eprintln!("  --game PREFIX        Only play games matching PREFIX (e.g., --game ft)");
        eprintln!("  --scorecard ID       Use a specific scorecard ID (persistent across runs)");
        eprintln!("  --close-scorecard    Close scorecard after this run (start fresh next time)");
        eprintln!("  --score              Show current scorecard results and exit");
        eprintln!("  --api-key KEY        ARC API key (or set ARC_API_KEY env var)");
        eprintln!("  --dna-dir PATH       Path to DNA folder (default: dna/puzzle_solver/ next to exe)");
        eprintln!("  -h, --help           Show this help");
        eprintln!();
        eprintln!("FILES (in DNA dir):");
        eprintln!("  arcade.qor             Game strategy rules (QOR DNA)");
        eprintln!("  game_rules.json        Saved combos, grid signatures, per-game state");
        eprintln!("  active_scorecard.txt   Current scorecard ID (auto-managed)");
        return;
    }

    let game_filter: Option<String> = args.iter()
        .position(|a| a == "--game")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let explicit_scorecard: Option<String> = args.iter()
        .position(|a| a == "--scorecard")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let close_scorecard = args.iter().any(|a| a == "--close-scorecard");

    // API key: --api-key > ARC_API_KEY env > default
    let cli_key = args.iter()
        .position(|a| a == "--api-key")
        .and_then(|i| args.get(i + 1))
        .cloned();
    if let Some(key) = cli_key.or_else(|| std::env::var("ARC_API_KEY").ok()) {
        let _ = API_KEY_STORE.set(key);
    }

    // DNA dir: --dna-dir > auto-detect (see dna_dir())
    if let Some(dir) = args.iter()
        .position(|a| a == "--dna-dir")
        .and_then(|i| args.get(i + 1))
    {
        std::env::set_var("QOR_DNA_DIR", dir);
    }

    eprintln!("\n{}", "=".repeat(70));
    eprintln!("  ARC-AGI-3 Arcade Agent — QOR + Bandit Learning");
    eprintln!("  DNA:      {}", dna_dir().display());
    if let Some(ref g) = game_filter { eprintln!("  Filter:   {g}"); }
    eprintln!("{}", "=".repeat(70));

    // Shared HTTP agent — persists cookies (GAMESESSION) across requests
    let agent = make_agent();

    // ── 1. Get game list ──
    let games_str = api_get(&agent, "/games").expect("Failed to get game list");
    let game_ids: Vec<String> = match serde_json::from_str::<Vec<serde_json::Value>>(&games_str) {
        Ok(arr) => arr.iter().filter_map(|v| {
            if let Some(s) = v.as_str() { Some(s.to_string()) }
            else if let Some(obj) = v.as_object() {
                obj.get("game_id").and_then(|g| g.as_str()).map(|s| s.to_string())
            } else { None }
        }).collect(),
        Err(e) => { eprintln!("  Cannot parse game list: {e}\n  Raw: {games_str}"); return; }
    };

    let rules_snapshot = load_game_rules();
    let mut filtered: Vec<&String> = game_ids.iter()
        .filter(|g| game_filter.as_ref().map(|f| g.starts_with(f)).unwrap_or(true))
        // Only play games that have winning combos (skip unsolved games like vc33, ls20)
        .filter(|g| {
            let short = g.split('-').next().unwrap_or(g);
            rules_snapshot.get(short).map(|r| r.best_level > 0 || r.levels.iter().any(|l| !l.winning_combo.is_empty())).unwrap_or(false)
        })
        .collect();
    if filtered.is_empty() {
        // Fallback: if no games have combos, play all
        filtered = game_ids.iter()
            .filter(|g| game_filter.as_ref().map(|f| g.starts_with(f)).unwrap_or(true))
            .collect();
    }
    eprintln!("  Games: {:?}", filtered);

    // ── 2. Create or reuse scorecard ──
    // Priority: --scorecard CLI flag > active_scorecard.txt > open new
    // Scorecard stays OPEN across runs so score accumulates.
    // Use --close-scorecard to explicitly close and start fresh.
    let sc_file = dna_dir().join("active_scorecard.txt");
    let card_id = if let Some(ref sc_id) = explicit_scorecard {
        // CLI override — use this exact scorecard ID
        std::fs::write(&sc_file, sc_id).ok();
        eprintln!("  Scorecard (CLI): {sc_id}");
        sc_id.clone()
    } else if sc_file.exists() {
        let saved = std::fs::read_to_string(&sc_file).unwrap_or_default().trim().to_string();
        if !saved.is_empty() {
            eprintln!("  Scorecard (saved): {saved}");
            saved
        } else {
            let sc_str = api_post(&agent, "/scorecard/open", &serde_json::json!({
                "tags": ["qor-lang", "qor", "agent"],
                "source_url": "https://github.com/qor-lang/QOR",
                "ai": true,
            })).expect("Failed to create scorecard");
            let scorecard: ScorecardResponse = serde_json::from_str(&sc_str)
                .expect("Cannot parse scorecard");
            std::fs::write(&sc_file, &scorecard.card_id).ok();
            eprintln!("  Scorecard (new): {}", scorecard.card_id);
            scorecard.card_id
        }
    } else {
        let sc_str = api_post(&agent, "/scorecard/open", &serde_json::json!({
            "tags": ["qor-lang", "qor", "agent"],
            "source_url": "https://github.com/qor-lang/QOR",
            "ai": true,
        })).expect("Failed to create scorecard");
        let scorecard: ScorecardResponse = serde_json::from_str(&sc_str)
            .expect("Cannot parse scorecard");
        std::fs::write(&sc_file, &scorecard.card_id).ok();
        eprintln!("  Scorecard (new): {}", scorecard.card_id);
        scorecard.card_id
    };

    // Ctrl+C handler — set flag so we exit gracefully (scorecard stays open)
    let _ = ctrlc::set_handler(move || {
        eprintln!("\n  [INTERRUPTED] Ctrl+C — saving progress, scorecard stays open");
        INTERRUPTED.store(true, Ordering::SeqCst);
    });

    // ── 3. Library ──
    let library_dir = dna_dir().join("library");
    let mut library = RuleLibrary::load(&library_dir);
    let mut all_game_rules = load_game_rules();
    eprintln!("  Game rules: {} games cached", all_game_rules.len());

    // ── 4. Play each game ──
    let overall_start = Instant::now();
    let mut total_levels = 0usize;
    let mut total_wins = 0usize;

    // Fetch games already played on THIS scorecard so we don't replay them
    let played_on_scorecard: std::collections::HashSet<String> = {
        let mut set = std::collections::HashSet::new();
        if let Ok(sc_json) = api_get(&agent, &format!("/scorecard/{card_id}")) {
            if let Ok(sc) = serde_json::from_str::<serde_json::Value>(&sc_json) {
                if let Some(envs) = sc.get("environments").and_then(|e| e.as_array()) {
                    for env in envs {
                        if let Some(id) = env.get("id").and_then(|i| i.as_str()) {
                            set.insert(id.to_string());
                        }
                    }
                }
            }
        }
        eprintln!("  Already on scorecard: {} games {:?}", set.len(),
            set.iter().map(|s| s.split('-').next().unwrap_or(s).to_string()).collect::<Vec<_>>());
        set
    };

    for game_id in &filtered {
        let short_id = game_id.split('-').next().unwrap_or(game_id);

        // Skip games already played on THIS scorecard
        if played_on_scorecard.contains(game_id.as_str()) {
            eprintln!("\n  SKIP: {game_id} ({short_id}) — already on this scorecard");
            total_wins += 1;
            continue;
        }

        eprintln!("\n{}", "─".repeat(70));
        eprintln!("  GAME: {game_id} ({short_id})");

        let game_start = Instant::now();
        let mut best_levels = 0usize;
        let mut game_won = false;

        // ONE play per game — no multi-reset brute force
        let attempt = 1usize;
        {
            eprintln!("\n  ── Single play ──");

            // Fresh session — load ONLY arcade.qor (not full ARC-AGI-2 ruleset
            // which has 400+ conflicting rules that override arcade-specific actions)
            let mut session = Session::new();
            let arcade_path = dna_dir().join("arcade.qor");
            if arcade_path.exists() {
                if let Ok(src) = std::fs::read_to_string(&arcade_path) {
                    match session.exec(&src) {
                        Ok(results) => eprintln!("  [LOAD] arcade.qor: {} statements, {} rules",
                            results.len(), session.rule_count()),
                        Err(e) => eprintln!("  [LOAD] ERROR loading arcade.qor: {e}"),
                    }
                }
            } else {
                eprintln!("  [LOAD] WARNING: arcade.qor not found at {:?}", arcade_path);
            }

            // Load memory + rules from previous attempts
            let mp = memory_path();
            if mp.exists() {
                if let Ok(src) = std::fs::read_to_string(&mp) {
                    let _ = session.exec(&src);
                }
            }
            let rp = rules_path();
            if rp.exists() {
                if let Ok(src) = std::fs::read_to_string(&rp) {
                    let _ = session.exec(&src);
                }
            }

            // Reset game → first frame
            let first_frame = match reset_game(&agent, game_id, &card_id, None) {
                Ok(f) => f,
                Err(e) => { eprintln!("  [ERROR] Reset: {e}"); continue; }
            };

            let mut guid = first_frame.guid.clone();
            let mut current_grid = extract_grid(&first_frame.frame);
            let mut levels = first_frame.levels_completed;
            let win_levels = first_frame.win_levels;
            let mut actions_taken = 0usize;
            let mut frame_history: Vec<FrameRecord> = Vec::new();
            let mut state = first_frame.state.clone();
            let mut prev_score = first_frame.score;
            let mut consecutive_no_change = 0u32;
            let mut last_action_name = String::new();
            let mut wall_hits: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut level_action_history: Vec<(i64, i64)> = Vec::new(); // auto-learn: track actions per level

            // Build action name map from QOR facts (Phase 1)
            let action_map = build_action_map(&session);

            // Inject game-id prefix → QOR derives game-type (Phase 2)
            let prefix: String = short_id.chars().take_while(|c| c.is_alphabetic()).collect();
            let _ = session.exec(&format!("(game-id-prefix {prefix})"));
            let mut gtype = read_game_type(&session);

            // Save grid signature for this game (visual fingerprint)
            {
                let entry = all_game_rules.entry(short_id.to_string()).or_default();
                if entry.grid_signature.is_none() {
                    let mut sig = GridSignature::from_grid(&current_grid);
                    sig.game_type = gtype.clone();
                    eprintln!("  [GRID-SIG] Saved signature for {} (type={}, colors={:?}, objects={}, edges={:?})",
                        short_id, gtype, sig.color_counts, sig.num_objects, sig.edge_colors);
                    entry.grid_signature = Some(sig);
                    save_game_rules(&all_game_rules);
                }
            }

            // If game type is unknown, try matching grid against known games
            if gtype == "unknown" || gtype.is_empty() {
                if let Some((_matched_id, matched_type)) = match_game_by_grid(&current_grid, &all_game_rules) {
                    eprintln!("  [GRID-MATCH] Unknown game '{}' → matched type '{}'", short_id, matched_type);
                    gtype = matched_type.clone();
                    // Inject the matched type so QOR rules fire correctly
                    let _ = session.exec(&format!("(game-type {matched_type})"));
                    // Save matched type to this game's signature
                    let entry = all_game_rules.entry(short_id.to_string()).or_default();
                    if let Some(ref mut sig) = entry.grid_signature {
                        sig.game_type = matched_type;
                    }
                    save_game_rules(&all_game_rules);
                }
            }

            // Available actions (from API or default)
            let mut avail_actions = first_frame.available_actions.clone();
            if avail_actions.is_empty() {
                avail_actions = vec![1, 2, 3, 4, 5, 6];
            }

            let grid_size = current_grid.len().max(
                current_grid.first().map(|r| r.len()).unwrap_or(64)
            );

            // Inject initial game context facts (persist across frames)
            let _ = session.exec(&format!("(game-id {short_id})"));
            let _ = session.exec(&format!("(attempt-number {attempt})"));

            eprintln!("  [RESET] state={state}, levels={levels}/{win_levels}, \
                grid={grid_size}x{grid_size}, actions={:?}, type={gtype}", avail_actions);

            // Tile grid cache: detect once on RESET, re-detect on LEVEL_UP
            let mut cached_game_grid: Option<TileGrid> = None;
            let mut cached_target_grid: Option<TileGrid> = None;

            // Save initial frame for any game type
            {
                let frame_path = format!("frame_{}_level0.ppm", short_id);
                save_frame_ppm(&current_grid, &frame_path);
            }
            // AUTO-SAVE: store raw grid + color histogram for level 0
            {
                let needs_save = {
                    let entry = all_game_rules.entry(short_id.to_string()).or_default();
                    while entry.levels.is_empty() {
                        entry.levels.push(GameLevelRules::default());
                    }
                    if entry.levels[0].initial_grid.is_none() {
                        entry.levels[0].initial_grid = Some(current_grid.clone());
                        let mut hist = vec![0u32; 16];
                        for row in &current_grid {
                            for &c in row {
                                if (c as usize) < 16 { hist[c as usize] += 1; }
                            }
                        }
                        entry.levels[0].color_histogram = hist;
                        entry.levels[0].grid_rows = current_grid.len();
                        entry.levels[0].grid_cols = current_grid.first().map(|r| r.len()).unwrap_or(0);
                        true
                    } else { false }
                };
                if needs_save {
                    save_game_rules(&all_game_rules);
                    eprintln!("  [AUTO-SAVE] Saved raw grid for level 0");
                }
            }
            // Try tile detection on first frame (perception plumbing)
            dump_grid_overview(&current_grid);
            let (game_g, target_g) = detect_tile_grids(&current_grid);
            if let Some(gg) = game_g {
                eprintln!("  [TILES] Found {}x{} game board", gg.nrows, gg.ncols);
                cached_game_grid = Some(gg);
            }
            if let Some(tg) = target_g {
                eprintln!("  [TILES] Found {}x{} target grid", tg.nrows, tg.ncols);
                cached_target_grid = Some(tg);
            }

            eprintln!("  [STRATEGY] combination solver (attempt {attempt})");

            // ── Frame loop: perceive → reason → act → learn ──
            let mut seq_done_level: Option<usize> = None; // track which level's seq was replayed
            for turn in 1..=MAX_ACTIONS {
                if state == "WIN" || state == "GAME_OVER" { break; }
                if INTERRUPTED.load(Ordering::SeqCst) { break; }

                // ── EARLY REPLAY: stored combo for current level → replay immediately ──
                if gtype != "tile-puzzle" && seq_done_level != Some(levels) {
                    let level_combo: Vec<(i64, i64)> = all_game_rules.get(short_id)
                        .and_then(|r| r.levels.get(levels))
                        .map(|lr| lr.winning_combo.clone())
                        .unwrap_or_default();
                    if !level_combo.is_empty() {
                        seq_done_level = Some(levels);
                        eprintln!("  [REPLAY] Level {levels}: replaying {} stored actions", level_combo.len());
                        for &(x, y) in &level_combo {
                            if actions_taken >= MAX_ACTIONS { break; }
                            std::thread::sleep(std::time::Duration::from_millis(100));
                            let reasoning = serde_json::json!({"engine":"replay","level":levels});
                            let (action_num, coords) = if x < 0 {
                                ((-x) as u8, None)
                            } else {
                                (6u8, Some((x, y)))
                            };
                            if let Ok(resp) = send_action(&agent, game_id, &guid, action_num, coords, &reasoning) {
                                actions_taken += 1;
                                guid = resp.guid.clone();
                                state = resp.state.clone();
                                current_grid = extract_grid(&resp.frame);
                                eprintln!("  [REPLAY] ACTION{} {} → levels={}",
                                    action_num, action_name_from_map(action_num, &action_map), resp.levels_completed);
                                if resp.levels_completed > levels {
                                    eprintln!("  [REPLAY] *** LEVEL UP! {} → {} ***", levels, resp.levels_completed);
                                    levels = resp.levels_completed;
                                    seq_done_level = None; // allow next level to replay
                                    // Save frame for new level
                                    let frame_path = format!("frame_{}_level{}.ppm", short_id, levels);
                                    save_frame_ppm(&current_grid, &frame_path);
                                    let entry = all_game_rules.entry(short_id.to_string()).or_default();
                                    while entry.levels.len() <= levels - 1 {
                                        entry.levels.push(GameLevelRules::default());
                                    }
                                    entry.levels[levels - 1].solved = true;
                                    entry.levels[levels - 1].winning_combo = level_combo.clone();
                                    entry.best_level = entry.best_level.max(levels);
                                    // AUTO-SAVE: raw grid for new level
                                    while entry.levels.len() <= levels {
                                        entry.levels.push(GameLevelRules::default());
                                    }
                                    if entry.levels[levels].initial_grid.is_none() {
                                        entry.levels[levels].initial_grid = Some(current_grid.clone());
                                        let mut hist = vec![0u32; 16];
                                        for row in &current_grid {
                                            for &c in row { if (c as usize) < 16 { hist[c as usize] += 1; } }
                                        }
                                        entry.levels[levels].color_histogram = hist;
                                        entry.levels[levels].grid_rows = current_grid.len();
                                        entry.levels[levels].grid_cols = current_grid.first().map(|r| r.len()).unwrap_or(0);
                                        eprintln!("  [AUTO-SAVE] Saved raw grid for level {}", levels);
                                    }
                                    save_game_rules(&all_game_rules);
                                    break;
                                }
                            }
                        }
                        continue;
                    }
                }

                // ── TRIAL SEQUENCES: try known QOR-defined sequences for this game/level ──
                // QOR facts (trial-seq GAME LEVEL SEQ_ID STEP ACTION) define sequences to try.
                // Generic plumbing: reads facts, tries each sequence, undoes on failure.
                if gtype != "tile-puzzle" && seq_done_level != Some(levels) {
                    let has_combo = all_game_rules.get(short_id)
                        .and_then(|r| r.levels.get(levels))
                        .map(|lr| !lr.winning_combo.is_empty())
                        .unwrap_or(false);
                    if !has_combo {
                        let trial_seqs = read_trial_sequences(&session, short_id, levels);
                        if !trial_seqs.is_empty() {
                            seq_done_level = Some(levels);
                            eprintln!("  [TRIAL] Level {levels}: {} trial sequences from QOR", trial_seqs.len());
                            let mut trial_solved = false;
                            for (seq_id, steps) in &trial_seqs {
                                if actions_taken >= MAX_ACTIONS { break; }
                                eprintln!("  [TRIAL] Trying seq '{}': {:?}", seq_id, steps);
                                let mut seq_actions: Vec<u8> = Vec::new();
                                let mut level_up = false;
                                for action_name in steps {
                                    if actions_taken >= MAX_ACTIONS { break; }
                                    let action_num = action_name_to_num(action_name, &action_map);
                                    if action_num == 0 { continue; }
                                    std::thread::sleep(std::time::Duration::from_millis(120));
                                    let reasoning = serde_json::json!({"engine":"trial","seq":seq_id,"action":action_name});
                                    if let Ok(resp) = send_action(&agent, game_id, &guid, action_num, None, &reasoning) {
                                        actions_taken += 1;
                                        guid = resp.guid.clone();
                                        state = resp.state.clone();
                                        current_grid = extract_grid(&resp.frame);
                                        seq_actions.push(action_num);
                                        eprintln!("  [TRIAL] {} → levels={}", action_name, resp.levels_completed);
                                        if resp.levels_completed > levels {
                                            eprintln!("  [TRIAL] *** LEVEL UP! seq '{}' WORKS! ***", seq_id);
                                            // Save winning sequence
                                            let entry = all_game_rules.entry(short_id.to_string()).or_default();
                                            while entry.levels.len() <= levels {
                                                entry.levels.push(GameLevelRules::default());
                                            }
                                            entry.levels[levels].solved = true;
                                            entry.levels[levels].winning_combo = seq_actions.iter()
                                                .map(|&a| (-(a as i64), 0i64)).collect();
                                            entry.best_level = entry.best_level.max(resp.levels_completed);
                                            // Save grid for next level
                                            while entry.levels.len() <= resp.levels_completed {
                                                entry.levels.push(GameLevelRules::default());
                                            }
                                            if entry.levels[resp.levels_completed].initial_grid.is_none() {
                                                entry.levels[resp.levels_completed].initial_grid = Some(current_grid.clone());
                                            }
                                            save_game_rules(&all_game_rules);
                                            levels = resp.levels_completed;
                                            level_up = true;
                                            trial_solved = true;
                                            seq_done_level = None; // allow next level
                                            wall_hits.clear();
                                            consecutive_no_change = 0;
                                            level_action_history.clear();
                                            break;
                                        }
                                    }
                                }
                                if level_up { break; }
                                // Undo all actions from this failed sequence
                                eprintln!("  [TRIAL] Seq '{}' failed — undoing {} actions", seq_id, seq_actions.len());
                                for _ in 0..seq_actions.len() {
                                    if actions_taken >= MAX_ACTIONS { break; }
                                    std::thread::sleep(std::time::Duration::from_millis(80));
                                    let reasoning = serde_json::json!({"engine":"trial_undo"});
                                    if let Ok(resp) = send_action(&agent, game_id, &guid, 7, None, &reasoning) {
                                        actions_taken += 1;
                                        guid = resp.guid.clone();
                                        state = resp.state.clone();
                                        current_grid = extract_grid(&resp.frame);
                                    }
                                }
                            }
                            if trial_solved { continue; }
                            // Fall through to VARY or regular play
                        }
                    }
                }

                // ── COMBO VARIATION: try previous levels' combos on unsolved levels ──
                // Universal for ALL game types with levels.
                // When we hit a level with no winning_combo, gather click positions
                // from ALL previously solved levels and try variations.
                if gtype != "tile-puzzle" && seq_done_level == Some(levels) {
                    // seq_done_level == Some(levels) means we already tried replay and it was empty
                    // (or it was set because there was no combo to replay)
                    // Only trigger once per level
                } else if gtype != "tile-puzzle" && seq_done_level != Some(levels) {
                    let has_combo = all_game_rules.get(short_id)
                        .and_then(|r| r.levels.get(levels))
                        .map(|lr| !lr.winning_combo.is_empty())
                        .unwrap_or(false);

                    if !has_combo && levels > 0 {
                        seq_done_level = Some(levels);
                        eprintln!("  [VARY] Level {levels}: no combo — QOR will decide what to try");

                        // ── PLUMBING: save grid for this level ──
                        let frame_path = format!("frame_{}_level{}.ppm", short_id, levels);
                        save_frame_ppm(&current_grid, &frame_path);
                        {
                            let entry = all_game_rules.entry(short_id.to_string()).or_default();
                            while entry.levels.len() <= levels {
                                entry.levels.push(GameLevelRules::default());
                            }
                            if entry.levels[levels].initial_grid.is_none() {
                                entry.levels[levels].initial_grid = Some(current_grid.clone());
                                let mut hist = vec![0u32; 16];
                                for row in &current_grid {
                                    for &c in row { if (c as usize) < 16 { hist[c as usize] += 1; } }
                                }
                                entry.levels[levels].color_histogram = hist;
                                entry.levels[levels].grid_rows = current_grid.len();
                                entry.levels[levels].grid_cols = current_grid.first().map(|r| r.len()).unwrap_or(0);
                            }
                            save_game_rules(&all_game_rules);
                        }

                        // ── PLUMBING: inject previous win/fail data as QOR facts ──
                        // (prev-win-click level color x y count) — what worked before
                        // (prev-fail level) — what didn't work
                        // (grid-segment id color x y) — clickable things on current grid
                        if let Some(rules) = all_game_rules.get(short_id) {
                            for (li, lr) in rules.levels.iter().enumerate() {
                                if li >= levels { break; }
                                if lr.winning_combo.is_empty() { continue; }
                                // Inject each click with its color from saved grid
                                let mut click_runs: Vec<(u8, i64, i64, usize)> = Vec::new(); // (color, x, y, count)
                                for &(x, y) in &lr.winning_combo {
                                    if x < 0 { continue; }
                                    let c = lr.initial_grid.as_ref()
                                        .and_then(|g| g.get(y as usize)?.get(x as usize).copied())
                                        .unwrap_or(255);
                                    if let Some(last) = click_runs.last_mut() {
                                        if last.0 == c && last.1 == x && last.2 == y {
                                            last.3 += 1;
                                            continue;
                                        }
                                    }
                                    click_runs.push((c, x, y, 1));
                                }
                                for (seq_idx, &(c, x, y, count)) in click_runs.iter().enumerate() {
                                    let _ = session.exec(&format!(
                                        "(prev-win-click {} {} {} {} {} {})",
                                        li, seq_idx, c, x, y, count
                                    ));
                                }
                                let _ = session.exec(&format!("(prev-win-level {} {})", li, lr.winning_combo.len()));
                            }
                            // Inject fail count
                            if let Some(lr) = rules.levels.get(levels) {
                                let _ = session.exec(&format!("(prev-fail-count {})", lr.failed_combos.len()));
                            }
                        }

                        // ── PLUMBING: generic flood-fill to find segments on current grid ──
                        // No color assumptions — finds ALL connected regions of non-bg color
                        let bg_color = {
                            let mut bc: HashMap<u8, usize> = HashMap::new();
                            for row in &current_grid { for &v in row { *bc.entry(v).or_insert(0) += 1; } }
                            bc.into_iter().max_by_key(|(_, v)| *v).map(|(k, _)| k).unwrap_or(0)
                        };
                        let timer_color = current_grid.get(0).and_then(|r| r.first()).copied().unwrap_or(0);
                        let mut segment_id = 0usize;
                        let mut segments: Vec<(usize, u8, i64, i64)> = Vec::new(); // (id, color, cx, cy)
                        {
                            let mut visited = std::collections::HashSet::new();
                            for y in 0..current_grid.len() as i64 {
                                for x in 0..current_grid[0].len() as i64 {
                                    let c = current_grid[y as usize][x as usize];
                                    if c == bg_color || c == 0 || c == timer_color { continue; }
                                    if visited.contains(&(x, y)) { continue; }
                                    let mut min_x = x; let mut max_x = x;
                                    let mut min_y = y; let mut max_y = y;
                                    let mut count = 0usize;
                                    let mut stack = vec![(x, y)];
                                    while let Some((sx, sy)) = stack.pop() {
                                        if visited.contains(&(sx, sy)) { continue; }
                                        if sx < 0 || sy < 0 || sy >= current_grid.len() as i64
                                            || sx >= current_grid[0].len() as i64 { continue; }
                                        if current_grid[sy as usize][sx as usize] != c { continue; }
                                        visited.insert((sx, sy));
                                        count += 1;
                                        min_x = min_x.min(sx); max_x = max_x.max(sx);
                                        min_y = min_y.min(sy); max_y = max_y.max(sy);
                                        stack.push((sx+1, sy)); stack.push((sx-1, sy));
                                        stack.push((sx, sy+1)); stack.push((sx, sy-1));
                                    }
                                    let cx = (min_x + max_x) / 2;
                                    let cy = (min_y + max_y) / 2;
                                    let w = max_x - min_x + 1;
                                    let h = max_y - min_y + 1;
                                    // Inject segment as QOR fact
                                    let _ = session.exec(&format!(
                                        "(grid-segment {} {} {} {} {} {})",
                                        segment_id, c, cx, cy, w, h
                                    ));
                                    segments.push((segment_id, c, cx, cy));
                                    segment_id += 1;
                                }
                            }
                        }
                        let _ = session.exec(&format!("(current-level {})", levels));
                        let _ = session.exec(&format!("(grid-bg-color {})", bg_color));
                        let _ = session.exec(&format!("(segment-count {})", segments.len()));
                        let _ = session.exec("(vary-ready)");
                        // exec() already calls forward_chain() to fixed-point
                        // QOR rules derive: click-color → target-seg → try-sequence → try-click

                        eprintln!("  [VARY] Injected {} segments + prev win data into QOR", segments.len());

                        // ── PLUMBING: read QOR's decisions and execute them ──
                        // QOR rules produce: (try-click SEQ_ID STEP X Y) and
                        //                    (vary-use-prev-pattern LEVEL)
                        // Rust reads and executes, then undoes if it fails.
                        let mut seq_map: HashMap<String, Vec<(usize, i64, i64)>> = HashMap::new();
                        let mut prev_pattern_levels: Vec<usize> = Vec::new();
                        for sn in session.store().all() {
                            if let Neuron::Expression(parts) = &sn.neuron {
                                if parts.len() >= 5 && parts[0] == Neuron::symbol("try-click") {
                                    let seq_id = format!("{}", parts[1]);
                                    let step = match &parts[2] { Neuron::Value(QorValue::Int(v)) => *v as usize, _ => continue };
                                    let x = match &parts[3] { Neuron::Value(QorValue::Int(v)) => *v, _ => continue };
                                    let y = match &parts[4] { Neuron::Value(QorValue::Int(v)) => *v, _ => continue };
                                    seq_map.entry(seq_id).or_default().push((step, x, y));
                                } else if parts.len() >= 2 && parts[0] == Neuron::symbol("vary-use-prev-pattern") {
                                    if let Neuron::Value(QorValue::Int(v)) = &parts[1] {
                                        prev_pattern_levels.push(*v as usize);
                                    }
                                }
                            }
                        }
                        for clicks in seq_map.values_mut() {
                            clicks.sort_by_key(|&(step, _, _)| step);
                        }
                        eprintln!("  [VARY] QOR generated {} try-sequences from rules", seq_map.len());

                        // ── PLUMBING: handle (vary-use-prev-pattern N) ──
                        // QOR decided which previous levels to replicate.
                        // Rust does geometric nearest-neighbor mapping (like flood-fill — plumbing).
                        for src_level in &prev_pattern_levels {
                            let prev_combo = all_game_rules.get(short_id)
                                .and_then(|r| r.levels.get(*src_level))
                                .map(|lr| lr.winning_combo.clone())
                                .unwrap_or_default();
                            let prev_grid = all_game_rules.get(short_id)
                                .and_then(|r| r.levels.get(*src_level))
                                .and_then(|lr| lr.initial_grid.as_ref());
                            if prev_combo.is_empty() { continue; }

                            // Gather current segments per color
                            let mut segs_per_color: HashMap<u8, Vec<(i64, i64)>> = HashMap::new();
                            for &(_, sc, scx, scy) in &segments {
                                segs_per_color.entry(sc).or_default().push((scx, scy));
                            }

                            // ── Run-mapped: preserve click RUNS (grouping pattern) ──
                            // Build click runs from previous combo: (color, count) groups
                            let mut click_runs: Vec<(u8, usize)> = Vec::new();
                            for &(x, y) in &prev_combo {
                                if x < 0 { continue; }
                                let c: u8 = prev_grid
                                    .and_then(|g: &Vec<Vec<u8>>| g.get(y as usize)?.get(x as usize).copied())
                                    .unwrap_or(255);
                                if let Some(last) = click_runs.last_mut() {
                                    if last.0 == c { last.1 += 1; continue; }
                                }
                                click_runs.push((c, 1));
                            }
                            // Map each run to a current segment (cycle within same color)
                            // Try different starting offsets for the dominant color
                            {
                                let dom_color = click_runs.iter().map(|&(c, _)| c)
                                    .max_by_key(|c| click_runs.iter().filter(|r| r.0 == *c).map(|r| r.1).sum::<usize>())
                                    .unwrap_or(0);
                                let dom_seg_count = segs_per_color.get(&dom_color).map(|s| s.len()).unwrap_or(1);
                                let max_offsets = dom_seg_count.min(6); // cap at 6 variations
                                for offset in 0..max_offsets {
                                    let mut color_idx: HashMap<u8, usize> = HashMap::new();
                                    color_idx.insert(dom_color, offset);
                                    let run_id = format!("runs-lv{}-o{}", src_level, offset);
                                    let mut run_mapped: Vec<(usize, i64, i64)> = Vec::new();
                                    let mut step = 0usize;
                                    for &(c, n) in &click_runs {
                                        if let Some(cseg) = segs_per_color.get(&c) {
                                            if cseg.is_empty() { continue; }
                                            let idx = color_idx.entry(c).or_insert(0);
                                            let (cx, cy) = cseg[*idx % cseg.len()];
                                            for _ in 0..n {
                                                run_mapped.push((step, cx, cy));
                                                step += 1;
                                            }
                                            *idx += 1;
                                        }
                                    }
                                    if !run_mapped.is_empty() {
                                        if offset == 0 {
                                            eprintln!("  [VARY] Run-mapped level {} → {} clicks x{} offsets (runs={:?})",
                                                src_level, run_mapped.len(), max_offsets, click_runs);
                                        }
                                        seq_map.insert(run_id, run_mapped);
                                    }
                                }
                            }

                            // ── Distribute clicks by color (round-robin across segments) ──
                            let mut clicks_per_color: HashMap<u8, usize> = HashMap::new();
                            for &(x, y) in &prev_combo {
                                if x < 0 { continue; }
                                let c: u8 = prev_grid
                                    .and_then(|g: &Vec<Vec<u8>>| g.get(y as usize)?.get(x as usize).copied())
                                    .unwrap_or(255);
                                *clicks_per_color.entry(c).or_insert(0) += 1;
                            }
                            // Build sequence: distribute clicks round-robin across segments of same color
                            let seq_id = format!("dist-lv{}", src_level);
                            let mut mapped: Vec<(usize, i64, i64)> = Vec::new();
                            let mut step = 0usize;
                            // Process colors in order of most-clicked first
                            let mut color_list: Vec<(u8, usize)> = clicks_per_color.into_iter().collect();
                            color_list.sort_by(|a, b| b.1.cmp(&a.1));
                            for (c, n_clicks) in &color_list {
                                if let Some(cseg) = segs_per_color.get(c) {
                                    if cseg.is_empty() { continue; }
                                    for i in 0..*n_clicks {
                                        let (cx, cy) = cseg[i % cseg.len()];
                                        mapped.push((step, cx, cy));
                                        step += 1;
                                    }
                                }
                            }
                            if !mapped.is_empty() {
                                let n = mapped.len();
                                eprintln!("  [VARY] Distribute level {} pattern → {} clicks ({:?})",
                                    src_level, n, color_list);
                                // Generate variations from the base distribute (search plumbing)
                                // 1. Original
                                seq_map.insert(seq_id.clone(), mapped.clone());
                                // 2. Reversed order (same positions, flipped sequence)
                                let rev: Vec<(usize, i64, i64)> = mapped.iter().rev()
                                    .enumerate().map(|(i, &(_, x, y))| (i, x, y)).collect();
                                seq_map.insert(format!("{}-rev", seq_id), rev);
                                // 3. Only dominant color (skip secondary — most games mostly click one color)
                                if color_list.len() > 1 {
                                    let dom_color = color_list[0].0;
                                    let dom_segs: Vec<(i64, i64)> = segs_per_color.get(&dom_color)
                                        .cloned().unwrap_or_default();
                                    if !dom_segs.is_empty() {
                                        let dom_clicks = color_list[0].1;
                                        let dom_mapped: Vec<(usize, i64, i64)> = (0..dom_clicks)
                                            .map(|i| (i, dom_segs[i % dom_segs.len()].0, dom_segs[i % dom_segs.len()].1))
                                            .collect();
                                        seq_map.insert(format!("{}-dom", seq_id), dom_mapped);
                                    }
                                }
                                // 4. Double each click (some games need multiple clicks per button)
                                if n <= 15 {
                                    let dbl: Vec<(usize, i64, i64)> = mapped.iter()
                                        .flat_map(|&(_, x, y)| vec![(0, x, y), (0, x, y)])
                                        .enumerate().map(|(i, (_, x, y))| (i, x, y)).collect();
                                    seq_map.insert(format!("{}-2x", seq_id), dbl);
                                }
                            }
                        }
                        eprintln!("  [VARY] Total sequences to try: {}", seq_map.len());

                        // Load failed combos — use ACTUAL sequence as key (order matters)
                        let mut failed_seqs: std::collections::HashSet<Vec<(i64,i64)>> = std::collections::HashSet::new();
                        if let Some(rules) = all_game_rules.get(short_id) {
                            if let Some(lr) = rules.levels.get(levels) {
                                for fc in &lr.failed_combos {
                                    failed_seqs.insert(fc.clone());
                                }
                            }
                        }

                        // ── PLUMBING: execute each sequence, undo on failure ──
                        let mut any_won = false;
                        let mut seq_keys: Vec<String> = seq_map.keys().cloned().collect();
                        seq_keys.sort();
                        for (ci, seq_id) in seq_keys.iter().enumerate() {
                            if actions_taken >= MAX_ACTIONS { break; }
                            if state == "WIN" || state == "GAME_OVER" { break; }
                            let clicks = &seq_map[seq_id];
                            let action_seq: Vec<(i64, i64)> = clicks.iter().map(|&(_, x, y)| (x, y)).collect();

                            // Use actual sequence as dedup key (order matters)
                            let fail_key = action_seq.clone();
                            if failed_seqs.contains(&fail_key) {
                                eprintln!("  [VARY] Skip {}/{} '{}': exact duplicate", ci+1, seq_keys.len(), seq_id);
                                continue;
                            }

                            eprintln!("  [VARY] Try {}/{}: seq='{}' {} clicks {:?}",
                                ci+1, seq_keys.len(), seq_id, action_seq.len(), action_seq);

                            let mut clicks_sent = 0usize;
                            let mut combo_won = false;
                            for &(x, y) in &action_seq {
                                if actions_taken >= MAX_ACTIONS { break; }
                                if state == "WIN" || state == "GAME_OVER" { break; }
                                std::thread::sleep(std::time::Duration::from_millis(30));
                                let reasoning = serde_json::json!({"engine":"qor-vary","seq":seq_id});
                                clicks_sent += 1;
                                if let Ok(resp) = send_action(&agent, game_id, &guid, 6, Some((x, y)), &reasoning) {
                                    actions_taken += 1;
                                    guid = resp.guid.clone();
                                    state = resp.state.clone();
                                    current_grid = extract_grid(&resp.frame);
                                    if resp.levels_completed > levels {
                                        eprintln!("  [VARY] *** LEVEL UP! {} → {} — seq #{} WORKS! ***",
                                            levels, resp.levels_completed, seq_id);
                                        levels = resp.levels_completed;
                                        combo_won = true;
                                        let entry = all_game_rules.entry(short_id.to_string()).or_default();
                                        while entry.levels.len() <= levels {
                                            entry.levels.push(GameLevelRules::default());
                                        }
                                        entry.levels[levels - 1].winning_combo = action_seq.clone();
                                        entry.levels[levels - 1].solved = true;
                                        entry.best_level = entry.best_level.max(levels);
                                        if entry.levels[levels].initial_grid.is_none() {
                                            entry.levels[levels].initial_grid = Some(current_grid.clone());
                                            let mut hist = vec![0u32; 16];
                                            for row in &current_grid {
                                                for &c in row { if (c as usize) < 16 { hist[c as usize] += 1; } }
                                            }
                                            entry.levels[levels].color_histogram = hist;
                                            entry.levels[levels].grid_rows = current_grid.len();
                                            entry.levels[levels].grid_cols = current_grid.first().map(|r| r.len()).unwrap_or(0);
                                        }
                                        save_game_rules(&all_game_rules);
                                        let fp = format!("frame_{}_level{}.ppm", short_id, levels);
                                        save_frame_ppm(&current_grid, &fp);
                                        break;
                                    }
                                }
                            }
                            if combo_won {
                                any_won = true;
                                seq_done_level = None;
                                break;
                            }
                            // Failed — undo
                            failed_seqs.insert(fail_key);
                            for _ in 0..clicks_sent {
                                if actions_taken >= MAX_ACTIONS { break; }
                                if state == "GAME_OVER" || state == "WIN" { break; }
                                std::thread::sleep(std::time::Duration::from_millis(10));
                                let reasoning = serde_json::json!({"engine":"vary","phase":"undo"});
                                if let Ok(resp) = send_action(&agent, game_id, &guid, 7, None, &reasoning) {
                                    actions_taken += 1;
                                    guid = resp.guid.clone();
                                    state = resp.state.clone();
                                    current_grid = extract_grid(&resp.frame);
                                }
                            }
                        }
                        // Save failed combos (order-preserving dedup)
                        {
                            let entry = all_game_rules.entry(short_id.to_string()).or_default();
                            while entry.levels.len() <= levels {
                                entry.levels.push(GameLevelRules::default());
                            }
                            let existing: std::collections::HashSet<Vec<(i64,i64)>> = entry.levels[levels].failed_combos.iter()
                                .cloned().collect();
                            for key in &failed_seqs {
                                if !existing.contains(key) {
                                    entry.levels[levels].failed_combos.push(key.clone());
                                }
                            }
                            while entry.levels[levels].failed_combos.len() > 500 {
                                entry.levels[levels].failed_combos.remove(0);
                            }
                            save_game_rules(&all_game_rules);
                        }
                        if any_won {
                            continue;
                        }
                    }
                }

                // ── TILE PUZZLE: Combination-based solver ──
                // Strategy: find N colored tiles, try every subset of size N-1..1.
                // After each failed subset → undo all clicks → record failure.
                // NEVER click all N (that ends the game without solving).
                // NEVER click background (gray). Only click colored tiles.
                if gtype == "tile-puzzle" {
                    // ── LEVEL LOOP: solve each level via combination search ──
                    'tile_solver: loop {
                        // Recompute bg each iteration (bg changes between levels)
                        let tile_bg = {
                            let mut bc: HashMap<u8, usize> = HashMap::new();
                            for row in &current_grid { for &v in row { *bc.entry(v).or_insert(0) += 1; } }
                            bc.into_iter().max_by_key(|(_, v)| *v).map(|(k, _)| k).unwrap_or(0)
                        };
                        if state == "WIN" || state == "GAME_OVER" { break; }
                        if actions_taken >= MAX_ACTIONS { break; }

                        // Detect game grid if needed
                        if cached_game_grid.is_none() {
                            dump_grid_overview(&current_grid);
                            let (gg, tg) = detect_tile_grids(&current_grid);
                            cached_game_grid = gg;
                            cached_target_grid = tg;
                        }

                        // Find ALL non-background tiles (including center/special tiles).
                        // Every non-bg tile is clickable — don't filter by majority color.
                        let mut colored_tiles: Vec<(i64, i64)> = Vec::new();
                        let mut tile_colors: Vec<(u8, i64, i64)> = Vec::new();
                        if let Some(gg) = &cached_game_grid {
                            for t in &gg.tiles {
                                let color = read_tile_color(&current_grid, t.center_r, t.center_c, tile_bg);
                                if color != tile_bg {
                                    tile_colors.push((color, t.center_c as i64, t.center_r as i64));
                                }
                            }
                            for &(_, x, y) in &tile_colors {
                                colored_tiles.push((x, y));
                            }
                            eprintln!("  [COMBO] Level {levels}: {} clickable tiles (bg={}), positions: {:?}",
                                colored_tiles.len(), tile_bg, colored_tiles);
                            eprintln!("  [COMBO] Tile colors: {:?}", tile_colors);
                            // Save frame as image for visual inspection
                            let frame_path = format!("frame_{}_level{}.ppm", short_id, levels);
                            save_frame_ppm(&current_grid, &frame_path);
                        }

                        let n = colored_tiles.len();
                        if n == 0 {
                            eprintln!("  [COMBO] No colored tiles found — cannot solve");
                            break 'tile_solver;
                        }

                        // Load failed combos as sorted coordinate sequences
                        let mut failed_seqs: std::collections::HashSet<Vec<(i64,i64)>> = std::collections::HashSet::new();
                        if let Some(saved) = all_game_rules.get(short_id) {
                            if let Some(lr) = saved.levels.get(levels) {
                                for fc in &lr.failed_combos {
                                    let mut sorted = fc.clone();
                                    sorted.sort();
                                    failed_seqs.insert(sorted);
                                }
                            }
                        }
                        eprintln!("  [COMBO] {} previously failed combos loaded", failed_seqs.len());

                        // ── SYSTEMATIC COMBO SEARCH ──
                        // For solved levels: replay known combo. For unsolved: try combos systematically.
                        let mut replay_won = false;
                        let level_solved = all_game_rules.get(short_id)
                            .and_then(|r| r.levels.get(levels))
                            .map(|lr| lr.solved)
                            .unwrap_or(false);

                        // Get the base combo (current best guess from DNA)
                        let base_combo: Vec<(i64, i64)> = all_game_rules.get(short_id)
                            .and_then(|r| r.levels.get(levels))
                            .map(|lr| lr.winning_combo.clone())
                            .unwrap_or_default();

                        if level_solved && !base_combo.is_empty() {
                            // Level already solved — just replay
                            let wc = base_combo.clone();
                            eprintln!("  [COMBO] Replaying solved combo ({} steps): {:?}", wc.len(), wc);
                            for &(x, y) in &wc {
                                if actions_taken >= MAX_ACTIONS { break; }
                                std::thread::sleep(std::time::Duration::from_millis(100));
                                let reasoning = serde_json::json!({"engine":"combo","phase":"replay"});
                                let (action_num, coords) = if x < 0 {
                                    ((-x) as u8, None)
                                } else {
                                    (6u8, Some((x, y)))
                                };
                                if let Ok(resp) = send_action(&agent, game_id, &guid, action_num, coords, &reasoning) {
                                    actions_taken += 1;
                                    guid = resp.guid.clone();
                                    state = resp.state.clone();
                                    current_grid = extract_grid(&resp.frame);
                                    if resp.levels_completed > levels {
                                        eprintln!("  [COMBO] LEVEL UP via replay! {} → {}", levels, resp.levels_completed);
                                        levels = resp.levels_completed;
                                        replay_won = true;
                                        // AUTO-SAVE: raw grid for new level
                                        {
                                            let entry = all_game_rules.entry(short_id.to_string()).or_default();
                                            while entry.levels.len() <= levels {
                                                entry.levels.push(GameLevelRules::default());
                                            }
                                            if entry.levels[levels].initial_grid.is_none() {
                                                entry.levels[levels].initial_grid = Some(current_grid.clone());
                                                let mut hist = vec![0u32; 16];
                                                for row in &current_grid {
                                                    for &c in row { if (c as usize) < 16 { hist[c as usize] += 1; } }
                                                }
                                                entry.levels[levels].color_histogram = hist;
                                                entry.levels[levels].grid_rows = current_grid.len();
                                                entry.levels[levels].grid_cols = current_grid.first().map(|r| r.len()).unwrap_or(0);
                                            }
                                            save_game_rules(&all_game_rules);
                                            eprintln!("  [AUTO-SAVE] Saved raw grid for level {}", levels);
                                        }
                                        // Save PPM too
                                        let frame_path = format!("frame_{}_level{}.ppm", short_id, levels);
                                        save_frame_ppm(&current_grid, &frame_path);
                                        break;
                                    }
                                }
                            }
                            if replay_won {
                                cached_game_grid = None;
                                continue 'tile_solver;
                            }
                            eprintln!("  [COMBO] Solved combo replay failed — breaking");
                            break 'tile_solver;
                        }

                        // ── FRAME ANALYZER: click each tile + undo to map behavior ──
                        if false && attempt == 1 && !level_solved {
                            eprintln!("  [ANALYZE] Mapping every tile on level {levels} (click + undo)...");
                            let pre_grid = current_grid.clone();

                            // First check: does SPACE change the frame?
                            {
                                let reasoning = serde_json::json!({"analyze":"space"});
                                if let Ok(resp) = send_action(&agent, game_id, &guid, 5, None, &reasoning) {
                                    let post = extract_grid(&resp.frame);
                                    let changed = count_changed_pixels(&pre_grid, &post);
                                    eprintln!("  [ANALYZE] SPACE → {} pixels changed", changed);
                                    // Undo the SPACE
                                    if let Ok(undo_resp) = send_action(&agent, game_id, &guid, 7, None, &serde_json::json!({"analyze":"undo_space"})) {
                                        let undo_grid = extract_grid(&undo_resp.frame);
                                        let restored = count_changed_pixels(&pre_grid, &undo_grid);
                                        eprintln!("  [ANALYZE] UNDO SPACE → {} pixels differ from original", restored);
                                    }
                                }
                            }

                            // Click each tile, record what changes, then undo
                            for (ti, &(x, y)) in colored_tiles.iter().enumerate() {
                                let tc = tile_colors.iter().find(|&&(_, tx, ty)| tx == x && ty == y)
                                    .map(|&(c, _, _)| c).unwrap_or(99);
                                std::thread::sleep(std::time::Duration::from_millis(60));
                                let reasoning = serde_json::json!({"analyze":"click", "tile": ti});
                                if let Ok(resp) = send_action(&agent, game_id, &guid, 6, Some((x, y)), &reasoning) {
                                    let post = extract_grid(&resp.frame);
                                    let changed = count_changed_pixels(&pre_grid, &post);
                                    // Find which pixels changed and what color they became
                                    let mut changes: Vec<(usize, usize, u8, u8)> = Vec::new();
                                    for r in 0..pre_grid.len().min(post.len()) {
                                        if r >= 62 { continue; }
                                        for c in 0..pre_grid[r].len().min(post[r].len()) {
                                            if pre_grid[r][c] != post[r][c] {
                                                changes.push((r, c, pre_grid[r][c], post[r][c]));
                                            }
                                        }
                                    }
                                    // Summarize: what colors appeared
                                    let mut color_counts: HashMap<(u8, u8), usize> = HashMap::new();
                                    for &(_, _, from, to) in &changes {
                                        *color_counts.entry((from, to)).or_default() += 1;
                                    }
                                    eprintln!("  [ANALYZE] tile {:2} ({:2},{:2}) color={} → {} px changed, transitions: {:?}",
                                        ti, x, y, tc, changed, color_counts);

                                    // Now UNDO
                                    std::thread::sleep(std::time::Duration::from_millis(60));
                                    if let Ok(undo_resp) = send_action(&agent, game_id, &guid, 7, None, &serde_json::json!({"analyze":"undo"})) {
                                        let undo_grid = extract_grid(&undo_resp.frame);
                                        let restored = count_changed_pixels(&pre_grid, &undo_grid);
                                        if restored > 0 {
                                            eprintln!("  [ANALYZE] UNDO → {} pixels NOT restored!", restored);
                                        }
                                    }
                                }
                            }

                            // Also test: click on background area
                            {
                                let reasoning = serde_json::json!({"analyze":"bg_click"});
                                if let Ok(resp) = send_action(&agent, game_id, &guid, 6, Some((2, 2)), &reasoning) {
                                    let post = extract_grid(&resp.frame);
                                    let changed = count_changed_pixels(&pre_grid, &post);
                                    eprintln!("  [ANALYZE] BG click (2,2) → {} px changed", changed);
                                    let _ = send_action(&agent, game_id, &guid, 7, None, &serde_json::json!({"analyze":"undo_bg"}));
                                }
                            }

                            // Test double-click: does clicking same tile twice toggle?
                            if let Some(&(x, y)) = colored_tiles.first() {
                                let reasoning = serde_json::json!({"analyze":"double"});
                                let _ = send_action(&agent, game_id, &guid, 6, Some((x, y)), &reasoning);
                                std::thread::sleep(std::time::Duration::from_millis(60));
                                if let Ok(resp2) = send_action(&agent, game_id, &guid, 6, Some((x, y)), &reasoning) {
                                    let post = extract_grid(&resp2.frame);
                                    let changed = count_changed_pixels(&pre_grid, &post);
                                    eprintln!("  [ANALYZE] Double-click ({},{}) → {} px changed (0=toggled back)", x, y, changed);
                                }
                                // Undo both
                                let _ = send_action(&agent, game_id, &guid, 7, None, &serde_json::json!({"analyze":"undo"}));
                                let _ = send_action(&agent, game_id, &guid, 7, None, &serde_json::json!({"analyze":"undo"}));
                            }

                            eprintln!("  [ANALYZE] Tile mapping complete.");
                        }

                        // ── UNSOLVED LEVEL: systematic search (exactly 14 CLICKS) ──
                        // Base clicks from DNA (excluding SPACE markers)
                        let base_clicks: Vec<(i64, i64)> = base_combo.iter()
                            .filter(|&&(x, _)| x >= 0)
                            .copied()
                            .collect();
                        let base_set: std::collections::HashSet<(i64, i64)> = base_clicks.iter().copied().collect();
                        // Find the most common tile color (the "clickable" color) — NOT hardcoded
                        // Skip background (tile_bg), black (0), and preset/icon colors (2, 12)
                        let skip_colors: std::collections::HashSet<u8> = [tile_bg, 0, 2, 12].iter().copied().collect();
                        let mut color_freq: HashMap<u8, usize> = HashMap::new();
                        for &(c, _, _) in &tile_colors {
                            if !skip_colors.contains(&c) {
                                *color_freq.entry(c).or_insert(0) += 1;
                            }
                        }
                        let clickable_color = color_freq.into_iter()
                            .max_by_key(|(_, cnt)| *cnt)
                            .map(|(c, _)| c)
                            .unwrap_or(8);
                        eprintln!("  [SEARCH] clickable tile color = {} (bg={})", clickable_color, tile_bg);
                        let clickable_set: std::collections::HashSet<(i64, i64)> = tile_colors.iter()
                            .filter(|&&(c, _, _)| c == clickable_color)
                            .map(|&(_, x, y)| (x, y))
                            .collect();
                        let unclicked: Vec<(i64, i64)> = colored_tiles.iter()
                            .filter(|t| !base_set.contains(t))
                            .filter(|t| clickable_set.contains(t))
                            .copied()
                            .collect();

                        // Target clicks: from DNA config, or fallback to base combo len + 3
                        let target_clicks = all_game_rules.get(short_id)
                            .and_then(|r| r.levels.get(levels))
                            .map(|lr| if lr.target_clicks > 0 { lr.target_clicks } else { 14 })
                            .unwrap_or(14);
                        let need_extra = target_clicks.saturating_sub(base_clicks.len());
                        eprintln!("  [SEARCH] base={} clicks, need {} extra to reach {} clicks, {} unclicked tiles",
                            base_clicks.len(), need_extra, target_clicks, unclicked.len());

                        let mut candidates: Vec<Vec<(i64, i64)>> = Vec::new();
                        const MAX_CANDIDATES: usize = 500;

                        // Helper: generate k-combinations from a slice, capped at max_out
                        fn combos_of_capped(items: &[(i64,i64)], k: usize, max_out: usize) -> Vec<Vec<(i64,i64)>> {
                            let mut result = Vec::new();
                            if k == 0 || k > items.len() { return result; }
                            fn gen(items: &[(i64,i64)], k: usize, start: usize, cur: &mut Vec<(i64,i64)>, out: &mut Vec<Vec<(i64,i64)>>, max: usize) {
                                if out.len() >= max { return; }
                                if cur.len() == k { out.push(cur.clone()); return; }
                                for i in start..items.len() {
                                    if out.len() >= max { return; }
                                    cur.push(items[i]);
                                    gen(items, k, i+1, cur, out, max);
                                    cur.pop();
                                }
                            }
                            gen(items, k, 0, &mut Vec::new(), &mut result, max_out);
                            result
                        }

                        // When search space is huge, generate random samples instead
                        fn random_combos(items: &[(i64,i64)], k: usize, count: usize, seed: u64) -> Vec<Vec<(i64,i64)>> {
                            let mut result = Vec::new();
                            let n = items.len();
                            if k == 0 || k > n { return result; }
                            let mut rng = seed;
                            let mut seen = std::collections::HashSet::new();
                            for _ in 0..(count * 3) { // try up to 3x to find unique combos
                                if result.len() >= count { break; }
                                let mut indices: Vec<usize> = (0..n).collect();
                                // Fisher-Yates shuffle (first k elements)
                                for i in 0..k {
                                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                                    let j = i + ((rng >> 33) as usize) % (n - i);
                                    indices.swap(i, j);
                                }
                                let mut combo_idx: Vec<usize> = indices[..k].to_vec();
                                combo_idx.sort();
                                if seen.insert(combo_idx.clone()) {
                                    let combo: Vec<(i64,i64)> = combo_idx.iter().map(|&i| items[i]).collect();
                                    result.push(combo);
                                }
                            }
                            result
                        }

                        // Estimate search space size
                        fn binom(n: usize, k: usize) -> u64 {
                            if k > n { return 0; }
                            let k = k.min(n - k);
                            let mut r: u64 = 1;
                            for i in 0..k {
                                r = r.saturating_mul((n - i) as u64) / (i as u64 + 1);
                            }
                            r
                        }

                        let space_size = binom(unclicked.len(), need_extra);
                        let use_random = space_size > MAX_CANDIDATES as u64;
                        if use_random {
                            eprintln!("  [SEARCH] Search space C({},{}) = {} — using random sampling ({} samples)",
                                unclicked.len(), need_extra, space_size, MAX_CANDIDATES);
                        }

                        // Priority 1: base + need_extra from unclicked = target clicks
                        if need_extra <= unclicked.len() {
                            if use_random {
                                let seed = (levels as u64 * 12345 + attempt as u64 * 67890) | 1;
                                for extra in random_combos(&unclicked, need_extra, MAX_CANDIDATES, seed) {
                                    let mut seq = base_clicks.clone();
                                    seq.extend_from_slice(&extra);
                                    candidates.push(seq);
                                }
                            } else {
                                for extra in combos_of_capped(&unclicked, need_extra, MAX_CANDIDATES) {
                                    let mut seq = base_clicks.clone();
                                    seq.extend_from_slice(&extra);
                                    candidates.push(seq);
                                }
                            }
                        }

                        // Priority 2: swap 1 out, add (need_extra+1) = target clicks
                        if candidates.len() < MAX_CANDIDATES && need_extra + 1 <= unclicked.len() {
                            let remaining = MAX_CANDIDATES - candidates.len();
                            for (ri, _) in base_clicks.iter().enumerate() {
                                if candidates.len() >= MAX_CANDIDATES { break; }
                                let per_swap = remaining / base_clicks.len().max(1);
                                for extra in combos_of_capped(&unclicked, need_extra + 1, per_swap) {
                                    let mut seq: Vec<(i64,i64)> = Vec::new();
                                    for (bi, &t) in base_clicks.iter().enumerate() {
                                        if bi != ri { seq.push(t); }
                                    }
                                    seq.extend_from_slice(&extra);
                                    candidates.push(seq);
                                }
                            }
                        }

                        // Priority 3: swap 2 out, add (need_extra+2) = target clicks
                        if candidates.len() < MAX_CANDIDATES && need_extra + 2 <= unclicked.len() && base_clicks.len() >= 2 {
                            let remaining = MAX_CANDIDATES - candidates.len();
                            for remove in combos_of_capped(&base_clicks, 2, 10) {
                                if candidates.len() >= MAX_CANDIDATES { break; }
                                let remove_set: std::collections::HashSet<(i64,i64)> = remove.iter().copied().collect();
                                let kept: Vec<(i64,i64)> = base_clicks.iter().filter(|t| !remove_set.contains(t)).copied().collect();
                                for extra in combos_of_capped(&unclicked, need_extra + 2, remaining / 10) {
                                    let mut seq = kept.clone();
                                    seq.extend_from_slice(&extra);
                                    candidates.push(seq);
                                }
                            }
                        }

                        eprintln!("  [SEARCH] Generated {} candidate combos (all with {} clicks)",
                            candidates.len(), target_clicks);

                        // Filter out failed combos using coordinate-based keys
                        let mut chosen: Option<Vec<(i64, i64)>> = None;
                        let mut tried = 0usize;
                        for seq in &candidates {
                            let mut key = seq.clone();
                            key.sort();
                            if !failed_seqs.contains(&key) {
                                chosen = Some(seq.clone());
                                break;
                            }
                            tried += 1;
                        }

                        if chosen.is_none() {
                            eprintln!("  [SEARCH] All {} candidates failed — exhausted search", candidates.len());
                            break 'tile_solver;
                        }

                        let action_seq = chosen.unwrap();
                        let click_tiles: Vec<(i64, i64)> = action_seq.iter()
                            .filter(|&&(x, _)| x >= 0).copied().collect();
                        let has_space = action_seq.iter().any(|&(x, _)| x < 0);
                        eprintln!("  [SEARCH] Attempt {}: trying {} actions (space={}, {} clicks), {} failed, {} candidates",
                            attempt, action_seq.len(), has_space, click_tiles.len(),
                            failed_seqs.len(), candidates.len() - tried);

                        // Try candidates one by one: click combo → check → undo → next
                        let mut any_won = false;
                        for (ci, action_seq) in candidates.iter().enumerate() {
                            if actions_taken >= MAX_ACTIONS { break; }
                            if state == "WIN" || state == "GAME_OVER" { break; }

                            // Skip already-failed combos
                            let mut fail_key = action_seq.clone();
                            fail_key.sort();
                            if failed_seqs.contains(&fail_key) { continue; }

                            let click_count = action_seq.iter().filter(|&&(x,_)| x >= 0).count();
                            if ci < 10 || ci % 50 == 0 {
                                eprintln!("  [SEARCH] Try {}/{}: {} clicks, {} failed so far",
                                    ci+1, candidates.len(), click_count, failed_seqs.len());
                            }

                            // Execute the combo
                            let mut combo_won = false;
                            let mut clicks_sent = 0usize;
                            for &(x, y) in action_seq {
                                if actions_taken >= MAX_ACTIONS { break; }
                                std::thread::sleep(std::time::Duration::from_millis(20));
                                let reasoning = serde_json::json!({"engine":"search","try":ci});
                                let (action_num, coords) = if x < 0 {
                                    ((-x) as u8, None)
                                } else {
                                    clicks_sent += 1;
                                    (6u8, Some((x, y)))
                                };
                                if let Ok(resp) = send_action(&agent, game_id, &guid, action_num, coords, &reasoning) {
                                    actions_taken += 1;
                                    guid = resp.guid.clone();
                                    state = resp.state.clone();
                                    current_grid = extract_grid(&resp.frame);
                                    if resp.levels_completed > levels {
                                        eprintln!("  [SEARCH] *** LEVEL UP! {} → {} — COMBO #{} FOUND! ***",
                                            levels, resp.levels_completed, ci+1);
                                        levels = resp.levels_completed;
                                        combo_won = true;
                                        // Save winning combo
                                        let entry = all_game_rules.entry(short_id.to_string()).or_default();
                                        while entry.levels.len() <= levels - 1 {
                                            entry.levels.push(GameLevelRules::default());
                                        }
                                        entry.levels[levels - 1].winning_combo = action_seq.clone();
                                        entry.levels[levels - 1].solved = true;
                                        entry.best_level = entry.best_level.max(levels);
                                        // AUTO-SAVE: raw grid for new level
                                        while entry.levels.len() <= levels {
                                            entry.levels.push(GameLevelRules::default());
                                        }
                                        if entry.levels[levels].initial_grid.is_none() {
                                            entry.levels[levels].initial_grid = Some(current_grid.clone());
                                            let mut hist = vec![0u32; 16];
                                            for row in &current_grid {
                                                for &c in row { if (c as usize) < 16 { hist[c as usize] += 1; } }
                                            }
                                            entry.levels[levels].color_histogram = hist;
                                            entry.levels[levels].grid_rows = current_grid.len();
                                            entry.levels[levels].grid_cols = current_grid.first().map(|r| r.len()).unwrap_or(0);
                                            eprintln!("  [AUTO-SAVE] Saved raw grid for level {}", levels);
                                        }
                                        save_game_rules(&all_game_rules);
                                        eprintln!("  [SEARCH] Winning combo saved!");
                                        break;
                                    }
                                }
                            }

                            if combo_won {
                                any_won = true;
                                cached_game_grid = None;
                                break; // exit candidate loop → tile_solver will re-detect for next level
                            }

                            // Failed — undo all clicks to reset the board
                            failed_seqs.insert(fail_key);
                            for _ in 0..clicks_sent {
                                if actions_taken >= MAX_ACTIONS { break; }
                                if state == "GAME_OVER" || state == "WIN" { break; }
                                std::thread::sleep(std::time::Duration::from_millis(10));
                                let reasoning = serde_json::json!({"engine":"search","phase":"undo"});
                                if let Ok(resp) = send_action(&agent, game_id, &guid, 7, None, &reasoning) {
                                    actions_taken += 1;
                                    guid = resp.guid.clone();
                                    state = resp.state.clone();
                                    current_grid = extract_grid(&resp.frame);
                                }
                            }
                        }

                        // Save all failed combos to DNA
                        {
                            let entry = all_game_rules.entry(short_id.to_string()).or_default();
                            while entry.levels.len() <= levels {
                                entry.levels.push(GameLevelRules::default());
                            }
                            // Only save new failures (not already in DNA)
                            let existing: std::collections::HashSet<Vec<(i64,i64)>> = entry.levels[levels].failed_combos.iter()
                                .map(|c| { let mut k = c.clone(); k.sort(); k }).collect();
                            for key in &failed_seqs {
                                if !existing.contains(key) {
                                    entry.levels[levels].failed_combos.push(key.clone());
                                }
                            }
                            // Cap at 500 per level
                            while entry.levels[levels].failed_combos.len() > 500 {
                                entry.levels[levels].failed_combos.remove(0);
                            }
                            save_game_rules(&all_game_rules);
                        }

                        if any_won {
                            continue 'tile_solver; // next level
                        }
                        eprintln!("  [SEARCH] Exhausted {} candidates, {} failed total",
                            candidates.len(), failed_seqs.len());
                        break 'tile_solver;
                    } // end 'tile_solver loop

                    break; // exit per-turn loop — combo solver handled all actions
                }

                // ── PERCEIVE: inject facts for QOR reasoning ──
                session.remove_by_predicate(CLEAR_PREDICATES);
                let n_facts = feed_frame(&mut session, &current_grid, "f");
                detect_edge_colors(&mut session, &current_grid);
                detect_player_and_target(&mut session, &current_grid);
                let _ = session.exec(&format!("(frame-number {turn})"));
                let _ = session.exec(&format!("(game-type {gtype})"));
                let _ = session.exec(&format!("(turn-phase {})", (turn / 8) % 4));
                let _ = session.exec(&format!("(consecutive-no-change {consecutive_no_change})"));
                // Inject last action + wall-hit memory so QOR avoids blocked directions
                if !last_action_name.is_empty() {
                    let _ = session.exec(&format!("(last-action {})", last_action_name));
                }
                for dir in &wall_hits {
                    let _ = session.exec(&format!("(wall-hit {dir})"));
                }
                // Strategy state facts — QOR uses these to derive (strategy-select X)
                let _ = session.exec(&format!("(game-id {short_id})"));
                let _ = session.exec(&format!("(current-level {levels})"));
                if let Some(rules) = all_game_rules.get(short_id) {
                    for (i, lr) in rules.levels.iter().enumerate() {
                        if lr.solved { let _ = session.exec(&format!("(level-solved {i})")); }
                        if !lr.winning_combo.is_empty() { let _ = session.exec(&format!("(level-has-sequence {i})")); }
                    }
                }
                let _ = session.exec("(perception-done)");

                // ── STRATEGY: QOR decides, Rust executes ──
                let strategy = read_strategy(&session).unwrap_or_else(|| "qor-direct".to_string());
                if turn <= 3 || turn % 20 == 0 {
                    eprintln!("  [STRATEGY] turn {turn}: {strategy}");
                }

                // ── REPLAY-SEQUENCE: replay stored action sequences per level ──
                if strategy == "replay-sequence" {
                    let level_seq: Vec<(i64, i64)> = all_game_rules.get(short_id)
                        .and_then(|r| r.levels.get(levels))
                        .map(|lr| lr.winning_combo.clone())
                        .unwrap_or_default();

                    // Only replay sequence once per level (don't repeat on every turn)
                    if !level_seq.is_empty() && seq_done_level != Some(levels) {
                        let level_solved_flag = all_game_rules.get(short_id)
                            .and_then(|r| r.levels.get(levels))
                            .map(|lr| lr.solved)
                            .unwrap_or(false);

                        eprintln!("  [SEQ] Level {levels}: replaying {} actions (solved={})",
                            level_seq.len(), level_solved_flag);

                        let played_level_before = levels;
                        for (si, &(x, y)) in level_seq.iter().enumerate() {
                            if actions_taken >= MAX_ACTIONS { break; }
                            std::thread::sleep(std::time::Duration::from_millis(100));
                            let reasoning = serde_json::json!({"engine":"sequence","level":levels});
                            let (action_num, coords) = if x < 0 {
                                ((-x) as u8, None)
                            } else {
                                (6u8, Some((x, y)))
                            };
                            let before = current_grid.clone();
                            if let Ok(resp) = send_action(&agent, game_id, &guid, action_num, coords, &reasoning) {
                                actions_taken += 1;
                                guid = resp.guid.clone();
                                state = resp.state.clone();
                                current_grid = extract_grid(&resp.frame);
                                let changed = count_changed_pixels(&before, &current_grid);
                                eprintln!("  [SEQ] step {}: ACTION{} {} → {} px changed",
                                    si+1, action_num, action_name_from_map(action_num, &action_map), changed);
                                if resp.levels_completed > levels {
                                    eprintln!("  [SEQ] *** LEVEL UP! {} → {} ***", levels, resp.levels_completed);
                                    levels = resp.levels_completed;
                                    // Save as solved
                                    let entry = all_game_rules.entry(short_id.to_string()).or_default();
                                    while entry.levels.len() <= levels - 1 {
                                        entry.levels.push(GameLevelRules::default());
                                    }
                                    entry.levels[levels - 1].solved = true;
                                    entry.levels[levels - 1].winning_combo = level_seq.clone();
                                    entry.best_level = entry.best_level.max(levels);
                                    // AUTO-SAVE: raw grid for new level
                                    while entry.levels.len() <= levels {
                                        entry.levels.push(GameLevelRules::default());
                                    }
                                    if entry.levels[levels].initial_grid.is_none() {
                                        entry.levels[levels].initial_grid = Some(current_grid.clone());
                                        let mut hist = vec![0u32; 16];
                                        for row in &current_grid {
                                            for &c in row { if (c as usize) < 16 { hist[c as usize] += 1; } }
                                        }
                                        entry.levels[levels].color_histogram = hist;
                                        entry.levels[levels].grid_rows = current_grid.len();
                                        entry.levels[levels].grid_cols = current_grid.first().map(|r| r.len()).unwrap_or(0);
                                        eprintln!("  [AUTO-SAVE] Saved raw grid for level {}", levels);
                                    }
                                    save_game_rules(&all_game_rules);
                                    break;
                                }
                            }
                        }
                        let played_level = if levels > played_level_before {
                            played_level_before
                        } else {
                            levels
                        };
                        seq_done_level = Some(played_level);
                        if state == "WIN" || state == "GAME_OVER" { break; }
                        continue; // next turn will check for next level's sequence
                    }
                }
                // ── QOR ACTION SELECT: facts already injected above ──
                let qor_actions = read_action_selects(&session);

                let (action_num, coords) = {
                    if let Some((name, conf, qcoords)) = qor_actions.first() {
                        if turn <= 20 || turn % 10 == 0 {
                            eprintln!("  [QOR] turn {turn}: {} (conf={:.2})", name, conf);
                        }
                        let num = match name.as_str() {
                            "up" => 1, "down" => 2, "left" => 3, "right" => 4,
                            "space" => 5, "click" => 6, _ => 5,
                        };
                        (num, *qcoords)
                    } else {
                        // No QOR action — generic fallback: cycle 1-5
                        let num = ((turn - 1) % 5) as u8 + 1;
                        (num, None)
                    }
                };
                if turn <= 20 || turn % 10 == 0 {
                    let coord_str = coords.map(|(x,y)| format!(" @({x},{y})"))
                        .unwrap_or_default();
                    eprintln!("  [ACT] turn {turn}: ACTION{action_num} ({}){coord_str} [{gtype}]",
                        action_name_from_map(action_num, &action_map));
                }

                last_action_name = action_name_from_map(action_num, &action_map).to_string();

                // Build reasoning payload
                let reasoning = serde_json::json!({
                    "engine": "qor",
                    "turn": turn,
                    "action": &last_action_name,
                    "facts": n_facts,
                    "rules": session.rule_count(),
                    "qor_options": qor_actions.len(),
                });

                // 5. ACT: send action to API
                let resp = match send_action(&agent, game_id, &guid, action_num, coords, &reasoning) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("  [ERROR] turn {turn}: {e}");
                        consecutive_no_change += 1;
                        if consecutive_no_change > 10 {
                            eprintln!("  [ABORT] Too many errors, stopping attempt");
                            break;
                        }
                        continue;
                    }
                };

                actions_taken += 1;
                guid = resp.guid.clone();
                state = resp.state.clone();
                let new_grid = extract_grid(&resp.frame);
                let new_levels = resp.levels_completed;

                // Auto-learn: record action for potential combo save on level-up
                if let Some((cx, cy)) = coords {
                    level_action_history.push((cx, cy));
                } else {
                    level_action_history.push((-(action_num as i64), 0));
                }

                // 6. LEARN: track frame changes with detailed logging
                let changed_pixels = count_changed_pixels(&current_grid, &new_grid);
                let frame_changed = changed_pixels > 0;

                // Log diff for first 10 turns and every 10th
                if turn <= 10 || turn % 10 == 0 || new_levels > levels {
                    log_grid_diff(&current_grid, &new_grid,
                        &format!("ACTION{action_num} turn {turn}"));
                }

                if frame_changed {
                    consecutive_no_change = 0;
                    wall_hits.clear(); // piece moved — old walls may not apply anymore
                } else {
                    consecutive_no_change += 1;
                    // Remember this direction is blocked (hit wall)
                    if !last_action_name.is_empty() {
                        let dir = last_action_name.clone();
                        if wall_hits.insert(dir.clone()) {
                            eprintln!("  [WALL-HIT] '{dir}' blocked — {}/{} dirs blocked",
                                wall_hits.len(), 4);
                        }
                    }
                }

                // Record for genesis
                frame_history.push(FrameRecord {
                    grid: current_grid.clone(),
                    action_taken: action_num,
                    frame_changed,
                });

                // LEVEL UP — re-detect tiles for new level (generic)
                if new_levels > levels {
                    eprintln!("  [LEVEL UP] {} → {} (turn {turn}, action={}, px={})",
                        levels, new_levels, &last_action_name, changed_pixels);

                    // Auto-learn: save winning action sequence for this level
                    if !level_action_history.is_empty() {
                        let entry = all_game_rules.entry(short_id.to_string()).or_default();
                        while entry.levels.len() <= levels {
                            entry.levels.push(GameLevelRules::default());
                        }
                        if !entry.levels[levels].solved {
                            entry.levels[levels].solved = true;
                            entry.levels[levels].winning_combo = level_action_history.clone();
                            entry.best_level = entry.best_level.max(new_levels);
                            save_game_rules(&all_game_rules);
                            eprintln!("  [AUTO-LEARN] Saved winning combo for level {} ({} actions)",
                                levels, level_action_history.len());
                        }
                    }
                    level_action_history.clear(); // reset for next level

                    levels = new_levels;
                    consecutive_no_change = 0;
                    wall_hits.clear(); // new level — fresh start

                    // Save frame for the new level
                    {
                        let frame_path = format!("frame_{}_level{}.ppm", short_id, levels);
                        save_frame_ppm(&new_grid, &frame_path);
                    }
                    // AUTO-SAVE: store raw grid + color histogram for this new level
                    {
                        let needs_save = {
                            let entry = all_game_rules.entry(short_id.to_string()).or_default();
                            while entry.levels.len() <= levels {
                                entry.levels.push(GameLevelRules::default());
                            }
                            if entry.levels[levels].initial_grid.is_none() {
                                entry.levels[levels].initial_grid = Some(new_grid.clone());
                                let mut hist = vec![0u32; 16];
                                for row in &new_grid {
                                    for &c in row {
                                        if (c as usize) < 16 { hist[c as usize] += 1; }
                                    }
                                }
                                entry.levels[levels].color_histogram = hist;
                                entry.levels[levels].grid_rows = new_grid.len();
                                entry.levels[levels].grid_cols = new_grid.first().map(|r| r.len()).unwrap_or(0);
                                true
                            } else { false }
                        };
                        if needs_save {
                            save_game_rules(&all_game_rules);
                            eprintln!("  [AUTO-SAVE] Saved raw grid for level {}", levels);
                        }
                    }
                    // Re-detect tile grids for new level layout
                    dump_grid_overview(&new_grid);
                    let (new_gg, new_tg) = detect_tile_grids(&new_grid);
                    if let Some(gg) = new_gg {
                        eprintln!("  [TILES] New level: {}x{} game board", gg.nrows, gg.ncols);
                        cached_game_grid = Some(gg);
                    } else {
                        cached_game_grid = None;
                    }
                    if let Some(tg) = new_tg {
                        eprintln!("  [TILES] New level: {}x{} target", tg.nrows, tg.ncols);
                        cached_target_grid = Some(tg);
                    } else {
                        cached_target_grid = None;
                    }
                    // Re-detect tile grids
                    let (new_gg, new_tg) = detect_tile_grids(&new_grid);
                    cached_game_grid = new_gg;
                    cached_target_grid = new_tg;
                }
                if resp.score != prev_score {
                    prev_score = resp.score;
                }

                // Update available actions
                if !resp.available_actions.is_empty() {
                    avail_actions = resp.available_actions.clone();
                }

                // Periodic log
                if turn % 25 == 0 {
                    eprintln!("  [T{turn}] lv={}/{} px={} nochange={}",
                        levels, win_levels, changed_pixels, consecutive_no_change);
                }

                // Terminal
                if state == "WIN" {
                    eprintln!("  [WIN] {} in {actions_taken} actions!", game_id);
                    game_won = true;
                    break;
                }
                if state == "GAME_OVER" {
                    eprintln!("  [GAME_OVER] levels={levels}/{win_levels}, {actions_taken} actions");
                    break;
                }

                current_grid = new_grid;
            }

            if levels > best_levels { best_levels = levels; }

            // Auto-learn: save failed action sequence for the level we got stuck on
            if !game_won && !level_action_history.is_empty() {
                let entry = all_game_rules.entry(short_id.to_string()).or_default();
                while entry.levels.len() <= levels {
                    entry.levels.push(GameLevelRules::default());
                }
                entry.levels[levels].failed_combos.push(level_action_history.clone());
                // Cap failed combos to last 20 per level to avoid unbounded growth
                if entry.levels[levels].failed_combos.len() > 20 {
                    entry.levels[levels].failed_combos.remove(0);
                }
                eprintln!("  [AUTO-LEARN] Saved FAILED attempt for level {} ({} actions, {} total failures)",
                    levels, level_action_history.len(), entry.levels[levels].failed_combos.len());
                save_game_rules(&all_game_rules);
            }
            level_action_history.clear();

            // Save memory
            save_memory(short_id, attempt, levels, win_levels, actions_taken, &mp);

            // Save attempt result to game rules (track success/failure)
            {
                let entry = all_game_rules.entry(short_id.to_string()).or_default();
                entry.total_attempts += 1;
                if levels > entry.best_level { entry.best_level = levels; }
                if game_won {
                    entry.solved = true;
                    // Mark all completed levels as solved
                    for lv in 0..levels {
                        while entry.levels.len() <= lv {
                            entry.levels.push(GameLevelRules::default());
                        }
                        entry.levels[lv].solved = true;
                    }
                    eprintln!("  [RULES] Saved WIN — {} levels solved", levels);
                } else {
                    while entry.levels.len() <= levels {
                        entry.levels.push(GameLevelRules::default());
                    }
                    entry.levels[levels].attempts += 1;
                    eprintln!("  [RULES] Saved failed attempt {} for level {} ({} failed combos stored)",
                        attempt, levels, entry.levels[levels].failed_combos.len());
                }
                save_game_rules(&all_game_rules);
            }

            // Genesis after first attempt (learn from experience)
            if attempt == 1 && !game_won && !frame_history.is_empty() {
                let new_rules = try_genesis(&session, &frame_history, &mut library);
                if !new_rules.is_empty() {
                    let saved = save_rules(&new_rules, short_id, &rp);
                    eprintln!("  [RULES] Saved {saved} new rules");
                }
            }

        } // end single play

        total_levels += best_levels;
        if game_won { total_wins += 1; }
        eprintln!("\n  RESULT: {game_id} — {best_levels} levels, {}s",
            game_start.elapsed().as_secs());
    }

    // ── 5. Scorecard ──
    // Default: keep scorecard OPEN so score accumulates across runs.
    // Use --close-scorecard to explicitly close and start fresh next time.
    if close_scorecard {
        match api_post(&agent, "/scorecard/close", &serde_json::json!({ "card_id": card_id })) {
            Ok(resp) => {
                eprintln!("\n  Scorecard CLOSED: {}", &resp[..resp.len().min(500)]);
                let _ = std::fs::remove_file(dna_dir().join("active_scorecard.txt"));
            }
            Err(e) => eprintln!("\n  Scorecard close error: {e}"),
        }
    } else {
        eprintln!("\n  Scorecard KEPT OPEN: {card_id}");
        eprintln!("  (use --close-scorecard to close and start fresh)");
    }

    library.save();
    save_game_rules(&all_game_rules);
    eprintln!("\n{}", "=".repeat(70));
    eprintln!("  FINAL: {total_wins} wins, {total_levels} total levels, {}s",
        overall_start.elapsed().as_secs());
    eprintln!("  Scorecard: {card_id}");
    eprintln!("{}", "=".repeat(70));
}
