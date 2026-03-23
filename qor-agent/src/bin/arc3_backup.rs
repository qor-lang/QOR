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
use std::time::Instant;

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

const API_BASE: &str = "https://three.arcprize.org/api";
const API_KEY: &str = "5d0731de-5ead-41b3-94d4-cf31a6579da4";
const MAX_ACTIONS: usize = 400;
const MAX_RESETS: usize = 50;
const GENESIS_BUDGET_MS: u64 = 10_000;

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
        .set("X-API-Key", API_KEY)
        .set("Accept", "application/json")
        .call()?;
    Ok(resp.into_string()?)
}

fn api_post(agent: &ureq::Agent, path: &str, body: &serde_json::Value) -> Result<String, Box<dyn std::error::Error>> {
    let url = format!("{API_BASE}{path}");
    let body_str = body.to_string();
    for retry in 0..3 {
        let result = agent.post(&url)
            .set("X-API-Key", API_KEY)
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

// ── Game-type detection ────────────────────────────────────────────────

fn game_type(game_id: &str) -> &'static str {
    let short = game_id.split('-').next().unwrap_or("");
    if short.starts_with("ft") { return "tile-puzzle"; }
    if short.starts_with("ls") { return "room-explore"; }
    if short.starts_with("vc") { return "room-explore"; }
    if short.starts_with("as") { return "gravity"; }
    if short.starts_with("lp") { return "board-game"; }
    if short.starts_with("sp") { return "growth"; }
    "unknown"
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

const CLEAR_PREDICATES: &[&str] = &[
    "grid-cell", "grid-size", "grid-object", "grid-obj-cell", "grid-obj-bbox",
    "color-cell-count", "num-colors", "num-objects",
    "cell-changed", "frame-changes", "frame-changed",
    "action-select", "frame-number", "game-state", "game-type",
    "levels-completed", "win-levels", "consecutive-no-change",
    "last-action", "game-id", "attempt-number", "turn-phase",
    "perception-done",
    "tile", "tile-solid", "tile-mixed", "tile-colors", "tile-needs-click",
    "game-tile", "target-tile", "has-target",
    "game-board-rows", "game-board-cols",
    "pair-delta", "pair-size", "pair-color-diff", "pair-cell-change",
    "grid-neighbor", "enclosed-cell", "enclosed-region", "enclosed-region-cell",
    "shape-sig", "input-period", "effective-type",
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
    let game_filter: Option<String> = args.iter()
        .position(|a| a == "--game")
        .and_then(|i| args.get(i + 1))
        .cloned();

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
    let sc_file = dna_dir().join("active_scorecard.txt");
    let card_id = if sc_file.exists() {
        let saved = std::fs::read_to_string(&sc_file).unwrap_or_default().trim().to_string();
        if !saved.is_empty() {
            eprintln!("  Reusing scorecard: {saved}");
            saved
        } else {
            let sc_str = api_post(&agent, "/scorecard/open", &serde_json::json!({
                "tags": ["qor-agent", "arc3", "agent"],
                "source_url": "https://github.com/qor-lang/QOR",
                "ai": true,
            })).expect("Failed to create scorecard");
            let scorecard: ScorecardResponse = serde_json::from_str(&sc_str)
                .expect("Cannot parse scorecard");
            std::fs::write(&sc_file, &scorecard.card_id).ok();
            eprintln!("  New scorecard: {}", scorecard.card_id);
            scorecard.card_id
        }
    } else {
        let sc_str = api_post(&agent, "/scorecard/open", &serde_json::json!({
            "tags": ["qor-agent", "arc3", "agent"],
            "source_url": "https://github.com/qor-lang/QOR",
            "ai": true,
        })).expect("Failed to create scorecard");
        let scorecard: ScorecardResponse = serde_json::from_str(&sc_str)
            .expect("Cannot parse scorecard");
        std::fs::write(&sc_file, &scorecard.card_id).ok();
        eprintln!("  New scorecard: {}", scorecard.card_id);
        scorecard.card_id
    };

    // Ctrl+C handler — set flag so we close scorecard gracefully
    let _ = ctrlc::set_handler(move || {
        eprintln!("\n  [INTERRUPTED] Ctrl+C — will close scorecard and exit");
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

    for game_id in &filtered {
        let short_id = game_id.split('-').next().unwrap_or(game_id);
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
            // Detect game type from game ID prefix
            let gtype = game_type(game_id);

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
                        // Only pick extras from RED tiles (color 8) — skip icons (0) and preset (2)
                        let red_set: std::collections::HashSet<(i64, i64)> = tile_colors.iter()
                            .filter(|&&(c, _, _)| c == 8)
                            .map(|&(_, x, y)| (x, y))
                            .collect();
                        let unclicked: Vec<(i64, i64)> = colored_tiles.iter()
                            .filter(|t| !base_set.contains(t))
                            .filter(|t| red_set.contains(t))
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

                        // NO SPACE — tiles start RED, clicks paint ORANGE

                        // Helper: generate all k-combinations from a slice
                        fn combos_of(items: &[(i64,i64)], k: usize) -> Vec<Vec<(i64,i64)>> {
                            let mut result = Vec::new();
                            if k == 0 || k > items.len() { return result; }
                            fn gen(items: &[(i64,i64)], k: usize, start: usize, cur: &mut Vec<(i64,i64)>, out: &mut Vec<Vec<(i64,i64)>>) {
                                if cur.len() == k { out.push(cur.clone()); return; }
                                for i in start..items.len() {
                                    cur.push(items[i]);
                                    gen(items, k, i+1, cur, out);
                                    cur.pop();
                                }
                            }
                            gen(items, k, 0, &mut Vec::new(), &mut result);
                            result
                        }

                        // Priority 1: base + need_extra from unclicked = 14 clicks
                        if need_extra <= unclicked.len() {
                            for extra in combos_of(&unclicked, need_extra) {
                                let mut seq = base_clicks.clone();
                                seq.extend_from_slice(&extra);
                                candidates.push(seq);
                            }
                        }

                        // Priority 2: swap 1 out, add (need_extra+1) = target clicks
                        if need_extra + 1 <= unclicked.len() {
                            for (ri, _) in base_clicks.iter().enumerate() {
                                for extra in combos_of(&unclicked, need_extra + 1) {
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
                        if need_extra + 2 <= unclicked.len() && base_clicks.len() >= 2 {
                            for remove in combos_of(&base_clicks, 2) {
                                let remove_set: std::collections::HashSet<(i64,i64)> = remove.iter().copied().collect();
                                let kept: Vec<(i64,i64)> = base_clicks.iter().filter(|t| !remove_set.contains(t)).copied().collect();
                                for extra in combos_of(&unclicked, need_extra + 2) {
                                    let mut seq = kept.clone();
                                    seq.extend_from_slice(&extra);
                                    candidates.push(seq);
                                }
                            }
                        }

                        // Priority 4: swap 3 out, add (need_extra+3) = target clicks
                        if need_extra + 3 <= unclicked.len() && base_clicks.len() >= 3 {
                            for remove in combos_of(&base_clicks, 3) {
                                let remove_set: std::collections::HashSet<(i64,i64)> = remove.iter().copied().collect();
                                let kept: Vec<(i64,i64)> = base_clicks.iter().filter(|t| !remove_set.contains(t)).copied().collect();
                                for extra in combos_of(&unclicked, need_extra + 3) {
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

                        // Execute the combo
                        let mut combo_won = false;
                        for &(x, y) in &action_seq {
                            if actions_taken >= MAX_ACTIONS { break; }
                            std::thread::sleep(std::time::Duration::from_millis(80));
                            let reasoning = serde_json::json!({"engine":"search","phase":"try"});
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
                                    eprintln!("  [SEARCH] *** LEVEL UP! {} → {} — COMBO FOUND! ***", levels, resp.levels_completed);
                                    levels = resp.levels_completed;
                                    combo_won = true;
                                    // Save winning combo to DNA
                                    let entry = all_game_rules.entry(short_id.to_string()).or_default();
                                    while entry.levels.len() <= levels - 1 {
                                        entry.levels.push(GameLevelRules::default());
                                    }
                                    entry.levels[levels - 1].winning_combo = action_seq.clone();
                                    entry.levels[levels - 1].solved = true;
                                    entry.best_level = entry.best_level.max(levels);
                                    save_game_rules(&all_game_rules);
                                    eprintln!("  [SEARCH] Winning combo saved to DNA!");
                                    break;
                                }
                            }
                        }
                        if combo_won {
                            cached_game_grid = None;
                            continue 'tile_solver;
                        }

                        // Failed — save full action sequence (with coordinates) so dedup works
                        let mut fail_key = action_seq.clone();
                        fail_key.sort();
                        failed_seqs.insert(fail_key);

                        let entry = all_game_rules.entry(short_id.to_string()).or_default();
                        while entry.levels.len() <= levels {
                            entry.levels.push(GameLevelRules::default());
                        }
                        entry.levels[levels].failed_combos.push(action_seq.clone());
                        save_game_rules(&all_game_rules);

                        eprintln!("  [SEARCH] FAILED — saved (total failed: {})", failed_seqs.len());
                        break 'tile_solver; // next attempt starts fresh
                    } // end 'tile_solver loop

                    break; // exit per-turn loop — combo solver handled all actions
                }

                // ── ACTION-SEQUENCE GAMES: replay known sequences per level ──
                if gtype == "room-explore" {
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
                        let pre_seq_grid = current_grid.clone();
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
                                    si+1, action_num, action_name(action_num), changed);
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
                                    save_game_rules(&all_game_rules);
                                    break;
                                }
                            }
                        }
                        // Mark this level's sequence as done (use pre-replay level)
                        // If level changed (LEVEL UP), seq_done_level won't match new level
                        // so the next level's sequence will play correctly
                        let played_level = if levels > played_level_before {
                            played_level_before // level changed — don't block next level
                        } else {
                            levels // level didn't change — block re-replay
                        };
                        seq_done_level = Some(played_level);
                        // After sequence replay, skip to next turn
                        if state == "WIN" || state == "GAME_OVER" { break; }
                        continue; // next turn will check for next level's sequence
                    }
                }

                // ── NON-TILE GAMES: QOR-driven action selection ──
                session.remove_by_predicate(CLEAR_PREDICATES);
                let n_facts = feed_frame(&mut session, &current_grid, "f");
                let _ = session.exec(&format!("(frame-number {turn})"));
                let _ = session.exec(&format!("(game-type {gtype})"));
                let _ = session.exec(&format!("(turn-phase {})", (turn / 8) % 4));
                let _ = session.exec(&format!("(consecutive-no-change {consecutive_no_change})"));
                let _ = session.exec("(perception-done)");
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
                        action_name(action_num));
                }

                last_action_name = action_name(action_num).to_string();

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
                } else {
                    consecutive_no_change += 1;
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
                    levels = new_levels;
                    consecutive_no_change = 0;

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
                    // Failed attempt — clicks already saved per-click, just update attempt count
                    while entry.levels.len() <= levels {
                        entry.levels.push(GameLevelRules::default());
                    }
                    entry.levels[levels].attempts += 1;
                    eprintln!("  [RULES] Saved failed attempt {} for level {} ({} clicks recorded)",
                        attempt, levels, entry.levels[levels].click_history.len());
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

    // ── 5. Close scorecard ──
    match api_post(&agent, "/scorecard/close", &serde_json::json!({ "card_id": card_id })) {
        Ok(resp) => {
            eprintln!("\n  Scorecard closed: {}", &resp[..resp.len().min(500)]);
            // Remove saved scorecard so next run creates fresh one
            let _ = std::fs::remove_file(dna_dir().join("active_scorecard.txt"));
        }
        Err(e) => eprintln!("\n  Scorecard close error: {e}"),
    }

    library.save();
    eprintln!("\n{}", "=".repeat(70));
    eprintln!("  FINAL: {total_wins} wins, {total_levels} total levels, {}s",
        overall_start.elapsed().as_secs());
    eprintln!("{}", "=".repeat(70));
}
