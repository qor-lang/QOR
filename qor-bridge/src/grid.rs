// ── QOR Grid Perception ──────────────────────────────────────────────
//
// Ingests 2D grids and converts them to QOR facts.
// Uses flood-fill (BFS) to detect connected components (objects).
//
// Grid facts emitted:
//   (grid-size id rows cols)              — dimensions
//   (grid-cell id row col color)          — individual cell
//   (grid-object id obj_id color size)    — detected object
//   (grid-obj-cell id obj_id row col)     — cells belonging to object
//   (grid-obj-bbox id obj_id min_r min_c max_r max_c) — bounding box

use std::collections::{HashMap, HashSet, VecDeque};

use ndarray::Array2;
use qor_core::neuron::{Neuron, QorValue, Statement};
use qor_core::truth_value::TruthValue;

/// A 2D grid of colored cells (0 = background).
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    pub rows: usize,
    pub cols: usize,
    pub cells: Vec<Vec<u8>>,
}

/// A connected component detected by flood fill.
#[derive(Debug, Clone, PartialEq)]
pub struct GridObject {
    pub id: usize,
    pub color: u8,
    pub cells: Vec<(usize, usize)>, // (row, col)
}

impl GridObject {
    /// Bounding box: (min_row, min_col, max_row, max_col)
    pub fn bbox(&self) -> (usize, usize, usize, usize) {
        let min_r = self.cells.iter().map(|(r, _)| *r).min().unwrap_or(0);
        let min_c = self.cells.iter().map(|(_, c)| *c).min().unwrap_or(0);
        let max_r = self.cells.iter().map(|(r, _)| *r).max().unwrap_or(0);
        let max_c = self.cells.iter().map(|(_, c)| *c).max().unwrap_or(0);
        (min_r, min_c, max_r, max_c)
    }
}

impl Grid {
    /// Create a grid from nested vectors. Returns error if rows are ragged.
    pub fn from_vecs(cells: Vec<Vec<u8>>) -> Result<Grid, String> {
        if cells.is_empty() {
            return Ok(Grid { rows: 0, cols: 0, cells: vec![] });
        }
        let cols = cells[0].len();
        for (i, row) in cells.iter().enumerate() {
            if row.len() != cols {
                return Err(format!(
                    "ragged grid: row 0 has {} cols but row {} has {}",
                    cols, i, row.len()
                ));
            }
        }
        let rows = cells.len();
        Ok(Grid { rows, cols, cells })
    }

    /// Get cell value at (row, col).
    pub fn cell_at(&self, row: usize, col: usize) -> Option<u8> {
        self.cells.get(row).and_then(|r| r.get(col)).copied()
    }

    /// Grid dimensions as (rows, cols).
    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Convert grid cells to ndarray Array2 for efficient transforms.
    pub fn to_array2(&self) -> Array2<u8> {
        let mut arr = Array2::zeros((self.rows, self.cols));
        for r in 0..self.rows {
            for c in 0..self.cols {
                arr[[r, c]] = self.cells[r][c];
            }
        }
        arr
    }

    /// Create Grid from ndarray Array2.
    pub fn from_array2(arr: &Array2<u8>) -> Grid {
        let (rows, cols) = arr.dim();
        let cells: Vec<Vec<u8>> = (0..rows)
            .map(|r| arr.row(r).to_vec())
            .collect();
        Grid { rows, cols, cells }
    }

    /// Detect connected components via BFS flood fill.
    /// Color 0 = background (skipped). 4-connected neighbors.
    pub fn objects(&self) -> Vec<GridObject> {
        let mut visited = vec![vec![false; self.cols]; self.rows];
        let mut objects = Vec::new();
        let mut obj_id = 0;

        for r in 0..self.rows {
            for c in 0..self.cols {
                if visited[r][c] || self.cells[r][c] == 0 {
                    continue;
                }
                let color = self.cells[r][c];
                let mut component = Vec::new();
                let mut queue = VecDeque::new();
                queue.push_back((r, c));
                visited[r][c] = true;

                while let Some((cr, cc)) = queue.pop_front() {
                    component.push((cr, cc));
                    // 4-connected neighbors
                    let neighbors: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                    for (dr, dc) in &neighbors {
                        let nr = cr as isize + dr;
                        let nc = cc as isize + dc;
                        if nr >= 0 && nr < self.rows as isize && nc >= 0 && nc < self.cols as isize {
                            let nr = nr as usize;
                            let nc = nc as usize;
                            if !visited[nr][nc] && self.cells[nr][nc] == color {
                                visited[nr][nc] = true;
                                queue.push_back((nr, nc));
                            }
                        }
                    }
                }

                objects.push(GridObject {
                    id: obj_id,
                    color,
                    cells: component,
                });
                obj_id += 1;
            }
        }

        objects
    }

    /// Convert grid to QOR statements.
    pub fn to_statements(&self, grid_id: &str) -> Vec<Statement> {
        let mut stmts = Vec::new();
        let tv = TruthValue::new(0.99, 0.99);
        let id = Neuron::symbol(grid_id);

        // (grid-size id rows cols)
        stmts.push(Statement::Fact {
            neuron: Neuron::expression(vec![
                Neuron::symbol("grid-size"),
                id.clone(),
                Neuron::int_val(self.rows as i64),
                Neuron::int_val(self.cols as i64),
            ]),
            tv: Some(tv),
            decay: None,
        });

        // (input-period id period) — minimum row-repeat period for periodic tiling
        let period = (1..=self.rows).find(|&p| {
            (0..self.rows).all(|r| {
                let ref_r = r % p;
                (0..self.cols).all(|c| self.cells[r][c] == self.cells[ref_r][c])
            })
        }).unwrap_or(self.rows);
        stmts.push(Statement::Fact {
            neuron: Neuron::expression(vec![
                Neuron::symbol("input-period"),
                id.clone(),
                Neuron::int_val(period as i64),
            ]),
            tv: Some(tv),
            decay: None,
        });

        // (grid-cell id row col color) for ALL cells (including background/zero)
        for r in 0..self.rows {
            for c in 0..self.cols {
                let color = self.cells[r][c];
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("grid-cell"),
                        id.clone(),
                        Neuron::int_val(r as i64),
                        Neuron::int_val(c as i64),
                        Neuron::int_val(color as i64),
                    ]),
                    tv: Some(tv),
                    decay: None,
                });
            }
        }

        // (grid-neighbor id r c r2 c2) for ALL cells (including background)
        // Emitted for all cells so QOR rules can reason about background adjacency.
        for r in 0..self.rows {
            for c in 0..self.cols {
                let neighbors: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                for (dr, dc) in &neighbors {
                    let nr = r as i64 + dr;
                    let nc = c as i64 + dc;
                    if nr >= 0 && nr < self.rows as i64 && nc >= 0 && nc < self.cols as i64 {
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("grid-neighbor"),
                                id.clone(),
                                Neuron::int_val(r as i64),
                                Neuron::int_val(c as i64),
                                Neuron::int_val(nr),
                                Neuron::int_val(nc),
                            ]),
                            tv: Some(tv),
                            decay: None,
                        });
                    }
                }
            }
        }

        // ── Enclosed cells: zero cells NOT reachable from grid border ──
        // BFS from border zeros; anything zero NOT reached is "enclosed".
        // These facts enable QOR rules for enclosed-region fill predictions.
        if self.rows >= 3 && self.cols >= 3 {
            let reachable = self.border_reachable_zeros();
            for r in 0..self.rows {
                for c in 0..self.cols {
                    if self.cells[r][c] == 0 && !reachable[r][c] {
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("enclosed-cell"),
                                id.clone(),
                                Neuron::int_val(r as i64),
                                Neuron::int_val(c as i64),
                            ]),
                            tv: Some(tv),
                            decay: None,
                        });
                    }
                }
            }

            // ── Enclosed regions: connected components among enclosed cells ──
            // Groups enclosed cells into individual regions, emitting per-region
            // area and membership facts for size-dependent fill rules.
            let mut enc_visited = vec![vec![false; self.cols]; self.rows];
            let mut region_id = 0usize;
            for r in 0..self.rows {
                for c in 0..self.cols {
                    if self.cells[r][c] == 0 && !reachable[r][c] && !enc_visited[r][c] {
                        let mut region_cells = Vec::new();
                        let mut q = VecDeque::new();
                        q.push_back((r, c));
                        enc_visited[r][c] = true;
                        while let Some((cr, cc)) = q.pop_front() {
                            region_cells.push((cr, cc));
                            for (dr, dc) in [(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                                let nr = cr as i32 + dr;
                                let nc = cc as i32 + dc;
                                if nr >= 0 && nc >= 0 {
                                    let (nr, nc) = (nr as usize, nc as usize);
                                    if nr < self.rows && nc < self.cols
                                        && self.cells[nr][nc] == 0
                                        && !reachable[nr][nc]
                                        && !enc_visited[nr][nc]
                                    {
                                        enc_visited[nr][nc] = true;
                                        q.push_back((nr, nc));
                                    }
                                }
                            }
                        }
                        // (enclosed-region id region_id area)
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("enclosed-region"),
                                id.clone(),
                                Neuron::int_val(region_id as i64),
                                Neuron::int_val(region_cells.len() as i64),
                            ]),
                            tv: Some(tv),
                            decay: None,
                        });
                        // (enclosed-region-cell id region_id r c)
                        for &(cr, cc) in &region_cells {
                            stmts.push(Statement::Fact {
                                neuron: Neuron::expression(vec![
                                    Neuron::symbol("enclosed-region-cell"),
                                    id.clone(),
                                    Neuron::int_val(region_id as i64),
                                    Neuron::int_val(cr as i64),
                                    Neuron::int_val(cc as i64),
                                ]),
                                tv: Some(tv),
                                decay: None,
                            });
                        }
                        region_id += 1;
                    }
                }
            }
        }

        // ── Shape signatures for non-zero colors ──
        // For each color with a small footprint (<=30 cells, bbox<=5x5),
        // compute a bitmap signature within its bounding box. Shape keys may
        // be disconnected (e.g. inverted-V), so no connectivity check.
        {
            let mut color_cells: HashMap<u8, Vec<(usize, usize)>> = HashMap::new();
            for r in 0..self.rows {
                for c in 0..self.cols {
                    if self.cells[r][c] > 0 {
                        color_cells.entry(self.cells[r][c]).or_default().push((r, c));
                    }
                }
            }
            for (&color, cells) in &color_cells {
                if cells.len() > 30 { continue; }
                let cell_set: HashSet<(usize, usize)> = cells.iter().copied().collect();
                // Compute bounding box
                let min_r = cells.iter().map(|&(r, _)| r).min().unwrap();
                let min_c = cells.iter().map(|&(_, c)| c).min().unwrap();
                let max_r = cells.iter().map(|&(r, _)| r).max().unwrap();
                let max_c = cells.iter().map(|&(_, c)| c).max().unwrap();
                let h = max_r - min_r + 1;
                let w = max_c - min_c + 1;
                if h * w > 25 { continue; }
                // Bitmap signature: row-major scan of bbox
                let mut sig: i64 = 0;
                for r in 0..h {
                    for c in 0..w {
                        sig <<= 1;
                        if cell_set.contains(&(min_r + r, min_c + c)) {
                            sig |= 1;
                        }
                    }
                }
                // (shape-sig id color signature)
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("shape-sig"),
                        id.clone(),
                        Neuron::int_val(color as i64),
                        Neuron::int_val(sig),
                    ]),
                    tv: Some(tv),
                    decay: None,
                });
            }
        }

        // Objects — IDs are grid-scoped to prevent cross-grid accidental matching
        let objects = self.objects();
        let grid_id_str = grid_id;
        for obj in &objects {
            let obj_sym = Neuron::symbol(&format!("{}_obj{}", grid_id_str, obj.id));

            // (grid-object id obj_id color size)
            stmts.push(Statement::Fact {
                neuron: Neuron::expression(vec![
                    Neuron::symbol("grid-object"),
                    id.clone(),
                    obj_sym.clone(),
                    Neuron::int_val(obj.color as i64),
                    Neuron::int_val(obj.cells.len() as i64),
                ]),
                tv: Some(tv),
                decay: None,
            });

            // (grid-obj-cell id obj_id row col) for each cell
            for &(r, c) in &obj.cells {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("grid-obj-cell"),
                        id.clone(),
                        obj_sym.clone(),
                        Neuron::int_val(r as i64),
                        Neuron::int_val(c as i64),
                    ]),
                    tv: Some(tv),
                    decay: None,
                });
            }

            // (grid-obj-bbox id obj_id min_r min_c max_r max_c)
            let (min_r, min_c, max_r, max_c) = obj.bbox();
            stmts.push(Statement::Fact {
                neuron: Neuron::expression(vec![
                    Neuron::symbol("grid-obj-bbox"),
                    id.clone(),
                    obj_sym.clone(),
                    Neuron::int_val(min_r as i64),
                    Neuron::int_val(min_c as i64),
                    Neuron::int_val(max_r as i64),
                    Neuron::int_val(max_c as i64),
                ]),
                tv: Some(tv),
                decay: None,
            });

            // (bbox-width id obj_id width) and (bbox-height id obj_id height)
            let w = (max_c - min_c + 1) as i64;
            let h = (max_r - min_r + 1) as i64;
            stmts.push(Statement::Fact {
                neuron: Neuron::expression(vec![
                    Neuron::symbol("bbox-width"),
                    id.clone(),
                    obj_sym.clone(),
                    Neuron::int_val(w),
                ]),
                tv: Some(tv),
                decay: None,
            });
            stmts.push(Statement::Fact {
                neuron: Neuron::expression(vec![
                    Neuron::symbol("bbox-height"),
                    id.clone(),
                    obj_sym.clone(),
                    Neuron::int_val(h),
                ]),
                tv: Some(tv),
                decay: None,
            });
        }

        // ── Spatial relationships between objects (pre-computed for speed) ──
        // O(objects²) in Rust = microseconds. Replaces O(n²) QOR rule derivation.
        let tv_spatial = Some(TruthValue::new(0.90, 0.85));
        for i in 0..objects.len() {
            let (r1a, c1a, r2a, c2a) = objects[i].bbox();
            let sym_a = Neuron::symbol(&format!("{}_obj{}", grid_id_str, objects[i].id));

            for j in (i + 1)..objects.len() {
                let (r1b, c1b, r2b, c2b) = objects[j].bbox();
                let sym_b = Neuron::symbol(&format!("{}_obj{}", grid_id_str, objects[j].id));

                // (above g a b) — A's bottom row < B's top row
                if r2a < r1b {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("above"), id.clone(), sym_a.clone(), sym_b.clone(),
                        ]),
                        tv: tv_spatial, decay: None,
                    });
                } else if r2b < r1a {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("above"), id.clone(), sym_b.clone(), sym_a.clone(),
                        ]),
                        tv: tv_spatial, decay: None,
                    });
                }

                // (left-of g a b) — A's right col < B's left col
                if c2a < c1b {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("left-of"), id.clone(), sym_a.clone(), sym_b.clone(),
                        ]),
                        tv: tv_spatial, decay: None,
                    });
                } else if c2b < c1a {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("left-of"), id.clone(), sym_b.clone(), sym_a.clone(),
                        ]),
                        tv: tv_spatial, decay: None,
                    });
                }

                // (h-aligned g a b) — same row range
                if r1a == r1b && r2a == r2b {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("h-aligned"), id.clone(), sym_a.clone(), sym_b.clone(),
                        ]),
                        tv: tv_spatial, decay: None,
                    });
                }

                // (v-aligned g a b) — same col range
                if c1a == c1b && c2a == c2b {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("v-aligned"), id.clone(), sym_a.clone(), sym_b.clone(),
                        ]),
                        tv: tv_spatial, decay: None,
                    });
                }

                // (contains g outer inner) — inner bbox fully inside outer bbox
                if r1a < r1b && c1a < c1b && r2a > r2b && c2a > c2b {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("contains"), id.clone(), sym_a.clone(), sym_b.clone(),
                        ]),
                        tv: tv_spatial, decay: None,
                    });
                } else if r1b < r1a && c1b < c1a && r2b > r2a && c2b > c2a {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("contains"), id.clone(), sym_b.clone(), sym_a.clone(),
                        ]),
                        tv: tv_spatial, decay: None,
                    });
                }

                // (same-color g a b color) — two objects with same color
                if objects[i].color == objects[j].color {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("same-color"), id.clone(), sym_a.clone(), sym_b.clone(),
                            Neuron::int_val(objects[i].color as i64),
                        ]),
                        tv: tv_spatial, decay: None,
                    });
                }

                // (bbox-overlap g a b) — bounding boxes overlap
                if r1a < r2b && r1b < r2a && c1a < c2b && c1b < c2a {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("bbox-overlap"), id.clone(), sym_a.clone(), sym_b.clone(),
                        ]),
                        tv: tv_spatial, decay: None,
                    });
                }
            }
        }

        // ── Color distribution facts ──
        // (color-cell-count g color count) — replaces O(n²) repeated-color rule
        {
            let mut color_counts: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();
            for r in &self.cells {
                for &c in r {
                    if c > 0 {
                        *color_counts.entry(c).or_insert(0) += 1;
                    }
                }
            }
            for (&color, &count) in &color_counts {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("color-cell-count"),
                        id.clone(),
                        Neuron::int_val(color as i64),
                        Neuron::int_val(count as i64),
                    ]),
                    tv: Some(tv),
                    decay: None,
                });
            }
        }

        // (object-count id count)
        stmts.push(Statement::Fact {
            neuron: Neuron::expression(vec![
                Neuron::symbol("object-count"),
                id.clone(),
                Neuron::int_val(objects.len() as i64),
            ]),
            tv: Some(tv),
            decay: None,
        });

        // (color-count id count) + (has-color id color) — pre-computed for speed
        let mut colors = std::collections::HashSet::new();
        for r in &self.cells {
            for &c in r {
                if c > 0 {
                    colors.insert(c);
                }
            }
        }
        stmts.push(Statement::Fact {
            neuron: Neuron::expression(vec![
                Neuron::symbol("color-count"),
                id.clone(),
                Neuron::int_val(colors.len() as i64),
            ]),
            tv: Some(tv),
            decay: None,
        });
        for &color in &colors {
            stmts.push(Statement::Fact {
                neuron: Neuron::expression(vec![
                    Neuron::symbol("has-color"),
                    id.clone(),
                    Neuron::int_val(color as i64),
                ]),
                tv: Some(tv),
                decay: None,
            });
        }

        // (content-bbox id r1 c1 r2 c2) — bounding box of all non-zero cells
        if let Some((min_r, min_c, max_r, max_c)) = Grid::content_bbox(self) {
            stmts.push(Statement::Fact {
                neuron: Neuron::expression(vec![
                    Neuron::symbol("content-bbox"),
                    id.clone(),
                    Neuron::int_val(min_r as i64),
                    Neuron::int_val(min_c as i64),
                    Neuron::int_val(max_r as i64),
                    Neuron::int_val(max_c as i64),
                ]),
                tv: Some(tv),
                decay: None,
            });
        }

        // (separator-row id row color) — full row of uniform non-zero color
        for r in 0..self.rows {
            if self.cols == 0 { continue; }
            let first = self.cells[r][0];
            if first > 0 && self.cells[r].iter().all(|&c| c == first) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("separator-row"),
                        id.clone(),
                        Neuron::int_val(r as i64),
                        Neuron::int_val(first as i64),
                    ]),
                    tv: Some(tv),
                    decay: None,
                });
            }
        }

        // (separator-col id col color) — full column of uniform non-zero color
        for c in 0..self.cols {
            if self.rows == 0 { continue; }
            let first = self.cells[0][c];
            if first > 0 && (0..self.rows).all(|r| self.cells[r][c] == first) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("separator-col"),
                        id.clone(),
                        Neuron::int_val(c as i64),
                        Neuron::int_val(first as i64),
                    ]),
                    tv: Some(tv),
                    decay: None,
                });
            }
        }

        // ── Region facts — when separators exist, compute rectangular zones ──
        {
            let sep_rows: Vec<usize> = (0..self.rows).filter(|&r| {
                self.cols > 0 && {
                    let first = self.cells[r][0];
                    first > 0 && self.cells[r].iter().all(|&c| c == first)
                }
            }).collect();
            let sep_cols: Vec<usize> = (0..self.cols).filter(|&c| {
                self.rows > 0 && {
                    let first = self.cells[0][c];
                    first > 0 && (0..self.rows).all(|r| self.cells[r][c] == first)
                }
            }).collect();

            if !sep_rows.is_empty() || !sep_cols.is_empty() {
                // Build row boundaries
                let mut row_bounds = vec![0usize];
                for &sr in &sep_rows { row_bounds.push(sr); }
                row_bounds.push(self.rows);
                row_bounds.sort();
                row_bounds.dedup();

                // Build col boundaries
                let mut col_bounds = vec![0usize];
                for &sc in &sep_cols { col_bounds.push(sc); }
                col_bounds.push(self.cols);
                col_bounds.sort();
                col_bounds.dedup();

                let sep_row_set: std::collections::HashSet<usize> = sep_rows.iter().copied().collect();
                let sep_col_set: std::collections::HashSet<usize> = sep_cols.iter().copied().collect();

                let mut region_id = 0usize;
                for ri in 0..row_bounds.len().saturating_sub(1) {
                    let r_start = row_bounds[ri];
                    let r_end = row_bounds[ri + 1];
                    for ci in 0..col_bounds.len().saturating_sub(1) {
                        let c_start = col_bounds[ci];
                        let c_end = col_bounds[ci + 1];

                        // Skip degenerate regions
                        if r_start >= r_end || c_start >= c_end { continue; }

                        let reg_sym = Neuron::int_val(region_id as i64);

                        // (grid-region id region_id min_r min_c max_r max_c)
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("grid-region"),
                                id.clone(),
                                reg_sym.clone(),
                                Neuron::int_val(r_start as i64),
                                Neuron::int_val(c_start as i64),
                                Neuron::int_val((r_end - 1) as i64),
                                Neuron::int_val((c_end - 1) as i64),
                            ]),
                            tv: Some(tv),
                            decay: None,
                        });

                        // (region-cell id region_id r c) for non-separator cells
                        for r in r_start..r_end {
                            if sep_row_set.contains(&r) { continue; }
                            for c in c_start..c_end {
                                if sep_col_set.contains(&c) { continue; }
                                stmts.push(Statement::Fact {
                                    neuron: Neuron::expression(vec![
                                        Neuron::symbol("region-cell"),
                                        id.clone(),
                                        reg_sym.clone(),
                                        Neuron::int_val(r as i64),
                                        Neuron::int_val(c as i64),
                                    ]),
                                    tv: Some(tv),
                                    decay: None,
                                });
                            }
                        }

                        region_id += 1;
                    }
                }
            }
        }

        // ── Diagonal neighbors — separate predicate from grid-neighbor ──
        // (grid-diag-neighbor id r c r2 c2) for 4 diagonal directions
        for r in 0..self.rows {
            for c in 0..self.cols {
                let diags: [(i64, i64); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
                for (dr, dc) in &diags {
                    let nr = r as i64 + dr;
                    let nc = c as i64 + dc;
                    if nr >= 0 && nr < self.rows as i64 && nc >= 0 && nc < self.cols as i64 {
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("grid-diag-neighbor"),
                                id.clone(),
                                Neuron::int_val(r as i64),
                                Neuron::int_val(c as i64),
                                Neuron::int_val(nr),
                                Neuron::int_val(nc),
                            ]),
                            tv: Some(tv),
                            decay: None,
                        });
                    }
                }
            }
        }

        // ── Color cell count — frequency of each color per grid ──
        // (color-cell-count id color count)
        {
            let mut color_counts: HashMap<u8, usize> = HashMap::new();
            for r in 0..self.rows {
                for c in 0..self.cols {
                    *color_counts.entry(self.cells[r][c]).or_insert(0) += 1;
                }
            }
            for (&color, &count) in &color_counts {
                if color == 0 { continue; } // skip background
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("color-cell-count"),
                        id.clone(),
                        Neuron::int_val(color as i64),
                        Neuron::int_val(count as i64),
                    ]),
                    tv: Some(tv),
                    decay: None,
                });
            }
        }

        stmts
    }

    /// Check equality with another grid.
    pub fn equals(&self, other: &Grid) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.cells == other.cells
    }

    /// Crop a sub-grid from (r0,c0) to (r0+rows,c0+cols).
    pub fn crop(&self, r0: usize, c0: usize, rows: usize, cols: usize) -> Option<Grid> {
        if r0 + rows > self.rows || c0 + cols > self.cols {
            return None;
        }
        let cells: Vec<Vec<u8>> = (r0..r0 + rows)
            .map(|r| self.cells[r][c0..c0 + cols].to_vec())
            .collect();
        Grid::from_vecs(cells).ok()
    }

    /// Reflect horizontally (left ↔ right).
    pub fn reflect_h(&self) -> Grid {
        let arr = self.to_array2();
        let reflected = arr.slice(ndarray::s![.., ..;-1]).to_owned();
        Grid::from_array2(&reflected)
    }

    /// Reflect vertically (top ↔ bottom).
    pub fn reflect_v(&self) -> Grid {
        let arr = self.to_array2();
        let reflected = arr.slice(ndarray::s![..;-1, ..]).to_owned();
        Grid::from_array2(&reflected)
    }

    /// Rotate 90 degrees clockwise.
    pub fn rotate_90(&self) -> Grid {
        let arr = self.to_array2();
        // Transpose then reverse each row = 90° CW
        let t = arr.t().to_owned();
        let rotated = t.slice(ndarray::s![.., ..;-1]).to_owned();
        Grid::from_array2(&rotated)
    }

    /// Rotate 180 degrees.
    pub fn rotate_180(&self) -> Grid {
        let arr = self.to_array2();
        let rotated = arr.slice(ndarray::s![..;-1, ..;-1]).to_owned();
        Grid::from_array2(&rotated)
    }

    /// Rotate 270 degrees clockwise (= 90 counter-clockwise).
    pub fn rotate_270(&self) -> Grid {
        let arr = self.to_array2();
        // Reverse each row then transpose = 270° CW
        let reversed = arr.slice(ndarray::s![.., ..;-1]).to_owned();
        let rotated = reversed.t().to_owned();
        Grid::from_array2(&rotated)
    }

    /// Transpose (swap rows and columns).
    pub fn transpose(&self) -> Grid {
        let arr = self.to_array2();
        let t = arr.t().to_owned();
        Grid::from_array2(&t)
    }

    // ── Pair-level transform detection (perception) ────────────────────

    /// Compare two grids and emit perception facts about their relationship.
    /// These are PERCEPTION facts — detecting what transform occurred.
    /// All solving logic remains in .qor rules.
    ///
    /// Emits:
    /// - Cell-level: cell-kept, cell-removed, cell-added, recolored (pre-computed here for speed)
    /// - Transform-level: pair-identity, pair-reflect-h/v, pair-rotate-*, pair-shift, etc.
    pub fn compare_pair(input: &Grid, output: &Grid, in_id: &str, out_id: &str) -> Vec<Statement> {
        let mut stmts = Vec::new();
        let tv = Some(TruthValue::new(0.99, 0.99));
        let tv_cell = Some(TruthValue::new(0.90, 0.90));

        // ── Cell-level comparison (pre-computed for speed) ────────────────
        // Pre-computed here instead of QOR rules — O(1) insert vs O(n) chain.
        //
        // Semantics (matching original QOR rules):
        //   cell-kept:    in_color != 0 AND out has SAME color at (r,c)
        //   cell-removed: in_color != 0 AND out does NOT have in_color at (r,c)
        //   cell-added:   out_color != 0 AND in does NOT have out_color at (r,c)
        //   recolored:    both non-zero AND different colors at (r,c)
        //
        // Note: recolored cells are ALSO cell-removed + cell-added (matching old rules).

        let max_r = input.rows.max(output.rows);
        let max_c = input.cols.max(output.cols);

        for r in 0..max_r {
            for c in 0..max_c {
                let in_color = if r < input.rows && c < input.cols { input.cells[r][c] } else { 0 };
                let out_color = if r < output.rows && c < output.cols { output.cells[r][c] } else { 0 };

                if in_color != 0 && out_color == in_color {
                    // cell-kept
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("cell-kept"),
                            Neuron::symbol(in_id), Neuron::symbol(out_id),
                            Neuron::int_val(r as i64), Neuron::int_val(c as i64),
                            Neuron::int_val(in_color as i64),
                        ]),
                        tv: tv_cell, decay: None,
                    });
                }

                if in_color != 0 && out_color != in_color {
                    // cell-removed (original color gone — includes recolored)
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("cell-removed"),
                            Neuron::symbol(in_id), Neuron::symbol(out_id),
                            Neuron::int_val(r as i64), Neuron::int_val(c as i64),
                            Neuron::int_val(in_color as i64),
                        ]),
                        tv: tv_cell, decay: None,
                    });
                }

                if out_color != 0 && in_color != out_color {
                    // cell-added (new color appeared — includes recolored)
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("cell-added"),
                            Neuron::symbol(in_id), Neuron::symbol(out_id),
                            Neuron::int_val(r as i64), Neuron::int_val(c as i64),
                            Neuron::int_val(out_color as i64),
                        ]),
                        tv: tv_cell, decay: None,
                    });
                }

                if in_color != 0 && out_color != 0 && in_color != out_color {
                    // recolored
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("recolored"),
                            Neuron::symbol(in_id), Neuron::symbol(out_id),
                            Neuron::int_val(r as i64), Neuron::int_val(c as i64),
                            Neuron::int_val(in_color as i64),
                            Neuron::int_val(out_color as i64),
                        ]),
                        tv: tv_cell, decay: None,
                    });
                }
            }
        }

        // ── Transform-level detection ────────────────────────────────────

        // Identity
        if input.equals(output) {
            stmts.push(Statement::Fact {
                neuron: Neuron::expression(vec![
                    Neuron::symbol("pair-identity"), Neuron::symbol(in_id), Neuron::symbol(out_id),
                ]),
                tv, decay: None,
            });
        }

        // Same-size spatial transforms
        if input.rows == output.rows && input.cols == output.cols {
            if input.reflect_h().equals(output) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("pair-reflect-h"), Neuron::symbol(in_id), Neuron::symbol(out_id),
                    ]),
                    tv, decay: None,
                });
            }
            if input.reflect_v().equals(output) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("pair-reflect-v"), Neuron::symbol(in_id), Neuron::symbol(out_id),
                    ]),
                    tv, decay: None,
                });
            }
            if input.rotate_180().equals(output) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("pair-rotate-180"), Neuron::symbol(in_id), Neuron::symbol(out_id),
                    ]),
                    tv, decay: None,
                });
            }

            // Shift detection
            if let Some((dr, dc)) = Grid::detect_shift(input, output) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("pair-shift"), Neuron::symbol(in_id), Neuron::symbol(out_id),
                        Neuron::int_val(dr as i64), Neuron::int_val(dc as i64),
                    ]),
                    tv, decay: None,
                });
            }
        }

        // Rotation (changes dimensions for non-square)
        if input.rotate_90().equals(output) {
            stmts.push(Statement::Fact {
                neuron: Neuron::expression(vec![
                    Neuron::symbol("pair-rotate-90"), Neuron::symbol(in_id), Neuron::symbol(out_id),
                ]),
                tv, decay: None,
            });
        }
        if input.rotate_270().equals(output) {
            stmts.push(Statement::Fact {
                neuron: Neuron::expression(vec![
                    Neuron::symbol("pair-rotate-270"), Neuron::symbol(in_id), Neuron::symbol(out_id),
                ]),
                tv, decay: None,
            });
        }

        // Transpose
        if input.transpose().equals(output) {
            stmts.push(Statement::Fact {
                neuron: Neuron::expression(vec![
                    Neuron::symbol("pair-transpose"), Neuron::symbol(in_id), Neuron::symbol(out_id),
                ]),
                tv, decay: None,
            });
        }

        // Crop detection (output smaller than input)
        if output.rows <= input.rows && output.cols <= input.cols
            && (output.rows < input.rows || output.cols < input.cols)
        {
            if let Some((r0, c0)) = Grid::detect_crop(input, output) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("pair-crop"), Neuron::symbol(in_id), Neuron::symbol(out_id),
                        Neuron::int_val(r0 as i64), Neuron::int_val(c0 as i64),
                    ]),
                    tv, decay: None,
                });
            }

            // Crop-to-bbox: output == input cropped to content bounding box
            if let Some((min_r, min_c, max_r, max_c)) = Grid::content_bbox(input) {
                let bbox_h = max_r - min_r + 1;
                let bbox_w = max_c - min_c + 1;
                if bbox_h == output.rows && bbox_w == output.cols {
                    if let Some(cropped) = input.crop(min_r, min_c, bbox_h, bbox_w) {
                        if cropped.equals(output) {
                            stmts.push(Statement::Fact {
                                neuron: Neuron::expression(vec![
                                    Neuron::symbol("pair-crop-to-bbox"),
                                    Neuron::symbol(in_id), Neuron::symbol(out_id),
                                ]),
                                tv, decay: None,
                            });
                        }
                    }
                }
            }
        }

        // Scale detection (output is integer multiple of input)
        if output.rows > input.rows && output.cols > input.cols
            && output.rows % input.rows == 0 && output.cols % input.cols == 0
        {
            let sr = output.rows / input.rows;
            let sc = output.cols / input.cols;
            if sr == sc {
                // Uniform scale — verify all cells match
                let mut matches = true;
                for r in 0..output.rows {
                    for c in 0..output.cols {
                        if input.cells[r / sr][c / sc] != output.cells[r][c] {
                            matches = false;
                            break;
                        }
                    }
                    if !matches { break; }
                }
                if matches {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("pair-scale-up"), Neuron::symbol(in_id), Neuron::symbol(out_id),
                            Neuron::int_val(sr as i64),
                        ]),
                        tv, decay: None,
                    });
                }
            }
        }

        // ── Symmetry completion detection ────────────────────────────────
        // Check if output = input completed to be horizontally/vertically symmetric.
        // Requires: output is symmetric, input is not, AND all input cells preserved.
        if input.rows == output.rows && input.cols == output.cols {
            // Verify all non-zero input cells are preserved in output
            let mut input_cells_preserved = true;
            for r in 0..input.rows {
                for c in 0..input.cols {
                    if input.cells[r][c] != 0 && output.cells[r][c] != input.cells[r][c] {
                        input_cells_preserved = false;
                        break;
                    }
                }
                if !input_cells_preserved { break; }
            }

            if input_cells_preserved {
                let ref_h = output.reflect_h();
                if ref_h.equals(output) && !input.reflect_h().equals(input) {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("pair-symmetry-complete"),
                            Neuron::symbol(in_id), Neuron::symbol(out_id),
                            Neuron::symbol("h"),
                        ]),
                        tv, decay: None,
                    });
                }
                let ref_v = output.reflect_v();
                if ref_v.equals(output) && !input.reflect_v().equals(input) {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("pair-symmetry-complete"),
                            Neuron::symbol(in_id), Neuron::symbol(out_id),
                            Neuron::symbol("v"),
                        ]),
                        tv, decay: None,
                    });
                }
            }
        }

        // ── Flood fill detection ────────────────────────────────────────
        // Check if output = input with enclosed zero regions filled with a single color.
        // Uses BFS to verify filled cells are in regions NOT reachable from the border.
        if input.rows == output.rows && input.cols == output.cols {
            // First check: non-zero cells in input must be preserved
            let mut input_preserved = true;
            for r in 0..input.rows {
                for c in 0..input.cols {
                    if input.cells[r][c] != 0 && input.cells[r][c] != output.cells[r][c] {
                        input_preserved = false;
                        break;
                    }
                }
                if !input_preserved { break; }
            }

            // Find filled cells and verify they are enclosed (BFS from border)
            if input_preserved {
                // BFS: find all zero-cells reachable from the border (NOT enclosed)
                let mut border_reachable = vec![vec![false; input.cols]; input.rows];
                let mut queue = std::collections::VecDeque::new();
                for r in 0..input.rows {
                    for c in 0..input.cols {
                        if input.cells[r][c] == 0
                            && (r == 0 || r == input.rows - 1 || c == 0 || c == input.cols - 1)
                            && !border_reachable[r][c]
                        {
                            border_reachable[r][c] = true;
                            queue.push_back((r, c));
                        }
                    }
                }
                while let Some((r, c)) = queue.pop_front() {
                    for (dr, dc) in [(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                        let nr = r as i32 + dr;
                        let nc = c as i32 + dc;
                        if nr >= 0 && nc >= 0 {
                            let (nr, nc) = (nr as usize, nc as usize);
                            if nr < input.rows && nc < input.cols
                                && !border_reachable[nr][nc]
                                && input.cells[nr][nc] == 0
                            {
                                border_reachable[nr][nc] = true;
                                queue.push_back((nr, nc));
                            }
                        }
                    }
                }

                // Now check: filled cells must be in enclosed regions (not border-reachable)
                let mut fill_color: Option<u8> = None;
                let mut is_fill = true;
                let mut fill_count = 0;
                for r in 0..input.rows {
                    for c in 0..input.cols {
                        if input.cells[r][c] == 0 && output.cells[r][c] != 0 {
                            // This cell was filled — verify it's enclosed
                            if border_reachable[r][c] {
                                is_fill = false; // Filled a non-enclosed cell
                                break;
                            }
                            fill_count += 1;
                            match fill_color {
                                None => fill_color = Some(output.cells[r][c]),
                                Some(fc) if fc != output.cells[r][c] => { is_fill = false; }
                                _ => {}
                            }
                        }
                    }
                    if !is_fill { break; }
                }
                // Also verify: enclosed zero cells that WEREN'T filled → not a complete fill
                // (We allow partial fills — just check that what WAS filled is correct)

                if is_fill && fill_count > 0 {
                    if let Some(fc) = fill_color {
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("pair-flood-fill"),
                                Neuron::symbol(in_id), Neuron::symbol(out_id),
                                Neuron::int_val(fc as i64),
                            ]),
                            tv, decay: None,
                        });
                    }
                }
            }
        }

        // ── Color remap detection (per-pair) ────────────────────────────
        if let Some(remap) = Grid::detect_color_remap(input, output) {
            for (&from, &to) in &remap {
                if from != to {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("color-remap"),
                            Neuron::symbol(in_id), Neuron::symbol(out_id),
                            Neuron::int_val(from as i64), Neuron::int_val(to as i64),
                        ]),
                        tv, decay: None,
                    });
                }
            }
            stmts.push(Statement::Fact {
                neuron: Neuron::expression(vec![
                    Neuron::symbol("pair-color-remap"),
                    Neuron::symbol(in_id), Neuron::symbol(out_id),
                ]),
                tv, decay: None,
            });
        }

        // ── Composed transforms (transform + color remap) ───────────────
        // For each geometric transform T, check if T(input) → output via color remap.
        for (name, transformed) in [
            ("reflect-h", input.reflect_h()),
            ("reflect-v", input.reflect_v()),
            ("rotate-90", input.rotate_90()),
            ("rotate-180", input.rotate_180()),
            ("rotate-270", input.rotate_270()),
            ("transpose", input.transpose()),
        ] {
            if transformed.rows == output.rows && transformed.cols == output.cols
                && !transformed.equals(output)
            {
                if Grid::detect_color_remap(&transformed, output).is_some() {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("pair-composed"),
                            Neuron::symbol(in_id), Neuron::symbol(out_id),
                            Neuron::symbol(name), Neuron::symbol("color-remap"),
                        ]),
                        tv, decay: None,
                    });
                }
            }
        }

        // ── Gravity detection ───────────────────────────────────────────
        // Check if output looks like input with non-zero cells "fallen" downward
        if input.rows == output.rows && input.cols == output.cols {
            // For each column, collect non-zero cells, stack at bottom
            let mut gravity_match = true;
            let mut has_movement = false;
            for c in 0..input.cols {
                let mut col_colors: Vec<u8> = Vec::new();
                for r in 0..input.rows {
                    if input.cells[r][c] != 0 {
                        col_colors.push(input.cells[r][c]);
                    }
                }
                // Stack at bottom
                let empty = input.rows - col_colors.len();
                for r in 0..input.rows {
                    let expected = if r < empty { 0 } else { col_colors[r - empty] };
                    if expected != output.cells[r][c] {
                        gravity_match = false;
                        break;
                    }
                    if expected != input.cells[r][c] {
                        has_movement = true;
                    }
                }
                if !gravity_match { break; }
            }
            if gravity_match && has_movement {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("pair-gravity-down"),
                        Neuron::symbol(in_id), Neuron::symbol(out_id),
                    ]),
                    tv, decay: None,
                });
            }
        }

        // ── Row/Col fill detection ─────────────────────────────────────
        // Checks if non-zero "marker" cells in input cause entire rows/cols
        // to be filled with that marker's color in the output.
        if input.rows == output.rows && input.cols == output.cols {
            // Row fill: each non-zero input cell → its entire row filled with that color
            let row_fill = Grid::detect_row_fill(input, output);
            if row_fill {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("pair-row-fill"),
                        Neuron::symbol(in_id), Neuron::symbol(out_id),
                    ]),
                    tv, decay: None,
                });
            }

            // Col fill: each non-zero input cell → its entire column filled
            let col_fill = Grid::detect_col_fill(input, output);
            if col_fill {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("pair-col-fill"),
                        Neuron::symbol(in_id), Neuron::symbol(out_id),
                    ]),
                    tv, decay: None,
                });
            }

            // Cross fill: each non-zero input cell → cross (row + col) filled
            let cross_fill = Grid::detect_cross_fill(input, output);
            if cross_fill {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("pair-cross-fill"),
                        Neuron::symbol(in_id), Neuron::symbol(out_id),
                    ]),
                    tv, decay: None,
                });
            }
        }

        // ── Region fill from seed detection ────────────────────────────
        // Checks if enclosed zero-regions in the input get filled with the
        // color of a "seed" marker inside that region.
        if input.rows == output.rows && input.cols == output.cols && input.rows >= 3 {
            if let Some(bg_color) = Grid::detect_region_fill_seed(input, output) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("pair-region-fill-seed"),
                        Neuron::symbol(in_id), Neuron::symbol(out_id),
                        Neuron::int_val(bg_color as i64),
                    ]),
                    tv, decay: None,
                });
            }
        }

        // ── Noise removal detection ─────────────────────────────────────
        // Checks if output = input with "noise" cells removed.
        // Variants: isolated pixels, minority color, small objects.
        if input.rows == output.rows && input.cols == output.cols {
            // Variant 1: remove isolated pixels (no same-color 4-neighbor)
            if Grid::detect_noise_isolated(input, output) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("pair-noise-remove-isolated"),
                        Neuron::symbol(in_id), Neuron::symbol(out_id),
                    ]),
                    tv, decay: None,
                });
            }

            // Variant 2: remove minority color(s)
            if let Some(keep_color) = Grid::detect_noise_minority(input, output) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("pair-noise-remove-minority"),
                        Neuron::symbol(in_id), Neuron::symbol(out_id),
                        Neuron::int_val(keep_color as i64),
                    ]),
                    tv, decay: None,
                });
            }

            // Variant 3: keep only largest object
            if Grid::detect_keep_largest(input, output) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("pair-keep-largest"),
                        Neuron::symbol(in_id), Neuron::symbol(out_id),
                    ]),
                    tv, decay: None,
                });
            }

            // Variant 4: remove small objects (below size threshold)
            if let Some(threshold) = Grid::detect_noise_small_objects(input, output) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("pair-noise-remove-small"),
                        Neuron::symbol(in_id), Neuron::symbol(out_id),
                        Neuron::int_val(threshold as i64),
                    ]),
                    tv, decay: None,
                });
            }
        }

        // ── Color histogram detection ──────────────────────────────────
        // Checks if output is a sorted bar chart of color frequencies from input.
        if Grid::detect_color_histogram(input, output) {
            stmts.push(Statement::Fact {
                neuron: Neuron::expression(vec![
                    Neuron::symbol("pair-color-histogram"),
                    Neuron::symbol(in_id), Neuron::symbol(out_id),
                ]),
                tv, decay: None,
            });
        }

        stmts
    }

    // ── Cross-pair observations (Phase 1A) ─────────────────────────────

    /// Compute higher-level consistency facts across ALL training pairs.
    /// Called after all compare_pair() calls, before prediction.
    /// Emits obs-consistent and obs-parameter facts.
    pub fn compute_observations(
        pair_facts: &[Vec<Statement>],
        pair_grids: &[(Grid, Grid)],
    ) -> Vec<Statement> {
        let mut stmts = Vec::new();
        let tv = Some(TruthValue::new(0.95, 0.95));
        let n = pair_grids.len();
        if n == 0 { return stmts; }

        // Helper: check if a predicate appears in ALL pair fact sets
        let all_have = |pred: &str| -> bool {
            pair_facts.iter().all(|facts| {
                facts.iter().any(|s| {
                    if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = s {
                        parts.first().map(|p| p.to_string() == pred).unwrap_or(false)
                    } else { false }
                })
            })
        };

        // obs-consistent same-size — all pairs have same input/output dimensions
        if pair_grids.iter().all(|(i, o)| i.rows == o.rows && i.cols == o.cols) {
            stmts.push(Statement::Fact {
                neuron: Neuron::expression(vec![
                    Neuron::symbol("obs-consistent"), Neuron::symbol("same-size"),
                ]),
                tv, decay: None,
            });
        }

        // obs-consistent output-smaller — all outputs smaller than inputs
        if pair_grids.iter().all(|(i, o)| o.rows <= i.rows && o.cols <= i.cols
            && (o.rows < i.rows || o.cols < i.cols))
        {
            stmts.push(Statement::Fact {
                neuron: Neuron::expression(vec![
                    Neuron::symbol("obs-consistent"), Neuron::symbol("output-smaller"),
                ]),
                tv, decay: None,
            });
        }

        // obs-consistent output-larger — all outputs larger than inputs
        if pair_grids.iter().all(|(i, o)| o.rows >= i.rows && o.cols >= i.cols
            && (o.rows > i.rows || o.cols > i.cols))
        {
            stmts.push(Statement::Fact {
                neuron: Neuron::expression(vec![
                    Neuron::symbol("obs-consistent"), Neuron::symbol("output-larger"),
                ]),
                tv, decay: None,
            });
        }

        // Cache objects() results (expensive flood-fill BFS)
        let cached_objects: Vec<(Vec<GridObject>, Vec<GridObject>)> = pair_grids.iter()
            .map(|(i, o)| (i.objects(), o.objects()))
            .collect();

        // obs-consistent object-count-same — all pairs preserve object count
        if cached_objects.iter().all(|(io, oo)| io.len() == oo.len()) {
            stmts.push(Statement::Fact {
                neuron: Neuron::expression(vec![
                    Neuron::symbol("obs-consistent"), Neuron::symbol("object-count-same"),
                ]),
                tv, decay: None,
            });
        }

        // obs-consistent pure-color-change — only colors differ (same positions)
        let pure_color = pair_grids.iter().all(|(i, o)| {
            if i.rows != o.rows || i.cols != o.cols { return false; }
            for r in 0..i.rows {
                for c in 0..i.cols {
                    let ic = i.cells[r][c];
                    let oc = o.cells[r][c];
                    if (ic == 0) != (oc == 0) { return false; }
                }
            }
            true
        });
        if pure_color {
            stmts.push(Statement::Fact {
                neuron: Neuron::expression(vec![
                    Neuron::symbol("obs-consistent"), Neuron::symbol("pure-color-change"),
                ]),
                tv, decay: None,
            });
        }

        // obs-parameter color-remap — consistent remap across ALL pairs
        let remaps: Vec<Option<std::collections::HashMap<u8, u8>>> = pair_grids.iter()
            .map(|(i, o)| Grid::detect_color_remap(i, o))
            .collect();
        if remaps.iter().all(|r| r.is_some()) {
            let first = remaps[0].as_ref().unwrap();
            let all_consistent = remaps.iter().skip(1).all(|r| {
                let r = r.as_ref().unwrap();
                first.iter().all(|(k, v)| r.get(k) == Some(v))
            });
            if all_consistent {
                for (&from, &to) in first {
                    if from != to {
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("obs-parameter"),
                                Neuron::symbol("color-remap"),
                                Neuron::int_val(from as i64),
                                Neuron::int_val(to as i64),
                            ]),
                            tv, decay: None,
                        });
                    }
                }
            }
        }

        // obs-consistent for each geometric transform
        for pred in &[
            "pair-reflect-h", "pair-reflect-v", "pair-rotate-90",
            "pair-rotate-180", "pair-rotate-270", "pair-transpose",
            "pair-crop-to-bbox", "pair-gravity-down", "pair-color-remap",
            "pair-row-fill", "pair-col-fill", "pair-cross-fill",
            "pair-region-fill-seed", "pair-color-histogram",
            "pair-noise-remove-isolated", "pair-noise-remove-minority",
            "pair-keep-largest", "pair-noise-remove-small",
        ] {
            if all_have(pred) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("obs-consistent"),
                        Neuron::symbol(pred.strip_prefix("pair-").unwrap_or(pred)),
                    ]),
                    tv, decay: None,
                });
            }
        }

        // obs-consistent for composed transforms
        // Check if ALL pairs have the same pair-composed with same (t1, t2)
        let composed: Vec<Vec<(String, String)>> = pair_facts.iter().map(|facts| {
            facts.iter().filter_map(|s| {
                if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = s {
                    if parts.len() == 5 && parts[0] == Neuron::symbol("pair-composed") {
                        return Some((parts[3].to_string(), parts[4].to_string()));
                    }
                }
                None
            }).collect()
        }).collect();

        if !composed.is_empty() && composed.iter().all(|c| !c.is_empty()) {
            // Find composed transforms present in ALL pairs
            for (t1, t2) in &composed[0] {
                if composed.iter().all(|c| c.iter().any(|(a, b)| a == t1 && b == t2)) {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("obs-consistent"),
                            Neuron::symbol("composed"),
                            Neuron::symbol(t1),
                            Neuron::symbol(t2),
                        ]),
                        tv, decay: None,
                    });
                }
            }
        }

        // obs-consistent has-enclosed-regions — all inputs have enclosed zero regions
        let all_enclosed = pair_grids.iter().all(|(input, _)| {
            if input.rows < 3 || input.cols < 3 { return false; }
            let mut border_reachable = vec![vec![false; input.cols]; input.rows];
            let mut queue = std::collections::VecDeque::new();
            for r in 0..input.rows {
                for c in 0..input.cols {
                    if input.cells[r][c] == 0
                        && (r == 0 || r == input.rows - 1 || c == 0 || c == input.cols - 1)
                        && !border_reachable[r][c]
                    {
                        border_reachable[r][c] = true;
                        queue.push_back((r, c));
                    }
                }
            }
            while let Some((r, c)) = queue.pop_front() {
                for (dr, dc) in [(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr >= 0 && nc >= 0 {
                        let (nr, nc) = (nr as usize, nc as usize);
                        if nr < input.rows && nc < input.cols
                            && !border_reachable[nr][nc] && input.cells[nr][nc] == 0
                        {
                            border_reachable[nr][nc] = true;
                            queue.push_back((nr, nc));
                        }
                    }
                }
            }
            // Has at least one zero cell not reachable from border
            (0..input.rows).any(|r| {
                (0..input.cols).any(|c| input.cells[r][c] == 0 && !border_reachable[r][c])
            })
        });
        if all_enclosed {
            stmts.push(Statement::Fact {
                neuron: Neuron::expression(vec![
                    Neuron::symbol("obs-consistent"), Neuron::symbol("has-enclosed-regions"),
                ]),
                tv, decay: None,
            });

            // Determine fill color: what color replaces enclosed zero cells in output?
            // Must be consistent across ALL training pairs.
            let fill_colors: Vec<Option<u8>> = pair_grids.iter().map(|(input, output)| {
                let reachable = input.border_reachable_zeros();
                let mut fill: Option<u8> = None;
                for r in 0..input.rows.min(output.rows) {
                    for c in 0..input.cols.min(output.cols) {
                        if input.cells[r][c] == 0 && !reachable[r][c] && output.cells[r][c] > 0 {
                            match fill {
                                None => fill = Some(output.cells[r][c]),
                                Some(f) if f != output.cells[r][c] => return None, // inconsistent
                                _ => {}
                            }
                        }
                    }
                }
                fill
            }).collect();
            if !fill_colors.is_empty() && fill_colors.iter().all(|c| c.is_some()) {
                let first = fill_colors[0].unwrap();
                if fill_colors.iter().all(|c| c.unwrap() == first) {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("obs-enclosed-fill-color"),
                            Neuron::int_val(first as i64),
                        ]),
                        tv, decay: None,
                    });
                }
            }

            // Also detect the wall color (what forms the enclosure boundary)
            let wall_colors: Vec<Option<u8>> = pair_grids.iter().map(|(input, _)| {
                let reachable = input.border_reachable_zeros();
                let mut wall: Option<u8> = None;
                for r in 0..input.rows {
                    for c in 0..input.cols {
                        if input.cells[r][c] == 0 && !reachable[r][c] {
                            // Check 4-neighbors for non-zero cells (boundary)
                            for (dr, dc) in [(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                                let nr = r as i32 + dr;
                                let nc = c as i32 + dc;
                                if nr >= 0 && nc >= 0 {
                                    let (nr, nc) = (nr as usize, nc as usize);
                                    if nr < input.rows && nc < input.cols && input.cells[nr][nc] > 0 {
                                        match wall {
                                            None => wall = Some(input.cells[nr][nc]),
                                            Some(w) if w != input.cells[nr][nc] => return None,
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                wall
            }).collect();
            if !wall_colors.is_empty() && wall_colors.iter().all(|c| c.is_some()) {
                let first = wall_colors[0].unwrap();
                if wall_colors.iter().all(|c| c.unwrap() == first) {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("obs-enclosed-wall-color"),
                            Neuron::int_val(first as i64),
                        ]),
                        tv, decay: None,
                    });
                }
            }
        }

        // obs-enclosed-size-fill: per-region area→fill mapping
        // When enclosed regions have DIFFERENT fill colors, map area→color.
        if all_enclosed {
            let region_size_fills: Vec<Vec<(usize, u8)>> = pair_grids.iter().map(|(input, output)| {
                let reachable = input.border_reachable_zeros();
                let mut visited = vec![vec![false; input.cols]; input.rows];
                let mut regions = Vec::new();
                for r in 0..input.rows {
                    for c in 0..input.cols {
                        if input.cells[r][c] == 0 && !reachable[r][c] && !visited[r][c] {
                            let mut cells = Vec::new();
                            let mut q = VecDeque::new();
                            q.push_back((r, c));
                            visited[r][c] = true;
                            while let Some((cr, cc)) = q.pop_front() {
                                cells.push((cr, cc));
                                for (dr, dc) in [(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                                    let nr = cr as i32 + dr;
                                    let nc = cc as i32 + dc;
                                    if nr >= 0 && nc >= 0 {
                                        let (nr, nc) = (nr as usize, nc as usize);
                                        if nr < input.rows && nc < input.cols
                                            && input.cells[nr][nc] == 0
                                            && !reachable[nr][nc]
                                            && !visited[nr][nc]
                                        {
                                            visited[nr][nc] = true;
                                            q.push_back((nr, nc));
                                        }
                                    }
                                }
                            }
                            let mut fill: Option<u8> = None;
                            let mut consistent = true;
                            for &(cr, cc) in &cells {
                                if cr < output.rows && cc < output.cols && output.cells[cr][cc] > 0 {
                                    match fill {
                                        None => fill = Some(output.cells[cr][cc]),
                                        Some(f) if f != output.cells[cr][cc] => { consistent = false; break; },
                                        _ => {}
                                    }
                                }
                            }
                            if consistent {
                                if let Some(f) = fill {
                                    regions.push((cells.len(), f));
                                }
                            }
                        }
                    }
                }
                regions
            }).collect();

            // Build size→fill mapping; emit only if each size maps to exactly one color
            let mut size_fill_map: HashMap<usize, HashSet<u8>> = HashMap::new();
            for pair_regions in &region_size_fills {
                for &(size, fill) in pair_regions {
                    size_fill_map.entry(size).or_default().insert(fill);
                }
            }
            for (size, fills) in &size_fill_map {
                if fills.len() == 1 {
                    let fill = *fills.iter().next().unwrap();
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("obs-enclosed-size-fill"),
                            Neuron::int_val(*size as i64),
                            Neuron::int_val(fill as i64),
                        ]),
                        tv, decay: None,
                    });
                }
            }
        }

        // obs-shape-remap: shape-key determines color remap
        // Pattern: input has 2 non-zero colors; the smaller one is a "key" whose shape
        // determines what color the larger one gets remapped to in the output.
        {
            let mut shape_remap_valid = true;
            let mut shape_maps: HashMap<i64, u8> = HashMap::new();
            let mut target_color: Option<u8> = None;
            let mut key_color: Option<u8> = None;

            for (input, output) in pair_grids.iter() {
                if !shape_remap_valid { break; }
                let mut color_counts: HashMap<u8, usize> = HashMap::new();
                for r in 0..input.rows {
                    for c in 0..input.cols {
                        if input.cells[r][c] > 0 {
                            *color_counts.entry(input.cells[r][c]).or_default() += 1;
                        }
                    }
                }
                if color_counts.len() != 2 { shape_remap_valid = false; break; }

                let mut colors: Vec<(u8, usize)> = color_counts.into_iter().collect();
                colors.sort_by_key(|&(_, count)| count);
                let this_key = colors[0].0;
                let this_target = colors[1].0;

                match key_color {
                    None => key_color = Some(this_key),
                    Some(k) if k != this_key => { shape_remap_valid = false; break; },
                    _ => {}
                }
                match target_color {
                    None => target_color = Some(this_target),
                    Some(t) if t != this_target => { shape_remap_valid = false; break; },
                    _ => {}
                }

                // Verify key cells become 0, target cells become one consistent color
                let mut target_out: Option<u8> = None;
                for r in 0..input.rows {
                    for c in 0..input.cols {
                        if !shape_remap_valid { break; }
                        if input.cells[r][c] == this_key && r < output.rows && c < output.cols {
                            if output.cells[r][c] != 0 { shape_remap_valid = false; }
                        }
                        if input.cells[r][c] == this_target && r < output.rows && c < output.cols {
                            match target_out {
                                None => target_out = Some(output.cells[r][c]),
                                Some(t) if t != output.cells[r][c] => { shape_remap_valid = false; },
                                _ => {}
                            }
                        }
                    }
                }
                if !shape_remap_valid { break; }

                // Compute key shape bitmap signature
                let key_cells: Vec<(usize, usize)> = (0..input.rows).flat_map(|r| {
                    (0..input.cols).filter_map(move |c| {
                        if input.cells[r][c] == this_key { Some((r, c)) } else { None }
                    })
                }).collect();
                if key_cells.is_empty() { shape_remap_valid = false; break; }

                let min_r = key_cells.iter().map(|&(r, _)| r).min().unwrap();
                let min_c = key_cells.iter().map(|&(_, c)| c).min().unwrap();
                let max_r = key_cells.iter().map(|&(r, _)| r).max().unwrap();
                let max_c = key_cells.iter().map(|&(_, c)| c).max().unwrap();
                let h = max_r - min_r + 1;
                let w = max_c - min_c + 1;
                if h * w > 25 { shape_remap_valid = false; break; }

                let cell_set: HashSet<(usize, usize)> = key_cells.iter().copied().collect();
                let mut sig: i64 = 0;
                for dr in 0..h {
                    for dc in 0..w {
                        sig <<= 1;
                        if cell_set.contains(&(min_r + dr, min_c + dc)) { sig |= 1; }
                    }
                }

                if let Some(out_c) = target_out {
                    match shape_maps.get(&sig) {
                        None => { shape_maps.insert(sig, out_c); },
                        Some(&existing) if existing != out_c => { shape_remap_valid = false; },
                        _ => {}
                    }
                }
            }

            if shape_remap_valid && !shape_maps.is_empty() {
                if let (Some(key), Some(target)) = (key_color, target_color) {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("obs-shape-remap"),
                            Neuron::int_val(target as i64),
                        ]),
                        tv, decay: None,
                    });
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("obs-shape-key-color"),
                            Neuron::int_val(key as i64),
                        ]),
                        tv, decay: None,
                    });
                    for (sig, color) in &shape_maps {
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("obs-shape-map"),
                                Neuron::int_val(*sig),
                                Neuron::int_val(*color as i64),
                            ]),
                            tv, decay: None,
                        });
                    }
                }
            }
        }

        // obs-parameter shift — consistent shift across all pairs
        let shifts: Vec<Option<(i32, i32)>> = pair_grids.iter()
            .map(|(i, o)| Grid::detect_shift(i, o))
            .collect();
        if shifts.iter().all(|s| s.is_some()) {
            let first = shifts[0].unwrap();
            if shifts.iter().all(|s| s.unwrap() == first) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("obs-parameter"), Neuron::symbol("shift-dr"),
                        Neuron::int_val(first.0 as i64),
                    ]),
                    tv, decay: None,
                });
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("obs-parameter"), Neuron::symbol("shift-dc"),
                        Neuron::int_val(first.1 as i64),
                    ]),
                    tv, decay: None,
                });
            }
        }

        // ── FIX 4: Per-cell diff analysis + color-map aggregation ──
        // For same-size pairs, compare each cell: input vs output.
        // Aggregate into color-map facts validated across ALL pairs.
        {
            use std::collections::HashMap;

            // Build per-pair color maps: HashMap<(from, to), count>
            let pair_color_maps: Vec<HashMap<(u8, u8), usize>> = pair_grids.iter()
                .filter(|(i, o)| i.rows == o.rows && i.cols == o.cols)
                .map(|(i, o)| {
                    let mut map: HashMap<(u8, u8), usize> = HashMap::new();
                    for r in 0..i.rows {
                        for c in 0..i.cols {
                            *map.entry((i.cells[r][c], o.cells[r][c])).or_insert(0) += 1;
                        }
                    }
                    map
                })
                .collect();

            if !pair_color_maps.is_empty() && pair_color_maps.len() == pair_grids.len() {
                // Find color mappings present in ALL pairs
                let first_map = &pair_color_maps[0];
                for (&(from, to), _) in first_map {
                    if from == to { continue; } // Skip identity mappings
                    let universal = pair_color_maps.iter().all(|m| m.contains_key(&(from, to)));
                    if universal {
                        // (obs-color-map from to) — "color `from` becomes `to` in all pairs"
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("obs-color-map"),
                                Neuron::int_val(from as i64),
                                Neuron::int_val(to as i64),
                            ]),
                            tv, decay: None,
                        });
                    }
                }

                // Count changed vs unchanged cells per pair, report consistency
                let change_ratios: Vec<f64> = pair_grids.iter()
                    .filter(|(i, o)| i.rows == o.rows && i.cols == o.cols)
                    .map(|(i, o)| {
                        let total = i.rows * i.cols;
                        let changed = (0..i.rows).flat_map(|r| (0..i.cols)
                            .filter(move |&c| i.cells[r][c] != o.cells[r][c])).count();
                        if total > 0 { changed as f64 / total as f64 } else { 0.0 }
                    })
                    .collect();

                if !change_ratios.is_empty() {
                    let avg_change = change_ratios.iter().sum::<f64>() / change_ratios.len() as f64;
                    // Quantize: few-cells-change (<10%), moderate (10-50%), most-cells-change (>50%)
                    let label = if avg_change < 0.10 { "few-cells-change" }
                        else if avg_change < 0.50 { "moderate-change" }
                        else { "most-cells-change" };
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("obs-consistent"), Neuron::symbol(label),
                        ]),
                        tv, decay: None,
                    });
                }

                // Per-pair cell-diff facts for template instantiation
                // (cell-diff pair-id row col from-color to-color)
                // Cap at 200 diff facts total to avoid bloat
                let mut diff_count = 0;
                for (idx, (i, o)) in pair_grids.iter().enumerate() {
                    if i.rows != o.rows || i.cols != o.cols { continue; }
                    let pair_id = format!("t{}i", idx);
                    for r in 0..i.rows {
                        for c in 0..i.cols {
                            if i.cells[r][c] != o.cells[r][c] {
                                if diff_count < 200 {
                                    stmts.push(Statement::Fact {
                                        neuron: Neuron::expression(vec![
                                            Neuron::symbol("cell-diff"),
                                            Neuron::symbol(&pair_id),
                                            Neuron::int_val(r as i64),
                                            Neuron::int_val(c as i64),
                                            Neuron::int_val(i.cells[r][c] as i64),
                                            Neuron::int_val(o.cells[r][c] as i64),
                                        ]),
                                        tv, decay: None,
                                    });
                                    diff_count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        // ── Universal relationship measurements ──────────────────────
        // Raw numeric facts about input→output relationships.
        // Domain-agnostic — QOR classification rules interpret these.

        // Size ratio: when output dimensions are exact integer multiples of input
        {
            let ratios: Vec<(usize, usize)> = pair_grids.iter()
                .filter_map(|(i, o)| {
                    if i.rows > 0 && i.cols > 0
                        && o.rows % i.rows == 0 && o.cols % i.cols == 0
                        && (o.rows > i.rows || o.cols > i.cols)
                    {
                        Some((o.rows / i.rows, o.cols / i.cols))
                    } else {
                        None
                    }
                })
                .collect();

            if ratios.len() == pair_grids.len() && !ratios.is_empty() {
                let first = ratios[0];
                if ratios.iter().all(|r| *r == first) {
                    // (obs-size-ratio row-factor col-factor)
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("obs-size-ratio"),
                            Neuron::int_val(first.0 as i64),
                            Neuron::int_val(first.1 as i64),
                        ]),
                        tv, decay: None,
                    });
                    // Same factor in both dimensions = uniform scale
                    if first.0 == first.1 {
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("obs-uniform-scale"),
                                Neuron::int_val(first.0 as i64),
                            ]),
                            tv, decay: None,
                        });
                        // Emit tile-block indices so tiling rules can enumerate blocks
                        // e.g., for factor 3: (tile-block 0), (tile-block 1), (tile-block 2)
                        for i in 0..first.0 {
                            stmts.push(Statement::Fact {
                                neuron: Neuron::expression(vec![
                                    Neuron::symbol("tile-block"),
                                    Neuron::int_val(i as i64),
                                ]),
                                tv, decay: None,
                            });
                        }

                        // Detect whether ALL input cells are non-zero (uniform tiling)
                        // vs some are zero (conditional/self-multiplication tiling).
                        let all_nonzero = pair_grids.iter().all(|(inp, _)| {
                            inp.cells.iter().all(|row| row.iter().all(|&c| c != 0))
                        });
                        if all_nonzero {
                            stmts.push(Statement::Fact {
                                neuron: Neuron::expression(vec![
                                    Neuron::symbol("obs-all-inputs-nonzero"),
                                ]),
                                tv, decay: None,
                            });
                        } else {
                            stmts.push(Statement::Fact {
                                neuron: Neuron::expression(vec![
                                    Neuron::symbol("obs-inputs-have-zeros"),
                                ]),
                                tv, decay: None,
                            });
                        }

                        // ── Tile with alternating horizontal reflection ──
                        // Check if even block-rows tile normally while odd block-rows
                        // tile with each row reversed (columns mirrored).
                        // Pattern: block-row 0 = normal, block-row 1 = reflect-h, etc.
                        let factor = first.0;
                        if factor >= 2 {
                            let alt_reflect_h = pair_grids.iter().all(|(input, output)| {
                                let ir = input.rows;
                                let ic = input.cols;
                                for br in 0..factor {
                                    let reflected = br % 2 == 1;
                                    for bc in 0..factor {
                                        for sr in 0..ir {
                                            for sc in 0..ic {
                                                let or = br * ir + sr;
                                                let oc = if reflected {
                                                    bc * ic + (ic - 1 - sc)
                                                } else {
                                                    bc * ic + sc
                                                };
                                                if or < output.rows && oc < output.cols {
                                                    if output.cells[or][oc] != input.cells[sr][sc] {
                                                        return false;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                true
                            });
                            if alt_reflect_h {
                                stmts.push(Statement::Fact {
                                    neuron: Neuron::expression(vec![
                                        Neuron::symbol("obs-tile-alt-reflect-h"),
                                    ]),
                                    tv, decay: None,
                                });
                            }
                        }
                    }
                }
            }
        }

        // Input-in-output: does the input pattern appear verbatim in the output?
        // Checks if input cells appear at (0,0) in the output (top-left embedding)
        {
            let all_embedded = pair_grids.iter().all(|(i, o)| {
                if o.rows < i.rows || o.cols < i.cols { return false; }
                for r in 0..i.rows {
                    for c in 0..i.cols {
                        if i.cells[r][c] != o.cells[r][c] { return false; }
                    }
                }
                true
            });
            if all_embedded {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("obs-consistent"), Neuron::symbol("input-embedded-in-output"),
                    ]),
                    tv, decay: None,
                });
            }
        }

        // Value set analysis: new values introduced, values removed
        {
            use std::collections::HashSet;
            let value_sets: Vec<(HashSet<u8>, HashSet<u8>)> = pair_grids.iter()
                .map(|(i, o)| {
                    let iv: HashSet<u8> = i.cells.iter().flat_map(|r| r.iter().copied()).filter(|&v| v > 0).collect();
                    let ov: HashSet<u8> = o.cells.iter().flat_map(|r| r.iter().copied()).filter(|&v| v > 0).collect();
                    (iv, ov)
                })
                .collect();

            // Values preserved (same set)
            if value_sets.iter().all(|(iv, ov)| iv == ov) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("obs-consistent"), Neuron::symbol("value-set-preserved"),
                    ]),
                    tv, decay: None,
                });
            }

            // New values introduced in output
            if value_sets.iter().all(|(iv, ov)| ov.iter().any(|v| !iv.contains(v))) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("obs-consistent"), Neuron::symbol("new-values-introduced"),
                    ]),
                    tv, decay: None,
                });
            }

            // Values removed from output
            if value_sets.iter().all(|(iv, ov)| iv.iter().any(|v| !ov.contains(v))) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("obs-consistent"), Neuron::symbol("values-removed"),
                    ]),
                    tv, decay: None,
                });
            }

            // Output is subset of input values
            if value_sets.iter().all(|(iv, ov)| ov.is_subset(iv)) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("obs-consistent"), Neuron::symbol("output-values-subset"),
                    ]),
                    tv, decay: None,
                });
            }
        }

        // ── Deep pattern observations (from Python analysis) ──────────
        // These help QOR recognize WHAT TYPE of problem it's looking at.
        {
            use std::collections::HashSet;

            // Same-size analysis: addition, removal, recoloring
            let same_size_pairs: Vec<&(Grid, Grid)> = pair_grids.iter()
                .filter(|(i, o)| i.rows == o.rows && i.cols == o.cols)
                .collect();

            if !same_size_pairs.is_empty() && same_size_pairs.len() == pair_grids.len() {
                // Analyze what types of cell changes occur
                let mut all_pure_add = true;   // only 0→nonzero
                let mut all_pure_remove = true; // only nonzero→0
                let mut all_pure_recolor = true; // only nonzero→different nonzero
                let mut fill_colors: HashSet<u8> = HashSet::new();
                let mut total_changes = 0usize;
                let mut total_cells = 0usize;

                for (i, o) in &same_size_pairs {
                    total_cells += i.rows * i.cols;
                    for r in 0..i.rows {
                        for c in 0..i.cols {
                            let iv = i.cells[r][c];
                            let ov = o.cells[r][c];
                            if iv != ov {
                                total_changes += 1;
                                if iv != 0 { all_pure_add = false; }
                                if ov != 0 { all_pure_remove = false; }
                                if iv == 0 || ov == 0 { all_pure_recolor = false; }
                                if ov != 0 { fill_colors.insert(ov); }
                            }
                        }
                    }
                }

                if total_changes > 0 {
                    if all_pure_add {
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("obs-consistent"), Neuron::symbol("pure-addition"),
                            ]),
                            tv, decay: None,
                        });
                    }
                    if all_pure_remove {
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("obs-consistent"), Neuron::symbol("pure-removal"),
                            ]),
                            tv, decay: None,
                        });
                    }
                    if all_pure_recolor {
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("obs-consistent"), Neuron::symbol("pure-recolor"),
                            ]),
                            tv, decay: None,
                        });
                    }
                    // Single-color fill: all added/changed cells use exactly one color
                    if fill_colors.len() == 1 {
                        let c = *fill_colors.iter().next().unwrap();
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("obs-fill-color"),
                                Neuron::int_val(c as i64),
                            ]),
                            tv, decay: None,
                        });
                    }
                    // Change magnitude: few / localized / moderate / global
                    let ratio = total_changes as f64 / total_cells.max(1) as f64;
                    if ratio < 0.15 {
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("obs-consistent"), Neuron::symbol("few-cells-change"),
                            ]),
                            tv, decay: None,
                        });
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("obs-consistent"), Neuron::symbol("localized-change"),
                            ]),
                            tv, decay: None,
                        });
                    } else if ratio < 0.25 {
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("obs-consistent"), Neuron::symbol("localized-change"),
                            ]),
                            tv, decay: None,
                        });
                    } else if ratio < 0.50 {
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("obs-consistent"), Neuron::symbol("moderate-change"),
                            ]),
                            tv, decay: None,
                        });
                    } else {
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("obs-consistent"), Neuron::symbol("global-change"),
                            ]),
                            tv, decay: None,
                        });
                    }
                }
            }

            // Output size classification
            let all_smaller = pair_grids.iter().all(|(i, o)| {
                (o.rows * o.cols) < (i.rows * i.cols)
            });
            let all_larger = pair_grids.iter().all(|(i, o)| {
                (o.rows * o.cols) > (i.rows * i.cols)
            });
            if all_smaller {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("obs-consistent"), Neuron::symbol("output-smaller"),
                    ]),
                    tv, decay: None,
                });
                // Output tiny (≤3x3)
                if pair_grids.iter().all(|(_, o)| o.rows <= 3 && o.cols <= 3) {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("obs-consistent"), Neuron::symbol("output-tiny"),
                        ]),
                        tv, decay: None,
                    });
                }
            }
            if all_larger {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("obs-consistent"), Neuron::symbol("output-larger"),
                    ]),
                    tv, decay: None,
                });
            }

            // Border/padding detection: does output = input surrounded by uniform border?
            if all_larger {
                let border_info: Vec<Option<(usize, usize, u8)>> = pair_grids.iter()
                    .map(|(i, o)| {
                        // Check if input appears in the center of output
                        let pad_top = (o.rows - i.rows) / 2;
                        let pad_left = (o.cols - i.cols) / 2;
                        // Check if input matches at (pad_top, pad_left)
                        let mut matches = true;
                        for r in 0..i.rows {
                            for c in 0..i.cols {
                                if r + pad_top >= o.rows || c + pad_left >= o.cols
                                    || i.cells[r][c] != o.cells[r + pad_top][c + pad_left]
                                {
                                    matches = false;
                                    break;
                                }
                            }
                            if !matches { break; }
                        }
                        if !matches { return None; }
                        // Check if border cells are all the same color
                        let mut border_color = None;
                        for r in 0..o.rows {
                            for c in 0..o.cols {
                                let in_input = r >= pad_top && r < pad_top + i.rows
                                    && c >= pad_left && c < pad_left + i.cols;
                                if !in_input {
                                    match border_color {
                                        None => border_color = Some(o.cells[r][c]),
                                        Some(bc) if bc != o.cells[r][c] => return None,
                                        _ => {}
                                    }
                                }
                            }
                        }
                        border_color.map(|bc| (pad_top, pad_left, bc))
                    })
                    .collect();

                if border_info.iter().all(|b| b.is_some()) && !border_info.is_empty() {
                    let first = border_info[0].unwrap();
                    if border_info.iter().all(|b| b.unwrap() == first) {
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("obs-border-added"),
                                Neuron::int_val(first.0 as i64), // pad_top
                                Neuron::int_val(first.1 as i64), // pad_left
                                Neuron::int_val(first.2 as i64), // border_color
                            ]),
                            tv, decay: None,
                        });
                    }
                }
            }

            // Separator analysis: emit obs-has-separator if any pair has separators
            let has_sep = pair_grids.iter().any(|(inp, _)| {
                // Check for full rows of uniform non-zero color
                let has_row_sep = (0..inp.rows).any(|r| {
                    let first = inp.cells[r][0];
                    first != 0 && inp.cells[r].iter().all(|&v| v == first)
                });
                // Check for full cols of uniform non-zero color
                let has_col_sep = (0..inp.cols).any(|c| {
                    let first = inp.cells[0][c];
                    first != 0 && (0..inp.rows).all(|r| inp.cells[r][c] == first)
                });
                has_row_sep || has_col_sep
            });
            if has_sep {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("obs-consistent"), Neuron::symbol("has-separator"),
                    ]),
                    tv, decay: None,
                });
            }

            // Separator binary operation detection: AND/XOR/OR between sub-grids
            // Detect if output = f(left_subgrid, right_subgrid) for vertical separators
            if has_sep {
                // Collect separator cols from each input grid
                let sep_col_sets: Vec<std::collections::HashSet<usize>> = pair_grids.iter().map(|(inp, _)| {
                    (0..inp.cols).filter(|&c| {
                        inp.rows > 0 && {
                            let first = inp.cells[0][c];
                            first > 0 && (0..inp.rows).all(|r| inp.cells[r][c] == first)
                        }
                    }).collect()
                }).collect();

                // Find separator columns common to ALL pairs (intersection)
                let common_seps: Vec<usize> = if sep_col_sets.is_empty() {
                    vec![]
                } else {
                    let mut common = sep_col_sets[0].clone();
                    for s in &sep_col_sets[1..] {
                        common = common.intersection(s).copied().collect();
                    }
                    let mut v: Vec<usize> = common.into_iter().collect();
                    v.sort();
                    v
                };

                // Single common vertical separator across all pairs
                if common_seps.len() == 1 {
                    let sep = common_seps[0];
                    let left_w = sep;
                    let right_start = sep + 1;

                    // Check all pairs have matching sub-grid widths and output size
                    let all_match = pair_grids.iter().all(|(inp, out)| {
                        let right_w = inp.cols.saturating_sub(right_start);
                        left_w == right_w
                            && out.rows == inp.rows
                            && out.cols == left_w
                    });

                    if all_match && left_w > 0 {
                        let mut and_val: Option<u8> = None;
                        let mut xor_val: Option<u8> = None;
                        let mut and_ok = true;
                        let mut xor_ok = true;

                        for (inp, out) in pair_grids.iter() {
                            for r in 0..out.rows {
                                for c in 0..left_w {
                                    let left = inp.cells[r][c];
                                    let right = inp.cells[r][right_start + c];
                                    let oval = out.cells[r][c];

                                    // AND: both nonzero → value, else 0
                                    if and_ok {
                                        if left != 0 && right != 0 {
                                            if oval == 0 { and_ok = false; }
                                            else {
                                                match and_val {
                                                    Some(v) if v != oval => { and_ok = false; }
                                                    None => { and_val = Some(oval); }
                                                    _ => {}
                                                }
                                            }
                                        } else if oval != 0 {
                                            and_ok = false;
                                        }
                                    }

                                    // XOR: exactly one nonzero → value, else 0
                                    if xor_ok {
                                        if (left != 0) ^ (right != 0) {
                                            if oval == 0 { xor_ok = false; }
                                            else {
                                                match xor_val {
                                                    Some(v) if v != oval => { xor_ok = false; }
                                                    None => { xor_val = Some(oval); }
                                                    _ => {}
                                                }
                                            }
                                        } else if oval != 0 {
                                            xor_ok = false;
                                        }
                                    }
                                }
                            }
                        }

                        if and_ok && and_val.is_some() {
                            let v = and_val.unwrap() as i64;
                            stmts.push(Statement::Fact {
                                neuron: Neuron::expression(vec![
                                    Neuron::symbol("obs-separator-and"), Neuron::int_val(v),
                                ]),
                                tv, decay: None,
                            });
                        }
                        if xor_ok && xor_val.is_some() {
                            let v = xor_val.unwrap() as i64;
                            stmts.push(Statement::Fact {
                                neuron: Neuron::expression(vec![
                                    Neuron::symbol("obs-separator-xor"), Neuron::int_val(v),
                                ]),
                                tv, decay: None,
                            });
                        }
                    }
                }
            }

            // Periodic tiling detection: output rows = cyclic repeat of input rows
            // with consistent color remap. Uses minimum period (not input height).
            // Emits obs-periodic-tile + output-row facts.
            {
                let all_larger_same_w = pair_grids.iter().all(|(i, o)| {
                    o.rows > i.rows && o.cols == i.cols
                });
                if all_larger_same_w {
                    // Find minimum row period for each input grid
                    let periods: Vec<usize> = pair_grids.iter().map(|(inp, _)| {
                        let mut period = inp.rows;
                        for p in 1..inp.rows {
                            let is_periodic = (0..inp.rows).all(|r| {
                                let ref_r = r % p;
                                (0..inp.cols).all(|c| inp.cells[r][c] == inp.cells[ref_r][c])
                            });
                            if is_periodic { period = p; break; }
                        }
                        period
                    }).collect();

                    // Check if output = periodic tile of input with color remap
                    let mut remap: Option<(u8, u8)> = None; // (from, to)
                    let mut remap_ok = true;
                    let mut tile_ok = true;

                    for (idx, (inp, out)) in pair_grids.iter().enumerate() {
                        let period = periods[idx];
                        for r in 0..out.rows {
                            let src_r = r % period;
                            for c in 0..out.cols {
                                let iv = inp.cells[src_r][c];
                                let ov = out.cells[r][c];
                                if iv == ov { continue; } // identical cell is fine
                                if iv == 0 && ov == 0 { continue; }
                                // Different — check if consistent remap
                                if iv != 0 && ov != 0 {
                                    match remap {
                                        Some((f, t)) if f == iv && t == ov => {} // consistent
                                        Some(_) => { remap_ok = false; }
                                        None => { remap = Some((iv, ov)); }
                                    }
                                } else {
                                    tile_ok = false;
                                    break;
                                }
                            }
                            if !tile_ok { break; }
                        }
                        if !tile_ok { break; }
                    }

                    if tile_ok && remap_ok && remap.is_some() {
                        let (from, to) = remap.unwrap();
                        let out_h = pair_grids[0].1.rows;
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("obs-periodic-tile"),
                                Neuron::int_val(out_h as i64),
                                Neuron::int_val(from as i64),
                                Neuron::int_val(to as i64),
                            ]),
                            tv, decay: None,
                        });
                        // Emit row indices so QOR rules can iterate output rows
                        for r in 0..out_h {
                            stmts.push(Statement::Fact {
                                neuron: Neuron::expression(vec![
                                    Neuron::symbol("obs-row-idx"),
                                    Neuron::int_val(r as i64),
                                ]),
                                tv, decay: None,
                            });
                        }
                    }
                }
            }

            // Object count analysis
            let obj_counts: Vec<(usize, usize)> = pair_grids.iter()
                .map(|(i, o)| {
                    let ic = i.objects().len();
                    let oc = o.objects().len();
                    (ic, oc)
                })
                .collect();
            if !obj_counts.is_empty() {
                let all_same_count = obj_counts.iter().all(|(ic, oc)| ic == oc);
                let all_more = obj_counts.iter().all(|(ic, oc)| oc > ic);
                let all_fewer = obj_counts.iter().all(|(ic, oc)| oc < ic);
                if all_same_count {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("obs-consistent"), Neuron::symbol("same-object-count"),
                        ]),
                        tv, decay: None,
                    });
                }
                if all_more {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("obs-consistent"), Neuron::symbol("more-objects"),
                        ]),
                        tv, decay: None,
                    });
                }
                if all_fewer {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("obs-consistent"), Neuron::symbol("fewer-objects"),
                        ]),
                        tv, decay: None,
                    });
                }
            }
        }

        // ── Reduction-specific observations ───────────────────────────
        {
            use std::collections::HashSet;

            // Output-is-subgrid: output appears verbatim somewhere in input
            let all_subgrid = pair_grids.iter().all(|(inp, out)| {
                if out.rows > inp.rows || out.cols > inp.cols { return false; }
                for sr in 0..=(inp.rows - out.rows) {
                    for sc in 0..=(inp.cols - out.cols) {
                        let mut ok = true;
                        'check: for r in 0..out.rows {
                            for c in 0..out.cols {
                                if inp.cells[sr + r][sc + c] != out.cells[r][c] {
                                    ok = false;
                                    break 'check;
                                }
                            }
                        }
                        if ok { return true; }
                    }
                }
                false
            });
            if all_subgrid && pair_grids.iter().all(|(i, o)| i.rows > o.rows || i.cols > o.cols) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("obs-consistent"), Neuron::symbol("output-is-subgrid"),
                    ]),
                    tv, decay: None,
                });
            }

            // Output-binary: output has at most 2 unique values
            let all_binary = pair_grids.iter().all(|(_, out)| {
                let vals: HashSet<u8> = out.cells.iter().flat_map(|r| r.iter().copied()).collect();
                vals.len() <= 2
            });
            if all_binary {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("obs-consistent"), Neuron::symbol("output-binary"),
                    ]),
                    tv, decay: None,
                });
            }

            // Output-1-row or output-1-col
            if pair_grids.iter().all(|(_, o)| o.rows == 1) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("obs-consistent"), Neuron::symbol("output-1-row"),
                    ]),
                    tv, decay: None,
                });
            }
            if pair_grids.iter().all(|(_, o)| o.cols == 1) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("obs-consistent"), Neuron::symbol("output-1-col"),
                    ]),
                    tv, decay: None,
                });
            }

            // Position-dependent recolor: same-size, colors change, but mapping
            // is NOT consistent (same input color maps to different output colors
            // at different positions)
            let all_pos_dep = pair_grids.iter().all(|(inp, out)| {
                if inp.rows != out.rows || inp.cols != out.cols { return false; }
                let mut map: std::collections::HashMap<u8, HashSet<u8>> = std::collections::HashMap::new();
                for r in 0..inp.rows {
                    for c in 0..inp.cols {
                        let iv = inp.cells[r][c];
                        let ov = out.cells[r][c];
                        map.entry(iv).or_default().insert(ov);
                    }
                }
                // Position-dependent if any input color maps to multiple output colors
                map.values().any(|outs| outs.len() > 1)
            });
            if all_pos_dep {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("obs-consistent"), Neuron::symbol("position-dependent-recolor"),
                    ]),
                    tv, decay: None,
                });
            }

            // Mixed-size: training pairs have different size relationships
            if pair_grids.len() >= 2 {
                let rels: Vec<u8> = pair_grids.iter().map(|(i, o)| {
                    let it = i.rows * i.cols;
                    let ot = o.rows * o.cols;
                    if ot > it { 1 } else if ot < it { 2 } else { 0 }
                }).collect();
                let has_expand = rels.iter().any(|&r| r == 1);
                let has_reduce = rels.iter().any(|&r| r == 2);
                let has_same = rels.iter().any(|&r| r == 0);
                if (has_expand && has_reduce) || (has_expand && has_same && !pair_grids.iter().all(|(i,o)| i.rows*i.cols <= o.rows*o.cols))
                    || (has_reduce && has_same && !pair_grids.iter().all(|(i,o)| i.rows*i.cols >= o.rows*o.cols))
                {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("obs-consistent"), Neuron::symbol("mixed-size"),
                        ]),
                        tv, decay: None,
                    });
                }
            }

            // Same-cell-count: same number of non-zero cells in input and output
            let all_same_nz = pair_grids.iter().all(|(inp, out)| {
                let in_nz: usize = inp.cells.iter().flat_map(|r| r.iter()).filter(|&&v| v != 0).count();
                let out_nz: usize = out.cells.iter().flat_map(|r| r.iter()).filter(|&&v| v != 0).count();
                in_nz == out_nz
            });
            if all_same_nz {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("obs-consistent"), Neuron::symbol("same-cell-count"),
                    ]),
                    tv, decay: None,
                });
            }

            // Output-fixed-height / output-fixed-width: all outputs have same dimension
            if pair_grids.len() >= 2 {
                let out_heights: Vec<usize> = pair_grids.iter().map(|(_, o)| o.rows).collect();
                let out_widths: Vec<usize> = pair_grids.iter().map(|(_, o)| o.cols).collect();
                if out_heights.iter().all(|h| *h == out_heights[0]) && pair_grids.iter().any(|(i, o)| i.rows != o.rows) {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("obs-output-fixed-height"),
                            Neuron::int_val(out_heights[0] as i64),
                        ]),
                        tv, decay: None,
                    });
                }
                if out_widths.iter().all(|w| *w == out_widths[0]) && pair_grids.iter().any(|(i, o)| i.cols != o.cols) {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("obs-output-fixed-width"),
                            Neuron::int_val(out_widths[0] as i64),
                        ]),
                        tv, decay: None,
                    });
                }
            }

            // Fixed output size: all outputs have exact same dimensions
            if pair_grids.len() >= 2 {
                let out_sizes: Vec<(usize, usize)> = pair_grids.iter()
                    .map(|(_, o)| (o.rows, o.cols)).collect();
                if out_sizes.iter().all(|s| *s == out_sizes[0])
                    && pair_grids.iter().any(|(i, o)| i.rows != o.rows || i.cols != o.cols)
                {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("obs-output-fixed-size"),
                            Neuron::int_val(out_sizes[0].0 as i64),
                            Neuron::int_val(out_sizes[0].1 as i64),
                        ]),
                        tv, decay: None,
                    });
                }
            }

            // Same-height reduction: output has same height, width reduced
            if pair_grids.iter().all(|(i, o)| i.rows == o.rows && o.cols < i.cols) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("obs-consistent"), Neuron::symbol("same-height-reduction"),
                    ]),
                    tv, decay: None,
                });
            }

            // Same-width reduction: output has same width, height reduced
            if pair_grids.iter().all(|(i, o)| i.cols == o.cols && o.rows < i.rows) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("obs-consistent"), Neuron::symbol("same-width-reduction"),
                    ]),
                    tv, decay: None,
                });
            }

            // Same-width expansion: output has same width, height grows
            if pair_grids.iter().all(|(i, o)| i.cols == o.cols && o.rows > i.rows) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("obs-consistent"), Neuron::symbol("same-width-expansion"),
                    ]),
                    tv, decay: None,
                });
            }

            // Output-padded: output = input + constant offset in each dimension
            if pair_grids.len() >= 2 {
                let offsets: Vec<(i64, i64)> = pair_grids.iter()
                    .map(|(i, o)| (o.rows as i64 - i.rows as i64, o.cols as i64 - i.cols as i64))
                    .collect();
                if offsets.iter().all(|o| *o == offsets[0]) && (offsets[0].0 != 0 || offsets[0].1 != 0) {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("obs-output-padded"),
                            Neuron::int_val(offsets[0].0),
                            Neuron::int_val(offsets[0].1),
                        ]),
                        tv, decay: None,
                    });
                }
            }

            // Same-height expansion: output has same height, width grows
            if pair_grids.iter().all(|(i, o)| i.rows == o.rows && o.cols > i.cols) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("obs-consistent"), Neuron::symbol("same-height-expansion"),
                    ]),
                    tv, decay: None,
                });
            }

            // Content-dependent expansion: clean ratio but NOT simple scale/tile
            // Each input cell likely maps to a block whose content depends on cell value
            if pair_grids.iter().all(|(i, o)| {
                i.rows > 0 && i.cols > 0 && o.rows > i.rows && o.cols > i.cols
                    && o.rows % i.rows == 0 && o.cols % i.cols == 0
            }) {
                let ratios: Vec<(usize, usize)> = pair_grids.iter()
                    .map(|(i, o)| (o.rows / i.rows, o.cols / i.cols))
                    .collect();
                if ratios.iter().all(|r| *r == ratios[0]) {
                    // Check if it's NOT simple scale-up (at least one cell differs)
                    let is_scale = pair_grids.iter().all(|(i, o)| {
                        let (fr, fc) = ratios[0];
                        (0..o.rows).all(|r| (0..o.cols).all(|c| {
                            o.cells[r][c] == i.cells[r / fr][c / fc]
                        }))
                    });
                    if !is_scale {
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("obs-content-dependent-expansion"),
                                Neuron::int_val(ratios[0].0 as i64),
                                Neuron::int_val(ratios[0].1 as i64),
                            ]),
                            tv, decay: None,
                        });
                    }
                }
            }

            // Content-dependent reduction: clean ratio, output smaller
            if pair_grids.iter().all(|(i, o)| {
                o.rows > 0 && o.cols > 0 && i.rows > o.rows && i.cols > o.cols
                    && i.rows % o.rows == 0 && i.cols % o.cols == 0
            }) {
                let ratios: Vec<(usize, usize)> = pair_grids.iter()
                    .map(|(i, o)| (i.rows / o.rows, i.cols / o.cols))
                    .collect();
                if ratios.iter().all(|r| *r == ratios[0]) {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("obs-content-dependent-reduction"),
                            Neuron::int_val(ratios[0].0 as i64),
                            Neuron::int_val(ratios[0].1 as i64),
                        ]),
                        tv, decay: None,
                    });
                }
            }

            // Content-preserving: output contains exactly the same non-zero values
            // (possibly rearranged)
            let all_content_preserved = pair_grids.iter().all(|(inp, out)| {
                let mut in_vals: Vec<u8> = inp.cells.iter().flat_map(|r| r.iter().copied()).filter(|&v| v != 0).collect();
                let mut out_vals: Vec<u8> = out.cells.iter().flat_map(|r| r.iter().copied()).filter(|&v| v != 0).collect();
                in_vals.sort();
                out_vals.sort();
                in_vals == out_vals
            });
            if all_content_preserved && pair_grids.iter().any(|(i, o)| {
                (0..i.rows.min(o.rows)).any(|r| (0..i.cols.min(o.cols)).any(|c| i.cells[r][c] != o.cells[r][c]))
            }) {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("obs-consistent"), Neuron::symbol("content-rearranged"),
                    ]),
                    tv, decay: None,
                });
            }
        }

        stmts
    }

    /// BFS from border: returns a 2D bool grid where `true` means the zero-valued
    /// cell at (r,c) is reachable from the grid border following zero-valued cells.
    /// Non-zero cells are never reachable. Used for enclosed-cell detection.
    fn border_reachable_zeros(&self) -> Vec<Vec<bool>> {
        let mut reachable = vec![vec![false; self.cols]; self.rows];
        let mut queue = std::collections::VecDeque::new();
        for r in 0..self.rows {
            for c in 0..self.cols {
                if self.cells[r][c] == 0
                    && (r == 0 || r == self.rows - 1 || c == 0 || c == self.cols - 1)
                {
                    reachable[r][c] = true;
                    queue.push_back((r, c));
                }
            }
        }
        while let Some((r, c)) = queue.pop_front() {
            for (dr, dc) in [(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr >= 0 && nc >= 0 {
                    let (nr, nc) = (nr as usize, nc as usize);
                    if nr < self.rows && nc < self.cols
                        && !reachable[nr][nc] && self.cells[nr][nc] == 0
                    {
                        reachable[nr][nc] = true;
                        queue.push_back((nr, nc));
                    }
                }
            }
        }
        reachable
    }

    /// Get bounding box of all non-zero cells. Returns None if no non-zero cells.
    fn content_bbox(grid: &Grid) -> Option<(usize, usize, usize, usize)> {
        let mut min_r = grid.rows;
        let mut min_c = grid.cols;
        let mut max_r = 0usize;
        let mut max_c = 0usize;
        let mut found = false;
        for r in 0..grid.rows {
            for c in 0..grid.cols {
                if grid.cells[r][c] != 0 {
                    found = true;
                    min_r = min_r.min(r);
                    min_c = min_c.min(c);
                    max_r = max_r.max(r);
                    max_c = max_c.max(c);
                }
            }
        }
        if found { Some((min_r, min_c, max_r, max_c)) } else { None }
    }

    /// Detect a uniform shift between two same-size grids.
    /// Returns Some((dr, dc)) if shifting all non-zero cells by (dr,dc) matches output.
    /// Tries all candidate positions for the anchor color to avoid false anchors.
    fn detect_shift(input: &Grid, output: &Grid) -> Option<(i32, i32)> {
        if input.rows != output.rows || input.cols != output.cols {
            return None;
        }

        // Find first non-zero cell in input as anchor
        let mut in_first = None;
        let mut in_color = 0u8;
        'outer: for r in 0..input.rows {
            for c in 0..input.cols {
                if input.cells[r][c] != 0 {
                    in_first = Some((r as i32, c as i32));
                    in_color = input.cells[r][c];
                    break 'outer;
                }
            }
        }
        let (ir, ic) = in_first?;

        // Try ALL positions in output with same color as candidate anchors
        for or in 0..output.rows {
            for oc in 0..output.cols {
                if output.cells[or][oc] != in_color { continue; }
                let dr = or as i32 - ir;
                let dc = oc as i32 - ic;
                if dr == 0 && dc == 0 { continue; }
                if Self::verify_shift(input, output, dr, dc) {
                    return Some((dr, dc));
                }
            }
        }
        None
    }

    /// Verify that a given (dr, dc) shift exactly maps input to output.
    fn verify_shift(input: &Grid, output: &Grid, dr: i32, dc: i32) -> bool {
        // All non-zero input cells must match output at shifted position
        for r in 0..input.rows {
            for c in 0..input.cols {
                if input.cells[r][c] != 0 {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr < 0 || nr >= input.rows as i32 || nc < 0 || nc >= input.cols as i32 {
                        return false;
                    }
                    if output.cells[nr as usize][nc as usize] != input.cells[r][c] {
                        return false;
                    }
                }
            }
        }
        // All non-zero output cells must have corresponding input
        for r in 0..output.rows {
            for c in 0..output.cols {
                if output.cells[r][c] != 0 {
                    let sr = r as i32 - dr;
                    let sc = c as i32 - dc;
                    if sr < 0 || sr >= input.rows as i32 || sc < 0 || sc >= input.cols as i32 {
                        return false;
                    }
                    if input.cells[sr as usize][sc as usize] != output.cells[r][c] {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Detect if output is a crop of input. Returns Some((r0, c0)) offset.
    /// Uses row-hash shortcircuit to avoid O(n⁴) brute force.
    fn detect_crop(input: &Grid, output: &Grid) -> Option<(usize, usize)> {
        if output.rows > input.rows || output.cols > input.cols
            || output.rows == 0 || output.cols == 0
        {
            return None;
        }

        // Pre-compute simple hash of each output row for fast rejection
        let out_row_hashes: Vec<u64> = (0..output.rows)
            .map(|r| {
                let mut h: u64 = 0;
                for c in 0..output.cols {
                    h = h.wrapping_mul(31).wrapping_add(output.cells[r][c] as u64);
                }
                h
            })
            .collect();

        for r0 in 0..=(input.rows - output.rows) {
            'offset: for c0 in 0..=(input.cols - output.cols) {
                // Quick hash check on first row before full comparison
                let mut h: u64 = 0;
                for c in 0..output.cols {
                    h = h.wrapping_mul(31).wrapping_add(input.cells[r0][c0 + c] as u64);
                }
                if h != out_row_hashes[0] {
                    continue;
                }

                // Full comparison
                for r in 0..output.rows {
                    for c in 0..output.cols {
                        if input.cells[r0 + r][c0 + c] != output.cells[r][c] {
                            continue 'offset;
                        }
                    }
                }
                return Some((r0, c0));
            }
        }
        None
    }

    /// Emit predict-cell facts for every cell in a transformed grid.
    fn grid_to_predictions(grid: &Grid) -> Vec<Statement> {
        let tv = Some(TruthValue::new(0.95, 0.95));
        let mut stmts = Vec::new();
        for r in 0..grid.rows {
            for c in 0..grid.cols {
                if grid.cells[r][c] != 0 {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("predict-cell"),
                            Neuron::int_val(r as i64),
                            Neuron::int_val(c as i64),
                            Neuron::int_val(grid.cells[r][c] as i64),
                        ]),
                        tv, decay: None,
                    });
                }
            }
        }
        stmts
    }

    /// Predict output by reflecting horizontally.
    pub fn predict_reflect_h(input: &Grid) -> Vec<Statement> {
        Self::grid_to_predictions(&input.reflect_h())
    }

    /// Predict output by reflecting vertically.
    pub fn predict_reflect_v(input: &Grid) -> Vec<Statement> {
        Self::grid_to_predictions(&input.reflect_v())
    }

    /// Predict output by rotating 90 degrees clockwise.
    pub fn predict_rotate_90(input: &Grid) -> Vec<Statement> {
        Self::grid_to_predictions(&input.rotate_90())
    }

    /// Predict output by rotating 180 degrees.
    pub fn predict_rotate_180(input: &Grid) -> Vec<Statement> {
        Self::grid_to_predictions(&input.rotate_180())
    }

    /// Predict output by rotating 270 degrees clockwise.
    pub fn predict_rotate_270(input: &Grid) -> Vec<Statement> {
        Self::grid_to_predictions(&input.rotate_270())
    }

    /// Predict output by transposing.
    pub fn predict_transpose(input: &Grid) -> Vec<Statement> {
        Self::grid_to_predictions(&input.transpose())
    }

    /// Predict output by scaling up by given factor.
    pub fn predict_scale_up(input: &Grid, factor: usize) -> Vec<Statement> {
        let tv = Some(TruthValue::new(0.95, 0.95));
        let mut stmts = Vec::new();
        for r in 0..input.rows {
            for c in 0..input.cols {
                if input.cells[r][c] != 0 {
                    for dr in 0..factor {
                        for dc in 0..factor {
                            stmts.push(Statement::Fact {
                                neuron: Neuron::expression(vec![
                                    Neuron::symbol("predict-cell"),
                                    Neuron::int_val((r * factor + dr) as i64),
                                    Neuron::int_val((c * factor + dc) as i64),
                                    Neuron::int_val(input.cells[r][c] as i64),
                                ]),
                                tv, decay: None,
                            });
                        }
                    }
                }
            }
        }
        stmts
    }

    /// Predict output by cropping to content bounding box.
    pub fn predict_crop_to_bbox(input: &Grid) -> Vec<Statement> {
        if let Some((min_r, min_c, max_r, max_c)) = Self::content_bbox(input) {
            let h = max_r - min_r + 1;
            let w = max_c - min_c + 1;
            if let Some(cropped) = input.crop(min_r, min_c, h, w) {
                return Self::grid_to_predictions(&cropped);
            }
        }
        Vec::new()
    }

    /// Predict output by shifting all non-zero cells by (dr, dc).
    pub fn predict_shift(input: &Grid, dr: i32, dc: i32) -> Vec<Statement> {
        let tv = Some(TruthValue::new(0.95, 0.95));
        let mut stmts = Vec::new();
        for r in 0..input.rows {
            for c in 0..input.cols {
                if input.cells[r][c] != 0 {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr >= 0 && nr < input.rows as i32 && nc >= 0 && nc < input.cols as i32 {
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("predict-cell"),
                                Neuron::int_val(nr as i64),
                                Neuron::int_val(nc as i64),
                                Neuron::int_val(input.cells[r][c] as i64),
                            ]),
                            tv, decay: None,
                        });
                    }
                }
            }
        }
        stmts
    }

    /// Predict output by cropping at given offset.
    pub fn predict_crop(input: &Grid, r0: i32, c0: i32, out_rows: usize, out_cols: usize) -> Vec<Statement> {
        let tv = Some(TruthValue::new(0.95, 0.95));
        let mut stmts = Vec::new();
        for r in 0..out_rows {
            for c in 0..out_cols {
                let sr = r as i32 + r0;
                let sc = c as i32 + c0;
                let color = if sr >= 0 && sr < input.rows as i32 && sc >= 0 && sc < input.cols as i32 {
                    input.cells[sr as usize][sc as usize]
                } else {
                    0
                };
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("predict-cell"),
                        Neuron::int_val(r as i64),
                        Neuron::int_val(c as i64),
                        Neuron::int_val(color as i64),
                    ]),
                    tv, decay: None,
                });
            }
        }
        stmts
    }

    /// Predict output by horizontal symmetry completion (mirror existing content).
    pub fn predict_symmetry_h(input: &Grid) -> Vec<Statement> {
        let reflected = input.reflect_h();
        let tv = Some(TruthValue::new(0.90, 0.90));
        let mut stmts = Vec::new();
        for r in 0..input.rows {
            for c in 0..input.cols {
                // Keep original non-zero, fill with reflected where original is zero
                let color = if input.cells[r][c] != 0 {
                    input.cells[r][c]
                } else {
                    reflected.cells[r][c]
                };
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("predict-cell"),
                        Neuron::int_val(r as i64),
                        Neuron::int_val(c as i64),
                        Neuron::int_val(color as i64),
                    ]),
                    tv, decay: None,
                });
            }
        }
        stmts
    }

    /// Predict output by vertical symmetry completion.
    pub fn predict_symmetry_v(input: &Grid) -> Vec<Statement> {
        let reflected = input.reflect_v();
        let tv = Some(TruthValue::new(0.90, 0.90));
        let mut stmts = Vec::new();
        for r in 0..input.rows {
            for c in 0..input.cols {
                let color = if input.cells[r][c] != 0 {
                    input.cells[r][c]
                } else {
                    reflected.cells[r][c]
                };
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("predict-cell"),
                        Neuron::int_val(r as i64),
                        Neuron::int_val(c as i64),
                        Neuron::int_val(color as i64),
                    ]),
                    tv, decay: None,
                });
            }
        }
        stmts
    }

    /// Predict output by flood-filling enclosed zero regions with given color.
    pub fn predict_flood_fill(input: &Grid, fill_color: u8) -> Vec<Statement> {
        // Find border-reachable zero cells
        let mut border_reachable = vec![vec![false; input.cols]; input.rows];
        let mut queue = std::collections::VecDeque::new();
        for r in 0..input.rows {
            for c in 0..input.cols {
                if input.cells[r][c] == 0
                    && (r == 0 || r == input.rows - 1 || c == 0 || c == input.cols - 1)
                    && !border_reachable[r][c]
                {
                    border_reachable[r][c] = true;
                    queue.push_back((r, c));
                }
            }
        }
        while let Some((r, c)) = queue.pop_front() {
            for (dr, dc) in [(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr >= 0 && nc >= 0 {
                    let (nr, nc) = (nr as usize, nc as usize);
                    if nr < input.rows && nc < input.cols
                        && !border_reachable[nr][nc]
                        && input.cells[nr][nc] == 0
                    {
                        border_reachable[nr][nc] = true;
                        queue.push_back((nr, nc));
                    }
                }
            }
        }

        let tv = Some(TruthValue::new(0.90, 0.90));
        let mut stmts = Vec::new();
        for r in 0..input.rows {
            for c in 0..input.cols {
                let color = if input.cells[r][c] != 0 {
                    input.cells[r][c]
                } else if !border_reachable[r][c] {
                    fill_color // enclosed → fill
                } else {
                    0
                };
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("predict-cell"),
                        Neuron::int_val(r as i64),
                        Neuron::int_val(c as i64),
                        Neuron::int_val(color as i64),
                    ]),
                    tv, decay: None,
                });
            }
        }
        stmts
    }

    // ── Color remap detection and prediction ────────────────────────────

    /// Detect consistent color remapping between two grids.
    /// Returns a map of (from_color -> to_color) if every cell maps consistently.
    pub fn detect_color_remap(input: &Grid, output: &Grid) -> Option<std::collections::HashMap<u8, u8>> {
        if input.rows != output.rows || input.cols != output.cols {
            return None;
        }
        let mut remap: std::collections::HashMap<u8, u8> = std::collections::HashMap::new();
        for r in 0..input.rows {
            for c in 0..input.cols {
                let ic = input.cells[r][c];
                let oc = output.cells[r][c];
                if let Some(&existing) = remap.get(&ic) {
                    if existing != oc {
                        return None; // Inconsistent mapping
                    }
                } else {
                    remap.insert(ic, oc);
                }
            }
        }
        // Must have at least one actual remap (not all identity)
        if remap.iter().all(|(k, v)| k == v) {
            return None;
        }
        Some(remap)
    }

    /// Predict output by applying a color remap to the input.
    pub fn predict_color_remap(input: &Grid, remap: &std::collections::HashMap<u8, u8>) -> Vec<Statement> {
        let tv = Some(TruthValue::new(0.95, 0.95));
        let mut stmts = Vec::new();
        for r in 0..input.rows {
            for c in 0..input.cols {
                let color = input.cells[r][c];
                let mapped = remap.get(&color).copied().unwrap_or(color);
                if mapped != 0 {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("predict-cell"),
                            Neuron::int_val(r as i64),
                            Neuron::int_val(c as i64),
                            Neuron::int_val(mapped as i64),
                        ]),
                        tv, decay: None,
                    });
                }
            }
        }
        stmts
    }

    /// Apply a named transform to a grid, returning the transformed grid.
    pub fn apply_transform(&self, transform: &str) -> Option<Grid> {
        match transform {
            "reflect-h" => Some(self.reflect_h()),
            "reflect-v" => Some(self.reflect_v()),
            "rotate-90" => Some(self.rotate_90()),
            "rotate-180" => Some(self.rotate_180()),
            "rotate-270" => Some(self.rotate_270()),
            "transpose" => Some(self.transpose()),
            "crop-to-bbox" => {
                let (min_r, min_c, max_r, max_c) = Self::content_bbox(self)?;
                self.crop(min_r, min_c, max_r - min_r + 1, max_c - min_c + 1)
            }
            "gravity-down" => Some(self.apply_gravity_down()),
            _ => None,
        }
    }

    /// Apply gravity (cells fall down) and return the resulting grid.
    pub fn apply_gravity_down(&self) -> Grid {
        let mut cells = vec![vec![0u8; self.cols]; self.rows];
        for c in 0..self.cols {
            let mut col_colors: Vec<u8> = Vec::new();
            for r in 0..self.rows {
                if self.cells[r][c] != 0 {
                    col_colors.push(self.cells[r][c]);
                }
            }
            let empty = self.rows - col_colors.len();
            for (i, &color) in col_colors.iter().enumerate() {
                cells[empty + i][c] = color;
            }
        }
        Grid { rows: self.rows, cols: self.cols, cells }
    }

    /// Apply a color remap to a grid, returning the remapped grid.
    pub fn apply_color_remap(&self, remap: &std::collections::HashMap<u8, u8>) -> Grid {
        let cells: Vec<Vec<u8>> = self.cells.iter().map(|row| {
            row.iter().map(|&c| remap.get(&c).copied().unwrap_or(c)).collect()
        }).collect();
        Grid { rows: self.rows, cols: self.cols, cells }
    }

    /// Predict the output grid by applying gravity (cells fall down in each column).
    /// Returns predict-cell statements for the test input.
    pub fn predict_gravity_down(input: &Grid) -> Vec<Statement> {
        let tv = Some(TruthValue::new(0.90, 0.90));
        let mut stmts = Vec::new();

        for c in 0..input.cols {
            // Collect non-zero cells in this column, top to bottom
            let mut col_colors: Vec<u8> = Vec::new();
            for r in 0..input.rows {
                if input.cells[r][c] != 0 {
                    col_colors.push(input.cells[r][c]);
                }
            }
            // Stack at bottom of column
            let empty = input.rows - col_colors.len();
            for (i, &color) in col_colors.iter().enumerate() {
                let r = empty + i;
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("predict-cell"),
                        Neuron::int_val(r as i64),
                        Neuron::int_val(c as i64),
                        Neuron::int_val(color as i64),
                    ]),
                    tv, decay: None,
                });
            }
        }

        stmts
    }

    // ── Row/Col/Cross Fill Detection ───────────────────────────────────

    /// Detect row-fill: for each non-zero marker in input, the entire row
    /// in output is filled with that marker's color. Requires sparse markers
    /// (not already full rows) and actual changes from input to output.
    fn detect_row_fill(input: &Grid, output: &Grid) -> bool {
        if input.rows == 0 || input.cols == 0 { return false; }
        // Build per-row color set from input markers
        let mut row_colors: Vec<Vec<u8>> = vec![Vec::new(); input.rows];
        let mut any_marker = false;
        let mut total_markers = 0usize;
        for r in 0..input.rows {
            for c in 0..input.cols {
                if input.cells[r][c] != 0 {
                    any_marker = true;
                    total_markers += 1;
                    if !row_colors[r].contains(&input.cells[r][c]) {
                        row_colors[r].push(input.cells[r][c]);
                    }
                }
            }
        }
        if !any_marker { return false; }

        // Sparsity check: markers should be sparse (< 50% of cells)
        // Otherwise it's not really "filling" — input is already mostly full
        let total_cells = input.rows * input.cols;
        if total_markers * 2 > total_cells { return false; }

        // Must have at least one row with an empty cell that gets filled
        let mut any_fill = false;

        // Verify: every output cell matches expected row fill
        for r in 0..output.rows {
            for c in 0..output.cols {
                if row_colors[r].is_empty() {
                    // No marker in this row — output should be 0
                    if output.cells[r][c] != 0 { return false; }
                } else if row_colors[r].len() == 1 {
                    // Single color fills the row
                    if output.cells[r][c] != row_colors[r][0] { return false; }
                    if input.cells[r][c] == 0 { any_fill = true; }
                } else {
                    // Multiple markers — output should be one of the marker colors
                    if !row_colors[r].contains(&output.cells[r][c]) { return false; }
                    if input.cells[r][c] == 0 { any_fill = true; }
                }
            }
        }
        any_fill
    }

    /// Detect col-fill: for each non-zero marker in input, the entire column
    /// in output is filled with that marker's color. Requires sparse markers.
    fn detect_col_fill(input: &Grid, output: &Grid) -> bool {
        if input.rows == 0 || input.cols == 0 { return false; }
        let mut col_colors: Vec<Vec<u8>> = vec![Vec::new(); input.cols];
        let mut any_marker = false;
        let mut total_markers = 0usize;
        for r in 0..input.rows {
            for c in 0..input.cols {
                if input.cells[r][c] != 0 {
                    any_marker = true;
                    total_markers += 1;
                    if !col_colors[c].contains(&input.cells[r][c]) {
                        col_colors[c].push(input.cells[r][c]);
                    }
                }
            }
        }
        if !any_marker { return false; }

        // Sparsity check
        let total_cells = input.rows * input.cols;
        if total_markers * 2 > total_cells { return false; }

        let mut any_fill = false;
        for r in 0..output.rows {
            for c in 0..output.cols {
                if col_colors[c].is_empty() {
                    if output.cells[r][c] != 0 { return false; }
                } else if col_colors[c].len() == 1 {
                    if output.cells[r][c] != col_colors[c][0] { return false; }
                    if input.cells[r][c] == 0 { any_fill = true; }
                } else {
                    if !col_colors[c].contains(&output.cells[r][c]) { return false; }
                    if input.cells[r][c] == 0 { any_fill = true; }
                }
            }
        }
        any_fill
    }

    /// Detect cross-fill: each non-zero marker paints its row AND column.
    /// At intersections, the output cell matches one of the contributing colors.
    /// Requires sparse markers and actual fill changes.
    fn detect_cross_fill(input: &Grid, output: &Grid) -> bool {
        if input.rows == 0 || input.cols == 0 { return false; }
        let mut row_colors: Vec<Vec<u8>> = vec![Vec::new(); input.rows];
        let mut col_colors: Vec<Vec<u8>> = vec![Vec::new(); input.cols];
        let mut any_marker = false;
        let mut total_markers = 0usize;

        for r in 0..input.rows {
            for c in 0..input.cols {
                if input.cells[r][c] != 0 {
                    any_marker = true;
                    total_markers += 1;
                    let color = input.cells[r][c];
                    if !row_colors[r].contains(&color) { row_colors[r].push(color); }
                    if !col_colors[c].contains(&color) { col_colors[c].push(color); }
                }
            }
        }
        if !any_marker { return false; }

        // Sparsity: markers should be < 30% of cells for cross pattern
        let total_cells = input.rows * input.cols;
        if total_markers * 3 > total_cells { return false; }

        // Need at least some rows AND some cols that are empty in input
        let empty_rows = (0..input.rows).filter(|r| row_colors[*r].is_empty()).count();
        let empty_cols = (0..input.cols).filter(|c| col_colors[*c].is_empty()).count();
        if empty_rows == 0 && empty_cols == 0 { return false; }

        let mut any_fill = false;
        for r in 0..output.rows {
            for c in 0..output.cols {
                let mut allowed = Vec::new();
                for &color in &row_colors[r] {
                    if !allowed.contains(&color) { allowed.push(color); }
                }
                for &color in &col_colors[c] {
                    if !allowed.contains(&color) { allowed.push(color); }
                }
                if allowed.is_empty() {
                    if output.cells[r][c] != 0 { return false; }
                } else {
                    if !allowed.contains(&output.cells[r][c]) {
                        return false;
                    }
                    if input.cells[r][c] == 0 { any_fill = true; }
                }
            }
        }
        any_fill
    }

    /// Predict output by row-fill: marker rows filled with marker color.
    pub fn predict_row_fill(input: &Grid) -> Vec<Statement> {
        let tv = Some(TruthValue::new(0.90, 0.90));
        let mut stmts = Vec::new();

        // For each row, find the dominant marker color
        for r in 0..input.rows {
            let mut colors: Vec<u8> = Vec::new();
            for c in 0..input.cols {
                if input.cells[r][c] != 0 && !colors.contains(&input.cells[r][c]) {
                    colors.push(input.cells[r][c]);
                }
            }
            if colors.len() == 1 {
                // Fill entire row with this color
                for c in 0..input.cols {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("predict-cell"),
                            Neuron::int_val(r as i64),
                            Neuron::int_val(c as i64),
                            Neuron::int_val(colors[0] as i64),
                        ]),
                        tv, decay: None,
                    });
                }
            } else if colors.len() > 1 {
                // Multiple marker colors in same row — use first one
                for c in 0..input.cols {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("predict-cell"),
                            Neuron::int_val(r as i64),
                            Neuron::int_val(c as i64),
                            Neuron::int_val(colors[0] as i64),
                        ]),
                        tv, decay: None,
                    });
                }
            }
            // No markers in this row → no predict-cell (stays 0)
        }
        stmts
    }

    /// Predict output by col-fill: marker columns filled with marker color.
    pub fn predict_col_fill(input: &Grid) -> Vec<Statement> {
        let tv = Some(TruthValue::new(0.90, 0.90));
        let mut stmts = Vec::new();
        let mut grid = vec![vec![0u8; input.cols]; input.rows];

        for c in 0..input.cols {
            let mut colors: Vec<u8> = Vec::new();
            for r in 0..input.rows {
                if input.cells[r][c] != 0 && !colors.contains(&input.cells[r][c]) {
                    colors.push(input.cells[r][c]);
                }
            }
            if !colors.is_empty() {
                let color = colors[0];
                for r in 0..input.rows {
                    grid[r][c] = color;
                }
            }
        }

        for r in 0..input.rows {
            for c in 0..input.cols {
                if grid[r][c] != 0 {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("predict-cell"),
                            Neuron::int_val(r as i64),
                            Neuron::int_val(c as i64),
                            Neuron::int_val(grid[r][c] as i64),
                        ]),
                        tv, decay: None,
                    });
                }
            }
        }
        stmts
    }

    /// Predict output by cross-fill: each marker paints its row + column.
    /// At intersections, the marker cell's own color wins.
    pub fn predict_cross_fill(input: &Grid) -> Vec<Statement> {
        let tv = Some(TruthValue::new(0.90, 0.90));
        let mut grid = vec![vec![0u8; input.cols]; input.rows];

        // First pass: find markers and their colors
        let mut markers: Vec<(usize, usize, u8)> = Vec::new();
        for r in 0..input.rows {
            for c in 0..input.cols {
                if input.cells[r][c] != 0 {
                    markers.push((r, c, input.cells[r][c]));
                }
            }
        }

        // Paint crosses — later markers overwrite earlier at intersections
        for &(mr, mc, color) in &markers {
            for c in 0..input.cols {
                grid[mr][c] = color;
            }
            for r in 0..input.rows {
                grid[r][mc] = color;
            }
        }

        // Emit predictions
        let mut stmts = Vec::new();
        for r in 0..input.rows {
            for c in 0..input.cols {
                if grid[r][c] != 0 {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("predict-cell"),
                            Neuron::int_val(r as i64),
                            Neuron::int_val(c as i64),
                            Neuron::int_val(grid[r][c] as i64),
                        ]),
                        tv, decay: None,
                    });
                }
            }
        }
        stmts
    }

    // ── Region Fill from Seed Detection ─────────────────────────────────

    /// Detect region-fill-from-seed: enclosed zero-regions in the input
    /// contain a single non-zero "seed" cell. The output fills the entire
    /// region with the seed's color (the seed itself is also that color).
    /// Returns Some(wall_color) — the color forming the walls/borders.
    /// Returns None if the pattern doesn't match.
    fn detect_region_fill_seed(input: &Grid, output: &Grid) -> Option<u8> {
        if input.rows < 3 || input.cols < 3 { return None; }

        // Identify the "wall" color — the most common non-zero color in input
        let mut color_counts = [0u32; 256];
        for r in 0..input.rows {
            for c in 0..input.cols {
                color_counts[input.cells[r][c] as usize] += 1;
            }
        }
        // Wall color = most frequent non-zero color
        let wall_color = (1..256)
            .max_by_key(|&i| color_counts[i])
            .filter(|&i| color_counts[i] > 0)?;

        // BFS to find connected regions of non-wall cells
        let mut visited = vec![vec![false; input.cols]; input.rows];
        let mut any_fill = false;

        for sr in 0..input.rows {
            for sc in 0..input.cols {
                if visited[sr][sc] || input.cells[sr][sc] == wall_color as u8 {
                    continue;
                }

                // BFS this region
                let mut region_cells: Vec<(usize, usize)> = Vec::new();
                let mut seeds: Vec<(usize, usize, u8)> = Vec::new(); // non-zero, non-wall cells
                let mut queue = VecDeque::new();
                queue.push_back((sr, sc));
                visited[sr][sc] = true;
                let mut touches_border = false;

                while let Some((r, c)) = queue.pop_front() {
                    region_cells.push((r, c));
                    let cell = input.cells[r][c];
                    if cell != 0 && cell != wall_color as u8 {
                        seeds.push((r, c, cell));
                    }
                    if r == 0 || r == input.rows - 1 || c == 0 || c == input.cols - 1 {
                        touches_border = true;
                    }

                    for (dr, dc) in [(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                        let nr = r as i32 + dr;
                        let nc = c as i32 + dc;
                        if nr >= 0 && nc >= 0 {
                            let (nr, nc) = (nr as usize, nc as usize);
                            if nr < input.rows && nc < input.cols
                                && !visited[nr][nc]
                                && input.cells[nr][nc] != wall_color as u8
                            {
                                visited[nr][nc] = true;
                                queue.push_back((nr, nc));
                            }
                        }
                    }
                }

                // Skip border-touching regions (not enclosed)
                if touches_border { continue; }
                if region_cells.is_empty() { continue; }

                // Need exactly 1 seed color in this region
                if seeds.is_empty() {
                    // No seed — output should keep region as-is (all 0)
                    for &(r, c) in &region_cells {
                        if output.cells[r][c] != 0 && output.cells[r][c] != input.cells[r][c] {
                            return None;
                        }
                    }
                    continue;
                }

                let seed_color = seeds[0].2;
                if seeds.iter().any(|s| s.2 != seed_color) {
                    return None; // Multiple seed colors in one region
                }

                // Verify output: entire region filled with seed color
                for &(r, c) in &region_cells {
                    if output.cells[r][c] != seed_color { return None; }
                }
                any_fill = true;
            }
        }

        // Verify walls preserved in output
        for r in 0..input.rows {
            for c in 0..input.cols {
                if input.cells[r][c] == wall_color as u8
                    && output.cells[r][c] != wall_color as u8
                {
                    return None;
                }
            }
        }

        if any_fill { Some(wall_color as u8) } else { None }
    }

    /// Predict output by region-fill-from-seed.
    /// Finds enclosed regions, fills each with its seed marker color.
    pub fn predict_region_fill_seed(input: &Grid, wall_color: u8) -> Vec<Statement> {
        let tv = Some(TruthValue::new(0.90, 0.90));
        let mut grid = input.cells.clone();

        // BFS to find non-wall connected regions
        let mut visited = vec![vec![false; input.cols]; input.rows];

        for sr in 0..input.rows {
            for sc in 0..input.cols {
                if visited[sr][sc] || input.cells[sr][sc] == wall_color {
                    continue;
                }

                let mut region_cells: Vec<(usize, usize)> = Vec::new();
                let mut seed_color: Option<u8> = None;
                let mut queue = VecDeque::new();
                queue.push_back((sr, sc));
                visited[sr][sc] = true;
                let mut touches_border = false;

                while let Some((r, c)) = queue.pop_front() {
                    region_cells.push((r, c));
                    let cell = input.cells[r][c];
                    if cell != 0 && cell != wall_color {
                        seed_color = Some(cell);
                    }
                    if r == 0 || r == input.rows - 1 || c == 0 || c == input.cols - 1 {
                        touches_border = true;
                    }

                    for (dr, dc) in [(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                        let nr = r as i32 + dr;
                        let nc = c as i32 + dc;
                        if nr >= 0 && nc >= 0 {
                            let (nr, nc) = (nr as usize, nc as usize);
                            if nr < input.rows && nc < input.cols
                                && !visited[nr][nc]
                                && input.cells[nr][nc] != wall_color
                            {
                                visited[nr][nc] = true;
                                queue.push_back((nr, nc));
                            }
                        }
                    }
                }

                // Only fill enclosed regions that have a seed
                if !touches_border {
                    if let Some(sc) = seed_color {
                        for &(r, c) in &region_cells {
                            grid[r][c] = sc;
                        }
                    }
                }
            }
        }

        // Emit predict-cell for ALL cells in result (including background=0)
        let mut stmts = Vec::new();
        for r in 0..input.rows {
            for c in 0..input.cols {
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("predict-cell"),
                        Neuron::int_val(r as i64),
                        Neuron::int_val(c as i64),
                        Neuron::int_val(grid[r][c] as i64),
                    ]),
                    tv, decay: None,
                });
            }
        }
        stmts
    }

    // ── Color Histogram Detection ───────────────────────────────────────

    /// Detect color-histogram: output is a bar chart of input color frequencies.
    /// The output has one column per non-background color, sorted by frequency
    /// (most frequent first or least frequent first), with bars growing from bottom.
    fn detect_color_histogram(input: &Grid, output: &Grid) -> bool {
        if input.rows == 0 || input.cols == 0 || output.rows == 0 || output.cols == 0 {
            return false;
        }

        // Count non-zero colors in input
        let mut color_counts: Vec<(u8, usize)> = Vec::new();
        let mut counts = [0usize; 256];
        for r in 0..input.rows {
            for c in 0..input.cols {
                let v = input.cells[r][c];
                if v != 0 { counts[v as usize] += 1; }
            }
        }
        for (color, &count) in counts.iter().enumerate() {
            if count > 0 {
                color_counts.push((color as u8, count));
            }
        }

        if color_counts.is_empty() { return false; }

        // Number of colors should match output cols
        if output.cols != color_counts.len() { return false; }

        // Try both sort orders: descending and ascending by count
        for sort_desc in [true, false] {
            let mut sorted = color_counts.clone();
            if sort_desc {
                sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            } else {
                sorted.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
            }

            // Check if output matches: bars from bottom, height = count
            // Output row count must accommodate the tallest bar
            let max_count = sorted.iter().map(|&(_, c)| c).max().unwrap_or(0);
            if output.rows != max_count { continue; }

            let mut matches = true;
            for (col_idx, &(color, count)) in sorted.iter().enumerate() {
                let empty_rows = output.rows - count;
                for r in 0..output.rows {
                    let expected = if r < empty_rows { 0 } else { color };
                    if output.cells[r][col_idx] != expected {
                        matches = false;
                        break;
                    }
                }
                if !matches { break; }
            }
            if matches { return true; }
        }

        // Try if the bars are horizontal instead (one row per color, bars from left)
        if output.rows == color_counts.len() {
            for sort_desc in [true, false] {
                let mut sorted = color_counts.clone();
                if sort_desc {
                    sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                } else {
                    sorted.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
                }

                let max_count = sorted.iter().map(|&(_, c)| c).max().unwrap_or(0);
                if output.cols != max_count { continue; }

                let mut matches = true;
                for (row_idx, &(color, count)) in sorted.iter().enumerate() {
                    for c in 0..output.cols {
                        let expected = if c < count { color } else { 0 };
                        if output.cells[row_idx][c] != expected {
                            matches = false;
                            break;
                        }
                    }
                    if !matches { break; }
                }
                if matches { return true; }
            }
        }

        false
    }

    // ── Noise Removal Detection ────────────────────────────────────────

    /// Detect noise removal: isolated pixels (no same-color 4-connected neighbor).
    /// Output = input minus all isolated single-pixel cells.
    fn detect_noise_isolated(input: &Grid, output: &Grid) -> bool {
        if input.rows == 0 || input.cols == 0 { return false; }
        let mut any_removed = false;

        for r in 0..input.rows {
            for c in 0..input.cols {
                let ic = input.cells[r][c];
                let oc = output.cells[r][c];

                if ic == 0 {
                    // Background: output should stay 0
                    if oc != 0 { return false; }
                    continue;
                }

                // Check if this cell is isolated (no same-color 4-neighbor)
                let isolated = ![(0i32, 1), (0, -1), (1, 0), (-1, 0)].iter().any(|&(dr, dc)| {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    nr >= 0 && nc >= 0
                        && (nr as usize) < input.rows && (nc as usize) < input.cols
                        && input.cells[nr as usize][nc as usize] == ic
                });

                if isolated {
                    // Should be removed in output
                    if oc != 0 { return false; }
                    any_removed = true;
                } else {
                    // Should be preserved
                    if oc != ic { return false; }
                }
            }
        }
        any_removed
    }

    /// Detect noise removal: keep only the majority non-zero color.
    /// Returns Some(keep_color) if output = input with only one color kept.
    fn detect_noise_minority(input: &Grid, output: &Grid) -> Option<u8> {
        let mut counts = [0usize; 256];
        for r in 0..input.rows {
            for c in 0..input.cols {
                let v = input.cells[r][c];
                if v != 0 { counts[v as usize] += 1; }
            }
        }

        // Find majority color
        let majority = (1..256)
            .max_by_key(|&i| counts[i])
            .filter(|&i| counts[i] > 0)?;

        // Need at least 2 colors for noise removal to make sense
        let color_count = (1..256).filter(|&i| counts[i] > 0).count();
        if color_count < 2 { return None; }

        // Verify: output keeps only majority color cells
        for r in 0..input.rows {
            for c in 0..input.cols {
                let ic = input.cells[r][c];
                let oc = output.cells[r][c];
                if ic == majority as u8 {
                    if oc != ic { return None; }
                } else {
                    if oc != 0 { return None; }
                }
            }
        }
        Some(majority as u8)
    }

    /// Detect keep-largest: output = only the largest connected object from input.
    fn detect_keep_largest(input: &Grid, output: &Grid) -> bool {
        let objects = input.objects();
        if objects.len() < 2 { return false; }

        // Find the largest object
        let largest = objects.iter().max_by_key(|o| o.cells.len());
        let largest = match largest {
            Some(l) => l,
            None => return false,
        };

        // Build expected output: only largest object cells, rest = 0
        for r in 0..input.rows {
            for c in 0..input.cols {
                let in_largest = largest.cells.contains(&(r, c));
                let expected = if in_largest { input.cells[r][c] } else { 0 };
                if output.cells[r][c] != expected { return false; }
            }
        }
        true
    }

    /// Detect noise removal by object size threshold.
    /// Returns Some(threshold) where all objects with size <= threshold are removed.
    fn detect_noise_small_objects(input: &Grid, output: &Grid) -> Option<usize> {
        let objects = input.objects();
        if objects.len() < 2 { return None; }

        // Find which objects are kept vs removed
        let mut kept_sizes = Vec::new();
        let mut removed_sizes = Vec::new();

        for obj in &objects {
            // Check if all cells of this object are preserved in output
            let preserved = obj.cells.iter().all(|&(r, c)| output.cells[r][c] == obj.color);
            if preserved {
                kept_sizes.push(obj.cells.len());
            } else {
                // Check if all cells are removed (set to 0)
                let all_zero = obj.cells.iter().all(|&(r, c)| output.cells[r][c] == 0);
                if !all_zero { return None; } // Partial removal = not this pattern
                removed_sizes.push(obj.cells.len());
            }
        }

        if removed_sizes.is_empty() || kept_sizes.is_empty() { return None; }

        let max_removed = removed_sizes.iter().max().copied().unwrap_or(0);
        let min_kept = kept_sizes.iter().min().copied().unwrap_or(0);

        // There should be a clean threshold: all removed < threshold <= all kept
        if max_removed < min_kept {
            // Also verify no new cells were added
            for r in 0..output.rows {
                for c in 0..output.cols {
                    if output.cells[r][c] != 0 && input.cells[r][c] == 0 {
                        return None;
                    }
                }
            }
            Some(max_removed)
        } else {
            None
        }
    }

    /// Predict output by removing isolated single-pixel noise.
    pub fn predict_noise_isolated(input: &Grid) -> Vec<Statement> {
        let tv = Some(TruthValue::new(0.90, 0.90));
        let mut stmts = Vec::new();

        for r in 0..input.rows {
            for c in 0..input.cols {
                let color = input.cells[r][c];

                // Background stays background
                if color == 0 {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("predict-cell"),
                            Neuron::int_val(r as i64),
                            Neuron::int_val(c as i64),
                            Neuron::int_val(0i64),
                        ]),
                        tv, decay: None,
                    });
                    continue;
                }

                let has_neighbor = [(0i32, 1), (0, -1), (1, 0), (-1, 0)].iter().any(|&(dr, dc)| {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    nr >= 0 && nc >= 0
                        && (nr as usize) < input.rows && (nc as usize) < input.cols
                        && input.cells[nr as usize][nc as usize] == color
                });

                // Keep non-isolated cells, remove isolated ones (→ 0)
                let out_color = if has_neighbor { color } else { 0 };
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("predict-cell"),
                        Neuron::int_val(r as i64),
                        Neuron::int_val(c as i64),
                        Neuron::int_val(out_color as i64),
                    ]),
                    tv, decay: None,
                });
            }
        }
        stmts
    }

    /// Predict output by keeping only the majority color.
    pub fn predict_noise_minority(input: &Grid, keep_color: u8) -> Vec<Statement> {
        let tv = Some(TruthValue::new(0.90, 0.90));
        let mut stmts = Vec::new();

        for r in 0..input.rows {
            for c in 0..input.cols {
                // Keep cells matching keep_color, set everything else to 0
                let color = if input.cells[r][c] == keep_color { keep_color } else { 0 };
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("predict-cell"),
                        Neuron::int_val(r as i64),
                        Neuron::int_val(c as i64),
                        Neuron::int_val(color as i64),
                    ]),
                    tv, decay: None,
                });
            }
        }
        stmts
    }

    /// Predict output by keeping only the largest object.
    pub fn predict_keep_largest(input: &Grid) -> Vec<Statement> {
        let tv = Some(TruthValue::new(0.90, 0.90));
        let objects = input.objects();
        let largest = match objects.iter().max_by_key(|o| o.cells.len()) {
            Some(l) => l,
            None => return Vec::new(),
        };

        // Build a set of largest-object cells for O(1) lookup
        let largest_cells: std::collections::HashSet<(usize, usize)> =
            largest.cells.iter().copied().collect();

        // Output ALL cells: largest object keeps its color, everything else → 0 (background)
        let mut stmts = Vec::new();
        for r in 0..input.rows {
            for c in 0..input.cols {
                let color = if largest_cells.contains(&(r, c)) {
                    largest.color
                } else {
                    0
                };
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("predict-cell"),
                        Neuron::int_val(r as i64),
                        Neuron::int_val(c as i64),
                        Neuron::int_val(color as i64),
                    ]),
                    tv, decay: None,
                });
            }
        }
        stmts
    }

    /// Predict output by removing small objects (size <= threshold).
    pub fn predict_noise_small_objects(input: &Grid, threshold: usize) -> Vec<Statement> {
        let tv = Some(TruthValue::new(0.90, 0.90));
        let objects = input.objects();

        // Build a set of cells belonging to large-enough objects
        let mut keep_cells: std::collections::HashMap<(usize, usize), u8> =
            std::collections::HashMap::new();
        for obj in &objects {
            if obj.cells.len() > threshold {
                for &(r, c) in &obj.cells {
                    keep_cells.insert((r, c), obj.color);
                }
            }
        }

        // Output ALL cells: kept objects retain color, removed → 0
        let mut stmts = Vec::new();
        for r in 0..input.rows {
            for c in 0..input.cols {
                let color = keep_cells.get(&(r, c)).copied().unwrap_or(0) as i64;
                stmts.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("predict-cell"),
                        Neuron::int_val(r as i64),
                        Neuron::int_val(c as i64),
                        Neuron::int_val(color),
                    ]),
                    tv, decay: None,
                });
            }
        }
        stmts
    }

    /// Predict output by color-histogram: vertical bars, descending frequency.
    pub fn predict_color_histogram(input: &Grid, vertical: bool, descending: bool) -> Vec<Statement> {
        let tv = Some(TruthValue::new(0.90, 0.90));

        // Count non-zero colors
        let mut counts = [0usize; 256];
        for r in 0..input.rows {
            for c in 0..input.cols {
                let v = input.cells[r][c];
                if v != 0 { counts[v as usize] += 1; }
            }
        }

        let mut color_counts: Vec<(u8, usize)> = Vec::new();
        for (color, &count) in counts.iter().enumerate() {
            if count > 0 {
                color_counts.push((color as u8, count));
            }
        }

        if descending {
            color_counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        } else {
            color_counts.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
        }

        let max_count = color_counts.iter().map(|&(_, c)| c).max().unwrap_or(0);
        let mut stmts = Vec::new();

        if vertical {
            // Vertical bars: cols = colors, rows = max count, bars from bottom
            for (col_idx, &(color, count)) in color_counts.iter().enumerate() {
                let empty_rows = max_count - count;
                for r in 0..max_count {
                    if r >= empty_rows {
                        stmts.push(Statement::Fact {
                            neuron: Neuron::expression(vec![
                                Neuron::symbol("predict-cell"),
                                Neuron::int_val(r as i64),
                                Neuron::int_val(col_idx as i64),
                                Neuron::int_val(color as i64),
                            ]),
                            tv, decay: None,
                        });
                    }
                }
            }
        } else {
            // Horizontal bars: rows = colors, cols = max count, bars from left
            for (row_idx, &(color, count)) in color_counts.iter().enumerate() {
                for c in 0..count {
                    stmts.push(Statement::Fact {
                        neuron: Neuron::expression(vec![
                            Neuron::symbol("predict-cell"),
                            Neuron::int_val(row_idx as i64),
                            Neuron::int_val(c as i64),
                            Neuron::int_val(color as i64),
                        ]),
                        tv, decay: None,
                    });
                }
            }
        }

        stmts
    }

    /// Compare predicted vs expected grids — cell accuracy (0.0 to 1.0).
    pub fn cell_accuracy(predicted: &[Vec<u8>], expected: &[Vec<u8>]) -> f64 {
        let total = expected.iter().map(|r| r.len()).sum::<usize>();
        if total == 0 { return 0.0; }
        let mut correct = 0;
        for (r, row) in expected.iter().enumerate() {
            for (c, &val) in row.iter().enumerate() {
                let p = predicted.get(r).and_then(|pr| pr.get(c).copied()).unwrap_or(255);
                if p == val { correct += 1; }
            }
        }
        correct as f64 / total as f64
    }

    /// Extract cell predictions from a session for the given predicate.
    /// Handles both 4-arg `(pred r c v)` and 5-arg `(pred id r c v)` formats.
    /// Uses confidence-based selection (highest confidence wins per cell).
    pub fn extract_predictions(
        session: &qor_runtime::eval::Session,
        pred_name: &str,
        rows: usize,
        cols: usize,
    ) -> Option<Vec<Vec<u8>>> {
        use qor_core::neuron::Neuron;
        let mut cells = vec![vec![0u8; cols]; rows];
        let mut conf = vec![vec![0.0f64; cols]; rows];
        let mut any = false;
        let pred_sym = Neuron::symbol(pred_name);
        for sn in session.all_facts() {
            if let Neuron::Expression(parts) = &sn.neuron {
                let indices = if parts.len() == 4 && parts[0] == pred_sym {
                    Some((1, 2, 3))
                } else if parts.len() == 5 && parts[0] == pred_sym {
                    Some((2, 3, 4))
                } else {
                    None
                };
                if let Some((ri, ci, vi)) = indices {
                    if let (Some(r), Some(c), Some(v)) = (
                        parts[ri].as_usize(), parts[ci].as_usize(), parts[vi].as_usize(),
                    ) {
                        let c_val = sn.tv.confidence;
                        if r < rows && c < cols && c_val >= conf[r][c] {
                            cells[r][c] = v as u8;
                            conf[r][c] = c_val;
                            any = true;
                        }
                    }
                }
            }
        }
        if any { Some(cells) } else { None }
    }
}

// ── Deep Grid Perception ────────────────────────────────────────────────
//
// Detects ARC-like train/test pair structure in grid facts.
// Remaps long IDs (train-0-input) to short IDs (t0i, t0o, ti).
// Runs cross-pair analysis (compare_pair, compute_observations).
// Returns enriched facts ready for QOR reasoning.

/// Extract an integer from a Neuron value.
fn neuron_as_i64(n: &Neuron) -> Option<i64> {
    match n {
        Neuron::Value(QorValue::Int(i)) => Some(*i),
        Neuron::Value(QorValue::Float(f)) => Some(*f as i64),
        _ => None,
    }
}

/// Remap a symbol name using the ID mapping.
/// Handles both direct matches (grid IDs) and prefix matches (object IDs).
fn remap_symbol(sym: &str, id_map: &HashMap<String, String>) -> Option<String> {
    // Direct match
    if let Some(new_id) = id_map.get(sym) {
        return Some(new_id.clone());
    }
    // Prefix match for object IDs: "train-0-input_obj0" → "t0i_obj0"
    for (old, new) in id_map {
        if sym.starts_with(old.as_str()) {
            return Some(format!("{}{}", new, &sym[old.len()..]));
        }
    }
    None
}

/// Remap all symbol neurons in a fact using the ID mapping.
fn remap_fact(fact: &Statement, id_map: &HashMap<String, String>) -> Statement {
    match fact {
        Statement::Fact { neuron: Neuron::Expression(parts), tv, decay } => {
            let new_parts: Vec<Neuron> = parts.iter().map(|p| {
                if let Neuron::Symbol(s) = p {
                    if let Some(new_s) = remap_symbol(s, id_map) {
                        return Neuron::symbol(&new_s);
                    }
                }
                p.clone()
            }).collect();
            Statement::Fact {
                neuron: Neuron::Expression(new_parts),
                tv: *tv,
                decay: *decay,
            }
        }
        other => other.clone(),
    }
}

/// Deep grid perception for ARC-like data.
///
/// Detects training pair structure (train-N-input / train-N-output / test-N-input),
/// remaps to short IDs (t0i, t0o, ti), adds structural facts, and runs
/// cross-pair analysis (compare_pair, compute_observations).
///
/// If no ARC structure is detected, returns the input facts unchanged.
pub fn deep_grid_perception(grid_facts: Vec<Statement>) -> Vec<Statement> {
    // 1. Parse grid IDs and cell data from the facts
    let mut grid_sizes: HashMap<String, (usize, usize)> = HashMap::new();
    let mut grid_cells_map: HashMap<String, Vec<(usize, usize, u8)>> = HashMap::new();

    for fact in &grid_facts {
        if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = fact {
            if parts.len() < 3 { continue; }
            if let Neuron::Symbol(pred) = &parts[0] {
                match pred.as_str() {
                    "grid-size" if parts.len() >= 4 => {
                        if let (Neuron::Symbol(id), Some(rows), Some(cols)) =
                            (&parts[1], neuron_as_i64(&parts[2]), neuron_as_i64(&parts[3]))
                        {
                            grid_sizes.insert(id.clone(), (rows as usize, cols as usize));
                        }
                    }
                    "grid-cell" if parts.len() >= 5 => {
                        if let (Neuron::Symbol(id), Some(r), Some(c), Some(color)) = (
                            &parts[1],
                            neuron_as_i64(&parts[2]),
                            neuron_as_i64(&parts[3]),
                            neuron_as_i64(&parts[4]),
                        ) {
                            grid_cells_map
                                .entry(id.clone())
                                .or_default()
                                .push((r as usize, c as usize, color as u8));
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // 2. Detect ARC train/test pair structure
    let mut train_pairs: Vec<(String, String)> = Vec::new();
    let mut test_inputs: Vec<String> = Vec::new();
    let mut test_outputs: Vec<String> = Vec::new();

    let mut sorted_ids: Vec<&String> = grid_sizes.keys().collect();
    sorted_ids.sort();

    for id in &sorted_ids {
        if id.ends_with("-input") {
            let base = &id[..id.len() - 6]; // strip "-input"
            let output_id = format!("{}-output", base);
            if base.starts_with("train") && grid_sizes.contains_key(&output_id) {
                train_pairs.push((id.to_string(), output_id));
            } else if base.starts_with("test") {
                test_inputs.push(id.to_string());
            }
        } else if id.ends_with("-output") && id.starts_with("test") {
            test_outputs.push(id.to_string());
        }
    }

    // Build set of test output IDs — these get converted to expected-* facts
    let test_output_set: std::collections::HashSet<&str> =
        test_outputs.iter().map(|s| s.as_str()).collect();

    // If no ARC structure detected, return facts as-is
    if train_pairs.is_empty() {
        return grid_facts;
    }

    train_pairs.sort();
    test_inputs.sort();

    // 3. Build ID mapping: long → short
    let mut id_map: HashMap<String, String> = HashMap::new();
    for (i, (inp, outp)) in train_pairs.iter().enumerate() {
        id_map.insert(inp.clone(), format!("t{}i", i));
        id_map.insert(outp.clone(), format!("t{}o", i));
    }
    for (i, test_id) in test_inputs.iter().enumerate() {
        let short = if i == 0 { "ti".to_string() } else { format!("ti{}", i) };
        id_map.insert(test_id.clone(), short);
    }

    // 4. Re-emit grid facts with short IDs, excluding test output facts
    let mut result: Vec<Statement> = Vec::new();
    for fact in &grid_facts {
        // Skip test output grid facts — they become expected-* facts below
        if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = fact {
            if parts.len() >= 2 {
                if let Neuron::Symbol(id) = &parts[1] {
                    if test_output_set.contains(id.as_str()) {
                        continue;
                    }
                    // Also skip object-level facts for test outputs
                    if test_outputs.iter().any(|tid| id.starts_with(tid.as_str())) {
                        continue;
                    }
                }
            }
        }
        result.push(remap_fact(fact, &id_map));
    }

    // 5. Emit structural facts
    let tv = Some(TruthValue::new(0.99, 0.99));
    for (i, _) in train_pairs.iter().enumerate() {
        result.push(Statement::Fact {
            neuron: Neuron::expression(vec![
                Neuron::symbol("train-pair"),
                Neuron::symbol(&format!("t{}i", i)),
                Neuron::symbol(&format!("t{}o", i)),
            ]),
            tv,
            decay: None,
        });
    }
    for (i, _) in test_inputs.iter().enumerate() {
        let short = if i == 0 { "ti".to_string() } else { format!("ti{}", i) };
        result.push(Statement::Fact {
            neuron: Neuron::expression(vec![
                Neuron::symbol("test-input"),
                Neuron::symbol(&short),
            ]),
            tv,
            decay: None,
        });
    }

    // 5b. Emit expected-predict-cell facts from test output grids (for scoring)
    for test_out_id in &test_outputs {
        if let Some(cells) = grid_cells_map.get(test_out_id) {
            for &(r, c, color) in cells {
                // 3-arg format: (expected-predict-cell r c color)
                result.push(Statement::Fact {
                    neuron: Neuron::expression(vec![
                        Neuron::symbol("expected-predict-cell"),
                        Neuron::int_val(r as i64),
                        Neuron::int_val(c as i64),
                        Neuron::int_val(color as i64),
                    ]),
                    tv,
                    decay: None,
                });
            }
        }
    }

    // 6. Reconstruct Grid objects and run compare_pair for each training pair
    let mut pair_grids: Vec<(Grid, Grid)> = Vec::new();
    let mut pair_facts_all: Vec<Vec<Statement>> = Vec::new();

    for (i, (inp_id, outp_id)) in train_pairs.iter().enumerate() {
        let in_short = format!("t{}i", i);
        let out_short = format!("t{}o", i);

        if let (Some(&(in_rows, in_cols)), Some(&(out_rows, out_cols))) =
            (grid_sizes.get(inp_id), grid_sizes.get(outp_id))
        {
            let in_cells = reconstruct_grid_cells(
                in_rows, in_cols, grid_cells_map.get(inp_id),
            );
            let out_cells = reconstruct_grid_cells(
                out_rows, out_cols, grid_cells_map.get(outp_id),
            );

            if let (Ok(in_grid), Ok(out_grid)) =
                (Grid::from_vecs(in_cells), Grid::from_vecs(out_cells))
            {
                let pfacts = Grid::compare_pair(&in_grid, &out_grid, &in_short, &out_short);
                pair_facts_all.push(pfacts.clone());
                result.extend(pfacts);
                pair_grids.push((in_grid, out_grid));
            }
        }
    }

    // 7. Compute cross-pair observations
    if !pair_grids.is_empty() {
        let obs = Grid::compute_observations(&pair_facts_all, &pair_grids);
        result.extend(obs);
    }

    result
}

/// Reconstruct a 2D cell grid from parsed cell coordinates.
fn reconstruct_grid_cells(
    rows: usize,
    cols: usize,
    cells: Option<&Vec<(usize, usize, u8)>>,
) -> Vec<Vec<u8>> {
    let mut grid = vec![vec![0u8; cols]; rows];
    if let Some(cells) = cells {
        for &(r, c, color) in cells {
            if r < rows && c < cols {
                grid[r][c] = color;
            }
        }
    }
    grid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_from_vecs() {
        let g = Grid::from_vecs(vec![
            vec![0, 1, 0],
            vec![1, 1, 0],
        ]).unwrap();
        assert_eq!(g.rows, 2);
        assert_eq!(g.cols, 3);
        assert_eq!(g.cell_at(0, 1), Some(1));
        assert_eq!(g.cell_at(1, 2), Some(0));
        assert_eq!(g.dimensions(), (2, 3));
    }

    #[test]
    fn test_grid_from_vecs_ragged_error() {
        let r = Grid::from_vecs(vec![
            vec![0, 1],
            vec![1, 1, 0],
        ]);
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("ragged"));
    }

    #[test]
    fn test_grid_objects_single() {
        let g = Grid::from_vecs(vec![
            vec![0, 1, 0],
            vec![0, 1, 0],
            vec![0, 1, 1],
        ]).unwrap();
        let objs = g.objects();
        assert_eq!(objs.len(), 1);
        assert_eq!(objs[0].color, 1);
        assert_eq!(objs[0].cells.len(), 4);
    }

    #[test]
    fn test_grid_objects_two_colors() {
        let g = Grid::from_vecs(vec![
            vec![1, 1, 0, 2, 2],
            vec![0, 0, 0, 0, 0],
        ]).unwrap();
        let objs = g.objects();
        assert_eq!(objs.len(), 2);
        let colors: Vec<u8> = objs.iter().map(|o| o.color).collect();
        assert!(colors.contains(&1));
        assert!(colors.contains(&2));
    }

    #[test]
    fn test_grid_objects_disconnected() {
        let g = Grid::from_vecs(vec![
            vec![1, 0, 1],
            vec![0, 0, 0],
        ]).unwrap();
        let objs = g.objects();
        assert_eq!(objs.len(), 2);
        assert_eq!(objs[0].color, 1);
        assert_eq!(objs[1].color, 1);
        assert_eq!(objs[0].cells.len(), 1);
        assert_eq!(objs[1].cells.len(), 1);
    }

    #[test]
    fn test_grid_background_ignored() {
        let g = Grid::from_vecs(vec![
            vec![0, 0, 0],
            vec![0, 1, 0],
            vec![0, 0, 0],
        ]).unwrap();
        let objs = g.objects();
        assert_eq!(objs.len(), 1);
        assert_eq!(objs[0].cells.len(), 1);
        assert_eq!(objs[0].cells[0], (1, 1));
    }

    #[test]
    fn test_grid_to_statements() {
        let g = Grid::from_vecs(vec![
            vec![0, 1],
            vec![1, 1],
        ]).unwrap();
        let stmts = g.to_statements("g1");

        let size_count = stmts.iter().filter(|s| {
            matches!(s, Statement::Fact { neuron: Neuron::Expression(parts), .. }
                if parts[0] == Neuron::symbol("grid-size"))
        }).count();
        let cell_count = stmts.iter().filter(|s| {
            matches!(s, Statement::Fact { neuron: Neuron::Expression(parts), .. }
                if parts[0] == Neuron::symbol("grid-cell"))
        }).count();
        let obj_count = stmts.iter().filter(|s| {
            matches!(s, Statement::Fact { neuron: Neuron::Expression(parts), .. }
                if parts[0] == Neuron::symbol("grid-object"))
        }).count();

        assert_eq!(size_count, 1);
        assert_eq!(cell_count, 4); // all cells including zeros
        assert_eq!(obj_count, 1);
    }

    #[test]
    fn test_grid_bbox() {
        let g = Grid::from_vecs(vec![
            vec![0, 0, 0, 0],
            vec![0, 1, 1, 0],
            vec![0, 0, 1, 0],
            vec![0, 0, 0, 0],
        ]).unwrap();
        let objs = g.objects();
        assert_eq!(objs.len(), 1);
        let (min_r, min_c, max_r, max_c) = objs[0].bbox();
        assert_eq!((min_r, min_c, max_r, max_c), (1, 1, 2, 2));
    }

    #[test]
    fn test_grid_empty() {
        let g = Grid::from_vecs(vec![]).unwrap();
        assert_eq!(g.rows, 0);
        assert_eq!(g.cols, 0);
        let stmts = g.to_statements("empty");
        // Must have grid-size, object-count, color-count at minimum
        assert!(stmts.len() >= 3);
        assert_eq!(g.objects().len(), 0);
    }

    #[test]
    fn test_grid_all_background() {
        let g = Grid::from_vecs(vec![
            vec![0, 0, 0],
            vec![0, 0, 0],
        ]).unwrap();
        let objs = g.objects();
        assert_eq!(objs.len(), 0);
        let stmts = g.to_statements("bg");
        // grid-size + object-count + color-count + neighbor facts for all 6 cells
        assert!(stmts.len() >= 3);
        // Verify neighbor facts exist for background cells
        let neighbor_count = stmts.iter().filter(|s| {
            if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = s {
                parts.first().map(|p| p.to_string() == "grid-neighbor").unwrap_or(false)
            } else { false }
        }).count();
        // 2x3 grid: each cell has 2-4 neighbors = 14 neighbor facts total
        assert!(neighbor_count > 0, "background cells should have neighbor facts");
    }

    #[test]
    fn test_grid_equals() {
        let g1 = Grid::from_vecs(vec![vec![1, 2], vec![3, 4]]).unwrap();
        let g2 = Grid::from_vecs(vec![vec![1, 2], vec![3, 4]]).unwrap();
        let g3 = Grid::from_vecs(vec![vec![1, 2], vec![3, 5]]).unwrap();
        assert!(g1.equals(&g2));
        assert!(!g1.equals(&g3));
    }

    // ── compare_pair tests ──────────────────────────────────────────────

    fn fact_name(s: &Statement) -> String {
        if let Statement::Fact { neuron: Neuron::Expression(parts), .. } = s {
            parts.first().map(|p| p.to_string()).unwrap_or_default()
        } else {
            String::new()
        }
    }

    #[test]
    fn test_compare_pair_identity() {
        let g = Grid::from_vecs(vec![vec![1, 0], vec![0, 2]]).unwrap();
        let stmts = Grid::compare_pair(&g, &g, "in", "out");
        let has_identity = stmts.iter().any(|s| fact_name(s) == "pair-identity");
        assert!(has_identity, "identical grids should produce pair-identity");
    }

    #[test]
    fn test_compare_pair_reflect_h() {
        let input = Grid::from_vecs(vec![vec![1, 0, 2]]).unwrap();
        let output = Grid::from_vecs(vec![vec![2, 0, 1]]).unwrap();
        let stmts = Grid::compare_pair(&input, &output, "in", "out");
        let has_reflect = stmts.iter().any(|s| fact_name(s) == "pair-reflect-h");
        assert!(has_reflect, "horizontally mirrored grid should produce pair-reflect-h");
    }

    #[test]
    fn test_compare_pair_gravity() {
        let input = Grid::from_vecs(vec![
            vec![1, 0],
            vec![0, 0],
        ]).unwrap();
        let output = Grid::from_vecs(vec![
            vec![0, 0],
            vec![1, 0],
        ]).unwrap();
        let stmts = Grid::compare_pair(&input, &output, "in", "out");
        let has_gravity = stmts.iter().any(|s| fact_name(s) == "pair-gravity-down");
        assert!(has_gravity, "gravity-dropped grid should produce pair-gravity-down");
    }

    #[test]
    fn test_compare_pair_flood_fill() {
        let input = Grid::from_vecs(vec![
            vec![1, 1, 1],
            vec![1, 0, 1],
            vec![1, 1, 1],
        ]).unwrap();
        let output = Grid::from_vecs(vec![
            vec![1, 1, 1],
            vec![1, 2, 1],
            vec![1, 1, 1],
        ]).unwrap();
        let stmts = Grid::compare_pair(&input, &output, "in", "out");
        let has_fill = stmts.iter().any(|s| fact_name(s) == "pair-flood-fill");
        assert!(has_fill, "enclosed fill should produce pair-flood-fill");
    }

    #[test]
    fn test_compare_pair_scale_up() {
        let input = Grid::from_vecs(vec![vec![1, 2], vec![3, 4]]).unwrap();
        let output = Grid::from_vecs(vec![
            vec![1, 1, 2, 2],
            vec![1, 1, 2, 2],
            vec![3, 3, 4, 4],
            vec![3, 3, 4, 4],
        ]).unwrap();
        let stmts = Grid::compare_pair(&input, &output, "in", "out");
        let has_scale = stmts.iter().any(|s| fact_name(s) == "pair-scale-up");
        assert!(has_scale, "2x scaled grid should produce pair-scale-up");
    }

    #[test]
    fn test_compare_pair_shift() {
        let input = Grid::from_vecs(vec![
            vec![1, 0, 0],
            vec![0, 0, 0],
            vec![0, 0, 0],
        ]).unwrap();
        let output = Grid::from_vecs(vec![
            vec![0, 0, 0],
            vec![0, 1, 0],
            vec![0, 0, 0],
        ]).unwrap();
        let stmts = Grid::compare_pair(&input, &output, "in", "out");
        let has_shift = stmts.iter().any(|s| fact_name(s) == "pair-shift");
        assert!(has_shift, "shifted grid should produce pair-shift");
    }
}
