//! CodinGame Tron Battle agent.
//!
//! Paste this single file into the CodinGame Rust IDE. Local extras:
//! `cargo run --release -- --bench` and `--profile`.
//!
//! **1v1:** iterative-deepening alpha-beta with a Voronoi territory eval.
//! When the two bikes can no longer reach each other, switch to greedy
//! space-fill (survive as long as possible in our chamber).
//! **FFA (3–4 players):** one 2-ply of greedy replies, no deep minimax
//! (deep 1v1 search suicides against multiple opponents).

#![allow(dead_code)]

use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

/// How many choose_move calls saw the two duelists in separate chambers.
static SEPARATED_MOVE_COUNT: AtomicU32 = AtomicU32::new(0);
/// Total choose_move calls (for the bench separated-rate print).
static CHOOSE_MOVE_COUNT: AtomicU32 = AtomicU32::new(0);
/// Separated calls that happened with almost-empty boards (suspicious).
static EARLY_SEPARATION_COUNT: AtomicU32 = AtomicU32::new(0);

const WIDTH: i32 = 30;
const HEIGHT: i32 = 20;
const BOARD_CELLS: usize = 600;
const MAX_PLAYERS: usize = 4;

/// Direction index 0..3 matches [`DIR_NAME`]. 4 means “no move”.
const DIR_X: [i32; 4] = [0, 0, -1, 1];
const DIR_Y: [i32; 4] = [-1, 1, 0, 0];
const DIR_NAME: [&str; 4] = ["UP", "DOWN", "LEFT", "RIGHT"];
const NO_MOVE: u8 = 4;

/// Sentinel distance: cell is unreachable in a BFS.
const UNREACHABLE: u16 = 0x7FFF;
/// Win/loss scores. Subtract ply so faster mates are preferred.
const MATE_SCORE: i32 = 1_000_000;
/// Returned from search when the time budget is exhausted (not a real eval).
const TIMEOUT_SCORE: i32 = i32::MIN / 4;

const TURN_BUDGET_MS: u64 = 75;
const FIRST_TURN_BUDGET_MS: u64 = 85;

/// True if `(col, row)` is on the 30×20 board.
///
/// Called from every neighbour walk (`State::apply`, BFS, flood, Voronoi, fill)
/// so we never index off the bitboards.
#[inline]
fn in_bounds(col: i32, row: i32) -> bool {
    col >= 0 && col < WIDTH && row >= 0 && row < HEIGHT
}

/// Flatten `(col, row)` to `row * 30 + col` in `0..600`.
///
/// Indexes `Scratch` arrays (`distance`, `visited_stamp`, `owner`, `bfs_queue`)
/// so BFS / Voronoi / fill can store one value per cell without a 2D array.
#[inline]
fn cell_index(col: i32, row: i32) -> usize {
    (row * WIDTH + col) as usize
}

/// Inverse of [`cell_index`]: `(col, row)` from a flat `0..600` index.
///
/// Used when dequeuing a BFS/flood cell so we can expand its 4-neighbours.
#[inline]
fn coords_from_index(index: usize) -> (i32, i32) {
    ((index as i32) % WIDTH, (index as i32) / WIDTH)
}

/// One bit per column in a single board row (`WIDTH <= 32`).
///
/// Occupancy and each player’s trail are 20 of these. XOR on death is O(20)
/// instead of walking 600 cells. Used everywhere a cell is tested or marked.
#[derive(Clone, Copy)]
struct RowBits {
    bits: [u32; 20],
}

impl RowBits {
    /// All-zero occupancy: every cell in this mask is empty.
    /// Used to construct a fresh [`State`] and to clear a dead player’s trail.
    #[inline]
    fn empty() -> Self {
        Self { bits: [0; 20] }
    }

    /// True if column `col` of row `row` is set (occupied by this mask).
    /// Hot-path legality / BFS test: “is this neighbour a wall or trail?”
    #[inline]
    fn is_set(self, col: i32, row: i32) -> bool {
        self.bits[row as usize] & (1u32 << col) != 0
    }

    /// Mark `(col, row)` occupied in this mask.
    /// Called from [`State::occupy`] for both the combined board and that player’s trail.
    #[inline]
    fn set(&mut self, col: i32, row: i32) {
        self.bits[row as usize] |= 1u32 << col;
    }

    /// Mark `(col, row)` empty in this mask.
    /// [`State::undo_step`] uses this so search can take a move back without cloning occupancy.
    #[inline]
    fn clear(&mut self, col: i32, row: i32) {
        self.bits[row as usize] &= !(1u32 << col);
    }

    /// XOR another mask into this one (used to erase a dead player’s trail).
    /// Because each trail cell is unique to that player, XOR is equivalent to
    /// clearing those bits from the combined occupancy.
    /// [`State::kill`] is the only caller.
    #[inline]
    fn xor_with(&mut self, trail_mask: RowBits) {
        for row in 0..20 {
            self.bits[row] ^= trail_mask.bits[row];
        }
    }
}

/// Full game position: occupancy, per-player trails, heads, who is alive.
///
/// This is the board [`Tracker`] maintains from CodinGame input and that
/// [`choose_move`] / search mutate with apply/undo. `Copy` so FFA can fork a
/// position for opponent replies without a heap clone.
#[derive(Clone, Copy)]
struct State {
    /// Cells that currently contain any trail (or a live head).
    occupied: RowBits,
    /// Each player’s ribbon. XOR’d out of `occupied` on death.
    trail: [RowBits; MAX_PLAYERS],
    /// Current head column for each player, or -1 if dead / unset.
    head_x: [i8; MAX_PLAYERS],
    /// Current head row for each player, or -1 if dead / unset.
    head_y: [i8; MAX_PLAYERS],
    /// Bit `player` set ⇒ that player is still alive.
    alive_mask: u8,
    /// How many player slots this match uses (2, 3, or 4), including dead ones.
    player_count: u8,
}

impl State {
    /// Empty board with `player_count` slots; nobody occupied yet.
    /// Heads are -1 and `alive_mask` is 0 until [`State::occupy`] is called.
    /// Used by [`Tracker`] on the first frame and by local `--bench` spawns.
    fn new(player_count: u8) -> Self {
        Self {
            occupied: RowBits::empty(),
            trail: [RowBits::empty(); MAX_PLAYERS],
            head_x: [-1; MAX_PLAYERS],
            head_y: [-1; MAX_PLAYERS],
            alive_mask: 0,
            player_count,
        }
    }

    /// True if `player` still has a ribbon on the board.
    /// Gates eval, search, and whose turn we simulate in FFA.
    #[inline]
    fn is_alive(self, player: usize) -> bool {
        self.alive_mask & (1 << player) != 0
    }

    /// Mark `(col, row)` as this player’s new head (and a trail cell).
    /// Also sets the alive bit. [`Tracker`] uses this for every newly reported
    /// head; [`State::apply`] uses it for a legal step in search.
    fn occupy(&mut self, player: usize, col: i32, row: i32) {
        self.occupied.set(col, row);
        self.trail[player].set(col, row);
        self.head_x[player] = col as i8;
        self.head_y[player] = row as i8;
        self.alive_mask |= 1 << player;
    }

    /// CodinGame death: the whole ribbon disappears and those cells become empty.
    /// Idempotent if already dead. [`Tracker`] calls this on four `-1`s; FFA
    /// search calls it when a simulated opponent has no legal move.
    fn kill(&mut self, player: usize) {
        if !self.is_alive(player) {
            return;
        }
        // Trails never overlap, so XOR removes exactly this player’s cells.
        self.occupied.xor_with(self.trail[player]);
        self.trail[player] = RowBits::empty();
        self.alive_mask &= !(1 << player);
        self.head_x[player] = -1;
        self.head_y[player] = -1;
    }

    /// True if direction `dir` (0=UP .. 3=RIGHT) from this player’s head is
    /// on-board and not already occupied.
    /// Building block for [`State::legal_moves`] and for validating a chosen
    /// direction before [`State::apply`] in the bench / fill rollout.
    #[inline]
    fn is_legal_dir(self, player: usize, dir: usize) -> bool {
        let col = self.head_x[player] as i32 + DIR_X[dir];
        let row = self.head_y[player] as i32 + DIR_Y[dir];
        in_bounds(col, row) && !self.occupied.is_set(col, row)
    }

    /// Every legal direction index for `player`, plus how many (`0..=4`).
    /// Unused slots in the array are left as 0 and must not be read past `count`.
    /// Root of [`choose_move`] and every search / greedy / fill policy.
    fn legal_moves(self, player: usize) -> ([u8; 4], usize) {
        let mut moves = [0u8; 4];
        let mut count = 0;
        for dir in 0..4 {
            if self.is_legal_dir(player, dir) {
                moves[count] = dir as u8;
                count += 1;
            }
        }
        (moves, count)
    }

    /// Step `player` one cell in `dir`. Returns false (and does nothing) if that
    /// cell is off-board or occupied. On success the old head stays as trail.
    /// Search, greedy, fill, and the local bench all advance the board this way.
    fn apply(&mut self, player: usize, dir: usize) -> bool {
        let col = self.head_x[player] as i32 + DIR_X[dir];
        let row = self.head_y[player] as i32 + DIR_Y[dir];
        if !in_bounds(col, row) || self.occupied.is_set(col, row) {
            return false;
        }
        self.occupy(player, col, row);
        true
    }

    /// Undo a successful [`State::apply`]: clear the new head cell, restore the old head.
    /// The old cell stays occupied (it is still trail).
    /// Lets 1v1 search try sibling moves on the same `State` without cloning.
    fn undo_step(&mut self, player: usize, old_col: i8, old_row: i8) {
        let col = self.head_x[player] as i32;
        let row = self.head_y[player] as i32;
        self.occupied.clear(col, row);
        self.trail[player].clear(col, row);
        self.head_x[player] = old_col;
        self.head_y[player] = old_row;
    }
}

/// Scratch space for BFS / flood / Voronoi so we do not allocate in search.
///
/// One instance is created in [`codingame`] / `--bench` / `--profile` and
/// reused every turn. Eval is the NPS bottleneck; these buffers avoid 600-cell
/// `Vec` allocations inside [`negamax_1v1`] leaves.
struct Scratch {
    /// Per-player distance maps, `UNREACHABLE` if not reached.
    distance: [[u16; BOARD_CELLS]; MAX_PLAYERS],
    bfs_queue: [u16; BOARD_CELLS],
    /// Generation stamp per cell; compared to `visit_generation` instead of clearing.
    visited_stamp: [u32; BOARD_CELLS],
    visit_generation: u32,
    /// Voronoi owner of each empty cell, or -1.
    owner: [i8; BOARD_CELLS],
}

impl Scratch {
    /// Allocate one reusable buffer set. Call once per process / game loop
    /// ([`codingame`], [`bench`], [`profile`]).
    fn new() -> Self {
        Self {
            distance: [[UNREACHABLE; BOARD_CELLS]; MAX_PLAYERS],
            bfs_queue: [0; BOARD_CELLS],
            visited_stamp: [0; BOARD_CELLS],
            visit_generation: 1,
            owner: [-1; BOARD_CELLS],
        }
    }

    /// Advance the visit generation used by flood/fill.
    /// If the `u32` wraps to 0, clear stamps so we never match a stale 0.
    /// [`flood_count`] and [`approx_fill`] call this instead of zeroing 600 cells.
    fn next_generation(&mut self) -> u32 {
        self.visit_generation = self.visit_generation.wrapping_add(1);
        if self.visit_generation == 0 {
            self.visited_stamp.fill(0);
            self.visit_generation = 1;
        }
        self.visit_generation
    }
}

/// Breadth-first distances from `player` into empty cells.
///
/// The head cell itself is occupied, so the search starts from its empty
/// 4-neighbours at distance 1. Unreached cells stay [`UNREACHABLE`].
/// `distance` and `queue` are caller-owned so Voronoi can reuse one queue.
///
/// **Where:** [`compute_voronoi`] (once per living player), [`main_opponent`],
/// and [`shares_space`].
/// **Why:** territory, “can we still meet?”, and “who is closest?” all need
/// empty-cell distances from a head.
fn bfs_from_player(
    state: &State,
    player: usize,
    distance: &mut [u16; BOARD_CELLS],
    queue: &mut [u16; BOARD_CELLS],
) {
    distance.fill(UNREACHABLE);
    if !state.is_alive(player) {
        return;
    }
    let mut queue_head = 0usize;
    let mut queue_tail = 0usize;
    let head_col = state.head_x[player] as i32;
    let head_row = state.head_y[player] as i32;
    // Seed the queue with every empty neighbour of the head.
    for dir in 0..4 {
        let col = head_col + DIR_X[dir];
        let row = head_row + DIR_Y[dir];
        if in_bounds(col, row) && !state.occupied.is_set(col, row) {
            let index = cell_index(col, row);
            distance[index] = 1;
            queue[queue_tail] = index as u16;
            queue_tail += 1;
        }
    }
    while queue_head < queue_tail {
        let index = queue[queue_head] as usize;
        queue_head += 1;
        let (col, row) = coords_from_index(index);
        let next_dist = distance[index] + 1;
        for dir in 0..4 {
            let next_col = col + DIR_X[dir];
            let next_row = row + DIR_Y[dir];
            if in_bounds(next_col, next_row) && !state.occupied.is_set(next_col, next_row) {
                let next_index = cell_index(next_col, next_row);
                if distance[next_index] == UNREACHABLE {
                    distance[next_index] = next_dist;
                    queue[queue_tail] = next_index as u16;
                    queue_tail += 1;
                }
            }
        }
    }
}

/// Count empty cells reachable from `player`, ignoring who else can reach them.
///
/// This is a chamber-size / remaining-space metric, not Voronoi. Uses
/// generation stamps on `scratch` so we never `fill()` the visit array.
///
/// **Where:** [`greedy_direction`], [`fill_direction`], the FFA 2-ply score in
/// [`choose_move`], [`endgame_eval`], and the `--bench` 900-turn tiebreak.
/// **Why:** “how much space is left in *my* pocket?” — used to hug walls,
/// avoid sealing off rooms, and pick a winner if the bench hits the turn cap.
fn flood_count(state: &State, player: usize, scratch: &mut Scratch) -> i32 {
    if !state.is_alive(player) {
        return 0;
    }
    let generation = scratch.next_generation();
    let mut queue_head = 0usize;
    let mut queue_tail = 0usize;
    let head_col = state.head_x[player] as i32;
    let head_row = state.head_y[player] as i32;
    for dir in 0..4 {
        let col = head_col + DIR_X[dir];
        let row = head_row + DIR_Y[dir];
        if in_bounds(col, row) && !state.occupied.is_set(col, row) {
            let index = cell_index(col, row);
            if scratch.visited_stamp[index] != generation {
                scratch.visited_stamp[index] = generation;
                scratch.bfs_queue[queue_tail] = index as u16;
                queue_tail += 1;
            }
        }
    }
    while queue_head < queue_tail {
        let index = scratch.bfs_queue[queue_head] as usize;
        queue_head += 1;
        let (col, row) = coords_from_index(index);
        for dir in 0..4 {
            let next_col = col + DIR_X[dir];
            let next_row = row + DIR_Y[dir];
            if in_bounds(next_col, next_row) && !state.occupied.is_set(next_col, next_row) {
                let next_index = cell_index(next_col, next_row);
                if scratch.visited_stamp[next_index] != generation {
                    scratch.visited_stamp[next_index] = generation;
                    scratch.bfs_queue[queue_tail] = next_index as u16;
                    queue_tail += 1;
                }
            }
        }
    }
    // Every dequeued / enqueued cell was unique, so the tail is the count.
    queue_tail as i32
}

/// Number of empty on-board 4-neighbours of `(col, row)`.
/// Higher degree means a more “open” cell (more future options).
///
/// **Where:** [`wall_neighbor_count`] (as `4 - this`) and Voronoi `edge_sum`
/// inside [`compute_voronoi`].
/// **Why:** leftover open edges in your territory induce wall-hugging when
/// two Voronoi scores are close (a1k0n).
fn empty_degree(state: &State, col: i32, row: i32) -> i32 {
    let mut count = 0;
    for dir in 0..4 {
        let next_col = col + DIR_X[dir];
        let next_row = row + DIR_Y[dir];
        if in_bounds(next_col, next_row) && !state.occupied.is_set(next_col, next_row) {
            count += 1;
        }
    }
    count
}

/// Occupied or off-board neighbours: `4 - empty_degree`.
/// Wall-hugging prefers cells with a high count (ride existing trails / edges).
///
/// **Where:** move ordering, [`greedy_direction`], [`fill_direction`], isolated
/// [`eval_1v1`], the 80-step endgame score in [`choose_move`], and `bot_wallhug`.
/// **Why:** riding a wall leaves more empty cells in front of you; a cheap
/// heuristic that is correct often enough to use at every ply.
fn wall_neighbor_count(state: &State, col: i32, row: i32) -> i32 {
    4 - empty_degree(state, col, row)
}

/// True if occupying `(col, row)` would split its empty 4-neighbours into
/// disconnected pockets.
///
/// We look at the 8-cell ring around the cell: if you cannot walk from one
/// open 4-neighbour to another along empty ring cells, this is a local cut.
///
/// **Where:** only [`dfs_fill_score`].
/// **Why:** a flood-fill count overcounts space behind a choke (you cannot
/// take both branches). At a cut we take `max(pocket)` instead of the sum.
fn is_local_cut(state: &State, col: i32, row: i32) -> bool {
    let mut open_neighbors = [(0i32, 0i32); 4];
    let mut open_count = 0;
    for dir in 0..4 {
        let next_col = col + DIR_X[dir];
        let next_row = row + DIR_Y[dir];
        if in_bounds(next_col, next_row) && !state.occupied.is_set(next_col, next_row) {
            open_neighbors[open_count] = (next_col, next_row);
            open_count += 1;
        }
    }
    // 0–1 exits: nothing to split. 4 exits: the ring almost always connects them.
    if open_count <= 1 || open_count >= 4 {
        return false;
    }

    // Eight cells around (col, row), clockwise from NW.
    const RING: [(i32, i32); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
    ];
    let mut ring_empty = [false; 8];
    for ring_i in 0..8 {
        let ring_col = col + RING[ring_i].0;
        let ring_row = row + RING[ring_i].1;
        ring_empty[ring_i] =
            in_bounds(ring_col, ring_row) && !state.occupied.is_set(ring_col, ring_row);
    }

    // Map each 4-neighbour onto a ring slot: UP=1, RIGHT=3, DOWN=5, LEFT=7.
    /// Which of the 8 ring indices corresponds to this orthogonal neighbour.
    /// Local to [`is_local_cut`]: UP=1, RIGHT=3, DOWN=5, LEFT=7 on the clockwise ring.
    fn neighbor_to_ring(col: i32, row: i32, neighbor_col: i32, neighbor_row: i32) -> usize {
        if neighbor_col == col && neighbor_row == row - 1 {
            1
        } else if neighbor_col == col + 1 && neighbor_row == row {
            3
        } else if neighbor_col == col && neighbor_row == row + 1 {
            5
        } else {
            7
        }
    }

    // Start BFS from the first open neighbour that sits on an empty ring cell.
    let mut start_slot = -1i32;
    for neighbor_i in 0..open_count {
        let (neighbor_col, neighbor_row) = open_neighbors[neighbor_i];
        let slot = neighbor_to_ring(col, row, neighbor_col, neighbor_row);
        if ring_empty[slot] {
            start_slot = slot as i32;
            break;
        }
    }
    if start_slot < 0 {
        return true;
    }

    let mut seen = [false; 8];
    let mut stack = [0i32; 8];
    let mut stack_len = 1;
    stack[0] = start_slot;
    seen[start_slot as usize] = true;
    while stack_len > 0 {
        stack_len -= 1;
        let slot = stack[stack_len] as usize;
        for delta in [-1i32, 1] {
            let neighbor_slot = ((slot as i32 + delta + 8) % 8) as usize;
            // Only walk 4-adjacent ring cells (skip diagonal-only pairs).
            let (a_col, a_row) = (col + RING[slot].0, row + RING[slot].1);
            let (b_col, b_row) = (col + RING[neighbor_slot].0, row + RING[neighbor_slot].1);
            if (a_col - b_col).abs() + (a_row - b_row).abs() != 1 {
                continue;
            }
            if ring_empty[neighbor_slot] && !seen[neighbor_slot] {
                seen[neighbor_slot] = true;
                stack[stack_len] = neighbor_slot as i32;
                stack_len += 1;
            }
        }
    }

    // Cut if any open 4-neighbour was not reached through the ring.
    for neighbor_i in 0..open_count {
        let (neighbor_col, neighbor_row) = open_neighbors[neighbor_i];
        let slot = neighbor_to_ring(col, row, neighbor_col, neighbor_row);
        if !seen[slot] {
            return true;
        }
    }
    false
}

/// Estimate how many empty cells a perfect space-fill from `(col, row)` can take.
///
/// Recurses through unvisited empty 4-neighbours. At a local cut we can only
/// commit to one pocket, so we take the max; otherwise we sum (the region is
/// still one connected “snake” we can traverse). Returns at least 1 for the
/// current cell. `generation` is the visit stamp from [`approx_fill`].
///
/// **Where:** only [`approx_fill`] (once per empty neighbour of the head).
/// **Why:** when 1v1 is cut off, remaining *life* is fillable cells, not raw
/// flood size. This is a cheap stand-in for a chamber tree.
fn dfs_fill_score(
    state: &State,
    col: i32,
    row: i32,
    scratch: &mut Scratch,
    generation: u32,
) -> i32 {
    let index = cell_index(col, row);
    scratch.visited_stamp[index] = generation;
    let mut next_cells = [(0i32, 0i32); 4];
    let mut next_count = 0;
    for dir in 0..4 {
        let next_col = col + DIR_X[dir];
        let next_row = row + DIR_Y[dir];
        if in_bounds(next_col, next_row)
            && !state.occupied.is_set(next_col, next_row)
            && scratch.visited_stamp[cell_index(next_col, next_row)] != generation
        {
            next_cells[next_count] = (next_col, next_row);
            next_count += 1;
        }
    }
    if next_count == 0 {
        return 1;
    }
    // A single exit: we must take it, no branching choice.
    if next_count == 1 {
        return 1 + dfs_fill_score(state, next_cells[0].0, next_cells[0].1, scratch, generation);
    }
    // A cut means walking here walls off some neighbours from each other.
    if is_local_cut(state, col, row) {
        let mut best_pocket = 0;
        for pocket_i in 0..next_count {
            let (pocket_col, pocket_row) = next_cells[pocket_i];
            if scratch.visited_stamp[cell_index(pocket_col, pocket_row)] != generation {
                let pocket = dfs_fill_score(state, pocket_col, pocket_row, scratch, generation);
                if pocket > best_pocket {
                    best_pocket = pocket;
                }
            }
        }
        1 + best_pocket
    } else {
        let mut total = 1;
        for next_i in 0..next_count {
            let (next_col, next_row) = next_cells[next_i];
            if scratch.visited_stamp[cell_index(next_col, next_row)] != generation {
                total += dfs_fill_score(state, next_col, next_row, scratch, generation);
            }
        }
        total
    }
}

/// How many cells `player` can still claim if they fill their chamber greedily.
///
/// Tries each empty neighbour of the head and returns the best
/// [`dfs_fill_score`].
///
/// **Where:** [`eval_1v1`] when the duelists no longer share space;
/// [`fill_direction`] (endgame policy); unused [`endgame_eval`].
/// **Why:** after a cut, Voronoi is meaningless — the game is “who fills more
/// of their own pocket.” This estimate is the leaf score for that regime.
fn approx_fill(state: &State, player: usize, scratch: &mut Scratch) -> i32 {
    if !state.is_alive(player) {
        return 0;
    }
    let generation = scratch.next_generation();
    let head_col = state.head_x[player] as i32;
    let head_row = state.head_y[player] as i32;
    let mut best = 0;
    for dir in 0..4 {
        let col = head_col + DIR_X[dir];
        let row = head_row + DIR_Y[dir];
        if in_bounds(col, row)
            && !state.occupied.is_set(col, row)
            && scratch.visited_stamp[cell_index(col, row)] != generation
        {
            let score = dfs_fill_score(state, col, row, scratch, generation);
            if score > best {
                best = score;
            }
        }
    }
    best
}

/// Result of partitioning empty cells by who can reach them first.
/// Produced by [`compute_voronoi`]; consumed by [`eval_1v1`], [`eval_ffa`],
/// and [`choose_move`] (the `still_connected` flag).
struct Voronoi {
    /// Empty cells this player uniquely reaches strictly sooner than anyone else.
    territory: [i32; MAX_PLAYERS],
    /// Sum of [`empty_degree`] on those unique cells (open frontier is better).
    edge_sum: [i32; MAX_PLAYERS],
    /// Empty cells this player can reach at all, including ties.
    reachable: [i32; MAX_PLAYERS],
    /// True if at least two living players can both reach some empty cell.
    still_connected: bool,
}

/// Multi-source Voronoi from every living head.
///
/// Runs one BFS per living player, then for each empty cell records the closest
/// player. Ties are unowned (`scratch.owner = -1`) so neither side gets the
/// territory. Also sets `still_connected` if any cell is reachable by two
/// living players.
///
/// **Where:** [`eval_1v1`] and [`eval_ffa`] every leaf; [`choose_move`] as a
/// second opinion before declaring 1v1 chambers separated.
/// **Why:** in the open game, “land I uniquely reach first” is the standard
/// Tron heuristic (a1k0n / GAI Challenge). Eval is dominated by this call.
fn compute_voronoi(state: &State, scratch: &mut Scratch) -> Voronoi {
    let player_count = state.player_count as usize;
    for player in 0..player_count {
        if state.is_alive(player) {
            bfs_from_player(
                state,
                player,
                &mut scratch.distance[player],
                &mut scratch.bfs_queue,
            );
        } else {
            scratch.distance[player].fill(UNREACHABLE);
        }
    }
    let mut territory = [0i32; MAX_PLAYERS];
    let mut edge_sum = [0i32; MAX_PLAYERS];
    let mut reachable = [0i32; MAX_PLAYERS];
    let mut still_connected = false;
    scratch.owner.fill(-1);

    for row in 0..HEIGHT {
        for col in 0..WIDTH {
            if state.occupied.is_set(col, row) {
                continue;
            }
            let index = cell_index(col, row);
            let mut best_dist = UNREACHABLE;
            let mut best_player = -1i8;
            let mut tie_count = 0;
            for player in 0..player_count {
                if !state.is_alive(player) {
                    continue;
                }
                let dist = scratch.distance[player][index];
                if dist < UNREACHABLE {
                    reachable[player] += 1;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_player = player as i8;
                    tie_count = 1;
                } else if dist == best_dist && dist < UNREACHABLE {
                    tie_count += 1;
                }
            }
            if best_dist == UNREACHABLE {
                continue;
            }
            // Interaction test: two living players can both path to this cell.
            let mut reacher_count = 0;
            for player in 0..player_count {
                if state.is_alive(player) && scratch.distance[player][index] < UNREACHABLE {
                    reacher_count += 1;
                }
            }
            if reacher_count >= 2 {
                still_connected = true;
            }
            if tie_count == 1 {
                let owner = best_player as usize;
                territory[owner] += 1;
                scratch.owner[index] = best_player;
                edge_sum[owner] += empty_degree(state, col, row);
            }
        }
    }
    Voronoi {
        territory,
        edge_sum,
        reachable,
        still_connected,
    }
}

/// How many of the four directions are legal for `player` (0–4).
/// A mobility of 1 is a forced move; 0 is death next turn.
///
/// **Where:** [`eval_1v1`] and [`eval_ffa`] (and unused [`endgame_eval`]).
/// **Why:** getting boxed in (mobility 1→0) is often worse than losing a
/// couple of Voronoi cells, so it is a cheap extra term in the leaf.
fn mobility(state: &State, player: usize) -> i32 {
    let mut count = 0;
    for dir in 0..4 {
        if state.is_legal_dir(player, dir) {
            count += 1;
        }
    }
    count
}

/// Pick the opponent we should treat as the main threat in a 3–4 player game.
///
/// Prefers the living player whose head is closest in our empty-cell BFS
/// (distance 0 if the heads are adjacent). If nobody shares reachable space,
/// fall back to nearest Manhattan distance. Returns `None` only if we are alone.
///
/// **Where:** [`choose_move`] when 3+ are alive, to pick an `opponent` for
/// move-ordering (aim at their head). Not used for the FFA eval itself.
/// **Why:** even in FFA we still sort moves “toward someone”; this picks who.
fn main_opponent(state: &State, our_id: usize, scratch: &mut Scratch) -> Option<usize> {
    let player_count = state.player_count as usize;
    let mut best_player = None;
    let mut best_dist = UNREACHABLE;
    bfs_from_player(
        state,
        our_id,
        &mut scratch.distance[our_id],
        &mut scratch.bfs_queue,
    );
    for player in 0..player_count {
        if player == our_id || !state.is_alive(player) {
            continue;
        }
        let head_col = state.head_x[player] as i32;
        let head_row = state.head_y[player] as i32;
        let mut dist = UNREACHABLE;
        for dir in 0..4 {
            let col = head_col + DIR_X[dir];
            let row = head_row + DIR_Y[dir];
            if in_bounds(col, row) {
                // Adjacent heads (our head occupies their neighbour cell).
                if state.occupied.is_set(col, row)
                    && col == state.head_x[our_id] as i32
                    && row == state.head_y[our_id] as i32
                {
                    dist = 0;
                } else if !state.occupied.is_set(col, row) {
                    dist = dist.min(scratch.distance[our_id][cell_index(col, row)]);
                }
            }
        }
        if dist < best_dist {
            best_dist = dist;
            best_player = Some(player);
        } else if dist == UNREACHABLE && best_dist == UNREACHABLE {
            // Nobody shares space yet: pick the geographically nearest head.
            let manhattan = (state.head_x[our_id] as i32 - head_col).abs()
                + (state.head_y[our_id] as i32 - head_row).abs();
            let previous = best_player.map(|previous_player| {
                (state.head_x[our_id] as i32 - state.head_x[previous_player] as i32).abs()
                    + (state.head_y[our_id] as i32 - state.head_y[previous_player] as i32).abs()
            });
            if best_player.is_none() || manhattan < previous.unwrap_or(i32::MAX) {
                best_player = Some(player);
            }
        }
    }
    best_player
}

/// True if `our_id` and `opponent` can still meet through empty cells.
///
/// Adjacent heads count as connected. Otherwise we BFS from us and check
/// whether any empty neighbour of the opponent is reachable.
///
/// **Where:** [`choose_move`] (together with Voronoi `still_connected`) to
/// decide open 1v1 vs chamber-fill; [`eval_1v1`] to pick Voronoi vs fill terms.
/// **Why:** a false “we are cut off” makes minimax ignore a fight we can still
/// reach. We would rather keep fighting than fill too early.
fn shares_space(state: &State, our_id: usize, opponent: usize, scratch: &mut Scratch) -> bool {
    if !state.is_alive(our_id) || !state.is_alive(opponent) {
        return false;
    }
    let our_col = state.head_x[our_id] as i32;
    let our_row = state.head_y[our_id] as i32;
    let opp_col = state.head_x[opponent] as i32;
    let opp_row = state.head_y[opponent] as i32;
    if (our_col - opp_col).abs() + (our_row - opp_row).abs() == 1 {
        return true;
    }
    // Can we path to any empty cell next to their head?
    bfs_from_player(
        state,
        our_id,
        &mut scratch.distance[our_id],
        &mut scratch.bfs_queue,
    );
    for dir in 0..4 {
        let col = opp_col + DIR_X[dir];
        let row = opp_row + DIR_Y[dir];
        if in_bounds(col, row)
            && !state.occupied.is_set(col, row)
            && scratch.distance[our_id][cell_index(col, row)] < UNREACHABLE
        {
            return true;
        }
    }
    false
}

/// Count our Voronoi cells that share an edge with an opponent Voronoi cell.
///
/// A larger front means more contact / contest. Requires [`compute_voronoi`]
/// to have filled `scratch.owner`.
///
/// **Where:** only the open-game branch of [`eval_1v1`].
/// **Why:** without it, search is happy to wander into dead space while the
/// opponent takes the middle. A small bonus keeps us on the battle line.
fn battlefront(state: &State, our_id: usize, opponent: usize, scratch: &Scratch) -> i32 {
    let mut front = 0;
    for row in 0..HEIGHT {
        for col in 0..WIDTH {
            if state.occupied.is_set(col, row) {
                continue;
            }
            let index = cell_index(col, row);
            if scratch.owner[index] != our_id as i8 {
                continue;
            }
            for dir in 0..4 {
                let next_col = col + DIR_X[dir];
                let next_row = row + DIR_Y[dir];
                if in_bounds(next_col, next_row)
                    && !state.occupied.is_set(next_col, next_row)
                    && scratch.owner[cell_index(next_col, next_row)] == opponent as i8
                {
                    front += 1;
                    break;
                }
            }
        }
    }
    front
}

/// Static 1v1 evaluation from `our_id`’s point of view (positive = good for our player).
///
/// Terminal: we are dead → `-MATE_SCORE + ply`; they are dead → `MATE_SCORE - ply`
/// (adding ply prefers faster wins / slower losses).
///
/// If the two bikes can no longer reach each other, compare chamber fill
/// estimates plus wall-hugging. Otherwise combine Voronoi territory, frontier
/// edges, raw reach, mobility, battlefront length, and a center bias that
/// fades as the board fills.
///
/// **Where:** leaves of [`negamax_1v1`]; [`oneply_direction`] when exactly one
/// rival is alive; `--bench` `bot_voronoi1`; unused [`rollout_score`].
/// **Why:** this is *the* 1v1 heuristic. Search only looks a few plies; the
/// leaf has to encode “who owns the remaining empty board.” An early version
/// added a huge separated-bonus here and minimax hallucinated fake cuts —
/// do not reintroduce that.
fn eval_1v1(state: &State, our_id: usize, opponent: usize, ply: i32, scratch: &mut Scratch) -> i32 {
    if !state.is_alive(our_id) {
        return -MATE_SCORE + ply;
    }
    if !state.is_alive(opponent) {
        return MATE_SCORE - ply;
    }
    let our_mobility = mobility(state, our_id);
    let opp_mobility = mobility(state, opponent);
    if !shares_space(state, our_id, opponent, scratch) {
        // Isolated chambers: the game is “who fills more of their own space”.
        let our_fill = approx_fill(state, our_id, scratch);
        let opp_fill = approx_fill(state, opponent, scratch);
        let fill_diff = our_fill - opp_fill;
        let hug = wall_neighbor_count(
            state,
            state.head_x[our_id] as i32,
            state.head_y[our_id] as i32,
        ) - wall_neighbor_count(
            state,
            state.head_x[opponent] as i32,
            state.head_y[opponent] as i32,
        );
        return fill_diff.signum() * 80 + fill_diff * 60 + hug * 5 + (our_mobility - opp_mobility);
    }
    let voronoi = compute_voronoi(state, scratch);
    let territory = voronoi.territory[our_id] - voronoi.territory[opponent];
    let edges = voronoi.edge_sum[our_id] - voronoi.edge_sum[opponent];
    let reach = voronoi.reachable[our_id] - voronoi.reachable[opponent];
    let mobility_diff = our_mobility - opp_mobility;
    let front = battlefront(state, our_id, opponent, scratch);
    let head_col = state.head_x[our_id] as i32;
    let head_row = state.head_y[our_id] as i32;
    let center_penalty = -((head_col - 14).abs() + (head_row - 9).abs());
    let occupied_count: i32 = (0..20)
        .map(|row| state.occupied.bits[row].count_ones())
        .sum::<u32>() as i32;
    // Center control matters early; fade it out as the board fills.
    let center_weight = (500 - occupied_count).max(0) / 80;
    territory * 50
        + edges * 12
        + reach * 3
        + mobility_diff * 6
        + front * 4
        + center_penalty * center_weight
}

/// Free-for-all evaluation from `our_id`’s point of view.
///
/// We are not trying to 1v1 a specific rival. Score our reachable / unique
/// territory, subtract the strongest remaining opponent’s territory and reach,
/// and add mobility so we do not get boxed in. Death / sole survivor use mate
/// scores like [`eval_1v1`].
///
/// **Where:** the 2-ply FFA branch of [`choose_move`]; [`oneply_direction`]
/// when 2+ rivals are alive; unused [`search_ffa`] / [`rollout_score`].
/// **Why:** deep 1v1 minimax in a 3–4 player game treats others as frozen walls
/// and suicides. This leaf prefers “survive with space” over picking a fight.
fn eval_ffa(state: &State, our_id: usize, ply: i32, scratch: &mut Scratch) -> i32 {
    if !state.is_alive(our_id) {
        return -MATE_SCORE + ply;
    }
    if state.alive_mask.count_ones() == 1 {
        return MATE_SCORE - ply;
    }
    let voronoi = compute_voronoi(state, scratch);
    let mut best_other_territory = 0;
    let mut best_other_reach = 0;
    let player_count = state.player_count as usize;
    for player in 0..player_count {
        if player == our_id || !state.is_alive(player) {
            continue;
        }
        best_other_territory = best_other_territory.max(voronoi.territory[player]);
        best_other_reach = best_other_reach.max(voronoi.reachable[player]);
    }
    voronoi.reachable[our_id] * 40 + voronoi.territory[our_id] * 25 - best_other_territory * 10
        + -best_other_reach * 4
        + mobility(state, our_id) * 20
}

/// Clock and move-ordering state for one call to [`choose_move`].
///
/// `killers[ply % 64]` stores up to two direction indices that caused a beta
/// cutoff at that ply; they are tried early next time. `timed_out` is sticky:
/// once set, search returns [`TIMEOUT_SCORE`] so the caller can discard a
/// partial iteration.
///
/// **Where:** constructed in [`choose_move`] for the 1v1 iterative-deepening
/// loop, then passed into [`negamax_1v1`]. Unused searches take one too.
/// **Why:** CodinGame kills a late bot; we must stop on a completed depth,
/// not mid-ply, and still undo every apply.
struct Search {
    deadline: Instant,
    nodes: u64,
    timed_out: bool,
    /// Two killer direction indices per ply slot; `NO_MOVE` if empty.
    killers: [[u8; 2]; 64],
}

impl Search {
    /// Start a search that must finish by `now + budget`.
    /// [`choose_move`] passes whatever is left of the turn after bookkeeping.
    fn new(budget: Duration) -> Self {
        Self {
            deadline: Instant::now() + budget,
            nodes: 0,
            timed_out: false,
            killers: [[NO_MOVE; 2]; 64],
        }
    }

    /// Count a visited node. Every 16 nodes, compare wall clock to `deadline`.
    /// Checking every node is too expensive relative to our tiny branching factor.
    /// Called at the top of [`negamax_1v1`] (and the unused searches).
    #[inline]
    fn check_time(&mut self) {
        self.nodes += 1;
        if (self.nodes & 15) == 0 && Instant::now() >= self.deadline {
            self.timed_out = true;
        }
    }
}

/// Sort `moves[0..move_count]` so the most promising directions are tried first.
///
/// Heuristic (higher is better): previous principal variation, killer move,
/// continuing in `last_dir`, wall-hugging, then closer to `(target_col, target_row)`
/// (usually the opponent’s head). Insertion sort is enough because `move_count ≤ 4`.
///
/// **Where:** [`choose_move`] at the root (each ID iteration) and inside
/// [`negamax_1v1`] at every interior node; unused [`search_endgame`].
/// **Why:** alpha-beta only cuts if we try a good move first. Branching is 2–4,
/// so this is cheap and still pays for itself.
fn order_moves(
    state: &State,
    player: usize,
    moves: &mut [u8],
    move_count: usize,
    pv_dir: u8,
    killer_dir: u8,
    last_dir: u8,
    target_col: i32,
    target_row: i32,
) {
    let head_col = state.head_x[player] as i32;
    let head_row = state.head_y[player] as i32;
    let mut keys = [0i32; 4];
    for move_i in 0..move_count {
        let dir = moves[move_i];
        let col = head_col + DIR_X[dir as usize];
        let row = head_row + DIR_Y[dir as usize];
        let mut key = 0;
        if dir == pv_dir {
            key += 10_000;
        }
        if dir == killer_dir {
            key += 3_000;
        }
        if dir == last_dir {
            key += 40;
        }
        key += wall_neighbor_count(state, col, row) * 8;
        key -= (col - target_col).abs() + (row - target_row).abs();
        keys[move_i] = key;
    }
    // Insertion sort, descending key.
    for move_i in 1..move_count {
        let mut insert_at = move_i;
        while insert_at > 0 && keys[insert_at] > keys[insert_at - 1] {
            keys.swap(insert_at, insert_at - 1);
            moves.swap(insert_at, insert_at - 1);
            insert_at -= 1;
        }
    }
}

/// Greedy one-ply policy: try each legal step and pick the one that maximises
/// remaining flood-fill size plus a small wall-hug bonus.
///
/// **Where:** unused [`search_ffa`] (what “they” would do); `--bench`
/// `bot_greedy`; `--profile` walks 30 plies of this to reach a midgame.
/// **Why:** a cheap space-taking model. Production FFA uses the stronger
/// [`oneply_direction`] for opponent replies instead.
/// Returns `None` if `player` has no legal move.
fn greedy_direction(state: &State, player: usize, scratch: &mut Scratch) -> Option<u8> {
    let (legal, move_count) = state.legal_moves(player);
    if move_count == 0 {
        return None;
    }
    let mut best_dir = legal[0];
    let mut best_score = i32::MIN;
    for move_i in 0..move_count {
        let dir = legal[move_i] as usize;
        let mut after = *state;
        after.apply(player, dir);
        let flood = flood_count(&after, player, scratch);
        let col = after.head_x[player] as i32;
        let row = after.head_y[player] as i32;
        let score = flood * 20 + wall_neighbor_count(&after, col, row) * 3;
        if score > best_score {
            best_score = score;
            best_dir = legal[move_i];
        }
    }
    Some(best_dir)
}

/// Negamax alpha-beta for a 1v1 duel.
///
/// Score is always from `to_move`’s point of view; the caller negates. `our_id`
/// and `opponent` are the two colours at the root and never swap — [`eval_1v1`]
/// is written from `our_id`’s side, then flipped if the opponent is to move.
///
/// `depth` is remaining plies to a leaf. `ply` is distance from the root
/// (used to prefer faster mates). `last_dir_*` help move ordering (prefer
/// continuing straight). Returns [`TIMEOUT_SCORE`] if the budget expired;
/// the caller must ignore that iteration.
///
/// **Where:** only the iterative-deepening loop in [`choose_move`], and only
/// when exactly two bikes are alive and still share space.
/// **Why:** sequential Tron is a two-player game once it is a duel. A few
/// plies of minimax plus Voronoi leaves beat greedy 1-ply; FFA must not
/// call this (it treats extra bikes as frozen walls).
fn negamax_1v1(
    state: &mut State,
    our_id: usize,
    opponent: usize,
    to_move: usize,
    depth: i32,
    ply: i32,
    mut alpha: i32,
    beta: i32,
    last_dir_ours: u8,
    last_dir_opponent: u8,
    search: &mut Search,
    scratch: &mut Scratch,
) -> i32 {
    search.check_time();
    if search.timed_out {
        return TIMEOUT_SCORE;
    }
    let next_player = if to_move == our_id { opponent } else { our_id };
    if !state.is_alive(our_id) {
        return -MATE_SCORE + ply;
    }
    if !state.is_alive(opponent) {
        return MATE_SCORE - ply;
    }

    // Leaf: side to move with no reply loses; otherwise static eval, flipped
    // so the value is from `to_move`’s perspective.
    if depth <= 0 {
        let move_count = state.legal_moves(to_move).1;
        if move_count == 0 {
            return if to_move == our_id {
                -MATE_SCORE + ply
            } else {
                MATE_SCORE - ply
            };
        }
        let score = eval_1v1(state, our_id, opponent, ply, scratch);
        return if to_move == our_id { score } else { -score };
    }

    let (mut moves, move_count) = state.legal_moves(to_move);
    if move_count == 0 {
        if to_move == our_id {
            return -MATE_SCORE + ply;
        } else {
            return MATE_SCORE - ply;
        }
    }

    let (target_col, target_row) = (
        state.head_x[next_player] as i32,
        state.head_y[next_player] as i32,
    );
    let last_dir = if to_move == our_id {
        last_dir_ours
    } else {
        last_dir_opponent
    };
    let killer_dir = search.killers[ply as usize % 64][0];
    order_moves(
        state, to_move, &mut moves, move_count, NO_MOVE, killer_dir, last_dir, target_col,
        target_row,
    );

    let mut best_score = -MATE_SCORE * 2;
    for move_i in 0..move_count {
        let dir = moves[move_i] as usize;
        let old_col = state.head_x[to_move];
        let old_row = state.head_y[to_move];
        state.apply(to_move, dir);
        let (next_last_ours, next_last_opponent) = if to_move == our_id {
            (moves[move_i], last_dir_opponent)
        } else {
            (last_dir_ours, moves[move_i])
        };
        let score = -negamax_1v1(
            state,
            our_id,
            opponent,
            next_player,
            depth - 1,
            ply + 1,
            -beta,
            -alpha,
            next_last_ours,
            next_last_opponent,
            search,
            scratch,
        );
        state.undo_step(to_move, old_col, old_row);
        if search.timed_out {
            return TIMEOUT_SCORE;
        }
        if score > best_score {
            best_score = score;
        }
        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            // Beta cutoff: remember this move as a killer for this ply.
            let slot = ply as usize % 64;
            if search.killers[slot][0] != moves[move_i] {
                search.killers[slot][1] = search.killers[slot][0];
                search.killers[slot][0] = moves[move_i];
            }
            break;
        }
    }
    best_score
}

/// Leaf evaluation used only by the unused [`search_endgame`].
/// Mix of fill estimate, flood size, wall-hug, and remaining mobility.
/// Not on the production path: isolated chambers use the 80-step
/// [`fill_direction`] rollout in [`choose_move`] instead.
fn endgame_eval(state: &State, our_id: usize, scratch: &mut Scratch) -> i32 {
    let fill = approx_fill(state, our_id, scratch);
    let flood = flood_count(state, our_id, scratch);
    let hug = wall_neighbor_count(
        state,
        state.head_x[our_id] as i32,
        state.head_y[our_id] as i32,
    );
    fill * 80 + flood * 20 + hug * 6 + mobility(state, our_id)
}

/// Unused solo survival search: maximise [`endgame_eval`] at a given depth.
///
/// **Where:** nowhere. `#![allow(dead_code)]` keeps it for experiments.
/// **Why it exists:** a true longest-path search in a chamber. Production
/// [`choose_move`] uses an 80-step greedy [`fill_direction`] rollout instead,
/// which is cheaper and scored about as well.
fn search_endgame(
    state: &mut State,
    our_id: usize,
    depth: i32,
    ply: i32,
    last_dir: u8,
    search: &mut Search,
    scratch: &mut Scratch,
) -> i32 {
    search.check_time();
    if search.timed_out {
        return TIMEOUT_SCORE;
    }
    let (mut moves, move_count) = state.legal_moves(our_id);
    if move_count == 0 {
        return ply;
    }
    if depth <= 0 {
        return endgame_eval(state, our_id, scratch) + ply;
    }
    order_moves(
        state,
        our_id,
        &mut moves,
        move_count,
        NO_MOVE,
        search.killers[ply as usize % 64][0],
        last_dir,
        state.head_x[our_id] as i32,
        state.head_y[our_id] as i32,
    );
    let mut best_score = i32::MIN / 2;
    for move_i in 0..move_count {
        let old_col = state.head_x[our_id];
        let old_row = state.head_y[our_id];
        state.apply(our_id, moves[move_i] as usize);
        let score = search_endgame(
            state,
            our_id,
            depth - 1,
            ply + 1,
            moves[move_i],
            search,
            scratch,
        );
        state.undo_step(our_id, old_col, old_row);
        if search.timed_out {
            return TIMEOUT_SCORE;
        }
        if score > best_score {
            best_score = score;
        }
    }
    best_score
}

/// Unused FFA search: we branch on our moves; every other player replies with
/// [`greedy_direction`] (or dies if they have no move).
///
/// **Where:** nowhere today. [`choose_move`] sets FFA `max_depth = 0` and uses
/// a 2-ply of [`oneply_direction`] + [`eval_ffa`] instead.
/// **Why it exists:** a deeper FFA tree to try later. Deep 1v1 minimax in FFA
/// suicides because it assumes others play “our” duel; this models them as
/// greedy fillers. Wiring it in is a listed next experiment.
fn search_ffa(
    state: &State,
    our_id: usize,
    depth: i32,
    ply: i32,
    search: &mut Search,
    scratch: &mut Scratch,
) -> i32 {
    search.check_time();
    if search.timed_out {
        return TIMEOUT_SCORE;
    }
    if !state.is_alive(our_id) {
        return -MATE_SCORE + ply;
    }
    if state.alive_mask.count_ones() == 1 {
        return MATE_SCORE - ply;
    }
    if depth <= 0 {
        return eval_ffa(state, our_id, ply, scratch);
    }
    let (moves, move_count) = state.legal_moves(our_id);
    if move_count == 0 {
        return -MATE_SCORE + ply;
    }
    let mut best_score = -MATE_SCORE * 2;
    for move_i in 0..move_count {
        let mut after = *state;
        after.apply(our_id, moves[move_i] as usize);
        // Copy the state so opponent deaths do not need undo; we just drop `after`.
        let player_count = after.player_count as usize;
        // Walk the remaining seats in seating order until we get back to us.
        let mut player = (our_id + 1) % player_count;
        while player != our_id {
            if after.is_alive(player) {
                if let Some(dir) = greedy_direction(&after, player, scratch) {
                    after.apply(player, dir as usize);
                } else {
                    after.kill(player);
                }
            }
            player = (player + 1) % player_count;
        }
        let score = search_ffa(&after, our_id, depth - 1, ply + 1, search, scratch);
        if search.timed_out {
            return TIMEOUT_SCORE;
        }
        if score > best_score {
            best_score = score;
        }
    }
    best_score
}

/// One-ply lookahead: try each of `player`’s legal moves, score the resulting
/// position, and return the best direction.
///
/// With one living opponent this uses [`eval_1v1`] (and an instant mate if they
/// have no reply). With several it uses [`eval_ffa`]. Returns `None` if dead.
///
/// **Where:** FFA [`choose_move`] — after we try a root move, every other
/// living player replies with this; unused [`rollout_score`].
/// **Why:** we need a “what would they do this turn?” model that is stronger
/// than raw flood-greedy but cheap enough to run N−1 times per root move.
fn oneply_direction(state: &mut State, player: usize, scratch: &mut Scratch) -> Option<u8> {
    let (legal, move_count) = state.legal_moves(player);
    if move_count == 0 {
        return None;
    }
    if move_count == 1 {
        return Some(legal[0]);
    }
    let player_count = state.player_count as usize;
    let living_others = (0..player_count)
        .filter(|&rival| rival != player && state.is_alive(rival))
        .count();
    let opponent = (0..player_count)
        .find(|&rival| rival != player && state.is_alive(rival))
        .unwrap_or(player);
    let mut best_dir = legal[0];
    let mut best_score = i32::MIN;
    for move_i in 0..move_count {
        let old_col = state.head_x[player];
        let old_row = state.head_y[player];
        state.apply(player, legal[move_i] as usize);
        let score = if living_others <= 1 {
            if state.is_alive(opponent) && state.legal_moves(opponent).1 == 0 {
                MATE_SCORE
            } else {
                eval_1v1(state, player, opponent, 1, scratch)
            }
        } else {
            eval_ffa(state, player, 1, scratch)
        };
        state.undo_step(player, old_col, old_row);
        if score > best_score {
            best_score = score;
            best_dir = legal[move_i];
        }
    }
    Some(best_dir)
}

/// Isolated-chamber policy: pick the step that keeps the most fillable space.
///
/// Score = approx-fill + remaining flood + wall-hug, with a small bonus for
/// continuing in `last_dir`. If a step shrinks the reachable set (we walked
/// into a pocket and walled ourselves off from the rest), apply a heavy
/// penalty. Returns `None` if there is no legal move.
///
/// **Where:** [`choose_move`] when nobody else is alive, and inside the 80-step
/// primed-greedy rollout when a 1v1 is cut off.
/// **Why:** after a cut, minimax + Voronoi is the wrong game. This is a1k0n’s
/// “take side pockets before the corridor” filler.
fn fill_direction(
    state: &mut State,
    player: usize,
    last_dir: u8,
    scratch: &mut Scratch,
) -> Option<u8> {
    let (legal, move_count) = state.legal_moves(player);
    if move_count == 0 {
        return None;
    }
    let flood_before = flood_count(state, player, scratch);
    let mut best_dir = legal[0];
    let mut best_score = i32::MIN;
    for move_i in 0..move_count {
        let old_col = state.head_x[player];
        let old_row = state.head_y[player];
        state.apply(player, legal[move_i] as usize);
        let flood_after = flood_count(state, player, scratch);
        let fill = approx_fill(state, player, scratch);
        let col = state.head_x[player] as i32;
        let row = state.head_y[player] as i32;
        let mut score = fill * 80 + flood_after * 30 + wall_neighbor_count(state, col, row) * 12;
        if legal[move_i] == last_dir {
            score += 8;
        }
        // flood_after should be about flood_before - 1 (we occupied one cell).
        // A bigger drop means we sealed off a region we can no longer reach.
        if flood_after + 1 < flood_before {
            score -= (flood_before - flood_after) * 400;
        }
        state.undo_step(player, old_col, old_row);
        if score > best_score {
            best_score = score;
            best_dir = legal[move_i];
        }
    }
    Some(best_dir)
}

/// Unused: play `first_dir` for us, then `steps` of [`oneply_direction`] for
/// every living player in seating order, then return a static eval.
///
/// **Where:** nowhere. Production [`choose_move`] does not call this.
/// **Why it exists:** an old “roll the position forward then eval” experiment.
fn rollout_score(
    state: &State,
    our_id: usize,
    first_dir: u8,
    steps: i32,
    scratch: &mut Scratch,
) -> i32 {
    let mut after = *state;
    if !after.apply(our_id, first_dir as usize) {
        return -MATE_SCORE;
    }
    let player_count = after.player_count as usize;
    let mut player = (our_id + 1) % player_count;
    let mut made = 0i32;
    while made < steps && after.alive_mask.count_ones() > 1 && after.is_alive(our_id) {
        if after.is_alive(player) {
            match oneply_direction(&mut after, player, scratch) {
                Some(dir) if after.is_legal_dir(player, dir as usize) => {
                    after.apply(player, dir as usize);
                }
                _ => after.kill(player),
            }
            made += 1;
        }
        player = (player + 1) % player_count;
    }
    if !after.is_alive(our_id) {
        return -MATE_SCORE + made;
    }
    if after.alive_mask.count_ones() == 1 {
        return MATE_SCORE - made;
    }
    let others: Vec<usize> = (0..player_count)
        .filter(|&rival| rival != our_id && after.is_alive(rival))
        .collect();
    if others.len() == 1 {
        eval_1v1(&after, our_id, others[0], made, scratch)
    } else {
        eval_ffa(&after, our_id, made, scratch)
    }
}

/// Choose a direction (0=UP .. 3=RIGHT) for `our_id` within `budget_ms`.
///
/// Policy:
/// - No legal moves → dummy `0` (we crash next turn anyway).
/// - One legal move → play it immediately.
/// - Nobody else alive → isolated [`fill_direction`].
/// - 1v1 and the two chambers are cut off → 80-step greedy fill rollout.
/// - 3+ living players → 2-ply: we move, others reply with [`oneply_direction`],
///   then [`eval_ffa`]. Iterative deepening is skipped (`max_depth = 0`).
/// - 1v1 still connected → iterative-deepening [`negamax_1v1`] up to depth 16
///   or the time budget. There is no 2-ply Voronoi warmup; ID starts from the
///   first ordered move so search gets the full remaining budget.
///
/// `last_dir` is the direction we played last turn (`NO_MOVE` on turn 1); it
/// is a small move-ordering / fill-continuation hint, not a hard constraint.
///
/// **Where:** [`codingame`] every turn; `--bench` `Bot::Agent`; `--profile`.
/// **Why:** this is the whole policy. Everything else is a helper it calls.
fn choose_move(
    state: &mut State,
    our_id: usize,
    last_dir: u8,
    budget_ms: u64,
    scratch: &mut Scratch,
) -> u8 {
    let (mut moves, move_count) = state.legal_moves(our_id);
    if move_count == 0 {
        return 0;
    }
    if move_count == 1 {
        return moves[0];
    }
    let start = Instant::now();

    let alive_others: Vec<usize> = (0..state.player_count as usize)
        .filter(|&player| player != our_id && state.is_alive(player))
        .collect();
    if alive_others.is_empty() {
        return fill_direction(state, our_id, last_dir, scratch).unwrap_or(moves[0]);
    }
    let is_duel = alive_others.len() == 1;
    let opponent = if is_duel {
        alive_others[0]
    } else {
        main_opponent(state, our_id, scratch).unwrap_or(alive_others[0])
    };

    let occupied_count: u32 = (0..20)
        .map(|row| state.occupied.bits[row].count_ones())
        .sum();
    // Declare “cut off” only if both the BFS test and Voronoi agree we cannot
    // meet. FFA is never treated as separated (we still need to fight).
    let still_connected = !is_duel
        || shares_space(state, our_id, opponent, scratch)
        || compute_voronoi(state, scratch).still_connected;
    let separated = is_duel && !still_connected;
    CHOOSE_MOVE_COUNT.fetch_add(1, Ordering::Relaxed);
    if separated {
        SEPARATED_MOVE_COUNT.fetch_add(1, Ordering::Relaxed);
        if occupied_count < 24 {
            EARLY_SEPARATION_COUNT.fetch_add(1, Ordering::Relaxed);
        }
    }

    let (target_col, target_row) = (state.head_x[opponent] as i32, state.head_y[opponent] as i32);
    order_moves(
        state, our_id, &mut moves, move_count, NO_MOVE, NO_MOVE, last_dir, target_col, target_row,
    );

    // Isolated chamber: try each first step, then greedy-fill up to 80 cells.
    if separated {
        let mut best_dir = moves[0];
        let mut best_score = i32::MIN;
        for move_i in 0..move_count {
            let old_col = state.head_x[our_id];
            let old_row = state.head_y[our_id];
            state.apply(our_id, moves[move_i] as usize);
            let mut simulated = *state;
            let mut extra = 0i32;
            let mut prev_dir = moves[move_i];
            while extra < 80 {
                match fill_direction(&mut simulated, our_id, prev_dir, scratch) {
                    Some(dir) if simulated.is_legal_dir(our_id, dir as usize) => {
                        simulated.apply(our_id, dir as usize);
                        prev_dir = dir;
                        extra += 1;
                    }
                    _ => break,
                }
            }
            let hug = wall_neighbor_count(
                state,
                state.head_x[our_id] as i32,
                state.head_y[our_id] as i32,
            );
            let score = extra * 50 + hug;
            state.undo_step(our_id, old_col, old_row);
            if score > best_score {
                best_score = score;
                best_dir = moves[move_i];
            }
        }
        return best_dir;
    }

    // FFA has no deep search (`max_depth = 0` below), so it still uses 2-ply
    // greedy. 1v1 skips that warmup so iterative deepening gets the full budget.
    let mut best_dir = moves[0];
    let mut best_score = 0i32;
    if !is_duel {
        best_score = i32::MIN;
        for move_i in 0..move_count {
            let old_col = state.head_x[our_id];
            let old_row = state.head_y[our_id];
            state.apply(our_id, moves[move_i] as usize);
            let mut after = *state;
            let player_count = after.player_count as usize;
            let mut player = (our_id + 1) % player_count;
            while player != our_id {
                if after.is_alive(player) {
                    if let Some(dir) = oneply_direction(&mut after, player, scratch) {
                        after.apply(player, dir as usize);
                    } else {
                        after.kill(player);
                    }
                }
                player = (player + 1) % player_count;
            }
            let mut score =
                eval_ffa(&after, our_id, 1, scratch) + flood_count(&after, our_id, scratch) * 20;
            state.undo_step(our_id, old_col, old_row);
            if moves[move_i] == last_dir {
                score += 4;
            }
            if score > best_score {
                best_score = score;
                best_dir = moves[move_i];
            }
        }
    }

    let remaining = Duration::from_millis(budget_ms.max(1)).saturating_sub(start.elapsed());
    let mut search = Search::new(remaining);
    let mut principal_dir = best_dir;
    // FFA: skip iterative deepening. 1v1: search as deep as time allows (cap 16).
    let max_depth = if state.alive_mask.count_ones() > 2 {
        0
    } else {
        16
    };
    for depth in 1..=max_depth {
        if Instant::now() >= search.deadline {
            break;
        }
        let mut iter_best_dir = principal_dir;
        let mut iter_best_score = -MATE_SCORE * 2;
        let mut completed_iteration = true;
        // Re-order using last iteration’s PV and root killers.
        order_moves(
            state,
            our_id,
            &mut moves,
            move_count,
            principal_dir,
            search.killers[0][0],
            last_dir,
            target_col,
            target_row,
        );
        for move_i in 0..move_count {
            let old_col = state.head_x[our_id];
            let old_row = state.head_y[our_id];
            state.apply(our_id, moves[move_i] as usize);
            let score = -negamax_1v1(
                state,
                our_id,
                opponent,
                opponent,
                depth - 1,
                1,
                -MATE_SCORE * 2,
                -iter_best_score,
                moves[move_i],
                NO_MOVE,
                &mut search,
                scratch,
            );
            state.undo_step(our_id, old_col, old_row);
            if search.timed_out {
                completed_iteration = false;
                break;
            }
            if score > iter_best_score {
                iter_best_score = score;
                iter_best_dir = moves[move_i];
            }
        }
        if completed_iteration {
            principal_dir = iter_best_dir;
            best_dir = iter_best_dir;
            // Forced mate / loss: no point searching deeper.
            if iter_best_score.abs() >= MATE_SCORE - 200 {
                break;
            }
        } else {
            // Timed out mid-iteration: keep the previous completed PV.
            break;
        }
    }

    if budget_ms >= TURN_BUDGET_MS {
        // CodinGame stderr is visible in the IDE; skip it under SPRT's short budget.
        eprintln!(
            "{} mm {} {}ms",
            DIR_NAME[best_dir as usize],
            best_score,
            start.elapsed().as_millis()
        );
    }
    best_dir
}

/// Reconstructs occupancy from CodinGame’s (start, head) pairs each turn.
///
/// The protocol never sends the full ribbon — only each player’s spawn cell
/// and current head. We occupy every newly reported head so trails stay in
/// sync. Death is four `-1`s; that player’s trail is XOR’d off the board.
///
/// **Where:** [`codingame`] owns one `Tracker` for the whole match.
/// **Why:** without this, search would only see heads, not walls. SPRT’s
/// `--seed-plies` still send every frame so this stays in sync during forced walks.
struct Tracker {
    state: State,
    seen_first_frame: bool,
}

impl Tracker {
    /// Placeholder state until the first CodinGame frame arrives.
    fn new() -> Self {
        Self {
            state: State::new(2),
            seen_first_frame: false,
        }
    }

    /// Apply one input frame.
    ///
    /// `coords[player] = (start_col, start_row, head_col, head_row)`.
    /// On the first frame we occupy both spawn and head (they differ after
    /// turn 1). On later frames we occupy only the new head if it moved.
    /// Called once per stdin turn from [`codingame`].
    fn update(&mut self, player_count: usize, coords: &[(i32, i32, i32, i32)]) {
        if !self.seen_first_frame {
            self.state = State::new(player_count as u8);
            for player in 0..player_count {
                let (start_col, start_row, head_col, head_row) = coords[player];
                if start_col < 0 {
                    continue;
                }
                self.state.occupy(player, start_col, start_row);
                if head_col != start_col || head_row != start_row {
                    self.state.occupy(player, head_col, head_row);
                }
            }
            self.seen_first_frame = true;
            return;
        }
        for player in 0..player_count {
            let (start_col, start_row, head_col, head_row) = coords[player];
            if start_col < 0 {
                self.state.kill(player);
                continue;
            }
            // Should not happen in a correct stream; re-seed if we missed a death.
            if !self.state.is_alive(player) {
                self.state.occupy(player, start_col, start_row);
            }
            let known_col = self.state.head_x[player] as i32;
            let known_row = self.state.head_y[player] as i32;
            if head_col != known_col || head_row != known_row {
                self.state.occupy(player, head_col, head_row);
            }
        }
    }
}

/// Time budget override for the SPRT referee: `--budget-ms N`, `--budget-ms=N`,
/// or env `TRON_BUDGET_MS`. `None` means use the CodinGame defaults
/// ([`FIRST_TURN_BUDGET_MS`] / [`TURN_BUDGET_MS`]).
///
/// **Where:** [`codingame`] at startup. Local `--bench` / `--profile` ignore this
/// and pass a budget into [`choose_move`] themselves.
fn parse_budget_ms() -> Option<u64> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--budget-ms" {
            return args.next().and_then(|value| value.parse().ok());
        }
        if let Some(value) = arg.strip_prefix("--budget-ms=") {
            return value.parse().ok();
        }
    }
    std::env::var("TRON_BUDGET_MS")
        .ok()
        .and_then(|value| value.parse().ok())
}

/// CodinGame stdin/stdout loop.
///
/// Each turn: read `N my_id`, then N lines of `start_x start_y head_x head_y`,
/// update [`Tracker`], print `UP|DOWN|LEFT|RIGHT`. First turn gets 85 ms,
/// later turns 75 ms, unless [`parse_budget_ms`] overrides (SPRT).
///
/// **Where:** [`main`] when there is no `--bench` / `--profile` flag (including
/// the CodinGame judge, which passes no argv).
fn codingame() {
    let budget_override = parse_budget_ms();
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let mut tracker = Tracker::new();
    let mut scratch = Scratch::new();
    let mut last_dir = NO_MOVE;
    let mut is_first_turn = true;
    loop {
        let line = match lines.next() {
            Some(Ok(text)) => text,
            _ => break,
        };
        let mut tokens = line.split_whitespace();
        let player_count: usize = tokens.next().unwrap().parse().unwrap();
        let our_id: usize = tokens.next().unwrap().parse().unwrap();
        let mut coords = [(0i32, 0i32, 0i32, 0i32); MAX_PLAYERS];
        for player in 0..player_count {
            let line = lines.next().unwrap().unwrap();
            let mut tokens = line.split_whitespace();
            // Four ints: spawn cell (constant) and current head.
            coords[player] = (
                tokens.next().unwrap().parse().unwrap(),
                tokens.next().unwrap().parse().unwrap(),
                tokens.next().unwrap().parse().unwrap(),
                tokens.next().unwrap().parse().unwrap(),
            );
        }
        tracker.update(player_count, &coords);
        let budget = budget_override.unwrap_or(if is_first_turn {
            FIRST_TURN_BUDGET_MS
        } else {
            TURN_BUDGET_MS
        });
        let dir = choose_move(&mut tracker.state, our_id, last_dir, budget, &mut scratch);
        last_dir = dir;
        is_first_turn = false;
        println!("{}", DIR_NAME[dir as usize]);
        let _ = io::stdout().flush();
    }
}

// ---------------------------------------------------------------------------
// Local self-play (`--bench`) so we can measure the agent before submitting.
// ---------------------------------------------------------------------------

/// Tiny xorshift64* RNG for local games (no `rand` crate; CodinGame is std-only).
/// Used only by `--bench` / `--profile` spawns and the random dummy opponent.
struct XorShift {
    state: u64,
}

impl XorShift {
    /// Seed must be non-zero; we OR in 1 so `new(0)` still works.
    /// [`bench`] / [`random_start`] / [`play_game`] each construct their own.
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    /// Advance the xorshift64* state and return the new 64-bit value.
    /// Seeds [`random_start`] games and picks a random legal move for `bot_random`.
    fn next_u64(&mut self) -> u64 {
        let mut bits = self.state;
        bits ^= bits << 13;
        bits ^= bits >> 7;
        bits ^= bits << 17;
        self.state = bits;
        bits
    }

    /// Uniform integer in `0..modulus`. Fine for local benches, not crypto.
    /// Spawn cells (`0..WIDTH`, `0..HEIGHT`) and random-bot move indices.
    fn gen_range(&mut self, modulus: u32) -> u32 {
        (self.next_u64() as u32) % modulus
    }
}

/// Place `player_count` bikes on unique random cells (CodinGame spawn rule).
/// `--bench` and `--profile` use this; SPRT uses the JSON opening books instead.
fn random_start(rng: &mut XorShift, player_count: usize) -> State {
    let mut state = State::new(player_count as u8);
    for player in 0..player_count {
        loop {
            let col = rng.gen_range(WIDTH as u32) as i32;
            let row = rng.gen_range(HEIGHT as u32) as i32;
            if !state.occupied.is_set(col, row) {
                state.occupy(player, col, row);
                break;
            }
        }
    }
    state
}

/// Local-bench opponent: uniform random legal direction.
/// Weak baseline in [`bench`]; also fills unused seats in [`play_game`]'s `[Bot; 4]`.
fn bot_random(state: &State, player: usize, rng: &mut XorShift) -> Option<u8> {
    let (legal, move_count) = state.legal_moves(player);
    if move_count == 0 {
        None
    } else {
        Some(legal[rng.gen_range(move_count as u32) as usize])
    }
}

/// Local-bench opponent: [`greedy_direction`] (flood + wall-hug).
/// One of the 1v1 matchups and a seat in the 4p mixed FFA smoke test.
fn bot_greedy(state: &State, player: usize, scratch: &mut Scratch) -> Option<u8> {
    greedy_direction(state, player, scratch)
}

/// Local-bench opponent: step onto the cell with the most occupied/edge neighbours.
/// A wall-hug dummy; if we lose to this, fill / hug terms are broken.
fn bot_wallhug(state: &State, player: usize) -> Option<u8> {
    let (legal, move_count) = state.legal_moves(player);
    if move_count == 0 {
        return None;
    }
    let mut best_dir = legal[0];
    let mut best_walls = -1;
    for move_i in 0..move_count {
        let col = state.head_x[player] as i32 + DIR_X[legal[move_i] as usize];
        let row = state.head_y[player] as i32 + DIR_Y[legal[move_i] as usize];
        let walls = wall_neighbor_count(state, col, row);
        if walls > best_walls {
            best_walls = walls;
            best_dir = legal[move_i];
        }
    }
    Some(best_dir)
}

/// Local-bench opponent: one-ply maximise of [`eval_1v1`] vs the first living rival.
/// The usual “strong greedy” reference; a broken leaf eval historically lost to this.
fn bot_voronoi1(state: &mut State, player: usize, scratch: &mut Scratch) -> Option<u8> {
    let (legal, move_count) = state.legal_moves(player);
    if move_count == 0 {
        return None;
    }
    let opponent = (0..state.player_count as usize)
        .find(|&rival| rival != player && state.is_alive(rival))
        .unwrap_or(player);
    let mut best_dir = legal[0];
    let mut best_score = i32::MIN;
    for move_i in 0..move_count {
        let old_col = state.head_x[player];
        let old_row = state.head_y[player];
        state.apply(player, legal[move_i] as usize);
        let score = eval_1v1(state, player, opponent, 1, scratch);
        state.undo_step(player, old_col, old_row);
        if score > best_score {
            best_score = score;
            best_dir = legal[move_i];
        }
    }
    Some(best_dir)
}

/// Which policy occupies a seat in a local [`play_game`].
/// Not used on CodinGame; `--bench` maps these onto dummy opponents.
#[derive(Clone, Copy)]
enum Bot {
    Agent,
    Greedy,
    Wall,
    Voronoi1,
    Random,
}

/// Play one local game to completion. Returns the winning player index.
///
/// Players move in seating order each “turn” (a full round). Illegal / missing
/// moves kill that player and clear their trail. Cap is 900 rounds so a fill
/// loop cannot run forever. If several players remain, the one with the largest
/// remaining flood wins (CodinGame would have already ended on a crash).
///
/// **Where:** [`bench`] only (1v1 matchups and the 4p smoke test).
/// **Why:** a fast sanity check before SPRT. Agent search here is 25 ms/turn.
fn play_game(mut state: State, bots: [Bot; 4], seed: u64) -> usize {
    let mut scratch = Scratch::new();
    let mut rng = XorShift::new(seed);
    let mut last_dir = [NO_MOVE; MAX_PLAYERS];
    let player_count = state.player_count as usize;
    let mut turn = 0u32;
    while state.alive_mask.count_ones() > 1 && turn < 900 {
        for player in 0..player_count {
            if !state.is_alive(player) {
                continue;
            }
            if state.alive_mask.count_ones() <= 1 {
                break;
            }
            let dir = match bots[player] {
                Bot::Agent => Some(choose_move(
                    &mut state,
                    player,
                    last_dir[player],
                    25,
                    &mut scratch,
                )),
                Bot::Greedy => bot_greedy(&state, player, &mut scratch),
                Bot::Wall => bot_wallhug(&state, player),
                Bot::Voronoi1 => bot_voronoi1(&mut state, player, &mut scratch),
                Bot::Random => bot_random(&state, player, &mut rng),
            };
            match dir {
                Some(chosen) if state.is_legal_dir(player, chosen as usize) => {
                    state.apply(player, chosen as usize);
                    last_dir[player] = chosen;
                }
                _ => state.kill(player),
            }
        }
        turn += 1;
    }
    let living: Vec<usize> = (0..player_count)
        .filter(|&player| state.is_alive(player))
        .collect();
    if living.len() == 1 {
        return living[0];
    }
    if living.is_empty() {
        return 0;
    }
    // Simultaneous last-crash / turn-cap: largest remaining chamber wins.
    let mut best_player = living[0];
    let mut best_flood = -1;
    for &player in &living {
        let flood = flood_count(&state, player, &mut scratch);
        if flood > best_flood {
            best_flood = flood;
            best_player = player;
        }
    }
    best_player
}

/// `cargo run --release -- --bench`: 12 games vs each dummy, plus 8 FFA games.
/// Prints win rates and how often 1v1 search thought the chambers were split.
///
/// **Where:** [`main`] when argv contains `--bench`. Not the SPRT harness
/// (`tools/sprt.sh` plays two compiled binaries instead).
fn bench() {
    let mut rng = XorShift::new(0xC0FFEE);
    let matchups: [(&str, Bot); 4] = [
        ("random", Bot::Random),
        ("wall-hug", Bot::Wall),
        ("greedy-fill", Bot::Greedy),
        ("voronoi-1ply", Bot::Voronoi1),
    ];
    let games = 12;
    eprintln!("Self-play: {games} games per matchup, 2 players, 30x20, 25ms/turn");
    for (name, opponent_bot) in matchups {
        let mut wins_as_first = 0;
        let mut first_games = 0;
        let mut agent_wins = 0;
        for game_i in 0..games {
            let seed = rng.next_u64();
            let start_state = random_start(&mut XorShift::new(seed), 2);
            // Alternate colours so first-mover bias does not dominate the score.
            let (bot0, bot1) = if game_i % 2 == 0 {
                (Bot::Agent, opponent_bot)
            } else {
                (opponent_bot, Bot::Agent)
            };
            let mut bots = [Bot::Random; 4];
            bots[0] = bot0;
            bots[1] = bot1;
            let winner = play_game(start_state, bots, seed ^ 0x9E3779B97F4A7C15);
            let agent_id = if game_i % 2 == 0 { 0 } else { 1 };
            if game_i % 2 == 0 {
                first_games += 1;
                if winner == 0 {
                    wins_as_first += 1;
                }
            }
            if winner == agent_id {
                agent_wins += 1;
            }
        }
        eprintln!(
            "  vs {name:12}  agent {agent_wins:3}/{games}  ({:.0}%)  as-p0 {wins_as_first}/{first_games}",
            100.0 * agent_wins as f64 / games as f64
        );
    }
    let separated = SEPARATED_MOVE_COUNT.load(Ordering::Relaxed);
    let total = CHOOSE_MOVE_COUNT.load(Ordering::Relaxed);
    let early = EARLY_SEPARATION_COUNT.load(Ordering::Relaxed);
    eprintln!(
        "  separated moves {separated}/{total} ({:.0}%) early {early}",
        if total > 0 {
            100.0 * separated as f64 / total as f64
        } else {
            0.0
        }
    );

    let mut ffa_wins = 0;
    let ffa_games = 8;
    // Four-player smoke test: us vs three different dummies.
    for _game_i in 0..ffa_games {
        let seed = rng.next_u64();
        let start_state = random_start(&mut XorShift::new(seed), 4);
        let bots = [Bot::Agent, Bot::Greedy, Bot::Wall, Bot::Voronoi1];
        let winner = play_game(start_state, bots, seed);
        if winner == 0 {
            ffa_wins += 1;
        }
    }
    eprintln!(
        "  FFA vs mixed     agent {ffa_wins:3}/{ffa_games}  ({:.0}%)",
        100.0 * ffa_wins as f64 / ffa_games as f64
    );
}

/// `cargo run --release -- --profile`: eval throughput plus one timed `choose_move`.
/// Walks 30 greedy plies first so we profile a midgame position, not an empty board.
///
/// **Where:** [`main`] when argv contains `--profile`. Use this to see whether
/// an eval change helped nodes/sec (Voronoi is the usual bottleneck).
fn profile() {
    let mut rng = XorShift::new(42);
    let mut state = random_start(&mut rng, 2);
    let mut scratch = Scratch::new();
    // Advance into a typical midgame so eval/search times are realistic.
    for _ in 0..30 {
        for player in 0..2 {
            if let Some(dir) = bot_greedy(&state, player, &mut scratch) {
                state.apply(player, dir as usize);
            }
        }
    }
    let eval_started = Instant::now();
    let eval_count = 2000;
    let mut checksum = 0i32;
    for _ in 0..eval_count {
        checksum ^= eval_1v1(&state, 0, 1, 0, &mut scratch);
    }
    let eval_time = eval_started.elapsed();
    eprintln!(
        "eval: {} in {:?} ({:.1}/ms) checksum {checksum}",
        eval_count,
        eval_time,
        eval_count as f64 / eval_time.as_secs_f64() / 1000.0
    );
    let choose_started = Instant::now();
    let dir = choose_move(&mut state, 0, NO_MOVE, TURN_BUDGET_MS, &mut scratch);
    eprintln!(
        "choose_move {} in {:?}",
        DIR_NAME[dir as usize],
        choose_started.elapsed()
    );
}

/// `--bench` / `--profile` for local work; otherwise the CodinGame I/O loop.
/// The judge (and SPRT) launch the binary with no flags, so [`codingame`] runs.
fn main() {
    if std::env::args().any(|arg| arg == "--bench") {
        bench();
    } else if std::env::args().any(|arg| arg == "--profile") {
        profile();
    } else {
        codingame();
    }
}
