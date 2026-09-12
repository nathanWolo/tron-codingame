//! CodinGame Tron Battle agent.
//!
//! Paste this single file into the CodinGame Rust IDE. Local extras live
//! beside it and are not part of the paste: `src/local.rs` (`--bench` /
//! `--profile`) and `src/voronoi_tests.rs` (`cargo test`). Engine-vs-engine
//! SPRT is `tools/sprt.sh`, not this file.
//!
//! **1v1:** iterative-deepening alpha-beta with a Voronoi territory eval.
//! When the two bikes can no longer reach each other, switch to greedy
//! space-fill (survive as long as possible in our chamber).
//! **FFA (3–4 players):** iterative-deepening paranoid search — we maximize
//! [`eval_ffa`]. The closest rival is a true Min; other bikes play one greedy
//! space-keeping reply. Deep 1v1 minimax (frozen extra bikes) is not used.

use std::io::{self, BufRead, Write};
#[cfg(any(test, feature = "local"))]
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

/// How many choose_move calls saw the two duelists in separate chambers.
#[cfg(any(test, feature = "local"))]
static SEPARATED_MOVE_COUNT: AtomicU32 = AtomicU32::new(0);
/// Total choose_move calls (for the bench separated-rate print).
#[cfg(any(test, feature = "local"))]
static CHOOSE_MOVE_COUNT: AtomicU32 = AtomicU32::new(0);
/// Separated calls that happened with almost-empty boards (suspicious).
#[cfg(any(test, feature = "local"))]
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
/// Extra plies a tactical 1v1 leaf may search (adjacent heads, one escape,
/// or a cut that appears this ply). Caps the horizon so a close fight cannot
/// run away with the 75 ms budget.
const QS_MAX: i32 = 4;
/// Half-window around the previous 1v1 ID score (eval units). ~320 Voronoi
/// cells: open-game depths should rarely fail. Mate and a sudden open→cut
/// still fail and re-search full-window.
const ASPIRATION_DELTA: i32 = 16_000;

/// Bits 0..29 are on the 30-wide board; 30 and 31 stay unused.
const COL_MASK: u32 = (1 << 30) - 1;
/// Even columns (0, 2, …, 28). With an even row this is `(col + row)` even;
/// with an odd row it is `(col + row)` odd. Used by the checkerboard fill bound.
const EVEN_COLS: u32 = 0x1555_5555;

/// Tunable 1v1 eval weights; `TRON_PARAMS` (`name=value,...`) overrides for
/// local tuning runs only.
#[derive(Clone, Copy)]
struct Params {
    terr: i32,
    edges: i32,
    mob: i32,
    front: i32,
    center_div: i32,
    /// Contested (equidistant) cells credited to the side that moves second,
    /// in eval units per cell.
    ties: i32,
    // FFA eval weights.
    f_reach: i32,
    f_terr: i32,
    f_oterr: i32,
    f_oreach: i32,
    f_edges: i32,
    f_mob: i32,
    doom: i32,
    /// 1v1 cut leaf: eval units per fill cell and the win/loss contempt.
    fill: i32,
    fill_sign: i32,
}

const DEFAULT_PARAMS: Params = Params {
    terr: 50,
    edges: 12,
    mob: 6,
    front: 4,
    center_div: 80,
    ties: 0,
    f_reach: 40,
    f_terr: 25,
    f_oterr: 10,
    f_oreach: 4,
    f_edges: 12,
    f_mob: 20,
    doom: 40,
    fill: 60,
    fill_sign: 80,
};

static PARAMS: std::sync::OnceLock<Params> = std::sync::OnceLock::new();

#[inline]
fn params() -> &'static Params {
    PARAMS.get_or_init(|| {
        let mut p = DEFAULT_PARAMS;
        if let Ok(spec) = std::env::var("TRON_PARAMS") {
            for item in spec.split(',') {
                let mut kv = item.split('=');
                let key = kv.next().unwrap_or("").trim();
                let value: i32 = kv.next().and_then(|v| v.trim().parse().ok()).unwrap_or(0);
                match key {
                    "terr" => p.terr = value,
                    "edges" => p.edges = value,
                    "mob" => p.mob = value,
                    "front" => p.front = value,
                    "center_div" => p.center_div = value,
                    "ties" => p.ties = value,
                    "f_reach" => p.f_reach = value,
                    "f_terr" => p.f_terr = value,
                    "f_oterr" => p.f_oterr = value,
                    "f_oreach" => p.f_oreach = value,
                    "f_edges" => p.f_edges = value,
                    "f_mob" => p.f_mob = value,
                    "doom" => p.doom = value,
                    "fill" => p.fill = value,
                    "fill_sign" => p.fill_sign = value,
                    _ => {}
                }
            }
        }
        p
    })
}

const TURN_BUDGET_MS: u64 = 95;
const FIRST_TURN_BUDGET_MS: u64 = 95;

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
/// Indexes `Scratch` arrays (`visited_stamp`, `bfs_queue`) so flood / fill
/// can store one value per cell without a 2D array.
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

/// Zobrist keys: one per cell for occupancy, one per (player, cell) for the
/// head position, and one per side to move. Generated at compile time with
/// splitmix64 so the paste stays a single file with no tables to copy.
const ZOBRIST_CELLS: usize = BOARD_CELLS * (MAX_PLAYERS + 1);
const ZOBRIST: [u64; ZOBRIST_CELLS] = {
    let mut keys = [0u64; ZOBRIST_CELLS];
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut i = 0;
    while i < ZOBRIST_CELLS {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        keys[i] = z ^ (z >> 31);
        i += 1;
    }
    keys
};

#[inline]
fn cell_key(col: i32, row: i32) -> u64 {
    ZOBRIST[cell_index(col, row)]
}

#[inline]
fn head_key(player: usize, col: i32, row: i32) -> u64 {
    ZOBRIST[BOARD_CELLS * (player + 1) + cell_index(col, row)]
}

/// XOR of [`cell_key`] over every set bit in a trail mask (used by kill / restore).
fn trail_hash(mask: &RowBits) -> u64 {
    let mut hash = 0u64;
    for row in 0..20 {
        let mut bits = mask.bits[row];
        while bits != 0 {
            let col = bits.trailing_zeros() as i32;
            bits &= bits - 1;
            hash ^= cell_key(col, row as i32);
        }
    }
    hash
}

/// Full game position: occupancy, per-player trails, heads, who is alive.
///
/// This is the board [`Tracker`] maintains from CodinGame input and that
/// [`choose_move`] / search mutate with apply/undo. `Copy` so fill rollouts
/// and tests can snapshot a position without a heap clone.
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
    /// Zobrist hash of occupancy plus every living head position.
    hash: u64,
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
            hash: 0,
        }
    }

    /// Recompute the Zobrist hash from scratch (tests check the incremental one).
    #[cfg(test)]
    fn full_hash(&self) -> u64 {
        let mut hash = trail_hash(&self.occupied);
        for player in 0..self.player_count as usize {
            if self.is_alive(player) {
                hash ^= head_key(player, self.head_x[player] as i32, self.head_y[player] as i32);
            }
        }
        hash
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
        self.hash ^= cell_key(col, row) ^ head_key(player, col, row);
        if self.head_x[player] >= 0 {
            self.hash ^= head_key(player, self.head_x[player] as i32, self.head_y[player] as i32);
        }
        self.head_x[player] = col as i8;
        self.head_y[player] = row as i8;
        self.alive_mask |= 1 << player;
    }

    /// Mark `(col, row)` as a wall that belongs to nobody (tests / local tools).
    /// Keeps the Zobrist hash in step with `occupied`.
    #[cfg(any(test, feature = "local"))]
    fn add_wall(&mut self, col: i32, row: i32) {
        if !self.occupied.is_set(col, row) {
            self.occupied.set(col, row);
            self.hash ^= cell_key(col, row);
        }
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
        self.hash ^= trail_hash(&self.trail[player])
            ^ head_key(player, self.head_x[player] as i32, self.head_y[player] as i32);
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
    /// Lets 1v1 and FFA search try sibling moves on the same `State` without cloning.
    fn undo_step(&mut self, player: usize, old_col: i8, old_row: i8) {
        let col = self.head_x[player] as i32;
        let row = self.head_y[player] as i32;
        self.occupied.clear(col, row);
        self.trail[player].clear(col, row);
        self.hash ^= cell_key(col, row)
            ^ head_key(player, col, row)
            ^ head_key(player, old_col as i32, old_row as i32);
        self.head_x[player] = old_col;
        self.head_y[player] = old_row;
    }

    /// Inverse of [`State::kill`]: XOR the saved ribbon back onto the board.
    /// FFA search uses this so Min can try sibling moves without cloning `State`.
    fn restore_killed(&mut self, player: usize, trail: RowBits, head_x: i8, head_y: i8) {
        self.trail[player] = trail;
        self.occupied.xor_with(trail);
        self.hash ^= trail_hash(&trail) ^ head_key(player, head_x as i32, head_y as i32);
        self.head_x[player] = head_x;
        self.head_y[player] = head_y;
        self.alive_mask |= 1 << player;
    }
}

/// Scratch space for flood / fill so we do not allocate in search.
///
/// One instance is created in [`codingame`] / `--bench` / `--profile` and
/// reused every turn. Voronoi no longer uses these buffers (row bitboards
/// live on the stack); fill / flood still need visit stamps.
struct Scratch {
    bfs_queue: [u16; BOARD_CELLS],
    /// Generation stamp per cell; compared to `visit_generation` instead of clearing.
    visited_stamp: [u32; BOARD_CELLS],
    visit_generation: u32,
    /// Cut cache: Zobrist key → `our_id`-POV fill score of a 1v1 position in
    /// which the bikes are already separated. Once cut, the two chambers are
    /// independent, so search treats such a node as terminal instead of
    /// expanding a subtree whose every leaf would rerun `approx_fill` twice.
    cut_keys: Vec<u64>,
    cut_scores: Vec<i32>,
}

const CUT_CACHE_BITS: u32 = 15;
const CUT_CACHE_SIZE: usize = 1 << CUT_CACHE_BITS;

impl Scratch {
    /// Allocate one reusable buffer set. Call once per process / game loop
    /// ([`codingame`], [`bench`], [`profile`]).
    fn new() -> Self {
        Self {
            bfs_queue: [0; BOARD_CELLS],
            visited_stamp: [0; BOARD_CELLS],
            visit_generation: 1,
            cut_keys: vec![0; CUT_CACHE_SIZE],
            cut_scores: vec![0; CUT_CACHE_SIZE],
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

/// Empty-cell row masks: bit `col` set iff `(col, row)` is on-board and free.
#[inline(always)]
fn empty_rows(state: &State) -> [u32; 20] {
    let mut empty = [0u32; 20];
    for row in 0..20 {
        empty[row] = !state.occupied.bits[row] & COL_MASK;
    }
    empty
}

/// One orthogonal step from `src` into `empty` cells.
#[inline(always)]
fn expand_mask(src: &[u32; 20], empty: &[u32; 20]) -> [u32; 20] {
    let mut out = [0u32; 20];
    let mut prev = 0u32;
    let mut curr = src[0];
    for row in 0..19 {
        let next = src[row + 1];
        out[row] = ((curr << 1) | (curr >> 1) | prev | next) & empty[row];
        prev = curr;
        curr = next;
    }
    out[19] = ((curr << 1) | (curr >> 1) | prev) & empty[19];
    out
}

/// One orthogonal step, clipped to the 30-wide board (no occupancy mask).
#[inline(always)]
fn expand_mask_board(src: &[u32; 20]) -> [u32; 20] {
    let mut out = [0u32; 20];
    let mut prev = 0u32;
    let mut curr = src[0];
    for row in 0..19 {
        let next = src[row + 1];
        out[row] = ((curr << 1) | (curr >> 1) | prev | next) & COL_MASK;
        prev = curr;
        curr = next;
    }
    out[19] = ((curr << 1) | (curr >> 1) | prev) & COL_MASK;
    out
}

/// Empty 4-neighbours of `player`’s head, as a row mask. The head cell itself
/// is occupied, so Voronoi / flood starts here at distance 1.
#[inline(always)]
fn head_seed(state: &State, player: usize, empty: &[u32; 20]) -> [u32; 20] {
    let mut seed = [0u32; 20];
    if !state.is_alive(player) {
        return seed;
    }
    let head_col = state.head_x[player] as i32;
    let head_row = state.head_y[player] as i32;
    for dir in 0..4 {
        let col = head_col + DIR_X[dir];
        let row = head_row + DIR_Y[dir];
        if in_bounds(col, row) {
            let bit = 1u32 << col;
            if empty[row as usize] & bit != 0 {
                seed[row as usize] |= bit;
            }
        }
    }
    seed
}

/// All empty cells reachable from `seed` (independent of other bikes).
fn flood_mask(seed: &[u32; 20], empty: &[u32; 20]) -> [u32; 20] {
    let mut filled = *seed;
    loop {
        let next = expand_mask(&filled, empty);
        let mut changed = false;
        for row in 0..20 {
            let add = next[row] & !filled[row];
            if add != 0 {
                filled[row] |= add;
                changed = true;
            }
        }
        if !changed {
            return filled;
        }
    }
}

#[inline]
fn mask_popcount(mask: &[u32; 20]) -> i32 {
    mask.iter().map(|row| row.count_ones()).sum::<u32>() as i32
}

/// Longest path through a bipartite set that starts on `start_parity`.
///
/// Same-colour surplus is unfillable: after the minority colour runs out you
/// cannot step onto the leftover majority cells. `start_parity` is 0 for even
/// `(col + row)`, 1 for odd — the colour of the first empty cell of the path.
///
/// **Where:** [`dfs_fill_score`] (per chamber) and [`checkerboard_reach_bound`].
/// **Why:** raw flood / uncut DFS count cells you cannot snake through.
#[inline]
fn checkerboard_path_bound(start_parity: i32, even: i32, odd: i32) -> i32 {
    let (same, other) = if start_parity == 0 {
        (even, odd)
    } else {
        (odd, even)
    };
    same.min(other) * 2 + i32::from(same > other)
}

/// Even / odd `(col + row)` popcounts of a 20-row occupancy mask.
fn mask_parity_counts(mask: &[u32; 20]) -> (i32, i32) {
    let mut even = 0i32;
    let mut odd = 0i32;
    for row in 0..20 {
        let bits = mask[row];
        let even_on_row = if row % 2 == 0 {
            EVEN_COLS
        } else {
            !EVEN_COLS & COL_MASK
        };
        even += (bits & even_on_row).count_ones() as i32;
        odd += (bits & !even_on_row & COL_MASK).count_ones() as i32;
    }
    (even, odd)
}

/// Checkerboard upper bound on filling every empty cell `player` can reach.
///
/// The first step from the occupied head always lands on the opposite colour,
/// so the path starts on `(head_col + head_row + 1) & 1`.
///
/// **Where:** [`fill_direction`] (remaining-space term) and tests.
/// **Why:** a flood count of 9 in a 5–4 colour pocket is not 9 extra lives.
fn checkerboard_reach_bound(state: &State, player: usize) -> i32 {
    if !state.is_alive(player) {
        return 0;
    }
    let empty = empty_rows(state);
    let reach = flood_mask(&head_seed(state, player, &empty), &empty);
    let (even, odd) = mask_parity_counts(&reach);
    let start_parity =
        (state.head_x[player] as i32 + state.head_y[player] as i32 + 1) & 1;
    checkerboard_path_bound(start_parity, even, odd)
}

fn masks_overlap(left: &[u32; 20], right: &[u32; 20]) -> bool {
    for row in 0..20 {
        if left[row] & right[row] != 0 {
            return true;
        }
    }
    false
}

/// Sum of empty 4-neighbours over every cell in `owned` (a1k0n leftover edges).
fn mask_edge_sum(owned: &[u32; 20], empty: &[u32; 20]) -> i32 {
    let mut sum = 0u32;
    for row in 0..20 {
        let owned_row = owned[row];
        if owned_row == 0 {
            continue;
        }
        let empty_row = empty[row];
        sum += (owned_row & (empty_row << 1)).count_ones();
        sum += (owned_row & (empty_row >> 1)).count_ones();
        if row > 0 {
            sum += (owned_row & empty[row - 1]).count_ones();
        }
        if row < 19 {
            sum += (owned_row & empty[row + 1]).count_ones();
        }
    }
    sum as i32
}

/// Count empty cells reachable from `player`, ignoring who else can reach them.
///
/// This is a chamber-size / remaining-space metric, not Voronoi. Uses
/// generation stamps on `scratch` so we never `fill()` the visit array.
///
/// **Where:** [`greedy_direction`], [`fill_direction`], and the `--bench`
/// 900-turn tiebreak.
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
/// **Where:** [`wall_neighbor_count`] (as `4 - this`).
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
/// [`eval_1v1`], the separated greedy rollout in [`choose_move`], and `bot_wallhug`.
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
            let (b_col, b_row) = (
                col + RING[neighbor_slot].0,
                row + RING[neighbor_slot].1,
            );
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

/// Fill estimate plus the even/odd cell counts of the chosen subtree.
/// `est` is already capped by [`checkerboard_path_bound`] for a path that
/// starts by occupying this subtree’s root cell.
struct FillScore {
    est: i32,
    even: i32,
    odd: i32,
}

#[inline]
fn cap_fill(col: i32, row: i32, est: i32, even: i32, odd: i32) -> i32 {
    est.min(checkerboard_path_bound((col + row) & 1, even, odd))
}

/// Estimate how many empty cells a perfect space-fill from `(col, row)` can take.
///
/// Recurses through unvisited empty 4-neighbours. At a local cut we can only
/// commit to one pocket, so we take the max; otherwise we sum (the region is
/// still one connected “snake” we can traverse). Returns at least 1 for the
/// current cell. `generation` is the visit stamp from [`approx_fill`].
/// The estimate is then min’d with the checkerboard bound on the chosen cells:
/// a simply-connected 5–4 colour blob is 8 lives, not 9.
///
/// **Where:** only [`approx_fill`] (once per empty neighbour of the head).
/// **Why:** when 1v1 is cut off, remaining *life* is fillable cells, not raw
/// flood size. Local cuts plus bipartite surplus are a cheap stand-in for a
/// chamber tree.
fn dfs_fill_score(
    state: &State,
    col: i32,
    row: i32,
    scratch: &mut Scratch,
    generation: u32,
) -> FillScore {
    let index = cell_index(col, row);
    scratch.visited_stamp[index] = generation;
    let mut even = i32::from((col + row) & 1 == 0);
    let mut odd = 1 - even;
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
        return FillScore {
            est: 1,
            even,
            odd,
        };
    }
    // A single exit: we must take it, no branching choice.
    if next_count == 1 {
        let child = dfs_fill_score(
            state,
            next_cells[0].0,
            next_cells[0].1,
            scratch,
            generation,
        );
        even += child.even;
        odd += child.odd;
        return FillScore {
            est: cap_fill(col, row, 1 + child.est, even, odd),
            even,
            odd,
        };
    }
    // A cut means walking here walls off some neighbours from each other.
    if is_local_cut(state, col, row) {
        let mut best_est = 0;
        let mut best_even = 0;
        let mut best_odd = 0;
        for pocket_i in 0..next_count {
            let (pocket_col, pocket_row) = next_cells[pocket_i];
            if scratch.visited_stamp[cell_index(pocket_col, pocket_row)] != generation {
                let pocket = dfs_fill_score(state, pocket_col, pocket_row, scratch, generation);
                if pocket.est > best_est {
                    best_est = pocket.est;
                    best_even = pocket.even;
                    best_odd = pocket.odd;
                }
            }
        }
        even += best_even;
        odd += best_odd;
        FillScore {
            est: cap_fill(col, row, 1 + best_est, even, odd),
            even,
            odd,
        }
    } else {
        let mut total = 1;
        for next_i in 0..next_count {
            let (next_col, next_row) = next_cells[next_i];
            if scratch.visited_stamp[cell_index(next_col, next_row)] != generation {
                let child = dfs_fill_score(state, next_col, next_row, scratch, generation);
                even += child.even;
                odd += child.odd;
                total += child.est;
            }
        }
        FillScore {
            est: cap_fill(col, row, total, even, odd),
            even,
            odd,
        }
    }
}

/// How many cells `player` can still claim if they fill their chamber greedily.
///
/// Tries each empty neighbour of the head and returns the best
/// [`dfs_fill_score`] (already checkerboard-capped per chamber).
///
/// **Where:** [`eval_1v1`] when the duelists no longer share space;
/// [`fill_direction`] (endgame policy).
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
            let score = dfs_fill_score(state, col, row, scratch, generation).est;
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
    /// Unique Voronoi cells as row bitboards; [`battlefront`] uses two of these.
    owned: [[u32; 20]; MAX_PLAYERS],
}

/// Multi-source Voronoi from every living head, as row-bitboard distance waves.
///
/// Each wave is one orthogonal step. A cell reached by exactly one bike this
/// wave is that bike’s territory; a cell reached by two or more is a tie and
/// stays unowned. Reachability is a separate independent flood (opponent
/// “owned” empty cells are still walkable). Same results as per-player BFS,
/// but one pass over 20 `u32`s per wave instead of N × 600-cell maps.
///
/// **Where:** [`eval_1v1`] and [`eval_ffa`] every leaf; [`choose_move`] as a
/// second opinion before declaring 1v1 chambers separated.
/// **Why:** in the open game, “land I uniquely reach first” is the standard
/// Tron heuristic (a1k0n / GAI Challenge). Eval is dominated by this call.
fn compute_voronoi(state: &State) -> Voronoi {
    compute_voronoi_ex(state, true)
}

fn compute_voronoi_ex(state: &State, with_edges: bool) -> Voronoi {
    let empty = empty_rows(state);
    let player_count = state.player_count as usize;

    let mut seeds = [[0u32; 20]; MAX_PLAYERS];
    let mut reachable = [0i32; MAX_PLAYERS];
    let mut still_connected = false;
    let mut reached_any = [0u32; 20];
    for player in 0..player_count {
        if !state.is_alive(player) {
            continue;
        }
        seeds[player] = head_seed(state, player, &empty);
        let flood = flood_mask(&seeds[player], &empty);
        reachable[player] = mask_popcount(&flood);
        if !still_connected && masks_overlap(&flood, &reached_any) {
            still_connected = true;
        }
        for row in 0..20 {
            reached_any[row] |= flood[row];
        }
    }

    let mut owned = [[0u32; 20]; MAX_PLAYERS];
    let mut frontier = seeds;
    let mut claimed = [0u32; 20];
    loop {
        let mut union = [0u32; 20];
        let mut multi = [0u32; 20];
        let mut any = false;
        for player in 0..player_count {
            if !state.is_alive(player) {
                continue;
            }
            for row in 0..20 {
                let cells = frontier[player][row] & !claimed[row];
                frontier[player][row] = cells;
                if cells != 0 {
                    any = true;
                }
                multi[row] |= cells & union[row];
                union[row] |= cells;
            }
        }
        if !any {
            break;
        }
        for player in 0..player_count {
            if !state.is_alive(player) {
                continue;
            }
            for row in 0..20 {
                owned[player][row] |= frontier[player][row] & !multi[row];
            }
        }
        for row in 0..20 {
            claimed[row] |= union[row];
        }
        let mut next = [[0u32; 20]; MAX_PLAYERS];
        for player in 0..player_count {
            if !state.is_alive(player) {
                continue;
            }
            next[player] = expand_mask(&frontier[player], &empty);
        }
        frontier = next;
    }

    let mut territory = [0i32; MAX_PLAYERS];
    let mut edge_sum = [0i32; MAX_PLAYERS];
    for player in 0..player_count {
        if !state.is_alive(player) {
            continue;
        }
        territory[player] = mask_popcount(&owned[player]);
        if with_edges {
            edge_sum[player] = mask_edge_sum(&owned[player], &empty);
        }
    }
    Voronoi {
        territory,
        edge_sum,
        reachable,
        still_connected,
        owned,
    }
}

/// FFA leaf Voronoi: the same claim loop as [`compute_voronoi_ex`], but only
/// one independent flood — our component. A rival whose empty head-neighbours
/// all lie inside it shares exactly that component, so their `reachable` is
/// our count; a rival touching a cell outside it is flooded separately. Same
/// `territory` / `edge_sum` / `reachable` / `owned` as the general version;
/// `still_connected` is not computed (always false).
///
/// **Where:** [`eval_ffa`]. **Why:** four independent floods were most of the
/// FFA leaf cost, and in a shared component they all count the same cells.
fn compute_voronoi_ffa(state: &State, our_id: usize) -> Voronoi {
    compute_voronoi_ffa_ex(state, our_id, 0)
}

/// [`compute_voronoi_ffa`] with `doomed` seats removed and their ribbons
/// counted as empty (see `Params::doom`: rivals with that many reachable cells
/// or fewer, sealed off from us, are treated as already gone — CodinGame erases
/// a dead bike’s whole ribbon, so their trail is space the survivors inherit).
fn compute_voronoi_ffa_ex(state: &State, our_id: usize, doomed: u8) -> Voronoi {
    let mut empty = empty_rows(state);
    let player_count = state.player_count as usize;
    let alive = state.alive_mask & !doomed;
    for player in 0..player_count {
        if doomed & (1 << player) != 0 {
            for row in 0..20 {
                empty[row] |= state.trail[player].bits[row];
            }
        }
    }
    let is_alive = |player: usize| alive & (1 << player) != 0;

    let mut seeds = [[0u32; 20]; MAX_PLAYERS];
    let mut reachable = [0i32; MAX_PLAYERS];
    for player in 0..player_count {
        if is_alive(player) {
            seeds[player] = head_seed(state, player, &empty);
        }
    }
    // Flood one component from our first empty neighbour. If our other
    // neighbours are not all inside it we touch several components; fall back
    // to the general version (rare).
    let mut first = [0u32; 20];
    for row in 0..20 {
        let bits = seeds[our_id][row];
        if bits != 0 {
            first[row] = bits & bits.wrapping_neg();
            break;
        }
    }
    let mut component = flood_mask(&first, &empty);
    let mut multi_component = false;
    for row in 0..20 {
        if seeds[our_id][row] & !component[row] != 0 {
            multi_component = true;
        }
    }
    if multi_component {
        // Our head touches several pockets: fall back to one flood per bike.
        component = flood_mask(&seeds[our_id], &empty);
    }
    let our_reach = mask_popcount(&component);
    reachable[our_id] = our_reach;
    for player in 0..player_count {
        if player == our_id || !is_alive(player) {
            continue;
        }
        let mut inside = false;
        let mut outside = false;
        for row in 0..20 {
            let bits = seeds[player][row];
            inside |= bits & component[row] != 0;
            outside |= bits & !component[row] != 0;
        }
        reachable[player] = if inside && !outside && !multi_component {
            our_reach
        } else if !inside && !outside {
            0
        } else {
            mask_popcount(&flood_mask(&seeds[player], &empty))
        };
    }

    let mut owned = [[0u32; 20]; MAX_PLAYERS];
    let mut frontier = seeds;
    let mut claimed = [0u32; 20];
    loop {
        let mut union = [0u32; 20];
        let mut multi = [0u32; 20];
        let mut any = false;
        for player in 0..player_count {
            if !is_alive(player) {
                continue;
            }
            for row in 0..20 {
                let cells = frontier[player][row] & !claimed[row];
                frontier[player][row] = cells;
                if cells != 0 {
                    any = true;
                }
                multi[row] |= cells & union[row];
                union[row] |= cells;
            }
        }
        if !any {
            break;
        }
        for player in 0..player_count {
            if !is_alive(player) {
                continue;
            }
            for row in 0..20 {
                owned[player][row] |= frontier[player][row] & !multi[row];
            }
        }
        for row in 0..20 {
            claimed[row] |= union[row];
        }
        let mut next = [[0u32; 20]; MAX_PLAYERS];
        for player in 0..player_count {
            if !is_alive(player) {
                continue;
            }
            next[player] = expand_mask(&frontier[player], &empty);
        }
        frontier = next;
    }

    let mut territory = [0i32; MAX_PLAYERS];
    let mut edge_sum = [0i32; MAX_PLAYERS];
    for player in 0..player_count {
        if !is_alive(player) {
            continue;
        }
        territory[player] = mask_popcount(&owned[player]);
        edge_sum[player] = mask_edge_sum(&owned[player], &empty);
    }
    Voronoi {
        territory,
        edge_sum,
        reachable,
        still_connected: false,
        owned,
    }
}

/// How many of the four directions are legal for `player` (0–4).
/// A mobility of 1 is a forced move; 0 is death next turn.
///
/// **Where:** [`eval_1v1`] and [`eval_ffa`].
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
fn main_opponent(state: &State, our_id: usize) -> Option<usize> {
    let player_count = state.player_count as usize;
    let our_col = state.head_x[our_id] as i32;
    let our_row = state.head_y[our_id] as i32;
    for player in 0..player_count {
        if player == our_id || !state.is_alive(player) {
            continue;
        }
        let head_col = state.head_x[player] as i32;
        let head_row = state.head_y[player] as i32;
        if (our_col - head_col).abs() + (our_row - head_row).abs() == 1 {
            return Some(player);
        }
    }

    let empty = empty_rows(state);
    let mut opp_seed = [[0u32; 20]; MAX_PLAYERS];
    for player in 0..player_count {
        if player != our_id && state.is_alive(player) {
            opp_seed[player] = head_seed(state, player, &empty);
        }
    }
    let mut frontier = head_seed(state, our_id, &empty);
    let mut seen = frontier;
    let mut dist = 1u16;
    let mut best_player = None;
    let mut best_dist = UNREACHABLE;
    loop {
        let mut any = false;
        for row in 0..20 {
            if frontier[row] != 0 {
                any = true;
                break;
            }
        }
        if !any {
            break;
        }
        for player in 0..player_count {
            if player == our_id || !state.is_alive(player) {
                continue;
            }
            if dist < best_dist && masks_overlap(&frontier, &opp_seed[player]) {
                best_dist = dist;
                best_player = Some(player);
            }
        }
        if best_dist < UNREACHABLE {
            return best_player;
        }
        let expanded = expand_mask(&frontier, &empty);
        let mut next = [0u32; 20];
        let mut grew = false;
        for row in 0..20 {
            next[row] = expanded[row] & !seen[row];
            if next[row] != 0 {
                grew = true;
            }
            seen[row] |= next[row];
        }
        if !grew {
            break;
        }
        frontier = next;
        dist += 1;
    }

    for player in 0..player_count {
        if player == our_id || !state.is_alive(player) {
            continue;
        }
        let head_col = state.head_x[player] as i32;
        let head_row = state.head_y[player] as i32;
        let manhattan = (our_col - head_col).abs() + (our_row - head_row).abs();
        let previous = best_player.map(|previous_player| {
            (our_col - state.head_x[previous_player] as i32).abs()
                + (our_row - state.head_y[previous_player] as i32).abs()
        });
        if best_player.is_none() || manhattan < previous.unwrap_or(i32::MAX) {
            best_player = Some(player);
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
fn shares_space(state: &State, our_id: usize, opponent: usize) -> bool {
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
    let empty = empty_rows(state);
    let our_flood = flood_mask(&head_seed(state, our_id, &empty), &empty);
    masks_overlap(&our_flood, &head_seed(state, opponent, &empty))
}

/// Count our Voronoi cells that share an edge with an opponent Voronoi cell.
///
/// A larger front means more contact / contest. Uses the `owned` bitboards
/// from [`compute_voronoi`].
///
/// **Where:** only the open-game branch of [`eval_1v1`].
/// **Why:** without it, search is happy to wander into dead space while the
/// opponent takes the middle. A small bonus keeps us on the battle line.
fn battlefront(our_owned: &[u32; 20], opp_owned: &[u32; 20]) -> i32 {
    let expanded = expand_mask_board(opp_owned);
    let mut front = 0u32;
    for row in 0..20 {
        front += (our_owned[row] & expanded[row]).count_ones();
    }
    front as i32
}

/// Manhattan distance between two living heads.
#[inline]
fn heads_manhattan(state: &State, a: usize, b: usize) -> i32 {
    (state.head_x[a] as i32 - state.head_x[b] as i32).abs()
        + (state.head_y[a] as i32 - state.head_y[b] as i32).abs()
}

/// True if `to_move` has a legal step that splits the two bikes.
///
/// **Where:** [`wants_quiescence`] when heads are close (`manh <= 3`) and
/// someone is already down to two escapes. Not used on quiet opening leaves.
fn one_ply_creates_cut(
    state: &mut State,
    our_id: usize,
    opponent: usize,
    to_move: usize,
) -> bool {
    if !shares_space(state, our_id, opponent) {
        return false;
    }
    let (moves, move_count) = state.legal_moves(to_move);
    for i in 0..move_count {
        let old_col = state.head_x[to_move];
        let old_row = state.head_y[to_move];
        state.apply(to_move, moves[i] as usize);
        let cut = !shares_space(state, our_id, opponent);
        state.undo_step(to_move, old_col, old_row);
        if cut {
            return true;
        }
    }
    false
}

/// Horizon extension: heads are adjacent / almost adjacent, a bike has one
/// escape, or a cut appears in one ply.
fn wants_quiescence(
    state: &mut State,
    our_id: usize,
    opponent: usize,
    qs_left: i32,
    to_move: usize,
    to_move_count: usize,
) -> bool {
    if qs_left <= 0 || !state.is_alive(our_id) || !state.is_alive(opponent) {
        return false;
    }
    let manh = heads_manhattan(state, our_id, opponent);
    if manh <= 2 {
        return true;
    }
    let our_m = if to_move == our_id {
        to_move_count as i32
    } else {
        mobility(state, our_id)
    };
    let opp_m = if to_move == opponent {
        to_move_count as i32
    } else {
        mobility(state, opponent)
    };
    if our_m <= 1 || opp_m <= 1 {
        return true;
    }
    // One-ply cut: skip while both still have room (opening leaves are manh 3
    // on an empty board; a full BFS there is wasted). Once someone is down to
    // two escapes, a corridor pinch this ply is worth checking.
    (our_m <= 2 || opp_m <= 2)
        && manh <= 3
        && one_ply_creates_cut(state, our_id, opponent, to_move)
}

/// Two independent BFS floods plus Voronoi ownership in one wave loop.
///
/// Both frontiers expand through every empty cell (so `reach_*` is the raw
/// flood, same as [`flood_mask`]). A cell first touched this wave by exactly
/// one bike — and never before by the other — is owned; a simultaneous first
/// touch is a tie. Ownership matches [`compute_voronoi`]: if the other bike
/// already reached a cell in an earlier wave they are strictly closer.
/// `connected` is true if any empty cell is reachable by both.
///
/// **Where:** [`eval_1v1`]. One loop replaces two floods plus a claim loop.
struct DuelVoronoi {
    owned_a: [u32; 20],
    owned_b: [u32; 20],
    connected: bool,
    /// Cells both bikes reach in the same wave (contested).
    ties: i32,
}

fn duel_voronoi(empty: &[u32; 20], seed_a: [u32; 20], seed_b: [u32; 20]) -> DuelVoronoi {
    // Claim-based waves: a frontier never re-enters a cell either bike has
    // already claimed, so it dies out at the territory border. Two bikes are
    // connected iff one wave ever touches a cell the other has claimed (or
    // both touch a cell in the same wave).
    let mut front_a = seed_a;
    let mut front_b = seed_b;
    let mut owned_a = [0u32; 20];
    let mut owned_b = [0u32; 20];
    let mut claimed_a = seed_a;
    let mut claimed_b = seed_b;
    let mut touch = 0u32;
    for row in 0..20 {
        let tie = seed_a[row] & seed_b[row];
        touch |= tie;
        owned_a[row] = seed_a[row] & !tie;
        owned_b[row] = seed_b[row] & !tie;
    }
    loop {
        let step_a = expand_mask(&front_a, empty);
        let step_b = expand_mask(&front_b, empty);
        let mut any = 0u32;
        for row in 0..20 {
            let hit_a = step_a[row] & !claimed_a[row];
            let hit_b = step_b[row] & !claimed_b[row];
            touch |= (hit_a & claimed_b[row]) | (hit_b & claimed_a[row]);
            let new_a = hit_a & !claimed_b[row];
            let new_b = hit_b & !claimed_a[row];
            let tie = new_a & new_b;
            owned_a[row] |= new_a & !tie;
            owned_b[row] |= new_b & !tie;
            claimed_a[row] |= new_a;
            claimed_b[row] |= new_b;
            front_a[row] = new_a;
            front_b[row] = new_b;
            any |= new_a | new_b;
        }
        if any == 0 {
            break;
        }
    }
    let mut ties = 0i32;
    for row in 0..20 {
        ties += (claimed_a[row] & claimed_b[row]).count_ones() as i32;
    }
    DuelVoronoi {
        owned_a,
        owned_b,
        connected: touch != 0,
        ties,
    }
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
/// Open-game path: flood us first (same test as [`shares_space`]), then flood
/// them and paint. That skips the extra independent flood [`compute_voronoi`]
/// would redo for us.
///
/// **Where:** leaves of [`negamax_1v1`]; `--bench` `bot_voronoi1`.
/// **Why:** this is *the* 1v1 heuristic. Search only looks a few plies; the
/// leaf has to encode “who owns the remaining empty board.” An early version
/// added a huge separated-bonus here and minimax hallucinated fake cuts —
/// do not reintroduce that. Switching to fill inside the tree is also a
/// scale change; keep it only for positions that are already cut, not as a
/// bonus for “looking cut.”
fn eval_1v1(
    state: &State,
    our_id: usize,
    opponent: usize,
    to_move: usize,
    ply: i32,
    scratch: &mut Scratch,
) -> i32 {
    if !state.is_alive(our_id) {
        return -MATE_SCORE + ply;
    }
    if !state.is_alive(opponent) {
        return MATE_SCORE - ply;
    }
    let our_mobility = mobility(state, our_id);
    let opp_mobility = mobility(state, opponent);
    let our_col = state.head_x[our_id] as i32;
    let our_row = state.head_y[our_id] as i32;
    let opp_col = state.head_x[opponent] as i32;
    let opp_row = state.head_y[opponent] as i32;
    let adjacent = (our_col - opp_col).abs() + (our_row - opp_row).abs() == 1;
    let empty = empty_rows(state);
    let our_seed = head_seed(state, our_id, &empty);
    let opp_seed = head_seed(state, opponent, &empty);
    // A third living bike still owns Voronoi cells; only the true duel can
    // paint just this pair. Search 1v1 is always two alive.
    let duel_only = state.alive_mask.count_ones() == 2;
    let (connected, duel) = if duel_only {
        let duel = duel_voronoi(&empty, our_seed, opp_seed);
        (duel.connected, Some(duel))
    } else {
        let our_flood = flood_mask(&our_seed, &empty);
        (masks_overlap(&our_flood, &opp_seed), None)
    };
    if !adjacent && !connected {
        let our_fill = approx_fill(state, our_id, scratch);
        let opp_fill = approx_fill(state, opponent, scratch);
        let fill_diff = our_fill - opp_fill;
        let hug = wall_neighbor_count(state, our_col, our_row)
            - wall_neighbor_count(state, opp_col, opp_row);
        let p = params();
        let score = fill_diff.signum() * p.fill_sign
            + fill_diff * p.fill
            + hug * 5
            + (our_mobility - opp_mobility);
        // Remember this cut position so a later, deeper iteration can stop
        // here instead of expanding two independent chambers.
        let slot = (state.hash >> (64 - CUT_CACHE_BITS)) as usize;
        scratch.cut_keys[slot] = state.hash;
        scratch.cut_scores[slot] = score;
        return score;
    }
    let ties = duel.as_ref().map_or(0, |d| d.ties);
    let voronoi = if let Some(duel) = duel {
        let mut owned = [[0u32; 20]; MAX_PLAYERS];
        let mut territory = [0i32; MAX_PLAYERS];
        let mut edge_sum = [0i32; MAX_PLAYERS];
        let reachable = [0i32; MAX_PLAYERS];
        owned[our_id] = duel.owned_a;
        owned[opponent] = duel.owned_b;
        territory[our_id] = mask_popcount(&duel.owned_a);
        territory[opponent] = mask_popcount(&duel.owned_b);
        edge_sum[our_id] = mask_edge_sum(&duel.owned_a, &empty);
        edge_sum[opponent] = mask_edge_sum(&duel.owned_b, &empty);
        Voronoi {
            territory,
            edge_sum,
            reachable,
            still_connected: true,
            owned,
        }
    } else {
        compute_voronoi(state)
    };
    let territory = voronoi.territory[our_id] - voronoi.territory[opponent];
    let edges = voronoi.edge_sum[our_id] - voronoi.edge_sum[opponent];
    let mobility_diff = our_mobility - opp_mobility;
    let front = battlefront(&voronoi.owned[our_id], &voronoi.owned[opponent]);
    let center_penalty = -((our_col - 14).abs() + (our_row - 9).abs());
    let occupied_count = mask_popcount(&state.occupied.bits);
    let p = params();
    let center_weight = if p.center_div > 0 {
        (500 - occupied_count).max(0) / p.center_div
    } else {
        0
    };
    // Contested cells favour the bike that moves second (it can answer the
    // first mover's commitment), so credit them against the side to move.
    let tie_score = if to_move == our_id { -ties } else { ties } * p.ties;
    territory * p.terr
        + edges * p.edges
        + mobility_diff * p.mob
        + front * p.front
        + center_penalty * center_weight
        + tie_score
}

/// Free-for-all evaluation from `our_id`’s point of view.
///
/// We are not trying to 1v1 a specific rival. Score our reachable / unique
/// territory, our unique-cell edge sum (open frontier), subtract the strongest
/// remaining opponent’s territory and reach, add mobility so we do not get
/// boxed in, and (while the board is still empty) a 1v1-style pull toward
/// (14, 9). Death / sole survivor use mate scores like [`eval_1v1`].
///
/// **Where:** leaves of [`paranoid_max`].
/// **Why:** deep 1v1 minimax in a 3–4 player game treats others as frozen walls
/// and suicides. This leaf prefers “survive with space” over picking a fight.
fn eval_ffa(state: &State, our_id: usize, ply: i32) -> i32 {
    if !state.is_alive(our_id) {
        return -MATE_SCORE + ply;
    }
    if state.alive_mask.count_ones() == 1 {
        return MATE_SCORE - ply;
    }
    let mut voronoi = compute_voronoi_ffa(state, our_id);
    let player_count = state.player_count as usize;
    // A rival sealed into a tiny pocket dies before we do and its ribbon
    // vanishes: score the board as if that had already happened.
    let mut doomed = 0u8;
    for player in 0..player_count {
        if player != our_id
            && state.is_alive(player)
            && voronoi.reachable[player] <= params().doom
            && voronoi.reachable[player] < voronoi.reachable[our_id]
        {
            doomed |= 1 << player;
        }
    }
    if doomed != 0 {
        voronoi = compute_voronoi_ffa_ex(state, our_id, doomed);
    }
    let mut best_other_territory = 0;
    let mut best_other_reach = 0;
    for player in 0..player_count {
        if player == our_id || !state.is_alive(player) || doomed & (1 << player) != 0 {
            continue;
        }
        best_other_territory = best_other_territory.max(voronoi.territory[player]);
        best_other_reach = best_other_reach.max(voronoi.reachable[player]);
    }
    let our_col = state.head_x[our_id] as i32;
    let our_row = state.head_y[our_id] as i32;
    let center_penalty = -((our_col - 14).abs() + (our_row - 9).abs());
    let occupied_count = mask_popcount(&state.occupied.bits);
    let center_weight = (500 - occupied_count).max(0) / 80;
    let p = params();
    voronoi.reachable[our_id] * p.f_reach + voronoi.territory[our_id] * p.f_terr
        - best_other_territory * p.f_oterr
        - best_other_reach * p.f_oreach
        + voronoi.edge_sum[our_id] * p.f_edges
        + mobility(state, our_id) * p.f_mob
        + center_penalty * center_weight
}

/// Clock and move-ordering state for one call to [`choose_move`].
///
/// `killers[ply % 64]` stores up to two direction indices that caused a beta
/// cutoff at that ply; they are tried early next time. `timed_out` is sticky:
/// once set, search returns [`TIMEOUT_SCORE`] so the caller can discard a
/// partial iteration.
///
/// **Where:** constructed in [`choose_move`] for the 1v1 and FFA
/// iterative-deepening loops, then passed into [`negamax_1v1`] / paranoid search.
/// **Why:** CodinGame kills a late bot; we must stop on a completed depth,
/// not mid-ply, and still undo every apply.
struct Search {
    deadline: Instant,
    nodes: u64,
    timed_out: bool,
    /// Two killer direction indices per ply slot; `NO_MOVE` if empty.
    killers: [[u8; 2]; 64],
    /// Butterfly history: `history[player][dir]`, bumped on beta cutoffs.
    history: [[i16; 4]; MAX_PLAYERS],
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
            history: [[0; 4]; MAX_PLAYERS],
        }
    }

    /// Count a visited node. Every 64 nodes, compare wall clock to `deadline`.
    /// Checking every node is too expensive relative to our tiny branching factor.
    /// Called at the top of [`negamax_1v1`], [`paranoid_max`], and [`paranoid_min`].
    #[inline]
    fn check_time(&mut self) {
        self.nodes += 1;
        if (self.nodes & 63) == 0 && Instant::now() >= self.deadline {
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
/// [`negamax_1v1`] / [`paranoid_max`] / [`paranoid_min`].
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
    history: &[i16; 4],
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
        key += (history[dir as usize] as i32).min(2_000);
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

/// Negamax alpha-beta for a 1v1 duel.
///
/// Score is always from `to_move`’s point of view; the caller negates. `our_id`
/// and `opponent` are the two colours at the root and never swap — [`eval_1v1`]
/// is written from `our_id`’s side, then flipped if the opponent is to move.
///
/// `depth` is remaining plies to a leaf. `ply` is distance from the root
/// (used to prefer faster mates). `qs_left` is remaining quiescence extensions.
/// `last_dir_*` help move ordering (prefer continuing straight). Returns
/// [`TIMEOUT_SCORE`] if the budget expired; the caller must ignore that
/// iteration.
///
/// **Where:** only the iterative-deepening loop in [`choose_move`], and only
/// when exactly two bikes are alive and still share space.
/// **Why:** sequential Tron is a two-player game once it is a duel. A few
/// plies of minimax plus Voronoi leaves beat greedy 1-ply; FFA must not
/// call this (it treats extra bikes as frozen walls). Horizon nodes that are
/// still tactical ([`wants_quiescence`]) search extra plies instead of eval.
fn negamax_1v1(
    state: &mut State,
    our_id: usize,
    opponent: usize,
    to_move: usize,
    mut depth: i32,
    ply: i32,
    mut alpha: i32,
    beta: i32,
    last_dir_ours: u8,
    last_dir_opponent: u8,
    mut qs_left: i32,
    search: &mut Search,
    scratch: &mut Scratch,
) -> i32 {
    search.check_time();
    if search.timed_out {
        return TIMEOUT_SCORE;
    }
    let next_player = if to_move == our_id { opponent } else { our_id };
    // Every return is from `to_move`’s point of view so the caller can negate.
    // Scoring terminals from `our_id` instead made a forced win come back as
    // `-MATE` after the root negation — search then “found mate” on move 1
    // of an empty board and stopped deepening.
    if !state.is_alive(to_move) {
        return -MATE_SCORE + ply;
    }
    if !state.is_alive(next_player) {
        return MATE_SCORE - ply;
    }

    let (mut moves, move_count) = state.legal_moves(to_move);
    if move_count == 0 {
        return -MATE_SCORE + ply;
    }

    // Already cut (seen as a leaf in an earlier iteration): the chambers are
    // independent, so the fill comparison is the value. Terminal.
    if depth > 0 {
        let slot = (state.hash >> (64 - CUT_CACHE_BITS)) as usize;
        if scratch.cut_keys[slot] == state.hash {
            let score = scratch.cut_scores[slot];
            return if to_move == our_id { score } else { -score };
        }
    }

    // Leaf: side to move with no reply loses. Tactical leaves (adjacent heads,
    // one escape, or a cut this ply) extend instead of static eval.
    if depth <= 0 {
        if !wants_quiescence(state, our_id, opponent, qs_left, to_move, move_count) {
            let score = eval_1v1(state, our_id, opponent, to_move, ply, scratch);
            return if to_move == our_id { score } else { -score };
        }
        depth = 1;
        qs_left -= 1;
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
        target_row, &search.history[to_move],
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
            qs_left,
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
            search.history[to_move][moves[move_i] as usize] = search.history[to_move]
                [moves[move_i] as usize]
                .saturating_add((depth * depth).max(1).min(64) as i16);
            break;
        }
    }
    best_score
}

/// Root 1v1 search at `depth` with window `[alpha, beta]` from `our_id`’s seat.
///
/// `moves` must already be ordered. Returns `(best_dir, score, completed)`.
/// On timeout the board is still clean; `completed` is false and the caller
/// must discard this iteration.
///
/// **Where:** the 1v1 iterative-deepening loop in [`choose_move`]. Aspiration
/// calls this with `[prev ± ASPIRATION_DELTA]`, then again full-window on fail.
fn search_root_1v1(
    state: &mut State,
    our_id: usize,
    opponent: usize,
    moves: &[u8; 4],
    move_count: usize,
    depth: i32,
    mut alpha: i32,
    beta: i32,
    search: &mut Search,
    scratch: &mut Scratch,
) -> (u8, i32, bool) {
    let mut best_dir = moves[0];
    let mut best_score = -MATE_SCORE * 2;
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
            -beta,
            -alpha,
            moves[move_i],
            NO_MOVE,
            QS_MAX,
            search,
            scratch,
        );
        state.undo_step(our_id, old_col, old_row);
        if search.timed_out {
            return (best_dir, best_score, false);
        }
        if score > best_score {
            best_score = score;
            best_dir = moves[move_i];
        }
        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            break;
        }
    }
    (best_dir, best_score, true)
}

/// Next seat after `player`, wrapping in `0..player_count`.
#[inline]
fn next_seat(player: usize, player_count: usize) -> usize {
    (player + 1) % player_count
}

/// First living rival, preferring `threat` if they are still up.
#[inline]
fn living_threat(state: &State, our_id: usize, threat: usize) -> usize {
    if threat != our_id && state.is_alive(threat) {
        return threat;
    }
    let player_count = state.player_count as usize;
    for player in 0..player_count {
        if player != our_id && state.is_alive(player) {
            return player;
        }
    }
    threat
}

/// One legal step that keeps the most reachable space for `player`.
fn greedy_space_dir(
    state: &mut State,
    player: usize,
    moves: &[u8],
    move_count: usize,
    scratch: &mut Scratch,
) -> u8 {
    let _ = scratch;
    let mut best_dir = moves[0];
    let mut best = i32::MIN;
    for move_i in 0..move_count {
        let old_col = state.head_x[player];
        let old_row = state.head_y[player];
        state.apply(player, moves[move_i] as usize);
        // Same count as `flood_count`, on row bitboards instead of a cell BFS.
        let empty = empty_rows(state);
        let space = mask_popcount(&flood_mask(&head_seed(state, player, &empty), &empty));
        let hug = wall_neighbor_count(
            state,
            state.head_x[player] as i32,
            state.head_y[player] as i32,
        );
        state.undo_step(player, old_col, old_row);
        let score = space * 20 + hug;
        if score > best {
            best = score;
            best_dir = moves[move_i];
        }
    }
    best_dir
}

/// Max node of paranoid FFA search: it is `our_id`’s turn.
///
/// `depth` is remaining *our* plies before a leaf. Score is always from our
/// point of view (no negamax flip). Alpha-beta uses that same orientation.
///
/// **Where:** [`paranoid_min`] after a full opponent round, when depth remains.
/// **Why:** the Max half of “we vs a colluding field.”
fn paranoid_max(
    state: &mut State,
    our_id: usize,
    threat: usize,
    depth: i32,
    ply: i32,
    mut alpha: i32,
    beta: i32,
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
        return eval_ffa(state, our_id, ply);
    }
    let (mut moves, move_count) = state.legal_moves(our_id);
    if move_count == 0 {
        return -MATE_SCORE + ply;
    }
    let player_count = state.player_count as usize;
    let mut target_col = state.head_x[our_id] as i32;
    let mut target_row = state.head_y[our_id] as i32;
    for player in 0..player_count {
        if player != our_id && state.is_alive(player) {
            target_col = state.head_x[player] as i32;
            target_row = state.head_y[player] as i32;
            break;
        }
    }
    order_moves(
        state,
        our_id,
        &mut moves,
        move_count,
        NO_MOVE,
        search.killers[ply as usize % 64][0],
        NO_MOVE,
        target_col,
        target_row,
        &search.history[our_id],
    );
    let mut best_score = -MATE_SCORE * 2;
    for move_i in 0..move_count {
        let old_col = state.head_x[our_id];
        let old_row = state.head_y[our_id];
        state.apply(our_id, moves[move_i] as usize);
        let score = paranoid_min(
            state,
            our_id,
            next_seat(our_id, player_count),
            threat,
            depth - 1,
            ply + 1,
            alpha,
            beta,
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
        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            let slot = ply as usize % 64;
            if search.killers[slot][0] != moves[move_i] {
                search.killers[slot][1] = search.killers[slot][0];
                search.killers[slot][0] = moves[move_i];
            }
            search.history[our_id][moves[move_i] as usize] = search.history[our_id]
                [moves[move_i] as usize]
                .saturating_add((depth * depth).max(1).min(64) as i16);
            break;
        }
    }
    best_score
}

/// Min node: `to_move` is an opponent (or a dead seat we skip). `threat`
/// colludes to **minimize** our [`eval_ffa`]; other bikes play one greedy
/// space-keeping step (they fill, they do not hunt us).
///
/// Seating order is preserved. When the walk returns to `our_id`, the round is
/// over: leaf eval if `depth == 0`, else [`paranoid_max`].
///
/// **Where:** FFA root loop in [`choose_move`] after we apply a candidate, and
/// from [`paranoid_max`] after each of our moves.
/// **Why:** defensive FFA — assume every other bike tries to ruin our score,
/// not to fill their own pocket.
fn paranoid_min(
    state: &mut State,
    our_id: usize,
    mut to_move: usize,
    threat: usize,
    depth: i32,
    ply: i32,
    alpha: i32,
    mut beta: i32,
    search: &mut Search,
    scratch: &mut Scratch,
) -> i32 {
    search.check_time();
    if search.timed_out {
        return TIMEOUT_SCORE;
    }
    let player_count = state.player_count as usize;
    let threat = living_threat(state, our_id, threat);
    while to_move != our_id && !state.is_alive(to_move) {
        to_move = next_seat(to_move, player_count);
    }
    if to_move == our_id {
        return paranoid_max(state, our_id, threat, depth, ply, alpha, beta, search, scratch);
    }
    if !state.is_alive(our_id) {
        return -MATE_SCORE + ply;
    }
    if state.alive_mask.count_ones() == 1 {
        return MATE_SCORE - ply;
    }
    let (mut moves, move_count) = state.legal_moves(to_move);
    if move_count == 0 {
        let trail = state.trail[to_move];
        let head_x = state.head_x[to_move];
        let head_y = state.head_y[to_move];
        state.kill(to_move);
        let score = paranoid_min(
            state,
            our_id,
            next_seat(to_move, player_count),
            threat,
            depth,
            ply + 1,
            alpha,
            beta,
            search,
            scratch,
        );
        state.restore_killed(to_move, trail, head_x, head_y);
        return score;
    }
    if to_move != threat {
        let dir = greedy_space_dir(state, to_move, &moves, move_count, scratch);
        let old_col = state.head_x[to_move];
        let old_row = state.head_y[to_move];
        state.apply(to_move, dir as usize);
        let score = paranoid_min(
            state,
            our_id,
            next_seat(to_move, player_count),
            threat,
            depth,
            ply + 1,
            alpha,
            beta,
            search,
            scratch,
        );
        state.undo_step(to_move, old_col, old_row);
        if search.timed_out {
            return TIMEOUT_SCORE;
        }
        return score;
    }
    order_moves(
        state,
        to_move,
        &mut moves,
        move_count,
        NO_MOVE,
        search.killers[ply as usize % 64][1],
        NO_MOVE,
        state.head_x[our_id] as i32,
        state.head_y[our_id] as i32,
        &search.history[to_move],
    );
    let mut best_score = MATE_SCORE * 2;
    for move_i in 0..move_count {
        let old_col = state.head_x[to_move];
        let old_row = state.head_y[to_move];
        state.apply(to_move, moves[move_i] as usize);
        let score = paranoid_min(
            state,
            our_id,
            next_seat(to_move, player_count),
            threat,
            depth,
            ply + 1,
            alpha,
            beta,
            search,
            scratch,
        );
        state.undo_step(to_move, old_col, old_row);
        if search.timed_out {
            return TIMEOUT_SCORE;
        }
        if score < best_score {
            best_score = score;
        }
        if score < beta {
            beta = score;
        }
        if alpha >= beta {
            search.history[to_move][moves[move_i] as usize] = search.history[to_move]
                [moves[move_i] as usize]
                .saturating_add((depth.max(1) * depth.max(1)).max(1).min(64) as i16);
            break;
        }
    }
    best_score
}

/// Isolated-chamber policy: pick the step that keeps the most fillable space.
///
/// Score = approx-fill + remaining flood + wall-hug, with a small bonus for
/// continuing in `last_dir`. If a step shrinks the reachable set (we walked
/// into a pocket and walled ourselves off from the rest), apply a heavy
/// penalty. Returns `None` if there is no legal move.
///
/// **Where:** [`choose_move`] when nobody else is alive, and inside the
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
        let space = flood_after.min(checkerboard_reach_bound(state, player));
        let col = state.head_x[player] as i32;
        let row = state.head_y[player] as i32;
        let mut score = fill * 80 + space * 30 + wall_neighbor_count(state, col, row) * 12;
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

/// Choose a direction (0=UP .. 3=RIGHT) for `our_id` within `budget_ms`.
///
/// Policy:
/// - No legal moves → dummy `0` (we crash next turn anyway).
/// - One legal move → play it immediately.
/// - Nobody else alive → isolated [`fill_direction`].
/// - 1v1 and the two chambers are cut off → greedy fill to completion
///   (this path skips minimax, so the turn budget is available).
/// - 3+ living players → iterative-deepening paranoid search ([`paranoid_min`]
///   after each of our moves). Closest rival is Min; others play one greedy
///   space-keeping reply. No 2-ply warmup; ID starts from the first ordered
///   move, cap 50.
/// - 1v1 still connected → iterative-deepening [`negamax_1v1`] up to depth 50
///   or the time budget. After depth 1, each iteration uses an aspiration
///   window of [`ASPIRATION_DELTA`] around the previous score and re-searches
///   full-window on fail-high/fail-low (skipped near mate). Tactical leaves
///   (adjacent heads, one escape, or a cut this ply) extend up to [`QS_MAX`]
///   extra plies. There is no 2-ply Voronoi warmup; ID starts from the first
///   ordered move so search gets the full remaining budget.
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

    let mut other_ids = [0usize; MAX_PLAYERS];
    let mut other_count = 0usize;
    for player in 0..state.player_count as usize {
        if player != our_id && state.is_alive(player) {
            other_ids[other_count] = player;
            other_count += 1;
        }
    }
    if other_count == 0 {
        return fill_direction(state, our_id, last_dir, scratch).unwrap_or(moves[0]);
    }
    let is_duel = other_count == 1;
    let opponent = if is_duel {
        other_ids[0]
    } else {
        main_opponent(state, our_id).unwrap_or(other_ids[0])
    };

    #[cfg(any(test, feature = "local"))]
    let occupied_count: u32 = (0..20)
        .map(|row| state.occupied.bits[row].count_ones())
        .sum();
    // Declare “cut off” only if both the BFS test and Voronoi agree we cannot
    // meet. FFA is never treated as separated: a rival’s death reopens the
    // board, and the paranoid search (which sees rivals run out of moves and
    // vanish inside its horizon) filled sealed chambers better than the
    // primed rollout did, even with the rollout modelling those deaths
    // (4p at 20 ms: −8 / +8 vs +11 Elo for plain search).
    let still_connected = !is_duel
        || shares_space(state, our_id, opponent)
        || compute_voronoi(state).still_connected;
    let separated = is_duel && !still_connected;
    #[cfg(any(test, feature = "local"))]
    {
        CHOOSE_MOVE_COUNT.fetch_add(1, Ordering::Relaxed);
        if separated {
            SEPARATED_MOVE_COUNT.fetch_add(1, Ordering::Relaxed);
            if occupied_count < 24 {
                EARLY_SEPARATION_COUNT.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    let (target_col, target_row) = (state.head_x[opponent] as i32, state.head_y[opponent] as i32);
    order_moves(
        state, our_id, &mut moves, move_count, NO_MOVE, NO_MOVE, last_dir, target_col, target_row,
        &[0; 4],
    );

    // Isolated chamber: try each first step, then greedy-fill until death.
    // Minimax does not run on this path, so use the turn budget; cap at
    // board size only as a fuse if apply ever failed to occupy.
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
            while extra < BOARD_CELLS as i32 {
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

    // Open 1v1 and FFA both skip a 2-ply warmup so iterative deepening gets
    // the full remaining budget. Seed is the first ordered legal move.
    let mut best_dir = moves[0];
    let mut best_score = 0i32;

    let remaining = Duration::from_millis(budget_ms.max(1)).saturating_sub(start.elapsed());
    let mut search = Search::new(remaining);
    let mut principal_dir = best_dir;
    let mut completed_depth = 0i32;

    if !is_duel {
        // FFA: each ID ply is our move, then a Min round (closest rival
        // branches; other bikes play one greedy space-keeping reply).
        let max_ffa_depth = 50;
        let player_count = state.player_count as usize;
        for depth in 1..=max_ffa_depth {
            if Instant::now() >= search.deadline {
                break;
            }
            let mut iter_best_dir = principal_dir;
            let mut iter_best_score = -MATE_SCORE * 2;
            let mut completed_iteration = true;
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
                &search.history[our_id],
            );
            for move_i in 0..move_count {
                let old_col = state.head_x[our_id];
                let old_row = state.head_y[our_id];
                state.apply(our_id, moves[move_i] as usize);
                let score = paranoid_min(
                    state,
                    our_id,
                    next_seat(our_id, player_count),
                    opponent,
                    depth - 1,
                    1,
                    iter_best_score,
                    MATE_SCORE * 2,
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
                completed_depth = depth;
                principal_dir = iter_best_dir;
                best_dir = iter_best_dir;
                best_score = iter_best_score;
                if iter_best_score.abs() >= MATE_SCORE - 200 {
                    break;
                }
            } else {
                break;
            }
        }
    } else {
        // 1v1: search as deep as time allows (cap 50). Depth 1 is full-window;
        // later depths aspirate around the previous score and re-search on fail.
        let max_depth = 50;
        let mut prev_score: Option<i32> = None;
        for depth in 1..=max_depth {
            if Instant::now() >= search.deadline {
                break;
            }
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
                &search.history[our_id],
            );

            let mut window_alpha = -MATE_SCORE * 2;
            let mut window_beta = MATE_SCORE * 2;
            let mut aspired = false;
            if let Some(prev) = prev_score {
                if prev.abs() < MATE_SCORE - 200 {
                    window_alpha = prev.saturating_sub(ASPIRATION_DELTA);
                    window_beta = prev.saturating_add(ASPIRATION_DELTA);
                    aspired = true;
                }
            }

            let (mut iter_best_dir, mut iter_best_score, mut completed_iteration) =
                search_root_1v1(
                    state,
                    our_id,
                    opponent,
                    &moves,
                    move_count,
                    depth,
                    window_alpha,
                    window_beta,
                    &mut search,
                    scratch,
                );
            if !completed_iteration {
                break;
            }
            if aspired
                && (iter_best_score <= window_alpha || iter_best_score >= window_beta)
            {
                if Instant::now() >= search.deadline {
                    break;
                }
                principal_dir = iter_best_dir;
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
                    &search.history[our_id],
                );
                let re = search_root_1v1(
                    state,
                    our_id,
                    opponent,
                    &moves,
                    move_count,
                    depth,
                    -MATE_SCORE * 2,
                    MATE_SCORE * 2,
                    &mut search,
                    scratch,
                );
                iter_best_dir = re.0;
                iter_best_score = re.1;
                completed_iteration = re.2;
                if !completed_iteration {
                    break;
                }
            }

            principal_dir = iter_best_dir;
            best_dir = iter_best_dir;
            best_score = iter_best_score;
            completed_depth = depth;
            prev_score = Some(iter_best_score);
            if iter_best_score.abs() >= MATE_SCORE - 200 {
                break;
            }
        }
    }

    eprintln!(
        "{} mm {} {}ms n={} d={} a={}",
        DIR_NAME[best_dir as usize],
        best_score,
        start.elapsed().as_millis().max(1),
        search.nodes,
        completed_depth,
        state.alive_mask.count_ones(),
    );
    let _ = std::io::stderr().flush();
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
/// update [`Tracker`], print `UP|DOWN|LEFT|RIGHT`. Each turn gets 95 ms,
/// unless [`parse_budget_ms`] overrides (SPRT).
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

#[cfg(any(test, feature = "local"))]
mod local;

#[cfg(test)]
mod voronoi_tests;

fn main() {
    #[cfg(feature = "local")]
    if local::run() {
        return;
    }
    codingame();
}
