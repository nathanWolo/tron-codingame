//! CodinGame Tron Battle, two players, sequential turns.
//!
//! The board, the row-bitboard floods and the space-filling estimators are
//! ported from the alpha-beta engine in `tron-codingame/src/main.rs`.

pub const W: i32 = 30;
pub const H: i32 = 20;
pub const CELLS: usize = 600;
pub const DIR_X: [i32; 4] = [0, 0, -1, 1];
pub const DIR_Y: [i32; 4] = [-1, 1, 0, 0];
pub const DIR_NAME: [&str; 4] = ["UP", "DOWN", "LEFT", "RIGHT"];
pub const NO_MOVE: u8 = 4;
pub const COL_MASK: u32 = (1 << 30) - 1;
pub const EVEN_COLS: u32 = 0x1555_5555;
/// Distance of a cell a bike cannot reach.
pub const UNREACH: u8 = 255;

#[inline]
pub fn in_bounds(x: i32, y: i32) -> bool {
    x >= 0 && x < W && y >= 0 && y < H
}

#[inline]
pub fn cell(x: i32, y: i32) -> usize {
    (y * W + x) as usize
}

/// A position: occupancy (trails and heads), the two heads and who moves next.
#[derive(Clone, Copy)]
pub struct State {
    pub occ: [u32; 20],
    pub hx: [i8; 2],
    pub hy: [i8; 2],
    pub to_move: u8,
}

impl State {
    pub fn new(p0: (i32, i32), p1: (i32, i32)) -> State {
        let mut s = State { occ: [0; 20], hx: [0; 2], hy: [0; 2], to_move: 0 };
        s.place(0, p0.0, p0.1);
        s.place(1, p1.0, p1.1);
        s
    }

    /// Put `p`'s head on `(x, y)`; the cell becomes trail.
    #[inline]
    pub fn place(&mut self, p: usize, x: i32, y: i32) {
        self.occ[y as usize] |= 1u32 << x;
        self.hx[p] = x as i8;
        self.hy[p] = y as i8;
    }

    #[inline]
    pub fn is_occ(&self, x: i32, y: i32) -> bool {
        self.occ[y as usize] >> x & 1 != 0
    }

    #[inline]
    pub fn free(&self, x: i32, y: i32) -> bool {
        in_bounds(x, y) && !self.is_occ(x, y)
    }

    /// Bit `d` set if `p` can step in direction `d`.
    #[inline]
    pub fn legal_mask(&self, p: usize) -> u8 {
        let (x, y) = (self.hx[p] as i32, self.hy[p] as i32);
        let mut m = 0u8;
        for d in 0..4 {
            if self.free(x + DIR_X[d], y + DIR_Y[d]) {
                m |= 1 << d;
            }
        }
        m
    }

    /// Step `p` one cell in a legal direction.
    #[inline]
    pub fn step(&mut self, p: usize, d: usize) {
        let (x, y) = (self.hx[p] as i32 + DIR_X[d], self.hy[p] as i32 + DIR_Y[d]);
        self.place(p, x, y);
    }

    /// Undo a `step`: clear the head cell and put the head back.
    #[inline]
    pub fn undo(&mut self, p: usize, ox: i8, oy: i8) {
        self.occ[self.hy[p] as usize] &= !(1u32 << self.hx[p]);
        self.hx[p] = ox;
        self.hy[p] = oy;
    }

    /// The player to move steps in `d`, then the turn passes.
    #[inline]
    pub fn play(&mut self, d: usize) {
        self.step(self.to_move as usize, d);
        self.to_move ^= 1;
    }

    pub fn occupied_count(&self) -> u32 {
        self.occ.iter().map(|r| r.count_ones()).sum()
    }
}

// --- Row bitboards ------------------------------------------------------------------

#[inline(always)]
pub fn empty_rows(st: &State) -> [u32; 20] {
    let mut e = [0u32; 20];
    for r in 0..20 {
        e[r] = !st.occ[r] & COL_MASK;
    }
    e
}

/// One orthogonal step from `src` into `empty` cells.
#[inline(always)]
pub fn expand(src: &[u32; 20], empty: &[u32; 20]) -> [u32; 20] {
    let mut out = [0u32; 20];
    let mut prev = 0u32;
    let mut curr = src[0];
    for r in 0..19 {
        let next = src[r + 1];
        out[r] = ((curr << 1) | (curr >> 1) | prev | next) & empty[r];
        prev = curr;
        curr = next;
    }
    out[19] = ((curr << 1) | (curr >> 1) | prev) & empty[19];
    out
}

/// Empty 4-neighbours of `p`'s head.
#[inline(always)]
pub fn head_seed(st: &State, p: usize, empty: &[u32; 20]) -> [u32; 20] {
    let mut seed = [0u32; 20];
    let (hx, hy) = (st.hx[p] as i32, st.hy[p] as i32);
    for d in 0..4 {
        let (x, y) = (hx + DIR_X[d], hy + DIR_Y[d]);
        if in_bounds(x, y) {
            seed[y as usize] |= (1u32 << x) & empty[y as usize];
        }
    }
    seed
}

pub fn flood(seed: &[u32; 20], empty: &[u32; 20]) -> [u32; 20] {
    let mut filled = *seed;
    loop {
        let next = expand(&filled, empty);
        let mut changed = false;
        for r in 0..20 {
            let add = next[r] & !filled[r];
            if add != 0 {
                filled[r] |= add;
                changed = true;
            }
        }
        if !changed {
            return filled;
        }
    }
}

#[inline]
pub fn popcount(mask: &[u32; 20]) -> i32 {
    mask.iter().map(|r| r.count_ones()).sum::<u32>() as i32
}

#[inline]
pub fn overlap(a: &[u32; 20], b: &[u32; 20]) -> bool {
    for r in 0..20 {
        if a[r] & b[r] != 0 {
            return true;
        }
    }
    false
}

pub fn parity_counts(mask: &[u32; 20]) -> (i32, i32) {
    let (mut even, mut odd) = (0i32, 0i32);
    for r in 0..20 {
        let even_on_row = if r % 2 == 0 { EVEN_COLS } else { !EVEN_COLS & COL_MASK };
        even += (mask[r] & even_on_row).count_ones() as i32;
        odd += (mask[r] & !even_on_row & COL_MASK).count_ones() as i32;
    }
    (even, odd)
}

/// Longest path through a bipartite set whose first cell has `start_parity`.
#[inline]
pub fn checkerboard_path_bound(start_parity: i32, even: i32, odd: i32) -> i32 {
    let (same, other) = if start_parity == 0 { (even, odd) } else { (odd, even) };
    same.min(other) * 2 + i32::from(same > other)
}

// --- Distances and the features every evaluation starts from ---------------------------

/// What the network and the search need to know about a position.
pub struct Feat {
    /// Steps for the mover (`[0]`) and the opponent (`[1]`) to reach each cell
    /// through empty cells; a head's empty neighbours are at 1. `UNREACH` if never.
    pub dist: [[u8; CELLS]; 2],
    pub reach: [[u32; 20]; 2],
    /// Some empty cell is reachable by both bikes.
    pub connected: bool,
}

impl Feat {
    pub fn new() -> Feat {
        Feat { dist: [[UNREACH; CELLS]; 2], reach: [[0; 20]; 2], connected: false }
    }
}

/// Breadth-first distances from `p`'s head, as row-bitboard waves.
pub fn bfs_dist(st: &State, p: usize, empty: &[u32; 20], dist: &mut [u8; CELLS]) -> [u32; 20] {
    dist.fill(UNREACH);
    let mut frontier = head_seed(st, p, empty);
    let mut visited = frontier;
    let mut d = 1u8;
    loop {
        let mut any = false;
        for r in 0..20 {
            let mut bits = frontier[r];
            while bits != 0 {
                let x = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                dist[r * 30 + x] = d;
                any = true;
            }
        }
        if !any {
            break;
        }
        let mut next = expand(&frontier, empty);
        for r in 0..20 {
            next[r] &= !visited[r];
            visited[r] |= next[r];
        }
        frontier = next;
        if d < 254 {
            d += 1;
        }
    }
    visited
}

pub fn features(st: &State, f: &mut Feat) {
    let empty = empty_rows(st);
    let me = st.to_move as usize;
    f.reach[0] = bfs_dist(st, me, &empty, &mut f.dist[0]);
    f.reach[1] = bfs_dist(st, me ^ 1, &empty, &mut f.dist[1]);
    f.connected = overlap(&f.reach[0], &f.reach[1]);
}

/// True if the bikes can still reach a common empty cell.
pub fn connected(st: &State) -> bool {
    let empty = empty_rows(st);
    let a = flood(&head_seed(st, 0, &empty), &empty);
    let b = flood(&head_seed(st, 1, &empty), &empty);
    overlap(&a, &b)
}

// --- Space filling (the cut-off endgame) -----------------------------------------------

/// Visit stamps for the fill DFS, so nothing is cleared per call.
pub struct Scratch {
    stamp: [u32; CELLS],
    generation: u32,
}

impl Scratch {
    pub fn new() -> Scratch {
        Scratch { stamp: [0; CELLS], generation: 1 }
    }

    fn next_generation(&mut self) -> u32 {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.stamp.fill(0);
            self.generation = 1;
        }
        self.generation
    }
}

pub fn flood_count(st: &State, p: usize) -> i32 {
    let empty = empty_rows(st);
    popcount(&flood(&head_seed(st, p, &empty), &empty))
}

/// Checkerboard upper bound on filling every empty cell `p` can reach.
pub fn checkerboard_reach_bound(st: &State, p: usize) -> i32 {
    let empty = empty_rows(st);
    let reach = flood(&head_seed(st, p, &empty), &empty);
    let (even, odd) = parity_counts(&reach);
    let start_parity = (st.hx[p] as i32 + st.hy[p] as i32 + 1) & 1;
    checkerboard_path_bound(start_parity, even, odd)
}

fn empty_degree(st: &State, x: i32, y: i32) -> i32 {
    let mut n = 0;
    for d in 0..4 {
        if st.free(x + DIR_X[d], y + DIR_Y[d]) {
            n += 1;
        }
    }
    n
}

pub fn wall_neighbor_count(st: &State, x: i32, y: i32) -> i32 {
    4 - empty_degree(st, x, y)
}

/// True if occupying `(x, y)` would split its empty 4-neighbours into
/// pockets that the 8-cell ring around it does not connect.
fn is_local_cut(st: &State, x: i32, y: i32) -> bool {
    // Ring clockwise from NW; the 4-neighbours sit at slots 1 (up), 3 (right),
    // 5 (down), 7 (left).
    const RING: [(i32, i32); 8] =
        [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)];
    let mut ring = [false; 8];
    for i in 0..8 {
        ring[i] = st.free(x + RING[i].0, y + RING[i].1);
    }
    let open = [ring[1], ring[3], ring[5], ring[7]];
    let open_count = open.iter().filter(|&&o| o).count();
    if open_count <= 1 || open_count >= 4 {
        return false;
    }
    let start = [1usize, 3, 5, 7].into_iter().find(|&s| ring[s]).unwrap();
    // Walk the ring both ways from `start` through empty cells. Consecutive
    // ring slots are always 4-adjacent.
    let mut seen = [false; 8];
    seen[start] = true;
    let mut i = (start + 1) % 8;
    while ring[i] && !seen[i] {
        seen[i] = true;
        i = (i + 1) % 8;
    }
    let mut i = (start + 7) % 8;
    while ring[i] && !seen[i] {
        seen[i] = true;
        i = (i + 7) % 8;
    }
    for s in [1usize, 3, 5, 7] {
        if ring[s] && !seen[s] {
            return true;
        }
    }
    false
}

struct FillScore {
    est: i32,
    even: i32,
    odd: i32,
}

#[inline]
fn cap_fill(x: i32, y: i32, est: i32, even: i32, odd: i32) -> i32 {
    est.min(checkerboard_path_bound((x + y) & 1, even, odd))
}

/// Estimate of the cells a space-fill starting on `(x, y)` can take: sum the
/// branches, except at a local cut (take the best pocket), capped by the
/// checkerboard bound.
fn dfs_fill_score(st: &State, x: i32, y: i32, sc: &mut Scratch, generation: u32) -> FillScore {
    sc.stamp[cell(x, y)] = generation;
    let mut even = i32::from((x + y) & 1 == 0);
    let mut odd = 1 - even;
    let mut next = [(0i32, 0i32); 4];
    let mut n = 0;
    for d in 0..4 {
        let (nx, ny) = (x + DIR_X[d], y + DIR_Y[d]);
        if st.free(nx, ny) && sc.stamp[cell(nx, ny)] != generation {
            next[n] = (nx, ny);
            n += 1;
        }
    }
    if n == 0 {
        return FillScore { est: 1, even, odd };
    }
    if n == 1 {
        let child = dfs_fill_score(st, next[0].0, next[0].1, sc, generation);
        even += child.even;
        odd += child.odd;
        return FillScore { est: cap_fill(x, y, 1 + child.est, even, odd), even, odd };
    }
    if is_local_cut(st, x, y) {
        let (mut best_est, mut best_even, mut best_odd) = (0, 0, 0);
        for i in 0..n {
            let (px, py) = next[i];
            if sc.stamp[cell(px, py)] != generation {
                let pocket = dfs_fill_score(st, px, py, sc, generation);
                if pocket.est > best_est {
                    best_est = pocket.est;
                    best_even = pocket.even;
                    best_odd = pocket.odd;
                }
            }
        }
        even += best_even;
        odd += best_odd;
        FillScore { est: cap_fill(x, y, 1 + best_est, even, odd), even, odd }
    } else {
        let mut total = 1;
        for i in 0..n {
            let (nx, ny) = next[i];
            if sc.stamp[cell(nx, ny)] != generation {
                let child = dfs_fill_score(st, nx, ny, sc, generation);
                even += child.even;
                odd += child.odd;
                total += child.est;
            }
        }
        FillScore { est: cap_fill(x, y, total, even, odd), even, odd }
    }
}

/// Estimate of how many more moves `p` can make in its own pocket.
pub fn approx_fill(st: &State, p: usize, sc: &mut Scratch) -> i32 {
    let generation = sc.next_generation();
    let (hx, hy) = (st.hx[p] as i32, st.hy[p] as i32);
    let mut best = 0;
    for d in 0..4 {
        let (x, y) = (hx + DIR_X[d], hy + DIR_Y[d]);
        if st.free(x, y) && sc.stamp[cell(x, y)] != generation {
            best = best.max(dfs_fill_score(st, x, y, sc, generation).est);
        }
    }
    best
}

/// The greedy space-filling step: keep the fill estimate and the remaining
/// space high, hug walls, and never seal off a region.
pub fn fill_direction(st: &mut State, p: usize, last_dir: u8, sc: &mut Scratch) -> Option<u8> {
    let legal = st.legal_mask(p);
    if legal == 0 {
        return None;
    }
    let flood_before = flood_count(st, p);
    let mut best_dir = NO_MOVE;
    let mut best_score = i32::MIN;
    for d in 0..4u8 {
        if legal >> d & 1 == 0 {
            continue;
        }
        let (ox, oy) = (st.hx[p], st.hy[p]);
        st.step(p, d as usize);
        let flood_after = flood_count(st, p);
        let fill = approx_fill(st, p, sc);
        let space = flood_after.min(checkerboard_reach_bound(st, p));
        let (x, y) = (st.hx[p] as i32, st.hy[p] as i32);
        let mut score = fill * 80 + space * 30 + wall_neighbor_count(st, x, y) * 12;
        if d == last_dir {
            score += 8;
        }
        if flood_after + 1 < flood_before {
            score -= (flood_before - flood_after) * 400;
        }
        st.undo(p, ox, oy);
        if score > best_score {
            best_score = score;
            best_dir = d;
        }
    }
    Some(best_dir)
}

/// Moves `p` makes if it steps in `first` and then fills greedily to the end.
pub fn greedy_len(st: &State, p: usize, first: u8, sc: &mut Scratch) -> i32 {
    let mut s = *st;
    s.step(p, first as usize);
    let mut n = 1;
    let mut prev = first;
    while let Some(d) = fill_direction(&mut s, p, prev, sc) {
        s.step(p, d as usize);
        prev = d;
        n += 1;
    }
    n
}

/// Primed greedy fill: try each first step, fill greedily to the end, keep the
/// longest. Returns the step and the number of moves it leads to (0 if stuck).
pub fn primed_fill(st: &State, p: usize, sc: &mut Scratch) -> (u8, i32) {
    let legal = st.legal_mask(p);
    let mut best = (NO_MOVE, 0);
    let mut best_score = i32::MIN;
    for d in 0..4u8 {
        if legal >> d & 1 == 0 {
            continue;
        }
        let len = greedy_len(st, p, d, sc);
        let (x, y) = (st.hx[p] as i32 + DIR_X[d as usize], st.hy[p] as i32 + DIR_Y[d as usize]);
        let score = len * 50 + wall_neighbor_count(st, x, y);
        if score > best_score {
            best_score = score;
            best = (d, len);
        }
    }
    best
}

/// Who wins once the bikes are cut off: the mover makes `a` more moves and the
/// opponent `b`; the mover moves first, so it needs `a > b`.
/// Returns the winner and whether the cheap estimates decided it.
pub fn adjudicate(st: &State, sc: &mut Scratch) -> (usize, bool) {
    let me = st.to_move as usize;
    let a = approx_fill(st, me, sc);
    let b = approx_fill(st, me ^ 1, sc);
    if (a - b).abs() > 10 {
        return (if a > b { me } else { me ^ 1 }, true);
    }
    let a = primed_fill(st, me, sc).1;
    let b = primed_fill(st, me ^ 1, sc).1;
    (if a > b { me } else { me ^ 1 }, false)
}

/// Search value of a cut-off position for the mover, from the cheap estimates.
pub fn cutoff_value(st: &State, sc: &mut Scratch) -> f32 {
    let me = st.to_move as usize;
    let a = approx_fill(st, me, sc);
    let b = approx_fill(st, me ^ 1, sc);
    (((a - b) as f32 - 0.5) / 3.0).tanh()
}
