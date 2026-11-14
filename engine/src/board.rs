//! CodinGame Tron Battle: two to four players, sequential turns in seat order.
//!
//! A bike that cannot move when its turn comes is out, and its trail leaves
//! the board. The last bike standing wins; the others place in the reverse
//! order of their deaths.
//!
//! The row-bitboard floods and the space-filling estimators are ported from
//! the alpha-beta engine in `legacy/alpha-beta/src/main.rs`.

use std::time::{Duration, Instant};

pub const W: i32 = 30;
pub const H: i32 = 20;
pub const CELLS: usize = 600;
pub const MAXP: usize = 4;
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

/// A position. After `play` (and `new`) it is settled: the game is over or the
/// player to move has a legal move.
#[derive(Clone, Copy)]
pub struct State {
    /// Trails and heads of the bikes still in the game.
    pub occ: [u32; 20],
    /// Each player's trail (all zero once it is out).
    pub trail: [[u32; 20]; MAXP],
    pub hx: [i8; MAXP],
    pub hy: [i8; MAXP],
    /// Placement of a player that is out: `n` for the first out, 2 for the
    /// runner-up; 1 for the winner once the game is over; 0 while undecided.
    pub rank: [u8; MAXP],
    /// The last direction each player moved (`NO_MOVE` before its first move).
    pub last: [u8; MAXP],
    /// Players in the game, dead ones included (2..=4).
    pub n: u8,
    /// Bit `p`: player `p` is still in the game.
    pub alive: u8,
    pub to_move: u8,
}

impl State {
    pub fn new(starts: &[(i32, i32)]) -> State {
        let n = starts.len();
        let mut s = State {
            occ: [0; 20],
            trail: [[0; 20]; MAXP],
            hx: [-1; MAXP],
            hy: [-1; MAXP],
            rank: [0; MAXP],
            last: [NO_MOVE; MAXP],
            n: n as u8,
            alive: (1u8 << n) - 1,
            to_move: 0,
        };
        for (p, &(x, y)) in starts.iter().enumerate() {
            s.place(p, x, y);
        }
        s.settle();
        s
    }

    /// Put `p`'s head on `(x, y)`; the cell becomes trail.
    #[inline]
    pub fn place(&mut self, p: usize, x: i32, y: i32) {
        self.occ[y as usize] |= 1u32 << x;
        self.trail[p][y as usize] |= 1u32 << x;
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

    #[inline]
    pub fn is_alive(&self, p: usize) -> bool {
        self.alive >> p & 1 != 0
    }

    #[inline]
    pub fn alive_count(&self) -> u32 {
        self.alive.count_ones()
    }

    /// One bike (or none) is left.
    #[inline]
    pub fn over(&self) -> bool {
        self.alive & self.alive.wrapping_sub(1) == 0
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

    /// Step `p` one cell in a legal direction (turn order untouched).
    #[inline]
    pub fn step(&mut self, p: usize, d: usize) {
        let (x, y) = (self.hx[p] as i32 + DIR_X[d], self.hy[p] as i32 + DIR_Y[d]);
        self.place(p, x, y);
    }

    /// Undo a `step`: clear the head cell and put the head back.
    #[inline]
    pub fn undo(&mut self, p: usize, ox: i8, oy: i8) {
        let bit = !(1u32 << self.hx[p]);
        self.occ[self.hy[p] as usize] &= bit;
        self.trail[p][self.hy[p] as usize] &= bit;
        self.hx[p] = ox;
        self.hy[p] = oy;
    }

    /// The next player after `p` in seat order that is still in the game.
    #[inline]
    pub fn next_alive(&self, p: usize) -> usize {
        let n = self.n as usize;
        let mut q = p;
        for _ in 0..n {
            q = if q + 1 == n { 0 } else { q + 1 };
            if self.is_alive(q) {
                return q;
            }
        }
        p
    }

    /// Take `p` out of the game: its trail leaves the board and it places last
    /// among the bikes that were still in.
    pub fn kill(&mut self, p: usize) {
        if !self.is_alive(p) {
            return;
        }
        self.rank[p] = self.alive_count() as u8;
        for r in 0..20 {
            self.occ[r] &= !self.trail[p][r];
            self.trail[p][r] = 0;
        }
        self.alive &= !(1 << p);
        if self.alive_count() == 1 {
            self.rank[self.alive.trailing_zeros() as usize] = 1;
        }
    }

    /// Remove movers that have no legal move until someone can move or the
    /// game is over.
    pub fn settle(&mut self) {
        while !self.over() && self.legal_mask(self.to_move as usize) == 0 {
            let p = self.to_move as usize;
            self.kill(p);
            self.to_move = self.next_alive(p) as u8;
        }
    }

    /// The player to move steps in the legal direction `d`; the turn passes.
    #[inline]
    pub fn play(&mut self, d: usize) {
        let p = self.to_move as usize;
        self.step(p, d);
        self.last[p] = d as u8;
        self.to_move = self.next_alive(p) as u8;
        self.settle();
    }

    /// The player to move is taken out (it crashed or played an illegal move).
    pub fn forfeit(&mut self) {
        let p = self.to_move as usize;
        self.kill(p);
        self.to_move = self.next_alive(p) as u8;
        self.settle();
    }

    /// The players still in, in turn order from the player to move.
    #[inline]
    pub fn slots(&self) -> ([u8; MAXP], usize) {
        let mut s = [0u8; MAXP];
        let mut k = 0;
        let n = self.n as usize;
        let mut p = self.to_move as usize;
        for _ in 0..n {
            if self.is_alive(p) {
                s[k] = p as u8;
                k += 1;
            }
            p = if p + 1 == n { 0 } else { p + 1 };
        }
        (s, k)
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

/// What the network and the search need to know about a position. Everything
/// is indexed by slot: slot 0 is the player to move, the following slots are
/// the other players still in, in turn order.
pub struct Feat {
    pub slot: [u8; MAXP],
    pub nslots: usize,
    /// Steps for each slot's bike to reach each cell through empty cells; a
    /// head's empty neighbours are at 1. `UNREACH` if never.
    pub dist: [[u8; CELLS]; MAXP],
    pub reach: [[u32; 20]; MAXP],
    /// Some empty cell is reachable by the mover and by another bike.
    pub mover_connected: bool,
}

impl Feat {
    pub fn new() -> Feat {
        Feat {
            slot: [0; MAXP],
            nslots: 0,
            dist: [[UNREACH; CELLS]; MAXP],
            reach: [[0; 20]; MAXP],
            mover_connected: false,
        }
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

/// Fill `f` for a position that is not over.
pub fn features(st: &State, f: &mut Feat) {
    let empty = empty_rows(st);
    let (slot, n) = st.slots();
    f.slot = slot;
    f.nslots = n;
    f.mover_connected = false;
    for k in 0..n {
        f.reach[k] = bfs_dist(st, slot[k] as usize, &empty, &mut f.dist[k]);
        if k > 0 && overlap(&f.reach[0], &f.reach[k]) {
            f.mover_connected = true;
        }
    }
}

// --- Space filling (a bike alone in its region) ----------------------------------------

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
    let start = [1usize, 3, 5, 7].iter().copied().find(|&s| ring[s]).unwrap();
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
/// space high, hug walls, and never seal off a region. `st` is restored.
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

// paste:skip-begin
/// Moves `p` makes if it steps in `first` and then fills greedily to the end,
/// with the rest of the board frozen.
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

/// Who wins once the last two bikes are cut off from each other: the mover
/// makes `a` more moves and the other `b`; the mover moves first, so it needs
/// `a > b`. Returns the winner and whether the cheap estimates decided it.
pub fn adjudicate(st: &State, sc: &mut Scratch) -> (usize, bool) {
    let me = st.to_move as usize;
    let opp = st.next_alive(me);
    let a = approx_fill(st, me, sc);
    let b = approx_fill(st, opp, sc);
    if (a - b).abs() > 10 {
        return (if a > b { me } else { opp }, true);
    }
    let a = primed_fill(st, me, sc).1;
    let b = primed_fill(st, opp, sc).1;
    (if a > b { me } else { opp }, false)
}
// paste:skip-end

/// A turn's deadline, with a margin learned as the turn goes: `ok` stays true
/// while two more steps as long as the slowest step so far would still fit.
/// (A debug build is tens of times slower than a release build; this keeps
/// both inside the limit.)
pub struct Clock {
    deadline: Instant,
    last: Instant,
    worst: Duration,
}

impl Clock {
    pub fn new(deadline: Instant) -> Clock {
        Clock { deadline, last: Instant::now(), worst: Duration::from_secs(0) }
    }

    /// Call before each step of work.
    pub fn ok(&mut self) -> bool {
        let now = Instant::now();
        self.worst = self.worst.max(now.duration_since(self.last));
        self.last = now;
        now + self.worst * 2 < self.deadline
    }
}

/// `greedy_len` under a clock: `None` if time ran out before the fill ended.
fn greedy_len_until(st: &State, p: usize, first: u8, sc: &mut Scratch, clock: &mut Clock) -> Option<i32> {
    let mut s = *st;
    s.step(p, first as usize);
    let mut n = 1;
    let mut prev = first;
    loop {
        if !clock.ok() {
            return None;
        }
        match fill_direction(&mut s, p, prev, sc) {
            Some(d) => {
                s.step(p, d as usize);
                prev = d;
                n += 1;
            }
            None => return Some(n),
        }
    }
}

/// The primed greedy fill under a clock: try each first step, fill greedily
/// to the end, keep the longest. The plain greedy step is tried first and is
/// the answer if no fill can be finished in time; a first step whose fill
/// was cut short is not considered. `NO_MOVE` if `p` is stuck.
pub fn primed_fill_until(st: &State, p: usize, sc: &mut Scratch, clock: &mut Clock) -> u8 {
    let mut s = *st;
    let greedy = match fill_direction(&mut s, p, st.last[p], sc) {
        Some(d) => d,
        None => return NO_MOVE,
    };
    let legal = st.legal_mask(p);
    let mut best = greedy;
    let mut best_score = i32::MIN;
    for i in 0..5u8 {
        let d = if i == 0 { greedy } else { i - 1 };
        if (i > 0 && d == greedy) || legal >> d & 1 == 0 {
            continue;
        }
        let len = match greedy_len_until(st, p, d, sc, clock) {
            Some(len) => len,
            None => break,
        };
        let (x, y) = (st.hx[p] as i32 + DIR_X[d as usize], st.hy[p] as i32 + DIR_Y[d as usize]);
        let score = len * 50 + wall_neighbor_count(st, x, y);
        if score > best_score {
            best_score = score;
            best = d;
        }
    }
    best
}

/// Search value in [-1, 1] for the mover of a position where the last two
/// bikes are cut off, from the cheap estimates.
pub fn cutoff_value(st: &State, sc: &mut Scratch) -> f32 {
    let me = st.to_move as usize;
    let a = approx_fill(st, me, sc);
    let b = approx_fill(st, st.next_alive(me), sc);
    (((a - b) as f32 - 0.5) / 3.0).tanh()
}
