//! CodinGame Tron Battle agent.
//!
//! Paste `src/main.rs` into the CodinGame Rust IDE.
//! Local: `cargo run --release -- --bench`  (self-play)
//!        `cargo run --release -- --profile`
#![allow(dead_code)]
//!
//! 1v1: iterative-deepening alpha-beta with Voronoi + edge territory,
//!      chamber-aware fill when players separate.
//! FFA: our moves searched; others modeled as greedy space-takers.
//!
//! Paste this file into the CodinGame IDE (Rust). Local `--bench` runs self-play.

use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

static SEP_N: AtomicU32 = AtomicU32::new(0);
static MOVE_N: AtomicU32 = AtomicU32::new(0);
static EARLY_SEP: AtomicU32 = AtomicU32::new(0);

const W: i32 = 30;
const H: i32 = 20;
const SIZE: usize = 600;
const MAX_P: usize = 4;

const DX: [i32; 4] = [0, 0, -1, 1];
const DY: [i32; 4] = [-1, 1, 0, 0];
const DIR_NAME: [&str; 4] = ["UP", "DOWN", "LEFT", "RIGHT"];

const INF: u16 = 0x7FFF;
const MATE: i32 = 1_000_000;
const TIMEOUT_SCORE: i32 = i32::MIN / 4;

const TURN_BUDGET_MS: u64 = 75;
const FIRST_TURN_BUDGET_MS: u64 = 85;

#[inline]
fn in_b(x: i32, y: i32) -> bool {
    x >= 0 && x < W && y >= 0 && y < H
}
#[inline]
fn idx(x: i32, y: i32) -> usize {
    (y * W + x) as usize
}
#[inline]
fn xy(i: usize) -> (i32, i32) {
    ((i as i32) % W, (i as i32) / W)
}

#[derive(Clone, Copy)]
struct Rows {
    r: [u32; 20],
}

impl Rows {
    #[inline]
    fn empty() -> Self {
        Self { r: [0; 20] }
    }
    #[inline]
    fn get(self, x: i32, y: i32) -> bool {
        self.r[y as usize] & (1u32 << x) != 0
    }
    #[inline]
    fn set(&mut self, x: i32, y: i32) {
        self.r[y as usize] |= 1u32 << x;
    }
    #[inline]
    fn clear(&mut self, x: i32, y: i32) {
        self.r[y as usize] &= !(1u32 << x);
    }
    #[inline]
    fn xor_with(&mut self, o: Rows) {
        for i in 0..20 {
            self.r[i] ^= o.r[i];
        }
    }
}

#[derive(Clone, Copy)]
struct State {
    occ: Rows,
    trail: [Rows; MAX_P],
    hx: [i8; MAX_P],
    hy: [i8; MAX_P],
    alive: u8,
    n: u8,
}

impl State {
    fn new(n: u8) -> Self {
        Self {
            occ: Rows::empty(),
            trail: [Rows::empty(); MAX_P],
            hx: [-1; MAX_P],
            hy: [-1; MAX_P],
            alive: 0,
            n,
        }
    }

    #[inline]
    fn is_alive(self, p: usize) -> bool {
        self.alive & (1 << p) != 0
    }

    fn occupy(&mut self, p: usize, x: i32, y: i32) {
        self.occ.set(x, y);
        self.trail[p].set(x, y);
        self.hx[p] = x as i8;
        self.hy[p] = y as i8;
        self.alive |= 1 << p;
    }

    fn kill(&mut self, p: usize) {
        if !self.is_alive(p) {
            return;
        }
        self.occ.xor_with(self.trail[p]);
        self.trail[p] = Rows::empty();
        self.alive &= !(1 << p);
        self.hx[p] = -1;
        self.hy[p] = -1;
    }

    #[inline]
    fn legal_dir(self, p: usize, d: usize) -> bool {
        let x = self.hx[p] as i32 + DX[d];
        let y = self.hy[p] as i32 + DY[d];
        in_b(x, y) && !self.occ.get(x, y)
    }

    fn legal_list(self, p: usize) -> ([u8; 4], usize) {
        let mut m = [0u8; 4];
        let mut n = 0;
        for d in 0..4 {
            if self.legal_dir(p, d) {
                m[n] = d as u8;
                n += 1;
            }
        }
        (m, n)
    }

    fn apply(&mut self, p: usize, d: usize) -> bool {
        let x = self.hx[p] as i32 + DX[d];
        let y = self.hy[p] as i32 + DY[d];
        if !in_b(x, y) || self.occ.get(x, y) {
            return false;
        }
        self.occupy(p, x, y);
        true
    }

    fn undo_step(&mut self, p: usize, old_x: i8, old_y: i8) {
        let x = self.hx[p] as i32;
        let y = self.hy[p] as i32;
        self.occ.clear(x, y);
        self.trail[p].clear(x, y);
        self.hx[p] = old_x;
        self.hy[p] = old_y;
    }
}

struct Bufs {
    dist: [[u16; SIZE]; MAX_P],
    q: [u16; SIZE],
    vis: [u32; SIZE],
    stamp: u32,
    owner: [i8; SIZE],
}

impl Bufs {
    fn new() -> Self {
        Self {
            dist: [[INF; SIZE]; MAX_P],
            q: [0; SIZE],
            vis: [0; SIZE],
            stamp: 1,
            owner: [-1; SIZE],
        }
    }

    fn tick(&mut self) -> u32 {
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.vis.fill(0);
            self.stamp = 1;
        }
        self.stamp
    }
}

fn bfs_player(s: &State, p: usize, dist: &mut [u16; SIZE], q: &mut [u16; SIZE]) {
    dist.fill(INF);
    if !s.is_alive(p) {
        return;
    }
    let mut qh = 0usize;
    let mut qt = 0usize;
    let hx = s.hx[p] as i32;
    let hy = s.hy[p] as i32;
    // Heads occupy cells; search starts from empty neighbours.
    for d in 0..4 {
        let x = hx + DX[d];
        let y = hy + DY[d];
        if in_b(x, y) && !s.occ.get(x, y) {
            let i = idx(x, y);
            dist[i] = 1;
            q[qt] = i as u16;
            qt += 1;
        }
    }
    while qh < qt {
        let i = q[qh] as usize;
        qh += 1;
        let (x, y) = xy(i);
        let nd = dist[i] + 1;
        for d in 0..4 {
            let nx = x + DX[d];
            let ny = y + DY[d];
            if in_b(nx, ny) && !s.occ.get(nx, ny) {
                let j = idx(nx, ny);
                if dist[j] == INF {
                    dist[j] = nd;
                    q[qt] = j as u16;
                    qt += 1;
                }
            }
        }
    }
}

fn flood_count(s: &State, p: usize, b: &mut Bufs) -> i32 {
    if !s.is_alive(p) {
        return 0;
    }
    let st = b.tick();
    let mut qh = 0usize;
    let mut qt = 0usize;
    let hx = s.hx[p] as i32;
    let hy = s.hy[p] as i32;
    for d in 0..4 {
        let x = hx + DX[d];
        let y = hy + DY[d];
        if in_b(x, y) && !s.occ.get(x, y) {
            let i = idx(x, y);
            if b.vis[i] != st {
                b.vis[i] = st;
                b.q[qt] = i as u16;
                qt += 1;
            }
        }
    }
    while qh < qt {
        let i = b.q[qh] as usize;
        qh += 1;
        let (x, y) = xy(i);
        for d in 0..4 {
            let nx = x + DX[d];
            let ny = y + DY[d];
            if in_b(nx, ny) && !s.occ.get(nx, ny) {
                let j = idx(nx, ny);
                if b.vis[j] != st {
                    b.vis[j] = st;
                    b.q[qt] = j as u16;
                    qt += 1;
                }
            }
        }
    }
    qt as i32
}

fn empty_deg(s: &State, x: i32, y: i32) -> i32 {
    let mut c = 0;
    for d in 0..4 {
        let nx = x + DX[d];
        let ny = y + DY[d];
        if in_b(nx, ny) && !s.occ.get(nx, ny) {
            c += 1;
        }
    }
    c
}

fn wall_neighbors(s: &State, x: i32, y: i32) -> i32 {
    4 - empty_deg(s, x, y)
}

/// Local 4-neighbour cut test via the 8-ring around (x, y).
fn is_local_cut(s: &State, x: i32, y: i32) -> bool {
    let mut pts = [(0i32, 0i32); 4];
    let mut n = 0;
    for d in 0..4 {
        let nx = x + DX[d];
        let ny = y + DY[d];
        if in_b(nx, ny) && !s.occ.get(nx, ny) {
            pts[n] = (nx, ny);
            n += 1;
        }
    }
    if n <= 1 {
        return false;
    }
    if n >= 4 {
        return false;
    }
    // BFS on empty 8-ring with 4-connectivity, see if all pts connect.
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
    for i in 0..8 {
        let rx = x + RING[i].0;
        let ry = y + RING[i].1;
        ring_empty[i] = in_b(rx, ry) && !s.occ.get(rx, ry);
    }
    // Map each 4-neighbour to a ring index: UP=1, RIGHT=3, DOWN=5, LEFT=7
    let mut start = -1i32;
    for i in 0..n {
        let (px, py) = pts[i];
        let k = if px == x && py == y - 1 {
            1
        } else if px == x + 1 && py == y {
            3
        } else if px == x && py == y + 1 {
            5
        } else {
            7
        };
        if ring_empty[k] {
            start = k as i32;
            break;
        }
    }
    if start < 0 {
        return true;
    }
    let mut seen = [false; 8];
    let mut stack = [0i32; 8];
    let mut sp = 1;
    stack[0] = start;
    seen[start as usize] = true;
    while sp > 0 {
        sp -= 1;
        let u = stack[sp] as usize;
        for delta in [-1i32, 1] {
            let v = ((u as i32 + delta + 8) % 8) as usize;
            // 4-connectivity on the ring: adjacent ring cells that share an edge.
            // Consecutive ring indices are 8-adjacent; skip diagonal-only pairs
            // when either is a corner and they don't 4-connect.
            let (ax, ay) = (x + RING[u].0, y + RING[u].1);
            let (bx, by) = (x + RING[v].0, y + RING[v].1);
            if (ax - bx).abs() + (ay - by).abs() != 1 {
                continue;
            }
            if ring_empty[v] && !seen[v] {
                seen[v] = true;
                stack[sp] = v as i32;
                sp += 1;
            }
        }
    }
    for i in 0..n {
        let (px, py) = pts[i];
        let k = if px == x && py == y - 1 {
            1
        } else if px == x + 1 && py == y {
            3
        } else if px == x && py == y + 1 {
            5
        } else {
            7
        };
        if !seen[k] {
            return true;
        }
    }
    false
}

fn dfs_fill(s: &State, x: i32, y: i32, b: &mut Bufs, st: u32) -> i32 {
    let i = idx(x, y);
    b.vis[i] = st;
    let mut nbs = [(0i32, 0i32); 4];
    let mut n = 0;
    for d in 0..4 {
        let nx = x + DX[d];
        let ny = y + DY[d];
        if in_b(nx, ny) && !s.occ.get(nx, ny) && b.vis[idx(nx, ny)] != st {
            nbs[n] = (nx, ny);
            n += 1;
        }
    }
    if n == 0 {
        return 1;
    }
    if n == 1 {
        return 1 + dfs_fill(s, nbs[0].0, nbs[0].1, b, st);
    }
    if is_local_cut(s, x, y) {
        let mut best = 0;
        for k in 0..n {
            if b.vis[idx(nbs[k].0, nbs[k].1)] != st {
                let v = dfs_fill(s, nbs[k].0, nbs[k].1, b, st);
                if v > best {
                    best = v;
                }
            }
        }
        1 + best
    } else {
        let mut sum = 1;
        for k in 0..n {
            if b.vis[idx(nbs[k].0, nbs[k].1)] != st {
                sum += dfs_fill(s, nbs[k].0, nbs[k].1, b, st);
            }
        }
        sum
    }
}

fn approx_fill(s: &State, p: usize, b: &mut Bufs) -> i32 {
    if !s.is_alive(p) {
        return 0;
    }
    let st = b.tick();
    let hx = s.hx[p] as i32;
    let hy = s.hy[p] as i32;
    let mut best = 0;
    for d in 0..4 {
        let x = hx + DX[d];
        let y = hy + DY[d];
        if in_b(x, y) && !s.occ.get(x, y) && b.vis[idx(x, y)] != st {
            let v = dfs_fill(s, x, y, b, st);
            if v > best {
                best = v;
            }
        }
    }
    best
}

struct Voronoi {
    terr: [i32; MAX_P],
    edges: [i32; MAX_P],
    reach: [i32; MAX_P],
    connected: bool,
}

fn voronoi(s: &State, b: &mut Bufs) -> Voronoi {
    let n = s.n as usize;
    for p in 0..n {
        if s.is_alive(p) {
            bfs_player(s, p, &mut b.dist[p], &mut b.q);
        } else {
            b.dist[p].fill(INF);
        }
    }
    let mut terr = [0i32; MAX_P];
    let mut edges = [0i32; MAX_P];
    let mut reach = [0i32; MAX_P];
    let mut connected = false;
    b.owner.fill(-1);

    for y in 0..H {
        for x in 0..W {
            if s.occ.get(x, y) {
                continue;
            }
            let i = idx(x, y);
            let mut best = INF;
            let mut who = -1i8;
            let mut ties = 0;
            for p in 0..n {
                if !s.is_alive(p) {
                    continue;
                }
                let d = b.dist[p][i];
                if d < INF {
                    reach[p] += 1;
                }
                if d < best {
                    best = d;
                    who = p as i8;
                    ties = 1;
                } else if d == best && d < INF {
                    ties += 1;
                }
            }
            if best == INF {
                continue;
            }
            // Two live players both reaching this cell => still interacting.
            let mut reachers = 0;
            for p in 0..n {
                if s.is_alive(p) && b.dist[p][i] < INF {
                    reachers += 1;
                }
            }
            if reachers >= 2 {
                connected = true;
            }
            if ties == 1 {
                let p = who as usize;
                terr[p] += 1;
                b.owner[i] = who;
                edges[p] += empty_deg(s, x, y);
            }
        }
    }
    Voronoi {
        terr,
        edges,
        reach,
        connected,
    }
}

fn mobility(s: &State, p: usize) -> i32 {
    let mut c = 0;
    for d in 0..4 {
        if s.legal_dir(p, d) {
            c += 1;
        }
    }
    c
}

fn main_opponent(s: &State, me: usize, b: &mut Bufs) -> Option<usize> {
    let n = s.n as usize;
    let mut best = None;
    let mut best_d = INF;
    bfs_player(s, me, &mut b.dist[me], &mut b.q);
    for p in 0..n {
        if p == me || !s.is_alive(p) {
            continue;
        }
        // Distance = min dist of our map to a neighbour of their head.
        let hx = s.hx[p] as i32;
        let hy = s.hy[p] as i32;
        let mut d = INF;
        for dir in 0..4 {
            let x = hx + DX[dir];
            let y = hy + DY[dir];
            if in_b(x, y) {
                if s.occ.get(x, y) && x == s.hx[me] as i32 && y == s.hy[me] as i32 {
                    d = 0;
                } else if !s.occ.get(x, y) {
                    d = d.min(b.dist[me][idx(x, y)]);
                }
            }
        }
        // Prefer opponents we still share space with; else nearest manhattan.
        if d < best_d {
            best_d = d;
            best = Some(p);
        } else if d == INF && best_d == INF {
            let md = (s.hx[me] as i32 - hx).abs() + (s.hy[me] as i32 - hy).abs();
            if best.is_none() || md < (s.hx[me] as i32 - s.hx[best.unwrap()] as i32).abs() {
                best = Some(p);
            }
        }
    }
    best
}

fn shares_space(s: &State, me: usize, opp: usize, b: &mut Bufs) -> bool {
    if !s.is_alive(me) || !s.is_alive(opp) {
        return false;
    }
    let hx = s.hx[me] as i32;
    let hy = s.hy[me] as i32;
    let ox = s.hx[opp] as i32;
    let oy = s.hy[opp] as i32;
    if (hx - ox).abs() + (hy - oy).abs() == 1 {
        return true;
    }
    bfs_player(s, me, &mut b.dist[me], &mut b.q);
    for d in 0..4 {
        let x = ox + DX[d];
        let y = oy + DY[d];
        if in_b(x, y) && !s.occ.get(x, y) && b.dist[me][idx(x, y)] < INF {
            return true;
        }
    }
    false
}

fn battlefront(s: &State, me: usize, opp: usize, b: &Bufs) -> i32 {
    let mut front = 0;
    for y in 0..H {
        for x in 0..W {
            if s.occ.get(x, y) {
                continue;
            }
            let i = idx(x, y);
            if b.owner[i] != me as i8 {
                continue;
            }
            for d in 0..4 {
                let nx = x + DX[d];
                let ny = y + DY[d];
                if in_b(nx, ny) && !s.occ.get(nx, ny) && b.owner[idx(nx, ny)] == opp as i8 {
                    front += 1;
                    break;
                }
            }
        }
    }
    front
}

fn eval_1v1(s: &State, me: usize, opp: usize, ply: i32, b: &mut Bufs) -> i32 {
    if !s.is_alive(me) {
        return -MATE + ply;
    }
    if !s.is_alive(opp) {
        return MATE - ply;
    }
    let my_m = mobility(s, me);
    let op_m = mobility(s, opp);
    if !shares_space(s, me, opp, b) {
        let my_f = approx_fill(s, me, b);
        let op_f = approx_fill(s, opp, b);
        let diff = my_f - op_f;
        let sign = diff.signum();
        let hug = wall_neighbors(s, s.hx[me] as i32, s.hy[me] as i32)
            - wall_neighbors(s, s.hx[opp] as i32, s.hy[opp] as i32);
        return sign * 80 + diff * 60 + hug * 5 + (my_m - op_m);
    }
    let v = voronoi(s, b);
    let terr = v.terr[me] - v.terr[opp];
    let ed = v.edges[me] - v.edges[opp];
    let rh = v.reach[me] - v.reach[opp];
    let mob = my_m - op_m;
    let front = battlefront(s, me, opp, b);
    let hx = s.hx[me] as i32;
    let hy = s.hy[me] as i32;
    let center = -((hx - 14).abs() + (hy - 9).abs());
    let occ_est = (0..20).map(|y| s.occ.r[y].count_ones()).sum::<u32>() as i32;
    let center_w = (500 - occ_est).max(0) / 80;
    terr * 50 + ed * 12 + rh * 3 + mob * 6 + front * 4 + center * center_w
}

fn eval_ffa(s: &State, me: usize, ply: i32, b: &mut Bufs) -> i32 {
    if !s.is_alive(me) {
        return -MATE + ply;
    }
    let alive_n = s.alive.count_ones();
    if alive_n == 1 {
        return MATE - ply;
    }
    let v = voronoi(s, b);
    let mut best_other_t = 0;
    let mut best_other_r = 0;
    let n = s.n as usize;
    for p in 0..n {
        if p == me || !s.is_alive(p) {
            continue;
        }
        best_other_t = best_other_t.max(v.terr[p]);
        best_other_r = best_other_r.max(v.reach[p]);
    }
    let my_f = v.reach[me];
    let mob = mobility(s, me);
    my_f * 40 + v.terr[me] * 25 - best_other_t * 10 - best_other_r * 4 + mob * 20
}

struct Search {
    deadline: Instant,
    nodes: u64,
    timed_out: bool,
    killers: [[u8; 2]; 64],
}

impl Search {
    fn new(budget: Duration) -> Self {
        Self {
            deadline: Instant::now() + budget,
            nodes: 0,
            timed_out: false,
            killers: [[4; 2]; 64],
        }
    }
    #[inline]
    fn check(&mut self) {
        self.nodes += 1;
        if (self.nodes & 15) == 0 && Instant::now() >= self.deadline {
            self.timed_out = true;
        }
    }
}

fn order_moves(
    s: &State,
    p: usize,
    moves: &mut [u8],
    n: usize,
    pv: u8,
    killer: u8,
    last: u8,
    tx: i32,
    ty: i32,
) {
    let hx = s.hx[p] as i32;
    let hy = s.hy[p] as i32;
    let mut key = [0i32; 4];
    for i in 0..n {
        let d = moves[i];
        let x = hx + DX[d as usize];
        let y = hy + DY[d as usize];
        let mut k = 0;
        if d == pv {
            k += 10_000;
        }
        if d == killer {
            k += 3_000;
        }
        if d == last {
            k += 40;
        }
        k += wall_neighbors(s, x, y) * 8;
        k -= (x - tx).abs() + (y - ty).abs();
        key[i] = k;
    }
    for i in 1..n {
        let mut j = i;
        while j > 0 && key[j] > key[j - 1] {
            key.swap(j, j - 1);
            moves.swap(j, j - 1);
            j -= 1;
        }
    }
}

fn greedy_dir(s: &State, p: usize, b: &mut Bufs) -> Option<u8> {
    let (mv, n) = s.legal_list(p);
    if n == 0 {
        return None;
    }
    let mut best = mv[0];
    let mut best_sc = i32::MIN;
    for i in 0..n {
        let d = mv[i] as usize;
        let mut t = *s;
        t.apply(p, d);
        let flood = flood_count(&t, p, b);
        let x = t.hx[p] as i32;
        let y = t.hy[p] as i32;
        let sc = flood * 20 + wall_neighbors(&t, x, y) * 3;
        if sc > best_sc {
            best_sc = sc;
            best = mv[i];
        }
    }
    Some(best)
}

fn negamax_1v1(
    s: &mut State,
    me: usize,
    opp: usize,
    to_move: usize,
    depth: i32,
    ply: i32,
    mut alpha: i32,
    beta: i32,
    last_me: u8,
    last_opp: u8,
    se: &mut Search,
    b: &mut Bufs,
) -> i32 {
    se.check();
    if se.timed_out {
        return TIMEOUT_SCORE;
    }
    let other = if to_move == me { opp } else { me };
    if !s.is_alive(me) {
        return -MATE + ply;
    }
    if !s.is_alive(opp) {
        return MATE - ply;
    }
    if depth <= 0 {
        let n = s.legal_list(to_move).1;
        if n == 0 {
            return if to_move == me {
                -MATE + ply
            } else {
                MATE - ply
            };
        }
        let sc = eval_1v1(s, me, opp, ply, b);
        return if to_move == me { sc } else { -sc };
    }
    let (mut moves, n) = s.legal_list(to_move);
    if n == 0 {
        // Player to move dies.
        if to_move == me {
            return -MATE + ply;
        } else {
            return MATE - ply;
        }
    }
    let (tx, ty) = (s.hx[other] as i32, s.hy[other] as i32);
    let last = if to_move == me { last_me } else { last_opp };
    let killer = se.killers[ply as usize % 64][0];
    order_moves(s, to_move, &mut moves, n, 4, killer, last, tx, ty);

    let mut best = -MATE * 2;
    for i in 0..n {
        let d = moves[i] as usize;
        let ox = s.hx[to_move];
        let oy = s.hy[to_move];
        s.apply(to_move, d);
        let (nm, no) = if to_move == me {
            (moves[i], last_opp)
        } else {
            (last_me, moves[i])
        };
        let sc = -negamax_1v1(
            s,
            me,
            opp,
            other,
            depth - 1,
            ply + 1,
            -beta,
            -alpha,
            nm,
            no,
            se,
            b,
        );
        s.undo_step(to_move, ox, oy);
        if se.timed_out {
            return TIMEOUT_SCORE;
        }
        if sc > best {
            best = sc;
        }
        if sc > alpha {
            alpha = sc;
        }
        if alpha >= beta {
            let slot = ply as usize % 64;
            if se.killers[slot][0] != moves[i] {
                se.killers[slot][1] = se.killers[slot][0];
                se.killers[slot][0] = moves[i];
            }
            break;
        }
    }
    best
}

fn endgame_eval(s: &State, me: usize, b: &mut Bufs) -> i32 {
    let fill = approx_fill(s, me, b);
    let flood = flood_count(s, me, b);
    let hug = wall_neighbors(s, s.hx[me] as i32, s.hy[me] as i32);
    fill * 80 + flood * 20 + hug * 6 + mobility(s, me)
}

fn search_endgame(
    s: &mut State,
    me: usize,
    depth: i32,
    ply: i32,
    last: u8,
    se: &mut Search,
    b: &mut Bufs,
) -> i32 {
    se.check();
    if se.timed_out {
        return TIMEOUT_SCORE;
    }
    let (mut moves, n) = s.legal_list(me);
    if n == 0 {
        return ply;
    }
    if depth <= 0 {
        return endgame_eval(s, me, b) + ply;
    }
    order_moves(
        s,
        me,
        &mut moves,
        n,
        4,
        se.killers[ply as usize % 64][0],
        last,
        s.hx[me] as i32,
        s.hy[me] as i32,
    );
    let mut best = i32::MIN / 2;
    for i in 0..n {
        let ox = s.hx[me];
        let oy = s.hy[me];
        s.apply(me, moves[i] as usize);
        let sc = search_endgame(s, me, depth - 1, ply + 1, moves[i], se, b);
        s.undo_step(me, ox, oy);
        if se.timed_out {
            return TIMEOUT_SCORE;
        }
        if sc > best {
            best = sc;
        }
    }
    best
}

fn search_ffa(
    s: &State,
    me: usize,
    depth: i32,
    ply: i32,
    se: &mut Search,
    b: &mut Bufs,
) -> i32 {
    se.check();
    if se.timed_out {
        return TIMEOUT_SCORE;
    }
    if !s.is_alive(me) {
        return -MATE + ply;
    }
    if s.alive.count_ones() == 1 {
        return MATE - ply;
    }
    if depth <= 0 {
        return eval_ffa(s, me, ply, b);
    }
    let (moves, n) = s.legal_list(me);
    if n == 0 {
        return -MATE + ply;
    }
    let mut best = -MATE * 2;
    for i in 0..n {
        let mut t = *s;
        t.apply(me, moves[i] as usize);
        // Greedy replies; copy avoids death-undo pain.
        let npl = t.n as usize;
        let mut p = (me + 1) % npl;
        while p != me {
            if t.is_alive(p) {
                if let Some(d) = greedy_dir(&t, p, b) {
                    t.apply(p, d as usize);
                } else {
                    t.kill(p);
                }
            }
            p = (p + 1) % npl;
        }
        let sc = search_ffa(&t, me, depth - 1, ply + 1, se, b);
        if se.timed_out {
            return TIMEOUT_SCORE;
        }
        if sc > best {
            best = sc;
        }
    }
    best
}

fn oneply_dir(s: &mut State, p: usize, b: &mut Bufs) -> Option<u8> {
    let (mv, n) = s.legal_list(p);
    if n == 0 {
        return None;
    }
    if n == 1 {
        return Some(mv[0]);
    }
    let npl = s.n as usize;
    let others: usize = (0..npl).filter(|&o| o != p && s.is_alive(o)).count();
    let opp = (0..npl).find(|&o| o != p && s.is_alive(o)).unwrap_or(p);
    let mut best = mv[0];
    let mut best_sc = i32::MIN;
    for i in 0..n {
        let ox = s.hx[p];
        let oy = s.hy[p];
        s.apply(p, mv[i] as usize);
        let sc = if others <= 1 {
            if s.is_alive(opp) && s.legal_list(opp).1 == 0 {
                MATE
            } else {
                eval_1v1(s, p, opp, 1, b)
            }
        } else {
            eval_ffa(s, p, 1, b)
        };
        s.undo_step(p, ox, oy);
        if sc > best_sc {
            best_sc = sc;
            best = mv[i];
        }
    }
    Some(best)
}

fn fill_dir(s: &mut State, p: usize, last: u8, b: &mut Bufs) -> Option<u8> {
    let (mv, n) = s.legal_list(p);
    if n == 0 {
        return None;
    }
    let base = flood_count(s, p, b);
    let mut best = mv[0];
    let mut best_sc = i32::MIN;
    for i in 0..n {
        let ox = s.hx[p];
        let oy = s.hy[p];
        s.apply(p, mv[i] as usize);
        let remain = flood_count(s, p, b);
        let fill = approx_fill(s, p, b);
        let x = s.hx[p] as i32;
        let y = s.hy[p] as i32;
        let mut sc = fill * 80 + remain * 30 + wall_neighbors(s, x, y) * 12;
        if mv[i] == last {
            sc += 8;
        }
        if remain + 1 < base {
            sc -= (base - remain) * 400;
        }
        s.undo_step(p, ox, oy);
        if sc > best_sc {
            best_sc = sc;
            best = mv[i];
        }
    }
    Some(best)
}

fn rollout_score(
    s: &State,
    me: usize,
    first: u8,
    steps: i32,
    b: &mut Bufs,
) -> i32 {
    let mut t = *s;
    if !t.apply(me, first as usize) {
        return -MATE;
    }
    let npl = t.n as usize;
    let mut p = (me + 1) % npl;
    let mut made = 0i32;
    while made < steps && t.alive.count_ones() > 1 && t.is_alive(me) {
        if t.is_alive(p) {
            let mv = oneply_dir(&mut t, p, b);
            match mv {
                Some(d) if t.legal_dir(p, d as usize) => {
                    t.apply(p, d as usize);
                }
                _ => t.kill(p),
            }
            made += 1;
        }
        p = (p + 1) % npl;
    }
    if !t.is_alive(me) {
        return -MATE + made;
    }
    if t.alive.count_ones() == 1 {
        return MATE - made;
    }
    let others: Vec<usize> = (0..npl).filter(|&o| o != me && t.is_alive(o)).collect();
    if others.len() == 1 {
        eval_1v1(&t, me, others[0], made, b)
    } else {
        eval_ffa(&t, me, made, b)
    }
}

fn choose_move(
    s: &mut State,
    me: usize,
    last: u8,
    budget_ms: u64,
    b: &mut Bufs,
) -> u8 {
    let (mut moves, n) = s.legal_list(me);
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return moves[0];
    }
    let start = Instant::now();

    let alive_others: Vec<usize> = (0..s.n as usize)
        .filter(|&p| p != me && s.is_alive(p))
        .collect();
    if alive_others.is_empty() {
        return fill_dir(s, me, last, b).unwrap_or(moves[0]);
    }
    let duel = alive_others.len() == 1;
    let opp = if duel {
        alive_others[0]
    } else {
        main_opponent(s, me, b).unwrap_or(alive_others[0])
    };

    let occ_n: u32 = (0..20).map(|y| s.occ.r[y].count_ones()).sum();
    let connected = !duel || shares_space(s, me, opp, b) || voronoi(s, b).connected;
    let separated = duel && !connected;
    MOVE_N.fetch_add(1, Ordering::Relaxed);
    if separated {
        SEP_N.fetch_add(1, Ordering::Relaxed);
        if occ_n < 24 {
            EARLY_SEP.fetch_add(1, Ordering::Relaxed);
        }
    }

    let (tx, ty) = (s.hx[opp] as i32, s.hy[opp] as i32);
    order_moves(s, me, &mut moves, n, 4, 4, last, tx, ty);

    if separated {
        let mut best = moves[0];
        let mut best_sc = i32::MIN;
        for i in 0..n {
            let ox = s.hx[me];
            let oy = s.hy[me];
            s.apply(me, moves[i] as usize);
            let mut t = *s;
            let mut extra = 0i32;
            let mut prev = moves[i];
            while extra < 80 {
                match fill_dir(&mut t, me, prev, b) {
                    Some(d) if t.legal_dir(me, d as usize) => {
                        t.apply(me, d as usize);
                        prev = d;
                        extra += 1;
                    }
                    _ => break,
                }
            }
            let hug = wall_neighbors(s, s.hx[me] as i32, s.hy[me] as i32);
            let sc = extra * 50 + hug;
            s.undo_step(me, ox, oy);
            if sc > best_sc {
                best_sc = sc;
                best = moves[i];
            }
        }
        return best;
    }

    // 2-ply against greedy Voronoi (the standard strong baseline).
    let mut best = moves[0];
    let mut best_sc = i32::MIN;
    for i in 0..n {
        let ox = s.hx[me];
        let oy = s.hy[me];
        s.apply(me, moves[i] as usize);
        let mut sc = if duel {
            if s.legal_list(opp).1 == 0 {
                MATE - 1
            } else if let Some(od) = oneply_dir(s, opp, b) {
                let ox2 = s.hx[opp];
                let oy2 = s.hy[opp];
                if s.apply(opp, od as usize) {
                    let v = eval_1v1(s, me, opp, 2, b);
                    s.undo_step(opp, ox2, oy2);
                    v
                } else {
                    MATE - 1
                }
            } else {
                MATE - 1
            }
        } else {
            let mut t = *s;
            let npl = t.n as usize;
            let mut p = (me + 1) % npl;
            while p != me {
                if t.is_alive(p) {
                    if let Some(d) = oneply_dir(&mut t, p, b) {
                        t.apply(p, d as usize);
                    } else {
                        t.kill(p);
                    }
                }
                p = (p + 1) % npl;
            }
            eval_ffa(&t, me, 1, b) + flood_count(&t, me, b) * 20
        };
        s.undo_step(me, ox, oy);
        if moves[i] == last {
            sc += 4;
        }
        if sc > best_sc {
            best_sc = sc;
            best = moves[i];
        }
    }

    // 1v1-style search vs the most relevant opponent; others stay as walls.
    let mut se = Search::new(Duration::from_millis(budget_ms.max(1)).saturating_sub(start.elapsed()));
    let mut pv = best;
    let max_d = if s.alive.count_ones() > 2 { 0 } else { 16 };
    for depth in 1..=max_d {
        if Instant::now() >= se.deadline {
            break;
        }
        let mut iter_best = pv;
        let mut iter_sc = -MATE * 2;
        let mut complete = true;
        order_moves(s, me, &mut moves, n, pv, se.killers[0][0], last, tx, ty);
        for i in 0..n {
            let ox = s.hx[me];
            let oy = s.hy[me];
            s.apply(me, moves[i] as usize);
            let deep = -negamax_1v1(
                s,
                me,
                opp,
                opp,
                depth - 1,
                1,
                -MATE * 2,
                -iter_sc,
                moves[i],
                4,
                &mut se,
                b,
            );
            s.undo_step(me, ox, oy);
            if se.timed_out {
                complete = false;
                break;
            }
            if deep > iter_sc {
                iter_sc = deep;
                iter_best = moves[i];
            }
        }
        if complete {
            pv = iter_best;
            best = iter_best;
            if iter_sc.abs() >= MATE - 200 {
                break;
            }
        } else {
            break;
        }
    }

    if budget_ms >= TURN_BUDGET_MS {
        eprintln!(
            "{} mm {} {}ms",
            DIR_NAME[best as usize],
            best_sc,
            start.elapsed().as_millis()
        );
    }
    best
}

/// Reconstruct occupancy from successive CodinGame inputs.
struct Tracker {
    state: State,
    seen: bool,
}

impl Tracker {
    fn new() -> Self {
        Self {
            state: State::new(2),
            seen: false,
        }
    }

    fn update(&mut self, n: usize, coords: &[(i32, i32, i32, i32)]) {
        if !self.seen {
            self.state = State::new(n as u8);
            for p in 0..n {
                let (x0, y0, x1, y1) = coords[p];
                if x0 < 0 {
                    continue;
                }
                self.state.occupy(p, x0, y0);
                if x1 != x0 || y1 != y0 {
                    self.state.occupy(p, x1, y1);
                }
            }
            self.seen = true;
            return;
        }
        for p in 0..n {
            let (x0, y0, x1, y1) = coords[p];
            if x0 < 0 {
                self.state.kill(p);
                continue;
            }
            if !self.state.is_alive(p) {
                // Should not happen; re-seed.
                self.state.occupy(p, x0, y0);
            }
            let hx = self.state.hx[p] as i32;
            let hy = self.state.hy[p] as i32;
            if x1 != hx || y1 != hy {
                self.state.occupy(p, x1, y1);
            }
        }
    }
}

fn parse_budget_ms() -> Option<u64> {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--budget-ms" {
            return args.next().and_then(|v| v.parse().ok());
        }
        if let Some(v) = a.strip_prefix("--budget-ms=") {
            return v.parse().ok();
        }
    }
    std::env::var("TRON_BUDGET_MS")
        .ok()
        .and_then(|v| v.parse().ok())
}

fn codingame() {
    let budget_override = parse_budget_ms();
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let mut tr = Tracker::new();
    let mut bufs = Bufs::new();
    let mut last = 4u8;
    let mut first = true;
    loop {
        let line = match lines.next() {
            Some(Ok(s)) => s,
            _ => break,
        };
        let mut it = line.split_whitespace();
        let n: usize = it.next().unwrap().parse().unwrap();
        let p: usize = it.next().unwrap().parse().unwrap();
        let mut coords = [(0i32, 0i32, 0i32, 0i32); MAX_P];
        for i in 0..n {
            let line = lines.next().unwrap().unwrap();
            let mut it = line.split_whitespace();
            coords[i] = (
                it.next().unwrap().parse().unwrap(),
                it.next().unwrap().parse().unwrap(),
                it.next().unwrap().parse().unwrap(),
                it.next().unwrap().parse().unwrap(),
            );
        }
        tr.update(n, &coords);
        let budget = budget_override.unwrap_or(if first {
            FIRST_TURN_BUDGET_MS
        } else {
            TURN_BUDGET_MS
        });
        let mv = choose_move(&mut tr.state, p, last, budget, &mut bufs);
        last = mv;
        first = false;
        println!("{}", DIR_NAME[mv as usize]);
        let _ = io::stdout().flush();
    }
}

// ---------------------------------------------------------------------------
// Local self-play (`--bench`) so we can measure the agent before submitting.
// ---------------------------------------------------------------------------

struct XorShift {
    s: u64,
}
impl XorShift {
    fn new(seed: u64) -> Self {
        Self { s: seed | 1 }
    }
    fn next(&mut self) -> u64 {
        let mut x = self.s;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.s = x;
        x
    }
    fn gen_range(&mut self, n: u32) -> u32 {
        (self.next() as u32) % n
    }
}

fn random_start(rng: &mut XorShift, n: usize) -> State {
    let mut s = State::new(n as u8);
    for p in 0..n {
        loop {
            let x = rng.gen_range(W as u32) as i32;
            let y = rng.gen_range(H as u32) as i32;
            if !s.occ.get(x, y) {
                s.occupy(p, x, y);
                break;
            }
        }
    }
    s
}

fn bot_random(s: &State, p: usize, rng: &mut XorShift) -> Option<u8> {
    let (mv, n) = s.legal_list(p);
    if n == 0 {
        None
    } else {
        Some(mv[rng.gen_range(n as u32) as usize])
    }
}

fn bot_greedy(s: &State, p: usize, b: &mut Bufs) -> Option<u8> {
    greedy_dir(s, p, b)
}

fn bot_wallhug(s: &State, p: usize) -> Option<u8> {
    let (mv, n) = s.legal_list(p);
    if n == 0 {
        return None;
    }
    let mut best = mv[0];
    let mut best_w = -1;
    for i in 0..n {
        let x = s.hx[p] as i32 + DX[mv[i] as usize];
        let y = s.hy[p] as i32 + DY[mv[i] as usize];
        let w = wall_neighbors(s, x, y);
        if w > best_w {
            best_w = w;
            best = mv[i];
        }
    }
    Some(best)
}

fn bot_voronoi1(s: &mut State, p: usize, b: &mut Bufs) -> Option<u8> {
    let (mv, n) = s.legal_list(p);
    if n == 0 {
        return None;
    }
    let opp = (0..s.n as usize).find(|&o| o != p && s.is_alive(o)).unwrap_or(p);
    let mut best = mv[0];
    let mut best_sc = i32::MIN;
    for i in 0..n {
        let ox = s.hx[p];
        let oy = s.hy[p];
        s.apply(p, mv[i] as usize);
        let sc = eval_1v1(s, p, opp, 1, b);
        s.undo_step(p, ox, oy);
        if sc > best_sc {
            best_sc = sc;
            best = mv[i];
        }
    }
    Some(best)
}

#[derive(Clone, Copy)]
enum Bot {
    Agent,
    Greedy,
    Wall,
    Voronoi1,
    Random,
}

fn play_game(mut s: State, bots: [Bot; 4], seed: u64) -> usize {
    let mut b = Bufs::new();
    let mut rng = XorShift::new(seed);
    let mut last = [4u8; MAX_P];
    let n = s.n as usize;
    let mut turn = 0u32;
    while s.alive.count_ones() > 1 && turn < 900 {
        for p in 0..n {
            if !s.is_alive(p) {
                continue;
            }
            if s.alive.count_ones() <= 1 {
                break;
            }
            let mv = match bots[p] {
                Bot::Agent => Some(choose_move(&mut s, p, last[p], 25, &mut b)),
                Bot::Greedy => bot_greedy(&s, p, &mut b),
                Bot::Wall => bot_wallhug(&s, p),
                Bot::Voronoi1 => bot_voronoi1(&mut s, p, &mut b),
                Bot::Random => bot_random(&s, p, &mut rng),
            };
            match mv {
                Some(d) if s.legal_dir(p, d as usize) => {
                    s.apply(p, d as usize);
                    last[p] = d;
                }
                _ => s.kill(p),
            }
        }
        turn += 1;
    }
    let alive: Vec<usize> = (0..n).filter(|&p| s.is_alive(p)).collect();
    if alive.len() == 1 {
        return alive[0];
    }
    if alive.is_empty() {
        return 0;
    }
    let mut best_p = alive[0];
    let mut best_f = -1;
    for &p in &alive {
        let f = flood_count(&s, p, &mut b);
        if f > best_f {
            best_f = f;
            best_p = p;
        }
    }
    best_p
}

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
    for (name, opp) in matchups {
        let mut as_first = 0;
        let mut first_games = 0;
        let mut w0 = 0;
        let mut _w1 = 0;
        for g in 0..games {
            let seed = rng.next();
            let s = random_start(&mut XorShift::new(seed), 2);
            // Alternate colours.
            let (b0, b1) = if g % 2 == 0 {
                (Bot::Agent, opp)
            } else {
                (opp, Bot::Agent)
            };
            let mut bots = [Bot::Random; 4];
            bots[0] = b0;
            bots[1] = b1;
            let winner = play_game(s, bots, seed ^ 0x9E3779B97F4A7C15);
            let agent_id = if g % 2 == 0 { 0 } else { 1 };
            if g % 2 == 0 {
                first_games += 1;
                if winner == 0 {
                    as_first += 1;
                }
            }
            if winner == agent_id {
                w0 += 1;
            } else {
                _w1 += 1;
            }
        }
        eprintln!(
            "  vs {name:12}  agent {w0:3}/{games}  ({:.0}%)  as-p0 {as_first}/{first_games}",
            100.0 * w0 as f64 / games as f64
        );
    }
    let sep = SEP_N.load(Ordering::Relaxed);
    let tot = MOVE_N.load(Ordering::Relaxed);
    let early = EARLY_SEP.load(Ordering::Relaxed);
    eprintln!(
        "  separated moves {sep}/{tot} ({:.0}%) early {early}",
        if tot > 0 {
            100.0 * sep as f64 / tot as f64
        } else {
            0.0
        }
    );

    // 4-player survival sample
    let mut wins = 0;
    let ffa_games = 8;
    for g in 0..ffa_games {
        let seed = rng.next();
        let s = random_start(&mut XorShift::new(seed), 4);
        let bots = [Bot::Agent, Bot::Greedy, Bot::Wall, Bot::Voronoi1];
        let winner = play_game(s, bots, seed);
        if winner == 0 {
            wins += 1;
        }
        let _ = g;
    }
    eprintln!(
        "  FFA vs mixed     agent {wins:3}/{ffa_games}  ({:.0}%)",
        100.0 * wins as f64 / ffa_games as f64
    );
}

fn profile() {
    let mut rng = XorShift::new(42);
    let mut s = random_start(&mut rng, 2);
    let mut b = Bufs::new();
    // Play 30 greedy plies to get a midgame-ish board.
    for _ in 0..30 {
        for p in 0..2 {
            if let Some(d) = bot_greedy(&s, p, &mut b) {
                s.apply(p, d as usize);
            }
        }
    }
    let t0 = Instant::now();
    let n_eval = 2000;
    let mut acc = 0i32;
    for _ in 0..n_eval {
        acc ^= eval_1v1(&s, 0, 1, 0, &mut b);
    }
    let ev = t0.elapsed();
    eprintln!(
        "eval: {} in {:?} ({:.1}/ms) checksum {acc}",
        n_eval,
        ev,
        n_eval as f64 / ev.as_secs_f64() / 1000.0
    );
    let t1 = Instant::now();
    let mv = choose_move(&mut s, 0, 4, TURN_BUDGET_MS, &mut b);
    eprintln!(
        "choose_move {} in {:?}",
        DIR_NAME[mv as usize],
        t1.elapsed()
    );
}

fn main() {
    if std::env::args().any(|a| a == "--bench") {
        bench();
    } else if std::env::args().any(|a| a == "--profile") {
        profile();
    } else {
        codingame();
    }
}
