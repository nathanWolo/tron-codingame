//! PUCT search for sequential Tron with two to four players.
//!
//! The settings follow slinky's AlphaZero (`baselines/az.py`): priors and leaf
//! values from the network, exact values at finished positions, unvisited
//! moves valued at the node's mean minus a first-play-urgency reduction, and
//! Dirichlet noise in the root priors during self-play.
//!
//! Turns are sequential, so each node belongs to one player, who picks the
//! move that is best for itself (max^n): every node keeps one value per
//! player and nothing is negated on the way up.
//!
//! Values are placements on the scale of the root: +1 for winning, -1 for
//! being the next bike out among those in the game at the root, evenly spaced
//! between. The network speaks on the scale of the bikes in the game at the
//! position it sees, so a leaf where bikes have died since the root is
//! rescaled (slinky's `rescale_q` does the same for its duel values).
//!
//! Three kinds of position are never left to the network's policy:
//! - the last two bikes cut off from each other: a leaf valued by the fill
//!   estimates;
//! - a mover alone in its region while others still fight: it takes the
//!   greedy space-filling step;
//! - a mover with one legal move: it takes it.
//! The last two are passed through without a network evaluation (up to
//! `MAX_CHAIN` in a row), so a simulation ends at a position with a choice.

use crate::board::{cutoff_value, features, fill_direction, Feat, Scratch, State, MAXP};
use crate::net::{Net, Work};

/// Forced positions one simulation may pass through before it must stop and evaluate.
const MAX_CHAIN: u32 = 6;

#[derive(Clone, Copy)]
pub struct Params {
    /// Exploration constant on values in [-1, 1] (slinky's 0.5 on [0, 1] is 1.0 here).
    pub c_puct: f32,
    pub fpu_reduction: f32,
}

pub struct Node {
    child: [u32; 4],
    prior: [f32; 4],
    n: u32,
    /// Sum of backed-up values, one per player.
    w: [f32; MAXP],
    /// The moves the search considers: the legal ones, or the one forced step.
    legal: u8,
    terminal: bool,
    mover: u8,
    /// The value of a terminal node.
    v0: [f32; MAXP],
}

/// What `reset` found at the root.
pub enum Root {
    /// The game is over.
    Over,
    /// The last two bikes are cut off: a fill race, no search.
    Cutoff,
    /// One legal move, or (`fill`) a mover alone in its region: no search.
    Forced { mv: u8, fill: bool },
    /// A tree was started; run simulations.
    Open,
}

pub struct Search {
    nodes: Vec<Node>,
    feat: Box<Feat>,
    work: Work,
    pub sc: Scratch,
    path: Vec<u32>,
    /// Network evaluations since the last `reset`.
    pub evals: u32,
    /// Bikes in the game at the root, minus one: the span of the value scale.
    span: f32,
}

pub struct RootResult {
    /// The moves the search considered.
    pub legal: u8,
    pub prior: [f32; 4],
    pub visits: [u32; 4],
    /// The root's mean value for the player to move.
    pub q: f32,
    /// The mover's mean value after each move (0 for unvisited moves).
    pub child_q: [f32; 4],
}

// paste:skip-begin
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn below(&mut self, n: u32) -> u32 {
        ((self.next_u64() >> 32) * n as u64 >> 32) as u32
    }

    fn normal(&mut self) -> f32 {
        let u1 = (1.0 - self.f32()).max(1e-12);
        let u2 = self.f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// Gamma(alpha, 1), Marsaglia and Tsang.
    pub fn gamma(&mut self, alpha: f32) -> f32 {
        if alpha < 1.0 {
            let u = (1.0 - self.f32()).max(1e-12);
            return self.gamma(alpha + 1.0) * u.powf(1.0 / alpha);
        }
        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;
            let u = (1.0 - self.f32()).max(1e-12);
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }
}
// paste:skip-end

impl Search {
    pub fn new(net: &Net) -> Search {
        Search {
            nodes: Vec::with_capacity(4096),
            feat: Box::new(Feat::new()),
            work: net.work(),
            sc: Scratch::new(),
            path: Vec::with_capacity(256),
            evals: 0,
            span: 1.0,
        }
    }

    /// Values of the players that are out, on the root's scale.
    fn dead_values(&self, st: &State) -> [f32; MAXP] {
        let mut v = [0.0; MAXP];
        for p in 0..st.n as usize {
            if st.rank[p] != 0 {
                // A bike that was out before the root gets a value below -1; nothing reads it.
                v[p] = 1.0 - 2.0 * (st.rank[p] - 1) as f32 / self.span;
            }
        }
        v
    }

    /// Append the node for `st`. Returns its index and its value per player,
    /// or `None` for a forced position that was passed through unevaluated.
    fn make_node(&mut self, net: &Net, st: &State, pass_ok: bool) -> (u32, Option<[f32; MAXP]>) {
        let mut node = Node {
            child: [0; 4],
            prior: [0.0; 4],
            n: 0,
            w: [0.0; MAXP],
            legal: 0,
            terminal: true,
            mover: st.to_move,
            v0: self.dead_values(st),
        };
        let mut evaluated = true;
        if !st.over() {
            features(st, &mut self.feat);
            let ns = self.feat.nslots;
            let me = st.to_move as usize;
            if ns == 2 && !self.feat.mover_connected {
                let u = cutoff_value(st, &mut self.sc);
                node.v0[me] = 1.0 - (1.0 - u) / self.span;
                node.v0[self.feat.slot[1] as usize] = 1.0 - (1.0 + u) / self.span;
            } else {
                let mut legal = st.legal_mask(me);
                let mut forced = legal & (legal - 1) == 0;
                if !forced && !self.feat.mover_connected {
                    let mut s = *st;
                    legal = 1 << fill_direction(&mut s, me, st.last[me], &mut self.sc).unwrap();
                    forced = true;
                }
                node.legal = legal;
                node.terminal = false;
                if forced && pass_ok {
                    evaluated = false;
                } else {
                    let (logits, u) = net.eval(&mut self.work, st, &self.feat);
                    self.evals += 1;
                    let mut mx = f32::NEG_INFINITY;
                    for a in 0..4 {
                        if legal >> a & 1 != 0 {
                            mx = mx.max(logits[a]);
                        }
                    }
                    let mut sum = 0.0;
                    for a in 0..4 {
                        if legal >> a & 1 != 0 {
                            node.prior[a] = (logits[a] - mx).exp();
                            sum += node.prior[a];
                        }
                    }
                    for a in 0..4 {
                        node.prior[a] /= sum;
                    }
                    // From the scale of the bikes in the game here to the root's.
                    let k = (ns - 1) as f32 / self.span;
                    for s in 0..ns {
                        node.v0[self.feat.slot[s] as usize] = 1.0 - (1.0 - u[s]) * k;
                    }
                }
            }
        }
        let v = node.v0;
        self.nodes.push(node);
        ((self.nodes.len() - 1) as u32, if evaluated { Some(v) } else { None })
    }

    /// Start a new tree at `st` and say whether it needs a search.
    pub fn reset(&mut self, net: &Net, st: &State) -> Root {
        self.evals = 0;
        self.nodes.clear();
        if st.over() {
            return Root::Over;
        }
        self.span = (st.alive_count() - 1) as f32;
        features(st, &mut self.feat);
        let me = st.to_move as usize;
        if self.feat.nslots == 2 && !self.feat.mover_connected {
            return Root::Cutoff;
        }
        let legal = st.legal_mask(me);
        if legal & (legal - 1) == 0 {
            return Root::Forced { mv: legal.trailing_zeros() as u8, fill: false };
        }
        if !self.feat.mover_connected {
            let mut s = *st;
            let mv = fill_direction(&mut s, me, st.last[me], &mut self.sc).unwrap();
            return Root::Forced { mv, fill: true };
        }
        let (_, v) = self.make_node(net, st, false);
        let root = &mut self.nodes[0];
        root.n = 1;
        root.w = v.unwrap();
        Root::Open
    }

    // paste:skip-begin
    /// Mix Dirichlet noise into the root priors (legal moves only).
    pub fn add_root_noise(&mut self, rng: &mut Rng, eps: f32, alpha: f32) {
        let legal = self.nodes[0].legal;
        let mut g = [0.0f32; 4];
        let mut sum = 0.0;
        for a in 0..4 {
            if legal >> a & 1 != 0 {
                g[a] = rng.gamma(alpha).max(1e-12);
                sum += g[a];
            }
        }
        let root = &mut self.nodes[0];
        for a in 0..4 {
            if legal >> a & 1 != 0 {
                root.prior[a] = (1.0 - eps) * root.prior[a] + eps * g[a] / sum;
            }
        }
    }

    /// The root's mean value for every player (below -1 for a bike that was out before the root).
    pub fn root_values(&self) -> [f32; MAXP] {
        let root = &self.nodes[0];
        let mut v = [0.0; MAXP];
        for k in 0..MAXP {
            v[k] = root.w[k] / root.n.max(1) as f32;
        }
        v
    }

    /// The tree after a search, as `key=value` words for the match recorder:
    /// its nodes, its deepest line, the mean depth of a simulation, every
    /// player's value, the visits of the four moves, and the most-visited
    /// line as the cell each bike steps to (`player:x:y`).
    pub fn stats_line(&self, root: &State) -> String {
        let n = self.nodes.len();
        if n == 0 {
            return String::new();
        }
        // A child is always created after its parent, so one pass in index order gives every depth.
        let mut depth = vec![0u32; n];
        let (mut deepest, mut through) = (0u32, 0u64);
        for i in 0..n {
            for a in 0..4 {
                let c = self.nodes[i].child[a] as usize;
                if c != 0 {
                    depth[c] = depth[i] + 1;
                }
            }
            deepest = deepest.max(depth[i]);
            if i > 0 {
                through += self.nodes[i].n as u64;
            }
        }
        let sims = self.nodes[0].n.saturating_sub(1).max(1);
        let r = self.root_result();
        let v = self.root_values();
        let mut line = String::new();
        let mut st = *root;
        let mut idx = 0usize;
        let mut plies = 0;
        while !self.nodes[idx].terminal && plies < 60 {
            let node = &self.nodes[idx];
            let best = (0..4).filter(|&a| node.child[a] != 0).max_by_key(|&a| self.nodes[node.child[a] as usize].n);
            let Some(a) = best else { break };
            let p = st.to_move as usize;
            st.play(a);
            if !line.is_empty() {
                line.push(';');
            }
            line.push_str(&format!("{}:{}:{}", p, st.hx[p], st.hy[p]));
            idx = node.child[a] as usize;
            plies += 1;
        }
        // How many nodes sit at each depth: the tree's shape.
        let mut at = vec![0u32; deepest as usize + 1];
        for i in 0..n {
            at[depth[i] as usize] += 1;
        }
        let shape: Vec<String> = at.iter().map(|c| c.to_string()).collect();
        format!(
            "nodes={} maxd={} avgd={:.1} pvd={} v={:.3},{:.3},{:.3},{:.3} vis={},{},{},{} dh={} pv={}",
            n, deepest, through as f64 / sims as f64, plies, v[0], v[1], v[2], v[3],
            r.visits[0], r.visits[1], r.visits[2], r.visits[3], shape.join(","), line
        )
    }
    // paste:skip-end

    pub fn simulate(&mut self, net: &Net, root: &State, p: &Params) {
        let mut st = *root;
        let mut idx = 0usize;
        self.path.clear();
        self.path.push(0);
        let mut chain = 0;
        let value;
        loop {
            let node = &self.nodes[idx];
            if node.terminal {
                value = node.v0;
                break;
            }
            let mut best = node.legal.trailing_zeros() as usize;
            if node.legal & (node.legal - 1) != 0 {
                let m = node.mover as usize;
                let sqrt_n = (node.n.max(1) as f32).sqrt();
                let mean = node.w[m] / node.n.max(1) as f32;
                let fpu = (mean - p.fpu_reduction).max(-1.0);
                let mut best_score = f32::NEG_INFINITY;
                for a in 0..4 {
                    if node.legal >> a & 1 == 0 {
                        continue;
                    }
                    let c = node.child[a] as usize;
                    let (q, n) = if c != 0 {
                        let ch = &self.nodes[c];
                        (ch.w[m] / ch.n as f32, ch.n)
                    } else {
                        (fpu, 0)
                    };
                    let score = q + p.c_puct * node.prior[a] * sqrt_n / (1 + n) as f32;
                    if score > best_score {
                        best_score = score;
                        best = a;
                    }
                }
            }
            let c = node.child[best];
            st.play(best);
            if c == 0 {
                let (new, v) = self.make_node(net, &st, chain < MAX_CHAIN);
                self.nodes[idx].child[best] = new;
                self.path.push(new);
                match v {
                    Some(v) => {
                        value = v;
                        break;
                    }
                    None => {
                        idx = new as usize;
                        chain += 1;
                        continue;
                    }
                }
            }
            idx = c as usize;
            self.path.push(c);
        }
        for &i in self.path.iter() {
            let node = &mut self.nodes[i as usize];
            node.n += 1;
            for k in 0..MAXP {
                node.w[k] += value[k];
            }
        }
    }

    pub fn root_result(&self) -> RootResult {
        let root = &self.nodes[0];
        let m = root.mover as usize;
        let mut r = RootResult {
            legal: root.legal,
            prior: root.prior,
            visits: [0; 4],
            q: root.w[m] / root.n.max(1) as f32,
            child_q: [0.0; 4],
        };
        for a in 0..4 {
            let c = root.child[a] as usize;
            if c != 0 {
                r.visits[a] = self.nodes[c].n;
                r.child_q[a] = self.nodes[c].w[m] / self.nodes[c].n as f32;
            }
        }
        r
    }
}

/// The most visited legal move; ties go to the higher value, then to the
/// higher prior (which decides alone if there was no time to search).
pub fn best_move(r: &RootResult) -> usize {
    let mut best = 4;
    for a in 0..4 {
        if r.legal >> a & 1 == 0 {
            continue;
        }
        if best == 4
            || r.visits[a] > r.visits[best]
            || (r.visits[a] == r.visits[best] && (r.child_q[a], r.prior[a]) > (r.child_q[best], r.prior[best]))
        {
            best = a;
        }
    }
    best
}

// paste:skip-begin
/// A move sampled in proportion to the visit counts.
pub fn sample_move(r: &RootResult, rng: &mut Rng) -> usize {
    let total: u32 = r.visits.iter().sum();
    if total == 0 {
        return best_move(r);
    }
    let mut x = rng.below(total);
    for a in 0..4 {
        if x < r.visits[a] {
            return a;
        }
        x -= r.visits[a];
    }
    best_move(r)
}
// paste:skip-end
