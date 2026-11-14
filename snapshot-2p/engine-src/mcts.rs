//! PUCT search for sequential two-player Tron.
//!
//! The settings follow slinky's AlphaZero (`baselines/az.py`): priors and leaf
//! values from the network, exact values at finished positions, unvisited
//! moves valued at the node's mean minus a first-play-urgency reduction, and
//! Dirichlet noise in the root priors during self-play. Turns alternate here,
//! so this is plain PUCT with a sign flip per ply, not slinky's decoupled
//! simultaneous-move search. A position where the bikes are cut off is a leaf
//! valued by the fill estimates, never by the network.

use crate::board::{cutoff_value, features, Feat, Scratch, State};
use crate::net::{Net, Work};

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
    /// Sum of backed-up values, from the point of view of the player to move here.
    w: f32,
    legal: u8,
    terminal: bool,
    v0: f32,
}

pub struct Search {
    nodes: Vec<Node>,
    feat: Box<Feat>,
    work: Work,
    pub sc: Scratch,
    path: Vec<u32>,
    /// Network evaluations since the last `reset`.
    pub evals: u32,
    /// Keep the subtree of the new root when `reset` is given a position one
    /// or two plies below the last root (as slinky's native engine does).
    pub reuse: bool,
    root_state: Option<State>,
}

fn same_position(a: &State, b: &State) -> bool {
    a.occ == b.occ && a.hx == b.hx && a.hy == b.hy && a.to_move == b.to_move
}

pub struct RootResult {
    pub visits: [u32; 4],
    /// The root's mean value for the player to move.
    pub q: f32,
    /// The mover's mean value after each move (0 for unvisited moves).
    pub child_q: [f32; 4],
}

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

impl Search {
    pub fn new(net: &Net) -> Search {
        Search {
            nodes: Vec::with_capacity(4096),
            feat: Box::new(Feat::new()),
            work: net.work(),
            sc: Scratch::new(),
            path: Vec::with_capacity(256),
            evals: 0,
            reuse: false,
            root_state: None,
        }
    }

    /// Append the node for `st` and return its index and its value for the mover.
    fn make_node(&mut self, net: &Net, st: &State) -> (u32, f32) {
        let me = st.to_move as usize;
        let legal = st.legal_mask(me);
        let mut node =
            Node { child: [0; 4], prior: [0.0; 4], n: 0, w: 0.0, legal, terminal: true, v0: -1.0 };
        if legal != 0 {
            features(st, &mut self.feat);
            if !self.feat.connected {
                node.v0 = cutoff_value(st, &mut self.sc);
            } else {
                let (logits, v) = net.eval(&mut self.work, st, &self.feat);
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
                node.terminal = false;
                node.v0 = v;
            }
        }
        let v = node.v0;
        self.nodes.push(node);
        ((self.nodes.len() - 1) as u32, v)
    }

    /// Start a new tree at `st`. Returns false if `st` needs no search: the
    /// mover is stuck or the bikes are cut off.
    pub fn reset(&mut self, net: &Net, st: &State) -> bool {
        self.evals = 0;
        if self.reuse {
            if let Some(r) = self.find(st) {
                self.reroot(r);
                self.root_state = Some(*st);
                return !self.nodes[0].terminal;
            }
        }
        self.nodes.clear();
        let (_, v) = self.make_node(net, st);
        let root = &mut self.nodes[0];
        root.n = 1;
        root.w = v;
        self.root_state = Some(*st);
        !root.terminal
    }

    /// The node of `st` if it is the root, a child or a grandchild of the root.
    fn find(&self, st: &State) -> Option<u32> {
        let rs = self.root_state?;
        if self.nodes.is_empty() {
            return None;
        }
        if same_position(&rs, st) {
            return Some(0);
        }
        for a in 0..4 {
            let c = self.nodes[0].child[a];
            if c == 0 {
                continue;
            }
            let mut s1 = rs;
            s1.play(a);
            if same_position(&s1, st) {
                return Some(c);
            }
            for b in 0..4 {
                let g = self.nodes[c as usize].child[b];
                if g == 0 {
                    continue;
                }
                let mut s2 = s1;
                s2.play(b);
                if same_position(&s2, st) {
                    return Some(g);
                }
            }
        }
        None
    }

    /// Keep only the subtree under node `r`, which becomes node 0.
    fn reroot(&mut self, r: u32) {
        if r == 0 {
            return;
        }
        let old = std::mem::take(&mut self.nodes);
        let mut new: Vec<Node> = Vec::with_capacity(old.len().max(4096));
        let mut order: Vec<u32> = vec![r];
        let mut i = 0;
        while i < order.len() {
            let o = &old[order[i] as usize];
            let mut node = Node {
                child: [0; 4],
                prior: o.prior,
                n: o.n,
                w: o.w,
                legal: o.legal,
                terminal: o.terminal,
                v0: o.v0,
            };
            for a in 0..4 {
                if o.child[a] != 0 {
                    order.push(o.child[a]);
                    node.child[a] = (order.len() - 1) as u32;
                }
            }
            new.push(node);
            i += 1;
        }
        self.nodes = new;
    }

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

    pub fn simulate(&mut self, net: &Net, root: &State, p: &Params) {
        let mut st = *root;
        let mut idx = 0usize;
        self.path.clear();
        self.path.push(0);
        let value;
        loop {
            let node = &self.nodes[idx];
            if node.terminal {
                value = node.v0;
                break;
            }
            let sqrt_n = (node.n.max(1) as f32).sqrt();
            let mean = node.w / node.n.max(1) as f32;
            let fpu = (mean - p.fpu_reduction).max(-1.0);
            let mut best = 4usize;
            let mut best_score = f32::NEG_INFINITY;
            for a in 0..4 {
                if node.legal >> a & 1 == 0 {
                    continue;
                }
                let c = node.child[a] as usize;
                let (q, n) = if c != 0 {
                    let ch = &self.nodes[c];
                    (-ch.w / ch.n as f32, ch.n)
                } else {
                    (fpu, 0)
                };
                let score = q + p.c_puct * node.prior[a] * sqrt_n / (1 + n) as f32;
                if score > best_score {
                    best_score = score;
                    best = a;
                }
            }
            let c = node.child[best];
            st.play(best);
            if c == 0 {
                let (new, v) = self.make_node(net, &st);
                self.nodes[idx].child[best] = new;
                self.path.push(new);
                value = v;
                break;
            }
            idx = c as usize;
            self.path.push(c);
        }
        // `value` is for the mover at the last node; it flips every ply up.
        let mut v = value;
        for &i in self.path.iter().rev() {
            let node = &mut self.nodes[i as usize];
            node.n += 1;
            node.w += v;
            v = -v;
        }
    }

    pub fn root_result(&self) -> RootResult {
        let root = &self.nodes[0];
        let mut r = RootResult { visits: [0; 4], q: root.w / root.n.max(1) as f32, child_q: [0.0; 4] };
        for a in 0..4 {
            let c = root.child[a] as usize;
            if c != 0 {
                r.visits[a] = self.nodes[c].n;
                r.child_q[a] = -self.nodes[c].w / self.nodes[c].n as f32;
            }
        }
        r
    }

    pub fn root_legal(&self) -> u8 {
        self.nodes[0].legal
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// The most visited move; ties go to the higher value.
pub fn best_move(r: &RootResult) -> usize {
    let mut best = 0;
    for a in 1..4 {
        if r.visits[a] > r.visits[best]
            || (r.visits[a] == r.visits[best] && r.child_q[a] > r.child_q[best])
        {
            best = a;
        }
    }
    best
}

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
