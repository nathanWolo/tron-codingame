//! tronzero: an AlphaZero-style agent for CodinGame Tron Battle (1v1).
//!
//!   tronzero selfplay --net N --games G --out FILE [--sims 64 ...]   training games
//!   tronzero match --a N1 --b N2 --pairs P [--sims 64]               network against network
//!   tronzero bench --net N                                            evaluation speed
//!   tronzero dump --net N --n K --out PREFIX                          parity vectors for the trainer
//!   tronzero vsbase --net N --pairs P --budget-ms MS                  against the alpha-beta engine
//!   tronzero [--budget-ms MS] [--net N]                               CodinGame stdin/stdout loop
//!
//! Without `--net`, the CodinGame loop reads `TRONZERO_NET`, then `net.bin`
//! beside the executable.

mod board;
mod mcts;
mod net;

use board::*;
use mcts::{best_move, sample_move, Params, Rng, Search};
use net::Net;
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let key = format!("--{name}");
    for i in 0..args.len() {
        if args[i] == key {
            return args.get(i + 1).cloned();
        }
        if let Some(v) = args[i].strip_prefix(&format!("{key}=")) {
            return Some(v.to_string());
        }
    }
    None
}

fn arg_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    arg(name).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn load_net(path: &str) -> Net {
    Net::load(path).unwrap_or_else(|e| {
        eprintln!("cannot load network {path}: {e}");
        std::process::exit(2);
    })
}

/// Two distinct uniformly random cells, as CodinGame spawns the bikes.
fn random_start(rng: &mut Rng) -> State {
    let a = rng.below(600) as i32;
    let mut b = rng.below(599) as i32;
    if b >= a {
        b += 1;
    }
    State::new((a % 30, a / 30), (b % 30, b / 30))
}

// --- Self-play -----------------------------------------------------------------------

/// One training position, 104 bytes on disk (see `train/tz.py::SAMPLE_DTYPE`).
struct Sample {
    occ: [u32; 20],
    heads: [u8; 4], // mover x, y, opponent x, y
    visits: [u16; 4],
    legal: u8,
    flags: u8, // bit 0: forced move (no search)
    q: f32,
    ply: u16,
    sims: u16,
    mover: u8,
}

impl Sample {
    fn new(st: &State, ply: u32) -> Sample {
        let me = st.to_move as usize;
        Sample {
            occ: st.occ,
            heads: [st.hx[me] as u8, st.hy[me] as u8, st.hx[me ^ 1] as u8, st.hy[me ^ 1] as u8],
            visits: [0; 4],
            legal: st.legal_mask(me),
            flags: 0,
            q: 0.0,
            ply: ply.min(65535) as u16,
            sims: 0,
            mover: me as u8,
        }
    }

    fn write(&self, out: &mut Vec<u8>, winner: usize) {
        for r in self.occ {
            out.extend_from_slice(&r.to_le_bytes());
        }
        out.extend_from_slice(&self.heads);
        for v in self.visits {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let z: i8 = if winner == self.mover as usize { 1 } else { -1 };
        out.extend_from_slice(&[self.legal, z as u8, self.flags, 0]);
        out.extend_from_slice(&self.q.to_le_bytes());
        out.extend_from_slice(&self.ply.to_le_bytes());
        out.extend_from_slice(&self.sims.to_le_bytes());
    }
}

#[derive(Default)]
struct SpStats {
    games: u32,
    samples: u64,
    searched: u64,
    plies: u64,
    p0_wins: u32,
    crashes: u32,
    cut_by_estimate: u32,
    cut_by_fill: u32,
    evals: u64,
}

struct SpCfg {
    sims: u32,
    temp_plies: u32,
    noise_eps: f32,
    noise_alpha: f32,
    params: Params,
}

/// Play one self-play game; returns its samples and the winner.
fn selfplay_game(
    net: &Net,
    search: &mut Search,
    rng: &mut Rng,
    cfg: &SpCfg,
    stats: &mut SpStats,
) -> (Vec<Sample>, usize) {
    let mut st = random_start(rng);
    let mut samples: Vec<Sample> = Vec::with_capacity(128);
    let mut ply = 0u32;
    let winner;
    loop {
        let me = st.to_move as usize;
        let legal = st.legal_mask(me);
        if legal == 0 {
            winner = me ^ 1;
            stats.crashes += 1;
            break;
        }
        let forced = legal.count_ones() == 1;
        let open = if forced { connected(&st) } else { search.reset(net, &st) };
        if !open {
            let (w, by_estimate) = adjudicate(&st, &mut search.sc);
            winner = w;
            if by_estimate {
                stats.cut_by_estimate += 1;
            } else {
                stats.cut_by_fill += 1;
            }
            break;
        }
        let mut s = Sample::new(&st, ply);
        let mv;
        if forced {
            mv = legal.trailing_zeros() as usize;
            s.visits[mv] = 1;
            s.flags = 1;
        } else {
            if cfg.noise_eps > 0.0 {
                search.add_root_noise(rng, cfg.noise_eps, cfg.noise_alpha);
            }
            for _ in 0..cfg.sims {
                search.simulate(net, &st, &cfg.params);
            }
            stats.evals += search.evals as u64;
            stats.searched += 1;
            let r = search.root_result();
            for a in 0..4 {
                s.visits[a] = r.visits[a].min(65535) as u16;
            }
            s.q = r.q;
            s.sims = cfg.sims.min(65535) as u16;
            mv = if ply < cfg.temp_plies { sample_move(&r, rng) } else { best_move(&r) };
        }
        samples.push(s);
        st.play(mv);
        ply += 1;
    }
    stats.plies += ply as u64;
    (samples, winner)
}

fn cmd_selfplay() {
    let net = load_net(&arg("net").expect("--net"));
    let games: u32 = arg_or("games", 1000);
    let threads: usize = arg_or("threads", 14);
    let seed: u64 = arg_or("seed", 1);
    let out_path = arg("out").expect("--out");
    let cfg = SpCfg {
        sims: arg_or("sims", 64),
        temp_plies: arg_or("temp-plies", 20),
        noise_eps: arg_or("noise-eps", 0.25),
        noise_alpha: arg_or("noise-alpha", 0.5),
        params: Params { c_puct: arg_or("c-puct", 1.0), fpu_reduction: arg_or("fpu", 0.2) },
    };
    let reuse = arg_or("reuse", 0u32) != 0;
    let out = Mutex::new(std::io::BufWriter::new(
        std::fs::File::create(&out_path).expect("cannot create --out"),
    ));
    let total = Mutex::new(SpStats::default());
    let next = AtomicU32::new(0);
    let start = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let mut search = Search::new(&net);
                search.reuse = reuse;
                let mut buf: Vec<u8> = Vec::new();
                loop {
                    let g = next.fetch_add(1, Ordering::Relaxed);
                    if g >= games {
                        break;
                    }
                    let mut rng = Rng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ ((g as u64) << 20));
                    rng.next_u64();
                    let mut st = SpStats::default();
                    let (samples, winner) = selfplay_game(&net, &mut search, &mut rng, &cfg, &mut st);
                    buf.clear();
                    for s in &samples {
                        s.write(&mut buf, winner);
                    }
                    out.lock().unwrap().write_all(&buf).unwrap();
                    let mut t = total.lock().unwrap();
                    t.games += 1;
                    t.samples += samples.len() as u64;
                    t.searched += st.searched;
                    t.plies += st.plies;
                    t.p0_wins += (winner == 0) as u32;
                    t.crashes += st.crashes;
                    t.cut_by_estimate += st.cut_by_estimate;
                    t.cut_by_fill += st.cut_by_fill;
                    t.evals += st.evals;
                }
            });
        }
    });
    out.lock().unwrap().flush().unwrap();
    let t = total.lock().unwrap();
    let secs = start.elapsed().as_secs_f64();
    println!(
        "{{\"games\": {}, \"samples\": {}, \"searched\": {}, \"mean_plies\": {:.1}, \"p0_win\": {:.4}, \
         \"crashes\": {}, \"cut_by_estimate\": {}, \"cut_by_fill\": {}, \"evals\": {}, \"seconds\": {:.1}, \
         \"samples_per_s\": {:.0}, \"evals_per_s\": {:.0}}}",
        t.games,
        t.samples,
        t.searched,
        t.plies as f64 / t.games.max(1) as f64,
        t.p0_wins as f64 / t.games.max(1) as f64,
        t.crashes,
        t.cut_by_estimate,
        t.cut_by_fill,
        t.evals,
        secs,
        t.samples as f64 / secs,
        t.evals as f64 / secs,
    );
}

// --- Network against network ---------------------------------------------------------

/// Play one game between two searchers; `seat[p]` is the agent (0 or 1) of player `p`.
fn match_game(
    nets: [&Net; 2],
    searches: &mut [Search; 2],
    sims: [u32; 2],
    params: [&Params; 2],
    start: State,
) -> usize {
    let mut st = start;
    loop {
        let me = st.to_move as usize;
        let legal = st.legal_mask(me);
        if legal == 0 {
            return me ^ 1;
        }
        let forced = legal.count_ones() == 1;
        let open = if forced { connected(&st) } else { searches[me].reset(nets[me], &st) };
        if !open {
            return adjudicate(&st, &mut searches[me].sc).0;
        }
        let mv = if forced {
            legal.trailing_zeros() as usize
        } else {
            for _ in 0..sims[me] {
                searches[me].simulate(nets[me], &st, params[me]);
            }
            best_move(&searches[me].root_result())
        };
        st.play(mv);
    }
}

fn cmd_match() {
    let net_a = load_net(&arg("a").expect("--a"));
    let net_b = load_net(&arg("b").expect("--b"));
    let pairs: u32 = arg_or("pairs", 200);
    let threads: usize = arg_or("threads", 14);
    let seed: u64 = arg_or("seed", 7);
    let sims_a: u32 = arg_or("sims-a", arg_or("sims", 64));
    let sims_b: u32 = arg_or("sims-b", arg_or("sims", 64));
    // Search settings can differ per side: --c-puct-a, --fpu-b, --reuse-a 1, ...
    let side = |x: &str| Params {
        c_puct: arg_or(&format!("c-puct-{x}"), arg_or("c-puct", 1.0)),
        fpu_reduction: arg_or(&format!("fpu-{x}"), arg_or("fpu", 0.2)),
    };
    let (params_a, params_b) = (side("a"), side("b"));
    let reuse_a = arg_or("reuse-a", arg_or("reuse", 0u32)) != 0;
    let reuse_b = arg_or("reuse-b", arg_or("reuse", 0u32)) != 0;
    let next = AtomicU32::new(0);
    // wins of A as player 0, wins of A as player 1
    // then the pairs A lost 0-2, split 1-1 and won 2-0
    let wins = Mutex::new([0u32; 5]);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let mut sa = Search::new(&net_a);
                let mut sb = Search::new(&net_b);
                sa.reuse = reuse_a;
                sb.reuse = reuse_b;
                loop {
                    let g = next.fetch_add(1, Ordering::Relaxed);
                    if g >= pairs {
                        break;
                    }
                    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9) ^ ((g as u64) << 20));
                    rng.next_u64();
                    let start = random_start(&mut rng);
                    // A moves first, then B moves first, from the same spawn.
                    let mut s = [sa, sb];
                    let w0 = match_game([&net_a, &net_b], &mut s, [sims_a, sims_b], [&params_a, &params_b], start);
                    let [a, b] = s;
                    let mut s = [b, a];
                    let w1 = match_game([&net_b, &net_a], &mut s, [sims_b, sims_a], [&params_b, &params_a], start);
                    let [b, a] = s;
                    sa = a;
                    sb = b;
                    let mut w = wins.lock().unwrap();
                    w[0] += (w0 == 0) as u32;
                    w[1] += (w1 == 1) as u32;
                    w[2 + (w0 == 0) as usize + (w1 == 1) as usize] += 1;
                }
            });
        }
    });
    let w = wins.lock().unwrap();
    let n = 2 * pairs;
    let a_wins = w[0] + w[1];
    let score = a_wins as f64 / n as f64;
    let se = (score * (1.0 - score) / n as f64).sqrt();
    println!(
        "{{\"games\": {}, \"a_wins\": {}, \"score\": {:.4}, \"ci95\": {:.4}, \"a_wins_first\": {}, \"a_wins_second\": {}, \
         \"pairs_lost\": {}, \"pairs_split\": {}, \"pairs_won\": {}}}",
        n, a_wins, score, 1.96 * se, w[0], w[1], w[2], w[3], w[4]
    );
}

// --- Speed and parity ----------------------------------------------------------------

/// A random open position: a random spawn, then random legal moves.
fn random_position(rng: &mut Rng) -> State {
    loop {
        let mut st = random_start(rng);
        let plies = rng.below(160);
        let mut ok = true;
        for _ in 0..plies {
            let legal = st.legal_mask(st.to_move as usize);
            if legal == 0 {
                ok = false;
                break;
            }
            let k = rng.below(legal.count_ones());
            let mut m = legal;
            for _ in 0..k {
                m &= m - 1;
            }
            st.play(m.trailing_zeros() as usize);
        }
        if ok && st.legal_mask(st.to_move as usize) != 0 && connected(&st) {
            return st;
        }
    }
}

fn cmd_bench() {
    let net = load_net(&arg("net").expect("--net"));
    let mut rng = Rng(arg_or("seed", 3));
    let positions: Vec<State> = (0..2000).map(|_| random_position(&mut rng)).collect();
    let mut wk = net.work();
    let mut feat = Box::new(Feat::new());
    let reps = 20;
    let t = Instant::now();
    let mut sum = 0.0f32;
    for _ in 0..reps {
        for st in &positions {
            features(st, &mut feat);
            sum += net.eval(&mut wk, st, &feat).1;
        }
    }
    let per = t.elapsed().as_secs_f64() / (reps * positions.len()) as f64;
    let t = Instant::now();
    for _ in 0..reps {
        for st in &positions {
            features(st, &mut feat);
        }
    }
    let feat_per = t.elapsed().as_secs_f64() / (reps * positions.len()) as f64;
    println!(
        "eval {:.2} us (features {:.2} us), {:.0} evals/s, checksum {sum}",
        per * 1e6,
        feat_per * 1e6,
        1.0 / per
    );
    for (name, parts) in [("canvas", 1), ("conv0", 2), ("conv1", 3), ("conv2", 4)] {
        let t = Instant::now();
        for _ in 0..reps {
            for st in &positions {
                net.eval_parts(&mut wk, st, &feat, parts);
            }
        }
        let per = t.elapsed().as_secs_f64() / (reps * positions.len()) as f64;
        println!("  through {name}: {:.2} us", per * 1e6);
    }
    let params = Params { c_puct: 1.0, fpu_reduction: 0.2 };
    let mut search = Search::new(&net);
    let t = Instant::now();
    let mut sims = 0u64;
    for st in positions.iter().take(200) {
        if search.reset(&net, st) {
            for _ in 0..1000 {
                search.simulate(&net, st, &params);
            }
            sims += 1000;
        }
    }
    println!("search {:.2} us per simulation", t.elapsed().as_secs_f64() / sims as f64 * 1e6);
}

fn cmd_dump() {
    let net = load_net(&arg("net").expect("--net"));
    let n: u32 = arg_or("n", 256);
    let prefix = arg("out").expect("--out");
    let mut rng = Rng(arg_or("seed", 11));
    let mut wk = net.work();
    let mut feat = Box::new(Feat::new());
    let mut samples: Vec<u8> = Vec::new();
    let mut outputs: Vec<u8> = Vec::new();
    for _ in 0..n {
        let st = random_position(&mut rng);
        features(&st, &mut feat);
        let (logits, v) = net.eval(&mut wk, &st, &feat);
        Sample::new(&st, 0).write(&mut samples, 0);
        for x in logits {
            outputs.extend_from_slice(&x.to_le_bytes());
        }
        outputs.extend_from_slice(&v.to_le_bytes());
        outputs.extend_from_slice(&feat.dist[0]);
        outputs.extend_from_slice(&feat.dist[1]);
    }
    std::fs::write(format!("{prefix}.samples"), samples).unwrap();
    std::fs::write(format!("{prefix}.outputs"), outputs).unwrap();
    println!("wrote {n} positions to {prefix}.samples and {prefix}.outputs");
}

// --- CodinGame ---------------------------------------------------------------------

/// A legal move that keeps the most space; for games this agent has no search for.
fn fallback_move(st: &State, me: usize) -> u8 {
    let legal = st.legal_mask(me);
    let mut best = (legal.trailing_zeros().min(3) as u8, -1);
    for d in 0..4u8 {
        if legal >> d & 1 != 0 {
            let mut s = *st;
            s.step(me, d as usize);
            let space = flood_count(&s, me);
            if space > best.1 {
                best = (d, space);
            }
        }
    }
    best.0
}

/// The agent's move under a time budget: forced moves at once, the primed
/// greedy fill once the bikes are cut off, the search otherwise.
/// Returns the direction, the simulations run and the root value of the move.
fn timed_move(
    net: &Net,
    search: &mut Search,
    st: &State,
    params: &Params,
    deadline: Instant,
) -> (u8, u32, f32) {
    let legal = st.legal_mask(st.to_move as usize);
    if legal == 0 {
        return (0, 0, -1.0);
    }
    if legal.count_ones() == 1 {
        return (legal.trailing_zeros() as u8, 0, 0.0);
    }
    if !search.reset(net, st) {
        return (primed_fill(st, st.to_move as usize, &mut search.sc).0, 0, 0.0);
    }
    let mut sims = 0u32;
    while Instant::now() < deadline {
        for _ in 0..8 {
            search.simulate(net, st, params);
        }
        sims += 8;
    }
    let r = search.root_result();
    let dir = best_move(&r);
    (dir as u8, sims, r.child_q[dir])
}

fn cmd_codingame() {
    let net_path = arg("net").or_else(|| std::env::var("TRONZERO_NET").ok()).unwrap_or_else(|| {
        let exe = std::env::current_exe().expect("no executable path");
        exe.with_file_name("net.bin").to_string_lossy().into_owned()
    });
    let net = load_net(&net_path);
    let budget_ms: u64 = arg("budget-ms")
        .or_else(|| std::env::var("TRON_BUDGET_MS").ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(95);
    let params = Params { c_puct: arg_or("c-puct", 1.0), fpu_reduction: arg_or("fpu", 0.2) };
    let mut search = Search::new(&net);
    search.reuse = arg_or("reuse", 1u32) != 0;
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let mut occ = [0u32; 20];
    let mut heads = [(-1i32, -1i32); 4];
    let mut first = true;
    loop {
        let line = match lines.next() {
            Some(Ok(t)) => t,
            _ => break,
        };
        let start = Instant::now();
        let mut tok = line.split_whitespace();
        let n: usize = tok.next().unwrap().parse().unwrap();
        let my_id: usize = tok.next().unwrap().parse().unwrap();
        for p in 0..n {
            let line = lines.next().unwrap().unwrap();
            let v: Vec<i32> = line.split_whitespace().map(|t| t.parse().unwrap()).collect();
            if v[0] < 0 {
                // In a two-player game a death ends the game, so no trail needs removing.
                heads[p] = (-1, -1);
                continue;
            }
            if first {
                occ[v[1] as usize] |= 1 << v[0];
            }
            occ[v[3] as usize] |= 1 << v[2];
            heads[p] = (v[2], v[3]);
        }
        first = false;
        let opp = (0..n).find(|&p| p != my_id && heads[p].0 >= 0);
        let mut st = State { occ, hx: [0; 2], hy: [0; 2], to_move: 0 };
        st.hx[0] = heads[my_id].0 as i8;
        st.hy[0] = heads[my_id].1 as i8;
        let dir: u8;
        if n != 2 || opp.is_none() {
            dir = if st.legal_mask(0) == 0 { 0 } else { fallback_move(&st, 0) };
        } else {
            let o = opp.unwrap();
            st.hx[1] = heads[o].0 as i8;
            st.hy[1] = heads[o].1 as i8;
            let deadline = start + Duration::from_millis(budget_ms);
            let (d, sims, q) = timed_move(&net, &mut search, &st, &params, deadline);
            dir = d;
            if sims > 0 {
                eprintln!(
                    "{} mm {} {}ms n={} d=0 a=2",
                    DIR_NAME[dir as usize],
                    (q * 1000.0) as i32,
                    start.elapsed().as_millis().max(1),
                    sims
                );
            }
        }
        println!("{}", DIR_NAME[dir as usize]);
        let _ = std::io::stdout().flush();
    }
}

// --- Against the alpha-beta engine, in process -------------------------------------

/// `tron-codingame/src/main.rs`, compiled in unchanged (`build.sh` copies it
/// without its inner doc comment), so matches need no pipes or referee.
mod baseline {
    #![allow(dead_code, unused_imports, unused_variables, unused_mut)]
    #![allow(clippy::all)]
    include!("baseline_src.rs");

    pub struct Player {
        scratch: Scratch,
        last_dir: u8,
    }

    impl Player {
        pub fn new() -> Player {
            Player { scratch: Scratch::new(), last_dir: NO_MOVE }
        }

        /// The engine's move for player `me` (two-player game).
        pub fn choose(&mut self, occ: &[u32; 20], hx: [i8; 2], hy: [i8; 2], me: usize, budget_ms: u64) -> u8 {
            let mut st = State::new(2);
            st.occupied.bits = *occ;
            st.head_x[0] = hx[0];
            st.head_y[0] = hy[0];
            st.head_x[1] = hx[1];
            st.head_y[1] = hy[1];
            st.alive_mask = 3;
            let d = choose_move(&mut st, me, self.last_dir, budget_ms, &mut self.scratch);
            self.last_dir = d;
            d
        }
    }
}

/// A spawn followed by `seed_plies` random legal moves per side, as the SPRT
/// referee opens its games. `None` if a bike got stuck on the way.
fn random_opening(rng: &mut Rng, seed_plies: u32) -> Option<State> {
    let mut st = random_start(rng);
    for _ in 0..2 * seed_plies {
        let legal = st.legal_mask(st.to_move as usize);
        if legal == 0 {
            return None;
        }
        let mut m = legal;
        for _ in 0..rng.below(legal.count_ones()) {
            m &= m - 1;
        }
        st.play(m.trailing_zeros() as usize);
    }
    if st.legal_mask(0) == 0 || st.legal_mask(1) == 0 {
        return None;
    }
    Some(st)
}

fn cmd_vsbase() {
    let net = load_net(&arg("net").expect("--net"));
    let pairs: u32 = arg_or("pairs", 100);
    let threads: usize = arg_or("threads", 8);
    let seed: u64 = arg_or("seed", 99);
    let budget_ms: u64 = arg_or("budget-ms", 20);
    let seed_plies: u32 = arg_or("seed-plies", 4);
    let params = Params { c_puct: arg_or("c-puct", 1.0), fpu_reduction: arg_or("fpu", 0.2) };
    let reuse = arg_or("reuse", 1u32) != 0;
    let debug = arg_or("debug", 0u32) != 0;
    let next = AtomicU32::new(0);
    // AZ wins moving first, AZ wins moving second, total plies, AZ searches, AZ simulations
    let tally = Mutex::new([0u64; 15]);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let mut search = Search::new(&net);
                search.reuse = reuse;
                loop {
                    let g = next.fetch_add(1, Ordering::Relaxed);
                    if g >= pairs {
                        break;
                    }
                    let mut rng = Rng(seed.wrapping_mul(0xD6E8_FEB8_6659_FD93) ^ ((g as u64) << 20));
                    rng.next_u64();
                    let start = loop {
                        if let Some(st) = random_opening(&mut rng, seed_plies) {
                            break st;
                        }
                    };
                    let mut local = [0u64; 15];
                    for az_seat in 0..2usize {
                        let mut st = start;
                        let mut base = baseline::Player::new();
                        // At the first cut-off position: does the agent win the fill race
                        // by the primed greedy fill, and by the cheap estimate?
                        let mut cut: Option<(bool, bool)> = None;
                        let mut last_az: Option<(State, u8, u32, [u32; 4], [f32; 4])> = None;
                        let winner = loop {
                            let me = st.to_move as usize;
                            let legal = st.legal_mask(me);
                            if legal == 0 {
                                break me ^ 1;
                            }
                            if cut.is_none() && !connected(&st) {
                                let sc = &mut search.sc;
                                let mover_wins = primed_fill(&st, me, sc).1 > primed_fill(&st, me ^ 1, sc).1;
                                let est_wins = approx_fill(&st, me, sc) > approx_fill(&st, me ^ 1, sc);
                                cut = Some(((me == az_seat) == mover_wins, (me == az_seat) == est_wins));
                            }
                            let d = if me == az_seat {
                                let deadline = Instant::now() + Duration::from_millis(budget_ms);
                                let (d, sims, _) = timed_move(&net, &mut search, &st, &params, deadline);
                                if sims > 0 {
                                    local[3] += 1;
                                    local[4] += sims as u64;
                                    if debug {
                                        let r = search.root_result();
                                        last_az = Some((st, d, sims, r.visits, r.child_q));
                                    }
                                }
                                d
                            } else {
                                base.choose(&st.occ, st.hx, st.hy, me, budget_ms)
                            };
                            if legal >> d & 1 == 0 {
                                break me ^ 1; // an illegal move loses
                            }
                            st.play(d as usize);
                            local[2] += 1;
                        };
                        let won = (winner == az_seat) as u64;
                        local[az_seat] += won;
                        match cut {
                            Some((ahead, est_ahead)) => {
                                local[if ahead { 5 } else { 7 }] += 1;
                                local[if ahead { 6 } else { 8 }] += won;
                                local[11] += (ahead != est_ahead) as u64;
                            }
                            None => local[10 - won as usize] += 1,
                        }
                        if debug && cut.is_none() && won == 0 {
                            // The agent was trapped before any cut-off: show its last search.
                            if let Some((s, d, sims, visits, q)) = last_az {
                                let mut text = format!(
                                    "TRAPPED pair {g} seat {az_seat}: played {} after {sims} sims, visits {visits:?}, q {q:.2?}\n",
                                    DIR_NAME[d as usize]
                                );
                                for y in 0..20 {
                                    for x in 0..30 {
                                        let c = if (s.hx[az_seat], s.hy[az_seat]) == (x as i8, y as i8) {
                                            'A'
                                        } else if (s.hx[az_seat ^ 1], s.hy[az_seat ^ 1]) == (x as i8, y as i8) {
                                            'B'
                                        } else if s.is_occ(x, y) {
                                            '#'
                                        } else {
                                            '.'
                                        };
                                        text.push(c);
                                    }
                                    text.push('\n');
                                }
                                print!("{text}");
                            }
                        }
                    }
                    // Pairs the agent lost 0-2, split 1-1 and won 2-0.
                    local[12 + (local[0] + local[1]) as usize] += 1;
                    let mut t = tally.lock().unwrap();
                    for i in 0..15 {
                        t[i] += local[i];
                    }
                }
            });
        }
    });
    let t = tally.lock().unwrap();
    let n = 2 * pairs as u64;
    let wins = t[0] + t[1];
    let score = wins as f64 / n as f64;
    let se = (score * (1.0 - score) / n as f64).sqrt();
    let elo = if score > 0.0 && score < 1.0 { -400.0 * (1.0 / score - 1.0).log10() } else { f64::NAN };
    println!(
        "{{\"games\": {}, \"az_wins\": {}, \"score\": {:.4}, \"ci95\": {:.4}, \"elo\": {:.0}, \
         \"az_wins_first\": {}, \"az_wins_second\": {}, \"mean_plies\": {:.1}, \"sims_per_search\": {:.0}, \
         \"budget_ms\": {}, \"cut_ahead\": {}, \"cut_ahead_won\": {}, \"cut_behind\": {}, \"cut_behind_won\": {},          \"no_cut_won\": {}, \"no_cut_lost\": {}, \"estimate_disagrees\": {},          \"pairs_lost\": {}, \"pairs_split\": {}, \"pairs_won\": {}}}",
        n, wins, score, 1.96 * se, elo, t[0], t[1],
        t[2] as f64 / n as f64, t[4] as f64 / t[3].max(1) as f64, budget_ms,
        t[5], t[6], t[7], t[8], t[9], t[10], t[11], t[12], t[13], t[14]
    );
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("selfplay") => cmd_selfplay(),
        Some("match") => cmd_match(),
        Some("bench") => cmd_bench(),
        Some("dump") => cmd_dump(),
        Some("vsbase") => cmd_vsbase(),
        _ => cmd_codingame(),
    }
}
