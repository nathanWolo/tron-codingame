//! tronzero: an AlphaZero-style agent for CodinGame Tron Battle (2-4 players).
//!
//!   tronzero selfplay --net N --games G --out FILE [--sims 64 --mix 1,1,3 ...]  training games
//!   tronzero match --a N1 --b N2 --pairs P [--players 4 --sims 64 --book B]     network against network
//!   tronzero openings --net N --players P --count C --out B [--max-abs 0.15]    book of even spawns
//!   tronzero vsbase --net N --pairs P [--players 4 --budget-ms MS]              against the alpha-beta engine
//!   tronzero bench --net N                                                      evaluation speed
//!   tronzero dump --net N --n K --out PREFIX                                    parity vectors for the trainer
//!   tronzero check                                                              rule invariants on random games
//!   tronzero fillbench --net N                                                  time of the space fill
//!   tronzero [--budget-ms MS] [--net N | --packed TXT]                          CodinGame stdin/stdout loop
//!
//! Without `--net` or `--packed`, the CodinGame loop reads `TRONZERO_NET`,
//! then `net.bin` beside the executable. `--sims N` (or `TRON_SIMS`) plays a
//! fixed number of simulations per move instead of a time limit.

mod board;
mod mcts;
mod net;
mod play;

/// Milliseconds the CodinGame loop thinks per turn (the limit is 100) when no
/// `--budget-ms` is given. The paste carries the same two constants.
pub const TURN_BUDGET_MS: u64 = 95;
pub const FIRST_TURN_BUDGET_MS: u64 = 80;

use board::*;
use mcts::{best_move, sample_move, Params, Rng, Root, Search};
use play::{timed_move, Limit};
use net::Net;
use std::io::Write;
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

/// The search settings when no flag overrides them. The paste carries the
/// same two: `tools/make_paste.py` reads them from here.
pub const C_PUCT: f32 = 0.6;
pub const FPU_REDUCTION: f32 = 0.2;

fn search_params() -> Params {
    Params { c_puct: arg_or("c-puct", C_PUCT), fpu_reduction: arg_or("fpu", FPU_REDUCTION) }
}

/// `n` distinct uniformly random cells, as CodinGame spawns the bikes.
fn random_start(rng: &mut Rng, n: usize) -> State {
    let mut cells = [(0i32, 0i32); MAXP];
    let mut k = 0;
    while k < n {
        let c = rng.below(600) as i32;
        let xy = (c % 30, c / 30);
        if !cells[..k].contains(&xy) {
            cells[k] = xy;
            k += 1;
        }
    }
    State::new(&cells[..n])
}

/// A uniformly random legal move of the player to move.
fn random_move(st: &State, rng: &mut Rng) -> usize {
    let legal = st.legal_mask(st.to_move as usize);
    let mut m = legal;
    for _ in 0..rng.below(legal.count_ones()) {
        m &= m - 1;
    }
    m.trailing_zeros() as usize
}

/// A game's score for a seat in [0, 1]: 1 for winning, 0 for being first out,
/// evenly spaced between (in a duel: 1 or 0).
fn seat_score(st: &State, p: usize) -> f64 {
    1.0 - (st.rank[p] - 1) as f64 / (st.n - 1) as f64
}

/// With the last two bikes cut off, decide the fill race and end the game.
fn finish_cutoff(st: &mut State, sc: &mut Scratch) -> bool {
    let me = st.to_move as usize;
    let (winner, by_estimate) = adjudicate(st, sc);
    let loser = if winner == me { st.next_alive(me) } else { me };
    st.kill(loser);
    by_estimate
}

// --- Self-play -----------------------------------------------------------------------

/// One training position, 352 bytes on disk (see `train/tz.py::SAMPLE_DTYPE`).
/// Per-player fields are in slot order: the mover, then the others in turn order.
struct Sample {
    trail: [[u32; 20]; MAXP],
    heads: [u8; 8], // x, y per slot; 255 for an unused slot
    seat: [u8; MAXP],
    n_alive: u8,
    n_players: u8,
    legal: u8,
    flags: u8, // bit 0: one legal move; bit 1: space-filling step (neither is searched)
    visits: [u16; 4],
    q: f32,
    ply: u16,
    sims: u16,
}

impl Sample {
    fn new(st: &State, ply: u32) -> Sample {
        let (slot, ns) = st.slots();
        let mut s = Sample {
            trail: [[0; 20]; MAXP],
            heads: [255; 8],
            seat: [255; MAXP],
            n_alive: ns as u8,
            n_players: st.n,
            legal: st.legal_mask(st.to_move as usize),
            flags: 0,
            visits: [0; 4],
            q: 0.0,
            ply: ply.min(65535) as u16,
            sims: 0,
        };
        for k in 0..ns {
            let p = slot[k] as usize;
            s.trail[k] = st.trail[p];
            s.heads[2 * k] = st.hx[p] as u8;
            s.heads[2 * k + 1] = st.hy[p] as u8;
            s.seat[k] = p as u8;
        }
        s
    }

    /// `rank`: each player's final placement (1 = winner).
    fn write(&self, out: &mut Vec<u8>, rank: &[u8; MAXP]) {
        for t in &self.trail {
            for r in t {
                out.extend_from_slice(&r.to_le_bytes());
            }
        }
        out.extend_from_slice(&self.heads);
        for v in self.visits {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&[self.n_alive, self.n_players, self.legal, self.flags]);
        for k in 0..MAXP {
            out.push(if self.seat[k] == 255 { 0 } else { rank[self.seat[k] as usize] });
        }
        out.extend_from_slice(&self.q.to_le_bytes());
        out.extend_from_slice(&self.ply.to_le_bytes());
        out.extend_from_slice(&self.sims.to_le_bytes());
    }
}

#[derive(Default)]
struct SpStats {
    games: u32,
    games_by_n: [u32; 3],
    samples: u64,
    searched: u64,
    forced: u64,
    filled: u64,
    plies: u64,
    p0_wins: u32,
    cut_by_estimate: u32,
    cut_by_fill: u32,
    evals: u64,
}

struct SpCfg {
    sims: u32,
    temp_rounds: u32,
    noise_eps: f32,
    noise_alpha: f32,
    params: Params,
    /// Relative numbers of two-, three- and four-player games.
    mix: [u32; 3],
    /// Share of the unsearched positions (one legal move, space filling) that
    /// are recorded. They carry no policy target and come in long runs.
    keep_forced: f32,
}

/// Play one self-play game; returns its samples and the final position.
fn selfplay_game(
    net: &Net,
    search: &mut Search,
    rng: &mut Rng,
    cfg: &SpCfg,
    stats: &mut SpStats,
) -> (Vec<Sample>, State) {
    let mut x = rng.below(cfg.mix.iter().sum());
    let mut n = 2;
    for (i, &m) in cfg.mix.iter().enumerate() {
        if x < m {
            n = i + 2;
            break;
        }
        x -= m;
    }
    stats.games_by_n[n - 2] += 1;
    let mut st = random_start(rng, n);
    let mut samples: Vec<Sample> = Vec::with_capacity(256);
    let mut ply = 0u32;
    while !st.over() {
        let root = search.reset(net, &st);
        let mut s = Sample::new(&st, ply);
        let mut keep = true;
        let mv;
        match root {
            Root::Over => unreachable!(),
            Root::Cutoff => {
                if finish_cutoff(&mut st, &mut search.sc) {
                    stats.cut_by_estimate += 1;
                } else {
                    stats.cut_by_fill += 1;
                }
                break;
            }
            Root::Forced { mv: m, fill } => {
                mv = m as usize;
                s.visits[mv] = 1;
                s.flags = if fill { 2 } else { 1 };
                keep = rng.f32() < cfg.keep_forced;
                if fill {
                    stats.filled += 1;
                } else {
                    stats.forced += 1;
                }
            }
            Root::Open => {
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
                mv = if ply < cfg.temp_rounds * n as u32 { sample_move(&r, rng) } else { best_move(&r) };
            }
        }
        if keep {
            samples.push(s);
        }
        st.play(mv);
        ply += 1;
    }
    stats.plies += ply as u64;
    (samples, st)
}

fn cmd_selfplay() {
    let net = load_net(&arg("net").expect("--net"));
    let games: u32 = arg_or("games", 1000);
    let threads: usize = arg_or("threads", 14);
    let seed: u64 = arg_or("seed", 1);
    let out_path = arg("out").expect("--out");
    let mut mix = [1u32, 1, 3];
    if let Some(m) = arg("mix") {
        let v: Vec<u32> = m.split(',').map(|t| t.parse().expect("--mix a,b,c")).collect();
        assert!(v.len() == 3 && v.iter().sum::<u32>() > 0, "--mix a,b,c");
        mix = [v[0], v[1], v[2]];
    }
    let cfg = SpCfg {
        sims: arg_or("sims", 64),
        temp_rounds: arg_or("temp-rounds", 10),
        noise_eps: arg_or("noise-eps", 0.25),
        noise_alpha: arg_or("noise-alpha", 0.5),
        params: search_params(),
        mix,
        keep_forced: arg_or("keep-forced", 0.25),
    };
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
                let mut buf: Vec<u8> = Vec::new();
                loop {
                    let g = next.fetch_add(1, Ordering::Relaxed);
                    if g >= games {
                        break;
                    }
                    let mut rng = Rng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ ((g as u64) << 20));
                    rng.next_u64();
                    let mut st = SpStats::default();
                    let (samples, end) = selfplay_game(&net, &mut search, &mut rng, &cfg, &mut st);
                    buf.clear();
                    for s in &samples {
                        s.write(&mut buf, &end.rank);
                    }
                    out.lock().unwrap().write_all(&buf).unwrap();
                    let mut t = total.lock().unwrap();
                    t.games += 1;
                    for i in 0..3 {
                        t.games_by_n[i] += st.games_by_n[i];
                    }
                    t.samples += samples.len() as u64;
                    t.searched += st.searched;
                    t.forced += st.forced;
                    t.filled += st.filled;
                    t.plies += st.plies;
                    t.p0_wins += (end.rank[0] == 1) as u32;
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
        "{{\"games\": {}, \"games_2p\": {}, \"games_3p\": {}, \"games_4p\": {}, \"samples\": {}, \
         \"searched\": {}, \"forced\": {}, \"filled\": {}, \"mean_plies\": {:.1}, \"p0_win\": {:.4}, \
         \"cut_by_estimate\": {}, \"cut_by_fill\": {}, \"evals\": {}, \"seconds\": {:.1}, \
         \"samples_per_s\": {:.0}, \"evals_per_s\": {:.0}}}",
        t.games,
        t.games_by_n[0],
        t.games_by_n[1],
        t.games_by_n[2],
        t.samples,
        t.searched,
        t.forced,
        t.filled,
        t.plies as f64 / t.games.max(1) as f64,
        t.p0_wins as f64 / t.games.max(1) as f64,
        t.cut_by_estimate,
        t.cut_by_fill,
        t.evals,
        secs,
        t.samples as f64 / secs,
        t.evals as f64 / secs,
    );
}

// --- Matches -------------------------------------------------------------------------

/// Per-opening results of a match: the measured side's mean score in each
/// opening, its wins and the number of games.
#[derive(Default)]
struct Tally {
    scores: Vec<f64>,
    wins: u64,
    games: u64,
    rank_sum: u64,
    /// How often the measured side's seats placed first, second, third, fourth.
    placed: [u64; MAXP],
    seats: u64,
    plies: u64,
    searches: u64,
    sims: u64,
    /// Multi-player games that came down to the agent and one other bike: how
    /// many, in how many the agent then had the larger first-arrival territory,
    /// and its wins from ahead and from behind.
    finals: [u64; 4],
}

impl Tally {
    fn merge(&mut self, o: &Tally) {
        self.scores.extend_from_slice(&o.scores);
        self.wins += o.wins;
        self.games += o.games;
        self.rank_sum += o.rank_sum;
        for k in 0..MAXP {
            self.placed[k] += o.placed[k];
        }
        self.seats += o.seats;
        self.plies += o.plies;
        self.searches += o.searches;
        self.sims += o.sims;
        for k in 0..4 {
            self.finals[k] += o.finals[k];
        }
    }

    /// The JSON fields every match reports. `score` averages the openings;
    /// the interval comes from the spread between openings, since the games
    /// of one opening are not independent.
    fn json(&self) -> String {
        let n = self.scores.len().max(1) as f64;
        let mean = self.scores.iter().sum::<f64>() / n;
        let var = self.scores.iter().map(|s| (s - mean) * (s - mean)).sum::<f64>() / (n - 1.0).max(1.0);
        let lost = self.scores.iter().filter(|&&s| s < 0.5 - 1e-9).count();
        let won = self.scores.iter().filter(|&&s| s > 0.5 + 1e-9).count();
        format!(
            "\"openings\": {}, \"games\": {}, \"score\": {:.4}, \"ci95\": {:.4}, \"win_rate\": {:.4}, \
             \"mean_rank\": {:.3}, \"placed\": {:?}, \"mean_plies\": {:.1}, \"pairs_lost\": {}, \"pairs_split\": {}, \"pairs_won\": {}",
            self.scores.len(),
            self.games,
            mean,
            1.96 * (var / n).sqrt(),
            self.wins as f64 / self.games.max(1) as f64,
            self.rank_sum as f64 / self.seats.max(1) as f64,
            self.placed,
            self.plies as f64 / self.games.max(1) as f64,
            lost,
            self.scores.len() - lost - won,
            won
        )
    }
}

/// Seatings of two sides (0 is the measured one) that give each side every
/// seat equally often.
fn seatings(n: usize) -> &'static [[u8; MAXP]] {
    match n {
        2 => &[[0, 1, 0, 0], [1, 0, 0, 0]],
        3 => &[[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        _ => &[[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]],
    }
}

/// Play one game between two searchers at fixed simulations; `side[p]` is the
/// searcher of player `p`. Cut-off endings are adjudicated as in self-play.
fn match_game(
    nets: [&Net; 2],
    searches: &mut [Search; 2],
    sims: [u32; 2],
    params: [&Params; 2],
    side: &[u8; MAXP],
    start: State,
) -> State {
    let mut st = start;
    while !st.over() {
        let a = side[st.to_move as usize] as usize;
        let mv = match searches[a].reset(nets[a], &st) {
            Root::Over => unreachable!(),
            Root::Cutoff => {
                finish_cutoff(&mut st, &mut searches[a].sc);
                break;
            }
            Root::Forced { mv, .. } => mv as usize,
            Root::Open => {
                for _ in 0..sims[a] {
                    searches[a].simulate(nets[a], &st, params[a]);
                }
                best_move(&searches[a].root_result())
            }
        };
        st.play(mv);
    }
    st
}

/// Add one finished game to `t`; returns the mean score of side 0's seats.
fn tally_game(t: &mut Tally, end: &State, side: &[u8; MAXP]) -> f64 {
    let n = end.n as usize;
    let (mut score, mut seats) = (0.0, 0);
    for p in 0..n {
        if side[p] == 0 {
            score += seat_score(end, p);
            seats += 1;
            t.rank_sum += end.rank[p] as u64;
            t.placed[end.rank[p] as usize - 1] += 1;
            t.wins += (end.rank[p] == 1) as u64;
        }
    }
    t.seats += seats;
    t.games += 1;
    score / seats as f64
}

fn cmd_match() {
    let net_a = load_net(&arg("a").expect("--a"));
    let net_b = load_net(&arg("b").expect("--b"));
    let pairs: u32 = arg_or("pairs", 200);
    let n: usize = arg_or("players", 2);
    let threads: usize = arg_or("threads", 14);
    let seed: u64 = arg_or("seed", 7);
    let sims_a: u32 = arg_or("sims-a", arg_or("sims", 64));
    let sims_b: u32 = arg_or("sims-b", arg_or("sims", 64));
    // Search settings can differ per side: --c-puct-a, --fpu-b, ...
    let side = |x: &str| Params {
        c_puct: arg_or(&format!("c-puct-{x}"), arg_or("c-puct", C_PUCT)),
        fpu_reduction: arg_or(&format!("fpu-{x}"), arg_or("fpu", FPU_REDUCTION)),
    };
    let (params_a, params_b) = (side("a"), side("b"));
    // With --book the openings come from the file, in an order shuffled by
    // --seed and starting at --skip, so none repeats within a run and a second
    // run with the same seed and a later --skip plays different ones.
    let book = arg("book").map(|p| load_book(&p, n));
    let skip: usize = arg_or("skip", 0);
    let order = book.as_ref().map(|b| shuffled(b.len(), seed)).unwrap_or_default();
    if let Some(b) = &book {
        assert!(skip + pairs as usize <= b.len(), "the book has {} openings; --skip plus --pairs asks for more", b.len());
    }
    let next = AtomicU32::new(0);
    let total = Mutex::new(Tally::default());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let mut s = [Search::new(&net_a), Search::new(&net_b)];
                let mut local = Tally::default();
                loop {
                    let g = next.fetch_add(1, Ordering::Relaxed);
                    if g >= pairs {
                        break;
                    }
                    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9) ^ ((g as u64) << 20));
                    rng.next_u64();
                    let start = match &book {
                        Some(b) => State::new(&b[order[skip + g as usize]]),
                        None => random_start(&mut rng, n),
                    };
                    let mut score = 0.0;
                    for seating in seatings(n) {
                        let end = match_game(
                            [&net_a, &net_b], &mut s, [sims_a, sims_b], [&params_a, &params_b], seating, start,
                        );
                        score += tally_game(&mut local, &end, seating);
                    }
                    local.scores.push(score / seatings(n).len() as f64);
                }
                total.lock().unwrap().merge(&local);
            });
        }
    });
    println!("{{\"players\": {n}, {}}}", total.lock().unwrap().json());
}

// --- Opening books -------------------------------------------------------------------

/// Spawns from a book in the SPRT referee's format: JSON whose last key,
/// "spawns", lists the openings, each a list of `n` cells [x, y].
fn load_book(path: &str, n: usize) -> Vec<Vec<(i32, i32)>> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("cannot read book {path}: {e}");
        std::process::exit(2);
    });
    let at = text.find("\"spawns\"").expect("the book has no \"spawns\" key");
    let mut nums: Vec<i32> = Vec::new();
    let mut cur: Option<i32> = None;
    for ch in text[at + 8..].chars() {
        match ch.to_digit(10) {
            Some(d) => cur = Some(cur.unwrap_or(0) * 10 + d as i32),
            None => nums.extend(cur.take()),
        }
    }
    assert!(!nums.is_empty() && nums.len() % (2 * n) == 0, "{path} is not a list of {n}-player spawns");
    let book: Vec<Vec<(i32, i32)>> = nums.chunks(2 * n).map(|c| c.chunks(2).map(|xy| (xy[0], xy[1])).collect()).collect();
    assert!(book.iter().flatten().all(|&(x, y)| in_bounds(x, y)), "{path} has a cell off the board");
    book
}

/// The numbers 0..len in an order fixed by `seed`.
fn shuffled(len: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..len).collect();
    let mut rng = Rng(seed.wrapping_mul(0xA24B_AED4_963E_E407) ^ 0xB00C);
    rng.next_u64();
    for i in (1..len).rev() {
        idx.swap(i, rng.below(i as u32 + 1) as usize);
    }
    idx
}

/// Write a book of spawns the network judges even: after `--sims`
/// simulations from the spawn, no bike's value is further than `--max-abs`
/// from zero (values are placements in [-1, 1]; 0 is the middle of the field).
/// Spawns are drawn as CodinGame draws them and kept in the order drawn.
fn cmd_openings() {
    let net = load_net(&arg("net").expect("--net"));
    let n: usize = arg_or("players", 4);
    let count: usize = arg_or("count", 2000);
    let sims: u32 = arg_or("sims", 2000);
    let max_abs: f32 = arg_or("max-abs", 0.15);
    let threads: usize = arg_or("threads", 14);
    let seed: u64 = arg_or("seed", 1);
    let out = arg("out").expect("--out");
    assert!((2..=4).contains(&n));
    let params = search_params();
    let next = AtomicU32::new(0);
    // (index drawn, cells, the largest |value| of any bike)
    let kept = Mutex::new(Vec::<(u32, Vec<(i32, i32)>, f32)>::new());
    let all = Mutex::new(Vec::<f32>::new());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let mut search = Search::new(&net);
                loop {
                    if kept.lock().unwrap().len() >= count {
                        break;
                    }
                    let g = next.fetch_add(1, Ordering::Relaxed);
                    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((g as u64) << 20) ^ n as u64);
                    rng.next_u64();
                    let st = random_start(&mut rng, n);
                    // A spawn where the first bike has no choice is never even enough to keep.
                    let Root::Open = search.reset(&net, &st) else { continue };
                    for _ in 0..sims {
                        search.simulate(&net, &st, &params);
                    }
                    let v = search.root_values();
                    let worst = (0..n).map(|p| v[p].abs()).fold(0.0, f32::max);
                    all.lock().unwrap().push(worst);
                    if worst <= max_abs {
                        let cells = (0..n).map(|p| (st.hx[p] as i32, st.hy[p] as i32)).collect();
                        kept.lock().unwrap().push((g, cells, worst));
                    }
                }
            });
        }
    });
    let mut kept = kept.into_inner().unwrap();
    kept.sort_by_key(|k| k.0);
    kept.truncate(count);
    let mut all = all.into_inner().unwrap();
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f64| all[((all.len() - 1) as f64 * f) as usize];
    let rows: Vec<String> = kept
        .iter()
        .map(|(_, cells, _)| format!("[{}]", cells.iter().map(|(x, y)| format!("[{x}, {y}]")).collect::<Vec<_>>().join(", ")))
        .collect();
    let json = format!(
        "{{\"width\": 30, \"height\": 20, \"players\": {n}, \"count\": {}, \"source\": \"tronzero openings: uniform spawns kept when every bike's value after {sims} simulations is within {max_abs} of zero (seed {seed})\", \"spawns\": [\n{}\n]}}\n",
        kept.len(),
        rows.join(",\n")
    );
    std::fs::write(&out, json).expect("cannot write the book");
    println!(
        "{out}: {} openings for {n} players kept of {} drawn ({:.1}%); largest |value| over all drawn: median {:.3}, 25th pct {:.3}, 75th pct {:.3}, 90th pct {:.3}",
        kept.len(),
        all.len(),
        100.0 * kept.len() as f64 / all.len().max(1) as f64,
        q(0.5),
        q(0.25),
        q(0.75),
        q(0.9)
    );
}

// --- Speed and parity ----------------------------------------------------------------

/// A random unfinished position: a random spawn of two to four bikes, then
/// random legal moves (bikes may have died on the way).
fn random_position(rng: &mut Rng) -> State {
    loop {
        let n = 2 + rng.below(3) as usize;
        let mut st = random_start(rng, n);
        for _ in 0..rng.below(100 * n as u32) {
            if st.over() {
                break;
            }
            let mv = random_move(&st, rng);
            st.play(mv);
        }
        if !st.over() {
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
            sum += net.eval(&mut wk, st, &feat).1[0];
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
                features(st, &mut feat);
                net.eval_parts(&mut wk, st, &feat, parts);
            }
        }
        let per = t.elapsed().as_secs_f64() / (reps * positions.len()) as f64;
        println!("  features through {name}: {:.2} us", per * 1e6);
    }
    let params = search_params();
    let mut search = Search::new(&net);
    let t = Instant::now();
    let (mut sims, mut evals) = (0u64, 0u64);
    for st in positions.iter().take(300) {
        if let Root::Open = search.reset(&net, st) {
            for _ in 0..1000 {
                search.simulate(&net, st, &params);
            }
            sims += 1000;
            evals += search.evals as u64;
        }
    }
    println!(
        "search {:.2} us per simulation, {:.2} evaluations per simulation",
        t.elapsed().as_secs_f64() / sims as f64 * 1e6,
        evals as f64 / sims as f64
    );
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
        Sample::new(&st, 0).write(&mut samples, &[0; MAXP]);
        for x in logits.iter().chain(v.iter()) {
            outputs.extend_from_slice(&x.to_le_bytes());
        }
        for k in 0..MAXP {
            if k < feat.nslots {
                outputs.extend_from_slice(&feat.dist[k]);
            } else {
                outputs.extend_from_slice(&[UNREACH; CELLS]);
            }
        }
    }
    std::fs::write(format!("{prefix}.samples"), samples).unwrap();
    std::fs::write(format!("{prefix}.outputs"), outputs).unwrap();
    println!("wrote {n} positions to {prefix}.samples and {prefix}.outputs");
}

/// How long the primed space fill takes when it is given all the time it
/// wants, over games the network plays against itself.
fn cmd_fillbench() {
    let net = load_net(&arg("net").expect("--net"));
    let games: u32 = arg_or("games", 150);
    let sims: u32 = arg_or("sims", 16);
    let params = search_params();
    let mut search = Search::new(&net);
    let mut rng = Rng(arg_or("seed", 17));
    let mut times: Vec<(f64, i32)> = Vec::new();
    for g in 0..games {
        let mut st = random_start(&mut rng, 2 + (g % 3) as usize);
        while !st.over() {
            let me = st.to_move as usize;
            let mv = match search.reset(&net, &st) {
                Root::Over => unreachable!(),
                Root::Forced { mv, fill: false } => mv as usize,
                Root::Cutoff | Root::Forced { fill: true, .. } => {
                    let t = Instant::now();
                    let mut clock = Clock::new(t + Duration::from_secs(3600));
                    let d = primed_fill_until(&st, me, &mut search.sc, &mut clock);
                    times.push((t.elapsed().as_secs_f64() * 1e3, flood_count(&st, me)));
                    d as usize
                }
                Root::Open => {
                    for _ in 0..sims {
                        search.simulate(&net, &st, &params);
                    }
                    best_move(&search.root_result())
                }
            };
            st.play(mv);
        }
    }
    times.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let n = times.len();
    let over = |ms: f64| times.iter().filter(|t| t.0 > ms).count();
    println!(
        "{n} fills in {games} games: mean {:.2} ms, median {:.2}, 99th percentile {:.2}, max {:.2} ms (region of {} cells);          over 15 ms: {}, over 45 ms: {}, over 90 ms: {}",
        times.iter().map(|t| t.0).sum::<f64>() / n as f64,
        times[n / 2].0,
        times[n * 99 / 100].0,
        times[n - 1].0,
        times[n - 1].1,
        over(15.0),
        over(45.0),
        over(90.0)
    );
}

/// Random games, checking what the rest of the engine relies on.
fn cmd_check() {
    let mut rng = Rng(arg_or("seed", 5));
    let games: u32 = arg_or("games", 20000);
    let mut deaths_mid_game = 0u64;
    let mut plies = 0u64;
    for g in 0..games {
        let n = 2 + (g % 3) as usize;
        let mut st = random_start(&mut rng, n);
        while !st.over() {
            let me = st.to_move as usize;
            assert!(st.is_alive(me) && st.legal_mask(me) != 0, "unsettled position");
            let mut occ = [0u32; 20];
            for p in 0..n {
                for r in 0..20 {
                    assert!(occ[r] & st.trail[p][r] == 0, "trails overlap");
                    occ[r] |= st.trail[p][r];
                }
                let on_trail = st.trail[p][st.hy[p].max(0) as usize] >> st.hx[p].max(0) & 1 != 0;
                assert!(st.is_alive(p) == (st.rank[p] == 0), "rank of a live bike");
                assert!(!st.is_alive(p) || on_trail, "head off its trail");
                assert!(st.is_alive(p) || st.trail[p] == [0; 20], "dead bike left a trail");
            }
            assert!(occ == st.occ, "occupancy is not the union of the trails");
            let (slot, ns) = st.slots();
            assert!(ns == st.alive_count() as usize && slot[0] as usize == me);
            let before = st.alive_count();
            let mv = random_move(&st, &mut rng);
            st.play(mv);
            plies += 1;
            if !st.over() && st.alive_count() < before {
                deaths_mid_game += 1;
            }
        }
        let mut seen = [false; MAXP + 1];
        for p in 0..n {
            let r = st.rank[p] as usize;
            assert!(r >= 1 && r <= n && !seen[r], "placements are not a permutation");
            seen[r] = true;
        }
        assert!(st.rank[st.alive.trailing_zeros() as usize] == 1);
    }
    println!("{games} random games, {plies} plies, {deaths_mid_game} deaths before the end: all checks passed");
}

// --- CodinGame ---------------------------------------------------------------------

/// The CodinGame loop with weights from a file: `--packed FILE` is the text
/// `tools/make_paste.py` embeds in the paste, `--net FILE` a float32 file.
fn cmd_codingame() {
    let started = Instant::now();
    let net = match arg("packed") {
        Some(path) => Net::from_packed(std::fs::read_to_string(&path).expect("cannot read --packed").trim()),
        None => {
            let path = arg("net").or_else(|| std::env::var("TRONZERO_NET").ok()).unwrap_or_else(|| {
                let exe = std::env::current_exe().expect("no executable path");
                exe.with_file_name("net.bin").to_string_lossy().into_owned()
            });
            load_net(&path)
        }
    };
    play::run(&net, &search_params(), started);
}

// --- Against the alpha-beta engine, in process -------------------------------------

/// `legacy/alpha-beta/src/main.rs`, compiled in unchanged (`build.sh` copies it
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

        /// The engine's move for player `me`.
        #[allow(clippy::too_many_arguments)]
        pub fn choose(
            &mut self,
            occ: &[u32; 20],
            trail: &[[u32; 20]; 4],
            hx: [i8; 4],
            hy: [i8; 4],
            alive: u8,
            players: u8,
            me: usize,
            budget_ms: u64,
        ) -> u8 {
            let mut st = State::new(players);
            st.occupied.bits = *occ;
            for p in 0..players as usize {
                if alive >> p & 1 != 0 {
                    st.trail[p].bits = trail[p];
                    st.head_x[p] = hx[p];
                    st.head_y[p] = hy[p];
                }
            }
            st.alive_mask = alive;
            let d = choose_move(&mut st, me, self.last_dir, budget_ms, &mut self.scratch);
            self.last_dir = d;
            d
        }
    }
}

/// A spawn followed by `seed_plies` random legal moves per bike, as the SPRT
/// referee opens its games. `None` if a bike was lost on the way.
fn random_opening(rng: &mut Rng, n: usize, seed_plies: u32) -> Option<State> {
    let mut st = random_start(rng, n);
    for _ in 0..n as u32 * seed_plies {
        if st.alive_count() < n as u32 {
            return None;
        }
        let mv = random_move(&st, rng);
        st.play(mv);
    }
    if st.alive_count() < n as u32 {
        return None;
    }
    Some(st)
}

fn cmd_vsbase() {
    let net = load_net(&arg("net").expect("--net"));
    let pairs: u32 = arg_or("pairs", 100);
    let n: usize = arg_or("players", 2);
    // Seats the agent takes; the alpha-beta engine takes the rest.
    let agents: usize = arg_or("agents", 1);
    assert!((2..=4).contains(&n) && agents >= 1 && agents < n);
    let threads: usize = arg_or("threads", 8);
    let seed: u64 = arg_or("seed", 99);
    let budget_ms: u64 = arg_or("budget-ms", 20);
    let seed_plies: u32 = arg_or("seed-plies", 4);
    let params = search_params();
    let next = AtomicU32::new(0);
    let total = Mutex::new(Tally::default());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let mut search = Search::new(&net);
                let mut feat = Box::new(Feat::new());
                let mut local = Tally::default();
                loop {
                    let g = next.fetch_add(1, Ordering::Relaxed);
                    if g >= pairs {
                        break;
                    }
                    let mut rng = Rng(seed.wrapping_mul(0xD6E8_FEB8_6659_FD93) ^ ((g as u64) << 20));
                    rng.next_u64();
                    let start = loop {
                        if let Some(st) = random_opening(&mut rng, n, seed_plies) {
                            break st;
                        }
                    };
                    let mut score = 0.0;
                    // The same opening with the agent starting from every seat.
                    for first in 0..n {
                        let mut side = [1u8; MAXP];
                        for i in 0..agents {
                            side[(first + i) % n] = 0;
                        }
                        let mut st = start;
                        let mut base: Vec<baseline::Player> = (0..n).map(|_| baseline::Player::new()).collect();
                        // Whether the agent had the larger territory when the game came down to two.
                        let mut final_ahead: Option<bool> = None;
                        let mut final_seen = false;
                        while !st.over() {
                            let me = st.to_move as usize;
                            if n > 2 && agents == 1 && !final_seen && st.alive_count() == 2 {
                                final_seen = true;
                                if st.is_alive(first) {
                                    features(&st, &mut feat);
                                    let (mut mover, mut other) = (0, 0);
                                    for i in 0..CELLS {
                                        let (a, b) = (feat.dist[0][i], feat.dist[1][i]);
                                        if a != UNREACH && a <= b {
                                            mover += 1;
                                        } else if b != UNREACH {
                                            other += 1;
                                        }
                                    }
                                    final_ahead = Some(if me == first { mover > other } else { other > mover });
                                }
                            }
                            let d = if side[me] == 0 {
                                let deadline = Instant::now() + Duration::from_millis(budget_ms);
                                let (d, sims, _) = timed_move(&net, &mut search, &st, &params, Limit::Until(deadline));
                                if sims > 0 {
                                    local.searches += 1;
                                    local.sims += sims as u64;
                                }
                                d
                            } else {
                                base[me].choose(&st.occ, &st.trail, st.hx, st.hy, st.alive, st.n, me, budget_ms)
                            };
                            if d >= 4 || st.legal_mask(me) >> d & 1 == 0 {
                                st.forfeit(); // an illegal move is a crash
                            } else {
                                st.play(d as usize);
                            }
                            local.plies += 1;
                        }
                        score += tally_game(&mut local, &st, &side);
                        if let Some(ahead) = final_ahead {
                            let won = (st.rank[first] == 1) as u64;
                            local.finals[0] += 1;
                            local.finals[1] += ahead as u64;
                            local.finals[if ahead { 2 } else { 3 }] += won;
                        }
                    }
                    local.scores.push(score / n as f64);
                }
                total.lock().unwrap().merge(&local);
            });
        }
    });
    let t = total.lock().unwrap();
    println!(
        "{{\"players\": {n}, \"agents\": {agents}, \"budget_ms\": {budget_ms}, \"win_rate_at_par\": {:.4}, \
         \"sims_per_search\": {:.0}, \"finals\": {}, \"finals_ahead\": {}, \"finals_ahead_won\": {}, \
         \"finals_behind_won\": {}, {}}}",
        agents as f64 / n as f64,
        t.sims as f64 / t.searches.max(1) as f64,
        t.finals[0],
        t.finals[1],
        t.finals[2],
        t.finals[3],
        t.json()
    );
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("selfplay") => cmd_selfplay(),
        Some("match") => cmd_match(),
        Some("bench") => cmd_bench(),
        Some("dump") => cmd_dump(),
        Some("check") => cmd_check(),
        Some("fillbench") => cmd_fillbench(),
        Some("openings") => cmd_openings(),
        Some("vsbase") => cmd_vsbase(),
        _ => cmd_codingame(),
    }
}
