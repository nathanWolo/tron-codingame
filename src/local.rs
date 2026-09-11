//! Local `--bench` / `--profile` helpers. Not part of the CodinGame paste.
//!
//! SPRT lives in `tools/sprt.py`; this module is dummy-opponent self-play and
//! eval throughput. Enabled by the `local` feature (on by default) and by
//! `cargo test`.
use super::*;
use std::sync::atomic::Ordering;

// ---------------------------------------------------------------------------
// Local self-play (`--bench`) so we can measure the agent before submitting.
// ---------------------------------------------------------------------------

/// Tiny xorshift64* RNG for local games (no `rand` crate; CodinGame is std-only).
/// Used only by `--bench` / `--profile` spawns and the random dummy opponent.
pub(crate) struct XorShift {
    state: u64,
}

impl XorShift {
    /// Seed must be non-zero; we OR in 1 so `new(0)` still works.
    /// [`bench`] / [`random_start`] / [`play_game`] each construct their own.
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    /// Advance the xorshift64* state and return the new 64-bit value.
    /// Seeds [`random_start`] games and picks a random legal move for `bot_random`.
    pub(crate) fn next_u64(&mut self) -> u64 {
        let mut bits = self.state;
        bits ^= bits << 13;
        bits ^= bits >> 7;
        bits ^= bits << 17;
        self.state = bits;
        bits
    }

    /// Uniform integer in `0..modulus`. Fine for local benches, not crypto.
    /// Spawn cells (`0..WIDTH`, `0..HEIGHT`) and random-bot move indices.
    pub(crate) fn gen_range(&mut self, modulus: u32) -> u32 {
        (self.next_u64() as u32) % modulus
    }
}

/// Place `player_count` bikes on unique random cells (CodinGame spawn rule).
/// `--bench` and `--profile` use this; SPRT uses the JSON opening books instead.
pub(crate) fn random_start(rng: &mut XorShift, player_count: usize) -> State {
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
pub(crate) fn bot_random(state: &State, player: usize, rng: &mut XorShift) -> Option<u8> {
    let (legal, move_count) = state.legal_moves(player);
    if move_count == 0 {
        None
    } else {
        Some(legal[rng.gen_range(move_count as u32) as usize])
    }
}

/// Greedy one-ply policy: try each legal step and pick the one that maximises
/// remaining flood-fill size plus a small wall-hug bonus.
///
/// Dummy opponent for `--bench` / `--profile`, and the greedy-play test.
pub(crate) fn greedy_direction(state: &State, player: usize, scratch: &mut Scratch) -> Option<u8> {
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
/// Walks 12 greedy plies first so we profile a still-connected midgame.
///
/// **Where:** [`main`] when argv contains `--profile`. Use this to see whether
/// an eval change helped nodes/sec (Voronoi is the usual bottleneck).
fn profile() {
    let mut rng = XorShift::new(42);
    let mut state = random_start(&mut rng, 2);
    let mut scratch = Scratch::new();
    for _ in 0..12 {
        for player in 0..2 {
            if let Some(dir) = bot_greedy(&state, player, &mut scratch) {
                state.apply(player, dir as usize);
            }
        }
    }
    {
        let empty = empty_rows(&state);
        let sa = head_seed(&state, 0, &empty);
        let sb = head_seed(&state, 1, &empty);
        let n = 300000;
        let t = Instant::now();
        let mut acc = 0i32;
        for i in 0..n {
            let f = flood_mask(&std::hint::black_box(sa), &empty);
            acc ^= f[i % 20] as i32;
        }
        eprintln!("flood_mask: {:.1}/ms {acc}", n as f64 / t.elapsed().as_secs_f64() / 1000.0);
        let t = Instant::now();
        for i in 0..n {
            let d = duel_voronoi(&empty, std::hint::black_box(sa), sb);
            acc ^= d.owned_a[i % 20] as i32;
        }
        eprintln!("duel_voronoi: {:.1}/ms {acc}", n as f64 / t.elapsed().as_secs_f64() / 1000.0);
        let t = Instant::now();
        for i in 0..n {
            let v = compute_voronoi(std::hint::black_box(&state));
            acc ^= v.owned[0][i % 20] as i32;
        }
        eprintln!("compute_voronoi: {:.1}/ms {acc}", n as f64 / t.elapsed().as_secs_f64() / 1000.0);
        let t = Instant::now();
        for i in 0..n {
            let e = mask_edge_sum(&std::hint::black_box(sa), &empty);
            acc ^= e + i as i32;
        }
        eprintln!("edge_sum: {:.1}/ms {acc}", n as f64 / t.elapsed().as_secs_f64() / 1000.0);
    }
    let eval_started = Instant::now();
    let eval_count = 200000;
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

    let mut state4 = random_start(&mut rng, 4);
    for _ in 0..8 {
        for player in 0..4 {
            if let Some(dir) = bot_greedy(&state4, player, &mut scratch) {
                state4.apply(player, dir as usize);
            }
        }
    }
    let eval4_started = Instant::now();
    let mut checksum4 = 0i32;
    for _ in 0..eval_count {
        checksum4 ^= eval_ffa(&state4, 0, 0);
    }
    let eval4_time = eval4_started.elapsed();
    eprintln!(
        "eval_ffa: {} in {:?} ({:.1}/ms) checksum {checksum4}",
        eval_count,
        eval4_time,
        eval_count as f64 / eval4_time.as_secs_f64() / 1000.0
    );
    let choose4_started = Instant::now();
    let dir4 = choose_move(&mut state4, 0, NO_MOVE, TURN_BUDGET_MS, &mut scratch);
    eprintln!(
        "choose_move 4p {} in {:?}",
        DIR_NAME[dir4 as usize],
        choose4_started.elapsed()
    );
}

/// `--fill-eval`: read boards from stdin and print fill estimates.
///
/// Each board is 20 lines of 30 chars: `.` empty, `#` wall/trail, `A` head of
/// player 0, `B` head of player 1. Prints per player:
/// `flood approx_fill checkerboard greedy_steps`.
fn fill_eval() {
    use std::io::BufRead;
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let mut scratch = Scratch::new();
    loop {
        let mut state = State::new(2);
        let mut rows = 0;
        let mut heads = [(-1i32, -1i32); 2];
        while rows < 20 {
            let line = match lines.next() {
                Some(Ok(text)) => text,
                _ => return,
            };
            if line.trim().is_empty() {
                continue;
            }
            for (col, ch) in line.chars().take(30).enumerate() {
                match ch {
                    '#' => state.occupied.set(col as i32, rows as i32),
                    'A' => heads[0] = (col as i32, rows as i32),
                    'B' => heads[1] = (col as i32, rows as i32),
                    _ => {}
                }
            }
            rows += 1;
        }
        for player in 0..2 {
            if heads[player].0 >= 0 {
                state.occupy(player, heads[player].0, heads[player].1);
            }
        }
        let mut out = Vec::new();
        for player in 0..2 {
            if !state.is_alive(player) {
                out.push("0 0 0 0".to_string());
                continue;
            }
            let fl = flood_count(&state, player, &mut scratch);
            let af = approx_fill(&state, player, &mut scratch);
            let cb = checkerboard_reach_bound(&state, player);
            // Greedy rollout exactly like the separated root policy.
            let mut best = 0;
            let (moves, count) = state.legal_moves(player);
            for i in 0..count {
                let mut sim = state;
                sim.apply(player, moves[i] as usize);
                let mut extra = 1;
                let mut prev = moves[i];
                while extra < 600 {
                    match fill_direction(&mut sim, player, prev, &mut scratch) {
                        Some(dir) if sim.is_legal_dir(player, dir as usize) => {
                            sim.apply(player, dir as usize);
                            prev = dir;
                            extra += 1;
                        }
                        _ => break,
                    }
                }
                best = best.max(extra);
            }
            out.push(format!("{fl} {af} {cb} {best}"));
        }
        println!("{}", out.join(" | "));
    }
}

/// Handle `--bench` / `--profile`. Returns true if this process should exit.
pub(crate) fn run() -> bool {
    if std::env::args().any(|arg| arg == "--bench") {
        bench();
        true
    } else if std::env::args().any(|arg| arg == "--fill-eval") {
        fill_eval();
        true
    } else if std::env::args().any(|arg| arg == "--profile") {
        profile();
        true
    } else {
        false
    }
}
