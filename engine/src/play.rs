//! Play under a time limit: the agent's move for one position, and the
//! CodinGame stdin/stdout loop. This file is part of the CodinGame paste.

use crate::board::*;
use crate::mcts::{best_move, Params, Root, Search};
use crate::net::Net;
use std::io::{BufRead, Write};
use std::time::{Duration, Instant};

/// How long to think: until a deadline, or a fixed number of simulations (so
/// that two builds can be compared move for move).
#[derive(Clone, Copy)]
pub enum Limit {
    Until(Instant),
    Sims(u32),
}

/// The agent's move: a forced move at once, the primed greedy fill when it is
/// alone in its region, the search otherwise. Returns the direction, the
/// simulations run and the root value of the move.
pub fn timed_move(net: &Net, search: &mut Search, st: &State, params: &Params, limit: Limit) -> (u8, u32, f32) {
    let me = st.to_move as usize;
    let mut clock = Clock::new(match limit {
        Limit::Until(t) => t,
        Limit::Sims(_) => Instant::now() + Duration::from_secs(3600),
    });
    match search.reset(net, st) {
        Root::Over => (0, 0, 0.0),
        Root::Forced { mv, fill: false } => (mv, 0, 0.0),
        Root::Cutoff | Root::Forced { fill: true, .. } => {
            (primed_fill_until(st, me, &mut search.sc, &mut clock), 0, 0.0)
        }
        Root::Open => {
            let mut sims = 0u32;
            loop {
                let go = match limit {
                    Limit::Until(_) => clock.ok(),
                    Limit::Sims(n) => sims < n,
                };
                if !go {
                    break;
                }
                search.simulate(net, st, params);
                sims += 1;
            }
            let r = search.root_result();
            let dir = best_move(&r);
            (dir as u8, sims, r.child_q[dir])
        }
    }
}

/// A setting from `--name value` on the command line, else from the
/// environment variable `env`. CodinGame passes neither.
fn setting(name: &str, env: &str) -> Option<u64> {
    let args: Vec<String> = std::env::args().collect();
    let flag = format!("--{}", name);
    let from_args = args.iter().position(|a| *a == flag).and_then(|i| args.get(i + 1).cloned());
    from_args.or_else(|| std::env::var(env).ok()).and_then(|v| v.parse().ok())
}

/// The CodinGame loop: read a frame, keep the position up to date, print a
/// move. `started` is when the process began; whatever start-up took is
/// taken off the first turn's budget.
pub fn run(net: &Net, params: &Params, started: Instant) {
    let budget = setting("budget-ms", "TRON_BUDGET_MS");
    let sims = setting("sims", "TRON_SIMS");
    let mut search = Search::new(net);
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let setup = started.elapsed();
    // Every frame gives each bike's head; a bike that is out reports -1 and
    // its trail leaves the board.
    let mut st = State::new(&[]);
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
        if first {
            st.n = n as u8;
            st.alive = (1u8 << n) - 1;
        }
        let rows: Vec<Vec<i32>> = (0..n)
            .map(|_| lines.next().unwrap().unwrap().split_whitespace().map(|t| t.parse().unwrap()).collect())
            .collect();
        // Bikes that are out leave first: a bike still in may already stand on
        // a cell one of them vacated earlier in the round, and removing that
        // trail afterwards would erase the new head with it.
        for p in 0..n {
            if rows[p][0] < 0 {
                st.kill(p);
            }
        }
        for p in 0..n {
            let v = &rows[p];
            if v[0] < 0 {
                continue;
            }
            if first {
                st.place(p, v[0], v[1]);
            }
            let (px, py) = (st.hx[p] as i32, st.hy[p] as i32);
            if (px, py) != (v[2], v[3]) {
                if let Some(d) = (0..4).find(|&d| (px + DIR_X[d], py + DIR_Y[d]) == (v[2], v[3])) {
                    st.last[p] = d as u8;
                }
                st.place(p, v[2], v[3]);
            }
        }
        st.to_move = my_id as u8;
        let mut dir = 0u8;
        if !st.over() && st.is_alive(my_id) && st.legal_mask(my_id) != 0 {
            let limit = match (sims, budget) {
                (Some(n), _) => Limit::Sims(n as u32),
                (None, Some(ms)) => Limit::Until(start + Duration::from_millis(ms)),
                (None, None) if first => {
                    let ms = Duration::from_millis(crate::FIRST_TURN_BUDGET_MS);
                    Limit::Until(start + ms.checked_sub(setup).unwrap_or(Duration::from_millis(1)))
                }
                (None, None) => Limit::Until(start + Duration::from_millis(crate::TURN_BUDGET_MS)),
            };
            let (d, done, q) = timed_move(net, &mut search, &st, params, limit);
            dir = d.min(3);
            if done > 0 {
                eprintln!(
                    "{} mm {} {}ms n={} d=0 a={}",
                    DIR_NAME[dir as usize],
                    (q * 1000.0) as i32,
                    start.elapsed().as_millis().max(1),
                    done,
                    st.alive_count()
                );
            }
            // paste:skip-begin
            // For the match recorder: the same line again with the tree's statistics, or what replaced the search.
            if std::env::var("TZ_STATS").is_ok() {
                if done > 0 {
                    eprintln!(
                        "{} mm {} {}ms n={} d=0 a={} {}",
                        DIR_NAME[dir as usize],
                        (q * 1000.0) as i32,
                        start.elapsed().as_millis().max(1),
                        done,
                        st.alive_count(),
                        search.stats_line(&st)
                    );
                } else {
                    let forced = st.legal_mask(my_id).count_ones() == 1;
                    eprintln!("{} tz mode={} a={}", DIR_NAME[dir as usize], if forced { "forced" } else { "fill" }, st.alive_count());
                }
            }
            // paste:skip-end
        }
        first = false;
        println!("{}", DIR_NAME[dir as usize]);
        let _ = std::io::stdout().flush();
    }
}
