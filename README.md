# Tron Battle agent (CodinGame)

A Rust bot for [CodinGame Tron Battle](https://www.codingame.com/multiplayer/bot-programming/tron-battle). The whole agent is a **single file** (`src/main.rs`) so it can be pasted into the CodinGame IDE.

The bot plays light-cycle Tron on a 30×20 grid. In 1v1 it uses iterative-deepening minimax with a Voronoi territory evaluation. Once the two bikes can no longer reach each other, it switches to a survival / space-filling policy. Matches with 3–4 players use a shallower greedy model of the others.

---

## Game

You drive a light cycle. Each turn you output `UP`, `DOWN`, `LEFT`, or `RIGHT`. The cycle always moves one cell, leaves a solid trail, and dies if it hits a trail, a wall, or leaves the board. Last cycle alive wins.

### Rules that actually matter for an AI

- **Grid:** 30 wide × 20 high. Each player starts at a **random location** (CodinGame’s statement). There is no smaller official opening set: any pair of distinct cells among the 600 is a legal 1v1 spawn (600×599 ordered pairs). The example in the statement is `(9,5)` vs `(10,7)`.
- **Players:** 2 to 4. CodinGame’s statement lists `2 ≤ N ≤ 4`; some league text shows `N = 2` only. This bot supports both.
- **Turns are sequential, not simultaneous.** Player 0 moves, then 1, then 2, … wrapping around live players. This is different from the 2010 Google AI Challenge (simultaneous Tron). Head-on “both enter the same cell” collisions do not happen; the player who moves second cannot occupy the cell the first player just took.
- **Input does not send the full trail.** Each turn you get, per player, `(X0, Y0, X1, Y1)`: the *original* start (tail) and the *current* head. You must remember every head you have ever seen. When a player dies, all four values are `-1` and **their entire ribbon disappears**.
- **Time:** under 100 ms per turn. This bot budgets **85 ms** on the first turn and **75 ms** after that, so there is margin for CodinGame’s slower judges.
- **Invalid / late / crashing output loses.**

Example (2 players, we are player 0 at `(9,5)`, opponent at `(10,7)`):

```
2 0
9 5 9 5
10 7 10 7
```

A legal first move is any of `UP` `DOWN` `LEFT` `RIGHT` that stays on the board. This bot typically steps toward the opponent (often `DOWN` from that start).

---

## Repository

```
tron_codingame/
  Cargo.toml          # release: opt-level 3, LTO, abort-on-panic
  src/main.rs         # entire agent + local bench/profile
  tools/sprt.py           # SPRT referee (baseline vs dev binaries)
  tools/openings_1v1.json # frozen 2-player CG spawns
  tools/openings_3p.json
  tools/openings_4p.json
  tools/save_baseline.sh
  tools/sprt.sh
  README.md
```

No crates.io dependencies. CodinGame compiles a single Rust file with `std` only.

### Run locally

```bash
cargo run --release                 # CodinGame stdin/stdout loop
cargo run --release -- --bench      # self-play vs baseline bots
cargo run --release -- --profile    # eval throughput + one timed search
```

Release profile is aggressive because the same file is what you submit:

```toml
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"
```

### Submit to CodinGame

1. Open Tron Battle, language **Rust**.
2. Paste `src/main.rs`.
3. Play a few IDE games. Stderr lines look like `DOWN mm 0 75ms` (direction, leftover score, milliseconds used). The middle number is the FFA 2-ply score, or `0` in 1v1 (iterative deepening does not write it).
4. If you timeout, lower `TURN_BUDGET_MS` / `FIRST_TURN_BUDGET_MS` at the top of the file.

`--bench` and `--profile` never run on CodinGame (no CLI args there).

---

## SPRT (testing a change against a frozen baseline)

Use a Sequential Probability Ratio Test, the same style of stop-when-you-know used for chess-engine patches (fishtest / OpenBench). Two **compiled binaries** play each other over the real CodinGame stdin/stdout protocol.

An **N-player** game is **1 candidate vs N−1 copies of the baseline**. Games are played in **seat-rotated blocks** on the same spawn (colour-swap when N=2) so first-move advantage cancels. If every engine is equal, the candidate’s win rate is **1/N**. Elo 0 is that prior; LOS is P(dev is stronger than the equal field).

### Openings

CodinGame does **not** ship a named opening book. Starts are uniform random unique cells on the 30×20 grid. SPRT therefore:

1. Picks a spawn from `tools/openings_1v1.json` / `openings_3p.json` / `openings_4p.json` — 256 frozen N-tuples sampled from that distribution (2p entry 0 is the statement example `(9,5)` / `(10,7)`).
2. Plays `--seed-plies` (default **4**) referee-chosen random legal moves per side. Engines still receive every ply so their trail tracker stays in sync; the referee applies the seeded walk instead of the engine’s choice. Seat-rotated games reuse the same spawn and the same walk.

Pass `--book none` for a fresh uniform spawn every block, or `--seed-plies 0` to start from the spawn cells with no extra walk.

```bash
tools/sprt.sh --seed-plies 6
tools/sprt.sh --book none --seed-plies 0   # raw random spawns
tools/sprt.sh --players 3                  # 1v2 FFA
tools/sprt.sh --players 4                  # 1v3 FFA
```

### Workflow

```bash
# 1. Freeze the current bot as the reference
tools/save_baseline.sh          # cargo build --release → bin/tron-baseline

# 2. Change the agent in src/main.rs

# 3. Build the candidate and run SPRT
tools/sprt.sh                   # default: H0=0 Elo, H1=+10 Elo, α=β=0.05, 20 ms/turn
```

`tools/sprt.sh` rebuilds `target/release/tron` as **dev** and pits it against `bin/tron-baseline`. Extra flags are forwarded to `tools/sprt.py`.

### Hypotheses

| | meaning |
|---|---|
| **H0** (`--elo0`, default 0) | Dev is this many Elo vs an equal field (0 = win rate 1/N). |
| **H1** (`--elo1`, default 10) | Dev is this many Elo stronger than the field. |
| **Accept H1** (exit 0) | The patch gained about `elo1` (or more). Ship it. |
| **Accept H0** (exit 1) | No such gain. Revert or try again. |
| **Inconclusive** (exit 2) | Hit `--max-games` without crossing the bounds. |

The test uses a **GSPRT** on per-game scores in `[0, 1]` (win = 1, loss = 0, k-way first-place tie = 1/k). Expected score uses Plackett-Luce: `1 / (1 + (N−1) · 10^(−Elo/400))`, which is ordinary logistic Elo when N=2 and equals **1/N** at Elo 0. Variance is estimated from the actual scores. Bounds are `ln(β/(1-α))` and `ln((1-β)/α)` (≈ ±2.94 at 5%).

Useful variants:

```bash
# Stricter “is it even +5 Elo?”
tools/sprt.sh --elo0 0 --elo1 5

# No-regression: fail if the patch is −5 Elo or worse
tools/sprt.sh --elo0 -5 --elo1 0

# Just play 200 games and print Elo (no sequential stop)
tools/sprt.sh --fixed --max-games 200

# 3-player FFA (1 candidate vs 2 baselines); fair prior is 33.3%
tools/sprt.sh --players 3 --fixed --max-games 120

# More search (closer to CodinGame, slower)
tools/sprt.sh --budget-ms 40 --concurrency 4
```

Direct invocation if you already have two binaries:

```bash
python3 tools/sprt.py \
  --baseline bin/tron-baseline \
  --dev target/release/tron \
  --players 2 \
  --elo0 0 --elo1 10 \
  --budget-ms 20 \
  --concurrency 4 \
  --max-games 2000 \
  --log sprt-logs/run.jsonl
```

Both engines get `--budget-ms N` and `TRON_BUDGET_MS=N` so they use a fixed time (not the 75 ms CodinGame budget). The referee kills a side that fails to move within `budget-ms/1000 + 0.25` seconds, outputs an illegal direction, crashes, or walks into a wall — that side loses, and on death its trail is cleared, matching CodinGame.

Status line:

```
n=  80  44-2-34  56.2% vs 50.0%  Elo  +22.3 ±  39.1  LOS  72.4%  LLR  +0.81 [-2.94,+2.94]  SPRT[0,10] 2p
```

The percentage is the candidate’s mean score against the fair **1/N** prior. `LOS` is P(dev Elo > 0 | equal field), i.e. P(true win rate > 1/N), from a normal approximation. Do not ship on LOS alone; wait for SPRT to accept H1.

Logs (`--log`) are JSONL, one seat-rotated block per line. Binaries and logs: `bin/tron-*` and `sprt-logs/` are gitignored.

---

## Architecture

```
stdin turn
    │
    ▼
Tracker          reconstructs occupancy from successive (tail, head) pairs
    │
    ▼
choose_move
    ├─ 0 or 1 legal move → play it
    ├─ 3+ alive (FFA)
    │     2-ply: we move, others reply with oneply_direction
    │     score eval_ffa (no deep search)
    ├─ 1v1, bikes still share space
    │     iterative-deepening alpha-beta from the first ordered move
    │     (no 2-ply Voronoi warmup)
    └─ 1v1, bikes cut off (endgame)
          try each first move, then greedy-fill up to 80 steps
          pick the first move that survives longest
    │
    ▼
stdout: UP|DOWN|LEFT|RIGHT
```

`State` is `Copy` (~400 bytes). Search applies a move, recurses, then `undo_step`s. There is no heap allocation on the hot path except a couple of tiny `Vec`s at the root of `choose_move`.

Reusable BFS buffers live in `Scratch` (distance maps, queue, visit stamps) so evaluation does not allocate.

---

## Board representation

Occupancy is a **row bitboard**: 20 rows × `u32`, bits 0–29 used.

```text
occupied.bits[row] & (1 << col)  → cell (col, row) is a wall or trail
```

Each player also has their own `trail: [RowBits; 4]` so death can XOR that player’s ribbon out of `occupied` in one pass (CodinGame: dead player’s light disappears).

Heads are `i8` coordinates. Alive flags are a 4-bit mask. Directions:

| index | name  | (dx, dy) |
|------:|-------|----------|
| 0     | UP    | (0, -1)  |
| 1     | DOWN  | (0, +1)  |
| 2     | LEFT  | (-1, 0)  |
| 3     | RIGHT | (+1, 0)  |

You cannot reverse into your own trail: the cell you just left is occupied, so the reverse direction is illegal automatically.

---

## Trail reconstruction (`Tracker`)

CodinGame never sends the polyline, only start and current head.

**First turn**

- Occupy `(X0, Y0)` for every live player.
- If they already moved (you are not player 0), also occupy `(X1, Y1)`.

**Later turns**

- If a player’s four coords are `-1`, `kill` them: XOR their stored trail out of `occupied`.
- Otherwise, if `(X1, Y1)` differs from the head we stored, occupy that new cell.

We do **not** apply our own printed move to the tracker. The next input already includes our new head. Search mutates a copy/`&mut State` and undoes before returning, so the tracker stays consistent.

This is sequential, so between our turns each other player moves **once**. Seeing only their latest head is enough; we never miss intermediate cells.

---

## Algorithms

The design follows the 2010 Google AI Challenge Tron literature (especially a1k0n’s post-mortem and the Voronoi / wall-hug folklore), adapted to CodinGame’s sequential turns and vanishing trails.

### 1. Voronoi territory

For each live player, BFS from the **empty neighbours of their head** (the head cell itself is occupied). That gives a distance map over empty cells.

A cell belongs to a player if they are **strictly closest**. Ties are contested and owned by nobody. We also count:

- **territory** — cells owned
- **edges** — sum of empty 4-neighbours of owned cells (a1k0n: leftover open edges in your region; occupying a wall-adjacent cell leaves more edges, which induces wall-hugging when scores are close)
- **reach** — empty cells you can reach at all, even if the opponent is closer
- **still_connected** — true if any empty cell is reachable by two or more players
- **battlefront** — owned cells adjacent to opponent-owned cells

Open-game evaluation (1v1, still interacting):

```text
score = 50*(my_terr - opp_terr)
      + 12*(my_edges - opp_edges)
      +  3*(my_reach - opp_reach)
      +  6*(my_mobility - opp_mobility)
      +  4*battlefront
      + center_weight * (-manhattan to (14,9))
```

Center preference fades as the board fills (`center_weight = (500 - occupied) / 80`).

### 2. Sharing space vs. endgame

Two tests, either is enough to stay in “open” mode (we would rather keep fighting than ignore an opponent we can still reach):

1. **`shares_space`** — heads are orthogonally adjacent, **or** BFS from us reaches an empty neighbour of their head.
2. **`voronoi.still_connected`** — some empty cell is reachable by both.

Only if **both** say we are cut off do we enter endgame. An early-game sanity counter (`EARLY_SEPARATION_COUNT`) checks that we never mark “separated” while fewer than 24 cells are occupied; in benches that counter stayed at 0.

Once cut off, each player’s remaining life is independent. The one who can occupy more of their pocket wins.

### 3. Approximate fillable space (`approx_fill`)

Flood-fill size is an **upper bound** on survival: branches behind a choke cannot all be used, because you cannot return through your own trail.

`approx_fill` DFS-walks empty cells from the head. At a cell with several unvisited neighbours:

- if the cell is a **local cut** (empty 4-neighbours are not 4-connected through the 8-ring around the cell), take `1 + max(branch)`
- otherwise take `1 + sum(branches)` (same chamber)

A 1-wide loop still works: the first branch paints almost the whole loop; the other branch is tiny; `max` is nearly the full loop.

This is a cheap stand-in for the “tree of chambers” / articulation-point ideas from a1k0n and Iouri.

### 4. Greedy fill (`fill_direction`)

Used as the endgame policy. For each legal step, score:

```text
80 * approx_fill
+ 30 * flood_remaining
+ 12 * wall_neighbours(destination)
+  8 if continuing straight
- 400 * (cells lost beyond the one we just took)
```

The last term is the important one: if you walk past a 1-cell pocket, `flood_remaining` drops by more than 1 and the move is punished. That forces you to take side pockets before committing to a corridor.

### 5. Primed greedy endgame search

Finding a true longest path is NP-complete. a1k0n’s practical trick: **try each first move, then run the greedy filler to completion, pick the first move that “primes” the longest greedy life**.

We simulate up to 80 greedy steps after each candidate. Score = `50 * extra_steps + wall_hug`. That is the move we play when `separated`.

---

## Search (open 1v1)

### Iterative deepening + alpha-beta (negamax)

There is no 2-ply Voronoi warmup in 1v1 (it was SPRT-neutral and ate budget). Iterative deepening starts from the first move in `order_moves` so search gets the full remaining time:

```text
for depth = 1 ..= 16:
    search every root move with negamax(depth-1)
    if the iteration finishes before the deadline, keep its best move
    if it times out, discard this ply and use the previous depth
    if a mate score appears, stop
```

If even depth 1 cannot finish, we play that first ordered move. Root window: later moves are searched with beta = `-best_so_far` (fail-low pruning).

Leaf / no-move handling is sequential-correct:

- Player to move with 0 legal moves dies immediately (`±MATE_SCORE ± ply`).
- Eval is always from **our** perspective, then negated if it is the opponent’s turn (standard negamax).

`MATE_SCORE = 1_000_000`. Closer mates score slightly higher (`MATE_SCORE - ply`).

### Move ordering

Insertion-sort of at most 4 moves, keys:

1. Previous iteration’s PV move (`+10000`)
2. Killer move at this ply (`+3000`)
3. Continue straight (`+40`)
4. Wall-hug (occupied neighbours × 8)
5. Toward the opponent (minus Manhattan)

Killers: two slots per ply, updated on beta cutoffs.

### Time

`Search` timestamps a deadline. Every 16 nodes it samples `Instant::now()`. On timeout, negamax returns a sentinel; every frame still undoes its move, so the real board is never left dirty. The incomplete depth is thrown away.

Budgets: 85 ms first turn, 75 ms later (CodinGame limit 100 ms). SPRT passes `--budget-ms` / `TRON_BUDGET_MS` so both engines use a shorter fixed time.

---

## Free-for-all (3–4 players)

Max-N on four players is too bushy for 75 ms.

When more than two bikes are alive:

- Do **not** run deep 1v1 minimax (it treats others as frozen walls and suicides into fights).
- Instead: try each of our moves, then let every other player in turn order play `oneply_direction` (or die and vanish). Score `eval_ffa + 20 * our_flood`.

`eval_ffa` prefers surviving with space:

```text
40 * our_reach + 25 * our_voronoi
- 10 * best_enemy_voronoi - 4 * best_enemy_reach
+ 20 * our_mobility
```

When only two remain, the game becomes the 1v1 path (minimax + endgame fill). Dead players’ trails are already gone, so the board opens up — that is unique to this CodinGame ruleset.

`main_opponent` is the live enemy whose head we can reach soonest (BFS), used if we ever 1v1-search in a multi-player setting.

---

## Evaluation poison we had to remove

An early version added **`±12_000` whenever the two bikes were separated**, on top of the fill difference.

That looks reasonable at a *root* position that is truly cut. Inside a deep search tree it is lethal: the opponent, according to the same eval, “cooperates” into a line where we are barely ahead after a fake cut (`51` vs `50` cells). The `12_000` spike outranks real territory. Minimax then plays to chase that hallucination and **loses to 1-ply Voronoi** (~12–40% in self-play).

The fix: when separated, score only

```text
80 * sign(fill_diff) + 60 * fill_diff + 5 * hug_diff + mobility_diff
```

A real winning cut (200 vs 10) still dwarfs open-game scores. A 50–49 cut is a small edge, not a fake mate. After this change, the same minimax beat 1-ply Voronoi **~67–81%**.

Blending 1-ply with deep scores, or capping depth at 1, did not fix it; the explosion in the leaf eval was the bug.

---

## Local testing

`--bench` plays 12 games per 1v1 matchup (colours swapped) plus 8 four-player games. Opponent bots:

| name | policy |
|------|--------|
| random | uniform legal move |
| wall-hug | most occupied neighbours |
| greedy-fill | max `flood * 20 + walls * 3` |
| voronoi-1ply | 1-ply `eval_1v1` (the usual strong greedy) |

Agent search in the bench is capped at **25 ms/turn** so a full run finishes in about a minute. CodinGame play uses 75–85 ms and searches deeper.

Representative results (25 ms, 12 games, noisy but directional):

| opponent | agent wins |
|----------|------------|
| random | ~90%+ |
| wall-hug | ~85–90% |
| greedy-fill | ~90%+ |
| voronoi-1ply | ~67–80% |
| FFA vs mixed (greedy / wall / voronoi) | weak (~12% in 8 games; 25% is par) |

`--profile` reports evals/ms (tens of evals per millisecond in release; Voronoi is the bottleneck) and one timed `choose_move`. Midgame 1v1 often reaches **depth 8–11** in the 75 ms box.

If both players are still alive after 900 rounds, the bench awards the larger flood fill (it used to default to player 0, which biased colour-swap stats).

---

## Implementation notes

- **No reverse rule needed** — own trail blocks it.
- **Apply / undo:** `occupy` sets occupied + trail + head; `undo_step` clears the *current* head cell and restores the previous head. Search only applies moves already known legal.
- **Visit stamps** instead of `memset` on flood fills (`Scratch::next_generation`).
- **Integer-only eval**, no transposition table. A Zobrist TT was tried and measured ~0 Elo (hits were almost all previous-iteration depth-misses; sequential Tron transposes rarely), so it was reverted. Four-wide branching plus alpha-beta is enough on a 600-cell board.
- **Debug** goes to stderr (`eprintln!`), which CodinGame shows in the IDE and ignores for scoring.
- Dead helpers (`search_endgame`, `search_ffa`, `rollout_score`, …) remain in the file under `#![allow(dead_code)]` from earlier experiments. They are unused by `choose_move` today.

---

## What to paste vs. what to keep

| path | CodinGame? |
|------|------------|
| `src/main.rs` | yes, entire file |
| `Cargo.toml` | no (their compiler) |
| `--bench` / `--profile` | harmless if pasted; they only run with argv |

---

## Ideas that would still help

1. **Faster Voronoi** — eval is the nodes-per-second bottleneck; a cheaper distance / territory pass would buy real depth.
2. **Wire `search_ffa`** into `choose_move` when 3+ are alive (the function exists but FFA currently uses only the 2-ply greedy).
3. **True articulation points (Tarjan) + chamber tree** instead of the local 8-ring cut test; this is the contest-winning eval idea from a1k0n (ignore non-battlefront chambers when you must choose).
4. **Checkerboard bound** on fillable space (surplus of one colour is unfillable).
5. **Quiescence / search extension** when heads are adjacent or a cut appears in one ply.
6. **FFA survival:** stay away from multi-enemy squeezes; currently the 4-player mode is much weaker than 1v1.
7. **Self-play tuning** of the open-game weights (`50 / 12 / 3 / 6 / 4`) against the voronoi-1ply baseline and against copies of itself.

---

## References

- CodinGame, *Tron Battle* — sequential turns, 30×20, vanishing trails.
- a1k0n, [Google AI Challenge post-mortem](https://www.a1k0n.net/2010/03/04/google-ai-postmortem.html) — Voronoi, wall-hug / least-edges, primed greedy endgame, tree of chambers.
- Google AI Challenge 2010 forum folklore — Voronoi as the standard open-game heuristic.
- Iouri / “tree of chambers” — fillable space as a tree split at articulation points, not raw flood fill.
- Waterloo CS Club Tron write-ups — Voronoi inside minimax, articulation points as cut targets.
