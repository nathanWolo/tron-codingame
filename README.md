# Tron Battle agent (CodinGame)

A Rust bot for [CodinGame Tron Battle](https://www.codingame.com/multiplayer/bot-programming/tron-battle). The **shipped agent is a single file** (`src/main.rs`, under the 100k-character IDE limit) so it can be pasted into the CodinGame IDE. Local bench, unit tests, SPRT, and the board watcher live beside it and are not part of the paste.

The bot plays light-cycle Tron on a 30×20 grid. In 1v1 it uses iterative-deepening minimax with a **row-bitboard Voronoi** territory evaluation. Once the two bikes can no longer reach each other, it switches to a survival / space-filling policy. Matches with 3–4 players use iterative-deepening paranoid search: we maximize a survival eval, the closest rival is a true Min, and other bikes play one greedy space-keeping reply.

---

## Game

You drive a light cycle. Each turn you output `UP`, `DOWN`, `LEFT`, or `RIGHT`. The cycle always moves one cell, leaves a solid trail, and dies if it hits a trail, a wall, or leaves the board. Last cycle alive wins.

### Rules that actually matter for an AI

- **Grid:** 30 wide × 20 high. Each player starts at a **random location** (CodinGame’s statement). There is no smaller official opening set: any pair of distinct cells among the 600 is a legal 1v1 spawn (600×599 ordered pairs). The example in the statement is `(9,5)` vs `(10,7)`.
- **Players:** 2 to 4. CodinGame’s statement lists `2 ≤ N ≤ 4`; some league text shows `N = 2` only. This bot supports both.
- **Turns are sequential, not simultaneous.** Player 0 moves, then 1, then 2, … wrapping around live players. This is different from the 2010 Google AI Challenge (simultaneous Tron). Head-on “both enter the same cell” collisions do not happen; the player who moves second cannot occupy the cell the first player just took.
- **Input does not send the full trail.** Each turn you get, per player, `(X0, Y0, X1, Y1)`: the *original* start (tail) and the *current* head. You must remember every head you have ever seen. When a player dies, all four values are `-1` and **their entire ribbon disappears**.
- **Time:** under 100 ms per turn. This bot budgets **95 ms**, so there is a little margin for CodinGame’s slower judges.
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
  Cargo.toml          # release: opt-level 3, LTO, abort-on-panic; default `local` feature
  src/main.rs         # CodinGame paste (agent only)
  src/local.rs        # `--bench` / `--profile` dummy-opponent self-play
  src/voronoi_tests.rs # `cargo test` (not pasted)
  tools/sprt.py           # SPRT referee (baseline vs dev binaries)
  tools/watch.py          # live board: candidate vs baseline (Windows + Linux)
  tools/openings_1v1.json # frozen 2-player CG spawns
  tools/openings_3p.json
  tools/openings_4p.json
  tools/save_baseline.sh
  tools/save_baseline.cmd     # Windows freeze of bin/tron-baseline.exe
  tools/sprt.sh
  tools/watch.sh          # Linux/macOS wrapper
  tools/watch.cmd         # Windows wrapper (cmd; not blocked by ExecutionPolicy)
  README.md
```

No crates.io dependencies. CodinGame compiles a single Rust file with `std` only. Paste **`src/main.rs` only** — it is under the 100k-character IDE limit. `src/local.rs` and `src/voronoi_tests.rs` stay in this repo.

### Run locally

```bash
cargo run --release                 # CodinGame stdin/stdout loop
cargo run --release -- --bench      # self-play vs dummy bots (src/local.rs)
cargo run --release -- --profile    # eval throughput + one timed search
cargo test --bin tron               # Voronoi / search tests
cargo test --bin tron --release     # plus exhaustive empty-board pair checks
tools/save_baseline.sh              # freeze target/release/tron → bin/tron-baseline
tools/save_baseline.cmd             # same freeze on Windows (tron-baseline.exe)
tools/sprt.sh                       # SPRT: candidate vs frozen baseline
tools/watch.sh                      # Linux/macOS: live 30×20 viewer
python tools/watch.py --build       # same viewer on Windows or Linux
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
2. Paste `src/main.rs` (not `local.rs` or the tests).
3. Play a few IDE games. Stderr lines look like `DOWN mm 1840 75ms n=12345 d=11 a=2` (direction, last completed search score, milliseconds used, nodes, completed ID depth, living bikes). Open 1v1 and FFA both print that leftover score; isolated fill / endgame do not.
4. If you timeout, lower `TURN_BUDGET_MS` / `FIRST_TURN_BUDGET_MS` at the top of the file.

`--bench` and `--profile` live in `src/local.rs` (Cargo feature `local`, on by default). The CodinGame judge passes no argv and compiles only the pasted file, so they never run there. Engine-vs-engine testing is `tools/sprt.sh`. `cargo build --release --no-default-features` is a CodinGame-like binary (agent loop only).

---

## Watch a game

`tools/watch.py` rebuilds (with `--build`) and opens a Tk window where the candidate plays `bin/tron-baseline` using the **same CodinGame protocol** as SPRT (seed plies, sequential turns, death clears the ribbon). It prints the host OS at startup (`Windows` / `Linux` / `macOS`), puts that name in the window title, and picks I/O accordingly: Windows cannot `select()` on pipes, so engine stdout/stderr are read on threads (same as SPRT). Bare `tron` paths are resolved to `tron.exe` on Windows.

```bash
# Linux / macOS
tools/watch.sh                 # 80 ms/turn (closer to CodinGame)
tools/watch.sh --budget-ms 20  # same think time as default SPRT
tools/watch.sh --players 3     # 1 candidate vs 2 baselines
```

```powershell
# Windows — PowerShell will not run .sh, and Restricted policy blocks .ps1
python tools\watch.py --build
.\tools\watch.cmd
.\tools\watch.cmd --budget-ms 20
python tools\watch.py --build --players 3
```

Either OS, if the binaries already exist:

```bash
python tools/watch.py --baseline bin/tron-baseline --dev target/release/tron
```

Cyan is the candidate, orange (and extra hues in FFA) is the baseline. Empty cells are tinted by Voronoi owner; contested cells are dull. The sidebar is **OPEN** while the bikes still share space (Voronoi leaf) and **CUT** when they are in separate chambers (fill eval, ×60 scale). Each engine’s last stderr line is shown (`DIR mm SCORE Tms n=…`). Older frozen baselines that do not print `n=` show `(no stderr stats)`.

| key / control | action |
|---------------|--------|
| Space | pause / resume |
| `N` | new random opening |
| `R` | replay the same spawn and seed walk |
| `S` | step one ply |
| `V` | toggle Voronoi overlay |
| Swap + replay | flip seats on the same opening |
| Auto next game | start another game when one ends |
| Candidate moves first | candidate is P0 (uncheck → P1) |
| budget / delay / seed plies | spinboxes in the sidebar |

On Linux it needs `DISPLAY` or `WAYLAND_DISPLAY`. On Windows it uses the desktop session (no `DISPLAY` variable).

---

## SPRT (testing a change against a frozen baseline)

Use a Sequential Probability Ratio Test, the same style of stop-when-you-know used for chess-engine patches (fishtest / OpenBench). Two **compiled binaries** play each other over the real CodinGame stdin/stdout protocol.

An **N-player** game is **1 candidate vs N−1 copies of the baseline**. Games are played in **seat-rotated blocks** on the same spawn (colour-swap when N=2) so first-move advantage cancels. If every engine is equal, the candidate’s win rate is **1/N**. Elo 0 is that prior; LOS is P(dev is stronger than the equal field).

### Openings

CodinGame does **not** ship a named opening book. Starts are uniform random unique cells on the 30×20 grid. SPRT therefore:

1. Picks a spawn from `tools/openings_1v1.json` / `openings_3p.json` / `openings_4p.json` — 256 frozen N-tuples sampled from that distribution (2p entry 0 is the statement example `(9,5)` / `(10,7)`).
2. Plays `--seed-plies` (default **4**) referee-chosen random legal moves **per side**. A ply is one step by one bike, so 1v1 with 4 seed plies is eight forced steps. Engines still receive every frame so their trail tracker stays in sync; the referee **ignores** their output and applies the canned walk. Seat-rotated games reuse the same spawn and the same walk.

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
# Windows: tools\save_baseline.cmd  → bin\tron-baseline.exe

# 2. Change the agent in src/main.rs

# 3. Build the candidate and run SPRT
tools/sprt.sh                   # default: H0=0 Elo, H1=+10 Elo, α=β=0.05, 20 ms/turn
```

`tools/sprt.sh` rebuilds `target/release/tron` as **dev** and pits it against `bin/tron-baseline`. Extra flags are forwarded to `tools/sprt.py`.

### Frozen baseline

`bin/tron-baseline` is gitignored and matches this `src/main.rs`: bitboard Voronoi, mate scores from the side to move, checkerboard fill bound, 1v1 quiescence (up to 4 extra plies) and aspiration windows (`±16000`), iterative deepening cap 50, primed greedy fill **to completion** after a 1v1 cut, mixed FFA (closest rival is Min; others play one greedy space-keeping reply). Freeze it with `tools/save_baseline.sh` or `tools/save_baseline.cmd`. New patches SPRT against that file; Elo 0 means “as strong as this freeze.”

### Earlier SPRTs (previous freezes)

Against the pre-bitboard cell-BFS freeze, after scoring 1v1 terminals from the side to move:

| match | decision | n | score | Elo |
|-------|----------|---|-------|-----|
| 1v1 SPRT `[0, 10]`, 20 ms | **ACCEPT H1** | 446 | 62.6% vs 50% | **+89.2 ± 17.0** |
| 4p SPRT `[0, 10]`, 20 ms (1 vs 3) | **ACCEPT H1** | 1004 | 31.7% vs 25% | **+57.3 ± 11.8** |

The same 1v1 SPRT **before** the mate-sign fix accepted H0 at about **−35 Elo**: extra depth was real, but a forced win was coming back as `-MATE`, so deeper search looked worse.

Against the freeze that already had bitboard Voronoi, mate-sign, and paranoid FFA, checkerboard fill plus 1v1 quiescence and aspiration at **90 ms** accepted H1 (`n=1180`, 55.3%, **+36.6 ± 10.2**). At 20 ms those 1v1 patches were inconclusive. FFA eval did not change; 4p at 20 ms stayed a coin-flip.

Against that freeze’s 80-step endgame cap, running primed greedy fill to completion at **90 ms** 1v1 accepted H1 (`n=8548`, 51.3%, **+9.3 ± 3.8**). NPS was unchanged (604k vs 604k); the gain is ranking large pockets correctly after a cut.

Against the same freeze, mixed FFA search (closest rival is Min; other bikes play one greedy flood reply) at **20 ms** 4p accepted H1 (`n=1108`, 31.3% vs 25%, **+54.4 ± 11.3**, NPS 0.9×). Full-coalition Min was too scared; a single Min that hunted us (ordered toward our head) was ~0 Elo. Letting distant bikes keep their own space is the mixed model that converted extra depth into Elo.

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

# 4-player FFA (1 candidate vs 3 baselines); fair prior is 25%
tools/sprt.sh --players 4

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
  --max-games 10000 \
  --log sprt-logs/run.jsonl
```

Both engines get `--budget-ms N` and `TRON_BUDGET_MS=N` so they use a fixed time (not the 75 ms CodinGame budget). The referee kills a side that fails to move within `budget-ms/1000 + 0.25` seconds, outputs an illegal direction, crashes, or walks into a wall — that side loses, and on death its trail is cleared, matching CodinGame.

Status line:

```
n=  80  44-2-34  56.2% vs 50.0%  Elo  +22.3 ±  39.1  LOS  72.4%  LLR  +0.81 [-2.94,+2.94]  SPRT[0,10] 2p  NPS 610k vs 98k (6.2x)
```

The percentage is the candidate’s mean score against the fair **1/N** prior. `LOS` is P(dev Elo > 0 | equal field), i.e. P(true win rate > 1/N), from a normal approximation. Do not ship on LOS alone; wait for SPRT to accept H1.

`NPS` is search nodes per second, summed over turns that reported `n=` on stderr (the agent prints `DIR mm score Tms n=NODES` after iterative deepening). Old baselines that do not print `n=` show as `n/a` until you freeze a new one with `tools/save_baseline.sh`. The ratio is dev/baseline; >1 means the candidate is searching more nodes in the same budget.

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
    │     iterative-deepening paranoid search (cap 50 our-plies)
    │     closest rival is Min; others play one greedy space-keeping reply
    │     (no 2-ply warmup)
    ├─ 1v1, bikes still share space
    │     iterative-deepening alpha-beta from the first ordered move
    │     (no 2-ply Voronoi warmup)
    └─ 1v1, bikes cut off (endgame)
          try each first move, then greedy-fill to completion
          pick the first move that survives longest
    │
    ▼
stdout: UP|DOWN|LEFT|RIGHT
```

`State` is `Copy` (~400 bytes). 1v1 and FFA search both apply a move, recurse, then `undo_step` (FFA `kill` is paired with `restore_killed`). There is no heap allocation on the hot path except a couple of tiny `Vec`s at the root of `choose_move`.

Reusable fill/flood buffers live in `Scratch` (queue + visit stamps). Voronoi no longer uses those: it expands 20-row bitboards on the stack. The separated-rate counters (`CHOOSE_MOVE_COUNT` and friends) exist only with the `local` feature / `cargo test`; they are not in the CodinGame paste.

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

Empty cells are partitioned by who can reach them first. Distances are **row-bitboard waves**: each living head’s empty neighbours are a 30-bit seed, then every orthogonal step is a shift/OR across the 20 occupancy rows. A cell reached by exactly one bike that wave is that bike’s territory; a cell reached by two or more is a tie (unowned). Reachability is a separate independent flood — opponent-owned empty cells are still walkable, matching the old per-player BFS.

The head cell itself is occupied, so search starts from its empty 4-neighbours at distance 1.

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

`my_terr` is unique Voronoi cells. `my_reach` is the raw flood (including ties).

Center preference fades as the board fills (`center_weight = (500 - occupied) / 80`).

### 2. Sharing space vs. endgame

Two tests, either is enough to stay in “open” mode (we would rather keep fighting than ignore an opponent we can still reach):

1. **`shares_space`** — heads are orthogonally adjacent, **or** BFS from us reaches an empty neighbour of their head.
2. **`voronoi.still_connected`** — some empty cell is reachable by both.

Only if **both** say we are cut off do we enter endgame. With the `local` feature, an early-game sanity counter (`EARLY_SEPARATION_COUNT`) checks that we never mark “separated” while fewer than 24 cells are occupied; in benches that counter stayed at 0.

Once cut off, each player’s remaining life is independent. The one who can occupy more of their pocket wins.

### 3. Approximate fillable space (`approx_fill`)

Flood-fill size is an **upper bound** on survival: branches behind a choke cannot all be used, because you cannot return through your own trail.

`approx_fill` DFS-walks empty neighbours of the head. At a **local cut** (the 8-cell ring around a cell does not connect its empty 4-neighbours) we can only commit to one pocket (`max`); otherwise we sum. The 8-ring test treats a degree-4 plus-shape as not a cut (`open_count >= 4`).

Then min with the **checkerboard bound** on the chosen cells. The grid is bipartite: a path that starts on one colour can take at most `2·min(same, other) + [same > other]` cells. Surplus of the majority colour is unfillable, so a simply-connected 5–4 pocket is 8 lives, not 9. A plus-shape is capped at 3 lives from the stem, not 5.

A full articulation / chamber tree is left for later: this CodinGame map is an empty 30×20, so real cut vertices appear too late for Tarjan at every leaf to pay. Endgame still uses the uncut flood for “did we seal a pocket?”.

### 4. Greedy fill (`fill_direction`)

Used as the endgame policy. For each legal step, score:

```text
80 * approx_fill
+ 30 * min(flood_remaining, checkerboard_bound)
+ 12 * wall_neighbours(destination)
+  8 if continuing straight
- 400 * (cells lost beyond the one we just took)
```

The last term is the important one: if you walk past a 1-cell pocket, raw `flood_remaining` drops by more than 1 and the move is punished. That forces you to take side pockets before committing to a corridor. The checkerboard cap on the `* 30` term is only an upper bound on remaining *life*; seal detection still uses the uncut flood.

### 5. Primed greedy endgame search

Finding a true longest path is NP-complete. a1k0n’s practical trick: **try each first move, then run the greedy filler to completion, pick the first move that “primes” the longest greedy life**.

We simulate greedy fill **to completion** after each candidate (this path skips minimax, so the turn budget is available; a 600-cell fuse is only there if a step fails to occupy). Score = `50 * extra_steps + wall_hug`. That is the move we play when `separated`.

---

## Search (open 1v1)

### Iterative deepening + alpha-beta (negamax)
```text
for depth = 1 ..= 50:
    search every root move with negamax(depth-1)
    if the iteration finishes before the deadline, keep its best move
    if it times out, discard this ply and use the previous depth
    if a mate score appears, stop
```

If even depth 1 cannot finish, we play that first ordered move. Depth 1 is a full window. Later depths search in `(prev_score ± 16000)` (~320 Voronoi cells). Fail-high or fail-low re-searches full-window; a timeout on either pass keeps the previous depth. We do not aspirate when `|prev|` is near mate. Inside a window, later root moves still raise alpha (fail-low pruning).

Every negamax return is from **`to_move`’s** point of view so the caller can negate:

- Side to move with 0 legal moves (or already dead) → `-MATE_SCORE + ply` (they lose).
- The other bike already dead → `MATE_SCORE - ply`.
- Quiet leaf → `eval_1v1` from `our_id`, then negated if it is the opponent’s turn.
- Tactical leaf → search up to `QS_MAX` (4) extra plies instead of eval. A leaf is tactical if heads are within Manhattan 2, either bike has ≤1 legal move, or (when someone has ≤2 escapes and heads are within 3) a legal step would cut the two bikes apart.

`MATE_SCORE = 1_000_000`. Closer mates score slightly higher (`MATE_SCORE - ply`). Scoring terminals from `our_id` instead of `to_move` made a forced win come back as `-MATE` after the root negation — search then “found mate” on move 1 of an empty board and stopped deepening. That is why extra depth lost Elo until the sign was fixed.

### Move ordering

Insertion-sort of at most 4 moves, keys:

1. Previous iteration’s PV move (`+10000`)
2. Killer move at this ply (`+3000`)
3. Continue straight (`+40`)
4. Wall-hug (occupied neighbours × 8)
5. Toward the opponent (minus Manhattan)

Killers: two slots per ply, updated on beta cutoffs.

### Time

`Search` timestamps a deadline. Every 64 nodes it samples `Instant::now()`. On timeout, negamax returns a sentinel; every frame still undoes its move, so the real board is never left dirty. The incomplete depth is thrown away.

Budgets: 95 ms each turn (CodinGame limit 100 ms). SPRT passes `--budget-ms` / `TRON_BUDGET_MS` so both engines use a shorter fixed time.

---

## Free-for-all (3–4 players)

FFA uses **mixed paranoid search**. We maximize `eval_ffa`. The closest living rival (`main_opponent`) is a true Min; other living bikes play one greedy flood reply that keeps their own space. Depth counts *our* plies. After each of our moves, Min walks seating order (skipping the dead) until it is our turn again. Scores stay in our POV (no negamax flip); alpha-beta prunes on that orientation. Iterative deepening starts from the first `order_moves` candidate, cap 50 (same as 1v1; time still throws away an incomplete ply), no 2-ply warmup.

`eval_ffa` prefers surviving with space:

```text
40 * our_reach + 25 * our_voronoi
- 10 * best_enemy_voronoi - 4 * best_enemy_reach
+ 20 * our_mobility
```

When only two remain, the game becomes the 1v1 path (minimax + endgame fill). Dead players’ trails are already gone, so the board opens up — that is unique to this CodinGame ruleset.

`main_opponent` is the live enemy whose head we can reach soonest (BFS). FFA uses that head as the move-ordering target.

Switching FFA from greedy-reply search to paranoid minimax (20 ms/turn, 1 candidate vs N−1 copies of that older search): **+94.5 ± 16.3 Elo** in 3p (46.3% vs 33.3%) and **+38.8 ± 10.0 Elo** in 4p (29.4% vs 25.0%). Numbers vs later freezes are in [Earlier SPRTs](#earlier-sprts-previous-freezes).

---

## Tests

`src/voronoi_tests.rs` is compiled only by `cargo test` (not pasted). It compares the bitboard Voronoi against an independent cell-BFS implementation (territory, edges, reach, connected, owned, `shares_space`, battlefront, `eval_1v1` / `eval_ffa`, `main_opponent`), walks short minimax trees with undo checks, and asserts `choose_move` restores the board.

Search smokes: trapping a boxed-in opponent at depth 1 is a **win**, the statement opening at depth 8 is **not** a mate, and quiescence sees a one-cell-left forced death that depth-1 eval misses. Fill smokes: a 3×3 minority-colour entry is 8 not 9, and a plus-shape is 3 not 5. Release-only tests enumerate every empty-board 2-player pair and every pair with a mid-board wall.

```bash
cargo test --bin tron
cargo test --bin tron --release
```

---

## Local testing

`--bench` (`src/local.rs`) plays 12 games per 1v1 matchup (colours swapped) plus 8 four-player games. Opponent bots:

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

`--profile` reports 1v1 and FFA evals/ms plus one timed `choose_move` for each. Midgame 1v1 often reaches **depth 14–15** in the 95 ms box; 4p mixed-Min typically completes depth 6.

If both players are still alive after 900 rounds, the bench awards the larger flood fill (it used to default to player 0, which biased colour-swap stats).

---

## Implementation notes

- **No reverse rule needed** — own trail blocks it.
- **Apply / undo:** `occupy` sets occupied + trail + head; `undo_step` clears the *current* head cell and restores the previous head. 1v1 and FFA search both apply/undo on one `State` (FFA `kill` is paired with `restore_killed`). Search only applies moves already known legal.
- **Eval speed:** 1v1 leaves flood us once for the cut test and, with exactly two bikes alive, reuse that mask in the open-game Voronoi instead of a third independent flood. FFA eval skips `mask_edge_sum` (the FFA score does not use edges). AVX2/SSE2 row-expand was measured and lost: 20 `u32` rows with vertical neighbours is too small for a `#[target_feature]` call, and CodinGame is `rustc -O` (no Cargo LTO / `target-cpu=native`).
- **FFA search:** the closest rival is a true Min; other bikes play one greedy flood reply (they keep their own space instead of hunting us).
- **Move ordering:** killers plus a small per-player direction history table, bumped on beta cutoffs.
- **Visit stamps** instead of `memset` on flood fills (`Scratch::next_generation`).
- **Integer-only eval**, no transposition table. A Zobrist TT for *cutoffs* was tried and measured ~0 Elo (hits were almost all previous-iteration depth-misses; sequential Tron transposes rarely), so it was reverted. Caching depth-independent *properties* of a position (Voronoi, etc.) is a different idea — see [Ideas](#ideas-that-would-still-help). Four-wide branching plus alpha-beta is enough on a 600-cell board.
- **Debug** goes to stderr (`eprintln!` + flush), which CodinGame shows in the IDE and ignores for scoring. After a completed search the line is `DIR mm SCORE Tms n=NODES d=DEPTH a=ALIVE`. Isolated fill / one-legal-move turns do not print it. SPRT parses `n=` / `Tms` for the NPS column.

---

## What to paste vs. what to keep

| path | CodinGame? |
|------|------------|
| `src/main.rs` | **yes — paste this file only** (~78k characters, limit 100k) |
| `src/local.rs` | no (`--bench` / `--profile`) |
| `src/voronoi_tests.rs` | no (`cargo test`) |
| `Cargo.toml` | no (their compiler; no `local` feature there) |
| `tools/*` | no (SPRT, watcher, opening books) |

---

## Ideas that would still help

1. **Articulation / chamber tree** for fillable space — Tarjan APs + a1k0n battlefront (drop rear rooms). Tried; on this empty 30×20 the cuts appear too late and open-game Tarjan wrecked NPS. Revisit if late-game trail geometry looks worth it.
2. **Max-N FFA** — if the mixed “closest hunts, others fill” model is still too scared or too optimistic, let every seat maximize its own `eval_ffa`.
3. **Self-play tuning** of the open-game weights (`50 / 12 / 3 / 6 / 4`) against the voronoi-1ply baseline and against copies of itself.
4. **Property cache, not cutoff TT.** A Zobrist table for alpha-beta cutoffs was ~0 Elo: sequential Tron almost never transposes at equal depth, so the hits were previous-iteration depth-misses. What *does* repeat, regardless of remaining depth, is the board itself — occupancy, heads, whose turn. Cache depth-independent facts for that key (Voronoi territory / edges / reach, `shares_space`, maybe `approx_fill`) so sibling lines and iterative-deepening re-searches skip the bitboard waves. Same paste-size and undo constraints as everything else.
5. **Incremental Voronoi on make/unmake.** A step moves one head and walls one cell, but ownership is “strictly closest,” so both distance maps can change (paths through the new wall, flips far from the head). True reverse-delta BFS is messy; the robust undo is a stack snapshot of the maps. Voronoi only runs at leaves today, so updating on every interior `apply` can cost more than recomputing at the leaf. The interesting variant is parent→leaf: keep maps on the search stack and repair one ply at a leaf. On this 30×20 the current row-bitboard waves are already cheap, so measure against a full recompute before keeping it. `kill` (whole ribbon vanishes) is a rebuild anyway. Cousin of (4): a cache skips exact repeats; this tries to update when the board only moved one cell.

---

## References

- CodinGame, *Tron Battle* — sequential turns, 30×20, vanishing trails.
- a1k0n, [Google AI Challenge post-mortem](https://www.a1k0n.net/2010/03/04/google-ai-postmortem.html) — Voronoi, wall-hug / least-edges, primed greedy endgame, tree of chambers.
- Google AI Challenge 2010 forum folklore — Voronoi as the standard open-game heuristic.
- Iouri / “tree of chambers” — fillable space as a tree split at articulation points, not raw flood fill.
- Waterloo CS Club Tron write-ups — Voronoi inside minimax, articulation points as cut targets.
