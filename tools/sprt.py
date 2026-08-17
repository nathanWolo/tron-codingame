#!/usr/bin/env python3
"""SPRT tester: pit a frozen baseline binary against a candidate (dev) binary.

Speaks the CodinGame Tron Battle stdin/stdout protocol. An N-player game is
1 candidate vs N-1 copies of the baseline. Games are played in seat-rotated
blocks on the same spawn so first-move advantage cancels (colour-swap when
N=2).

CodinGame has no discrete opening book: each player starts at a random unique
cell on the 30×20 grid. SPRT cycles a frozen sample of that distribution
and then plays a few referee-chosen legal moves so openings vary while engine
trail tracking stays in sync.

Equal engines win 1/N of games (Plackett-Luce). Elo 0 is that prior; LOS is
P(dev is stronger than the equal field).

Typical workflow:

    tools/save_baseline.sh              # freeze current release as bin/tron-baseline
    # ... edit the agent ...
    cargo build --release
    tools/sprt.sh                       # SPRT [0, 10] Elo @ 5%

Or call this file directly:

    python3 tools/sprt.py \\
        --baseline bin/tron-baseline \\
        --dev target/release/tron \\
        --players 3 --elo0 0 --elo1 10 \\
        --budget-ms 20 --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import select
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_BOOK_BY_N = {
    2: TOOLS_DIR / "openings_1v1.json",
    3: TOOLS_DIR / "openings_3p.json",
    4: TOOLS_DIR / "openings_4p.json",
}
DEFAULT_BOOK = DEFAULT_BOOK_BY_N[2]

W, H = 30, 20
DIRS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

# ---------------------------------------------------------------------------
# SPRT / Elo
# ---------------------------------------------------------------------------


def expected_score(elo: float, n_players: int = 2) -> float:
    """Plackett-Luce P(first place) vs n-1 equal opponents.

    At 0 Elo this is 1/N. For N=2 it is the usual logistic Elo curve.
    """
    return 1.0 / (1.0 + (n_players - 1) * 10.0 ** (-elo / 400.0))


def score_to_elo(score: float, n_players: int = 2) -> float:
    """Elo advantage vs an equal field of n_players (0 means win rate 1/N)."""
    s = min(max(score, 1e-9), 1.0 - 1e-9)
    return 400.0 * math.log10((s / (1.0 - s)) * (n_players - 1))


def sprt_bounds(alpha: float, beta: float) -> tuple[float, float]:
    """Accept H0 if LLR <= lower; accept H1 if LLR >= upper."""
    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)
    return lower, upper


def gsprt_llr(
    mean: float,
    var: float,
    n: int,
    elo0: float,
    elo1: float,
    n_players: int = 2,
) -> float:
    """Generalized SPRT on per-game scores in [0, 1].

    Same form OpenBench / fishtest use: test whether the mean score matches
    the Plackett-Luce conversion of elo0 vs elo1, with variance estimated
    from play. H0 at 0 Elo is a 1/N win rate against N-1 equal baselines.
    """
    if n < 2 or var <= 1e-12:
        return 0.0
    t0 = expected_score(elo0, n_players)
    t1 = expected_score(elo1, n_players)
    return n * (t1 - t0) * (2.0 * mean - t0 - t1) / (2.0 * var)


def erf_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class WDL:
    wins: int = 0
    draws: int = 0
    losses: int = 0
    score_sum: float = 0.0
    score_sq: float = 0.0
    n_players: int = 2

    def add(self, score: float) -> None:
        self.score_sum += score
        self.score_sq += score * score
        if score >= 0.99:
            self.wins += 1
        elif score <= 0.01:
            self.losses += 1
        else:
            self.draws += 1

    @property
    def n(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def points(self) -> float:
        return self.score_sum

    @property
    def mean(self) -> float:
        if self.n:
            return self.score_sum / self.n
        return 1.0 / self.n_players

    def sample_variance(self) -> float:
        """Unbiased variance of per-game scores."""
        n = self.n
        if n < 2:
            p = 1.0 / self.n_players
            return p * (1.0 - p)
        m = self.mean
        return max(0.0, (self.score_sq - n * m * m) / (n - 1))

    def elo(self) -> float:
        if self.n == 0:
            return 0.0
        return score_to_elo(self.mean, self.n_players)

    def elo_se(self) -> float:
        n = self.n
        if n < 2:
            return float("inf")
        s = min(max(self.mean, 1e-9), 1.0 - 1e-9)
        # d(elo)/d(score) = 400 / ln(10) / (s(1-s))  (n_players is a constant)
        ds = math.sqrt(self.sample_variance() / n)
        return (400.0 / math.log(10.0)) * ds / (s * (1.0 - s))

    def los(self) -> float:
        """P(dev is stronger than the equal 1/N field), Elo ~ Normal(elo, se^2)."""
        se = self.elo_se()
        if not math.isfinite(se) or se <= 0:
            return 0.5
        return erf_cdf(self.elo() / se)


# ---------------------------------------------------------------------------
# Referee (CodinGame protocol)
# ---------------------------------------------------------------------------


@dataclass
class GameResult:
    score_dev: float  # 1 / 1/k / 0 from the candidate's view
    winner: Optional[int]
    turns: int
    reason: str
    start: list[tuple[int, int]]
    first_player: str
    seed_moves: list[list[str]]
    dev_seat: int
    n_players: int
    tied: list[int]
    dev_nodes: int = 0
    dev_ms: int = 0
    base_nodes: int = 0
    base_ms: int = 0


def parse_search_stats(line: str) -> tuple[int, int]:
    """Read `n=NODES` and `Tms` from an engine stderr line. (0, 0) if absent."""
    nodes = 0
    ms = 0
    for tok in line.split():
        if tok.startswith("n=") and tok[2:].isdigit():
            nodes = int(tok[2:])
        elif tok.endswith("ms") and tok[:-2].isdigit():
            ms = int(tok[:-2])
    return nodes, ms


def fmt_nps(nodes: int, ms: int) -> str:
    if nodes <= 0 or ms <= 0:
        return "n/a"
    nps = 1000.0 * nodes / ms
    if nps >= 1_000_000:
        return f"{nps / 1_000_000:.2f}M"
    if nps >= 1000:
        return f"{nps / 1000:.0f}k"
    return f"{nps:.0f}"


def fmt_nps_pair(dev_nodes: int, dev_ms: int, base_nodes: int, base_ms: int) -> str:
    dev = fmt_nps(dev_nodes, dev_ms)
    base = fmt_nps(base_nodes, base_ms)
    if dev == "n/a" or base == "n/a":
        return f"NPS {dev} vs {base}"
    ratio = (1000.0 * dev_nodes / dev_ms) / (1000.0 * base_nodes / base_ms)
    return f"NPS {dev} vs {base} ({ratio:.1f}x)"


@dataclass
class NodeClock:
    """Search nodes and think-time reported on engine stderr (`n=` / `Tms`)."""

    nodes: int = 0
    ms: int = 0

    def add(self, nodes: int, ms: int) -> None:
        if nodes <= 0:
            return
        self.nodes += nodes
        self.ms += max(ms, 1)


class Engine:
    def __init__(self, path: str, budget_ms: int, label: str, err_dir: Optional[Path]):
        self.label = label
        self.nodes = 0
        self.ms = 0
        if err_dir is not None:
            err_dir.mkdir(parents=True, exist_ok=True)
            self._err_file = open(err_dir / f"{label}.stderr", "w", encoding="utf-8")
        else:
            self._err_file = None
        env = os.environ.copy()
        env["TRON_BUDGET_MS"] = str(budget_ms)
        self.proc = subprocess.Popen(
            [path, "--budget-ms", str(budget_ms)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        if self.proc.stdin is None or self.proc.stdout is None or self.proc.stderr is None:
            raise RuntimeError(f"failed to open pipes for {path}")

    def send(self, text: str) -> None:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(text)
            self.proc.stdin.flush()
        except BrokenPipeError as e:
            raise RuntimeError(f"{self.label} stdin closed") from e

    def drain_stderr(self) -> None:
        """Parse search-stats lines already written (stderr is emitted before stdout)."""
        err = self.proc.stderr
        if err is None:
            return
        fd = err.fileno()
        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                return
            line = err.readline()
            if line == "":
                return
            if self._err_file is not None:
                self._err_file.write(line)
            nodes, ms = parse_search_stats(line)
            if nodes > 0:
                self.nodes += nodes
                self.ms += max(ms, 1)

    def read_line(self, timeout: float) -> str:
        assert self.proc.stdout is not None
        fd = self.proc.stdout.fileno()
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            raise TimeoutError(f"{self.label} timed out after {timeout:.2f}s")
        line = self.proc.stdout.readline()
        if line == "":
            raise RuntimeError(f"{self.label} exited (code {self.proc.poll()})")
        self.drain_stderr()
        return line.strip()

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.drain_stderr()
        except Exception:
            pass
        try:
            self.proc.kill()
            self.proc.wait(timeout=1.0)
        except Exception:
            pass
        if self._err_file is not None:
            try:
                self._err_file.close()
            except OSError:
                pass


def random_starts(rng: random.Random, n: int = 2) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    while len(cells) < n:
        c = (rng.randrange(W), rng.randrange(H))
        if c not in used:
            used.add(c)
            cells.append(c)
    return cells


@lru_cache(maxsize=8)
def load_book(path: str) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Frozen sample of CodinGame's uniform unique-cell spawn distribution."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for row in data["spawns"]:
        out.append(tuple((int(c[0]), int(c[1])) for c in row))
    return tuple(out)


def pick_starts(
    rng: random.Random,
    book: tuple[tuple[tuple[int, int], ...], ...],
    n: int,
) -> list[tuple[int, int]]:
    if book:
        eligible = [row for row in book if len(row) == n]
        if not eligible:
            raise RuntimeError(f"opening book has no {n}-player spawns")
        return list(eligible[rng.randrange(len(eligible))])
    return random_starts(rng, n)


def random_seed_moves(
    rng: random.Random,
    starts: list[tuple[int, int]],
    plies: int,
    max_tries: int = 64,
) -> list[list[str]]:
    """Referee-chosen legal walks from the spawn, sequential P0 .. P(n-1).

    Engines still receive every ply (so Tracker records the trail) but the
    referee applies these moves instead of the engine output. Retry if a walk
    boxes someone in before `plies` steps; give up and return empty walks.
    """
    n = len(starts)
    if plies <= 0:
        return [[] for _ in range(n)]
    for _ in range(max_tries):
        occ = set(starts)
        heads = list(starts)
        moves: list[list[str]] = [[] for _ in range(n)]
        ok = True
        for _ply in range(plies):
            for p in range(n):
                legal: list[tuple[str, int, int]] = []
                hx, hy = heads[p]
                for name, (dx, dy) in DIRS.items():
                    nx, ny = hx + dx, hy + dy
                    if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in occ:
                        legal.append((name, nx, ny))
                if not legal:
                    ok = False
                    break
                name, nx, ny = legal[rng.randrange(len(legal))]
                occ.add((nx, ny))
                heads[p] = (nx, ny)
                moves[p].append(name)
            if not ok:
                break
        if ok:
            return moves
    return [[] for _ in range(n)]


def play_game(
    paths: list[str],
    starts: list[tuple[int, int]],
    budget_ms: int,
    turn_timeout: float,
    labels: list[str],
    err_dir: Optional[str],
    seed_moves: Optional[list[list[str]]] = None,
    candidate: int = 0,
) -> GameResult:
    """Play one N-player game. paths[i] is the binary for player i.

    `candidate` is the seat whose score is reported (1 / 1/k / 0).
    """
    n = len(paths)
    if n != len(starts) or n != len(labels):
        raise ValueError("paths, starts, and labels must have the same length")
    engines = [
        Engine(paths[i], budget_ms, labels[i], Path(err_dir) if err_dir else None)
        for i in range(n)
    ]
    occ = [[False] * W for _ in range(H)]
    trails: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    start = list(starts)
    head = list(starts)
    alive = [True] * n
    for i, (x, y) in enumerate(start):
        occ[y][x] = True
        trails[i].append((x, y))

    def kill(p: int) -> None:
        if not alive[p]:
            return
        alive[p] = False
        for x, y in trails[p]:
            occ[y][x] = False
        trails[p].clear()

    def coords(p: int) -> str:
        if not alive[p]:
            return "-1 -1 -1 -1"
        x0, y0 = start[p]
        x1, y1 = head[p]
        return f"{x0} {y0} {x1} {y1}"

    def snapshot(mover: int) -> str:
        lines = [f"{n} {mover}"]
        for p in range(n):
            lines.append(coords(p))
        return "\n".join(lines) + "\n"

    forced = seed_moves if seed_moves is not None else [[] for _ in range(n)]
    seed_plies = max((len(m) for m in forced), default=0)

    def step(p: int, token: str, why: str) -> bool:
        """Apply a direction for player p. Returns False if they died."""
        nonlocal reason
        if token not in DIRS:
            kill(p)
            reason = f"{labels[p]} {why}"
            return False
        dx, dy = DIRS[token]
        nx, ny = head[p][0] + dx, head[p][1] + dy
        if nx < 0 or nx >= W or ny < 0 or ny >= H or occ[ny][nx]:
            kill(p)
            reason = f"{labels[p]} crash"
            return False
        occ[ny][nx] = True
        head[p] = (nx, ny)
        trails[p].append((nx, ny))
        return True

    def query(p: int) -> Optional[str]:
        """Send a frame and read one engine line. None if they died."""
        nonlocal reason
        engines[p].send(snapshot(p))
        try:
            raw = engines[p].read_line(turn_timeout)
        except TimeoutError:
            kill(p)
            reason = f"{labels[p]} timeout"
            return None
        except RuntimeError as e:
            kill(p)
            reason = str(e)
            return None
        return raw.split()[0].upper() if raw.split() else ""

    turns = 0
    reason = "unknown"
    winner: Optional[int] = None
    tied: list[int] = []
    try:
        # Opening: engines see every ply so Tracker records the trail; the
        # referee applies the seeded walk instead of the engine's choice.
        for ply in range(seed_plies):
            for p in range(n):
                if not alive[p]:
                    continue
                if query(p) is None:
                    continue
                token = forced[p][ply] if ply < len(forced[p]) else ""
                step(p, token, f"seed invalid '{token}'")

        while sum(alive) > 1 and turns < 900:
            for p in range(n):
                if not alive[p]:
                    continue
                if sum(alive) <= 1:
                    break
                token = query(p)
                if token is None:
                    continue
                if token not in DIRS:
                    kill(p)
                    reason = f"{labels[p]} invalid '{token}'"
                    continue
                step(p, token, f"invalid '{token}'")
            turns += 1
        live = [i for i, a in enumerate(alive) if a]
        tied: list[int] = []
        if len(live) == 1:
            winner = live[0]
            reason = "last standing"
        elif len(live) == 0:
            winner = None
            tied = list(range(n))
            reason = reason if reason != "unknown" else "all eliminated"
        else:
            # Turn cap: most remaining flood wins; k-way ties share 1/k.
            def flood(p: int) -> int:
                if not alive[p]:
                    return 0
                hx, hy = head[p]
                seen = set()
                q = []
                for dx, dy in DIRS.values():
                    x, y = hx + dx, hy + dy
                    if 0 <= x < W and 0 <= y < H and not occ[y][x]:
                        q.append((x, y))
                        seen.add((x, y))
                i = 0
                while i < len(q):
                    x, y = q[i]
                    i += 1
                    for dx, dy in DIRS.values():
                        nx, ny = x + dx, y + dy
                        if (
                            0 <= nx < W
                            and 0 <= ny < H
                            and not occ[ny][nx]
                            and (nx, ny) not in seen
                        ):
                            seen.add((nx, ny))
                            q.append((nx, ny))
                return len(seen)

            best = max(flood(p) for p in live)
            tied = [p for p in live if flood(p) == best]
            if len(tied) == 1:
                winner = tied[0]
                tied = []
                reason = "turn cap flood"
            else:
                winner = None
                reason = "turn cap draw"
    finally:
        for e in engines:
            e.close()

    if winner is not None:
        score = 1.0 if winner == candidate else 0.0
    elif candidate in tied:
        score = 1.0 / len(tied)
    else:
        score = 0.0
    dev_nodes = sum(e.nodes for e, lab in zip(engines, labels) if lab == "dev")
    dev_ms = sum(e.ms for e, lab in zip(engines, labels) if lab == "dev")
    base_nodes = sum(e.nodes for e, lab in zip(engines, labels) if lab.startswith("base"))
    base_ms = sum(e.ms for e, lab in zip(engines, labels) if lab.startswith("base"))
    return GameResult(
        score_dev=score,
        winner=winner,
        turns=turns,
        reason=reason,
        start=start,
        first_player=labels[0],
        seed_moves=forced,
        dev_seat=candidate,
        n_players=n,
        tied=tied,
        dev_nodes=dev_nodes,
        dev_ms=dev_ms,
        base_nodes=base_nodes,
        base_ms=base_ms,
    )


def _play_block_job(payload: dict) -> dict:
    """Worker: N games, candidate rotated through every seat, same opening."""
    n = int(payload.get("n_players", 2))
    rng = random.Random(payload["seed"])
    book_path = payload.get("book") or ""
    book = load_book(book_path) if book_path else ()
    starts = pick_starts(rng, book, n)
    seed_moves = random_seed_moves(rng, starts, int(payload.get("seed_plies", 0)))
    budget = payload["budget_ms"]
    timeout = payload["turn_timeout"]
    err = payload.get("err_dir")
    base, dev = payload["baseline"], payload["dev"]
    games = []
    dev_scores = []
    dev_nodes = 0
    dev_ms = 0
    base_nodes = 0
    base_ms = 0
    for seat in range(n):
        paths = [dev if i == seat else base for i in range(n)]
        labels = ["dev" if i == seat else f"base{i}" for i in range(n)]
        g = play_game(paths, starts, budget, timeout, labels, err, seed_moves, seat)
        games.append(asdict(g))
        dev_scores.append(g.score_dev)
        dev_nodes += g.dev_nodes
        dev_ms += g.dev_ms
        base_nodes += g.base_nodes
        base_ms += g.base_ms
    return {
        "seed": payload["seed"],
        "n_players": n,
        "start": starts,
        "seed_moves": seed_moves,
        "games": games,
        "dev_scores": dev_scores,
        "dev_nodes": dev_nodes,
        "dev_ms": dev_ms,
        "base_nodes": base_nodes,
        "base_ms": base_ms,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def fmt_status(
    wdl: WDL,
    llr: float,
    lo: float,
    hi: float,
    elo0: float,
    elo1: float,
    nps: NodeClock | None = None,
    nps_base: NodeClock | None = None,
) -> str:
    se = wdl.elo_se()
    se_s = f"{se:.1f}" if math.isfinite(se) else "inf"
    fair = 100.0 / wdl.n_players
    line = (
        f"n={wdl.n:4d}  {wdl.wins}-{wdl.draws}-{wdl.losses}  "
        f"{100.0 * wdl.mean:5.1f}% vs {fair:4.1f}%  "
        f"Elo {wdl.elo():+6.1f} ± {se_s:>5s}  "
        f"LOS {100.0 * wdl.los():5.1f}%  "
        f"LLR {llr:+6.2f} [{lo:+.2f},{hi:+.2f}]  "
        f"SPRT[{elo0:g},{elo1:g}] {wdl.n_players}p"
    )
    if nps is not None and nps_base is not None:
        line += "  " + fmt_nps_pair(nps.nodes, nps.ms, nps_base.nodes, nps_base.ms)
    return line


def run_sprt(args: argparse.Namespace, fixed: bool = False) -> int:
    baseline = os.path.abspath(args.baseline)
    dev = os.path.abspath(args.dev)
    for p, name in ((baseline, "baseline"), (dev, "dev")):
        if not os.path.isfile(p) or not os.access(p, os.X_OK):
            print(f"error: {name} binary not executable: {p}", file=sys.stderr)
            return 2

    n_players = int(args.players)
    lo, hi = sprt_bounds(args.alpha, args.beta)
    wdl = WDL(n_players=n_players)
    nps_dev = NodeClock()
    nps_base = NodeClock()
    scores: list[float] = []
    rng = random.Random(args.seed)
    log_f = open(args.log, "a") if args.log else None
    t0 = time.time()
    decided = None
    games_played = 0

    turn_timeout = args.timeout
    if turn_timeout <= 0:
        turn_timeout = args.budget_ms / 1000.0 + 0.25

    fair = 1.0 / n_players
    print(
        f"SPRT {n_players}p  1 dev vs {n_players - 1} baseline  "
        f"fair={100.0 * fair:.1f}%  "
        f"H0={args.elo0:g} Elo ({100.0 * expected_score(args.elo0, n_players):.1f}%)  "
        f"H1={args.elo1:g} Elo ({100.0 * expected_score(args.elo1, n_players):.1f}%)  "
        f"α={args.alpha:g} β={args.beta:g}  bounds [{lo:.3f}, {hi:.3f}]",
        flush=True,
    )
    print(
        f"baseline={baseline}\n"
        f"dev     ={dev}\n"
        f"book    ={args.book or 'none (uniform random spawns)'}\n"
        f"seed    ={args.seed_plies} random legal plies after spawn\n"
        f"budget  ={args.budget_ms} ms/turn  timeout={turn_timeout:.2f}s  "
        f"concurrency={args.concurrency}  max-games={args.max_games}",
        flush=True,
    )

    pending_max = max(1, args.max_games)
    block_count_target = (pending_max + n_players - 1) // n_players

    def make_payload(seed: int) -> dict:
        return {
            "seed": seed,
            "baseline": baseline,
            "dev": dev,
            "budget_ms": args.budget_ms,
            "turn_timeout": turn_timeout,
            "err_dir": args.engine_stderr,
            "book": args.book or "",
            "seed_plies": args.seed_plies,
            "n_players": n_players,
        }

    # Sequential SPRT with batched workers: submit `concurrency` seat-rotated
    # blocks, harvest, update LLR, stop when we can decide.
    with ProcessPoolExecutor(max_workers=args.concurrency) as pool:
        in_flight = {}
        submitted_blocks = 0

        def submit_one() -> None:
            nonlocal submitted_blocks
            if submitted_blocks >= block_count_target:
                return
            seed = rng.randrange(2**63)
            fut = pool.submit(_play_block_job, make_payload(seed))
            in_flight[fut] = seed
            submitted_blocks += 1

        for _ in range(min(args.concurrency, block_count_target)):
            submit_one()

        while in_flight:
            done = next(as_completed(list(in_flight)))
            in_flight.pop(done)
            try:
                result = done.result()
            except Exception as e:
                print(f"worker failed: {e}", file=sys.stderr)
                submit_one()
                continue

            for sc in result["dev_scores"]:
                wdl.add(sc)
                scores.append(sc)
                games_played += 1
            nps_dev.add(int(result.get("dev_nodes") or 0), int(result.get("dev_ms") or 0))
            nps_base.add(int(result.get("base_nodes") or 0), int(result.get("base_ms") or 0))

            if log_f:
                log_f.write(json.dumps(result) + "\n")
                log_f.flush()

            mean = wdl.mean
            var = wdl.sample_variance()
            llr = gsprt_llr(mean, var, wdl.n, args.elo0, args.elo1, n_players)

            if args.verbose or wdl.n % max(n_players, args.print_every) == 0:
                print(
                    fmt_status(wdl, llr, lo, hi, args.elo0, args.elo1, nps_dev, nps_base),
                    flush=True,
                )

            if (not fixed) and wdl.n >= args.min_games:
                if llr >= hi:
                    decided = "H1"
                elif llr <= lo:
                    decided = "H0"

            if decided or wdl.n >= args.max_games:
                for fut in list(in_flight):
                    try:
                        extra = fut.result()
                        for sc in extra["dev_scores"]:
                            wdl.add(sc)
                            scores.append(sc)
                            games_played += 1
                        nps_dev.add(
                            int(extra.get("dev_nodes") or 0), int(extra.get("dev_ms") or 0)
                        )
                        nps_base.add(
                            int(extra.get("base_nodes") or 0), int(extra.get("base_ms") or 0)
                        )
                        if log_f:
                            log_f.write(json.dumps(extra) + "\n")
                    except Exception:
                        pass
                in_flight.clear()
                break

            submit_one()

    if log_f:
        log_f.close()

    llr = gsprt_llr(
        wdl.mean, wdl.sample_variance(), wdl.n, args.elo0, args.elo1, n_players
    )
    elapsed = time.time() - t0
    print(
        fmt_status(wdl, llr, lo, hi, args.elo0, args.elo1, nps_dev, nps_base),
        flush=True,
    )
    print(f"time {elapsed:.1f}s  games/s {wdl.n / max(elapsed, 1e-6):.2f}", flush=True)
    print(
        f"search  {fmt_nps_pair(nps_dev.nodes, nps_dev.ms, nps_base.nodes, nps_base.ms)}"
        f"  dev {nps_dev.nodes} nodes / {nps_dev.ms}ms"
        f"  base {nps_base.nodes} nodes / {nps_base.ms}ms",
        flush=True,
    )

    if fixed:
        print("FIXED match complete (no SPRT decision).", flush=True)
        return 0
    if decided == "H1":
        print(
            f"ACCEPT H1: dev is stronger than a 1/{n_players} field "
            f"(target {args.elo1:g} Elo vs H0 {args.elo0:g}).",
            flush=True,
        )
        return 0
    if decided == "H0":
        print(
            f"ACCEPT H0: no {args.elo1:g} Elo gain vs a 1/{n_players} field "
            f"(dev looks like {args.elo0:g} Elo).",
            flush=True,
        )
        return 1
    print(
        f"INCONCLUSIVE: hit max-games={args.max_games} without crossing SPRT bounds.",
        flush=True,
    )
    return 2


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SPRT: baseline vs dev Tron binaries (CodinGame protocol)."
    )
    p.add_argument("--baseline", required=True, help="Path to frozen baseline engine")
    p.add_argument("--dev", required=True, help="Path to candidate engine")
    p.add_argument(
        "--players",
        type=int,
        default=2,
        choices=(2, 3, 4),
        help="Players per game: 1 candidate vs N-1 baselines (default 2)",
    )
    p.add_argument(
        "--elo0",
        type=float,
        default=0.0,
        help="H0 Elo of dev vs an equal field (default 0 = win rate 1/N)",
    )
    p.add_argument(
        "--elo1",
        type=float,
        default=10.0,
        help="H1 Elo of dev vs baseline (default 10 = small gain)",
    )
    p.add_argument("--alpha", type=float, default=0.05, help="False-H1 (type I) rate")
    p.add_argument("--beta", type=float, default=0.05, help="False-H0 (type II) rate")
    p.add_argument(
        "--budget-ms",
        type=int,
        default=20,
        help="Per-turn search budget passed to both engines (default 20)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Referee per-move timeout in seconds (default budget-ms/1000 + 0.25)",
    )
    p.add_argument("--concurrency", type=int, default=os.cpu_count() or 2)
    p.add_argument("--max-games", type=int, default=2000)
    p.add_argument(
        "--min-games",
        type=int,
        default=40,
        help="Do not stop for SPRT before this many games",
    )
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    p.add_argument(
        "--book",
        default="",
        help=(
            "JSON spawn book. Default: tools/openings_1v1.json (2p), "
            "openings_3p.json, or openings_4p.json. CodinGame has no official "
            "opening set; these are frozen samples of uniform unique-cell "
            "starts. Pass 'none' for a fresh random spawn every block."
        ),
    )
    p.add_argument(
        "--seed-plies",
        type=int,
        default=4,
        help=(
            "After spawn, play this many referee-chosen legal moves per side "
            "(default 4) so openings vary. Engines still receive every ply. "
            "0 = start from the spawn cells."
        ),
    )
    p.add_argument(
        "--log",
        default="",
        help="JSONL path for per-block results (default: none)",
    )
    p.add_argument(
        "--engine-stderr",
        default="",
        help="Directory to capture engine stderr (default: discard)",
    )
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--fixed",
        action="store_true",
        help="Play exactly --max-games and report Elo; do not apply SPRT stop",
    )
    return p


def main() -> int:
    args = build_argparser().parse_args()
    fixed = args.fixed
    if args.fixed:
        args.min_games = args.max_games
    if args.concurrency < 1:
        args.concurrency = 1
    if args.seed_plies < 0:
        args.seed_plies = 0
    raw_book = (args.book or "").strip()
    if raw_book.lower() in ("none", "off", "-"):
        args.book = ""
    elif raw_book == "":
        default = DEFAULT_BOOK_BY_N.get(int(args.players))
        if default is not None and default.is_file():
            args.book = str(default.resolve())
        else:
            args.book = ""
    else:
        book_path = Path(raw_book).expanduser()
        if not book_path.is_file():
            alt = Path(__file__).parent / raw_book
            if alt.is_file():
                book_path = alt
            else:
                print(f"error: opening book not found: {raw_book}", file=sys.stderr)
                return 2
        args.book = str(book_path.resolve())
    if args.engine_stderr == "":
        args.engine_stderr = None
    if args.log == "":
        args.log = None
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        return run_sprt(args, fixed=fixed)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
