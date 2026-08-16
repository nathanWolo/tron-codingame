#!/usr/bin/env python3
"""SPRT tester: pit a frozen baseline binary against a candidate (dev) binary.

Speaks the CodinGame Tron Battle stdin/stdout protocol. Games are played in
colour-swapped pairs on the same spawn so first-move advantage cancels.

Typical workflow:

    tools/save_baseline.sh              # freeze current release as bin/tron-baseline
    # ... edit the agent ...
    cargo build --release
    tools/sprt.sh                       # SPRT [0, 10] Elo @ 5%

Or call this file directly:

    python3 tools/sprt.py \\
        --baseline bin/tron-baseline \\
        --dev target/release/tron \\
        --elo0 0 --elo1 10 \\
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
from pathlib import Path
from typing import IO, Optional

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


def expected_score(elo: float) -> float:
    """Logistic expected score (BayesElo / Elo)."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def score_to_elo(score: float) -> float:
    s = min(max(score, 1e-9), 1.0 - 1e-9)
    return 400.0 * math.log10(s / (1.0 - s))


def sprt_bounds(alpha: float, beta: float) -> tuple[float, float]:
    """Accept H0 if LLR <= lower; accept H1 if LLR >= upper."""
    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)
    return lower, upper


def gsprt_llr(mean: float, var: float, n: int, elo0: float, elo1: float) -> float:
    """Generalized SPRT on game scores in {0, 0.5, 1}.

    Same form OpenBench / fishtest use: test whether the mean score matches
    the logistic conversion of elo0 vs elo1, with variance estimated from play.
    """
    if n < 2 or var <= 1e-12:
        return 0.0
    t0 = expected_score(elo0)
    t1 = expected_score(elo1)
    return n * (t1 - t0) * (2.0 * mean - t0 - t1) / (2.0 * var)


def erf_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class WDL:
    wins: int = 0
    draws: int = 0
    losses: int = 0

    def add(self, score: float) -> None:
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
        return self.wins + 0.5 * self.draws

    @property
    def mean(self) -> float:
        return self.points / self.n if self.n else 0.5

    def sample_variance(self) -> float:
        """Unbiased variance of per-game scores."""
        n = self.n
        if n < 2:
            return 0.25
        m = self.mean
        sse = (
            self.wins * (1.0 - m) ** 2
            + self.draws * (0.5 - m) ** 2
            + self.losses * (0.0 - m) ** 2
        )
        return sse / (n - 1)

    def elo(self) -> float:
        if self.n == 0:
            return 0.0
        return score_to_elo(self.mean)

    def elo_se(self) -> float:
        n = self.n
        if n < 2:
            return float("inf")
        s = min(max(self.mean, 1e-9), 1.0 - 1e-9)
        # d(elo)/d(score) = 400 / ln(10) / (s(1-s))
        ds = math.sqrt(self.sample_variance() / n)
        return (400.0 / math.log(10.0)) * ds / (s * (1.0 - s))

    def los(self) -> float:
        """P(dev is stronger), assuming Elo ~ Normal(elo, se^2)."""
        se = self.elo_se()
        if not math.isfinite(se) or se <= 0:
            return 0.5
        return erf_cdf(self.elo() / se)


# ---------------------------------------------------------------------------
# Referee (CodinGame protocol)
# ---------------------------------------------------------------------------


@dataclass
class GameResult:
    score_p0: float  # 1 / 0.5 / 0 from player 0's view
    winner: Optional[int]
    turns: int
    reason: str
    start: list[tuple[int, int]]
    first_player: str  # "dev" or "base" for P0


class Engine:
    def __init__(self, path: str, budget_ms: int, label: str, err_dir: Optional[Path]):
        self.label = label
        stderr: IO[str] | int
        if err_dir is not None:
            err_dir.mkdir(parents=True, exist_ok=True)
            self._err_file = open(err_dir / f"{label}.stderr", "w", encoding="utf-8")
            stderr = self._err_file
        else:
            self._err_file = None
            stderr = subprocess.DEVNULL
        env = os.environ.copy()
        env["TRON_BUDGET_MS"] = str(budget_ms)
        self.proc = subprocess.Popen(
            [path, "--budget-ms", str(budget_ms)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=env,
            text=True,
            bufsize=1,
        )
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError(f"failed to open pipes for {path}")

    def send(self, text: str) -> None:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(text)
            self.proc.stdin.flush()
        except BrokenPipeError as e:
            raise RuntimeError(f"{self.label} stdin closed") from e

    def read_line(self, timeout: float) -> str:
        assert self.proc.stdout is not None
        fd = self.proc.stdout.fileno()
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            raise TimeoutError(f"{self.label} timed out after {timeout:.2f}s")
        line = self.proc.stdout.readline()
        if line == "":
            raise RuntimeError(f"{self.label} exited (code {self.proc.poll()})")
        return line.strip()

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except OSError:
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


def play_game(
    paths: tuple[str, str],
    starts: list[tuple[int, int]],
    budget_ms: int,
    turn_timeout: float,
    labels: tuple[str, str],
    err_dir: Optional[str],
) -> GameResult:
    """Play one 1v1 game. paths[i] is the binary for player i."""
    n = 2
    engines = [
        Engine(paths[i], budget_ms, labels[i], Path(err_dir) if err_dir else None)
        for i in range(n)
    ]
    occ = [[False] * W for _ in range(H)]
    trails: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    start = list(starts)
    head = list(starts)
    alive = [True, True]
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

    turns = 0
    reason = "unknown"
    winner: Optional[int] = None
    try:
        while sum(alive) > 1 and turns < 900:
            for p in range(n):
                if not alive[p]:
                    continue
                if sum(alive) <= 1:
                    break
                engines[p].send(snapshot(p))
                try:
                    raw = engines[p].read_line(turn_timeout)
                except TimeoutError:
                    kill(p)
                    reason = f"{labels[p]} timeout"
                    continue
                except RuntimeError as e:
                    kill(p)
                    reason = str(e)
                    continue
                token = raw.split()[0].upper() if raw.split() else ""
                if token not in DIRS:
                    kill(p)
                    reason = f"{labels[p]} invalid '{raw}'"
                    continue
                dx, dy = DIRS[token]
                nx, ny = head[p][0] + dx, head[p][1] + dy
                if nx < 0 or nx >= W or ny < 0 or ny >= H or occ[ny][nx]:
                    kill(p)
                    reason = f"{labels[p]} crash"
                    continue
                occ[ny][nx] = True
                head[p] = (nx, ny)
                trails[p].append((nx, ny))
            turns += 1
        live = [i for i, a in enumerate(alive) if a]
        if len(live) == 1:
            winner = live[0]
            reason = "last standing"
        elif len(live) == 0:
            winner = None
            reason = reason if reason != "unknown" else "double elimination"
        else:
            # Turn cap: more remaining flood wins (draw if equal).
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

            f0, f1 = flood(0), flood(1)
            if f0 > f1:
                winner = 0
                reason = "turn cap flood"
            elif f1 > f0:
                winner = 1
                reason = "turn cap flood"
            else:
                winner = None
                reason = "turn cap draw"
    finally:
        for e in engines:
            e.close()

    if winner == 0:
        score = 1.0
    elif winner == 1:
        score = 0.0
    else:
        score = 0.5
    return GameResult(
        score_p0=score,
        winner=winner,
        turns=turns,
        reason=reason,
        start=start,
        first_player=labels[0],
    )


def _play_pair_job(payload: dict) -> dict:
    """Worker entry: two games, colour-swapped, same spawn."""
    rng = random.Random(payload["seed"])
    starts = random_starts(rng, 2)
    budget = payload["budget_ms"]
    timeout = payload["turn_timeout"]
    err = payload.get("err_dir")
    base, dev = payload["baseline"], payload["dev"]
    g1 = play_game((dev, base), starts, budget, timeout, ("dev", "base"), err)
    g2 = play_game((base, dev), starts, budget, timeout, ("base", "dev"), err)
    # Dev scores: g1 as P0, g2 as P1 (so 1 - g2.score_p0)
    return {
        "seed": payload["seed"],
        "start": starts,
        "g1": asdict(g1),
        "g2": asdict(g2),
        "dev_scores": [g1.score_p0, 1.0 - g2.score_p0],
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def fmt_status(wdl: WDL, llr: float, lo: float, hi: float, elo0: float, elo1: float) -> str:
    se = wdl.elo_se()
    se_s = f"{se:.1f}" if math.isfinite(se) else "inf"
    return (
        f"n={wdl.n:4d}  {wdl.wins}-{wdl.draws}-{wdl.losses}  "
        f"Elo {wdl.elo():+6.1f} ± {se_s:>5s}  "
        f"LOS {100.0 * wdl.los():5.1f}%  "
        f"LLR {llr:+6.2f} [{lo:+.2f},{hi:+.2f}]  "
        f"SPRT[{elo0:g},{elo1:g}]"
    )


def run_sprt(args: argparse.Namespace, fixed: bool = False) -> int:
    baseline = os.path.abspath(args.baseline)
    dev = os.path.abspath(args.dev)
    for p, name in ((baseline, "baseline"), (dev, "dev")):
        if not os.path.isfile(p) or not os.access(p, os.X_OK):
            print(f"error: {name} binary not executable: {p}", file=sys.stderr)
            return 2

    lo, hi = sprt_bounds(args.alpha, args.beta)
    wdl = WDL()
    scores: list[float] = []
    rng = random.Random(args.seed)
    log_f = open(args.log, "a") if args.log else None
    t0 = time.time()
    decided = None
    games_played = 0

    print(
        f"SPRT H0={args.elo0:g} Elo  H1={args.elo1:g} Elo  "
        f"α={args.alpha:g} β={args.beta:g}  bounds [{lo:.3f}, {hi:.3f}]",
        flush=True,
    )
    print(
        f"baseline={baseline}\n"
        f"dev     ={dev}\n"
        f"budget  ={args.budget_ms} ms/turn  timeout={args.timeout:.2f}s  "
        f"concurrency={args.concurrency}  max-games={args.max_games}",
        flush=True,
    )

    turn_timeout = args.timeout
    if turn_timeout <= 0:
        turn_timeout = args.budget_ms / 1000.0 + 0.25

    pending_max = max(1, args.max_games)
    pair_count_target = (pending_max + 1) // 2

    def make_payload(seed: int) -> dict:
        return {
            "seed": seed,
            "baseline": baseline,
            "dev": dev,
            "budget_ms": args.budget_ms,
            "turn_timeout": turn_timeout,
            "err_dir": args.engine_stderr,
        }

    # Sequential SPRT with batched workers: submit `concurrency` pairs, harvest,
    # update LLR, stop when we can decide (or hit max games / min games).
    with ProcessPoolExecutor(max_workers=args.concurrency) as pool:
        in_flight = {}
        submitted_pairs = 0

        def submit_one() -> None:
            nonlocal submitted_pairs
            if submitted_pairs >= pair_count_target:
                return
            seed = rng.randrange(2**63)
            fut = pool.submit(_play_pair_job, make_payload(seed))
            in_flight[fut] = seed
            submitted_pairs += 1

        for _ in range(min(args.concurrency, pair_count_target)):
            submit_one()

        while in_flight:
            # Wait for at least one pair.
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

            if log_f:
                log_f.write(json.dumps(result) + "\n")
                log_f.flush()

            mean = wdl.mean
            var = wdl.sample_variance()
            llr = gsprt_llr(mean, var, wdl.n, args.elo0, args.elo1)

            if args.verbose or wdl.n % max(2, args.print_every) == 0:
                print(fmt_status(wdl, llr, lo, hi, args.elo0, args.elo1), flush=True)

            if (not fixed) and wdl.n >= args.min_games:
                if llr >= hi:
                    decided = "H1"
                elif llr <= lo:
                    decided = "H0"

            if decided or wdl.n >= args.max_games:
                # Let remaining in-flight pairs finish so we don't leak procs,
                # but do not submit more.
                for fut in list(in_flight):
                    try:
                        extra = fut.result()
                        for sc in extra["dev_scores"]:
                            wdl.add(sc)
                            scores.append(sc)
                            games_played += 1
                        if log_f:
                            log_f.write(json.dumps(extra) + "\n")
                    except Exception:
                        pass
                in_flight.clear()
                break

            submit_one()

    if log_f:
        log_f.close()

    llr = gsprt_llr(wdl.mean, wdl.sample_variance(), wdl.n, args.elo0, args.elo1)
    elapsed = time.time() - t0
    print(fmt_status(wdl, llr, lo, hi, args.elo0, args.elo1), flush=True)
    print(f"time {elapsed:.1f}s  games/s {wdl.n / max(elapsed, 1e-6):.2f}", flush=True)

    if fixed:
        print("FIXED match complete (no SPRT decision).", flush=True)
        return 0
    if decided == "H1":
        print(
            f"ACCEPT H1: dev is stronger (target {args.elo1:g} Elo vs H0 {args.elo0:g}).",
            flush=True,
        )
        return 0
    if decided == "H0":
        print(
            f"ACCEPT H0: no {args.elo1:g} Elo gain (dev looks like {args.elo0:g} Elo).",
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
        "--elo0",
        type=float,
        default=0.0,
        help="H0 Elo of dev vs baseline (default 0 = equal)",
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
        "--log",
        default="",
        help="JSONL path for per-pair results (default: none)",
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
