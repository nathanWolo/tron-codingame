#!/usr/bin/env python3
"""New build against old build: the test a change has to pass before it ships.

Two folders hold the two sides, each an engine and the network it loads:

    gate/prev/   the last accepted engine (tronzero.exe + net.bin), frozen
    gate/dev/    the candidate

    python tools/gate.py freeze            copy bin/tronzero-dev.exe + a network into gate/prev
    python tools/gate.py dev               the same into gate/dev
    python tools/gate.py books             balanced opening books, judged by the frozen engine
    python tools/gate.py screen ...        fixed simulations, in process: networks and search settings
    python tools/gate.py run ...           the referee (sprt.py) at CodinGame's speed: any change

Both stages are sequential tests (SPRT, as sprt.py and fishtest run them):
they stop as soon as the result is clear, accepting either H0, the new side
is worth `--elo0` (default 0), or H1, it is worth `--elo1` (default 10), with
5% error rates, and give up undecided at a cap. `--fixed` plays the whole
cap instead. Every result is reported as Elo with a 95% interval.

`screen` plays both sides inside the candidate's build, so it can compare
two networks or two search settings (`-- --c-puct-a 0.6`), not two builds.
`run` plays the two builds as separate processes through the CodinGame
protocol, one candidate against copies of the old build, in every seat.

Duels start from the balanced book openings/balanced_2p.json: spawns where
the frozen engine gives both bikes an even chance. From random spawns nine
duels in ten are decided by the spawn and say nothing about the change; on
the book the same known difference (2,000 simulations against 1,000) read
0.605 +/- 0.038 instead of 0.518 +/- 0.022 over the same 400 games. With
three or four players a book made no difference (0.523 +/- 0.015 against
0.518 +/- 0.012), so those games start from uniform random spawns, as on
CodinGame; `--book FILE` overrides either choice.

Both stages are set to what the engine gets on CodinGame, measured there in
test games on 14 Nov 2026 (README, "CodinGame arena"). A 94 ms turn gave a
median of 1,250 simulations with four bikes alive, 1,550 with three and
1,800 with two: those are the simulations `screen` plays at. With 7 games
running this desktop gave 1,500 at 56 ms with four players and 1,740 in
duels, 1.6 to 2.0 times CodinGame's speed, so the default budget of `run`,
52 ms, stands in for CodinGame's 95 ms. Run at most 7 games at once and
nothing else on the machine.

Elo is the logistic of the score, as sprt.py converts it: of the mean
placement score in `screen` (0.5 is level; the two sides share each table),
of the candidate's win rate against the field in `run` (1/players is
level). The two scales are not the same thing and are not to be added. The
interval comes from the spread between openings, each counted once. A
`--fixed` result passes only if its whole interval is above 0 Elo.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = "tronzero.exe" if os.name == "nt" else "tronzero"
BUILT = os.path.join("bin", "tronzero-dev.exe" if os.name == "nt" else "tronzero-dev")
BOOK_SETTINGS = {2: (4000, 0.15), 3: (4000, 0.15), 4: (8000, 0.10)}  # openings, largest |value|
BOOK_BY_DEFAULT = (2,)  # player counts that start from their balanced book unless told otherwise
SCREEN_CAP = {2: 3000, 3: 800, 4: 600}  # openings a screen plays at most, by players
SCREEN_BATCH = {2: 100, 3: 25, 4: 25}  # openings between two looks at the result
SCREEN_SIMS = {2: 1800, 3: 1550, 4: 1250}  # simulations a 94 ms turn gave on CodinGame, by bikes alive


def path(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def sha256(file: str) -> str:
    h = hashlib.sha256()
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def book(n: int) -> str:
    return path("openings", f"balanced_{n}p.json")


def book_for(args: argparse.Namespace) -> str:
    """The book a match uses: --book, else the balanced book for duels, else none ("")."""
    if args.book:
        return "" if args.book == "none" else os.path.join(ROOT, args.book)
    return book(args.players) if args.players in BOOK_BY_DEFAULT else ""


def side(name: str) -> tuple[str, str]:
    exe, net = path("gate", name, EXE), path("gate", name, "net.bin")
    for f in (exe, net):
        if not os.path.isfile(f):
            sys.exit(f"{f} is missing: run `python tools/gate.py {'freeze' if name == 'prev' else 'dev'}` first")
    return exe, net


def stage(name: str, args: argparse.Namespace) -> None:
    folder = path("gate", name)
    note = os.path.join(folder, "ABOUT.txt")
    if name == "prev" and os.path.exists(note) and not args.force:
        sys.exit("gate/prev is already frozen; pass --force to replace it (only after a change has passed)")
    os.makedirs(folder, exist_ok=True)
    exe, net = os.path.join(folder, EXE), os.path.join(folder, "net.bin")
    shutil.copyfile(os.path.join(ROOT, args.exe), exe)
    shutil.copymode(os.path.join(ROOT, args.exe), exe)
    shutil.copyfile(os.path.join(ROOT, args.net), net)
    with open(note, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{'Frozen' if name == 'prev' else 'Staged'} {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
        f.write(f"engine  {args.exe.replace(os.sep, '/')}  sha256 {sha256(exe)}\n")
        f.write(f"network {args.net.replace(os.sep, '/')}  sha256 {sha256(net)}\n")
        if args.note:
            f.write(args.note + "\n")
    print(open(note, encoding="utf-8").read(), end="")


def record(kind: str, entry: dict) -> None:
    os.makedirs(path("gate", "logs"), exist_ok=True)
    entry = {"when": f"{datetime.datetime.now():%Y-%m-%d %H:%M}", "kind": kind, **entry}
    with open(path("gate", "logs", "results.jsonl"), "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry) + "\n")


def score_at(elo: float, n: int = 2) -> float:
    """The score an Elo advantage gives against an equal field (sprt.py's `expected_score`): 1/n at 0."""
    return 1.0 / (1.0 + (n - 1) * 10.0 ** (-elo / 400.0))


def elo_of(score: float, n: int = 2) -> float:
    s = min(max(score, 1e-9), 1.0 - 1e-9)
    return 400.0 * math.log10(s / (1.0 - s) * (n - 1))


def elo_interval(score: float, ci: float, n: int = 2) -> float:
    """Half-width in Elo of a score's interval `ci` (first-order, as sprt.py's `elo_se`)."""
    s = min(max(score, 1e-9), 1.0 - 1e-9)
    return 400.0 / math.log(10.0) * ci / (s * (1.0 - s))


def sprt_bounds(alpha: float, beta: float) -> tuple[float, float]:
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


def llr(mean: float, var: float, count: int, elo0: float, elo1: float, n: int = 2) -> float:
    """Generalised SPRT log-likelihood ratio of H1 against H0 from a mean and a variance (sprt.py's `gsprt_llr`)."""
    if count < 2 or var <= 1e-12:
        return 0.0
    t0, t1 = score_at(elo0, n), score_at(elo1, n)
    return count * (t1 - t0) * (2.0 * mean - t0 - t1) / (2.0 * var)


def verdict(decision: str, elo: float, ci: float, args: argparse.Namespace) -> str:
    if decision == "H1":
        return f"PASS (SPRT accepted H1, {args.elo1:g} Elo)"
    if decision == "H0":
        return f"NO GAIN (SPRT accepted H0, {args.elo0:g} Elo)"
    if decision == "undecided":
        return "UNDECIDED (reached the cap between the two bounds)"
    if elo - ci > 0:
        return "PASS (the whole interval is above 0)"
    if elo + ci < 0:
        return "WORSE (the whole interval is below 0)"
    return "NOT SHOWN (the interval includes 0)"


def cmd_books(args: argparse.Namespace) -> None:
    # The frozen network judges the spawns; the current build runs the command
    # (a frozen engine may predate it).
    _, net = side("prev")
    exe = path(BUILT)
    os.makedirs(path("openings"), exist_ok=True)
    for n in args.players:
        count, max_abs = BOOK_SETTINGS[n]
        subprocess.run(
            [exe, "openings", "--net", net, "--players", str(n), "--count", str(count), "--max-abs", str(max_abs),
             "--sims", "2000", "--seed", str(args.seed), "--threads", str(args.threads), "--out", book(n)],
            check=True,
        )


def cmd_screen(args: argparse.Namespace) -> None:
    exe, net = side("dev")
    _, old_net = side("prev")
    cap = args.openings or SCREEN_CAP[args.players]
    step = args.batch or SCREEN_BATCH[args.players]
    args.sims = args.sims or SCREEN_SIMS[args.players]
    lo, hi = sprt_bounds(args.alpha, args.beta)
    test = "fixed length" if args.fixed else f"SPRT[{args.elo0:g},{args.elo1:g}]"
    batches: list[tuple[int, float, float]] = []  # openings, mean score, variance between openings
    tally = {"games": 0, "pairs_lost": 0, "pairs_split": 0, "pairs_won": 0}
    done, mean, var, decision = 0, 0.5, 0.0, "fixed" if args.fixed else "undecided"
    while done < cap:
        k = min(step, cap - done)
        cmd = [exe, "match", "--a", net, "--b", old_net, "--players", str(args.players), "--pairs", str(k),
               "--sims", str(args.sims), "--threads", str(args.threads)]
        if book_for(args):  # one shuffled order of the book, read on from where the last batch stopped
            cmd += ["--book", book_for(args), "--seed", str(args.seed), "--skip", str(args.skip + done)]
        else:  # fresh uniform spawns for every batch
            cmd += ["--seed", str(args.seed * 100003 + len(batches))]
        cmd += args.extra
        res = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1])
        # The match reports its interval from the spread between openings: 1.96 * sqrt(variance / openings).
        batches.append((k, res["score"], (res["ci95"] / 1.96) ** 2 * k))
        for key in tally:
            tally[key] += res[key]
        done += k
        mean = sum(b[0] * b[1] for b in batches) / done
        var = (sum((b[0] - 1) * b[2] + b[0] * (b[1] - mean) ** 2 for b in batches)) / max(done - 1, 1)
        ci = 1.96 * (var / done) ** 0.5
        ratio = llr(mean, var, done, args.elo0, args.elo1)
        print(f"openings {done:5d}  games {tally['games']:6d}  Elo {elo_of(mean):+6.1f} +/- {elo_interval(mean, ci):5.1f}  "
              f"LLR {ratio:+5.2f} [{lo:+.2f},{hi:+.2f}]  {test}", flush=True)
        if not args.fixed and done >= args.min_openings and (ratio >= hi or ratio <= lo):
            decision = "H1" if ratio >= hi else "H0"
            break
    ci = 1.96 * (var / done) ** 0.5
    res = {"openings": done, **tally, "score": round(mean, 4), "ci95": round(ci, 4), "elo": round(elo_of(mean), 1),
           "elo_ci95": round(elo_interval(mean, ci), 1), "llr": round(llr(mean, var, done, args.elo0, args.elo1), 2)}
    v = verdict(decision, res["elo"], res["elo_ci95"], args)
    print(json.dumps(res))
    print(f"screen, {args.players} players, {res['games']} games on {done} openings at {args.sims} simulations: "
          f"Elo {res['elo']:+.1f} +/- {res['elo_ci95']:.1f} (95%), {test}: {v}")
    record("screen", {"players": args.players, "sims": args.sims, "seed": args.seed, "skip": args.skip,
                      "book": os.path.basename(book_for(args)) or "uniform spawns", "extra": args.extra,
                      "test": {"elo0": args.elo0, "elo1": args.elo1, "alpha": args.alpha, "beta": args.beta,
                               "fixed": args.fixed, "cap": cap, "decision": decision},
                      "result": res, "verdict": v.split(" (")[0]})


def summarise(log: str, n: int, elo0: float = 0.0, elo1: float = 10.0) -> dict:
    """The candidate's win rate and Elo from a referee log, each opening counted once; the
    likelihood ratio is the referee's own, over single games."""
    seen, blocks, repeats, per_game = set(), [], 0, []
    endings: dict[str, int] = {}
    nodes = {"dev": [0, 0], "base": [0, 0]}
    with open(log, encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            key = tuple(map(tuple, b["start"]))
            if key in seen:
                repeats += 1
                continue
            seen.add(key)
            blocks.append(sum(b["dev_scores"]) / len(b["dev_scores"]))
            per_game.extend(b["dev_scores"])
            for g in b["games"]:
                why = g["reason"]
                if any(w in why for w in ("timeout", "crash", "invalid")):
                    who = "new" if why.startswith("dev") else "old"
                    what = next(w for w in ("timeout", "crash", "invalid") if w in why)
                    endings[f"{who} {what}"] = endings.get(f"{who} {what}", 0) + 1
            for k in nodes:
                nodes[k][0] += int(b.get(f"{k}_nodes") or 0)
                nodes[k][1] += int(b.get(f"{k}_ms") or 0)
    m = len(blocks)
    mean = sum(blocks) / max(m, 1)
    var = sum((x - mean) ** 2 for x in blocks) / max(m - 1, 1)
    ci = 1.96 * (var / max(m, 1)) ** 0.5
    g = len(per_game)
    g_mean = sum(per_game) / max(g, 1)
    g_var = sum((x - g_mean) ** 2 for x in per_game) / max(g - 1, 1)
    return {
        "openings": m, "games": g, "repeated_openings_dropped": repeats, "win_rate": round(mean, 4),
        "ci95": round(ci, 4), "level": round(1 / n, 4), "elo": round(elo_of(mean, n), 1),
        "elo_ci95": round(elo_interval(mean, ci, n), 1), "llr": round(llr(g_mean, g_var, g, elo0, elo1, n), 2),
        "forfeits": endings,
        "sims_per_ms": {k: round(v[0] / v[1], 2) if v[1] else None for k, v in nodes.items()},
    }


def cmd_run(args: argparse.Namespace) -> None:
    new, _ = side("dev")
    old, _ = side("prev")
    os.makedirs(path("gate", "logs"), exist_ok=True)
    log = path("gate", "logs", f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}_{args.players}p.jsonl")
    env = dict(os.environ)
    env.pop("TRONZERO_NET", None)  # each side loads the net.bin beside it
    cmd = [sys.executable, os.path.join(ROOT, args.sprt), "--baseline", old, "--dev", new, "--players", str(args.players),
           "--budget-ms", str(args.budget_ms), "--book", book_for(args) or "none", "--seed-plies", "0",
           "--max-games", str(args.games), "--concurrency", str(args.workers), "--seed", str(args.seed), "--log", log,
           "--print-every", str(args.print_every), "--elo0", str(args.elo0), "--elo1", str(args.elo1),
           "--alpha", str(args.alpha), "--beta", str(args.beta)]
    if args.fixed:
        cmd.append("--fixed")
    code = subprocess.run(cmd, env=env).returncode  # sprt.py: 0 accepted H1 (or fixed), 1 accepted H0, 2 undecided
    if code not in (0, 1, 2):
        sys.exit(f"the referee failed (exit code {code})")
    decision = "fixed" if args.fixed else ("H1", "H0", "undecided")[code]
    test = "fixed length" if args.fixed else f"SPRT[{args.elo0:g},{args.elo1:g}]"
    res = summarise(log, args.players, args.elo0, args.elo1)
    v = verdict(decision, res["elo"], res["elo_ci95"], args)
    print(json.dumps(res))
    print(f"referee, {args.players} players at {args.budget_ms} ms, {res['games']} games on {res['openings']} openings: "
          f"Elo {res['elo']:+.1f} +/- {res['elo_ci95']:.1f} (95%; wins {100 * res['win_rate']:.1f}%, level {100 / args.players:.1f}%), "
          f"{test}: {v}")
    record("run", {"players": args.players, "budget_ms": args.budget_ms, "seed": args.seed,
                   "book": os.path.basename(book_for(args)) or "uniform spawns",
                   "test": {"elo0": args.elo0, "elo1": args.elo1, "alpha": args.alpha, "beta": args.beta,
                            "fixed": args.fixed, "cap": args.games, "decision": decision},
                   "log": os.path.relpath(log, ROOT).replace("\\", "/"), "result": res, "verdict": v.split(" (")[0]})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("freeze", "dev"):
        p = sub.add_parser(name)
        p.add_argument("--exe", default=BUILT)
        p.add_argument("--net", default="runs/n1/nets/0040.bin")
        p.add_argument("--note", default="", help="one line on what this build is")
        p.add_argument("--force", action="store_true")
    p = sub.add_parser("books")
    p.add_argument("--players", type=int, nargs="+", default=[2, 3, 4], choices=(2, 3, 4))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=14)
    p = sub.add_parser("screen")
    p.add_argument("--players", type=int, default=4, choices=(2, 3, 4))
    p.add_argument("--openings", type=int, default=0, help="the cap: openings played at most (default 600; duels 3000)")
    p.add_argument("--batch", type=int, default=0, help="openings between two looks at the result (default 25; duels 100)")
    p.add_argument("--min-openings", type=int, default=50, help="no decision before this many openings")
    p.add_argument("--sims", type=int, default=0,
                   help="simulations per move (default: what a CodinGame turn gives, 1250 with four players, "
                        "1550 with three, 1800 in duels)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--skip", type=int, default=0, help="start this far into the seed's order of book openings")
    p.add_argument("--book", default="", help="opening book, or 'none' for uniform spawns")
    p.add_argument("--threads", type=int, default=14)
    p.add_argument("extra", nargs="*", help="after --: settings for `tronzero match`, such as --c-puct-a 0.6")
    p = sub.add_parser("run")
    p.add_argument("--players", type=int, default=4, choices=(2, 3, 4))
    p.add_argument("--games", type=int, default=3000, help="the cap: games played at most")
    p.add_argument("--budget-ms", type=int, default=52, help="ms per move; 52 here stands for CodinGame's 95")
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--book", default="", help="opening book, or 'none' for uniform spawns")
    p.add_argument("--sprt", default="legacy/alpha-beta/tools/sprt.py")
    for name in ("screen", "run"):
        p = sub.choices[name]
        p.add_argument("--elo0", type=float, default=0.0, help="H0: the new side is worth this much Elo")
        p.add_argument("--elo1", type=float, default=10.0, help="H1: it is worth this much")
        p.add_argument("--alpha", type=float, default=0.05, help="chance of accepting H1 when H0 holds")
        p.add_argument("--beta", type=float, default=0.05, help="chance of accepting H0 when H1 holds")
        p.add_argument("--fixed", action="store_true", help="play the whole cap, no early stop")
    args = ap.parse_args()
    if args.cmd in ("freeze", "dev"):
        stage("prev" if args.cmd == "freeze" else "dev", args)
    else:
        {"books": cmd_books, "screen": cmd_screen, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
