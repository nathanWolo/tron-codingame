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

The default budget, 56 ms, stands in for CodinGame's 95 ms on this desktop
(crossfish measured 1.69 times CodinGame's speed here with 7 games running),
so run at most 7 games at once and nothing else on the machine.

A result counts as a pass only if its whole 95% interval is above level
(0.5 for `screen`, 1/players for `run`); the interval comes from the spread
between openings, each counted once.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = "tronzero.exe" if os.name == "nt" else "tronzero"
BUILT = os.path.join("bin", "tronzero-dev.exe" if os.name == "nt" else "tronzero-dev")
BOOK_SETTINGS = {2: (4000, 0.15), 3: (4000, 0.15), 4: (8000, 0.10)}  # openings, largest |value|
BOOK_BY_DEFAULT = (2,)  # player counts that start from their balanced book unless told otherwise


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


def verdict(mean: float, ci: float, level: float) -> str:
    if mean - ci > level:
        return "PASS (the whole interval is above level)"
    if mean + ci < level:
        return "WORSE (the whole interval is below level)"
    return "NOT SHOWN (the interval includes level)"


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
    cmd = [exe, "match", "--a", net, "--b", old_net, "--players", str(args.players), "--pairs", str(args.openings),
           "--sims", str(args.sims), "--seed", str(args.seed), "--threads", str(args.threads)]
    if book_for(args):
        cmd += ["--book", book_for(args), "--skip", str(args.skip)]
    cmd += args.extra
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1]
    res = json.loads(out)
    print(out)
    v = verdict(res["score"], res["ci95"], 0.5)
    print(f"screen, {args.players} players, {res['games']} games on {res['openings']} openings at {args.sims} simulations: "
          f"the new side scores {res['score']:.4f} +/- {res['ci95']:.4f} (level 0.5): {v}")
    record("screen", {"players": args.players, "sims": args.sims, "seed": args.seed, "skip": args.skip,
                      "book": os.path.basename(book_for(args)) or "uniform spawns",
                      "extra": args.extra, "result": res, "verdict": v.split(" (")[0]})


def summarise(log: str, n: int) -> dict:
    """Win rate of the candidate from a referee log, each opening counted once."""
    seen, blocks, repeats = set(), [], 0
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
    return {
        "openings": m, "games": m * n, "repeated_openings_dropped": repeats, "win_rate": round(mean, 4),
        "ci95": round(1.96 * (var / max(m, 1)) ** 0.5, 4), "level": round(1 / n, 4), "forfeits": endings,
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
           "--budget-ms", str(args.budget_ms), "--book", book_for(args) or "none", "--seed-plies", "0", "--fixed",
           "--max-games", str(args.games), "--concurrency", str(args.workers), "--seed", str(args.seed), "--log", log,
           "--print-every", str(args.print_every)]
    subprocess.run(cmd, check=True, env=env)
    res = summarise(log, args.players)
    v = verdict(res["win_rate"], res["ci95"], 1 / args.players)
    print(json.dumps(res))
    print(f"referee, {args.players} players at {args.budget_ms} ms, {res['games']} games on {res['openings']} openings: "
          f"the new build wins {100 * res['win_rate']:.1f}% +/- {100 * res['ci95']:.1f} (level {100 / args.players:.1f}%): {v}")
    record("run", {"players": args.players, "budget_ms": args.budget_ms, "seed": args.seed,
                   "book": os.path.basename(book_for(args)) or "uniform spawns",
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
    p.add_argument("--openings", type=int, default=200)
    p.add_argument("--sims", type=int, default=2000, help="simulations per move (about what 95 ms buys on CodinGame)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--skip", type=int, default=0, help="start this far into the seed's order of book openings")
    p.add_argument("--book", default="", help="opening book, or 'none' for uniform spawns")
    p.add_argument("--threads", type=int, default=14)
    p.add_argument("extra", nargs="*", help="after --: settings for `tronzero match`, such as --c-puct-a 0.6")
    p = sub.add_parser("run")
    p.add_argument("--players", type=int, default=4, choices=(2, 3, 4))
    p.add_argument("--games", type=int, default=1200)
    p.add_argument("--budget-ms", type=int, default=56)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--book", default="", help="opening book, or 'none' for uniform spawns")
    p.add_argument("--sprt", default="legacy/alpha-beta/tools/sprt.py")
    args = ap.parse_args()
    if args.cmd in ("freeze", "dev"):
        stage("prev" if args.cmd == "freeze" else "dev", args)
    else:
        {"books": cmd_books, "screen": cmd_screen, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
