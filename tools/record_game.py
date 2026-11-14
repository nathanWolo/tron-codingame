#!/usr/bin/env python3
"""Record matches for the replay page and the video.

One tronzero agent plays alpha-beta bots on the alpha-beta engine's referee,
driven exactly as its viewer (legacy/alpha-beta/tools/watch.py) drives it: the CodinGame
protocol over stdin/stdout, a crash or a timeout removes the bike and its
trail. Nothing is drawn; every move is written to a JSON file together with
what the engine printed about its search, for video/replay.html to play back.

    python tools/record_game.py --games 24 --seed 1 --out video/games

The agent thinks for a fixed number of simulations (--sims) so that a recording
does not depend on what else the machine is doing; the alpha-beta bots get
--budget-ms each move. The agent must be a build that prints its tree's
statistics when TZ_STATS is set (any build of this engine outside the paste).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "legacy", "alpha-beta", "tools"))

import sprt  # noqa: E402
import watch  # noqa: E402

AGENT = "tronzero"


class RecEngine(watch.WatchEngine):
    """The viewer's engine wrapper, made to say whether this move's line arrived.

    The viewer keeps each engine's last stderr line and waits 8 ms for it; on a
    busy machine the line can come later and would then be shown one move
    late. Here the referee waits for the line itself.
    """

    def __init__(self, path: str, budget_ms: int, label: str):
        self.fresh = threading.Event()
        self.is_agent = label == AGENT
        super().__init__(path, budget_ms, label)

    def _on_stderr_line(self, line: str) -> None:
        line = line.rstrip("\n")
        if not line:
            return
        if self.is_agent and "nodes=" not in line and " tz " not in line:
            return  # the plain line; the one with the tree's statistics follows it
        super()._on_stderr_line(line)
        self.fresh.set()

    def send(self, text: str) -> None:
        self.fresh.clear()
        super().send(text)

    def sync_stderr(self) -> None:
        # The agent prints a line for every move it has; the alpha-beta bot prints none once it is
        # only filling space, so it is not waited on for long.
        got = self.fresh.wait(0.3 if self.is_agent else 0.03)
        self.last_mm = dict(self.last_mm, fresh=got)


watch.WatchEngine = RecEngine  # play_watched_game looks the class up by name


def parse_stat(raw: str, is_agent: bool) -> dict:
    """An engine's stderr line as the fields the replay shows."""
    words = raw.split()
    kv = dict(w.split("=", 1) for w in words if "=" in w)
    if not is_agent:
        out = {"mode": "ab"}
        if "n" in kv and kv["n"].isdigit():
            out["nodes"] = int(kv["n"])
        if "d" in kv and kv["d"].isdigit():
            out["depth"] = int(kv["d"])
        for w in words:
            if w.endswith("ms") and w[:-2].isdigit():
                out["ms"] = int(w[:-2])
        return out
    if "mode" in kv:
        return {"mode": kv["mode"]}  # forced: one legal move; fill: alone in its region
    out = {"mode": "search"}
    try:
        out["q"] = int(words[2]) / 1000.0
        out["ms"] = int(next(w for w in words if w.endswith("ms"))[:-2])
        out["sims"] = int(kv["n"])
        out["nodes"] = int(kv["nodes"])
        out["maxd"] = int(kv["maxd"])
        out["avgd"] = float(kv["avgd"])
        out["pvd"] = int(kv["pvd"])
        out["v"] = [float(t) for t in kv["v"].split(",")]
        out["vis"] = [int(t) for t in kv["vis"].split(",")]
        out["dh"] = [int(t) for t in kv["dh"].split(",")] if "dh" in kv else []
        out["pv"] = [[int(t) for t in step.split(":")] for step in kv.get("pv", "").split(";") if step]
    except (KeyError, ValueError, IndexError, StopIteration):
        return {"mode": "unparsed", "raw": raw}
    return out


def play(args: argparse.Namespace, index: int) -> dict:
    n = args.players
    rng = random.Random(args.seed * 1000003 + index)
    starts = sprt.pick_starts(rng, (), n)
    seat = index % n if args.seat < 0 else args.seat
    labels, paths, k = [], [], 0
    for p in range(n):
        if p == seat:
            labels.append(AGENT)
            paths.append(os.path.join(ROOT, args.agent))
        else:
            k += 1
            labels.append(f"alpha-beta {k}")
            paths.append(os.path.join(ROOT, args.baseline))
    ctrl = watch.Control()
    ctrl.delay_ms = 0
    out: queue.Queue = queue.Queue()
    t0 = time.time()
    watch.play_watched_game(paths, starts, labels, [p == seat for p in range(n)], args.budget_ms,
                            args.timeout, [[] for _ in range(n)], ctrl, out)
    plies, over = [], None
    while not out.empty():
        s = out.get()
        if s["over"]:
            over = s["over"]
        if s["thinking"] or s["mover"] < 0:
            continue
        p = s["mover"]
        ply = {"p": p, "round": s["turn"], "mv": s["move"], "died": not s["alive"][p],
               "territory": s["territory"]}
        if s["alive"][p]:
            ply["x"], ply["y"] = s["heads"][p]
        mm = s["mm"][p]
        if mm.get("fresh") and mm.get("raw"):
            ply["stat"] = parse_stat(mm["raw"], p == seat)
        plies.append(ply)
    return {
        "meta": {"index": index, "seed": args.seed, "players": n, "agent_seat": seat, "labels": labels,
                 "sims": args.sims, "budget_ms": args.budget_ms, "net": args.net, "width": sprt.W, "height": sprt.H,
                 "recorded": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t0, 1)},
        "starts": [list(c) for c in starts],
        "plies": plies,
        "over": over,
    }


def describe(game: dict) -> dict:
    """What makes a game worth watching, for choosing one out of a batch."""
    seat = game["meta"]["agent_seat"]
    n = game["meta"]["players"]
    out_round = [None] * n
    for ply in game["plies"]:
        if ply["died"]:
            out_round[ply["p"]] = ply["round"]
    rounds = game["plies"][-1]["round"] + 1 if game["plies"] else 0
    deaths = sorted(r for r in out_round if r is not None)
    evals = [(ply["round"], ply["stat"]["v"][seat]) for ply in game["plies"]
             if ply["p"] == seat and ply.get("stat", {}).get("mode") == "search"]
    after = [v for r, v in evals if r >= 10]
    searched = len(evals)
    mine = sum(1 for ply in game["plies"] if ply["p"] == seat)
    over = game["over"] or {}
    return {
        "file": game.get("file", ""),
        "seat": seat,
        "won": over.get("winner") == seat,
        "reason": over.get("reason"),
        "rounds": rounds,
        "plies": len(game["plies"]),
        "out_rounds": out_round,
        "duel_rounds": rounds - deaths[-2] if len(deaths) >= 2 else 0,
        "first_out": deaths[0] if deaths else None,
        "eval_low": round(min(after), 2) if after else None,
        "eval_early": round(sum(v for r, v in evals if r < 10) / max(1, sum(1 for r, _ in evals if r < 10)), 2),
        "searched_share": round(searched / max(1, mine), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--first", type=int, default=0, help="index of the first game (spawns follow from seed and index)")
    ap.add_argument("--players", type=int, default=4, choices=(2, 3, 4))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seat", type=int, default=-1, help="the agent's seat; by default it rotates game by game")
    ap.add_argument("--agent", default="bin/tronzero-rec.exe")
    ap.add_argument("--net", default="runs/n1/nets/0040.bin")
    ap.add_argument("--sims", type=int, default=2000, help="the agent's simulations a move")
    ap.add_argument("--baseline", default="bin/tron-baseline.exe")
    ap.add_argument("--budget-ms", type=int, default=100, help="the alpha-beta bots' time a move")
    ap.add_argument("--timeout", type=float, default=3.0, help="seconds before the referee gives up on a move")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--out", default="video/games")
    args = ap.parse_args()

    os.environ["TZ_STATS"] = "1"
    os.environ["TRON_SIMS"] = str(args.sims)
    os.environ["TRONZERO_NET"] = os.path.join(ROOT, args.net)
    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    def one(index: int) -> dict:
        game = play(args, index)
        name = f"s{args.seed}_g{index:03d}.json"
        game["file"] = name
        with open(os.path.join(out_dir, name), "w", encoding="utf-8", newline="\n") as f:
            json.dump(game, f, separators=(",", ":"))
        d = describe(game)
        print(json.dumps(d), flush=True)
        return d

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(one, range(args.first, args.first + args.games)))
    wins = sum(1 for d in rows if d["won"])
    print(f"agent won {wins} of {len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
