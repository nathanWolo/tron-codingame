#!/usr/bin/env python3
"""Replay a logged game to a given ply and ask an engine what it would play.

usage: replay_ask.py ENGINE LOG GAME_INDEX PLY [--budget-ms N]

Feeds the engine every frame it would have seen as the player to move at PLY
(so its Tracker is in sync), then prints the engine's move and stderr for
that turn. GAME_INDEX is 0-based over all games in the log (all blocks).
"""
import subprocess
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from analyze_games import DIRS  # noqa: E402
from show_game import load  # noqa: E402


def main():
    engine, log, idx, ply = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    budget = "20"
    if "--budget-ms" in sys.argv:
        budget = sys.argv[sys.argv.index("--budget-ms") + 1]
    g = load(log, idx)
    n = g["n_players"]
    start = [tuple(s) for s in g["start"]]
    heads = list(start)
    moves = g["moves"]
    mover = int(moves[ply][0])
    proc = subprocess.Popen(
        [engine, "--budget-ms", budget],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def frame():
        lines = [f"{n} {mover}"]
        for p in range(n):
            lines.append(f"{start[p][0]} {start[p][1]} {heads[p][0]} {heads[p][1]}")
        return "\n".join(lines) + "\n"

    def ask():
        proc.stdin.write(frame())
        proc.stdin.flush()
        return proc.stdout.readline().strip()

    for i, m in enumerate(moves[: ply + 1]):
        p = int(m[0])
        if p == mover:
            out = ask()
            if i == ply:
                proc.stdin.close()
                err = proc.stderr.read()
                print(f"ply {ply} player {mover} logged move {m}  engine says {out}")
                print("stderr last lines:")
                print("\n".join(err.strip().split("\n")[-3:]))
                proc.kill()
                return
        dx, dy = DIRS[m[1]]
        heads[p] = (heads[p][0] + dx, heads[p][1] + dy)


if __name__ == "__main__":
    main()
