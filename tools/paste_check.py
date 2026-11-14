"""Check that two builds of the agent play the same moves.

    python tools/paste_check.py --a "bin/tronzero-paste.exe" \
        --b "bin/tronzero-dev.exe --net paste/net_q.bin" [--games 12] [--sims 200]

Both engines sit in the same seat of the same games and get the same frames
over the CodinGame protocol, with `TRON_SIMS` fixing the simulations per move
so their searches are deterministic. The other seats are simple wanderers
(keep straight, turn at random, never step into a wall while a free cell
exists). The move of engine A is the one played.

Reports the moves compared, the moves that differed, and any move of the
agent that crashed while a legal one existed.

Before the games, every engine gets one scripted position that its frame
reader must get right: a bike dies, and before the agent's next frame another
bike steps onto a cell the dead bike's trail has just vacated. An engine that
places heads before it removes the dead bike's trail loses that head from its
board and walks into it.

Without `--b` it times one engine instead: no `TRON_SIMS`, so the engine
thinks for its own budget, and the time from sending a frame to reading the
move is measured for every turn.

    python tools/paste_check.py --a "bin/tronzero-paste.exe" --games 6
"""

from __future__ import annotations

import argparse
import os
import random
import shlex
import subprocess
import sys
import time

W, H = 30, 20
DIRS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}


def start(cmd: str, sims, extra_env: dict) -> subprocess.Popen:
    env = dict(os.environ, **extra_env)
    if sims is not None:
        env["TRON_SIMS"] = str(sims)
    argv = shlex.split(cmd, posix=False)
    argv[0] = os.path.abspath(argv[0])
    return subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env,
    )


def play(cmds: list[str], sims, rng: random.Random, envs: list[dict], times: list) -> tuple[int, int, int, int]:
    n = rng.choice([2, 3, 4, 4])
    hero = rng.randrange(n)
    cells = rng.sample(range(W * H), n)
    start_xy = [(c % W, c // W) for c in cells]
    head = list(start_xy)
    occ = [[False] * W for _ in range(H)]
    trail: list[list[tuple[int, int]]] = [[xy] for xy in start_xy]
    for x, y in start_xy:
        occ[y][x] = True
    alive = [True] * n
    last = [None] * n
    engines = [start(c, sims, e) for c, e in zip(cmds, envs)]
    compared = differed = blunders = plies = 0

    def free(x: int, y: int) -> bool:
        return 0 <= x < W and 0 <= y < H and not occ[y][x]

    def legal(p: int) -> list[str]:
        return [d for d, (dx, dy) in DIRS.items() if free(head[p][0] + dx, head[p][1] + dy)]

    def kill(p: int) -> None:
        alive[p] = False
        for x, y in trail[p]:
            occ[y][x] = False
        trail[p] = []

    try:
        while sum(alive) > 1 and alive[hero]:
            for p in range(n):
                if not alive[p] or sum(alive) <= 1:
                    continue
                moves = legal(p)
                if p == hero:
                    frame = f"{n} {p}\n" + "".join(
                        (f"{start_xy[q][0]} {start_xy[q][1]} {head[q][0]} {head[q][1]}\n" if alive[q]
                         else "-1 -1 -1 -1\n")
                        for q in range(n)
                    )
                    answers = []
                    for e in engines:
                        t = time.perf_counter()
                        e.stdin.write(frame)
                        e.stdin.flush()
                        answers.append(e.stdout.readline().strip())
                        times.append((compared == 0, (time.perf_counter() - t) * 1000.0))
                    if moves:
                        compared += 1
                        differed += answers[0] != answers[-1]
                        blunders += answers[0] not in moves
                    choice = answers[0]
                elif not moves:
                    choice = "UP"
                elif last[p] in moves and rng.random() < 0.8:
                    choice = last[p]
                else:
                    choice = rng.choice(moves)
                if choice not in moves:
                    kill(p)
                    continue
                dx, dy = DIRS[choice]
                head[p] = (head[p][0] + dx, head[p][1] + dy)
                occ[head[p][1]][head[p][0]] = True
                trail[p].append(head[p])
                last[p] = choice
                plies += 1
    finally:
        for e in engines:
            e.kill()
    return compared, differed, blunders, plies


def vacated_cell_case(cmd: str, env: dict) -> bool:
    """Seat 2 crashes in round 6; in round 7 seat 0 steps onto (11, 10), which
    seat 2's trail held. The agent, seat 1 at (10, 10), then has one legal
    move, UP; RIGHT is seat 0's new head. Its earlier answers are ignored:
    the script moves it."""
    start_xy = [(9, 11), (10, 16), (14, 10)]
    script = [
        ["UP", "UP", "UP", "RIGHT", "RIGHT", "DOWN", "DOWN"],
        ["UP"] * 6,
        ["LEFT", "LEFT", "LEFT", "DOWN", "DOWN", "UP"],
    ]
    head = list(start_xy)
    trail = [[xy] for xy in start_xy]
    alive = [True] * 3
    engine = start(cmd, 200, env)
    try:
        for r in range(7):
            for p in range(3):
                if not alive[p]:
                    continue
                if p == 1:
                    engine.stdin.write("3 1\n" + "".join(
                        (f"{start_xy[q][0]} {start_xy[q][1]} {head[q][0]} {head[q][1]}\n" if alive[q]
                         else "-1 -1 -1 -1\n")
                        for q in range(3)))
                    engine.stdin.flush()
                    answer = engine.stdout.readline().strip()
                    if r == 6:
                        return answer == "UP"
                dx, dy = DIRS[script[p][r]]
                cell = (head[p][0] + dx, head[p][1] + dy)
                if any(cell in t for t in trail):
                    alive[p] = False
                    trail[p] = []
                    continue
                head[p] = cell
                trail[p].append(cell)
    finally:
        engine.kill()
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", required=True, help="command line of the first engine")
    ap.add_argument("--b", default="", help="command line of the second engine; without it, time the first")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--env-a", default="", help="NAME=VALUE set for the first engine only")
    args = ap.parse_args()
    env_a = dict([args.env_a.split("=", 1)]) if args.env_a else {}
    rng = random.Random(args.seed)
    total = [0, 0, 0, 0]
    times: list = []
    cmds = [args.a, args.b] if args.b else [args.a]
    reader_ok = True
    for name, cmd, env in zip("ab", cmds, [env_a, {}]):
        ok = vacated_cell_case(cmd, env)
        reader_ok = reader_ok and ok
        print(f"frame reader, engine {name}: {'ok' if ok else 'FAILED: it stepped onto a cell another bike had just taken'}")
    for g in range(args.games):
        r = play(cmds, args.sims if args.b else None, rng, [env_a, {}], times)
        total = [t + x for t, x in zip(total, r)]
    if args.b:
        print(f"{args.games} games, {total[3]} plies: {total[0]} agent moves compared, "
              f"{total[1]} differed, {total[2]} crashed with a legal move available")
        sys.exit(1 if total[1] or total[2] or not reader_ok else 0)
    firsts = sorted(ms for first, ms in times if first)
    rest = sorted(ms for first, ms in times if not first)
    print(f"{args.games} games, {total[0]} agent moves, {total[2]} crashed with a legal move available")
    print(f"  first turns: max {firsts[-1]:.1f} ms, median {firsts[len(firsts) // 2]:.1f} ms")
    print(f"  later turns: max {rest[-1]:.1f} ms, 99th percentile {rest[int(len(rest) * 0.99)]:.1f} ms, "
          f"median {rest[len(rest) // 2]:.1f} ms; over 100 ms: {sum(ms > 100 for ms in rest + firsts)}")
    sys.exit(1 if total[2] or not reader_ok else 0)


if __name__ == "__main__":
    main()
