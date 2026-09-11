#!/usr/bin/env python3
"""At each game's separation ply, compare flood / approx_fill / checkerboard /
greedy rollout against the steps the loser actually survived."""
import json
import subprocess
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from analyze_games import DIRS, flood, W, H  # noqa: E402


def sep_state(g):
    heads = [tuple(s) for s in g["start"]]
    occ = set(heads)
    for i, m in enumerate(g["moves"]):
        p = int(m[0])
        dx, dy = DIRS[m[1]]
        heads[p] = (heads[p][0] + dx, heads[p][1] + dy)
        occ.add(heads[p])
        f0 = flood(occ, heads[0])
        f1 = flood(occ, heads[1])
        adj = abs(heads[0][0] - heads[1][0]) + abs(heads[0][1] - heads[1][1]) == 1
        if not adj and not (f0 & f1):
            after = [0, 0]
            for m2 in g["moves"][i + 1:]:
                after[int(m2[0])] += 1
            return i, heads, occ, (len(f0), len(f1)), after
    return None


def board_text(heads, occ):
    rows = []
    for y in range(H):
        row = []
        for x in range(W):
            c = (x, y)
            if c == heads[0]:
                row.append("A")
            elif c == heads[1]:
                row.append("B")
            elif c in occ:
                row.append("#")
            else:
                row.append(".")
        rows.append("".join(row))
    return "\n".join(rows) + "\n"


def main():
    binary = sys.argv[1]
    boards = []
    meta = []
    for path in sys.argv[2:]:
        for line in open(path):
            d = json.loads(line)
            for g in d["games"]:
                r = sep_state(g)
                if r is None:
                    continue
                i, heads, occ, fl, after = r
                boards.append(board_text(heads, occ))
                meta.append((g["winner"], fl, after, i))
    out = subprocess.run([binary, "--fill-eval"], input="".join(boards), capture_output=True, text=True).stdout
    lines = out.strip().split("\n")
    assert len(lines) == len(boards), (len(lines), len(boards))
    errs = {"approx": [], "cb": [], "greedy": [], "flood": [], "exact": []}
    sign_wrong = {"approx": 0, "cb": 0, "greedy": 0, "flood": 0, "exact": 0}
    exact_gap = []
    n = 0
    for (winner, fl, after, ply), line in zip(meta, lines):
        parts = []
        for side in line.split("|"):
            toks = side.split()
            vals = list(map(int, toks[:4]))
            ex = toks[4]
            vals.append(int(ex.lstrip("~")))
            vals.append(not ex.startswith("~"))
            parts.append(vals)
        loser = 1 - winner
        actual_l = after[loser]
        # loser used all their space; compare estimates for the loser
        f, a, c, gr, ex, ex_ok = parts[loser]
        if ex_ok:
            exact_gap.append((ex - actual_l, f, actual_l, ex, ply))
        errs["flood"].append(f - actual_l)
        errs["approx"].append(a - actual_l)
        errs["cb"].append(c - actual_l)
        errs["greedy"].append(gr - actual_l)
        # sign test: does estimate predict the winner? (winner needs >= loser steps; mover order ignored)
        for key, idx in (("flood", 0), ("approx", 1), ("cb", 2), ("greedy", 3)):
            est_w = parts[winner][idx]
            est_l = parts[loser][idx]
            if est_w < est_l:
                sign_wrong[key] += 1
        n += 1
    print(f"{n} separations")
    exact_gap.sort()
    print(f"exact solved for {len(exact_gap)} losers; gap (optimal - actual) histogram:")
    from collections import Counter
    print(sorted(Counter(min(g[0], 10) for g in exact_gap).items()))
    print("worst:", exact_gap[-8:])
    for key in ("flood", "cb", "approx", "greedy"):
        e = sorted(errs[key])
        q = [e[int(len(e) * x)] for x in (0.05, 0.25, 0.5, 0.75, 0.95)]
        mean_abs = sum(abs(v) for v in e) / len(e)
        print(f"{key:7s} est-actual quantiles {q}  mean|err| {mean_abs:.2f}  predicted wrong winner {sign_wrong[key]}")


if __name__ == "__main__":
    main()
