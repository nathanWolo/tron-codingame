#!/usr/bin/env python3
"""Replay SPRT jsonl games (with `moves`) and report separation statistics."""
import json
import sys
from collections import Counter

W, H = 30, 20
DIRS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}


def flood(occ, head):
    hx, hy = head
    seen = set()
    q = []
    for dx, dy in DIRS.values():
        x, y = hx + dx, hy + dy
        if 0 <= x < W and 0 <= y < H and (x, y) not in occ:
            q.append((x, y))
            seen.add((x, y))
    i = 0
    while i < len(q):
        x, y = q[i]
        i += 1
        for dx, dy in DIRS.values():
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in occ and (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append((nx, ny))
    return seen


def replay(g):
    n = g["n_players"]
    if n != 2:
        return None
    heads = [tuple(s) for s in g["start"]]
    occ = set(heads)
    trails = [[h] for h in heads]
    alive = [True, True]
    sep_turn = None
    sep_info = None
    steps_after = [0, 0]
    for i, m in enumerate(g["moves"]):
        p = int(m[0])
        dx, dy = DIRS[m[1]]
        nx, ny = heads[p][0] + dx, heads[p][1] + dy
        occ.add((nx, ny))
        heads[p] = (nx, ny)
        trails[p].append((nx, ny))
        if sep_turn is not None:
            steps_after[p] += 1
        else:
            f0 = flood(occ, heads[0])
            f1 = flood(occ, heads[1])
            adj = abs(heads[0][0] - heads[1][0]) + abs(heads[0][1] - heads[1][1]) == 1
            if not adj and not (f0 & f1):
                sep_turn = i
                sep_info = (len(f0), len(f1), p)  # p = who just moved
    return sep_turn, sep_info, steps_after


def main():
    total = 0
    sep = 0
    margins = Counter()
    upset = 0
    equal_space = 0
    ratio = []
    for path in sys.argv[1:]:
        for line in open(path):
            d = json.loads(line)
            for g in d["games"]:
                if "moves" not in g:
                    continue
                r = replay(g)
                if r is None:
                    continue
                total += 1
                sep_turn, info, after = r
                if sep_turn is None:
                    continue
                sep += 1
                f0, f1, mover = info
                # next to move is 1 - mover
                margins[min(abs(f0 - f1), 20)] += 1
                winner = g["winner"]
                bigger = 0 if f0 > f1 else 1 if f1 > f0 else None
                if bigger is None:
                    equal_space += 1
                elif winner != bigger:
                    upset += 1
                # fill efficiency: steps taken after separation / flood size
                ratio.append((after[0] / max(f0, 1), after[1] / max(f1, 1), f0, f1, after))
    print(f"games {total}, separated {sep} ({100*sep/max(total,1):.0f}%)")
    print(f"upsets (smaller flood at separation won): {upset}, equal-space cases {equal_space}")
    print("margin histogram (|f0-f1| capped 20):", sorted(margins.items()))
    # fill efficiency for the loser (who used all their space) and winner
    eff = []
    for r0, r1, f0, f1, after in ratio:
        eff.append(r0)
        eff.append(r1)
    eff.sort()
    print("fill efficiency quantiles (steps/flood):", [round(eff[int(len(eff) * q)], 2) for q in (0.1, 0.25, 0.5, 0.75, 0.9)])
    small = [(f0, f1, after) for _, _, f0, f1, after in ratio if abs(f0 - f1) <= 3]
    print("close cases (margin<=3):", len(small))
    for s in small[:15]:
        print("  ", s)


if __name__ == "__main__":
    main()
