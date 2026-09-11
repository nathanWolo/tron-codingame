#!/usr/bin/env python3
"""Print board at a given ply (or at separation) for a game in an SPRT jsonl log.

usage: show_game.py LOG GAME_INDEX [PLY|sep|end]
"""
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from analyze_games import DIRS, flood, W, H  # noqa: E402


def load(path, idx):
    i = 0
    for line in open(path):
        d = json.loads(line)
        for g in d["games"]:
            if i == idx:
                return g
            i += 1
    raise SystemExit("no such game")


def board_at(g, upto):
    heads = [tuple(s) for s in g["start"]]
    owner = {h: p for p, h in enumerate(heads)}
    order = {}
    for i, m in enumerate(g["moves"][:upto]):
        p = int(m[0])
        dx, dy = DIRS[m[1]]
        heads[p] = (heads[p][0] + dx, heads[p][1] + dy)
        owner[heads[p]] = p
        order[heads[p]] = i
    return heads, owner


def sep_ply(g):
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
            return i + 1
    return len(g["moves"])


def show(g, upto):
    heads, owner = board_at(g, upto)
    occ = set(owner)
    fl = [flood(occ, h) for h in heads]
    print(f"ply {upto}/{len(g['moves'])}  heads {heads}  flood {[len(f) for f in fl]}  winner {g['winner']} dev_seat {g['dev_seat']} ({g['first_player']} first)")
    print("   " + "".join(str(c % 10) for c in range(W)))
    for y in range(H):
        row = []
        for x in range(W):
            c = (x, y)
            if c in heads:
                row.append("A" if heads.index(c) == 0 else "B")
            elif c in owner:
                row.append("a" if owner[c] == 0 else "b")
            elif c in fl[0] and c in fl[1]:
                row.append(".")
            elif c in fl[0]:
                row.append(",")
            elif c in fl[1]:
                row.append(";")
            else:
                row.append(" ")
        print(f"{y:2d} " + "".join(row))


if __name__ == "__main__":
    g = load(sys.argv[1], int(sys.argv[2]))
    arg = sys.argv[3] if len(sys.argv) > 3 else "sep"
    if arg == "sep":
        upto = sep_ply(g)
    elif arg == "end":
        upto = len(g["moves"])
    else:
        upto = int(arg)
    show(g, upto)
