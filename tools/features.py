#!/usr/bin/env python3
"""Extract eval features from logged 1v1 games for offline weight fitting.

usage: features.py OUT.npz LOG.jsonl [LOG.jsonl ...]

For each decisive 2-player game, positions at a few plies before separation
are featurised from player 0's point of view; label = 1 if player 0 won.
"""
import json
import sys
from collections import deque

import numpy as np

W, H = 30, 20
DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
NEAR, FAR = 8, 16


def bfs(occ, head):
    d = {}
    q = deque()
    for dx, dy in DIRS:
        c = (head[0] + dx, head[1] + dy)
        if 0 <= c[0] < W and 0 <= c[1] < H and c not in occ:
            d[c] = 1
            q.append(c)
    while q:
        x, y = q.popleft()
        for dx, dy in DIRS:
            c = (x + dx, y + dy)
            if 0 <= c[0] < W and 0 <= c[1] < H and c not in occ and c not in d:
                d[c] = d[(x, y)] + 1
                q.append(c)
    return d


def empty_deg(occ, c):
    n = 0
    for dx, dy in DIRS:
        x, y = c[0] + dx, c[1] + dy
        if 0 <= x < W and 0 <= y < H and (x, y) not in occ:
            n += 1
    return n


def dead_end_loss(owned, walk):
    """Peel tips (<=1 walkable neighbour); each round with k tips loses k-1."""
    rem = set(owned)
    walk = set(walk)
    lost = 0
    for _ in range(40):
        tips = []
        for c in rem:
            n = 0
            for dx, dy in DIRS:
                if (c[0] + dx, c[1] + dy) in walk:
                    n += 1
            if n <= 1:
                tips.append(c)
        if len(tips) <= 1:
            break
        lost += len(tips) - 1
        for c in tips:
            rem.discard(c)
            walk.discard(c)
    return lost


def features(occ, heads, to_move):
    d0 = bfs(occ, heads[0])
    d1 = bfs(occ, heads[1])
    if not (set(d0) & set(d1)):
        return None
    own = [set(), set()]
    ties = 0
    near = [0, 0]
    far = [0, 0]
    for c in set(d0) | set(d1):
        a = d0.get(c, 9999)
        b = d1.get(c, 9999)
        if a < b:
            own[0].add(c)
            if a <= NEAR:
                near[0] += 1
            if a > FAR:
                far[0] += 1
        elif b < a:
            own[1].add(c)
            if b <= NEAR:
                near[1] += 1
            if b > FAR:
                far[1] += 1
        else:
            ties += 1
    terr = [len(own[0]), len(own[1])]
    edges = [sum(empty_deg(occ, c) for c in own[p]) for p in range(2)]
    mob = [empty_deg(occ, heads[p]) for p in range(2)]
    front = 0
    for c in own[0]:
        for dx, dy in DIRS:
            if (c[0] + dx, c[1] + dy) in own[1]:
                front += 1
                break
    center = [-(abs(heads[p][0] - 14) + abs(heads[p][1] - 9)) for p in range(2)]
    reach = [len(d0), len(d1)]
    empty_cells = set()
    for x in range(W):
        for y in range(H):
            if (x, y) not in occ:
                empty_cells.add((x, y))
    walk = empty_cells | set(heads)
    teeth = [dead_end_loss(own[p], walk) for p in range(2)]
    cb = []
    for p in range(2):
        even = sum(1 for c in own[p] if (c[0] + c[1]) % 2 == 0)
        odd = len(own[p]) - even
        start = (heads[p][0] + heads[p][1] + 1) & 1
        same, other = (even, odd) if start == 0 else (odd, even)
        cb.append(len(own[p]) - (2 * min(same, other) + (1 if same > other else 0)))
    wall_adj = [sum(1 for c in own[p] if empty_deg(occ, c) < 4) for p in range(2)]
    corridor = []
    for p in range(2):
        n = 0
        for c in own[p]:
            up = (c[0], c[1] - 1) in empty_cells
            dn = (c[0], c[1] + 1) in empty_cells
            lf = (c[0] - 1, c[1]) in empty_cells
            rt = (c[0] + 1, c[1]) in empty_cells
            if (up and dn and not lf and not rt) or (lf and rt and not up and not dn):
                n += 1
        corridor.append(n)
    occupied = len(occ)
    return [
        terr[0] - terr[1],
        edges[0] - edges[1],
        mob[0] - mob[1],
        front,
        center[0] - center[1],
        ties,
        near[0] - near[1],
        far[0] - far[1],
        teeth[0] - teeth[1],
        cb[0] - cb[1],
        reach[0] - reach[1],
        1 if to_move == 0 else -1,
        occupied,
        corridor[0] - corridor[1],
        wall_adj[0] - wall_adj[1],
        (terr[0] - terr[1]) * (500 - occupied) / 500.0,
    ]


NAMES = [
    "terr", "edges", "mob", "front", "center", "ties", "near", "far", "teeth", "cb",
    "reach", "tomove", "occupied", "corridor", "wall_adj", "terr_x_phase",
]


def main():
    out = sys.argv[1]
    X = []
    y = []
    games = 0
    for path in sys.argv[2:]:
        for line in open(path):
            d = json.loads(line)
            for g in d["games"]:
                if g["n_players"] != 2 or g["winner"] is None or "moves" not in g:
                    continue
                games += 1
                heads = [tuple(s) for s in g["start"]]
                occ = set(heads)
                for i, m in enumerate(g["moves"]):
                    p = int(m[0])
                    dx, dy = DIRS["UDLR".index(m[1])]
                    heads[p] = (heads[p][0] + dx, heads[p][1] + dy)
                    occ.add(heads[p])
                    if i in (17, 29, 41, 53, 65):
                        f = features(occ, heads, 1 - p)
                        if f is None:
                            break
                        X.append(f)
                        y.append(1 if g["winner"] == 0 else 0)
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    np.savez(out, X=X, y=y, names=np.array(NAMES))
    print(f"{games} games, {len(y)} positions, P(p0 wins)={y.mean():.3f}")


if __name__ == "__main__":
    main()
