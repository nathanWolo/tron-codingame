#!/usr/bin/env python3
"""Watch the candidate binary play the frozen baseline on a 30×20 board.

Reuses the SPRT referee (CodinGame stdin/stdout protocol, seed plies, death
clears the ribbon). Paints Voronoi ownership and whether the bikes still share
space, plus each engine's last stderr line (`DIR mm SCORE Tms n=`).

    tools/watch.sh                  # build candidate, then open the window
    python3 tools/watch.py          # if binaries already exist
    python3 tools/watch.py --players 3 --budget-ms 20
"""

from __future__ import annotations

import argparse
import os
import queue
import random
import select
import sys
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from typing import Optional

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

import sprt  # noqa: E402
from sprt import (  # noqa: E402
    DEFAULT_BOOK_BY_N,
    DIRS,
    Engine,
    H,
    W,
    fmt_nps,
    load_book,
    parse_search_stats,
    pick_starts,
    random_seed_moves,
)

CELL = 24
PAD = 10
BOARD_PX_W = W * CELL
BOARD_PX_H = H * CELL

# Role colours (not seat): candidate is always cyan.
CANDIDATE_TRAIL = "#3ec6ff"
CANDIDATE_HEAD = "#dff6ff"
CANDIDATE_ZONE = "#123544"
BASELINE_TRAIL = ("#ff9f1c", "#ff5d8f", "#c4f542")
BASELINE_HEAD = ("#ffe0b0", "#ffd0e0", "#e8ffb0")
BASELINE_ZONE = ("#3a2a10", "#3a1524", "#2a3510")
CONTESTED = "#2a2430"
EMPTY = "#10141c"
WALL = "#0b0e14"
GRID = "#1c2230"
BG = "#0b0e14"
PANEL = "#121722"
FG = "#d7dde8"
MUTED = "#8b95a8"
ACCENT = "#3ec6ff"
CUT = "#ff6b6b"
OPEN = "#7ee787"


def role_colors(is_candidate: bool, baseline_index: int) -> tuple[str, str, str]:
    if is_candidate:
        return CANDIDATE_TRAIL, CANDIDATE_HEAD, CANDIDATE_ZONE
    i = baseline_index % len(BASELINE_TRAIL)
    return BASELINE_TRAIL[i], BASELINE_HEAD[i], BASELINE_ZONE[i]


def in_bounds(x: int, y: int) -> bool:
    return 0 <= x < W and 0 <= y < H


def neighbours(x: int, y: int):
    for dx, dy in DIRS.values():
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny):
            yield nx, ny


def mobility(occ: list[list[bool]], head: tuple[int, int]) -> int:
    hx, hy = head
    n = 0
    for nx, ny in neighbours(hx, hy):
        if not occ[ny][nx]:
            n += 1
    return n


def bfs_empty(occ: list[list[bool]], head: tuple[int, int]) -> dict[tuple[int, int], int]:
    """Distance from empty neighbours of `head` (agent Voronoi seed)."""
    dist: dict[tuple[int, int], int] = {}
    q: deque[tuple[int, int]] = deque()
    hx, hy = head
    for nx, ny in neighbours(hx, hy):
        if not occ[ny][nx]:
            dist[(nx, ny)] = 1
            q.append((nx, ny))
    while q:
        x, y = q.popleft()
        d = dist[(x, y)]
        for nx, ny in neighbours(x, y):
            if occ[ny][nx] or (nx, ny) in dist:
                continue
            dist[(nx, ny)] = d + 1
            q.append((nx, ny))
    return dist


def shares_space(
    occ: list[list[bool]],
    a: tuple[int, int],
    b: tuple[int, int],
) -> bool:
    ax, ay = a
    bx, by = b
    if abs(ax - bx) + abs(ay - by) == 1:
        return True
    dist = bfs_empty(occ, a)
    for nx, ny in neighbours(bx, by):
        if not occ[ny][nx] and (nx, ny) in dist:
            return True
    return False


def analyse(
    occ: list[list[bool]],
    heads: list[tuple[int, int]],
    alive: list[bool],
) -> dict:
    n = len(heads)
    distances = [
        bfs_empty(occ, heads[p]) if alive[p] else {} for p in range(n)
    ]
    owner = [[-3] * W for _ in range(H)]  # -3 occupied, -2 unreached, -1 tie, >=0 player
    territory = [0] * n
    reachable = [0] * n
    for y in range(H):
        for x in range(W):
            if occ[y][x]:
                owner[y][x] = -3
                continue
            best = 10**9
            who = -2
            ties = 0
            for p in range(n):
                if not alive[p]:
                    continue
                d = distances[p].get((x, y))
                if d is None:
                    continue
                reachable[p] += 1
                if d < best:
                    best = d
                    who = p
                    ties = 1
                elif d == best:
                    ties += 1
            if ties == 0:
                owner[y][x] = -2
            elif ties == 1:
                owner[y][x] = who
                territory[who] += 1
            else:
                owner[y][x] = -1

    connected = False
    if n >= 2 and alive[0] and alive[1]:
        connected = shares_space(occ, heads[0], heads[1])
        if n > 2:
            for p in range(n):
                if not alive[p]:
                    continue
                for q in range(p + 1, n):
                    if alive[q] and shares_space(occ, heads[p], heads[q]):
                        connected = True
                        break
    return {
        "owner": owner,
        "territory": territory,
        "reachable": reachable,
        "mobility": [
            mobility(occ, heads[p]) if alive[p] else 0 for p in range(n)
        ],
        "connected": connected,
    }


def parse_mm_line(line: str) -> dict:
    """`DOWN mm 1840 75ms n=12345` → dir/score/ms/nodes. Missing fields stay None."""
    out: dict = {"raw": line, "dir": None, "score": None, "ms": None, "nodes": None}
    parts = line.split()
    if parts:
        out["dir"] = parts[0]
    if len(parts) >= 3 and parts[1] == "mm":
        try:
            out["score"] = int(parts[2])
        except ValueError:
            pass
    nodes, ms = parse_search_stats(line)
    if ms:
        out["ms"] = ms
    if nodes:
        out["nodes"] = nodes
    return out


class WatchEngine(Engine):
    def __init__(self, path: str, budget_ms: int, label: str):
        super().__init__(path, budget_ms, label, None)
        self.last_stderr = ""
        self.last_mm: dict = parse_mm_line("")

    def drain_stderr(self) -> None:
        err = self.proc.stderr
        if err is None:
            return
        fd = err.fileno()
        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                return
            line = err.readline()
            if line == "":
                return
            line = line.rstrip("\n")
            if not line:
                continue
            self.last_stderr = line
            self.last_mm = parse_mm_line(line)
            nodes, ms = parse_search_stats(line)
            if nodes > 0:
                self.nodes += nodes
                self.ms += max(ms, 1)


class Control:
    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.paused = False
        self.abort = False
        self.shutdown = False
        self.step = False
        self.delay_ms = 180
        self.auto_next = True

    def request_abort(self) -> None:
        with self.cond:
            self.abort = True
            self.cond.notify_all()

    def request_shutdown(self) -> None:
        with self.cond:
            self.shutdown = True
            self.abort = True
            self.cond.notify_all()

    def set_paused(self, paused: bool) -> None:
        with self.cond:
            self.paused = paused
            self.cond.notify_all()

    def request_step(self) -> None:
        with self.cond:
            self.paused = True
            self.step = True
            self.cond.notify_all()

    def wait_pace(self) -> bool:
        deadline: Optional[float] = None
        while True:
            with self.cond:
                if self.shutdown or self.abort:
                    return False
                if self.step:
                    self.step = False
                    return True
                if self.paused:
                    deadline = None
                    self.cond.wait(timeout=0.1)
                    continue
                if deadline is None:
                    deadline = time.time() + max(0, self.delay_ms) / 1000.0
                remain = deadline - time.time()
                if remain <= 0:
                    return True
                self.cond.wait(timeout=min(0.05, remain))


def snapshot_state(
    occ,
    trails,
    heads,
    alive,
    starts,
    labels,
    is_cand,
    engines,
    turn: int,
    mover: int,
    move: str,
    seed: bool,
    thinking: bool,
    over: Optional[dict] = None,
) -> dict:
    occ_copy = [row[:] for row in occ]
    analysis = analyse(occ_copy, list(heads), list(alive))
    mm = [dict(e.last_mm) for e in engines]
    return {
        "occ": occ_copy,
        "trails": [list(t) for t in trails],
        "heads": list(heads),
        "alive": list(alive),
        "starts": list(starts),
        "labels": list(labels),
        "is_cand": list(is_cand),
        "mm": mm,
        "turn": turn,
        "mover": mover,
        "move": move,
        "seed": seed,
        "thinking": thinking,
        "over": over,
        **analysis,
    }


def play_watched_game(
    paths: list[str],
    starts: list[tuple[int, int]],
    labels: list[str],
    is_cand: list[bool],
    budget_ms: int,
    turn_timeout: float,
    seed_moves: list[list[str]],
    ctrl: Control,
    out: queue.Queue,
) -> None:
    n = len(paths)
    engines = [WatchEngine(paths[i], budget_ms, labels[i]) for i in range(n)]
    occ = [[False] * W for _ in range(H)]
    trails: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    start = list(starts)
    head = list(starts)
    alive = [True] * n
    for i, (x, y) in enumerate(start):
        occ[y][x] = True
        trails[i].append((x, y))

    reason = "unknown"
    winner: Optional[int] = None
    tied: list[int] = []

    def emit(**kwargs) -> None:
        out.put(snapshot_state(
            occ, trails, head, alive, start, labels, is_cand, engines, **kwargs
        ))

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

    def frame_text(mover: int) -> str:
        lines = [f"{n} {mover}"]
        for p in range(n):
            lines.append(coords(p))
        return "\n".join(lines) + "\n"

    def step(p: int, token: str, why: str) -> bool:
        nonlocal reason
        if token not in DIRS:
            kill(p)
            reason = f"{labels[p]} {why}"
            return False
        dx, dy = DIRS[token]
        nx, ny = head[p][0] + dx, head[p][1] + dy
        if nx < 0 or nx >= W or ny < 0 or ny >= H or occ[ny][nx]:
            kill(p)
            reason = f"{labels[p]} crash"
            return False
        occ[ny][nx] = True
        head[p] = (nx, ny)
        trails[p].append((nx, ny))
        return True

    def query(p: int) -> Optional[str]:
        nonlocal reason
        engines[p].send(frame_text(p))
        try:
            raw = engines[p].read_line(turn_timeout)
        except TimeoutError:
            kill(p)
            reason = f"{labels[p]} timeout"
            return None
        except RuntimeError as e:
            kill(p)
            reason = str(e)
            return None
        return raw.split()[0].upper() if raw.split() else ""

    def paced() -> bool:
        return ctrl.wait_pace()

    try:
        emit(turn=0, mover=-1, move="", seed=True, thinking=False)
        if not paced():
            return

        seed_plies = max((len(m) for m in seed_moves), default=0)
        for ply in range(seed_plies):
            for p in range(n):
                if not alive[p]:
                    continue
                emit(turn=0, mover=p, move="", seed=True, thinking=True)
                token_engine = query(p)
                if token_engine is None:
                    emit(turn=0, mover=p, move="", seed=True, thinking=False)
                    if not paced():
                        return
                    continue
                token = seed_moves[p][ply] if ply < len(seed_moves[p]) else ""
                step(p, token, f"seed invalid '{token}'")
                emit(turn=0, mover=p, move=token, seed=True, thinking=False)
                if not paced():
                    return

        turns = 0
        while sum(alive) > 1 and turns < 900:
            for p in range(n):
                if not alive[p] or sum(alive) <= 1:
                    continue
                emit(turn=turns, mover=p, move="", seed=False, thinking=True)
                token = query(p)
                if token is None:
                    emit(turn=turns, mover=p, move="", seed=False, thinking=False)
                    if not paced():
                        return
                    continue
                if token not in DIRS:
                    kill(p)
                    reason = f"{labels[p]} invalid '{token}'"
                    emit(turn=turns, mover=p, move=token, seed=False, thinking=False)
                    if not paced():
                        return
                    continue
                step(p, token, f"invalid '{token}'")
                emit(turn=turns, mover=p, move=token, seed=False, thinking=False)
                if not paced():
                    return
            turns += 1

        live = [i for i, a in enumerate(alive) if a]
        if len(live) == 1:
            winner = live[0]
            reason = "last standing"
        elif len(live) == 0:
            winner = None
            tied = list(range(n))
            reason = reason if reason != "unknown" else "all eliminated"
        else:
            def flood(p: int) -> int:
                return len(bfs_empty(occ, head[p])) if alive[p] else 0

            best = max(flood(p) for p in live)
            tied = [p for p in live if flood(p) == best]
            if len(tied) == 1:
                winner = tied[0]
                tied = []
                reason = "turn cap flood"
            else:
                winner = None
                reason = "turn cap draw"

        cand_seats = [i for i, c in enumerate(is_cand) if c]
        cand = cand_seats[0] if cand_seats else 0
        if winner is not None:
            score = 1.0 if winner == cand else 0.0
        elif cand in tied:
            score = 1.0 / max(len(tied), 1)
        else:
            score = 0.0
        emit(
            turn=turns,
            mover=-1,
            move="",
            seed=False,
            thinking=False,
            over={
                "winner": winner,
                "tied": tied,
                "reason": reason,
                "score_dev": score,
                "turns": turns,
            },
        )
    finally:
        for e in engines:
            e.close()


class Watcher:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.baseline = os.path.abspath(args.baseline)
        self.dev = os.path.abspath(args.dev)
        self.n_players = int(args.players)
        book_path = args.book
        self.book = load_book(book_path) if book_path else ()
        self.rng = random.Random(args.seed)
        self.ctrl = Control()
        self.ctrl.delay_ms = args.delay_ms
        self.out: queue.Queue = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self._restart_pending = False
        self._auto_token = 0
        self.frame: Optional[dict] = None
        self.last_opening: Optional[dict] = None
        self.fixed_opening: Optional[dict] = None
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.games = 0
        self.candidate_first = True
        self.show_voronoi = True
        self._cells: list[list[int]] = []
        self._heads: list[int] = []

        self.root = tk.Tk()
        self.root.title("Tron — candidate vs baseline")
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill=tk.BOTH, expand=True)

        board_wrap = tk.Frame(main, bg=BG, padx=PAD, pady=PAD)
        board_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(
            board_wrap,
            width=BOARD_PX_W,
            height=BOARD_PX_H,
            bg=WALL,
            highlightthickness=0,
        )
        self.canvas.pack()
        self._init_cells()

        side = tk.Frame(main, bg=PANEL, width=340, padx=14, pady=12)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        self.status = tk.Label(
            side, text="Ready", bg=PANEL, fg=ACCENT,
            font=("TkDefaultFont", 14, "bold"), anchor="w", justify="left",
        )
        self.status.pack(fill=tk.X, pady=(0, 4))
        self.substatus = tk.Label(
            side, text="", bg=PANEL, fg=MUTED, font=("TkDefaultFont", 10),
            anchor="w", justify="left", wraplength=310,
        )
        self.substatus.pack(fill=tk.X, pady=(0, 10))

        self.scoreboard = tk.Label(
            side, text="session  0–0–0  (0 games)", bg=PANEL, fg=FG,
            font=("TkFixedFont", 11), anchor="w",
        )
        self.scoreboard.pack(fill=tk.X, pady=(0, 12))

        self.player_frames: list[tk.Frame] = []
        self.player_labels: list[tk.Label] = []
        for _ in range(4):
            f = tk.Frame(side, bg=PANEL)
            lab = tk.Label(
                f, text="", bg=PANEL, fg=FG, font=("TkFixedFont", 10),
                anchor="w", justify="left",
            )
            lab.pack(fill=tk.X)
            self.player_frames.append(f)
            self.player_labels.append(lab)

        tk.Frame(side, bg="#2a3140", height=1).pack(fill=tk.X, pady=10)

        self.budget_var = tk.IntVar(value=args.budget_ms)
        self.delay_var = tk.IntVar(value=args.delay_ms)
        self.seed_plies_var = tk.IntVar(value=args.seed_plies)
        self.voronoi_var = tk.BooleanVar(value=True)
        self.auto_var = tk.BooleanVar(value=True)
        self.first_var = tk.BooleanVar(value=True)
        self.paused_var = tk.BooleanVar(value=False)

        self._spin(side, "budget ms", self.budget_var, 5, 200)
        self._spin(side, "move delay ms", self.delay_var, 0, 2000)
        self._spin(side, "seed plies", self.seed_plies_var, 0, 12)

        tk.Checkbutton(
            side, text="Voronoi overlay", variable=self.voronoi_var,
            command=self._redraw, bg=PANEL, fg=FG, selectcolor=BG,
            activebackground=PANEL, activeforeground=FG, highlightthickness=0,
        ).pack(anchor="w")
        tk.Checkbutton(
            side, text="Auto next game", variable=self.auto_var,
            bg=PANEL, fg=FG, selectcolor=BG,
            activebackground=PANEL, activeforeground=FG, highlightthickness=0,
        ).pack(anchor="w")
        tk.Checkbutton(
            side, text="Candidate moves first (P0)", variable=self.first_var,
            bg=PANEL, fg=FG, selectcolor=BG,
            activebackground=PANEL, activeforeground=FG, highlightthickness=0,
        ).pack(anchor="w")

        btns = tk.Frame(side, bg=PANEL)
        btns.pack(fill=tk.X, pady=(12, 0))
        self._btn(btns, "New game", self.new_game).pack(side=tk.LEFT, padx=(0, 6))
        self._btn(btns, "Replay", self.replay).pack(side=tk.LEFT, padx=(0, 6))
        self.pause_btn = self._btn(btns, "Pause", self.toggle_pause)
        self.pause_btn.pack(side=tk.LEFT)

        btns2 = tk.Frame(side, bg=PANEL)
        btns2.pack(fill=tk.X, pady=(6, 0))
        self._btn(btns2, "Step", self.step).pack(side=tk.LEFT, padx=(0, 6))
        self._btn(btns2, "Swap + replay", self.swap_replay).pack(side=tk.LEFT)

        help_txt = (
            "Space pause  ·  N new  ·  R replay  ·  S step\n"
            "V overlay  ·  same protocol as SPRT"
        )
        tk.Label(
            side, text=help_txt, bg=PANEL, fg=MUTED,
            font=("TkDefaultFont", 9), anchor="w", justify="left",
        ).pack(fill=tk.X, pady=(16, 0))

        self.root.bind("<space>", lambda _e: self.toggle_pause())
        self.root.bind("n", lambda _e: self.new_game())
        self.root.bind("N", lambda _e: self.new_game())
        self.root.bind("r", lambda _e: self.replay())
        self.root.bind("R", lambda _e: self.replay())
        self.root.bind("s", lambda _e: self.step())
        self.root.bind("S", lambda _e: self.step())
        self.root.bind("v", lambda _e: self._toggle_voronoi())
        self.root.bind("V", lambda _e: self._toggle_voronoi())

        self.root.after(40, self._pump)
        self.root.after(200, self.new_game)

    def _spin(self, parent, label: str, var: tk.IntVar, lo: int, hi: int) -> None:
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text=label, bg=PANEL, fg=MUTED, width=14, anchor="w").pack(side=tk.LEFT)
        tk.Spinbox(
            row, from_=lo, to=hi, textvariable=var, width=6,
            bg=BG, fg=FG, buttonbackground=PANEL, highlightthickness=0,
            command=self._sync_ctrl,
        ).pack(side=tk.LEFT)
        var.trace_add("write", lambda *_: self._sync_ctrl())

    def _btn(self, parent, text: str, cmd) -> tk.Button:
        return tk.Button(
            parent, text=text, command=cmd, bg="#1c2433", fg=FG,
            activebackground="#2a3548", activeforeground=FG,
            highlightthickness=0, relief=tk.FLAT, padx=8, pady=4,
        )

    def _init_cells(self) -> None:
        self._cells = [[0] * W for _ in range(H)]
        for y in range(H):
            for x in range(W):
                x0, y0 = x * CELL, y * CELL
                self._cells[y][x] = self.canvas.create_rectangle(
                    x0, y0, x0 + CELL, y0 + CELL,
                    fill=EMPTY, outline=GRID, width=1,
                )
        self._heads = [
            self.canvas.create_rectangle(0, 0, 0, 0, fill="", outline="", width=2)
            for _ in range(4)
        ]

    def _sync_ctrl(self) -> None:
        try:
            self.ctrl.delay_ms = int(self.delay_var.get())
        except (tk.TclError, ValueError):
            pass
        self.ctrl.auto_next = bool(self.auto_var.get())

    def _toggle_voronoi(self) -> None:
        self.voronoi_var.set(not self.voronoi_var.get())
        self._redraw()

    def toggle_pause(self) -> None:
        self.paused_var.set(not self.paused_var.get())
        paused = bool(self.paused_var.get())
        self.ctrl.set_paused(paused)
        self.pause_btn.configure(text="Resume" if paused else "Pause")

    def step(self) -> None:
        self.paused_var.set(True)
        self.pause_btn.configure(text="Resume")
        self.ctrl.request_step()

    def on_close(self) -> None:
        self.ctrl.request_shutdown()
        self.root.after(80, self.root.destroy)

    def new_game(self) -> None:
        self._auto_token += 1
        self.fixed_opening = None
        self.candidate_first = bool(self.first_var.get())
        self._start_game()

    def replay(self) -> None:
        if self.last_opening is None:
            return
        self._auto_token += 1
        self.fixed_opening = dict(self.last_opening)
        self._start_game()

    def swap_replay(self) -> None:
        if self.last_opening is None:
            return
        self._auto_token += 1
        self.candidate_first = not self.last_opening["candidate_first"]
        self.first_var.set(self.candidate_first)
        self.fixed_opening = dict(self.last_opening)
        self.fixed_opening["candidate_first"] = self.candidate_first
        self._start_game()

    def _start_game(self) -> None:
        self._sync_ctrl()
        self.ctrl.request_abort()
        old = self.worker
        if old is not None and old.is_alive():
            if not self._restart_pending:
                self._restart_pending = True
                self.root.after(30, self._start_game)
            return
        self._restart_pending = False
        self.ctrl = Control()
        self.ctrl.delay_ms = int(self.delay_var.get() or 0)
        self.ctrl.auto_next = bool(self.auto_var.get())
        self.ctrl.paused = bool(self.paused_var.get())
        self.out = queue.Queue()

        n = self.n_players
        if self.fixed_opening is not None:
            starts = list(self.fixed_opening["starts"])
            seed_moves = [list(m) for m in self.fixed_opening["seed_moves"]]
            candidate_first = bool(self.fixed_opening["candidate_first"])
        else:
            starts = pick_starts(self.rng, self.book, n)
            seed_moves = random_seed_moves(self.rng, starts, int(self.seed_plies_var.get()))
            candidate_first = bool(self.first_var.get())

        self.candidate_first = candidate_first
        cand_seat = 0 if candidate_first else 1
        paths = [self.dev if i == cand_seat else self.baseline for i in range(n)]
        labels: list[str] = []
        base_i = 0
        for i in range(n):
            if i == cand_seat:
                labels.append("candidate")
            elif n == 2:
                labels.append("baseline")
            else:
                labels.append(f"baseline{base_i}")
                base_i += 1
        is_cand = [i == cand_seat for i in range(n)]

        self.last_opening = {
            "starts": list(starts),
            "seed_moves": [list(m) for m in seed_moves],
            "candidate_first": candidate_first,
        }
        self.fixed_opening = None

        budget = int(self.budget_var.get())
        timeout = budget / 1000.0 + 0.25
        q = self.out
        ctrl = self.ctrl

        def run() -> None:
            try:
                play_watched_game(
                    paths, starts, labels, is_cand, budget, timeout, seed_moves, ctrl, q,
                )
            except Exception as e:
                q.put({"error": str(e)})
            if ctrl.auto_next and not ctrl.shutdown and not ctrl.abort:
                q.put({"auto_next": True})

        self.worker = threading.Thread(target=run, daemon=True)
        self.status.configure(text="Starting…", fg=ACCENT)
        self.worker.start()

    def _pump(self) -> None:
        try:
            while True:
                msg = self.out.get_nowait()
                if isinstance(msg, dict) and msg.get("error"):
                    self.status.configure(text="Engine error", fg=CUT)
                    self.substatus.configure(text=msg["error"])
                    continue
                if isinstance(msg, dict) and msg.get("auto_next"):
                    if self.auto_var.get() and not self.ctrl.shutdown:
                        token = self._auto_token
                        self.root.after(600, lambda t=token: self._maybe_auto_next(t))
                    continue
                self.frame = msg
                if msg.get("over"):
                    self._on_over(msg["over"])
                self._redraw()
        except queue.Empty:
            pass
        self.root.after(40, self._pump)

    def _maybe_auto_next(self, token: int) -> None:
        if token != self._auto_token:
            return
        if self.auto_var.get() and not self.ctrl.shutdown:
            self.new_game()

    def _on_over(self, over: dict) -> None:
        self.games += 1
        sc = float(over.get("score_dev") or 0.0)
        if sc >= 0.99:
            self.wins += 1
        elif sc <= 0.01:
            self.losses += 1
        else:
            self.draws += 1
        self.scoreboard.configure(
            text=f"session  {self.wins}–{self.draws}–{self.losses}  ({self.games} games)"
        )

    def _player_color(self, p: int, frame: dict) -> tuple[str, str, str]:
        is_cand = frame["is_cand"][p]
        b = 0
        if not is_cand:
            b = sum(1 for i in range(p) if not frame["is_cand"][i])
        return role_colors(is_cand, b)

    def _redraw(self) -> None:
        frame = self.frame
        if not frame or "heads" not in frame:
            return
        show_v = bool(self.voronoi_var.get())
        owner = frame["owner"]
        occ = frame["occ"]
        trails = frame["trails"]
        trail_of = [[-1] * W for _ in range(H)]
        for p, trail in enumerate(trails):
            for x, y in trail:
                trail_of[y][x] = p

        for y in range(H):
            for x in range(W):
                who = trail_of[y][x]
                if who >= 0:
                    fill = self._player_color(who, frame)[0]
                elif occ[y][x]:
                    fill = "#3a4150"
                elif show_v:
                    o = owner[y][x]
                    if o >= 0:
                        fill = self._player_color(o, frame)[2]
                    elif o == -1:
                        fill = CONTESTED
                    else:
                        fill = EMPTY
                else:
                    fill = EMPTY
                self.canvas.itemconfigure(self._cells[y][x], fill=fill)

        for i, item in enumerate(self._heads):
            if i >= len(frame["heads"]) or not frame["alive"][i] or not trails[i]:
                self.canvas.coords(item, 0, 0, 0, 0)
                self.canvas.itemconfigure(item, outline="")
                continue
            hx, hy = frame["heads"][i]
            x0, y0 = hx * CELL + 3, hy * CELL + 3
            trail_c, head_c, _ = self._player_color(i, frame)
            self.canvas.coords(item, x0, y0, x0 + CELL - 6, y0 + CELL - 6)
            thinking = frame["thinking"] and frame["mover"] == i
            self.canvas.itemconfigure(
                item,
                fill=head_c,
                outline="#ffffff" if thinking else trail_c,
                width=3 if thinking else 2,
            )
            self.canvas.tag_raise(item)

        self._update_sidebar(frame)

    def _update_sidebar(self, frame: dict) -> None:
        over = frame.get("over")
        connected = bool(frame.get("connected"))
        if over:
            winner = over.get("winner")
            if winner is None:
                self.status.configure(text="Draw", fg=MUTED)
            elif frame["is_cand"][winner]:
                self.status.configure(text="Candidate wins", fg=OPEN)
            else:
                self.status.configure(text="Baseline wins", fg=CUT)
            self.substatus.configure(
                text=f"{over.get('reason', '')}  ·  {over.get('turns', 0)} turns"
            )
        elif frame.get("thinking"):
            lab = frame["labels"][frame["mover"]] if frame["mover"] >= 0 else ""
            self.status.configure(text=f"{lab} thinking…", fg=ACCENT)
            self.substatus.configure(text=self._phase_line(frame, connected))
        else:
            phase = "OPEN" if connected else "CUT"
            self.status.configure(text=phase, fg=OPEN if connected else CUT)
            extra = ""
            if frame.get("seed"):
                extra = "opening (referee walk)  ·  "
            mv = ""
            if frame["mover"] >= 0 and frame.get("move"):
                mv = f"{frame['labels'][frame['mover']]} {frame['move']}  ·  "
            self.substatus.configure(text=extra + mv + self._phase_line(frame, connected))

        n = len(frame["heads"])
        for i in range(4):
            if i < n:
                self.player_frames[i].pack(fill=tk.X, pady=(0, 10))
                self.player_labels[i].configure(
                    text=self._player_text(frame, i),
                    fg=self._player_color(i, frame)[0],
                )
            else:
                self.player_frames[i].pack_forget()

    def _phase_line(self, frame: dict, connected: bool) -> str:
        turn = frame.get("turn", 0)
        if connected:
            return f"shared space  ·  Voronoi eval  ·  turn {turn}"
        return f"separate chambers  ·  fill eval (×60)  ·  turn {turn}"

    def _player_text(self, frame: dict, p: int) -> str:
        hx, hy = frame["heads"][p]
        mm = frame["mm"][p]
        score = mm.get("score")
        ms = mm.get("ms")
        nodes = mm.get("nodes")
        nps = fmt_nps(nodes or 0, ms or 0) if nodes and ms else "n/a"
        seat = f"P{p}"
        role = frame["labels"][p]
        alive = "alive" if frame["alive"][p] else "dead"
        score_s = f"{score:+d}" if isinstance(score, int) else "—"
        if isinstance(score, int) and abs(score) >= 800_000:
            score_s += "  (mate)"
        terr = frame["territory"][p]
        reach = frame["reachable"][p]
        mob = frame["mobility"][p]
        last = frame["move"] if frame["mover"] == p and frame.get("move") else (mm.get("dir") or "")
        stderr = mm.get("raw") or "(no stderr stats)"
        return (
            f"{role}  {seat}  {alive}\n"
            f"  head ({hx},{hy})  last {last or '—'}\n"
            f"  eval {score_s}   {ms or '—'}ms  n={nodes or '—'}  {nps}\n"
            f"  territory {terr}   reach {reach}   mob {mob}\n"
            f"  {stderr}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch candidate vs baseline Tron games")
    p.add_argument("--baseline", default=str(ROOT / "bin/tron-baseline"))
    p.add_argument("--dev", default=str(ROOT / "target/release/tron"))
    p.add_argument("--players", type=int, default=2, choices=(2, 3, 4))
    p.add_argument("--budget-ms", type=int, default=80)
    p.add_argument("--delay-ms", type=int, default=180)
    p.add_argument("--seed-plies", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--book",
        default="",
        help="Opening book JSON, 'none', or empty for the default N-player book",
    )
    args = p.parse_args()
    if not args.seed:
        args.seed = random.randrange(2**31)
    if args.book == "none":
        args.book = ""
    elif not args.book:
        args.book = str(DEFAULT_BOOK_BY_N[args.players])
    return args


def main() -> int:
    args = parse_args()
    for path, name in ((args.baseline, "baseline"), (args.dev, "dev")):
        if not os.path.isfile(path) or not os.access(path, os.X_OK):
            print(f"error: {name} binary not executable: {path}", file=sys.stderr)
            print("Build with cargo build --release and/or tools/save_baseline.sh", file=sys.stderr)
            return 2
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print("error: no DISPLAY; this GUI needs a local desktop session", file=sys.stderr)
        return 2
    Watcher(args)
    tk.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
