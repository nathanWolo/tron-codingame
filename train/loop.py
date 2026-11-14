"""The AlphaZero loop: self-play with the Rust engine, training in JAX, repeat.

    python train/loop.py --run runs/r1            # start, or resume if runs/r1 exists

Each generation:
  1. `tronzero selfplay` plays games with the newest network and writes
     `data/<gen>.bin` (visit counts and outcomes per position).
  2. The new positions go into a replay ring; the network takes gradient steps
     on mirrored samples from the newest part of the ring (about `--reuse`
     uses of each position overall).
  3. The network is exported as `nets/<gen + 1>.bin` for the next generation.

`touch <run>/PAUSE` holds the loop between generations (for benchmarks that
need the CPU); `touch <run>/STOP` ends it after the current generation.
Settings come from `config.json` on resume; flags given on resume override it.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tz  # noqa: E402

DEFAULTS = dict(
    channels="16,24,24",
    hidden=96,
    seed=0,
    sims=64,
    samples_per_gen=200_000,
    threads=14,
    temp_rounds=10,  # rounds (one move per player) whose moves are sampled from the visits
    mix="1,1,3",  # relative numbers of two-, three- and four-player games
    noise_eps=0.25,
    noise_alpha=0.5,
    c_puct=1.0,
    batch=512,
    reuse=4.0,
    lr=1e-3,
    weight_decay=1e-4,
    capacity=1_000_000,
    window_frac=0.4,
    min_window=200_000,
    q_mix=0.0,
    eval_every=5,
    eval_pairs=200,  # openings per player count against an older network
    eval_players="2,4",
    base_every=10,  # generations between matches against the alpha-beta engine (0: never)
    base_pairs=60,  # openings per player count; every opening is played from each seat
    base_players="2,4",
    base_budget_ms=20,
    base_threads=8,
    keep_data=8,  # self-play files kept on disk (the ring is refilled from them on resume)
    gens=100_000,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--exe", default="bin/tronzero.exe")
    for k, v in DEFAULTS.items():
        p.add_argument("--" + k.replace("_", "-"), type=type(v), default=None)
    return p.parse_args()


class Ring:
    """The newest `capacity` positions, with their distance maps."""

    FIELDS = {
        "trail": (np.uint32, (4, 20)),
        "heads": (np.uint8, (4, 2)),
        "visits": (np.uint16, (4,)),
        "legal": (np.uint8, ()),
        "flags": (np.uint8, ()),
        "n_alive": (np.uint8, ()),
        "rank": (np.uint8, (4,)),
        "q": (np.float32, ()),
        "dist": (np.uint8, (4, 600)),
    }

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = {k: np.zeros((capacity, *s), d) for k, (d, s) in self.FIELDS.items()}
        self.pos = 0
        self.filled = 0

    def add(self, arrays: dict) -> None:
        n = len(arrays["q"])
        if n > self.capacity:
            arrays = {k: v[-self.capacity :] for k, v in arrays.items()}
            n = self.capacity
        first = min(n, self.capacity - self.pos)
        for k in self.FIELDS:
            self.data[k][self.pos : self.pos + first] = arrays[k][:first]
            self.data[k][: n - first] = arrays[k][first:]
        self.pos = (self.pos + n) % self.capacity
        self.filled = min(self.capacity, self.filled + n)

    def sample(self, rng: np.random.Generator, batch: int, window: int) -> dict:
        back = rng.integers(0, min(window, self.filled), batch)
        idx = (self.pos - 1 - back) % self.capacity
        return {k: v[idx] for k, v in self.data.items()}


def make_step(weight_decay: float, q_mix: float):
    adam = optax.scale_by_adam()

    def loss(params, batch, fx, fy):
        return tz.loss_fn(params, batch, fx, fy, q_mix=q_mix)

    @jax.jit
    def step(params, opt_state, batch, key, lr):
        b = batch["q"].shape[0]
        fx, fy = jax.random.bernoulli(key, 0.5, (2, b))
        (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(params, batch, fx, fy)
        updates, opt_state = adam.update(grads, opt_state, params)
        params = jax.tree.map(
            lambda p, u: p - lr * (u + weight_decay * p), params, updates
        )
        return params, opt_state, metrics

    return adam, step


def run_engine(exe: str, args: list[str], cwd: str) -> dict:
    r = subprocess.run([exe, *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"engine failed: {args}\n{r.stdout}\n{r.stderr}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    args = parse_args()
    run = args.run
    exe = os.path.abspath(args.exe)
    os.makedirs(os.path.join(run, "nets"), exist_ok=True)
    os.makedirs(os.path.join(run, "data"), exist_ok=True)
    cfg_path = os.path.join(run, "config.json")
    cfg = dict(DEFAULTS)
    if os.path.exists(cfg_path):
        cfg.update(json.load(open(cfg_path)))
    for k in DEFAULTS:
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    print("config", cfg, flush=True)

    adam, step = make_step(cfg["weight_decay"], cfg["q_mix"])
    ckpt_path = os.path.join(run, "ckpt.pkl")
    if os.path.exists(ckpt_path):
        ck = pickle.load(open(ckpt_path, "rb"))
        params = jax.tree.map(jnp.asarray, ck["params"])
        opt_state = jax.tree.map(jnp.asarray, ck["opt_state"])
        gen, total_samples, total_steps = ck["gen"], ck["total_samples"], ck["total_steps"]
        games = ck["games"]
    else:
        channels = tuple(int(c) for c in cfg["channels"].split(","))
        params = tz.init_params(jax.random.key(cfg["seed"]), channels, cfg["hidden"])
        opt_state = adam.init(params)
        gen, total_samples, total_steps = 0, 0, 0
        games = max(200, cfg["samples_per_gen"] // 250)
        tz.export_net(params, os.path.join(run, "nets", "0000.bin"))
    print(f"parameters {tz.count_params(params)}, starting at generation {gen}", flush=True)

    ring = Ring(cfg["capacity"])
    # On resume, refill the ring from the newest data files.
    need, files = cfg["capacity"], []
    for g in range(gen - 1, -1, -1):
        path = os.path.join(run, "data", f"{g:04d}.bin")
        if need <= 0 or not os.path.exists(path):
            break
        files.append(path)
        need -= os.path.getsize(path) // tz.SAMPLE_DTYPE.itemsize
    for path in reversed(files):
        s = tz.load_samples(path)
        ring.add(tz.add_dists(s))
    if files:
        print(f"refilled the ring with {ring.filled} positions from {len(files)} files", flush=True)

    rng = np.random.default_rng(cfg["seed"] + 1000 * gen)
    key = jax.random.key(cfg["seed"] + 1 + gen)
    metrics_path = os.path.join(run, "metrics.jsonl")
    while gen < cfg["gens"]:
        while os.path.exists(os.path.join(run, "PAUSE")):
            time.sleep(2)
        if os.path.exists(os.path.join(run, "STOP")):
            print("STOP file found, exiting", flush=True)
            break
        t0 = time.monotonic()
        sp = run_engine(
            exe,
            [
                "selfplay", "--net", f"nets/{gen:04d}.bin", "--out", f"data/{gen:04d}.bin",
                "--games", str(games), "--sims", str(cfg["sims"]), "--threads", str(cfg["threads"]),
                "--seed", str(cfg["seed"] * 100_003 + gen + 1),
                "--temp-rounds", str(cfg["temp_rounds"]), "--noise-eps", str(cfg["noise_eps"]),
                "--noise-alpha", str(cfg["noise_alpha"]), "--c-puct", str(cfg["c_puct"]),
                "--mix", cfg["mix"],
            ],
            run,
        )
        t1 = time.monotonic()
        samples = tz.load_samples(os.path.join(run, "data", f"{gen:04d}.bin"))
        ring.add(tz.add_dists(samples))
        n_new = len(samples)
        del samples
        old_data = os.path.join(run, "data", f"{gen - cfg['keep_data']:04d}.bin")
        if cfg["keep_data"] and os.path.exists(old_data):
            os.remove(old_data)
        total_samples += n_new
        window = int(min(ring.filled, max(cfg["min_window"], cfg["window_frac"] * total_samples)))
        steps = max(1, int(n_new * cfg["reuse"] / cfg["batch"]))
        t2 = time.monotonic()
        acc = None
        for _ in range(steps):
            batch = ring.sample(rng, cfg["batch"], window)
            key, k = jax.random.split(key)
            params, opt_state, m = step(params, opt_state, batch, k, cfg["lr"])
            acc = m if acc is None else jax.tree.map(jnp.add, acc, m)
        m = {k: float(v) / steps for k, v in acc.items()}
        total_steps += steps
        t3 = time.monotonic()
        gen += 1
        tz.export_net(params, os.path.join(run, "nets", f"{gen:04d}.bin"))
        games = int(np.clip(cfg["samples_per_gen"] / max(1.0, n_new / sp["games"]), 200, 20000))
        row = {
            "gen": gen, "total_samples": total_samples, "total_steps": total_steps,
            "window": window, "steps": steps, **m,
            "selfplay": sp, "selfplay_s": round(t1 - t0, 1), "dist_s": round(t2 - t1, 1),
            "train_s": round(t3 - t2, 1), "lr": cfg["lr"], "sims": cfg["sims"],
        }
        if cfg["eval_every"] and gen % cfg["eval_every"] == 0:
            old = max(0, gen - 2 * cfg["eval_every"])
            row["eval"] = {"vs_gen": old}
            for n in cfg["eval_players"].split(","):
                row["eval"][n] = run_engine(
                    exe,
                    [
                        "match", "--a", f"nets/{gen:04d}.bin", "--b", f"nets/{old:04d}.bin",
                        "--players", n, "--pairs", str(cfg["eval_pairs"]), "--sims", str(cfg["sims"]),
                        "--threads", str(cfg["threads"]),
                    ],
                    run,
                )
        if cfg["base_every"] and gen % cfg["base_every"] == 0:
            row["vs_base"] = {}
            for n in cfg["base_players"].split(","):
                row["vs_base"][n] = run_engine(
                    exe,
                    [
                        "vsbase", "--net", f"nets/{gen:04d}.bin", "--players", n,
                        "--pairs", str(cfg["base_pairs"]), "--budget-ms", str(cfg["base_budget_ms"]),
                        "--threads", str(cfg["base_threads"]),
                    ],
                    run,
                )
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        with open(ckpt_path + ".tmp", "wb") as f:
            pickle.dump(
                {
                    "params": jax.tree.map(np.asarray, params),
                    "opt_state": jax.tree.map(np.asarray, opt_state),
                    "gen": gen, "total_samples": total_samples, "total_steps": total_steps,
                    "games": games,
                },
                f,
            )
        os.replace(ckpt_path + ".tmp", ckpt_path)
        # A score is the mean placement score over openings (0.5 is parity; in a duel it
        # is the win rate). Most openings are decided by the seats, so the openings
        # won, drawn and lost overall are shown too.
        def res(r):
            return f"{r['score']:.3f} ({r['pairs_won']}-{r['pairs_split']}-{r['pairs_lost']})"

        ev = ""
        if "eval" in row:
            ev += f" | vs gen {row['eval']['vs_gen']}:" + "".join(
                f" {n}p {res(r)}" for n, r in row["eval"].items() if n != "vs_gen"
            )
        if "vs_base" in row:
            ev += f" | vs alpha-beta @{cfg['base_budget_ms']}ms:" + "".join(
                f" {n}p {res(r)} win {r['win_rate']:.2f}" for n, r in row["vs_base"].items()
            )
        print(
            f"gen {gen:4d} | {total_samples / 1e6:6.2f}M pos | {sp['games']} games "
            f"({sp['games_2p']}/{sp['games_3p']}/{sp['games_4p']}), {sp['mean_plies']:.0f} plies, "
            f"searched {sp['searched'] / max(1, sp['samples']):.2f} | "
            f"pol {m['policy_loss']:.3f} (H {m['target_entropy']:.3f}) val {m['value_loss']:.3f} "
            f"win-acc {m['winner_acc']:.3f} | sp {t1 - t0:.0f}s dist {t2 - t1:.0f}s train {t3 - t2:.0f}s{ev}",
            flush=True,
        )


if __name__ == "__main__":
    main()
