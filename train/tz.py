"""tronzero's network, features and losses in JAX (two to four players).

The Rust engine (`engine/src/net.rs`) computes the same features and the same
forward pass; `check_parity` compares the two on positions the engine dumps.

Everything per player is in slot order: slot 0 is the player to move, the
following slots are the other players still in the game, in turn order.
"""

from __future__ import annotations

import struct

import jax
import jax.numpy as jnp
import numpy as np

W, H = 30, 20
CW, CH = 59, 39
NPLANES, NSCAL, NVAL = 16, 13, 4
UNREACH = 255

# One training position as `tronzero selfplay` writes it (352 bytes, little endian).
SAMPLE_DTYPE = np.dtype(
    [
        ("trail", "<u4", (4, 20)),  # per slot, row bitboards: bit x of row y is trail or head
        ("heads", "u1", (4, 2)),  # per slot: x, y (255 for an unused slot)
        ("visits", "<u2", (4,)),  # root visits of UP, DOWN, LEFT, RIGHT
        ("n_alive", "u1"),  # players still in the game (the used slots)
        ("n_players", "u1"),  # players the game started with
        ("legal", "u1"),  # bit d: the mover can step in direction d
        ("flags", "u1"),  # bit 0: one legal move; bit 1: space-filling step (not searched)
        ("rank", "u1", (4,)),  # per slot: final placement, 1 = winner (0 for an unused slot)
        ("q", "<f4"),  # the search's root value for the mover
        ("ply", "<u2"),
        ("sims", "<u2"),
    ]
)
assert SAMPLE_DTYPE.itemsize == 352


# --- Parameters ---------------------------------------------------------------------


def init_params(key, channels=(16, 24, 24), hidden=96):
    """He-initialised weights: three 3x3 stride-2 convs, a hidden layer, two heads."""
    ks = jax.random.split(key, 8)
    cin = (NPLANES, *channels[:-1])
    params = {}
    for i, (ci, co) in enumerate(zip(cin, channels)):
        params[f"conv{i}"] = {
            "w": jax.random.normal(ks[i], (3, 3, ci, co)) * np.sqrt(2.0 / (9 * ci)),
            "b": jnp.zeros((co,)),
        }
    flat = 8 * 5 * channels[-1]
    params["hidden"] = {
        "w": jax.random.normal(ks[3], (flat, hidden)) * np.sqrt(2.0 / flat),
        "b": jnp.zeros((hidden,)),
    }
    params["feat"] = {"w": jax.random.normal(ks[4], (NSCAL, hidden)) * 0.1}
    params["policy"] = {
        "w": jax.random.normal(ks[5], (hidden, 4)) * 0.01,
        "b": jnp.zeros((4,)),
    }
    params["value"] = {
        "w": jax.random.normal(ks[6], (hidden, NVAL)) * 0.01,
        "b": jnp.zeros((NVAL,)),
    }
    return params


def count_params(params) -> int:
    return sum(int(np.prod(x.shape)) for x in jax.tree.leaves(params))


def export_net(params, path) -> None:
    """Write the weights in the layout `Net::load` reads."""
    p = jax.tree.map(lambda x: np.asarray(x, np.float32), params)
    c = [p[f"conv{i}"]["w"].shape[3] for i in range(3)]
    hidden = p["hidden"]["w"].shape[1]
    with open(path, "wb") as f:
        f.write(b"TZN2")
        f.write(struct.pack("<7I", NPLANES, NSCAL, c[0], c[1], c[2], hidden, NVAL))
        for i in range(3):
            f.write(p[f"conv{i}"]["w"].tobytes())  # [ky, kx, cin, cout]
            f.write(p[f"conv{i}"]["b"].tobytes())
        f.write(p["hidden"]["w"].tobytes())  # [(y * 8 + x) * c3 + ch, hidden]
        f.write(p["hidden"]["b"].tobytes())
        f.write(p["feat"]["w"].tobytes())  # [scalar, hidden]
        f.write(np.ascontiguousarray(p["policy"]["w"].T).tobytes())  # [move, hidden]
        f.write(p["policy"]["b"].tobytes())
        f.write(np.ascontiguousarray(p["value"]["w"].T).tobytes())  # [slot, hidden]
        f.write(p["value"]["b"].tobytes())


# --- Features -----------------------------------------------------------------------


def unpack_rows(bits):
    """Row bitboards ``u32[..., 20]`` to ``bool[..., 20, 30]``."""
    cols = jnp.arange(W, dtype=jnp.uint32)
    return ((bits[..., None] >> cols) & 1).astype(bool)


def _neighbours(m):
    p = jnp.pad(m, ((0, 0), (1, 1), (1, 1)))
    return p[:, :-2, 1:-1] | p[:, 2:, 1:-1] | p[:, 1:-1, :-2] | p[:, 1:-1, 2:]


@jax.jit
def bfs_dist(empty, hx, hy):
    """Steps from a head to every cell through empty cells: ``u8[B, 20, 30]``.

    A head's empty neighbours are at 1; unreachable cells are 255. Distances
    are capped at 254, as in the engine's ``bfs_dist``.
    """
    b = empty.shape[0]
    head = jnp.zeros(empty.shape, bool).at[jnp.arange(b), hy, hx].set(True)
    frontier = _neighbours(head) & empty
    dist = jnp.where(frontier, 1, UNREACH).astype(jnp.uint8)

    def cond(s):
        return s[0].any()

    def body(s):
        frontier, visited, dist, d = s
        nxt = _neighbours(frontier) & empty & ~visited
        d = jnp.minimum(d + 1, 254)
        return nxt, visited | nxt, jnp.where(nxt, d.astype(jnp.uint8), dist), d

    _, _, dist, _ = jax.lax.while_loop(cond, body, (frontier, frontier, dist, jnp.int32(1)))
    return dist


def board_planes(trail, heads, dist, n_alive):
    """The board planes ``f32[B, 20, 30, 16]`` and the scalars ``f32[B, 13]``.

    Planes: 0 empty; 1-3 the heads of slots 1-3; 4-7 the cells each slot
    reaches first (ties to the earlier slot); 8-11 each slot's closeness
    ``(64 - min(d, 64)) / 64``; 12-15 each slot's trail.
    """
    b = trail.shape[0]
    slots = jnp.arange(4)
    used = slots[None, :] < n_alive[:, None].astype(jnp.int32)  # [B, 4]
    tr = unpack_rows(trail) & used[:, :, None, None]  # [B, 4, H, W]
    empty = ~tr.any(1)
    d = dist.reshape(b, 4, H, W).astype(jnp.int32)
    reach = empty[:, None] & (d != UNREACH)
    owned = reach & (jnp.argmin(d, axis=1)[:, None] == slots[None, :, None, None])
    close = jnp.where(reach, (64 - jnp.minimum(d, 64)) / 64.0, 0.0)
    hx = jnp.clip(heads[:, 1:, 0].astype(jnp.int32), 0, W - 1)
    hy = jnp.clip(heads[:, 1:, 1].astype(jnp.int32), 0, H - 1)
    head = (
        jnp.zeros((b, 3, H, W), bool)
        .at[jnp.arange(b)[:, None], jnp.arange(3)[None, :], hy, hx]
        .set(used[:, 1:])
    )
    f32 = jnp.float32
    planes = jnp.concatenate(
        [empty[:, None].astype(f32), head.astype(f32), owned.astype(f32), close.astype(f32), tr.astype(f32)],
        axis=1,
    ).transpose(0, 2, 3, 1)
    terr = owned.sum((2, 3)).astype(f32)  # [B, 4]
    scal = jnp.concatenate(
        [
            terr / 128.0,
            reach.sum((2, 3)) / 256.0,
            empty.sum((1, 2))[:, None] / 600.0,
            (n_alive[:, None].astype(jnp.int32) == jnp.array([2, 3, 4])[None, :]).astype(f32),
            ((terr[:, 0] - terr[:, 1:].max(1)) / 32.0)[:, None],
        ],
        axis=1,
    ).astype(f32)
    return planes, scal


def canvas(planes, mx, my):
    """Board planes on the 39x59 canvas centred on the mover's head; zero off the board."""
    b = planes.shape[0]
    by = jnp.arange(CH)[None, :] - 19 + my[:, None]
    bx = jnp.arange(CW)[None, :] - 29 + mx[:, None]
    valid = ((by >= 0) & (by < H))[:, :, None] & ((bx >= 0) & (bx < W))[:, None, :]
    g = planes[
        jnp.arange(b)[:, None, None],
        jnp.clip(by, 0, H - 1)[:, :, None],
        jnp.clip(bx, 0, W - 1)[:, None, :],
    ]
    return jnp.where(valid[..., None], g, 0.0)


def flip(planes, mx, my, flip_x, flip_y):
    """Mirror the board left-right and/or top-bottom, per sample."""
    planes = jnp.where(flip_x[:, None, None, None], planes[:, :, ::-1], planes)
    mx = jnp.where(flip_x, W - 1 - mx, mx)
    planes = jnp.where(flip_y[:, None, None, None], planes[:, ::-1], planes)
    my = jnp.where(flip_y, H - 1 - my, my)
    return planes, mx, my


def flip_moves(x, flip_x, flip_y):
    """Apply the same mirrors to per-move values ``[B, 4]`` (UP, DOWN, LEFT, RIGHT)."""
    x = jnp.where(flip_x[:, None], x[:, jnp.array([0, 1, 3, 2])], x)
    return jnp.where(flip_y[:, None], x[:, jnp.array([1, 0, 2, 3])], x)


# --- Network --------------------------------------------------------------------------


def conv_patches(x, w, b):
    """3x3 stride-2 pad-1 convolution and ReLU, as one matmul over extracted patches."""
    _, h, wd, _ = x.shape
    oh, ow = (h - 1) // 2 + 1, (wd - 1) // 2 + 1
    xp = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    taps = [
        xp[:, ky : ky + 2 * oh - 1 : 2, kx : kx + 2 * ow - 1 : 2, :]
        for ky in range(3)
        for kx in range(3)
    ]
    patches = jnp.concatenate(taps, axis=-1)  # [B, oh, ow, 9 * cin], tap-major
    return jax.nn.relu(patches @ w.reshape(-1, w.shape[-1]) + b)


def conv_lax(x, w, b):
    y = jax.lax.conv_general_dilated(
        x, w, (2, 2), ((1, 1), (1, 1)), dimension_numbers=("NHWC", "HWIO", "NHWC")
    )
    return jax.nn.relu(y + b)


def forward(params, cv, scal, conv=conv_patches):
    """Policy logits ``[B, 4]`` and per-slot values ``[B, 4]`` in [-1, 1]."""
    x = cv
    for i in range(3):
        x = conv(x, params[f"conv{i}"]["w"], params[f"conv{i}"]["b"])
    x = x.reshape(x.shape[0], -1)
    h = jax.nn.relu(x @ params["hidden"]["w"] + params["hidden"]["b"] + scal @ params["feat"]["w"])
    logits = h @ params["policy"]["w"] + params["policy"]["b"]
    value = jnp.tanh(h @ params["value"]["w"] + params["value"]["b"])
    return logits, value


def inputs(batch, flip_x=None, flip_y=None):
    """Canvases and scalars of a batch of samples (dict of arrays), optionally mirrored."""
    planes, scal = board_planes(batch["trail"], batch["heads"], batch["dist"], batch["n_alive"])
    mx = batch["heads"][:, 0, 0].astype(jnp.int32)
    my = batch["heads"][:, 0, 1].astype(jnp.int32)
    if flip_x is not None:
        planes, mx, my = flip(planes, mx, my, flip_x, flip_y)
    return canvas(planes, mx, my), scal


def legal_bits(legal):
    return ((legal[:, None] >> jnp.arange(4, dtype=jnp.uint8)) & 1).astype(bool)


def value_targets(rank, n_alive):
    """Per-slot targets in [-1, 1] and the mask of used slots.

    A slot's target is its final placement among the players still in the
    game at this position: +1 for the winner, -1 for the next one out,
    evenly spaced between.
    """
    a = n_alive[:, None].astype(jnp.float32)
    used = jnp.arange(4)[None, :] < n_alive[:, None].astype(jnp.int32)
    u = 1.0 - 2.0 * (rank.astype(jnp.float32) - 1.0) / jnp.maximum(a - 1.0, 1.0)
    return jnp.where(used, u, 0.0), used


def loss_fn(params, batch, flip_x, flip_y, conv=conv_patches, q_mix=0.0):
    cv, scal = inputs(batch, flip_x, flip_y)
    logits, value = forward(params, cv, scal, conv)
    # Policy: cross-entropy to the visit shares, on searched positions only.
    searched = (batch["flags"] == 0).astype(jnp.float32)
    n_searched = jnp.maximum(searched.sum(), 1.0)
    legal = flip_moves(legal_bits(batch["legal"]), flip_x, flip_y)
    visits = flip_moves(batch["visits"].astype(jnp.float32), flip_x, flip_y)
    target = visits / jnp.maximum(visits.sum(-1, keepdims=True), 1.0)
    logp = jax.nn.log_softmax(jnp.where(legal, logits, -1e9), axis=-1)
    ce = -jnp.sum(target * jnp.where(legal, logp, 0.0), axis=-1)
    policy_loss = (ce * searched).sum() / n_searched
    entropy = -jnp.sum(jnp.where(target > 0, target * jnp.log(jnp.maximum(target, 1e-9)), 0.0), -1)
    # Value: every used slot's placement, each position weighted equally.
    u, used = value_targets(batch["rank"], batch["n_alive"])
    u = u.at[:, 0].set((1.0 - q_mix) * u[:, 0] + q_mix * batch["q"])
    n_used = batch["n_alive"].astype(jnp.float32)
    value_loss = jnp.mean(jnp.sum(jnp.where(used, (value - u) ** 2, 0.0), -1) / n_used)
    winner = jnp.argmax(jnp.where(used, value, -2.0), -1)
    metrics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "target_entropy": (entropy * searched).sum() / n_searched,
        # How often the slot the network values highest is the eventual winner.
        "winner_acc": jnp.mean(jnp.take_along_axis(batch["rank"], winner[:, None], 1)[:, 0] == 1),
    }
    return policy_loss + value_loss, metrics


# --- Data -----------------------------------------------------------------------------

TRAIN_FIELDS = ("trail", "heads", "visits", "legal", "flags", "n_alive", "rank", "q")


def load_samples(path) -> np.ndarray:
    return np.fromfile(path, dtype=SAMPLE_DTYPE)


def add_dists(samples: np.ndarray, chunk: int = 32768) -> dict:
    """The arrays training needs, with every slot's distance map computed here."""
    n = len(samples)
    dist = np.empty((n, 4, H * W), np.uint8)
    for s in range(0, n, chunk):
        part = samples[s : s + chunk]
        m = len(part)
        if m < chunk:  # pad so the jitted BFS sees one shape
            part = np.concatenate([part, np.repeat(part[-1:], chunk - m)])
        used = np.arange(4)[None, :] < part["n_alive"][:, None]
        trail = unpack_rows(jnp.asarray(part["trail"])) & jnp.asarray(used)[:, :, None, None]
        empty = ~trail.any(1)
        for k in range(4):
            hx = np.minimum(part["heads"][:, k, 0], W - 1).astype(np.int32)
            hy = np.minimum(part["heads"][:, k, 1], H - 1).astype(np.int32)
            d = np.asarray(bfs_dist(empty, jnp.asarray(hx), jnp.asarray(hy))).reshape(chunk, -1)
            dist[s : s + m, k] = np.where(used[:m, k, None], d[:m], UNREACH)
    out = {k: samples[k].copy() for k in TRAIN_FIELDS}
    out["dist"] = dist
    return out


def check_parity(params, prefix, conv=conv_patches) -> dict:
    """Compare this file's features and forward pass with `tronzero dump` output."""
    samples = load_samples(prefix + ".samples")
    n = len(samples)
    rec = np.dtype([("logits", "<f4", (4,)), ("value", "<f4", (4,)), ("dist", "u1", (4, H * W))])
    out = np.fromfile(prefix + ".outputs", dtype=rec)
    assert len(out) == n, (len(out), n)
    data = add_dists(samples, chunk=n)
    batch = {k: jnp.asarray(v) for k, v in data.items()}
    cv, scal = inputs(batch)
    logits, value = forward(params, cv, scal, conv)
    return {
        "positions": n,
        "by_players_alive": np.bincount(samples["n_alive"], minlength=5)[2:].tolist(),
        "dist_equal": bool((data["dist"] == out["dist"]).all()),
        "max_logit_diff": float(np.abs(np.asarray(logits) - out["logits"]).max()),
        "max_value_diff": float(np.abs(np.asarray(value) - out["value"]).max()),
    }
