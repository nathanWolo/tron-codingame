"""tronzero's network, features and losses in JAX.

The Rust engine (`engine/src/net.rs`) computes the same features and the same
forward pass; `check_parity` compares the two on positions the engine dumps.
"""

from __future__ import annotations

import struct

import jax
import jax.numpy as jnp
import numpy as np

W, H = 30, 20
CW, CH = 59, 39
NPLANES, NSCAL = 6, 6
UNREACH = 255

# One training position as `tronzero selfplay` writes it (104 bytes, little endian).
SAMPLE_DTYPE = np.dtype(
    [
        ("occ", "<u4", (20,)),  # row bitboards: bit x of row y is a trail or head
        ("heads", "u1", (4,)),  # mover x, y, opponent x, y
        ("visits", "<u2", (4,)),  # root visits of UP, DOWN, LEFT, RIGHT
        ("legal", "u1"),  # bit d: the mover can step in direction d
        ("z", "i1"),  # +1 if the mover won the game, -1 if it lost
        ("flags", "u1"),  # bit 0: forced move, not searched
        ("pad", "u1"),
        ("q", "<f4"),  # the search's root value for the mover
        ("ply", "<u2"),
        ("sims", "<u2"),
    ]
)
assert SAMPLE_DTYPE.itemsize == 104


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
        "w": jax.random.normal(ks[6], (hidden, 1)) * 0.01,
        "b": jnp.zeros((1,)),
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
        f.write(b"TZN1")
        f.write(struct.pack("<6I", NPLANES, NSCAL, c[0], c[1], c[2], hidden))
        for i in range(3):
            f.write(p[f"conv{i}"]["w"].tobytes())  # [ky, kx, cin, cout]
            f.write(p[f"conv{i}"]["b"].tobytes())
        f.write(p["hidden"]["w"].tobytes())  # [(y * 8 + x) * c3 + ch, hidden]
        f.write(p["hidden"]["b"].tobytes())
        f.write(p["feat"]["w"].tobytes())  # [scalar, hidden]
        f.write(np.ascontiguousarray(p["policy"]["w"].T).tobytes())  # [move, hidden]
        f.write(p["policy"]["b"].tobytes())
        f.write(p["value"]["w"][:, 0].tobytes())
        f.write(p["value"]["b"].tobytes())


def save_params(params, path) -> None:
    flat = {}
    for name, layer in params.items():
        for k, v in layer.items():
            flat[f"{name}/{k}"] = np.asarray(v)
    np.savez(path, **flat)


def load_params(path):
    params = {}
    with np.load(path) as z:
        for key in z.files:
            name, k = key.split("/")
            params.setdefault(name, {})[k] = jnp.asarray(z[key])
    return params


# --- Features -----------------------------------------------------------------------


def unpack_occ(occ_bits):
    """Row bitboards ``u32[B, 20]`` to occupancy ``bool[B, 20, 30]``."""
    cols = jnp.arange(W, dtype=jnp.uint32)
    return ((occ_bits[:, :, None] >> cols) & 1).astype(bool)


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


def board_planes(occ_bits, heads, dist_me, dist_opp):
    """The six board planes ``f32[B, 20, 30, 6]`` and the scalars ``f32[B, 6]``."""
    b = occ_bits.shape[0]
    empty = ~unpack_occ(occ_bits)
    dm = dist_me.reshape(b, H, W).astype(jnp.int32)
    do = dist_opp.reshape(b, H, W).astype(jnp.int32)
    reach_me = empty & (dm != UNREACH)
    reach_opp = empty & (do != UNREACH)
    mine = reach_me & (dm <= do)
    theirs = reach_opp & (do < dm)
    opp_head = (
        jnp.zeros((b, H, W), bool)
        .at[jnp.arange(b), heads[:, 3].astype(jnp.int32), heads[:, 2].astype(jnp.int32)]
        .set(True)
    )
    near_me = jnp.where(reach_me, (64 - jnp.minimum(dm, 64)) / 64.0, 0.0)
    near_opp = jnp.where(reach_opp, (64 - jnp.minimum(do, 64)) / 64.0, 0.0)
    planes = jnp.stack(
        [
            empty.astype(jnp.float32),
            opp_head.astype(jnp.float32),
            mine.astype(jnp.float32),
            theirs.astype(jnp.float32),
            near_me.astype(jnp.float32),
            near_opp.astype(jnp.float32),
        ],
        axis=-1,
    )
    terr_me = mine.sum((1, 2)).astype(jnp.float32)
    terr_opp = theirs.sum((1, 2)).astype(jnp.float32)
    scal = jnp.stack(
        [
            terr_me / 128.0,
            terr_opp / 128.0,
            (terr_me - terr_opp) / 32.0,
            reach_me.sum((1, 2)) / 256.0,
            reach_opp.sum((1, 2)) / 256.0,
            empty.sum((1, 2)) / 600.0,
        ],
        axis=-1,
    ).astype(jnp.float32)
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
    """Policy logits ``[B, 4]`` and values ``[B]`` in [-1, 1] for canvases ``[B, 39, 59, 6]``."""
    x = cv
    for i in range(3):
        x = conv(x, params[f"conv{i}"]["w"], params[f"conv{i}"]["b"])
    x = x.reshape(x.shape[0], -1)
    h = jax.nn.relu(x @ params["hidden"]["w"] + params["hidden"]["b"] + scal @ params["feat"]["w"])
    logits = h @ params["policy"]["w"] + params["policy"]["b"]
    value = jnp.tanh(h @ params["value"]["w"] + params["value"]["b"])[:, 0]
    return logits, value


def inputs(batch, flip_x=None, flip_y=None):
    """Canvases and scalars of a batch of samples (dict of arrays), optionally mirrored."""
    planes, scal = board_planes(batch["occ"], batch["heads"], batch["dist_me"], batch["dist_opp"])
    mx = batch["heads"][:, 0].astype(jnp.int32)
    my = batch["heads"][:, 1].astype(jnp.int32)
    if flip_x is not None:
        planes, mx, my = flip(planes, mx, my, flip_x, flip_y)
    return canvas(planes, mx, my), scal


def legal_bits(legal):
    return ((legal[:, None] >> jnp.arange(4, dtype=jnp.uint8)) & 1).astype(bool)


def loss_fn(params, batch, flip_x, flip_y, conv=conv_patches):
    cv, scal = inputs(batch, flip_x, flip_y)
    logits, value = forward(params, cv, scal, conv)
    legal = flip_moves(legal_bits(batch["legal"]), flip_x, flip_y)
    visits = flip_moves(batch["visits"].astype(jnp.float32), flip_x, flip_y)
    target = visits / jnp.maximum(visits.sum(-1, keepdims=True), 1.0)
    logp = jax.nn.log_softmax(jnp.where(legal, logits, -1e9), axis=-1)
    policy_loss = -jnp.sum(target * jnp.where(legal, logp, 0.0), axis=-1).mean()
    z = batch["z"].astype(jnp.float32)
    value_loss = jnp.mean((value - z) ** 2)
    entropy = -jnp.sum(jnp.where(target > 0, target * jnp.log(jnp.maximum(target, 1e-9)), 0.0), -1)
    metrics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "target_entropy": entropy.mean(),
        "value_sign_acc": jnp.mean((value * z) > 0),
    }
    return policy_loss + value_loss, metrics


# --- Data -----------------------------------------------------------------------------


def load_samples(path) -> np.ndarray:
    return np.fromfile(path, dtype=SAMPLE_DTYPE)


def add_dists(samples: np.ndarray, chunk: int = 32768) -> dict:
    """The arrays training needs, with both distance maps computed here."""
    n = len(samples)
    dist_me = np.empty((n, H * W), np.uint8)
    dist_opp = np.empty((n, H * W), np.uint8)
    for s in range(0, n, chunk):
        part = samples[s : s + chunk]
        m = len(part)
        if m < chunk:  # pad so the jitted BFS sees one shape
            part = np.concatenate([part, np.repeat(part[-1:], chunk - m)])
        empty = ~unpack_occ(jnp.asarray(part["occ"]))
        hd = part["heads"].astype(np.int32)
        dm = bfs_dist(empty, jnp.asarray(hd[:, 0]), jnp.asarray(hd[:, 1]))
        do = bfs_dist(empty, jnp.asarray(hd[:, 2]), jnp.asarray(hd[:, 3]))
        dist_me[s : s + m] = np.asarray(dm).reshape(chunk, -1)[:m]
        dist_opp[s : s + m] = np.asarray(do).reshape(chunk, -1)[:m]
    return {
        "occ": samples["occ"].copy(),
        "heads": samples["heads"].copy(),
        "visits": samples["visits"].copy(),
        "legal": samples["legal"].copy(),
        "z": samples["z"].copy(),
        "dist_me": dist_me,
        "dist_opp": dist_opp,
    }


def check_parity(params, prefix, conv=conv_patches) -> dict:
    """Compare this file's features and forward pass with `tronzero dump` output."""
    samples = load_samples(prefix + ".samples")
    n = len(samples)
    rec = np.dtype([("logits", "<f4", (4,)), ("value", "<f4"), ("dist", "u1", (2, H * W))])
    out = np.fromfile(prefix + ".outputs", dtype=rec)
    assert len(out) == n, (len(out), n)
    data = add_dists(samples, chunk=n)
    dist_ok = bool(
        (data["dist_me"] == out["dist"][:, 0]).all() and (data["dist_opp"] == out["dist"][:, 1]).all()
    )
    batch = {k: jnp.asarray(v) for k, v in data.items()}
    cv, scal = inputs(batch)
    logits, value = forward(params, cv, scal, conv)
    return {
        "positions": n,
        "dist_equal": dist_ok,
        "max_logit_diff": float(np.abs(np.asarray(logits) - out["logits"]).max()),
        "max_value_diff": float(np.abs(np.asarray(value) - out["value"]).max()),
    }
