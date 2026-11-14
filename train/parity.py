"""Check the Rust engine against the JAX network on random positions.

    python train/parity.py <engine exe> <work dir>

Writes a random network, has the engine dump its features and outputs for a
few hundred positions, and compares them with `tz.py`. Also times the two
convolution formulations and compares the device's gradients with the CPU's.
"""

import os
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tz  # noqa: E402

exe, work = os.path.abspath(sys.argv[1]), sys.argv[2]
os.makedirs(work, exist_ok=True)
print("devices", jax.devices())

params = tz.init_params(jax.random.key(0))
# Biases of zero would hide background errors: make every bias non-zero.
params = jax.tree.map(lambda x: x, params)
for i, name in enumerate(["conv0", "conv1", "conv2", "hidden", "policy", "value"]):
    b = params[name]["b"]
    params[name]["b"] = jax.random.normal(jax.random.key(100 + i), b.shape) * 0.3
print("parameters", tz.count_params(params))
tz.export_net(params, os.path.join(work, "parity.bin"))
r = subprocess.run(
    [exe, "dump", "--net", "parity.bin", "--n", "512", "--out", "parity"],
    cwd=work, capture_output=True, text=True,
)
print(r.stdout.strip(), r.stderr.strip())
prefix = os.path.join(work, "parity")
for name, conv in [("patches", tz.conv_patches), ("lax", tz.conv_lax)]:
    print(name, tz.check_parity(params, prefix, conv))

# Gradients: this device against the CPU, and the time of a training step.
samples = tz.load_samples(prefix + ".samples")
data = tz.add_dists(samples, chunk=len(samples))
rng = np.random.default_rng(0)
data["visits"] = rng.integers(0, 30, data["visits"].shape).astype(np.uint16)
data["rank"] = np.where(np.arange(4)[None, :] < data["n_alive"][:, None], np.arange(4)[None, :] + 1, 0).astype(np.uint8)
batch = {k: jnp.asarray(v) for k, v in data.items()}
fx = jnp.asarray(rng.random(len(samples)) < 0.5)
fy = jnp.asarray(rng.random(len(samples)) < 0.5)
cpu = jax.devices("cpu")[0]
for name, conv in [("patches", tz.conv_patches), ("lax", tz.conv_lax)]:
    g = jax.jit(jax.grad(lambda p, b, x, y: tz.loss_fn(p, b, x, y, conv)[0]))
    t = time.time()
    dev = g(params, batch, fx, fy)
    jax.block_until_ready(dev)
    compile_s = time.time() - t
    t = time.time()
    for _ in range(20):
        dev = g(params, batch, fx, fy)
    jax.block_until_ready(dev)
    step_ms = (time.time() - t) / 20 * 1000
    put = lambda tree: jax.tree.map(lambda x: jax.device_put(x, cpu), tree)  # noqa: E731
    ref = g(put(params), put(batch), jax.device_put(fx, cpu), jax.device_put(fy, cpu))
    diff = max(
        float(np.abs(np.asarray(a) - np.asarray(b)).max())
        for a, b in zip(jax.tree.leaves(dev), jax.tree.leaves(ref))
    )
    scale = max(float(np.abs(np.asarray(b)).max()) for b in jax.tree.leaves(ref))
    print(f"{name}: grad max diff vs cpu {diff:.2e} (max grad {scale:.2e}), "
          f"compile {compile_s:.1f}s, {step_ms:.1f} ms per grad at batch {len(samples)}")
