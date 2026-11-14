import time, jax, jax.numpy as jnp
print('jax', jax.__version__, jax.devices())
x = jnp.ones((2048, 40, 60, 8)); w = jnp.ones((3, 3, 8, 16)) * 0.01
def loss(w, x):
    y = jax.lax.conv_general_dilated(x, w, (2, 2), 'SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return jnp.mean(jax.nn.relu(y) ** 2)
f = jax.jit(jax.value_and_grad(loss))
t = time.time(); v, g = f(w, x); print(float(v), float(jnp.abs(g).sum()), 'compile+run s', round(time.time() - t, 2))
t = time.time()
for _ in range(50): v, g = f(w, x)
print(float(v), 'ms per fwd+bwd', round((time.time() - t) / 50 * 1000, 2))
