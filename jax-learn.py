import jax.numpy as jnp
from jax import jit
import time
import jax


@jax.jit
def relu(x):
    return jnp.where(x > 0, x, 0)


x = jnp.arange(-5, 5)


relu(x).block_until_ready()

for _ in range(100):
    start_t = time.time()
    y = relu(x)
    y.block_until_ready()
    end_t = time.time()
    print((end_t - start_t) * 1_000_000)
