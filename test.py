import jax
from jax import numpy as jnp

@jax.jit
def func_with_warning(y):
    return jnp.identity(y.shape[-1]) + jnp.matmul(y, y)

print(func_with_warning(jnp.ones((50, 100, 100))).shape)
