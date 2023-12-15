import jax.numpy as jnp

from jax import grad
from jax import random


def test_jax():
    key = random.PRNGKey(0)
    random.normal(key, (10,))

    grad(jnp.tanh)
