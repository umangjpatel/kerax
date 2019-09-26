import jax.numpy as np
from jax import jit


@jit
def binary_crossentropy(a, y):
    return -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))
