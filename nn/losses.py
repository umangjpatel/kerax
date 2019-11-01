import jax.numpy as np
from jax import jit


@jit
def binary_crossentropy(a, y):
    return -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))


@jit
def mse(a, y):
    return np.mean((a - y) ** 2)


@jit
def mae(a, y):
    return np.mean(np.abs(a - y))


@jit
def rmse(a, y):
    return mse(a, y) ** 0.5
