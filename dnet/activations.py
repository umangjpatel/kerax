import jax.numpy as tensor
from jax import jit


@jit
def sigmoid(z: tensor.array) -> tensor.array:
    return 1 / (1 + tensor.exp(-z))


@jit
def relu(z: tensor.array) -> tensor.array:
    return tensor.maximum(0.0, z)
