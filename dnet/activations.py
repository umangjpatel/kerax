import jax.nn as activations
import jax.numpy as tensor
from jax import jit


@jit
def linear(z: tensor.array) -> tensor.array:
    return z


@jit
def sigmoid(z: tensor.array) -> tensor.array:
    return activations.sigmoid(z)


@jit
def tanh(z: tensor.array) -> tensor.array:
    return tensor.tanh(z)


@jit
def relu(z: tensor.array) -> tensor.array:
    return activations.relu(z)


@jit
def softplus(z: tensor.array) -> tensor.array:
    return activations.softplus(z)


@jit
def mish(z: tensor.array) -> tensor.array:
    return z * tanh(softplus(z))


@jit
def softmax(z: tensor.array) -> tensor.array:
    return activations.softmax(z, axis=1)
