import jax.numpy as tensor
from jax import jit


@jit
def binary_crossentropy(predictions: tensor.array, outputs: tensor.array) -> float:
    return -tensor.mean(outputs * tensor.log(predictions) + (1 - outputs) * tensor.log(1 - predictions))
