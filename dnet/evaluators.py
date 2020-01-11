import jax.numpy as tensor
from jax import jit


@jit
def binary_crossentropy(outputs: tensor.array, targets: tensor.array) -> float:
    output_labels = tensor.where(outputs > (1.0 - outputs), 1.0, 0.0)
    return tensor.mean(output_labels == targets)
