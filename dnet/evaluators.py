import jax.numpy as tensor
from jax import jit


@jit
def binary_crossentropy(predictions: tensor.array, outputs: tensor.array, threshold: float = 0.90) -> float:
    prediction_labels: tensor.array = tensor.where(predictions >= threshold, 1, 0)
    return tensor.mean(prediction_labels == outputs)
