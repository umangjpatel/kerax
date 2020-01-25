import jax.numpy as tensor
from jax import jit


@jit
def binary_crossentropy(outputs: tensor.array, targets: tensor.array) -> float:
    output_labels: tensor.array = tensor.where(outputs > 0.50, 1.0, 0.0)
    return tensor.mean(output_labels == targets)


@jit
def categorical_crossentropy(outputs: tensor.array, targets: tensor.array) -> float:
    target_class = tensor.argmax(targets, axis=1)
    predicted_class = tensor.argmax(outputs, axis=1)
    return tensor.mean(predicted_class == target_class)
