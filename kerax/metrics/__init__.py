from ..utils import Tensor, jnp


def binary_accuracy(predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
    """
    Computes the accuracy for a binary-classification task.
    :param predictions: Outputs of the network.
    :param targets: Expected outputs of the network.
    :param kwargs: Keyword arguments which contains the 'acc_threshold' value.
    :return: the mean accuracy
    """
    threshold = kwargs.get("acc_thresh", 0.50)
    assert 0 < threshold < 1, "Threshold should be between 0 and 1"
    predictions = jnp.where(predictions > threshold, 1.0, 0.0)
    return jnp.mean(predictions == targets)


def accuracy(predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
    """
    Computes the accuracy for a multi-class classification task
    :param predictions: Outputs of the network.
    :param targets: Expected outputs of the network.
    :param kwargs: Keyword arguments (if any)
    :return: the mean accuracy
    """
    predicted_class = jnp.argmax(predictions, axis=1)
    target_class = jnp.argmax(targets, axis=1)
    return jnp.mean(predicted_class == target_class)


__all__ = [
    "binary_accuracy",
    "accuracy"
]
