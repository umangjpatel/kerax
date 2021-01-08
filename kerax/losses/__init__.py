from ..utils import Tensor, jnp


def BCELoss(predictions: Tensor, targets: Tensor) -> Tensor:
    """
    BCE or Binary Cross Entropy loss function.
    Useful for binary classification tasks.
    :param predictions: Outputs of the network.
    :param targets: Expected outputs of the network.
    :return: binary cross-entropy loss value
    """
    return -jnp.mean(a=(targets * jnp.log(predictions) + (1 - targets) * jnp.log(1 - predictions)))


def CCELoss(predictions: Tensor, targets: Tensor) -> Tensor:
    """
    CCE or Categorical Cross Entropy loss function.
    Useful for multi-class classification task.
    :param predictions: Outputs of the network
    :param targets: Expected outputs of the network.
    :return: categorical cross-entopy loss value.
    """
    return -jnp.mean(jnp.sum(predictions * targets, axis=1))


__all__ = [
    "BCELoss",
    "CCELoss"
]
