from ..utils import Tensor, jnp


def BCELoss(predictions: Tensor, targets: Tensor) -> Tensor:
    return -jnp.mean(a=(targets * jnp.log(predictions) + (1 - targets) * jnp.log(1 - predictions)))


def CCELoss(predictions: Tensor, targets: Tensor) -> Tensor:
    return -jnp.mean(jnp.sum(predictions * targets, axis=1))


__all__ = [
    "BCELoss",
    "CCELoss"
]
