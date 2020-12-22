import jax.numpy as jnp

from ..utils.tensor import Tensor


def BCELoss(predictions: Tensor, targets: Tensor) -> Tensor:
    return - jnp.mean(a=(targets * jnp.log(predictions) + (1 - targets) * jnp.log(1 - predictions)))


__all__ = [
    "BCELoss"
]
