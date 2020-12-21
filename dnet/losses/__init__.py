from jax import jit
import jax.numpy as jnp
from dnet.utils.tensor import Tensor


@jit
def BCELoss(predictions: Tensor, targets: Tensor) -> Tensor:
    return - jnp.mean(a=(targets * jnp.log(predictions) + (1 - targets) * jnp.log(1 - predictions)))
