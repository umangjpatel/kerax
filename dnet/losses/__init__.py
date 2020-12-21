import jax.numpy as jnp
from dnet.utils.tensor import Tensor
from typing import List, Callable
from jax.interpreters.ad import JVPTracer


def BCELoss(params: List, net_apply: Callable, inputs: Tensor, targets: Tensor) -> Tensor:
    predictions: JVPTracer = net_apply(params, inputs)
    return - jnp.mean(a=(targets * jnp.log(predictions) + (1 - targets) * jnp.log(1 - predictions)))
