from typing import Union

from jax.experimental.stax import Dense, Flatten
from ..utils import jnp, random

from ..utils import Tensor


def Dropout(rate: Union[Tensor, float]):
    """Layer construction function for a dropout layer with given rate."""

    def init_fun(rng, input_shape):
        return input_shape, ()

    def apply_fun(params, inputs, **kwargs):
        mode = kwargs.get('mode', 'train')
        rng = random.PRNGKey(seed=0)
        if mode == 'train':
            keep = random.bernoulli(rng, rate, inputs.shape)
            return jnp.where(keep, inputs / rate, 0)
        else:
            return inputs

    return init_fun, apply_fun
