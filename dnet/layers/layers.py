from typing import Union

from jax.experimental.stax import Dense, Flatten
from dnet.utils.tensor import Tensor


def dropout(rate: Union[Tensor, float]):
    """Layer construction function for a dropout layer with given rate."""

    def init_fun(rng, input_shape):
        return input_shape, ()

    def apply_fun(params, inputs, **kwargs):
        import jax.numpy as jnp
        from jax import random
        mode = kwargs.get('mode', 'train')
        rng = random.PRNGKey(seed=0)
        if rng is None:
            msg = ("Dropout layer requires apply_fun to be called with a PRNG key "
                   "argument. That is, instead of `apply_fun(params, inputs)`, call "
                   "it like `apply_fun(params, inputs, rng)` where `rng` is a "
                   "jax.random.PRNGKey value.")
            raise ValueError(msg)
        if mode == 'train':
            keep = random.bernoulli(rng, rate, inputs.shape)
            return jnp.where(keep, inputs / rate, 0)
        else:
            return inputs

    return init_fun, apply_fun
