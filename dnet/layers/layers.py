from typing import Union

import jax.numpy as jnp
from jax import random
from jax.nn.initializers import glorot_normal, normal

from dnet.layers import activations
from dnet.utils.tensor import Tensor


def dense(units: int, activation: str, w_init=glorot_normal(), b_init=normal()):
    """Layer constructor function for a dense (fully-connected) layer."""
    _, act_apply_fn = getattr(activations, activation)

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (units,)
        k1, k2 = random.split(rng)
        w, b = w_init(k1, (input_shape[-1], units)), b_init(k2, (units,))
        return output_shape, (w, b)

    def apply_fun(params, inputs, **kwargs):
        w, b = params
        z = jnp.dot(inputs, w) + b
        return act_apply_fn(params, z, **kwargs)

    return init_fun, apply_fun


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
