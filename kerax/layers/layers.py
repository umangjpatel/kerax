from typing import Union

from jax.experimental.stax import Dense, Flatten
from ..utils import jnp, random

from ..utils import Tensor


def Dropout(rate: Union[Tensor, float]):
    """
    Implementation of the Dropout layer.
    When using the apply_fun, you can pass a mode kwarg.
    This is helpful when you're using the network for validation / prediction.
    :param rate: Probability / Rate at which we wish to 'knock off' the neurons.
    :return:
    """

    def init_fun(rng, input_shape):
        """
        Initializes the function. This layer doesn't perform any parameter initialization.
        :param rng: a PRNG key for randomization.
        :param input_shape: Shape of the inputs received from the previous layer.
        :return: a tuple of the output shape and initialized params. Since no params involved, sent an empty tuple.
        """
        return input_shape, ()

    def apply_fun(params, inputs, **kwargs):
        """
        Performs computation of the layer
        :param params: Parameters of the layer
        :param inputs: Inputs for the layer
        :param kwargs: Keyword arguments for additional info while computing the inputs
        :return: the computed outputs.
        """
        mode = kwargs.get('mode', 'train')
        rng = random.PRNGKey(seed=0)
        if mode == 'train':
            keep = random.bernoulli(rng, rate, inputs.shape)
            return jnp.where(keep, inputs / rate, 0)
        else:
            return inputs

    return init_fun, apply_fun
