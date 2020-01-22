from typing import Callable, Tuple

import jax.numpy as tensor
from jax import jit
from jax.experimental.stax import Sigmoid, Relu, Tanh, LogSoftmax
from jax.nn import functions


@jit
def sigmoid() -> Tuple[Callable, Callable]:
    return Sigmoid


@jit
def tanh() -> Tuple[Callable, Callable]:
    return Tanh


@jit
def relu() -> Tuple[Callable, Callable]:
    return Relu


@jit
def softmax() -> Tuple[Callable, Callable]:
    return LogSoftmax


@jit
def mish() -> Tuple[Callable, Callable]:
    init_fun: Callable = lambda rng, input_shape: (input_shape, ())
    apply_fun: Callable = lambda params, inputs, **kwargs: inputs * tensor.tanh(functions.softplus(inputs))
    return init_fun, apply_fun


@jit
def linear() -> Tuple[Callable, Callable]:
    init_fun: Callable = lambda rng, input_shape: (input_shape, ())
    apply_fun: Callable = lambda params, inputs, **kwargs: inputs
    return init_fun, apply_fun
