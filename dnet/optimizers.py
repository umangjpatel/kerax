import functools
from typing import Tuple, Callable

from jax import jit
from jax.experimental import optimizers


@functools.partial(jit, static_argnums=0)
def sgd(lr: float) -> Tuple[Callable, ...]:
    return optimizers.sgd(step_size=lr)


@functools.partial(jit, static_argnums=0)
def rmsprop(lr: float) -> Tuple[Callable, ...]:
    beta: float = 0.9
    return optimizers.rmsprop(step_size=lr, gamma=beta)


@functools.partial(jit, static_argnums=0)
def momentum(lr: float) -> Tuple[Callable, ...]:
    beta: float = 0.9
    return optimizers.momentum(step_size=lr, mass=beta)


@functools.partial(jit, static_argnums=0)
def adam(lr: float) -> Tuple[Callable, ...]:
    beta1: float = 0.9
    beta2: float = 0.999
    return optimizers.adam(step_size=lr, b1=beta1, b2=beta2)


@functools.partial(jit, static_argnums=0)
def adagrad(lr: float) -> Tuple[Callable, ...]:
    beta: float = 0.9
    return optimizers.adagrad(step_size=lr, momentum=beta)


@functools.partial(jit, static_argnums=0)
def sm3(lr: float) -> Tuple[Callable, ...]:
    beta: float = 0.9
    return optimizers.sm3(step_size=lr, momentum=beta)
