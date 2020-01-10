from typing import Tuple, Callable

from jax.experimental import optimizers


def sgd(lr: float) -> Tuple[Callable, Callable, Callable]:
    return optimizers.sgd(step_size=lr)


def rmsprop(lr: float, beta: float = 0.9) -> Tuple[Callable, Callable, Callable]:
    return optimizers.rmsprop(step_size=lr, gamma=beta)


def momentum(lr: float, beta: float = 0.9) -> Tuple[Callable, Callable, Callable]:
    return optimizers.momentum(step_size=lr, mass=beta)


def adam(lr: float, beta1: float = 0.9, beta2: float = 0.999) -> Tuple[Callable, Callable, Callable]:
    return optimizers.adam(step_size=lr, b1=beta1, b2=beta2)


def adagrad(lr: float, beta: float = 0.9) -> Tuple[Callable, Callable, Callable]:
    return optimizers.adagrad(step_size=lr, momentum=beta)
