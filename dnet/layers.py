from typing import Callable

from jax import jit

from dnet import activations


class FC:

    def __init__(self, units: int, activation: str) -> None:
        self.units: int = units
        self.activation: Callable = jit(getattr(activations, activation))
