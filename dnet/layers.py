from typing import Callable, List, Tuple

from jax.experimental.stax import Dense

from dnet import activations


class Layer:
    pass


class FC(Layer):

    def __init__(self, units: int, activation: str = "linear") -> None:
        self.units: int = units
        layer_activation: Tuple[Callable, Callable] = getattr(activations, activation)()
        self.layer: List = [Dense(out_dim=self.units), layer_activation]
