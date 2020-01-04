from torch import Tensor
from dnet import activations
from typing import Callable


class Layer:
    pass


class FC(Layer):

    def __init__(self, units: int, activation: str = "linear") -> None:
        self.units: int = units
        self.activation_name: str = activation
        self.activation: Callable[[Tensor], Tensor] = getattr(activations, activation)
