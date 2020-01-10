from typing import Callable, Optional, Dict

import jax.numpy as tensor
from jax import random

from dnet import activations
from dnet import weight_initializers


class Layer:
    pass


class FC(Layer):
    _prev_units: Optional[int] = None
    _key: tensor.array = None

    def __init__(self, units: int, activation: str = "linear", weight_scheme: str = "glorot_uniform",
                 bias_scheme="zeros",
                 input_dim: Optional[int] = None) -> None:
        self.units: int = units
        self.act_name: str = activation
        self.activation: Callable[[tensor.array], tensor.array] = getattr(activations, activation)
        self.weight_scheme: Callable = getattr(weight_initializers, weight_scheme)
        self.bias_scheme: Callable = getattr(weight_initializers, bias_scheme)
        if FC._prev_units is None:
            FC._prev_units = input_dim
            FC._key = random.PRNGKey(0)
        self.init_weights()

    def init_weights(self) -> None:
        FC._key, subkey = random.split(FC._key)
        self.weights: tensor.array = self.weight_scheme(key=FC._key, shape=(FC._prev_units, self.units))
        self.bias: tensor.array = self.bias_scheme(key=subkey, shape=(1, self.units))
        FC._prev_units = self.units

    def forward(self, params: Dict[str, tensor.array], inputs: tensor.array) -> tensor.array:
        z: tensor.array = tensor.dot(inputs, params.get("w")) + params.get("b")
        return self.activation(z)

    def update_weights(self, params: Dict[str, tensor.array]) -> None:
        self.weights = params.get("w")
        self.bias = params.get("b")

    def __str__(self) -> str:
        return f"""
        FC layer info :-
        {"-" * 10}
        # units => {self.units},
        Activation fn => {self.act_name},
        Weights dims => {self.weights.shape}
        Weights scheme => {self.weight_scheme},
        Bias dims => {self.bias.shape},
        Bias scheme => {self.bias_scheme}
        {"-" * 10}"""
