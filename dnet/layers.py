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
        self.init_params()

    def init_params(self) -> None:
        FC._key, subkey = random.split(FC._key)
        weights: tensor.array = self.weight_scheme(key=FC._key, shape=(FC._prev_units, self.units))
        bias: tensor.array = self.bias_scheme(key=subkey, shape=(1, self.units))
        self.params: Dict[str, tensor.array] = {"w": weights, "b": bias}
        FC._prev_units = self.units

    def forward(self, params: Dict[str, tensor.array], inputs: tensor.array) -> tensor.array:
        z: tensor.array = tensor.dot(inputs, params.get("w")) + params.get("b")
        return self.activation(z)

    def update_params(self, params: Dict[str, tensor.array]) -> None:
        self.params = params


class Dropout(Layer):
    _key: tensor.array = None

    def __init__(self, keep_prob: float) -> None:
        if Dropout._key is None:
            Dropout._key = random.PRNGKey(0)
        self.keep_prob: float = keep_prob
        self.init_params()

    def init_params(self) -> None:
        self.params: Dict[str, tensor.array] = {}

    def forward(self, params: Dict[str, tensor.array], inputs: tensor.array) -> tensor.array:
        d: tensor.array = random.bernoulli(key=Dropout._key, p=self.keep_prob) / self.keep_prob
        return inputs * d

    def update_params(self, params: Dict[str, tensor.array]) -> None:
        self.params = params


class BatchNorm(Layer):
    _key: tensor.array = None

    def __init__(self, units: int) -> None:
        if BatchNorm._key is None:
            BatchNorm._key = random.PRNGKey(0)
        self.units: int = units
        self.eps: float = 1e-08
        self.gamma_scheme: Callable = getattr(weight_initializers, "ones")
        self.beta_scheme: Callable = getattr(weight_initializers, "zeros")
        self.init_params()

    def init_params(self) -> None:
        BatchNorm._key, subkey = random.split(BatchNorm._key)
        gamma: tensor.array = self.gamma_scheme(key=BatchNorm._key, shape=(1, self.units))
        beta: tensor.array = self.beta_scheme(key=subkey, shape=(1, self.units))
        self.params: Dict[str, tensor.array] = {"g": gamma, "b": beta}

    def forward(self, params: Dict[str, tensor.array], inputs: tensor.array) -> tensor.array:
        mean: tensor.array = tensor.mean(inputs, axis=0, keepdims=True)
        variance: tensor.array = tensor.var(inputs, axis=0, keepdims=True)
        z_norm: tensor.array = (inputs - mean) / tensor.sqrt(variance + self.eps)
        return params.get("g") * z_norm + params.get("b")

    def update_params(self, params: Dict[str, tensor.array]) -> None:
        self.params = params
