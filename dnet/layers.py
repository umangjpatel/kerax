from typing import Callable, List, Tuple

from jax.experimental.stax import Dense, Flatten as Flat, Conv, MaxPool, BatchNorm as BatchNormalisation

from dnet import activations


class Layer:
    pass


class FC(Layer):

    def __init__(self, units: int, activation: str = "linear") -> None:
        layer_activation: Tuple[Callable, Callable] = getattr(activations, activation)()
        self.layer: List = [Dense(out_dim=units), layer_activation]


class Flatten(Layer):

    def __init__(self) -> None:
        self.layer: List = [Flat]


class Conv2D(Layer):

    def __init__(self, filters: int, kernel_size: Tuple[int, int], strides: Tuple[int, int] = (1, 1),
                 padding: str = "valid",
                 activation: str = "linear") -> None:
        layer_activation: Tuple[Callable, Callable] = getattr(activations, activation)()
        self.layer: List = [Conv(out_chan=filters, filter_shape=kernel_size, strides=strides, padding=padding.upper()),
                            layer_activation]


class MaxPool2D(Layer):

    def __init__(self, pool_size: Tuple[int, int], padding: str = "valid") -> None:
        self.layer: List = [MaxPool(window_shape=pool_size, padding=padding.upper(), spec="NHWC")]


class BatchNorm(Layer):

    def __init__(self):
        self.layer: List = [BatchNormalisation()]
