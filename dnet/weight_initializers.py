from typing import Tuple

import jax.numpy as tensor
from jax.nn import initializers


def zeros(key: tensor.array, shape: Tuple[int, int]):
    return initializers.zeros(key=key, shape=shape)


def ones(key: tensor.array, shape: Tuple[int, int]):
    return initializers.ones(key=key, shape=shape)


def normal(key: tensor.array, shape: Tuple[int, int], weight_scale: float = 0.1) -> tensor.array:
    return initializers.normal(stddev=weight_scale)(key=key, shape=shape)


def glorot_normal(key: tensor.array, shape: Tuple[int, int]):
    return initializers.glorot_normal()(key=key, shape=shape)


def glorot_uniform(key: tensor.array, shape: Tuple[int, int]):
    return initializers.glorot_uniform()(key=key, shape=shape)


def he_normal(key: tensor.array, shape: Tuple[int, int]):
    return initializers.he_normal()(key=key, shape=shape)


def he_uniform(key: tensor.array, shape: Tuple[int, int]):
    return initializers.he_uniform()(key=key, shape=shape)
