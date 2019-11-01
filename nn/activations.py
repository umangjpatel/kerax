import jax.numpy as np
from jax import jit


@jit
def elu(x, alpha=1.0):
    return relu(x) + np.minimum(0, alpha * (np.exp(x) - 1.0))


@jit
def leaky_relu(x, neg_slope=1e-2):
    return relu(x) + neg_slope * np.minimum(0.0, x)


@jit
def log_sigmoid(x):
    return np.log(sigmoid(x))


@jit
def relu_6(x):
    return np.minimum(relu(x), 6.0)


@jit
def celu(x, alpha=1.0):
    return relu(x) + np.minimum(0.0, alpha * (np.exp(x / alpha)) - 1.0)


@jit
def softplus(x, beta=1):
    return np.log(1 + np.exp(beta * x)) / beta


@jit
def softsign(x):
    return x / (1 + np.abs(x))


@jit
def tanh_shrink(x):
    return x - tanh(x)


@jit
def mish(x):
    return x * tanh(softplus(x))


@jit
def relu(x):
    return np.maximum(0, x)


@jit
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@jit
def tanh(x):
    return np.tanh(x)


@jit
def linear(x):
    return x
