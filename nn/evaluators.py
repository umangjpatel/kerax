import jax.numpy as np
from jax import jit


@jit
def binary_crossentropy(preds, targets):
    pred_labels = np.where(1 - preds > preds, 0, 1).flatten()
    return np.mean(pred_labels == targets)


@jit
def mse(preds, targets):
    return np.mean((preds - targets) ** 2)


@jit
def mae(preds, targets):
    return np.mean(np.abs(preds - targets))


@jit
def rmse(a, y):
    return mse(a, y) ** 0.5
