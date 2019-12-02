from typing import Dict, Tuple, List, Callable

import jax.numpy as tensor
from jax import grad, jit
from jax import random
from tqdm import tqdm

from dnet.layers import FC


class Optimizer:

    def __init__(self, layers: List[FC], loss: Callable, accuracy: Callable, epochs: int, lr: float) -> None:
        self.layers: List[FC] = layers
        self.loss_fn: Callable = loss
        self.accuracy_fn: Callable = accuracy
        self.epochs: int = epochs
        self.lr: float = lr
        self.network_params: List[Dict[str, tensor.array]] = []
        self.cost: List[float] = []
        self.accuracy: List[float] = []

    def train(self, inputs: tensor.array, outputs: tensor.array) -> None:
        raise NotImplementedError

    def init_network_params(self, input_shape: Tuple[int, int]) -> None:
        key: tensor.array = random.PRNGKey(0)
        subkey: tensor.array
        for i, layer in enumerate(self.layers):
            key, subkey = random.split(key)
            weight_shape: Tuple[int, int] = (layer.units, self.layers[i - 1].units if i != 0 else input_shape[0])
            w: tensor.array = random.normal(subkey, shape=weight_shape) * 0.01
            b: tensor.array = tensor.zeros(shape=(layer.units, 1))
            self.network_params.append({"w": w, "b": b})

    def compute_predictions(self, params: List[Dict[str, tensor.array]], inputs: tensor.array) -> tensor.array:
        a: tensor.array = inputs
        for i, layer in enumerate(self.layers):
            z: tensor.array = tensor.dot(params[i].get("w"), a) + params[i].get("b")
            a = layer.activation(z)
        return a

    def compute_cost(self, params: List[Dict[str, tensor.array]], inputs: tensor.array, outputs: tensor.array) -> float:
        predictions: tensor.array = self.compute_predictions(params, inputs)
        return self.loss_fn(predictions, outputs)

    def compute_accuracy(self, predictions: tensor.array, outputs: tensor.array) -> float:
        return self.accuracy_fn(predictions, outputs)

    def evaluate(self, inputs: tensor.array, outputs: tensor.array) -> float:
        predictions = self.compute_predictions(self.network_params, inputs)
        return self.compute_accuracy(predictions, outputs)


class SGD(Optimizer):

    def train(self, inputs: tensor.array, outputs: tensor.array) -> None:
        super().init_network_params(inputs.shape)
        grad_fn: Callable = jit(grad(self.compute_cost))
        for _ in tqdm(range(self.epochs), desc="Training the model"):
            grads: List = grad_fn(self.network_params, inputs, outputs)
            for i, layer_params in enumerate(grads):
                self.network_params[i]["w"] -= self.lr * layer_params.get("w")
                self.network_params[i]["b"] -= self.lr * layer_params.get("b")
            self.cost.append(self.compute_cost(self.network_params, inputs, outputs))
            self.accuracy.append(self.evaluate(inputs, outputs))