from typing import List, Callable

import jax.numpy as tensor

from dnet.layers import FC
from dnet.optimizers import Optimizer, SGD, Momentum


class Trainer:

    def __init__(self, layers: List[FC], loss: Callable, optimizer: str, accuracy: Callable, epochs: int, lr: float,
                 bs: int):
        self.optimizer: Optimizer
        if optimizer == "sgd":
            self.optimizer = SGD(layers, loss, accuracy, epochs, lr, bs)
        elif optimizer == "momentum":
            self.optimizer = Momentum(layers, loss, accuracy, epochs, lr, bs)

    def train(self, inputs: tensor.array, outputs: tensor.array):
        self.optimizer.train(inputs, outputs)

    def evaluate(self, inputs: tensor.array, outputs: tensor.array):
        return self.optimizer.evaluate(inputs, outputs)

    def get_cost(self) -> List[float]:
        return self.optimizer.cost

    def get_accuracy(self) -> List[float]:
        return self.optimizer.accuracy
