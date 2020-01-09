from typing import List, Callable, Tuple, Optional

import jax.numpy as tensor
import matplotlib.pyplot as plt

from dnet import losses
from dnet.layers import Layer
from dnet.trainer import Trainer


class Model:
    pass


class Sequential(Model):

    def __init__(self) -> None:
        self.layers: List[Layer] = []

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    def compile(self, loss: str, optimizer: str, lr: float = 3e-02, bs: Optional[int] = None) -> None:
        self.loss: Callable[[tensor.array], float] = getattr(losses, loss)
        self.optimizer: str = optimizer
        self.lr: float = lr
        self.bs: Optional[int] = bs

    def fit(self, inputs: tensor.array, targets: tensor.array, epochs: int,
            validation_data: Tuple[tensor.array, tensor.array]):
        self.inputs: tensor.array = inputs
        self.targets: tensor.array = targets
        self.val_inputs, self.val_targets = validation_data
        self.epochs = epochs
        self.trainer = Trainer(self.__dict__)
        self.trainer.train()

    def plot_losses(self):
        plt.plot(range(self.epochs), self.trainer.training_cost, color="red", label="Training")
        plt.plot(range(self.epochs), self.trainer.validation_cost, color="green", label="Validation")
        plt.title("Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
