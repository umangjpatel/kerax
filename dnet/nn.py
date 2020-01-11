from typing import List, Callable, Tuple, Optional

import jax.numpy as tensor
import matplotlib.pyplot as plt

from dnet import evaluators
from dnet import losses
from dnet import optimizers
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
        self.evaluator: Callable[[tensor.array, tensor.array], float] = getattr(evaluators, loss)
        self.lr: float = lr
        self.optimizer: Callable = getattr(optimizers, optimizer)
        self.bs: Optional[int] = bs

    def fit(self, inputs: tensor.array, targets: tensor.array, epochs: int,
            validation_data: Tuple[tensor.array, tensor.array]) -> None:
        self.inputs: tensor.array = inputs
        self.targets: tensor.array = targets
        self.val_inputs, self.val_targets = validation_data
        self.epochs: int = epochs
        self.trainer: Trainer = Trainer(self.__dict__)
        self.trainer.train()

    def predict(self, inputs: tensor.array):
        return self.trainer.compute_predictions(self.trainer.get_weights(), inputs)

    def plot_losses(self) -> None:
        plt.plot(range(self.epochs), self.trainer.training_cost, marker="o", color="red", label="Training")
        plt.plot(range(self.epochs), self.trainer.validation_cost, color="green", label="Validation")
        plt.title("Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_accuracy(self) -> None:
        plt.plot(range(self.epochs), self.trainer.training_accuracy, marker="o", color="red", label="Training")
        plt.plot(range(self.epochs), self.trainer.validation_accuracy, color="green", label="Validation")
        plt.title("Accuracy Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
