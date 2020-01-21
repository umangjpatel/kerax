from typing import List, Callable, Tuple

import jax.numpy as tensor
import matplotlib.pyplot as plt
from jax.experimental.stax import serial

from dnet import evaluators
from dnet import losses
from dnet import optimizers
from dnet.layers import Layer
from dnet.trainer import Trainer


class Model:
    pass


class Sequential(Model):

    def __init__(self):
        self.layers: List = []

    def add(self, network_layer: Layer) -> None:
        self.layers.extend(network_layer.layer)

    def compile(self, loss: str, optimizer: str, lr: float = 1e-02, bs: int = 32) -> None:
        self.lr: float = lr
        self.bs: int = bs
        self.loss: Callable[[tensor.array, tensor.array], float] = getattr(losses, loss)
        self.optimizer: Callable[[float], Tuple[Callable, ...]] = getattr(optimizers, optimizer)
        self.evaluator: Callable[[tensor.array, tensor.array], float] = getattr(evaluators, loss)
        self.serial_model: serial = serial(*self.layers)

    def fit(self, inputs: tensor.array, targets: tensor.array, epochs: int,
            validation_data: Tuple[tensor.array, tensor.array]) -> None:
        self.epochs: int = epochs
        self.inputs: tensor.array = inputs
        self.targets: tensor.array = targets
        self.val_inputs, self.val_targets = validation_data
        self.trainer: Trainer = Trainer(self.__dict__)
        self.trainer.train()

    def plot_losses(self) -> None:
        plt.plot(range(self.epochs), self.trainer.training_cost, color="red", marker="o", label="Training loss")
        plt.plot(range(self.epochs), self.trainer.validation_cost, color="green", label="Validation loss")
        plt.title("Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_accuracy(self) -> None:
        plt.plot(range(self.epochs), self.trainer.training_accuracy, color="red", marker="o", label="Training accuracy")
        plt.plot(range(self.epochs), self.trainer.validation_accuracy, color="green", label="Validation accuracy")
        plt.title("Accuracy Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
