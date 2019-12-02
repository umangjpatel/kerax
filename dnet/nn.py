from typing import List, Callable

import jax.numpy as tensor
import matplotlib.pyplot as plt
from jax import jit

from dnet import losses, evaluators
from dnet.layers import FC
from dnet.trainer import Trainer


class Model:
    pass


class Sequential(Model):

    def __init__(self) -> None:
        self.layers: List[FC] = []

    def add(self, layer: FC) -> None:
        self.layers.append(layer)

    def compile(self, loss: str, optimizer: str, epochs: int, lr: float = 1e-3) -> None:
        self.loss_fn: Callable = jit(getattr(losses, loss))
        self.optimizer: str = optimizer
        self.accuracy_fn: Callable = jit(getattr(evaluators, loss))
        self.lr: float = lr
        self.epochs: int = epochs

    def fit(self, inputs: tensor.array, outputs: tensor.array) -> None:
        self.trainer: Trainer = Trainer(self.layers, self.loss_fn, self.optimizer, self.accuracy_fn, self.epochs,
                                        self.lr)
        self.trainer.train(inputs, outputs)

    def evaluate(self, inputs: tensor.array, outputs: tensor.array) -> float:
        return self.trainer.evaluate(inputs, outputs)

    def plot_losses(self) -> None:
        plt.plot(range(self.epochs), self.trainer.get_cost(), color="red")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.show()

    def plot_accuracy(self) -> None:
        plt.plot(range(self.epochs), self.trainer.get_accuracy(), color="green")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.show()

    def plot_curves(self) -> None:
        fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 7), sharex=True)
        fig.suptitle("Training Curves")
        ax1.plot(range(self.epochs), self.trainer.get_cost(), "tab:red")
        ax1.set_title("Loss Curve")
        ax2.plot(range(self.epochs), self.trainer.get_accuracy(), "tab:green")
        ax2.set_title("Accuracy Curve")
        plt.show()
