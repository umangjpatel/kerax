from torch import Tensor
import matplotlib.pyplot as plt
from dnet.layers import Layer, FC
from dnet import losses
from dnet.trainer import Trainer
from typing import List, Tuple, Callable, Optional


class Model:

    def add(self, layer: Layer) -> None:
        raise NotImplementedError

    def summary(self) -> None:
        raise NotImplementedError

    def compile(self, optimizer: str, loss: str, lr: float) -> None:
        raise NotImplementedError

    def fit(self, inputs: Tensor, targets: Tensor, epochs: int, validation_data: Tuple[Tensor, Tensor]) -> None:
        raise NotImplementedError

    def plot_losses(self):
        raise NotImplementedError


class Sequential(Model):

    def __init__(self) -> None:
        self.layers: List[FC] = []

    def add(self, layer: FC) -> None:
        self.layers.append(layer)

    def summary(self) -> None:
        for i, layer in enumerate(self.layers):
            print(f"Layer {i + 1} : units -> {layer.units}, activation -> {layer.activation_name}")

    def compile(self, optimizer: str, loss: str, lr: float = 1e-02) -> None:
        self.optimizer: str = optimizer
        self.loss: Callable = getattr(losses, loss)
        self.lr: float = lr

    def fit(self, inputs: Tensor, targets: Tensor, epochs: int,
            validation_data: Tuple[Tensor, Tensor] = None) -> None:
        self.epochs: int = epochs
        self.inputs: Tensor = inputs
        self.targets: Tensor = targets
        self.validation_data: Optional[Tuple[Tensor, Tensor]] = validation_data
        self.trainer: Trainer = Trainer(self.__dict__)
        self.trainer.train()

    def plot_losses(self) -> None:
        plt.plot(range(self.epochs), self.trainer.training_cost, color="red", label="Training")
        plt.plot(range(self.epochs), self.trainer.validation_cost, color="green", label="Validation")
        plt.title("Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_accuracy(self) -> None:
        raise NotImplementedError
