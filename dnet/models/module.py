from typing import Callable

from ..optimizers import Optimizer
from ..utils import convert_to_tensor, serialization
from ..utils.interpreter import Interpreter
from ..utils.tensor import Tensor
from ..utils.trainer import Trainer


class Module:

    def __init__(self, layers=None):
        if layers is None:
            layers = []
        self.layers = layers
        self._trainer: Trainer = Trainer()

    def __add__(self, other):
        if isinstance(other, Module):
            layers = self.layers + other.layers
            return Module(layers=layers)
        else:
            raise Exception("Operation not allowed")

    def add(self, other):
        if isinstance(other, Module):
            self.layers += other.layers
        elif isinstance(other, list):
            self.layers += other
        else:
            raise Exception("Operation not allowed")

    def compile(self, loss: Callable, optimizer: Optimizer):
        self._trainer.compile(loss=loss, optimizer=optimizer)

    def fit(self, inputs: Tensor, targets: Tensor, epochs: int = 1, seed: int = 0):
        self.epochs = epochs
        self._trainer.init_network(self.layers)
        self._trainer.init_params(input_shape=list(inputs.shape), seed=seed)
        self._trainer.begin_training(epochs=epochs, inputs=inputs, targets=targets)

    def get_interpretation(self) -> Interpreter:
        return Interpreter(epochs=self.epochs, losses=self._trainer.losses)

    def save(self, file_name: str):
        serialization.save_module(file_name, layers=self.layers,
                                  loss=self._trainer._loss,
                                  optimizer=self._trainer._optimizer,
                                  params=self._trainer.trained_params)

    def load(self, file_name: str):
        deserialized_config = serialization.load_module(file_name)
        self.layers = deserialized_config.get("layers")
        self.compile(loss=deserialized_config.get("loss"),
                     optimizer=deserialized_config.get("optimizer"))
        self._trainer.trained_params = convert_to_tensor(deserialized_config.get("params"))
