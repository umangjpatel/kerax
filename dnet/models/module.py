from typing import Callable, Tuple

from ..optimizers import Optimizer
from ..utils import convert_to_tensor, serialization
from ..utils.interpreter import Interpreter
from ..utils.tensor import Tensor
from ..utils.trainer import Trainer


class Module:

    def __init__(self, layers=None):
        self._layers: list = [] if layers is None else layers
        self._epochs = None
        self._metrics = None
        self._trained_params = None
        self._loss_fn = None
        self._optimizer = None
        self._metrics = {"train_loss": [], "val_loss": []}
        self._seed = None

    def __add__(self, other):
        assert type(other) == Module, "Type is not 'Module'"
        layers = self._layers + other._layers
        return Module(layers=layers)

    def add(self, other):
        if isinstance(other, Module) and len(other._layers) > 0:
            self._layers += other._layers
        elif isinstance(other, list) and len(other) > 0:
            self._layers += other
        else:
            raise Exception("Operation not allowed")

    def compile(self, loss: Callable, optimizer: Optimizer):
        self._loss_fn = loss
        self._optimizer = optimizer

    def fit(self, inputs: Tensor, targets: Tensor, validation_data: Tuple[Tensor, Tensor], epochs: int = 1,
            seed: int = 0):
        self._epochs = epochs
        self._seed = seed
        self.__dict__ = Trainer(self.__dict__).train((inputs, targets), validation_data)

    def get_interpretation(self) -> Interpreter:
        return Interpreter(epochs=self._epochs, metrics=self._metrics)

    def save(self, file_name: str):
        serialization.save_module(file_name, layers=self._layers,
                                  loss=self._loss_fn,
                                  optimizer=self._optimizer,
                                  params=self._trained_params)

    def load(self, file_name: str):
        deserialized_config = serialization.load_module(file_name)
        self._layers = deserialized_config.get("layers")
        self.compile(loss=deserialized_config.get("loss"),
                     optimizer=deserialized_config.get("optimizer"))
        self._trained_params = convert_to_tensor(deserialized_config.get("params"))
