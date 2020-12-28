from typing import Callable, Tuple, List, Dict, Optional

from dnet.optimizers import Optimizer
from dnet.data import Dataloader
from dnet.utils import convert_to_tensor, serialization
from dnet.utils.interpreter import Interpreter
from dnet.utils.tensor import Tensor
from dnet.utils.trainer import Trainer


class Module:

    def __init__(self, layers=None):
        self._layers: List[Tuple[Callable, Callable]] = [] if layers is None else layers
        self._epochs: int = 1
        self._trained_params: List[Optional[Tuple[Tensor, Tensor]]] = []
        self._loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None
        self._optimizer: Optional[Optimizer] = None
        self._metrics: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self._seed: int = 0

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

    def compile(self, loss: Callable[[Tensor, Tensor], Tensor], optimizer: Optimizer):
        self._loss_fn = loss
        self._optimizer = optimizer

    def fit(self, data: Dataloader, epochs: int, seed: int = 0):
        assert epochs > 0, "Number of epochs must be greater than 0"
        self._epochs = epochs
        self._seed = seed
        self.__dict__ = Trainer(self.__dict__).train(data)

    def predict(self, inputs: Tensor):
        assert self._trained_params, "Module not yet trained / trained params not found"
        from jax.experimental.stax import serial
        _, forward_pass = serial(*self._layers)
        return forward_pass(self._trained_params, inputs, mode="predict")

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
