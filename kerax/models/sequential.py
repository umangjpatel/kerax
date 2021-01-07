from collections import defaultdict
from typing import Callable, Tuple, List, Dict, Optional

from ..data import Dataloader
from ..utils import Interpreter, Tensor, Trainer, convert_to_tensor, serialization, stax


class Sequential:

    def __init__(self, layers=None):
        self._layers: List[Tuple[Callable, Callable]] = [] if layers is None else layers
        self._epochs: int = 1
        self._trained_params: List[Optional[Tuple[Tensor, Tensor]]] = []
        self._loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None
        self._optimizer: Optional[Optimizer] = None
        self._metrics: Dict[str, Dict[str, List[float]]] = {"loss": defaultdict(list),
                                                            "loss_per_epoch": defaultdict(list)}
        self._metrics_fn: Optional[List[Callable]] = []
        self._seed: int = 0

    def __add__(self, other):
        assert type(other) == Sequential, "Type is not 'Sequential'"
        layers = self._layers + other._layers
        return Sequential(layers=layers)

    def add(self, other):
        if isinstance(other, Sequential) and len(other._layers) > 0:
            self._layers += other._layers
        elif isinstance(other, list) and len(other) > 0:
            self._layers += other
        else:
            return None

    def compile(self, loss: Callable, optimizer: Callable, metrics: List[Callable] = None):
        self._loss_fn = loss
        self._optimizer = optimizer
        self._metrics_fn = metrics
        for metric_fn in self._metrics_fn:
            self._metrics[metric_fn.__name__] = defaultdict(list)

    def fit(self, data: Dataloader, epochs: int, seed: int = 0):
        assert self._optimizer, "Call .compile() before .fit()"
        assert self._loss_fn, "Call .compile() before .fit()"
        assert epochs > 0, "Number of epochs must be greater than 0"
        self._epochs = epochs
        self._seed = seed
        self.__dict__ = Trainer(self.__dict__).train(data)

    def predict(self, inputs: Tensor):
        assert self._trained_params, "Module not yet trained / trained params not found"
        _, forward_pass = stax.serial(*self._layers)
        return forward_pass(self._trained_params, inputs, mode="predict")

    def get_interpretation(self) -> Interpreter:
        return Interpreter(epochs=self._epochs, metrics=self._metrics)

    def save(self, file_name: str):
        assert self._layers, "Layers not provided"
        assert self._loss_fn, "Loss function not provided"
        assert self._metrics_fn, "Metric functions not provided"
        assert self._optimizer, "Optimizer not provided"
        assert self._trained_params, "Model not trained yet..."
        serialization.save_module(file_name, layers=self._layers,
                                  loss=self._loss_fn,
                                  metrics=self._metrics_fn,
                                  optimizer=self._optimizer,
                                  params=self._trained_params)

    def load(self, file_name: str):
        deserialized_config = serialization.load_module(file_name)
        self._layers = deserialized_config.get("layers")
        self.compile(loss=deserialized_config.get("loss"),
                     optimizer=deserialized_config.get("optimizer"),
                     metrics=deserialized_config.get("metrics"))
        self._trained_params = convert_to_tensor(deserialized_config.get("params"))
