from collections import defaultdict
from typing import Callable, Tuple, List, Dict, Optional

from ..data import Dataloader
from ..utils import Interpreter, Tensor, Trainer, convert_to_tensor, serialization, stax


class Sequential:
    """
    Sequential / serial model.
    """

    def __init__(self, layers=None):
        """
        Initializes the model with layers
        :param layers: List of layers for the neural network
        """
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
        """
        Overriden method to support addition of a model's layers with another ones.
        :param other: a Sequential model consisting of some layers.
        :return: a Sequential model with the combined layers.
        """
        assert type(other) == Sequential, "Type is not 'Sequential'"
        assert other._layers, "Layers not provided"
        assert len(other._layers) > 0, "No layers found"
        layers = self._layers + other._layers
        return Sequential(layers=layers)

    def add(self, other):
        """
        Another helper method for adding layers into the model.
        :param other: Either a Sequential model or list of layers or a layer
        :return: None if the instance is not as expected in the API.
        """
        if isinstance(other, Sequential) and len(other._layers) > 0:
            self._layers += other._layers
        elif isinstance(other, list) and len(other) > 0:
            self._layers += other
        else:
            return None

    def compile(self, loss: Callable, optimizer: Callable, metrics: List[Callable] = None):
        """
        Compiles the model.
        :param loss: the loss function to be used.
        :param optimizer: the optimizer to be used.
        :param metrics: the metrics to be used.
        """
        self._loss_fn = loss
        self._optimizer = optimizer
        self._metrics_fn = metrics
        for metric_fn in self._metrics_fn:
            self._metrics[metric_fn.__name__] = defaultdict(list)

    def fit(self, data: Dataloader, epochs: int, seed: int = 0):
        """
        Trains the model
        :param data: Dataloader object containing the dataset.
        :param epochs: Number of times the entire dataset is used for training the model.
        :param seed: Seed for randomization.
        """
        assert self._optimizer, "Call .compile() before .fit()"
        assert self._loss_fn, "Call .compile() before .fit()"
        assert epochs > 0, "Number of epochs must be greater than 0"
        self._epochs = epochs
        self._seed = seed
        self.__dict__ = Trainer(self.__dict__).train(data)

    def predict(self, inputs: Tensor):
        """
        Uses the trained model for prediction
        :param inputs: Inputs to be used for prediction
        :return: the outputs computed by the trained model.
        """
        assert self._trained_params, "Module not yet trained / trained params not found"
        _, forward_pass = stax.serial(*self._layers)
        return forward_pass(self._trained_params, inputs, mode="predict")

    def get_interpretation(self) -> Interpreter:
        """
        Fetches the Interpreter object for graphical analysis of the training process.
        :return: the Interpreter object containing relevant information of the training results.
        """
        return Interpreter(epochs=self._epochs, metrics=self._metrics)

    def save(self, file_name: str):
        """
        Saves the model onto the disk.
        By default, it will be saved in the current directory.
        :param file_name: File name of the model to be saved (without the file extension)
        """
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
        """
        Loads the model from the disk.
        By default, it will be loaded from the current directory.
        :param file_name: File name of the model to be loaded (without the file extension)
        """
        deserialized_config = serialization.load_module(file_name)
        self._layers = deserialized_config.get("layers")
        self.compile(loss=deserialized_config.get("loss"),
                     optimizer=deserialized_config.get("optimizer"),
                     metrics=deserialized_config.get("metrics"))
        self._trained_params = convert_to_tensor(deserialized_config.get("params"))
