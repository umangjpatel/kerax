from dnet.utils.tensor import Tensor
from typing import List, Callable
from dnet.optimizers import Optimizer
from dnet.utils.trainer import Trainer
from dnet.interpreter import Interpreter
from dnet.utils.serialization import to_bytes, from_bytes


class Module:

    def __init__(self, layers: List[Callable]):
        self.layers = layers

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
        self._trainer: Trainer = Trainer()
        self._trainer.compile(loss=loss, optimizer=optimizer)

    def fit(self, inputs: Tensor, targets: Tensor, epochs: int = 1, seed: int = 0):
        self.epochs = epochs
        self._trainer.init_network(self.layers)
        self._trainer.init_params(input_shape=list(inputs.shape), seed=seed)
        self._trainer.begin_training(epochs=epochs, inputs=inputs, targets=targets)

    def get_interpretation(self) -> Interpreter:
        return Interpreter(epochs=self.epochs, losses=self._trainer.losses)

    def save(self, file_name: str):
        self.data: dict = {
            "params": self._trainer.trained_params,
        }
        with open(file_name + "_params.msgpack", "wb") as saved_file:
            serialized_data = to_bytes(self.data)
            saved_file.write(serialized_data)
        self.load(file_name)
        print("Data serialized into msgpack format")

    def load(self, file_name: str):
        with open(file_name + "_params.msgpack", "rb") as loaded_file:
            data = loaded_file.read()
            # TODO : Decouple self.data reference so that model can directly be loaded and inferenced...
            deserialized_data = from_bytes(target=self.data, encoded_bytes=data)
        print("Data deserialized from msgpack format")
        # deserialized_data["params"] consist of python array instead of DeviceArray
