from typing import List, Callable

from dnet.interpreter import Interpreter
from dnet.optimizers import Optimizer
from dnet.utils import serialization
from dnet.utils.tensor import Tensor
from dnet.utils.trainer import Trainer


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

    def save(self, fname: str):
        serialization.save_model(fname, layers=self.layers,
                                 loss=self._trainer._loss,
                                 optimizer=self._trainer._optimizer,
                                 params=self._trainer.trained_params)


# TODO : Decouple self.data reference so that model can directly be loaded and inferenced...
# TODO Solution : We can use dill to serialize the pytree data structures (of flattened and type info)
# TODO Then, we use the msgpack to load and unload the params.

"""
 For retraining the model, we need
 1) Layers (List for callables) -> Done
 2) Loss function (can be simply dilled) -> Done
 3) Optimizer to be dilled (opt_init, opr_update, get_params can be extracted easily) -> Done
 4) Trained params (can be serialized using Flax to msgpack conversion APIs)
  -> These params can be replaced when loading it
"""
