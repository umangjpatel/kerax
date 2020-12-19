from dnet.tensor import Tensor
from typing import Tuple, List, Callable
from dnet.optimizers import Optimizer
from dnet.trainer import Trainer


class Sequential:

    def __init__(self, layers):
        self.layers = layers

    def compile(self, loss: Callable, optimizer: Optimizer):
        self._trainer = Trainer()
        self._trainer.compile(loss=loss, optimizer=optimizer)

    def fit(self, inputs: Tensor, targets: Tensor, epochs: int = 1, seed: int = 0):
        self._trainer.init_network(self.layers)
        self._trainer.init_params(input_shape=list(inputs.shape), seed=seed)
        self._trainer.begin_training(epochs=epochs, inputs=inputs, targets=targets)
