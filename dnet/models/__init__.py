from dnet.tensor import Tensor
from typing import Tuple, List, Callable
from dnet.optimizers import Optimizer


class Sequential:

    def __init__(self, layers):
        self.layers = layers

    def compile(self, loss: Callable, optimizer: Optimizer):
        self._loss = loss
        self._opt_init, self._opt_update, self._get_params = optimizer
        print("Compiled the network")

    def fit(self, inputs: Tensor, targets: Tensor, seed: int = 0):
        self.seed = seed
        self._init_network()
        self._init_params(input_shape=list(inputs.shape))

    def _init_network(self):
        from jax.experimental import stax
        self._set_params, self._forward_pass = stax.serial(*self.layers)
        print("Finished setting up the network...")

    def _init_params(self, input_shape: List[int]):
        from jax.random import PRNGKey
        rng = PRNGKey(self.seed)
        del input_shape[0]
        input_shape.insert(0, -1)
        input_shape = tuple(input_shape)
        self.output_shape, self._net_params = self._set_params(rng=rng, input_shape=input_shape)
        print(self.output_shape, self._net_params[0][0].shape)
        print("Finished initializing network params...")
