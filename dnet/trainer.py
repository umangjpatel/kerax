from jax import jit, grad, value_and_grad
from functools import partial
from typing import Callable, List

from dnet.optimizers import Optimizer
from dnet.tensor import Tensor

"""
1) Parameters handling
2) Network forward pass
3) Optimizer step
4)
"""


class Trainer:

    def compile(self, loss: Callable, optimizer: Optimizer):
        self._loss: Callable = loss
        self._opt_init, self._opt_update, self._get_params = optimizer

    def init_network(self, layers: List):
        from jax.experimental.stax import serial
        self._set_params, self._forward_pass = serial(*layers)

    def init_params(self, input_shape: List[int], seed: int = 0):
        from jax.random import PRNGKey
        rng = PRNGKey(seed)
        del input_shape[0]
        input_shape.insert(0, -1)
        input_shape = tuple(input_shape)
        self.output_shape, self._net_params = self._set_params(rng=rng, input_shape=input_shape)

    def begin_training(self, epochs: int, inputs: Tensor, targets: Tensor):
        opt_state = self._opt_init(self._net_params)
        for i in range(epochs):
            opt_state = self._train(i, opt_state, inputs, targets)
        self.trained_params = self._get_params(opt_state)

    @partial(jit, static_argnums=(0,))
    def _train(self, i, opt_state, inputs, targets):
        params = self._get_params(opt_state)
        grads = grad(self._loss)(params, self._forward_pass, inputs, targets)
        return self._opt_update(i, grads, opt_state)