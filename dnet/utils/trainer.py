from functools import partial
from typing import Callable, List

from jax import jit, grad
from jax.experimental.stax import serial
from tqdm import tqdm

from .tensor import Tensor
from ..optimizers import Optimizer
from ..optimizers import OptimizerState


class Trainer:

    def __init__(self):
        self.trained_params: List = []

    def compile(self, loss: Callable, optimizer: Optimizer):
        self._loss: Callable = loss
        self._optimizer = optimizer
        self._opt_init, self._opt_update, self._get_params = optimizer

    def init_network(self, layers: List):
        self._set_params, self._forward_pass = serial(*layers)

    def init_params(self, input_shape: List[int], seed: int = 0):
        from jax.random import PRNGKey
        rng = PRNGKey(seed)
        del input_shape[0]
        input_shape.insert(0, -1)
        input_shape = tuple(input_shape)
        self.output_shape, self._net_params = self._set_params(rng=rng, input_shape=input_shape)
        if len(self.trained_params) != 0:
            self._net_params = self.trained_params

    def begin_training(self, epochs: int, inputs: Tensor, targets: Tensor):
        self.losses: List[float] = []
        opt_state: OptimizerState = self._opt_init(self._net_params)
        progress_bar = tqdm(iterable=range(epochs), desc="Training model", leave=True)
        for i in progress_bar:
            opt_state = self._train(i, opt_state, inputs, targets)
            params = self._get_params(opt_state)
            loss: Tensor = self.compute(params, self._forward_pass, inputs, targets)
            self.losses.append(loss.item())
            progress_bar.set_postfix_str(f"Loss : {loss}")
            progress_bar.refresh()
        self.trained_params = self._get_params(opt_state)

    @partial(jit, static_argnums=(0,))
    def _train(self, i, opt_state, inputs, targets):
        params = self._get_params(opt_state)
        grads = grad(self.compute)(params, self._forward_pass, inputs, targets)
        return self._opt_update(i, grads, opt_state)

    def compute(self, params: List, net_apply: Callable, inputs: Tensor, targets: Tensor):
        predictions: Tensor = net_apply(params, inputs)
        return jit(self._loss)(predictions=predictions, targets=targets)
