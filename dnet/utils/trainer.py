from functools import partial
from typing import Tuple, List

from jax import jit, grad
from jax.experimental.stax import serial
from jax.random import PRNGKey
from tqdm import tqdm

from .tensor import Tensor
from ..optimizers import OptimizerState


class Trainer:

    def __init__(self, config: dict):
        self.config = config
        self.opt_init, self.opt_update, self.fetch_params = config.get("_optimizer")
        self.setup_params, self.forward_pass = serial(*config.get("_layers"))

    def initialize_params(self, input_shape: List[int]):
        rng = PRNGKey(self.config.get("_seed"))
        input_shape[0] = -1
        input_shape = tuple(input_shape)
        _, params = self.setup_params(rng=rng, input_shape=input_shape)
        trained_params = self.config.get("_trained_params")
        if trained_params:
            params = trained_params
        return params

    def train(self, batch: Tuple[Tensor, Tensor], validation_data: Tuple[Tensor, Tensor]):
        losses = []
        network_params = self.initialize_params(list(batch[0].shape))
        opt_state: OptimizerState = self.opt_init(network_params)
        progress_bar = tqdm(iterable=range(self.config.get("_epochs")),
                            desc="Training model", leave=True)
        for i in progress_bar:
            opt_state = self.step(i, opt_state, batch)
            params = self.fetch_params(opt_state)
            losses.append(self.compute_loss(params, batch).item())
            progress_bar.set_postfix_str(f"Loss : {losses[-1]}")
            progress_bar.refresh()
        self.config["_metrics"] = {"losses": losses}
        self.config["_trained_params"] = self.fetch_params(opt_state)
        return self.config

    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch):
        params = self.fetch_params(opt_state)
        grads = grad(self.compute_loss)(params, batch)
        return self.opt_update(i, grads, opt_state)

    @partial(jit, static_argnums=(0,))
    def compute_loss(self, params, batch):
        inputs, targets = batch
        predictions = self.forward_pass(params, inputs)
        return jit(self.config.get("_loss_fn"))(predictions, targets)
