from functools import partial
from typing import Tuple, List, Optional, Dict, Any

from jax import jit, grad
from jax.experimental.stax import serial
from jax.random import PRNGKey
from tqdm import tqdm

from .tensor import Tensor
from ..optimizers import OptimizerState


class Trainer:

    def __init__(self, config: Dict[str, Any]):
        self.config: Dict[str, Any] = config
        self.opt_init, self.opt_update, self.fetch_params = config.get("_optimizer")
        self.setup_params, self.forward_pass = serial(*config.get("_layers"))

    def initialize_params(self, input_shape: List[int]):
        trained_params: List[Optional[Tuple[Tensor, Tensor]]] = self.config.get("_trained_params")
        if len(trained_params) > 0:
            return trained_params
        else:
            rng = PRNGKey(self.config.get("_seed"))
            input_shape[0] = -1
            input_shape = tuple(input_shape)
            _, params = self.setup_params(rng=rng, input_shape=input_shape)
            return params

    def train(self, batch: Tuple[Tensor, Tensor], validation_data: Tuple[Tensor, Tensor]):
        network_params = self.initialize_params(list(batch[0].shape))
        opt_state: OptimizerState = self.opt_init(network_params)
        progress_bar = tqdm(iterable=range(self.config.get("_epochs")),
                            desc="Training model", leave=True)
        for i in progress_bar:
            opt_state = self.step(i, opt_state, batch)
            params = self.fetch_params(opt_state)
            latest_metric = self.compute_metrics(params, batch, validation_data)
            progress_bar.set_postfix_str(latest_metric)
            progress_bar.refresh()
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
        predictions = self.forward_pass(params, inputs, mode="train")
        return jit(self.config.get("_loss_fn"))(predictions, targets)

    def compute_metrics(self, params, batch: Tuple[Tensor, Tensor],
                        validation_data: Tuple[Tensor, Tensor]) -> str:
        self.config.get("_metrics")["train_loss"].append(self.compute_loss(params, batch).item())
        self.config.get("_metrics")["val_loss"].append(self.compute_loss(params, validation_data).item())
        log_message: str = ""
        for metric in self.config.get("_metrics"):
            log_message += f' {metric} : {self.config.get("_metrics").get(metric)[-1]} ::'
        return log_message
