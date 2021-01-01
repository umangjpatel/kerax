import itertools
from functools import partial
from typing import Tuple, List, Optional, Dict, Any

from jax import jit, grad, device_put
from jax.experimental.stax import serial
from jax.random import PRNGKey
from tqdm import tqdm

from dnet.data import Dataloader
from dnet.optimizers import OptimizerState
from dnet.utils.tensor import Tensor


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

    def train(self, data: Dataloader):
        network_params = self.initialize_params(list(data.input_shape))
        opt_state: OptimizerState = self.opt_init(network_params)
        iter_count = itertools.count()
        progress_bar = tqdm(iterable=range(self.config.get("_epochs")),
                            desc="Training model", leave=True)
        for epoch in progress_bar:
            train_loss, valid_loss = 0.0, 0.0
            progress_bar.set_description(desc=f"Epoch {epoch + 1}")
            for _ in range(data.num_train_batches):
                train_batch = device_put(next(data.train_data))
                opt_state = self.step(next(iter_count), opt_state, train_batch)
                network_params = self.fetch_params(opt_state)
                train_loss += self.compute_loss(network_params, train_batch)
            for _ in range(data.num_val_batches):
                valid_batch = device_put(next(data.val_data))
                valid_loss += self.compute_loss(network_params, valid_batch)
            train_loss /= data.num_train_batches
            valid_loss /= data.num_val_batches
            progress_bar.set_postfix_str(self.compute_metrics(train_loss, valid_loss))
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

    def compute_metrics(self, train_loss, valid_loss) -> str:
        self.config.get("_metrics")["train_loss"].append(train_loss)
        self.config.get("_metrics")["val_loss"].append(valid_loss)
        log_message: str = ""
        for metric in self.config.get("_metrics"):
            log_message += f' {metric} : {self.config.get("_metrics").get(metric)[-1]} ::'
        return log_message
