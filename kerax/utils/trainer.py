import itertools
from functools import partial
from typing import Tuple, List, Optional, Dict, Any

from tqdm import tqdm

from . import Tensor, jit, grad, device_put, jnp, random, stax
from ..data import Dataloader
from ..optimizers import OptimizerState


class Trainer:
    """
    Trainer utility class for training the models.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Trainer class with particular configuration.
        :param config: Configuration containing the optimizer function and the layers.
        """
        self.config: Dict[str, Any] = config
        self.mode = "train"
        self.opt_init, self.opt_update, self.fetch_params = config.get("_optimizer")
        self.setup_params, self.forward_pass = stax.serial(*config.get("_layers"))

    def initialize_params(self, input_shape: List[int]):
        """
        Initializes the network parameters.
        If already trained, then it will return the trained parameters.
        :param input_shape: Shape of the inputs for properly initializing the parms.
        :return: the network parameters.
        """
        trained_params: List[Optional[Tuple[Tensor, Tensor]]] = self.config.get("_trained_params")
        if len(trained_params) > 0:
            return trained_params
        else:
            rng = random.PRNGKey(self.config.get("_seed"))
            input_shape[0] = -1
            input_shape = tuple(input_shape)
            _, params = self.setup_params(rng=rng, input_shape=input_shape)
            return params

    def train(self, data: Dataloader):
        """
        Trains the network
        :param data: Dataloader object containing the dataset.
        :return: the configuration of the training network in the form of dictionary.
        """
        network_params = self.initialize_params(list(data.input_shape))
        opt_state: OptimizerState = self.opt_init(network_params)
        iter_count = itertools.count()
        progress_bar = tqdm(iterable=range(self.config.get("_epochs")),
                            desc="Training model", leave=True)
        for epoch in progress_bar:
            progress_bar.set_description(desc=f"Epoch {epoch + 1}")
            self.mode = "train"
            for _ in range(data.num_train_batches):
                train_batch = device_put(next(data.train_data))
                opt_state = self.step(next(iter_count), opt_state, train_batch)
                network_params = self.fetch_params(opt_state)
                self.calculate_metrics(network_params, train_batch)
            network_params = self.fetch_params(opt_state)
            self.mode = "valid"
            for _ in range(data.num_val_batches):
                valid_batch = device_put(next(data.val_data))
                self.calculate_metrics(network_params, valid_batch)
            self.calculate_epoch_losses(data)
            progress_bar.set_postfix_str(self.pretty_print_metrics())
            progress_bar.refresh()
        self.config["_trained_params"] = self.fetch_params(opt_state)
        return self.config

    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch):
        """
        Training step for the optimization process.
        :param i: Iteration count
        :param opt_state: State of the optimizer
        :param batch: Batch of data for the optimizer to work with.
        :return: the updates state of the optimizer.
        """
        params = self.fetch_params(opt_state)
        grads = grad(self.compute_loss)(params, batch)
        return self.opt_update(i, grads, opt_state)

    @partial(jit, static_argnums=(0,))
    def compute_loss(self, params, batch):
        """
        Helper function to compute forward pass as well as the loss at every step in the training process.
        :param params: Network parameters
        :param batch: Batch of data to compute predictions and loss value
        :return: the computed loss value
        """
        inputs, targets = batch
        predictions = self.forward_pass(params, inputs, mode=self.mode)
        return jit(self.config.get("_loss_fn"))(predictions, targets)

    def calculate_metrics(self, params, batch):
        """
        Helper function that computes the metrics at every step in the training process.
        :param params: Network parameters
        :param batch: Batch of data to compute the metrics.
        """
        inputs, targets = batch
        predictions = self.forward_pass(params, inputs, mode=self.mode)
        self.config.get("_metrics")["loss"][self.mode].append(self.compute_loss(params, batch))
        for metric_fn in self.config.get("_metrics_fn"):
            self.config.get("_metrics")[metric_fn.__name__][self.mode].append(jit(metric_fn)(predictions, targets))

    def calculate_epoch_losses(self, data: Dataloader):
        """
        Caluclates the loss values (both training and validation) after every epoch of the training process.
        :param data: Dataloader object (used to fetch the number of batches)
        """
        self.config.get("_metrics")["loss_per_epoch"]["train"].append(
            jnp.mean(jnp.array(self.config.get("_metrics")["loss"]["train"][-data.num_train_batches:]))
        )
        self.config.get("_metrics")["loss_per_epoch"]["valid"].append(
            jnp.mean(jnp.array(self.config.get("_metrics")["loss"]["valid"][-data.num_val_batches:]))
        )

    def pretty_print_metrics(self) -> str:
        """
        Helper function to display the results (loss + metrics) during the training process
        :return: a string containing the values of the results.
        """
        return " :: ".join([f"{metric_type}_{metric_name} : {metric.get(metric_type)[-1]:.3f}"
                            for metric_name, metric in self.config.get("_metrics").items()
                            for metric_type in metric.keys() if "epoch" not in metric_name])
