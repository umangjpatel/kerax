import itertools
from typing import Dict
from typing import Generator, Callable, Iterator, Tuple, List

import jax.numpy as tensor
from jax import grad
from jax import random
from tqdm import tqdm

from dnet.dataloaders import BatchLoader


class Trainer:

    def __init__(self, model: Dict) -> None:
        for k, v in model.items():
            self.__dict__[k] = v
        self.init_params_fun, self.predict_fun = self.serial_model
        self.data_loader: BatchLoader = BatchLoader(self.inputs, self.targets, batch_size=self.bs)
        self.data_batches: Generator = self.data_loader.load_batch()
        self.opt_init, self.opt_update, self.get_params = self.optimizer(self.lr)
        self.init_network()

    def init_network(self) -> None:
        _, self.params = self.init_params_fun(random.PRNGKey(0), (-1, self.inputs.shape[-1]))
        self.opt_state: Callable = self.opt_init(self.params)
        self.count: Iterator[int] = itertools.count()
        self.training_accuracy: List[float] = []
        self.training_cost: List[float] = []
        self.validation_accuracy: List[float] = []
        self.validation_cost: List[float] = []

    def compute_predictions(self, params: List[Tuple[tensor.array, tensor.array]],
                            inputs: tensor.array) -> tensor.array:
        return self.predict_fun(params, inputs)

    def compute_cost(self, params: List[Tuple[tensor.array, tensor.array]],
                     batch: Tuple[tensor.array, tensor.array]) -> float:
        inputs, targets = batch
        outputs: tensor.array = self.compute_predictions(params, inputs)
        return self.loss(outputs, targets)

    def compute_accuracy(self, params: List[Tuple[tensor.array, tensor.array]],
                         batch: Tuple[tensor.array, tensor.array]) -> float:
        inputs, targets = batch
        outputs: tensor.array = self.compute_predictions(params, inputs)
        return self.evaluator(outputs, targets)

    def update(self, i: int, opt_state: Callable, batch: Tuple[tensor.array, tensor.array]) -> Callable:
        params: List[Tuple[tensor.array, tensor.array]] = self.get_params(opt_state)
        return self.opt_update(i, grad(self.compute_cost)(params, batch), opt_state)

    def train(self) -> None:
        epoch_bar: tqdm = tqdm(range(self.epochs))
        for epoch in epoch_bar:
            for _ in range(self.data_loader.num_batches):
                self.opt_state = self.update(next(self.count), self.opt_state, next(self.data_batches))
            params: List[Tuple[tensor.array, tensor.array]] = self.get_params(self.opt_state)
            self.update_metrics(epoch, epoch_bar, params)

    def update_metrics(self, epoch, epoch_bar, params) -> None:
        train_acc: float = self.compute_accuracy(params, (self.inputs, self.targets))
        train_cost: float = self.compute_cost(params, (self.inputs, self.targets))
        val_acc: float = self.compute_accuracy(params, (self.val_inputs, self.val_targets))
        val_cost: float = self.compute_cost(params, (self.val_inputs, self.val_targets))
        epoch_bar.set_description_str(desc=f"Epoch {epoch + 1}")
        epoch_bar.set_postfix_str(s=f"Validation accuracy => {val_acc}")
        self.training_cost.append(train_cost)
        self.training_accuracy.append(train_acc)
        self.validation_cost.append(val_cost)
        self.validation_accuracy.append(val_acc)
