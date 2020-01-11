import itertools
from typing import List, Dict, Callable, Generator, Iterator, Tuple

import jax.numpy as tensor
from jax import grad
from tqdm import tqdm

from dnet.dataloaders import DataLoader


class Trainer:

    def __init__(self, model: Dict):
        for k, v in model.items():
            self.__dict__[k] = v
        self.data_loader: DataLoader = DataLoader(self.inputs, self.targets, self.bs)
        self.batches: Generator = self.data_loader.load_batch()
        self.opt_init, self.opt_update, self.get_params = self.optimizer(self.lr)
        self.training_cost: List[float] = []
        self.validation_cost: List[float] = []
        self.training_accuracy: List[float] = []
        self.validation_accuracy: List[float] = []

    def get_weights(self) -> List[Dict[str, tensor.array]]:
        return [{"w": layer.weights, "b": layer.bias} for layer in self.layers]

    def compute_cost(self, params: List[Dict[str, tensor.array]],
                     batch: Tuple[tensor.array, tensor.array]) -> float:
        inputs, targets = batch
        outputs: tensor.array = self.compute_predictions(params, inputs)
        return self.loss(outputs, targets)

    def compute_predictions(self, params: List[Dict[str, tensor.array]], inputs: tensor.array) -> tensor.array:
        for param, layer in zip(params, self.layers):
            inputs = layer.forward(param, inputs)
        return inputs

    def update_params(self, i: int, opt_state: Callable, batch: Tuple[tensor.array, tensor.array]) -> Callable:
        params: List[Dict[str, tensor.array]] = self.get_params(opt_state)
        return self.opt_update(i, grad(self.compute_cost)(params, batch), opt_state)

    def update_layer_weights(self, params: List[Dict[str, tensor.array]]) -> None:
        for param, layer in zip(params, self.layers): layer.update_weights(param)

    def train(self) -> None:
        parameters: List[Dict[str, tensor.array]]
        self.opt_state: Callable = self.opt_init(self.get_weights())
        count: Iterator[int] = itertools.count()
        for epoch in range(self.epochs):
            for _ in tqdm(range(self.data_loader.num_batches), desc=f"Epoch {epoch + 1} : "):
                self.opt_state = self.update_params(next(count), self.opt_state, next(self.batches))
            parameters = self.get_params(self.opt_state)
            self.training_cost.append(self.compute_cost(parameters, (self.inputs, self.targets)))
            self.validation_cost.append(self.compute_cost(parameters, (self.val_inputs, self.val_targets)))
        self.update_layer_weights(parameters)
