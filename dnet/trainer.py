from typing import List, Dict, Callable

import jax.numpy as tensor
from jax import jit, value_and_grad
from tqdm import tqdm


class Trainer:

    def __init__(self, model: Dict):
        for k, v in model.items():
            self.__dict__[k] = v
        self.training_cost: List[float] = []
        self.validation_cost: List[float] = []

    def get_parameters(self) -> List[Dict[str, tensor.array]]:
        return [{"w": layer.weights, "b": layer.bias} for layer in self.layers]

    def compute_cost(self, params: List[Dict[str, tensor.array]], inputs: tensor.array, targets: tensor.array) -> float:
        outputs: tensor.array = self.compute_predictions(params, inputs)
        return self.loss(outputs, targets)

    def compute_predictions(self, params: List[Dict[str, tensor.array]], inputs: tensor.array) -> tensor.array:
        for param, layer in zip(params, self.layers):
            inputs = layer.forward(param, inputs)
        return inputs

    def update_weights(self, grads: List[Dict[str, tensor.array]]) -> None:
        for grad, layer in zip(grads, self.layers):
            layer.update_weights(grad, self.lr)

    def train(self):
        parameters: List[Dict[str, tensor.array]]
        grad_fn: Callable = jit(value_and_grad(self.compute_cost))
        for _ in tqdm(range(self.epochs), desc="Training your model"):
            parameters = self.get_parameters()
            loss, grads = grad_fn(parameters, self.inputs, self.targets)
            self.training_cost.append(loss)
            self.validation_cost.append(self.compute_cost(parameters, self.val_inputs, self.val_targets))
            self.update_weights(grads)
