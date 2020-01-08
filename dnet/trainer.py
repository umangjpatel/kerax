from typing import List, Dict

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
            z: tensor.array = tensor.dot(param.get("w"), inputs)
            inputs: tensor.array = layer.activation(z)
        return inputs

    def update_weights(self, params: List[Dict[str, tensor.array]], grads: List[Dict[str, tensor.array]]) -> List[
        Dict[str, tensor.array]]:
        for i, grad in enumerate(grads):
            params[i]["w"] -= self.lr * grad.get("w")
            params[i]["b"] -= self.lr * grad.get("b")
        return params

    def train(self):
        parameters: List[Dict[str, tensor.array]] = self.get_parameters()
        grad_fn = jit(value_and_grad(self.compute_cost))
        for _ in tqdm(range(self.epochs), desc="Training your model"):
            loss, grads = grad_fn(parameters, self.inputs, self.targets)
            self.training_cost.append(loss)
            self.validation_cost.append(self.compute_cost(parameters, self.val_inputs, self.val_targets))
            parameters = self.update_weights(parameters, grads)
