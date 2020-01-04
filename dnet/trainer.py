from torch import Tensor
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple


class Trainer:

    def __init__(self, items: Dict):
        for k, v in items.items():
            self.__dict__[k] = v
        self.training_cost: List[float] = []
        self.training_acc: List[float] = []
        self.validation_cost: List[float] = []
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device.upper()} for training...")
        self.init_weights()

    def init_weights(self):
        self.weights: List[Dict[str, Tensor]] = []
        for i, layer in enumerate(self.layers):
            w_shape: Tuple[int, int] = (
                layer.units, self.layers[i - 1].units if i > 0 else self.inputs.shape[0])
            w: Tensor = torch.randn(size=w_shape, dtype=torch.float32, device=self.device) * 0.01
            b: Tensor = torch.zeros(size=(layer.units, 1), dtype=torch.float32, device=self.device)
            self.weights.append({"w": w.requires_grad_(True), "b": b.requires_grad_(True)})

    def compute_predictions(self, inputs: Tensor) -> Tensor:
        a: Tensor = inputs
        for i, layer in enumerate(self.layers):
            w: Tensor = self.weights[i]["w"]
            b: Tensor = self.weights[i]["b"]
            z: Tensor = w @ a + b
            a = layer.activation(z)
        return a

    def update_weights(self) -> None:
        for i in range(len(self.weights)):
            self.weights[i]["w"].sub_(self.lr * self.weights[i]["w"].grad)
            self.weights[i]["b"].sub_(self.lr * self.weights[i]["b"].grad)
            self.weights[i]["w"].grad.zero_()
            self.weights[i]["b"].grad.zero_()

    def train(self) -> None:
        for _ in tqdm(range(self.epochs), desc="Training your model "):
            outputs: Tensor = self.compute_predictions(self.inputs)
            loss: Tensor = self.loss(outputs, self.targets)
            self.training_cost.append(loss.item())
            loss.backward()
            with torch.no_grad():
                self.update_weights()
                if self.validation_data:
                    val_outputs: Tensor = self.compute_predictions(self.validation_data[0])
                    self.validation_cost.append(self.loss(val_outputs, self.validation_data[1]).item())

    def evaluate(self, inputs: Tensor, targets: Tensor) -> None:
        raise NotImplementedError

    def predict(self, inputs: Tensor) -> Tensor:
        with torch.no_grad():
            predictions: Tensor = self.compute_predictions(inputs)
        return predictions
