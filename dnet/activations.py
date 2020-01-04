from torch import Tensor
import torch


def linear(z: Tensor) -> Tensor:
    return z


def sigmoid(z: Tensor) -> Tensor:
    return torch.sigmoid(z)


def tanh(z: Tensor) -> Tensor:
    return torch.tanh(z)


def relu(z: Tensor) -> Tensor:
    return torch.relu(z)
