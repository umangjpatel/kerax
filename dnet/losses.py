from torch import Tensor
import torch


def binary_crossentropy(outputs: Tensor, targets: Tensor) -> Tensor:
    return -torch.mean(targets * torch.log(outputs) + (1.0 - targets) * torch.log(1.0 - outputs))
