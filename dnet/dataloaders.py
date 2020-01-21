from typing import Generator

import jax.numpy as tensor
from numpy.random import RandomState


class BatchLoader:

    def __init__(self, inputs: tensor.array, targets: tensor.array, batch_size: int) -> None:
        self.random_generator: RandomState = RandomState(0)
        self.inputs: tensor.array = inputs
        self.targets: tensor.array = targets
        self.batch_size: int = batch_size
        self.perform_batching()

    def perform_batching(self) -> None:
        num_complete_batches, leftover = divmod(self.inputs.shape[0], self.batch_size)
        self.num_batches: int = num_complete_batches + bool(leftover)

    def load_batch(self) -> Generator:
        while True:
            permutation: tensor.array = tensor.array(self.random_generator.permutation(self.inputs.shape[0]))
            for i in range(self.num_batches):
                index: int = permutation[i * self.batch_size: (i + 1) * self.batch_size]
                yield self.inputs[index], self.targets[index]
