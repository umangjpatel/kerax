from typing import Iterator, NamedTuple

import jax.numpy as tensor
from jax import random

Batch = NamedTuple("Batch", [("inputs", tensor.array), ("outputs", tensor.array)])


class DataIterator:

    def __call__(self, inputs: tensor.array, outputs: tensor.array) -> Iterator[Batch]:
        raise NotImplementedError


class BatchIterator(DataIterator):

    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: tensor.array, outputs: tensor.array) -> Iterator[Batch]:
        key = random.PRNGKey(0)
        starts = tensor.arange(0, len(inputs[-1]), self.batch_size)
        if self.shuffle:
            starts = random.shuffle(key, starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[:, start:end]
            batch_outputs = outputs[:, start:end]
            yield Batch(batch_inputs, batch_outputs)
