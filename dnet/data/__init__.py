from typing import Tuple

from dnet.utils.tensor import Tensor


class Dataloader:

    def __init__(self, train_data: Tuple[Tensor, Tensor], val_data: Tuple[Tensor, Tensor], batch_size: int = 32):
        self.batch_size = batch_size
        self.train_data = train_data
        self.val_data = val_data
        self.input_shape = train_data[0].shape
        self._num_train = self.input_shape[0]
        num_complete_batches, leftover = divmod(self._num_train, batch_size)
        self.num_batches = num_complete_batches + bool(leftover)
        self.batch_stream = self.data_stream()

    def data_stream(self):
        import numpy.random as npr
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(self._num_train)
            for i in range(self.num_batches):
                batch_idx = perm[i * self.batch_size:(i + 1) * self.batch_size]
                yield self.train_data[0][batch_idx], self.train_data[1][batch_idx]

