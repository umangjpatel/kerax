from typing import Iterator, Tuple


class Dataloader:

    def __init__(self, train_data: Iterator, val_data: Iterator,
                 input_shape: Tuple[int, ...], batch_size: int,
                 num_train_batches: int, num_val_batches: int):
        assert train_data is not None, "Training data is empty"
        assert val_data is not None, "Validation data is empty"
        assert input_shape is not None, "Input shape not passed"
        assert batch_size > 0, "Invalid batch size passed"
        assert num_train_batches is not None, "Number of training batches is not passed"
        assert num_val_batches is not None, "Number of validation batches is not passed"

        self.train_data = train_data
        self.val_data = val_data
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_train_batches = num_train_batches
        self.num_val_batches = num_val_batches


__all__ = [
    "Dataloader"
]
