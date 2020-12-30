import math


class Dataloader:

    def __init__(self, train_data, val_data, input_shape, batch_size, info):
        self.train_data = train_data
        self.val_data = val_data
        self.input_shape = input_shape
        self.batch_size = batch_size
        if info is not None:
            self.num_train_batches = math.ceil(info.splits["train"].num_examples / batch_size)
            self.num_val_batches = math.ceil(info.splits["test"].num_examples / batch_size)
        else:
            self.num_train_batches = math.ceil(sum(1 for _ in train_data) / batch_size)
            self.num_val_batches = math.ceil(sum(1 for _ in val_data) / batch_size)
