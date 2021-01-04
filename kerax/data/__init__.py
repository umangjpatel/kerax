class Dataloader:

    def __init__(self, train_data, val_data, input_shape, batch_size, num_train_batches, num_val_batches):
        self.train_data = train_data
        self.val_data = val_data
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_train_batches = num_train_batches
        self.num_val_batches = num_val_batches

        assert self.num_train_batches is not None, "Please add num_train_batches"
        assert self.num_val_batches is not None, "Please add num_val_batches"
