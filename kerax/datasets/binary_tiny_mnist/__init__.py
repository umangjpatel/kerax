from ...data import Dataloader as __Dataloader__
from typing import Tuple


def load_dataset(batch_size: int) -> __Dataloader__:
    """
    Loads the reduced version of the MNIST dataset where it contains only data of digits 0s and 1s.
    :param batch_size: Number of examples to be included in a single batch.
    :return: Dataloader object consisting of training and validation data.
    """
    import numpy as np
    from ...data import Dataloader
    import pandas as pd
    from pathlib import Path

    def compute_dataset_info(x, y) -> Tuple[int, ...]:
        """
        Computes the number of examples and number of batches for a given dataset
        :param x: Inputs
        :param y: Targets
        :return: a tuple consisting of total number of examples and number of batches.
        """
        assert x.shape[0] == y.shape[0], "Number of examples do not match..."
        num_examples = x.shape[0]
        num_complete_batches, leftover = divmod(num_examples, batch_size)
        num_batches = num_complete_batches + bool(leftover)
        return num_examples, num_batches

    def data_stream(x, y, data_info):
        """
        Yields an iterator for streaming the dataset into respective batches efficiently.
        :param x: Inputs
        :param y: Targets
        :param data_info: a tuple consisting of total number of examples and number of batches.
        :return: a tuple consisting of a batch of inputs and targets for a dataset.
        """
        num_examples, num_batches = data_info
        rng = np.random.RandomState(0)
        while True:
            perm = rng.permutation(num_examples)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield x[batch_idx], y[batch_idx]

    def load() -> __Dataloader__:
        """
        Loads the dataset
        :return: a Dataloader object consisting of the training and validation data.
        """
        path: Path = Path(__file__).parent
        dataset: pd.DataFrame = pd.read_csv(path / "train.csv", header=None)

        # 80% training, 20% validation
        train_data: pd.DataFrame = dataset.sample(frac=0.8, random_state=0)
        val_data: pd.DataFrame = dataset.drop(train_data.index)

        train_labels: np.ndarray = np.expand_dims(train_data[0].values, axis=1)
        train_images: np.ndarray = train_data.iloc[:, 1:].values / 255.0

        val_labels: np.ndarray = np.expand_dims(val_data[0].values, axis=1)
        val_images: np.ndarray = val_data.iloc[:, 1:].values / 255.0

        train_data_info = compute_dataset_info(train_images, train_labels)
        val_data_info = compute_dataset_info(val_images, val_labels)

        train_gen = data_stream(train_images, train_labels, train_data_info)
        val_gen = data_stream(val_images, val_labels, val_data_info)

        input_shape = tuple([-1] + list(train_images.shape)[1:])

        return Dataloader(train_data=train_gen, val_data=val_gen,
                          batch_size=batch_size, input_shape=input_shape,
                          num_train_batches=train_data_info[1],
                          num_val_batches=val_data_info[1])

    return load()
