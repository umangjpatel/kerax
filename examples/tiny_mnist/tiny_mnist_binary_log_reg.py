from dnet.data import Dataloader
from dnet.layers import Dense, Relu, Sigmoid
from dnet.losses import BCELoss
from dnet.models import Module
from dnet.optimizers import SGD


def load_dataset(batch_size: int) -> Dataloader:
    import numpy as np
    import pandas as pd
    from pathlib import Path

    def compute_dataset_info(x, y):
        num_examples = x.shape[0]
        num_complete_batches, leftover = divmod(num_examples, batch_size)
        num_batches = num_complete_batches + bool(leftover)
        return num_examples, num_batches

    def data_stream(x, y, data_info):
        num_examples, num_batches = data_info
        rng = np.random.RandomState(0)
        while True:
            perm = rng.permutation(num_examples)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield x[batch_idx], y[batch_idx]

    def load():
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


data = load_dataset(batch_size=200)
model = Module([Dense(100), Relu, Dense(1), Sigmoid])
model.compile(loss=BCELoss, optimizer=SGD(step_size=0.003))
model.fit(data=data, epochs=10)
model.save(file_name="log_reg")
interp = model.get_interpretation()
interp.plot_losses()

new_model = Module()
new_model.load(file_name="log_reg")
# model already compiled when loaded from serialized file
new_model.fit(data=data, epochs=100)
new_model.save(file_name="log_reg_v2")
interp = new_model.get_interpretation()
interp.plot_losses()
