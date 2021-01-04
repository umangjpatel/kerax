import asyncio
import math
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds

from kerax.data import Dataloader
from kerax.layers import Flatten, Dense, Relu, LogSoftmax
from kerax.losses import CCELoss
from kerax.metrics import accuracy
from kerax.models import Module
from kerax.optimizers import RMSProp


async def tfds_load_data(batch_size: int) -> Dataloader:
    current_path = Path(__file__).parent
    ds, info = tfds.load(name="mnist", split=["train", "test"], as_supervised=True, with_info=True,
                         shuffle_files=True, data_dir=current_path, batch_size=batch_size)
    train_ds, valid_ds = ds
    train_ds = train_ds.map(lambda x, y: (tf.divide(tf.cast(x, dtype=tf.float32), 255.0), tf.one_hot(y, depth=10)))
    valid_ds = valid_ds.map(lambda x, y: (tf.divide(tf.cast(x, dtype=tf.float32), 255.0), tf.one_hot(y, depth=10)))
    train_ds, valid_ds = train_ds.cache().repeat(), valid_ds.cache().repeat()
    input_shape = tuple([-1] + list(info.features["image"].shape))
    num_train_batches = math.ceil(info.splits["train"].num_examples / batch_size)
    num_val_batches = math.ceil(info.splits["test"].num_examples / batch_size)
    return Dataloader(
        train_data=iter(tfds.as_numpy(train_ds)), val_data=iter(tfds.as_numpy(valid_ds)),
        input_shape=input_shape, batch_size=batch_size,
        num_train_batches=num_train_batches, num_val_batches=num_val_batches
    )


data = asyncio.run(tfds_load_data(batch_size=1024))
model = Module([Flatten, Dense(500), Relu, Dense(100), Relu, Dense(10), LogSoftmax])
model.compile(loss=CCELoss, optimizer=RMSProp(step_size=0.001), metrics=[accuracy])
model.fit(data, epochs=10)
model.save("tfds_mnist_v1")
interp = model.get_interpretation()
interp.plot_losses()
