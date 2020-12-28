from dnet.data import Dataloader
from dnet.utils.tensor import Tensor


def load_data() -> Dataloader:
    """Loads the binary tiny MNIST dataset (available on Google Colaboratory).
    :return: a dataloader consisting of training and validation data
    """
    import asyncio

    async def load() -> Dataloader:
        import pandas as pd
        import jax.numpy as jnp
        from jax import device_put
        from pathlib import Path

        path: Path = Path(__file__).parent
        data: pd.DataFrame = pd.read_csv(path / "train.csv", header=None)

        # 80% training, 20% validation
        train_data: pd.DataFrame = data.sample(frac=0.8, random_state=0)
        val_data: pd.DataFrame = data.drop(train_data.index)

        train_labels: Tensor = jnp.expand_dims(device_put(train_data[0].values), axis=1)
        train_images: Tensor = device_put(train_data.iloc[:, 1:].values) / 255.0

        val_labels: Tensor = jnp.expand_dims(device_put(val_data[0].values), axis=1)
        val_images: Tensor = device_put(val_data.iloc[:, 1:].values) / 255.0

        return Dataloader(train_data=(train_images, train_labels),
                          val_data=(val_images, val_labels))

    return asyncio.run(load())
