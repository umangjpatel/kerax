from typing import Tuple

from ...utils.tensor import Tensor


def load_data() -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
    """Loads the binary tiny MNIST dataset (available on Google Colaboratory).
    :return: tuple consisting of training images and training labels
    """
    import pandas as pd
    import jax.numpy as jnp
    from jax import device_put
    from pathlib import Path

    path: Path = Path(__file__).parent
    data: pd.DataFrame = pd.read_csv(path / "train.csv", header=None)

    train_data = data.sample(frac=0.8, random_state=42)  # 80% training, 20% validation
    val_data = data.drop(train_data.index)

    train_labels: Tensor = jnp.expand_dims(device_put(train_data[0].values), axis=1)
    train_images: Tensor = device_put(train_data.iloc[:, 1:].values) / 255.0

    val_labels: Tensor = jnp.expand_dims(device_put(val_data[0].values), axis=1)
    val_images: Tensor = device_put(val_data.iloc[:, 1:].values) / 255.0

    return (train_images, train_labels), (val_images, val_labels)
