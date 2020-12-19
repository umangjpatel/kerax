from dnet.tensor import Tensor


def load_data():
    import pandas as pd
    import jax.numpy as jnp
    from pathlib import Path

    path: Path = Path(__file__).parent
    train_data: pd.DataFrame = pd.read_csv(path / "train.csv", header=None)
    train_labels: Tensor = jnp.expand_dims(jnp.array(train_data[0].values), axis=1)
    train_images: Tensor = jnp.array(train_data.iloc[:, 1:].values) / 255.0
    return train_images, train_labels