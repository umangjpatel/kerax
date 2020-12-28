from dnet.data import Dataloader
from dnet.layers import Dense, Relu, Sigmoid
from dnet.losses import BCELoss
from dnet.models import Module
from dnet.optimizers import SGD
from dnet.utils.tensor import Tensor


def load_data() -> Dataloader:
    import asyncio

    async def load() -> Dataloader:
        import pandas as pd
        import jax.numpy as jnp
        from jax import device_put
        from pathlib import Path

        path: Path = Path(__file__).parent
        dataset: pd.DataFrame = pd.read_csv(path / "train.csv", header=None)

        # 80% training, 20% validation
        train_data: pd.DataFrame = dataset.sample(frac=0.8, random_state=0)
        val_data: pd.DataFrame = dataset.drop(train_data.index)

        train_labels: Tensor = jnp.expand_dims(device_put(train_data[0].values), axis=1)
        train_images: Tensor = device_put(train_data.iloc[:, 1:].values) / 255.0

        val_labels: Tensor = jnp.expand_dims(device_put(val_data[0].values), axis=1)
        val_images: Tensor = device_put(val_data.iloc[:, 1:].values) / 255.0

        return Dataloader(train_data=(train_images, train_labels),
                          val_data=(val_images, val_labels))

    return asyncio.run(load())


data = load_data()

model = Module([Dense(100), Relu, Dense(1), Sigmoid])
model.compile(loss=BCELoss, optimizer=SGD(step_size=0.01))
model.fit(data=data, epochs=10)
model.save(file_name="log_reg")
interp = model.get_interpretation()
interp.plot_losses()

model = Module()
model.load(file_name="log_reg")
# model already compiled when loaded from serialized file
model.fit(data=data, epochs=100)
model.save(file_name="log_reg_v2")
interp = model.get_interpretation()
interp.plot_losses()
