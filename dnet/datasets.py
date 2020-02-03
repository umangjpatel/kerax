import os
from pathlib import Path
from typing import Dict, Tuple

import jax.numpy as tensor
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds

tfds.disable_progress_bar()


def mnist(flatten: bool = False, one_hot_encoding: bool = False,
          data_dir: str = os.path.join("..", "datasets", "mnist")):
    path: Path = Path(data_dir)
    downloaded_data, info = tfds.load(name="mnist", batch_size=-1, data_dir=path, with_info=True)
    mnist_data: Dict[str, Dict[str, np.array]] = tfds.as_numpy(downloaded_data)
    train_data, valid_data = mnist_data.get("train"), mnist_data.get("test")
    input_shape: Tuple[int, ...] = info.features["image"].shape
    train_images, train_labels = tensor.asarray(train_data.get("image"), dtype=tensor.float32), tensor.asarray(
        train_data.get("label"), dtype=tensor.float32).reshape(-1, 1)
    valid_images, valid_labels = tensor.asarray(valid_data.get("image"), dtype=tensor.float32), tensor.asarray(
        valid_data.get("label"), dtype=tensor.float32).reshape(-1, 1)
    if flatten:
        train_images = train_images.reshape(-1, tensor.prod(list(input_shape)))
        valid_images = valid_images.reshape(-1, tensor.prod(list(input_shape)))
    if one_hot_encoding:
        train_labels = tensor.asarray(pd.get_dummies(train_labels), dtype=tensor.float32)
        valid_labels = tensor.asarray(pd.get_dummies(valid_labels), dtype=tensor.float32)
    return (train_images, train_labels), (valid_images, valid_labels)
