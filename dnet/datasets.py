from pathlib import Path
from typing import Collection

import jax.numpy as tensor
import pandas as pd
from fastai.datasets import untar_data, URLs
from fastai.vision.data import ImageList, ImageDataBunch, imagenet_stats
from fastai.vision.image import Transform
from fastai.vision.transform import get_transforms


def mnist_tiny(flatten: bool = False, one_hot_encoding: bool = False):
    path: Path = untar_data(URLs.MNIST_TINY)
    transforms: Collection[Transform] = get_transforms(do_flip=False)
    data: ImageDataBunch = (ImageList.from_folder(path=path)
                            .split_by_folder()
                            .label_from_folder()
                            .transform(transforms, size=28)
                            .databunch()
                            .normalize(imagenet_stats))
    train_items: tensor.array = tensor.array([item.data.transpose(0, -1).numpy() for item in data.train_ds.x])
    train_targets: tensor.array = tensor.array([item.data for item in data.train_ds.y]).reshape(-1, 1)
    val_items: tensor.array = tensor.array([item.data.transpose(0, -1).numpy() for item in data.valid_ds.x])
    val_targets: tensor.array = tensor.array([item.data for item in data.valid_ds.y]).reshape(-1, 1)
    if flatten:
        train_items = train_items.reshape((-1, tensor.prod(list(train_items.shape[1:]))))
        val_items = val_items.reshape((-1, tensor.prod(list(val_items.shape[1:]))))
    if one_hot_encoding:
        train_targets = tensor.asarray(pd.get_dummies(train_targets))
        val_targets = tensor.asarray(pd.get_dummies(val_targets))
    return (train_items, train_targets), (val_items, val_targets)
