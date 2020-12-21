from typing import Dict

import dill
import msgpack
from jax.tree_util import tree_flatten, tree_unflatten
from jax import device_put


def save_module(file_name: str, **config: dict):
    serialized_config = {}
    for k, v in config.items():
        item_dill: bytes = dill.dumps(v)
        item_msgpack: bytes = msgpack.packb(item_dill, use_bin_type=True)
        serialized_config[k] = item_msgpack

    with open(f"{file_name}.msgpack", "wb") as f:
        serialized_data = msgpack.packb(serialized_config)
        f.write(serialized_data)
        print("Saved model")


def load_module(file_name: str):
    with open(f"{file_name}.msgpack", "rb") as f:
        deserialized_data: bytes = f.read()
        deserialized_config: Dict[str, bytes] = msgpack.unpackb(deserialized_data)
        for k, v in deserialized_config.items():
            item_dill = msgpack.unpackb(v)
            deserialized_config[k] = dill.loads(item_dill)
    return deserialized_config


def convert_to_tensor(data):
    flat_data, data_tree_struct = tree_flatten(data)
    for i, item in enumerate(flat_data):
        flat_data[i] = device_put(item)
    return tree_unflatten(data_tree_struct, flat_data)
