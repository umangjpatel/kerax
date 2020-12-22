from jax.interpreters.xla import DeviceArray as Tensor


def convert_to_tensor(data):
    from jax.tree_util import tree_flatten, tree_unflatten
    from jax import device_put
    flat_data, data_tree_struct = tree_flatten(data)
    for i, item in enumerate(flat_data):
        flat_data[i] = device_put(item)
    return tree_unflatten(data_tree_struct, flat_data)
