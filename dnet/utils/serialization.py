import dill
import msgpack

"""
Blatantly copy-pasted from flax library
"""


def save_model(fname: str, **config):
    print(fname)
    layers_dill: bytes = dill.dumps(config.get("layers"))
    layers_msgpack: bytes = msgpack.packb(layers_dill, use_bin_type=True)
    print(f"Layers msgpack : {layers_msgpack}")

    opt_dill: bytes = dill.dumps(config.get("optimizer"))
    opt_msgpack: bytes = msgpack.packb(opt_dill, use_bin_type=True)
    print(f"Opt msgpack : {opt_msgpack}")

    loss_dill: bytes = dill.dumps(config.get("loss"))
    loss_msgpack: bytes = msgpack.packb(loss_dill, use_bin_type=True)
    print(f"Loss msgpack : {loss_msgpack}")

    params_dill: bytes = dill.dumps(config.get("params"))
    params_msgpack: bytes = msgpack.packb(params_dill, use_bin_type=True)
    print(f"Params msgpack : {params_msgpack}")
