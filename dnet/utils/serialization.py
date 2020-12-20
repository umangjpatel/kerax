import msgpack
import jax
import numpy as np
import enum


def _ndarray_to_bytes(arr):
    """Save ndarray to simple msgpack encoding."""
    if isinstance(arr, jax.xla.DeviceArray):
        arr = np.array(arr)
    if arr.dtype.hasobject or arr.dtype.isalignedstruct:
        raise ValueError('Object and structured dtypes not supported '
                         'for serialization of ndarrays.')
    tpl = (arr.shape, arr.dtype.name, arr.tobytes('C'))
    return msgpack.packb(tpl, use_bin_type=True)


def _dtype_from_name(name):
    """Handle JAX bfloat16 dtype correctly."""
    if name == b'bfloat16':
        return jax.numpy.bfloat16
    else:
        return np.dtype(name)


def _ndarray_from_bytes(data):
    """Load ndarray from simple msgpack encoding."""
    shape, dtype_name, buffer = msgpack.unpackb(data, raw=True)
    return np.frombuffer(buffer,
                         dtype=_dtype_from_name(dtype_name),
                         count=-1,
                         offset=0).reshape(shape, order='C')


class _MsgpackExtType(enum.IntEnum):
    """Messagepack custom type ids."""
    ndarray = 1
    native_complex = 2
    npscalar = 3


def _msgpack_ext_pack(x):
    """Messagepack encoders for custom types."""
    # print(x)
    if isinstance(x, (np.ndarray, jax.xla.DeviceArray)):
        return msgpack.ExtType(_MsgpackExtType.ndarray, _ndarray_to_bytes(x))
    if np.issctype(type(x)):
        # pack scalar as ndarray
        return msgpack.ExtType(_MsgpackExtType.npscalar, _ndarray_to_bytes(np.asarray(x)))
    elif isinstance(x, complex):
        return msgpack.ExtType(_MsgpackExtType.native_complex,
                               msgpack.packb((x.real, x.imag)))
    return x


def _msgpack_ext_unpack(code, data):
    """Messagepack decoders for custom types."""
    if code == _MsgpackExtType.ndarray:
        return _ndarray_from_bytes(data)
    elif code == _MsgpackExtType.native_complex:
        complex_tuple = msgpack.unpackb(data)
        return complex(complex_tuple[0], complex_tuple[1])
    elif code == _MsgpackExtType.npscalar:
        ar = _ndarray_from_bytes(data)
        return ar[()]  # unpack ndarray to scalar
    return msgpack.ExtType(code, data)


def msgpack_serialize(pytree):
    """Save data structure to bytes in msgpack format.
     Low-level function that only supports python trees with array leaves,
     for custom objects use `to_bytes`.
     Args:
       pytree: python tree of dict, list, tuple with python primitives
         and array leaves.
     Returns:
       msgpack-encoded bytes of pytree.
     """
    return msgpack.packb(pytree, default=_msgpack_ext_pack, strict_types=True)


def msgpack_restore(encoded_pytree):
    """Restore data structure from bytes in msgpack format.
    Low-level function that only supports python trees with array leaves,
    for custom objects use `from_bytes`.
    Args:
      encoded_pytree: msgpack-encoded bytes of python tree.
    Returns:
      Python tree of dict, list, tuple with python primitive
      and array leaves.
    """
    return msgpack.unpackb(
        encoded_pytree, ext_hook=_msgpack_ext_unpack, raw=False)
