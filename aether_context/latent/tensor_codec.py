"""ndarray <-> (TensorRefV1, bytes). Bytes are little-endian, C order, digest = sha256(bytes)."""

from __future__ import annotations

import hashlib
import math
from typing import cast

import numpy as np

from ..contracts.latent_v1 import Dtype, TensorRefV1

# Soft-prefix capsules are float tensors. The contract also allows bfloat16 (no NumPy dtype)
# and int8/uint8 (would need a codebook/dequant path that does not exist), so none are emitted
# or loaded here.
_NP_DTYPES = {"float16": "<f2", "float32": "<f4"}
SUPPORTED_DTYPES = tuple(_NP_DTYPES)


class TensorIntegrityError(ValueError):
    """Tensor bytes do not match their reference."""


class TensorShapeError(TensorIntegrityError):
    """The reference's shape is inconsistent with its dtype and byte_count."""


def tensor_bytes(array: np.ndarray, dtype: str = "float32") -> tuple[TensorRefV1, bytes]:
    if dtype not in _NP_DTYPES:
        raise ValueError(f"unsupported tensor dtype {dtype!r}")
    with np.errstate(over="ignore"):
        arr = np.ascontiguousarray(np.asarray(array).astype(np.dtype(_NP_DTYPES[dtype])))
    if arr.ndim < 1:
        raise ValueError("tensor must have at least one dimension")
    if not np.isfinite(arr).all():
        raise ValueError("tensor values must be finite in the storage dtype")
    raw = arr.tobytes(order="C")
    ref = TensorRefV1(digest=hashlib.sha256(raw).hexdigest(), dtype=cast(Dtype, dtype),
                      shape=[int(d) for d in arr.shape], byte_count=len(raw))
    return ref, raw


def load_tensor(ref: TensorRefV1, raw: bytes) -> np.ndarray:
    if ref.dtype not in _NP_DTYPES:
        raise TensorIntegrityError(f"unsupported tensor dtype {ref.dtype!r}")
    if len(raw) != ref.byte_count:
        raise TensorIntegrityError("tensor bytes do not match byte_count")
    dtype = np.dtype(_NP_DTYPES[ref.dtype])
    if any(d <= 0 for d in ref.shape) or math.prod(ref.shape) * dtype.itemsize != ref.byte_count:
        raise TensorShapeError("tensor shape does not match byte_count")
    if hashlib.sha256(raw).hexdigest() != ref.digest:
        raise TensorIntegrityError("tensor bytes do not match digest")
    return np.frombuffer(raw, dtype=dtype).reshape(ref.shape).copy()


__all__ = ["SUPPORTED_DTYPES", "TensorIntegrityError", "TensorShapeError", "tensor_bytes", "load_tensor"]
