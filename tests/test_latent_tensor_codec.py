"""Tensor bytes are separate hashed objects; manifests only reference them."""
import hashlib

import numpy as np
import pytest

pytest.importorskip("pydantic")

from aether_context.contracts import canonical  # noqa: E402
from aether_context.latent.tensor_codec import TensorIntegrityError, TensorShapeError, load_tensor, tensor_bytes  # noqa: E402


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_round_trip(dtype):
    arr = np.arange(24, dtype=np.float32).reshape(4, 6) / 7
    ref, raw = tensor_bytes(arr, dtype)
    assert ref.shape == [4, 6] and ref.dtype == dtype and ref.byte_count == len(raw)
    assert ref.digest == hashlib.sha256(raw).hexdigest()
    back = load_tensor(ref, raw)
    np.testing.assert_array_equal(back, arr.astype(dtype))
    canonical(ref)  # manifest stays float-free


def test_tampered_bytes_refused():
    ref, raw = tensor_bytes(np.ones((2, 3), np.float32), "float16")
    with pytest.raises(TensorIntegrityError, match="digest"):
        load_tensor(ref, bytes([raw[0] ^ 1]) + raw[1:])


def test_truncated_bytes_refused():
    ref, raw = tensor_bytes(np.ones((2, 3), np.float32), "float32")
    with pytest.raises(TensorIntegrityError, match="byte_count"):
        load_tensor(ref, raw[:-1])


def test_nonfinite_rejected():
    with pytest.raises(ValueError, match="finite"):
        tensor_bytes(np.array([[np.nan, 1.0]], np.float32), "float32")


def test_overflow_to_inf_rejected():
    with pytest.raises(ValueError, match="finite"):
        tensor_bytes(np.array([[1e6]], np.float32), "float16")


@pytest.mark.parametrize("dtype", ["bfloat16", "int8", "uint8"])
def test_unsupported_dtype(dtype):
    with pytest.raises(ValueError, match="dtype"):
        tensor_bytes(np.ones((1, 1), np.float32), dtype)


def test_shape_inconsistent_with_byte_count_refused():
    ref, raw = tensor_bytes(np.ones((2, 3), np.float32), "float16")
    forged = ref.model_copy(update={"shape": [5, 6]})
    with pytest.raises(TensorShapeError, match="byte_count"):
        load_tensor(forged, raw)
    assert issubclass(TensorShapeError, TensorIntegrityError)
