# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""TurboVec — scalar quantization of unit retrieval vectors (the context pool's compression codec).

The context pool stores L2-normalized 256-dim float32 vectors (1024 B/row). TurboVec quantizes each
row to ``bits``-bit codes + a per-row float32 scale, so the ON-DISK footprint drops to
``dim*bits/8`` bytes per row — measured at exactly 4.00x for the vectors file at 8-bit.

WHAT THIS DOES NOT DO — corrected 2026-07-30 after measurement. An earlier version of this
docstring claimed "the same byte ceiling then holds that many more slices in the hot set". That is
false, in two independent ways:

  * ``ContextPool._load`` dequantizes every row back to a float32 ``Slice.vector`` and retains it
    in ``self._slices``. Measured retained heap is IDENTICAL between an fp32 pool and an 8-bit pool
    (~3,960 B/slice at 512-token text). Quantization buys disk, not resident memory.
  * the pool's admission budget, :func:`context_pool.slice_cost_bytes`, takes only ``dim``. It is
    not quantization-aware, so a quantized pool is charged the same bytes per slice and admits the
    same number of slices. Verified behaviourally: an fp32 and an 8-bit pool under one ceiling
    evict at the same count.

So quantization does not increase hot-set capacity and does not reduce cold reads. Whole-pool disk
saving at 8-bit measured -23.8%, because the JSON sidecar (verbatim slice text) is ~2.1x the size
of the vectors file and is not compressed. See ATSv2 docs/POOL_RESIDENCY.md for the measurements.

Per-row SYMMETRIC quantization: ``code = round(v/scale) + half`` over ``scale = max|v| / half``, so the
row's dynamic range maps onto the integer grid. Dequantization recovers ``(code-half)*scale`` and
RE-NORMALIZES to unit length (cosine only cares about direction). Recall@k parity vs float32 is the gate
the tests enforce (>=0.98). ``bits=0`` is the identity (the float32 path) — TurboVec is fully reversible.

Pure numpy, no dependencies. Only 4- and 8-bit are supported (the codes pack to whole bytes cleanly).
"""
from __future__ import annotations

import numpy as np

SUPPORTED_BITS = (0, 4, 8)


def packed_bytes_per_row(dim: int, bits: int) -> int:
    """On-disk/in-RAM code bytes for one row at ``bits`` (excludes the 4-byte per-row scale)."""
    if bits == 0:
        return dim * 4
    if bits == 8:
        return dim
    if bits == 4:
        return (dim + 1) // 2
    raise ValueError(f"unsupported bits={bits}; use one of {SUPPORTED_BITS}")


def quantize(matrix: np.ndarray, bits: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """``(n, dim)`` float32 rows -> ``(codes uint8 (n, packed), scales float32 (n,))``.

    Rows are assumed ~unit (the pool stores unit vectors). ``bits=0`` returns the raw float32 bytes as
    uint8 with unit scales (identity passthrough), so callers can treat both paths uniformly.
    """
    m = np.ascontiguousarray(matrix, dtype=np.float32)
    n, dim = m.shape
    if bits == 0:
        return m.view(np.uint8).reshape(n, dim * 4), np.ones(n, dtype=np.float32)
    levels = (1 << bits) - 1
    half = levels >> 1
    maxabs = np.max(np.abs(m), axis=1)
    scales = np.where(maxabs > 1e-12, maxabs / half, 1.0).astype(np.float32)
    q = np.rint(m / scales[:, None]) + half
    q = np.clip(q, 0, levels).astype(np.uint8)               # (n, dim) integer codes
    if bits == 8:
        return np.ascontiguousarray(q), scales
    # 4-bit: pack two nibbles per byte (pad odd dim with a zero nibble).
    if dim & 1:
        q = np.pad(q, ((0, 0), (0, 1)))
    codes = ((q[:, 0::2] << 4) | q[:, 1::2]).astype(np.uint8)
    return np.ascontiguousarray(codes), scales


def dequantize(codes: np.ndarray, scales: np.ndarray, dim: int, bits: int = 4) -> np.ndarray:
    """Inverse of :func:`quantize` -> ``(n, dim)`` float32 UNIT rows (re-normalized for cosine)."""
    if bits == 0:
        return codes.reshape(codes.shape[0], dim * 4).view(np.float32).reshape(-1, dim).copy()
    levels = (1 << bits) - 1
    half = levels >> 1
    if bits == 8:
        q = codes.astype(np.float32)
    else:  # 4-bit: unpack the nibbles
        hi = (codes >> 4) & 0xF
        lo = codes & 0xF
        q = np.empty((codes.shape[0], hi.shape[1] * 2), dtype=np.float32)
        q[:, 0::2] = hi
        q[:, 1::2] = lo
        q = q[:, :dim]
    v = (q - half) * scales[:, None]
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return (v / norms).astype(np.float32)


def compression_ratio(dim: int, bits: int) -> float:
    """float32 row bytes / quantized row bytes (vector only) — the headline TurboVec multiple."""
    if bits == 0:
        return 1.0
    return (dim * 4) / (packed_bytes_per_row(dim, bits) + 4)
