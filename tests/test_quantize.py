# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""TurboVec codec: round-trip + the recall@k parity GATE that decides a safe quantization ratio.

Coherence rides on retrieval recall, so the codec is only safe at a ratio that keeps recall@k ~= the
float32 baseline. Measured on high-entropy random unit vectors (worst case): 8-bit holds recall >= 0.98
(~4x compression); 4-bit drops to ~0.83 and is therefore NOT recall-safe as a default — it stays
available but flagged. Real (structured) embeddings do better than random, but the default must be safe.
"""
import numpy as np

from aether_context.quantize import (compression_ratio, dequantize, packed_bytes_per_row, quantize)


def _unit(m):
    return (m / np.linalg.norm(m, axis=1, keepdims=True)).astype(np.float32)


def _recall_at_k(bits, n=3000, dim=256, q=200, k=10, seed=1):
    rng = np.random.default_rng(seed)
    base = _unit(rng.standard_normal((n, dim)))
    queries = _unit(rng.standard_normal((q, dim)))
    approx = dequantize(*quantize(base, bits), dim, bits)
    hits = 0
    for i in range(q):
        true_top = set(np.argsort(-(base @ queries[i]))[:k])
        appx_top = set(np.argsort(-(approx @ queries[i]))[:k])
        hits += len(true_top & appx_top)
    return hits / (q * k)


def test_bits0_is_identity():
    m = _unit(np.random.default_rng(0).standard_normal((5, 256)))
    back = dequantize(*quantize(m, 0), 256, 0)
    assert np.allclose(m, back, atol=1e-6)


def test_roundtrip_is_unit_and_packs_to_expected_bytes():
    m = _unit(np.random.default_rng(2).standard_normal((7, 256)))
    for bits in (4, 8):
        codes, scales = quantize(m, bits)
        assert codes.shape[1] == packed_bytes_per_row(256, bits)
        d = dequantize(codes, scales, 256, bits)
        assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-5)


def test_8bit_is_recall_safe():
    assert _recall_at_k(8) >= 0.98               # the coherence gate — 8-bit is the safe default


def test_4bit_is_lossy_not_default_safe():
    r4, r8 = _recall_at_k(4), _recall_at_k(8)
    assert r4 < r8                               # 4-bit trades recall for size; documented, flag-only


def test_compression_ratios():
    assert compression_ratio(256, 8) > 3.5 and compression_ratio(256, 4) > 7.0
