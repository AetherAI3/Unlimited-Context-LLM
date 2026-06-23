# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""TurboVec pool integration: quantized on-disk codes, exact in-RAM search, persist + reopen.

quantize_bits>0 stores compressed codes in the vectors file (smaller footprint) while keeping the
float32 unit vector resident, so live search is recall-EXACT vs the float32 pool; a reopen dequantizes
(8-bit, ~0.99 recall). bits=0 is the untouched default path."""
import numpy as np

from aether_context.config import PoolConfig
from aether_context.context_pool import ContextPool, Slice


def _unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def _pool(d, qbits):
    cfg = PoolConfig(pool_gb=5, dim=256, index="flat", dir=d, quantize_bits=qbits)
    return ContextPool(cfg, ceiling_bytes=10_000_000)


def _fill(pool, vecs):
    for i, v in enumerate(vecs):
        pool.add(Slice(id=f"s{i}", session="x", vector=v, text=f"t{i}", tokens=1))


def test_quantized_search_matches_float32(tmp_path):
    rng = np.random.default_rng(3)
    vecs = [_unit(rng.standard_normal(256)) for _ in range(200)]
    p0, p8 = _pool(tmp_path / "f32", 0), _pool(tmp_path / "q8", 8)
    _fill(p0, vecs)
    _fill(p8, vecs)
    q = vecs[7]
    r0 = [s.id for s in p0.search(q, k=5)]
    r8 = [s.id for s in p8.search(q, k=5)]
    assert r0 == r8                      # float32 kept in RAM -> live search is recall-EXACT
    assert p8.stats()["quantized"] is True and p0.stats()["quantized"] is False


def test_quantized_persists_and_reopens(tmp_path):
    rng = np.random.default_rng(5)
    d = tmp_path / "q8"
    p = _pool(d, 8)
    _fill(p, [_unit(rng.standard_normal(256)) for _ in range(50)])
    p.close()
    assert (d / "vectors.q8").exists()   # codes on disk, not vectors.f32
    p2 = _pool(d, 8)                      # reopen -> dequantizes on load
    assert len(p2) == 50 and p2.stats()["quantized"] is True
    assert len(p2.search(_unit(rng.standard_normal(256)), k=3)) == 3
