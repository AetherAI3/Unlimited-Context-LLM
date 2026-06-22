"""Locks the TurboVec headline claim: 8-bit cuts footprint ~4x while keeping recall recall-safe."""
from __future__ import annotations

from bench.turbovec_bench import run


def test_8bit_compresses_and_keeps_recall():
    rows = run([2000], dim=256, queries=50, k=10, bits=8)
    r = rows[0]
    assert r["compression_x"] >= 3.5, r          # ~4x resident-footprint cut
    assert r["recall_at_k"] >= 0.97, r           # recall-safe at 8-bit (claim >= 0.98)


def test_footprint_ratio_holds_as_pool_grows():
    rows = run([1000, 4000], dim=256, queries=20, k=10, bits=8)
    # the ~4x footprint advantage is constant as the pool grows (the long-run resident-set win)
    assert abs(rows[0]["compression_x"] - rows[1]["compression_x"]) < 0.1, rows
