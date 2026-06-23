"""TurboVec proof harness — footprint + recall@k + query latency, fp32 vs 8-bit, across a GROWING pool.

The headline TurboVec claim is: 8-bit scalar quantization cuts the resident vector footprint ~4x while
keeping recall >= 0.98, and the gain compounds as the pool grows (more vectors stay resident -> fewer
cold reads -> flatter latency under long runs). `drift_vs_window.py` measures end-to-end coherence; it does
NOT isolate footprint/recall/latency. This does, with just numpy + the real codec (no API key, no Session).

  python -m bench.turbovec_bench                 # default sweep
  python -m bench.turbovec_bench --sizes 10000,50000,100000 --dim 256 --queries 200 --k 10 --bits 8
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from aether_context.quantize import (compression_ratio, dequantize, packed_bytes_per_row, quantize)


def _unit(n: int, d: int, rng) -> np.ndarray:
    m = rng.standard_normal((n, d)).astype(np.float32)
    m /= (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    return m


def _topk(mat: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    sims = mat @ q.T                      # (N, Q) cosine (unit vectors)
    return np.argpartition(-sims, kth=min(k, mat.shape[0] - 1), axis=0)[:k].T  # (Q, k) indices


def _latency_ms(mat: np.ndarray, queries: np.ndarray, k: int, reps: int = 3) -> tuple[float, float]:
    times = []
    for _ in range(reps):
        for q in queries:
            t = time.perf_counter()
            sims = mat @ q
            np.argpartition(-sims, kth=min(k, mat.shape[0] - 1))[:k]
            times.append((time.perf_counter() - t) * 1000.0)
    times.sort()
    p50 = times[len(times) // 2]
    p99 = times[min(len(times) - 1, int(len(times) * 0.99))]
    return p50, p99


def run(sizes, dim, queries, k, bits, seed=7) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        mat = _unit(n, dim, rng)
        qs = _unit(queries, dim, rng)
        # fp32 baseline: exact top-k + footprint + latency
        fp32_topk = _topk(mat, qs, k)
        fp32_bytes = n * dim * 4
        fp32_p50, fp32_p99 = _latency_ms(mat, qs, k)
        # 8-bit: quantize -> dequantize -> exact top-k on the reconstruction
        codes, scales = quantize(mat, bits=bits)
        deq = dequantize(codes, scales, dim, bits=bits).astype(np.float32)
        q_topk = _topk(deq, qs, k)
        q_bytes = packed_bytes_per_row(dim, bits) * n + scales.nbytes
        q_p50, q_p99 = _latency_ms(deq, qs, k)
        # recall@k = overlap of returned index sets, averaged over queries
        recalls = [len(set(a.tolist()) & set(b.tolist())) / k for a, b in zip(fp32_topk, q_topk)]
        rows.append({
            "n": n, "dim": dim, "bits": bits,
            "fp32_mb": round(fp32_bytes / 1e6, 1), "q_mb": round(q_bytes / 1e6, 1),
            "compression_x": round(fp32_bytes / max(q_bytes, 1), 2),
            "codec_ratio_x": round(compression_ratio(dim, bits), 2),
            "recall_at_k": round(float(np.mean(recalls)), 4),
            "fp32_p50_ms": round(fp32_p50, 3), "fp32_p99_ms": round(fp32_p99, 3),
            "q_p50_ms": round(q_p50, 3), "q_p99_ms": round(q_p99, 3),
        })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TurboVec footprint/recall/latency proof")
    ap.add_argument("--sizes", default="10000,50000,100000")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--bits", type=int, default=8)
    a = ap.parse_args(argv)
    sizes = [int(s) for s in a.sizes.split(",") if s.strip()]
    rows = run(sizes, a.dim, a.queries, a.k, a.bits)
    hdr = ("n", "fp32_mb", "q_mb", "compression_x", "recall_at_k", "fp32_p99_ms", "q_p99_ms")
    print(f"TurboVec {a.bits}-bit vs fp32 | dim={a.dim} k={a.k} queries={a.queries}")
    print("  ".join(f"{h:>13}" for h in hdr))
    for r in rows:
        print("  ".join(f"{r[h]:>13}" for h in hdr))
    worst_recall = min(r["recall_at_k"] for r in rows)
    print(f"\nworst recall@{a.k} = {worst_recall:.4f} (claim: >= 0.98 at 8-bit)  |  "
          f"footprint cut ~{rows[-1]['compression_x']}x at n={sizes[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
