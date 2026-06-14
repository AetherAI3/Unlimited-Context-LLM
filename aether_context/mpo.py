# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""MPO context chain — chains the session's context space and improves selection accuracy.

Cosine/semantic search remains the retrieval mechanism; this does not replace it. The MPO
(Matrix Product Operator) chain links the session's slices into one connected space: when
cosine pulls an entry slice, the chain pulls in the slices most coupled to it, widening the
working set with the connected thread instead of isolated nearest-neighbors.

Deterministic, numpy-only, and purely additive — it only ever *adds* connected slices to a
result, never blocks or replaces a hit, so any failure degrades cleanly to plain cosine.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

log = logging.getLogger("aether_context.mpo")

DEFAULT_CHAIN_WIDTH: int = 8
DEFAULT_CHAIN_HOPS: int = 1
_FEAT: int = 8
_BOND: int = 4
_OUT: int = 4


@dataclass(frozen=True)
class ChainItem:
    """A retrieval candidate the chain operates on: id, unit vector, and ``c_t``."""

    id: str
    vector: np.ndarray
    c_t: tuple[float, float]


def _lift(x: float, p: int) -> np.ndarray:
    half = max(1, p // 2)
    ks = np.arange(1, half + 1, dtype=np.float64)
    f = np.concatenate([np.sin(np.pi * ks * x), np.cos(np.pi * ks * x)])
    if f.size < p:
        f = np.concatenate([f, np.zeros(p - f.size)])
    return f[:p]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < 1e-12 else v / n


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


class MpoChain:
    """Chains slices and expands a cosine hit into the slices most coupled to it."""

    def __init__(
        self,
        *,
        seed: int = 0,
        width: int = DEFAULT_CHAIN_WIDTH,
        hops: int = DEFAULT_CHAIN_HOPS,
    ) -> None:
        rng = np.random.default_rng(seed)
        p = _FEAT
        self.width = max(1, int(width))
        self.hops = max(1, int(hops))
        self._w0 = (rng.standard_normal((1, p, _BOND)) * 0.5).astype(np.float64)
        self._w1 = (rng.standard_normal((_BOND, p, _OUT)) * 0.5).astype(np.float64)

    def _embed(self, c_t: tuple[float, float]) -> np.ndarray:
        u = _lift(float(c_t[0]), _FEAT)
        v = _lift(float(c_t[1]), _FEAT)
        a = np.einsum("aib,i->ab", self._w0, u)[0]
        e = np.einsum("b,bjm,j->m", a, self._w1, v)
        return _unit(e)

    def coupling(self, vec_i, e_i, vec_h, e_h) -> float:
        """Similarity of two slices for chaining (higher = more coupled)."""
        return _cos(np.asarray(vec_i), np.asarray(vec_h)) * (0.5 + 0.5 * _cos(e_i, e_h))

    def expand(
        self,
        hit_ids: Sequence[str],
        candidates: Sequence[ChainItem],
        *,
        width: int | None = None,
        hops: int | None = None,
    ) -> list[str]:
        """Widen ``hit_ids`` with the candidates most coupled to them.

        Returns ordered ids: the hits first (in their given order), then the coupled slices in
        pull order, de-duplicated.
        """
        items = list(candidates)
        if not items:
            return list(hit_ids)
        width = self.width if width is None else max(1, int(width))
        hops = self.hops if hops is None else max(1, int(hops))

        a0 = _minmax(np.array([it.c_t[0] for it in items], dtype=np.float64))
        a1 = _minmax(np.array([it.c_t[1] for it in items], dtype=np.float64))
        embeds = {it.id: self._embed((a0[i], a1[i])) for i, it in enumerate(items)}
        by_id = {it.id: it for it in items}

        present = [h for h in hit_ids if h in by_id] or [items[0].id]
        selected: list[str] = []
        seen: set[str] = set()
        for h in present:
            if h not in seen:
                selected.append(h)
                seen.add(h)
        frontier = list(selected)

        for _ in range(hops):
            scored: list[tuple[float, str]] = []
            for it in items:
                if it.id in seen:
                    continue
                best = max(
                    self.coupling(it.vector, embeds[it.id], by_id[h].vector, embeds[h])
                    for h in frontier
                )
                scored.append((best, it.id))
            if not scored:
                break
            scored.sort(key=lambda p: p[0], reverse=True)
            picked = [sid for _, sid in scored[:width]]
            for sid in picked:
                if sid not in seen:
                    selected.append(sid)
                    seen.add(sid)
            frontier = picked
        return selected


__all__ = ["MpoChain", "ChainItem", "DEFAULT_CHAIN_WIDTH", "DEFAULT_CHAIN_HOPS"]
