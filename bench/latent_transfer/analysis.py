# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Per-document accuracy, paired bootstrap confidence intervals and promotion verdicts.

NumPy only. Results go to an ordinary JSON results file, so floats are fine here (nothing in
this module feeds the canonical-JSON digest, which forbids floats).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

#: Percentile bounds of the two-sided 95% bootstrap interval.
CI_LOW_PCT = 2.5
CI_HIGH_PCT = 97.5
DEFAULT_RESAMPLES = 2000


def per_doc(
    records: Iterable[Mapping[str, Any]],
    arm: str,
    family: str,
    key: str = "correct",
    expected_per_doc: int | None = None,
) -> dict[str, float]:
    """Mean of ``key`` per ``doc_id`` over the records of one arm and task family.

    If ``expected_per_doc`` is given, every document must have exactly that many records
    (a missing item would otherwise silently reweight the per-document mean).
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for rec in records:
        if rec["arm"] != arm or rec["family"] != family:
            continue
        doc_id = rec["doc_id"]
        sums[doc_id] = sums.get(doc_id, 0.0) + float(rec[key])
        counts[doc_id] = counts.get(doc_id, 0) + 1
    if expected_per_doc is not None:
        bad = {d: c for d, c in counts.items() if c != expected_per_doc}
        if bad:
            raise ValueError(f"docs with item count != {expected_per_doc}: {sorted(bad.items())}")
    return {doc_id: sums[doc_id] / counts[doc_id] for doc_id in sums}


def bootstrap_diff(
    a: Mapping[str, float],
    b: Mapping[str, float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    strict: bool = True,
) -> tuple[float, float, float, int]:
    """Paired bootstrap of ``mean(a - b)`` over the sorted shared doc ids.

    Returns ``(mean, lo, hi, n)`` with a percentile 95% interval. Raises ``ValueError`` when
    ``resamples`` is not positive, the arms share no documents, or (``strict``, the default) the
    two arms cover different document sets. ``strict=False`` uses the intersection.
    """
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if strict and set(a) != set(b):
        raise ValueError(
            f"arms cover different docs: only in a={sorted(set(a) - set(b))[:5]}, "
            f"only in b={sorted(set(b) - set(a))[:5]}"
        )
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("arms share no doc ids")
    diffs = np.array([a[d] - b[d] for d in shared], dtype=np.float64)
    n = len(shared)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [CI_LOW_PCT, CI_HIGH_PCT])
    return float(diffs.mean()), float(lo), float(hi), n


def verdict(diffs: Mapping[str, tuple[float, float, float, int]], margin: float) -> bool:
    """True iff there is at least one diff and every lower bound is at least ``margin``.

    An empty ``diffs`` is never a pass: no evidence cannot promote a claim.
    """
    return bool(diffs) and all(d[1] >= margin for d in diffs.values())
