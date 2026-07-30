# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Accounting must say what it measured, and must not lie about memory.

One number, `bytes_used()`, stood for four different quantities and was wrong
about most of them: it is a flat admission charge, not disk and not heap.
Measured against a real pool it undercounts disk by ~45% and understates retained
heap by roughly 1.8x. The published resident-RAM figure was wrong by ~85x, and
`status` computed it from a hardcoded constant while materialising the whole pool
to print two integers.

These pin the corrections. They do not re-litigate the measurements — those live
in ATSv2 docs/POOL_RESIDENCY.md — they pin that the code now distinguishes
observed from projected, and refuses a configuration that cannot fit in RAM.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from aether_context.config import PoolConfig
from aether_context.context_pool import (
    DEFAULT_HEAP_BUDGET_BYTES,
    Slice,
    MEASURED_HEAP_BYTES_PER_SLICE,
    METADATA_FILENAME,
    ContextPool,
    check_heap_budget,
    project_loaded_heap_bytes,
    slice_cost_bytes,
)


def pool(tmp_path, **over) -> ContextPool:
    return ContextPool(PoolConfig(dir=tmp_path / "pool", pool_gb=5, **over))


#: 512 tokens at ~4 bytes/token — the slice size the residency measurements used,
#: and the size at which the budget's 45% disk undercount actually shows. A
#: 512-BYTE slice is small enough that the flat charge exceeds real disk, which is
#: the opposite error and made an earlier version of these tests pass vacuously.
REALISTIC_TEXT = "x" * 2048


def fill(p: ContextPool, n: int, text: str = REALISTIC_TEXT) -> None:
    rng = np.random.default_rng(7)
    for i in range(n):
        p.add(
            Slice(
                id=f"s{i}",
                session="sess",
                vector=rng.normal(size=256).astype(np.float32),
                text=text,
                tokens=128,
                meta={"k": "v"},
                score=0.5,
            )
        )


# ------------------------------------------------------- observed vs projected


def test_disk_metrics_are_named_for_what_they_measure(tmp_path):
    p = pool(tmp_path)
    fill(p, 40)
    p._flush()
    m = p.disk_metrics()
    for key in (
        "vector_disk_bytes",
        "sidecar_disk_bytes",
        "metadata_disk_bytes",
        "total_disk_bytes",
        "loaded_heap_bytes_observed",
    ):
        assert key in m, f"{key} missing"
    p.close()


def test_every_metric_declares_observed_or_projected(tmp_path):
    # The original defect was not a wrong number so much as an unlabelled one.
    p = pool(tmp_path)
    fill(p, 10)
    p._flush()
    m = p.disk_metrics()
    assert m["measurement"]["loaded_heap_bytes_observed"] == "observed"
    assert m["measurement"]["total_disk_bytes"] == "observed"
    assert m["measurement"]["budget_bytes_projected"] == "projected"
    p.close()


def test_disk_bytes_are_read_from_the_filesystem_not_computed(tmp_path):
    p = pool(tmp_path)
    fill(p, 40)
    p._flush()
    on_disk = (tmp_path / "pool" / METADATA_FILENAME).stat().st_size
    assert p.sidecar_disk_bytes() == on_disk
    assert p.total_disk_bytes() >= p.vector_disk_bytes() + p.sidecar_disk_bytes()
    p.close()


def test_the_budget_is_not_the_disk_footprint(tmp_path):
    # The specific untruth: a flat 2,224 B/slice charge against ~3,229 B actually
    # written. Conflating them made a nominal 5 GB pool occupy 6.84 GiB.
    p = pool(tmp_path)
    fill(p, 60)
    p._flush()
    assert p.total_disk_bytes() > p.budget_bytes(), (
        "real disk should exceed the admission charge — if this inverts, the charge was "
        "changed and the reach math needs revisiting"
    )
    p.close()


def test_observed_heap_counts_the_retained_vectors(tmp_path):
    p = pool(tmp_path)
    fill(p, 50)
    heap = p.loaded_heap_bytes_observed()
    # 50 slices x 256 float32 = 51,200 B of vector alone, before text and meta.
    assert heap > 50 * 256 * 4
    p.close()


def test_an_unloaded_pool_reports_zero_heap_rather_than_projecting_one(tmp_path):
    p = pool(tmp_path)
    assert p.loaded_heap_bytes_observed() == 0
    p.close()


def test_bytes_used_still_works_for_the_governor(tmp_path):
    # Kept deliberately: eviction depends on it and this pass is a truthfulness
    # fix, not an eviction redesign.
    p = pool(tmp_path)
    fill(p, 12)
    assert p.bytes_used() == p.budget_bytes() == 12 * slice_cost_bytes(256)
    p.close()


# ----------------------------------------------------------- the heap budget


def test_a_five_gb_pool_is_refused_against_the_default_budget():
    # The published guidance was "5 GB pool, 8 GB machine". It does not fit once.
    within, message = check_heap_budget(5)
    assert within is False
    assert "FULLY RESIDENT" in message
    assert "8.8 GiB" in message or "GiB retained heap" in message


def test_a_small_pool_passes_and_says_the_number_is_projected(tmp_path):
    within, message = check_heap_budget(1)
    assert within is True
    assert "projected, not measured" in message


def test_the_projection_uses_the_measured_rate_not_the_admission_charge():
    # If this ever equals the budget figure, someone has quietly re-derived the
    # projection from the policy constant and it will understate RAM again.
    projected = project_loaded_heap_bytes(1)
    admits = (1024**3) // slice_cost_bytes(256)
    assert projected == admits * MEASURED_HEAP_BYTES_PER_SLICE
    assert projected != admits * slice_cost_bytes(256)


def test_the_budget_can_be_raised_deliberately():
    within, _ = check_heap_budget(5, budget_bytes=32 * 1024**3)
    assert within is True


@pytest.mark.parametrize("pool_gb", [1, 2, 5, 10])
def test_projection_scales_linearly(pool_gb):
    # Integer division on the admitted-slice count makes this approximate rather
    # than exact; a 0.1% tolerance still catches a projection that stops scaling.
    expected = pool_gb * project_loaded_heap_bytes(1)
    assert abs(project_loaded_heap_bytes(pool_gb) - expected) / expected < 0.001


def test_the_default_budget_is_a_real_ceiling_not_infinity():
    assert 0 < DEFAULT_HEAP_BUDGET_BYTES <= 64 * 1024**3


# ------------------------------------------- counting without materialising


def test_pool_counts_reads_the_sidecar_and_never_opens_the_pool(tmp_path, monkeypatch):
    """`status` used to construct a ContextPool just to print two integers.

    Opening materialises every slice — dequantized vectors and verbatim text —
    so asking how big a pool is cost the whole pool: on the order of 75 s and
    12 GiB for a full 5 GB configuration.
    """
    from aether_context import cli

    p = pool(tmp_path)
    fill(p, 25)
    p._flush()
    p.close()

    opened: list[str] = []
    real_init = ContextPool.__init__

    def spy(self, *a, **k):
        opened.append("open")
        return real_init(self, *a, **k)

    monkeypatch.setattr(ContextPool, "__init__", spy)
    used, capacity = cli._pool_counts(PoolConfig(dir=tmp_path / "pool", pool_gb=5))

    assert used == 25
    assert capacity > 0
    assert opened == [], "counting must not construct a ContextPool"


def test_counting_an_absent_pool_reports_nothing_rather_than_guessing(tmp_path):
    from aether_context import cli

    assert cli._pool_counts(PoolConfig(dir=tmp_path / "nope", pool_gb=5)) == (0, 0)


def test_counting_survives_an_unreadable_sidecar(tmp_path):
    from aether_context import cli

    d = tmp_path / "pool"
    d.mkdir(parents=True)
    (d / METADATA_FILENAME).write_text("{ not json", encoding="utf-8")
    assert cli._pool_counts(PoolConfig(dir=d, pool_gb=5)) == (0, 0)


def test_the_sidecar_carries_what_counting_needs(tmp_path):
    # Guards the assumption the cheap path rests on.
    p = pool(tmp_path)
    fill(p, 5)
    p._flush()
    p.close()
    header = json.loads((tmp_path / "pool" / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert header["count"] == 5
    assert isinstance(header.get("ceiling_bytes"), int)
    assert isinstance(header.get("dim"), int)


# ------------------------------------------------------------ the false claim


def test_the_quantize_docstring_no_longer_claims_a_larger_hot_set():
    # It said "the same byte ceiling then holds that many more slices in the hot
    # set". Both halves are false: _load dequantizes to float32 in heap, and
    # slice_cost_bytes is not quantization-aware so the ceiling admits the same
    # count either way.
    from aether_context import quantize

    doc = quantize.__doc__ or ""
    # The phrase still appears, quoted, because the docstring retracts it by name
    # — a silent deletion would let the same claim be reintroduced by someone who
    # never learned it was wrong. What must be present is the retraction.
    assert "WHAT THIS DOES NOT DO" in doc
    assert "false" in doc, "the retraction has to name the claim as false"
    assert "does not increase hot-set capacity" in doc
    assert "ON-DISK" in doc


def test_the_admission_charge_is_still_quantization_blind(tmp_path):
    # The property the corrected docstring now asserts. If this ever changes, the
    # docstring has to change with it.
    assert slice_cost_bytes(256) == 256 * 4 + 1200
    fp32 = pool(tmp_path / "a")
    q8 = ContextPool(PoolConfig(dir=tmp_path / "b" / "pool", pool_gb=5, quantize_bits=8))
    try:
        fill(fp32, 8)
        fill(q8, 8)
        assert fp32.budget_bytes() == q8.budget_bytes()
    finally:
        fp32.close()
        q8.close()
