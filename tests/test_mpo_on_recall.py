# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""The MPO chain must be reachable from EVERY retrieval path, not just the pager's.

The regression these tests exist for: ``MpoChain.expand`` was invoked only inside
``Session._cold_retrieve``, which is registered solely as the pager's ``retrieve_fn``. Any
caller that retrieved through the public :meth:`Session.recall` — the documented way to pull
context back into the working window — bypassed the pager and therefore silently received
isolated cosine neighbours instead of the connected thread. The chain was *configured* on and
*observably* absent, which is the worst of both worlds: no error, no log, just weaker context.

So the property under test is not "the chain exists" but "the chain runs on the path callers
actually use". These tests count real ``expand`` invocations rather than asserting on ordering,
because ordering can coincide while the chain never ran.

Style mirrors ``test_session.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aether_context import mpo
from aether_context.session import Session


def _session(tmp_pool_dir: Path, **kw) -> Session:
    params: dict = dict(model="mock", pool_gb=5, pool_dir=tmp_pool_dir)
    params.update(kw)
    return Session(**params)


def _fill(s: Session, n: int = 40) -> None:
    """Enough distinct slices that the chain has candidates to widen beyond k."""
    for i in range(n):
        s.remember(f"fact {i}: the reactor coolant pressure reading was {900 + i} kPa")


@pytest.fixture
def counting_expand(monkeypatch):
    """Count real MpoChain.expand calls without changing its behaviour."""
    calls: list[int] = []
    original = mpo.MpoChain.expand

    def counted(self, seed, items, *a, **kw):
        calls.append(len(items))
        return original(self, seed, items, *a, **kw)

    monkeypatch.setattr(mpo.MpoChain, "expand", counted)
    return calls


# --- the regression ----------------------------------------------------------
def test_recall_invokes_the_mpo_chain(tmp_pool_dir, counting_expand):
    """``recall()`` must widen through the chain — the bug was that it never did."""
    with _session(tmp_pool_dir, mpo_chain=True) as s:
        _fill(s)
        hits = s.recall("reactor coolant pressure", k=5)
    assert hits, "recall returned nothing; the fixture failed, not the chain"
    assert counting_expand, (
        "MpoChain.expand was never called from recall() — retrieval is serving plain cosine "
        "neighbours while reporting mpo_chain=True"
    )


def test_recall_without_the_chain_does_not_invoke_it(tmp_pool_dir, counting_expand):
    """The off switch must genuinely bypass expansion, so the on-case proves something."""
    with _session(tmp_pool_dir, mpo_chain=False) as s:
        _fill(s)
        hits = s.recall("reactor coolant pressure", k=5)
    assert hits
    assert not counting_expand, "chain expanded despite mpo_chain=False"


def test_chain_widens_beyond_k_before_selecting(tmp_pool_dir, counting_expand):
    """The chain must see a fanned-out candidate set, or it cannot add connected context."""
    k = 5
    with _session(tmp_pool_dir, mpo_chain=True, chain_fanout=4) as s:
        _fill(s, n=60)
        s.recall("reactor coolant pressure", k=k)
    assert counting_expand, "chain never ran"
    assert counting_expand[0] > k, (
        f"chain was handed {counting_expand[0]} candidates for k={k}; without fan-out it can "
        "only reorder the cosine top-k and adds no connected context"
    )


# --- provenance safety -------------------------------------------------------
def test_source_filter_is_not_widened_back_open(tmp_pool_dir):
    """A provenance-filtered recall must never be re-widened into excluded slices.

    ``sources`` exists so a caller can exclude the model's own spilled notes (SAFETY.md).
    Widening after filtering would quietly reintroduce them.
    """
    with _session(tmp_pool_dir, mpo_chain=True) as s:
        for i in range(30):
            s.remember(f"operator states threshold {i} is authoritative", source="user")
        s._encode_slice("model speculation: threshold may be far higher",
                        salience=0.9, tags={"kind": "spill", "source": "model"})
        hits = s.recall("threshold", k=8, sources={"user"})
    assert hits, "filtered recall returned nothing"
    for h in hits:
        assert h.meta.get("source") != "model", (
            "MPO widening reintroduced a model-sourced slice into a recall filtered to user "
            "provenance"
        )


# --- observability -----------------------------------------------------------
def test_status_reports_observed_chain_activity_not_just_config(tmp_pool_dir):
    """Readiness needs OBSERVED behaviour: did it run, what did it add, did it fall back."""
    with _session(tmp_pool_dir, mpo_chain=True, chain_width=8, chain_hops=1, chain_fanout=4) as s:
        _fill(s, n=60)
        before = s.status_dict()
        assert before["mpo_chain"] is True
        assert before["mpo_chain_expansions"] == 0, "nothing retrieved yet"
        s.recall("reactor coolant pressure", k=5)
        after = s.status_dict()
    assert after["mpo_chain_width"] == 8
    assert after["mpo_chain_hops"] == 1
    assert after["mpo_chain_fanout"] == 4
    assert after["mpo_chain_expansions"] >= 1, (
        "status reports the chain as enabled but never observed it run — configuration is not "
        "evidence"
    )


def test_chain_failure_is_counted_not_swallowed(tmp_pool_dir, monkeypatch):
    """Fail-soft must stay soft, but a silent fallback is indistinguishable from success."""
    def boom(self, seed, items, *a, **kw):
        raise RuntimeError("synthetic chain failure")

    monkeypatch.setattr(mpo.MpoChain, "expand", boom)
    with _session(tmp_pool_dir, mpo_chain=True) as s:
        _fill(s, n=60)
        hits = s.recall("reactor coolant pressure", k=5)
        status = s.status_dict()
    assert hits, "fail-soft broken: a chain error must still serve cosine results"
    assert status["mpo_chain_fallbacks"] >= 1, (
        "the chain failed and served degraded results while reporting nothing — a readiness "
        "surface would have called this healthy"
    )


def test_pager_cold_path_still_chains(tmp_pool_dir, counting_expand):
    """The original path must keep working after being refactored onto the shared primitive."""
    with _session(tmp_pool_dir, mpo_chain=True) as s:
        _fill(s, n=60)
        qvec = s.encoder.encode("reactor coolant pressure")
        s._cold_retrieve(None, np.asarray(qvec, dtype=np.float32), 5)
    assert counting_expand, "the pager cold path lost its chain expansion in the refactor"
