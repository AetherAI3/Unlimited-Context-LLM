# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""``pool_mode="separate"`` must actually separate, including on an empty namespace.

The regression: ``_recall_local`` widened an empty scoped search into a global one so that a
fact stayed reachable after a close-and-reopen under a new session id. That rescue is
indistinguishable, at the point of the search, from a second principal whose namespace is
merely new — so a fresh session over a shared pool dir read every other session's slices, and
those slices flowed onward into whatever prompt the caller was assembling.

An empty namespace has one correct answer: nothing. The rescue is still available, but a caller
now has to ask for it by name, and asking is a statement that the pool dir holds one principal.
"""
from __future__ import annotations

from pathlib import Path

from aether_context.session import Session


SECRET = "operator A private: the account floor is 47250 dollars"
QUERY = "what is the account floor"


def _session(pool_dir: Path, session_id: str, **kw) -> Session:
    return Session(model="mock", pool_gb=5, pool_dir=pool_dir, session_id=session_id, **kw)


def _plant(pool_dir: Path) -> None:
    a = _session(pool_dir, "principal-a")
    a.remember(SECRET, source="user")
    for i in range(20):
        a.remember(f"filler {i}: nothing of consequence", source="tool")
    a.close()


# --- the leak ----------------------------------------------------------------
def test_a_second_principal_cannot_recall_the_firsts_memory(tmp_pool_dir):
    """The headline: a new namespace over a shared pool dir must read nothing."""
    _plant(tmp_pool_dir)
    b = _session(tmp_pool_dir, "principal-b")
    try:
        hits = b.recall(QUERY, k=8)
        leaked = [h.text for h in hits if "operator A private" in h.text]
        assert not leaked, (
            f"principal-b recalled principal-a's slices: {leaked[:1]}. pool_mode='separate' "
            "is an isolation promise and this breaks it."
        )
    finally:
        b.close()


def test_an_empty_namespace_returns_nothing_rather_than_everything(tmp_pool_dir):
    _plant(tmp_pool_dir)
    b = _session(tmp_pool_dir, "principal-b")
    try:
        assert b.recall(QUERY, k=8) == []
    finally:
        b.close()


# --- the property that must not regress in the other direction ---------------
def test_the_same_principal_still_recalls_across_a_restart(tmp_pool_dir):
    """Isolation must not cost continuity — the same id keeps reading its own memory."""
    _plant(tmp_pool_dir)
    again = _session(tmp_pool_dir, "principal-a")
    try:
        hits = again.recall(QUERY, k=8)
        assert any("operator A private" in h.text for h in hits), (
            "the owning principal lost its own memory across a reopen"
        )
    finally:
        again.close()


# --- the rescue, now opt-in --------------------------------------------------
def test_the_global_rescue_is_available_when_explicitly_requested(tmp_pool_dir):
    """A caller that cannot derive a stable id can still opt back in, knowingly."""
    _plant(tmp_pool_dir)
    b = _session(tmp_pool_dir, "principal-b", recall_scope_fallback=True)
    try:
        hits = b.recall(QUERY, k=8)
        assert any("operator A private" in h.text for h in hits), (
            "opting in did not restore the reopen rescue"
        )
    finally:
        b.close()


def test_the_rescue_is_off_unless_asked_for(tmp_pool_dir):
    """Fail-closed: the safe behaviour is the one you get by not deciding."""
    s = _session(tmp_pool_dir, "principal-a")
    try:
        assert s.recall_scope_fallback is False
    finally:
        s.close()


# --- shared mode is unaffected ------------------------------------------------
def test_shared_mode_still_reaches_across_sessions(tmp_pool_dir):
    """``pool_mode='shared'`` is an explicit request for global reach; keep it working."""
    a = _session(tmp_pool_dir, "principal-a", pool_mode="shared")
    a.remember(SECRET, source="user")
    a.close()
    b = _session(tmp_pool_dir, "principal-b", pool_mode="shared")
    try:
        hits = b.recall(QUERY, k=8)
        assert any("operator A private" in h.text for h in hits), (
            "shared mode lost its global reach"
        )
    finally:
        b.close()
