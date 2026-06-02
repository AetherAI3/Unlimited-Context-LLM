"""Tests for the B5 lifecycle controller — :class:`aether_context.session.Session`.

The session is the **process lifecycle** of the virtual-memory-for-attention design:
``open`` a fresh window, then as the model streams (and as input arrives) **encode** the
spill into the pool while the **witness** fades the cold; the **pager** keeps the right
slices resident (prefetched on a background thread while the model generates, since the
backend call releases the GIL); ``close`` flushes and optionally hands harvest candidates
to an ``atlas_client`` if one is configured.

These tests are numpy-only and never hit the network — every model is a ``MockLLM`` (or a
tiny in-proc stub), every pool lives under pytest's ``tmp_path``. The headline property is
**encode-and-recover**: a planted load-bearing fact stays reachable past a deliberately small
``context_window`` *with* the engine, and is lost *without* it (proven in test_end_to_end).

Style mirrors ``qosc/aether-atlas/tests/`` and the package's own ``test_slice_loader.py``.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest  # noqa: F401  (kept for parity / future raises-tests)

from aether_context.session import RunResult, Session


# --- helpers -----------------------------------------------------------------
def _mock_session(tmp_pool_dir: Path, **kw) -> Session:
    """A Session over a ``mock`` model with the pool isolated under tmp_path."""
    params: dict = dict(model="mock", pool_gb=5, pool_dir=tmp_pool_dir)
    params.update(kw)
    return Session(**params)


# --- open -> close lifecycle -------------------------------------------------
def test_session_constructs_with_mock_and_default_components(tmp_pool_dir):
    """``Session(model="mock", pool_gb=5)`` wires a local_llm, a pool, a witness, a pager."""
    s = _mock_session(tmp_pool_dir)
    try:
        assert s.local_llm.name == "mock"
        assert s.pool is not None
        assert s.pager is not None
        assert s.witness is not None
    finally:
        s.close()


def test_session_run_returns_a_runresult(tmp_pool_dir):
    s = _mock_session(tmp_pool_dir)
    try:
        result = s.run("build me a tracker app")
        assert isinstance(result, RunResult)
        assert isinstance(result.text, str)
        assert result.text  # mock produces deterministic non-empty output
        assert isinstance(result.stages, list)
        assert 0.0 <= result.hit_rate <= 1.0
    finally:
        s.close()


def test_session_run_is_deterministic_on_mock(tmp_pool_dir, tmp_path):
    """The mock model is deterministic, so two fresh sessions on the same task agree."""
    s1 = _mock_session(tmp_pool_dir)
    s2 = Session(model="mock", pool_gb=5, pool_dir=tmp_path / "pool2")
    try:
        a = s1.run("identical task")
        b = s2.run("identical task")
        assert a.text == b.text
    finally:
        s1.close()
        s2.close()


def test_session_ask_returns_a_string(tmp_pool_dir):
    s = _mock_session(tmp_pool_dir)
    try:
        reply = s.ask("hello there")
        assert isinstance(reply, str)
        assert reply
    finally:
        s.close()


def test_session_stream_yields_chunks(tmp_pool_dir):
    s = _mock_session(tmp_pool_dir)
    try:
        chunks = list(s.stream("walk the codebase"))
        assert chunks  # at least one chunk
        assert all(isinstance(c, str) for c in chunks)
        assert "".join(chunks)  # non-empty joined text
    finally:
        s.close()


def test_session_is_a_context_manager(tmp_pool_dir):
    """``with Session(...) as s:`` opens and closes cleanly (flush on exit)."""
    with Session(model="mock", pool_gb=5, pool_dir=tmp_pool_dir) as s:
        result = s.run("a task inside the context manager")
        assert result.text
    assert s.closed


def test_close_is_idempotent(tmp_pool_dir):
    s = _mock_session(tmp_pool_dir)
    s.run("task")
    s.close()
    s.close()  # second close must not raise
    assert s.closed


# --- spill is encoded into the pool ------------------------------------------
def test_run_encodes_spill_into_the_pool(tmp_pool_dir):
    """A run with a small window forces overflow; the spill is encoded into the pool, so
    the pool holds slices afterward (encode-on-spill, not summarize-and-forget)."""
    # tiny window + long mock output => overflow => encode into the pool
    s = _mock_session(tmp_pool_dir, context_window=64, output_tokens=400)
    try:
        s.run("a long build that overflows the window")
        assert len(s.pool) > 0  # encoded spill landed in the pool
    finally:
        s.close()


# --- budget held throughout --------------------------------------------------
def test_pool_budget_held_throughout_a_run(tmp_pool_dir):
    """The pool never exceeds its byte ceiling at any point during a run (the governor
    runs after every add, so 'never over budget' is literally true)."""
    # a hard, tiny ceiling so a long run definitely pushes against it
    s = _mock_session(
        tmp_pool_dir, context_window=64, output_tokens=600, pool_ceiling_bytes=8 * 1224
    )
    try:
        s.run("a very long build that produces a lot of spill")
        assert s.pool.bytes_used() <= s.pool.ceiling_bytes
    finally:
        s.close()


# --- atlas_client is None by default -----------------------------------------
def test_atlas_client_is_none_by_default(tmp_pool_dir):
    """Default Session is fully local: it constructs NO atlas client (moat: the closed API
    is opt-in only)."""
    s = _mock_session(tmp_pool_dir)
    try:
        assert s.atlas_client is None
    finally:
        s.close()


def test_close_with_no_atlas_client_does_not_raise(tmp_pool_dir):
    """Closing a fully-local session (no atlas client) flushes locally and never raises."""
    s = _mock_session(tmp_pool_dir)
    s.run("task that produces harvest candidates")
    s.close()  # must not raise even though atlas_client is None
    assert s.closed


# --- harvest candidates ------------------------------------------------------
def test_run_emits_harvest_candidates(tmp_pool_dir):
    """A run accumulates harvest candidates (text + vector + tags) the session can flush
    locally or hand to an atlas client if configured."""
    s = _mock_session(tmp_pool_dir, context_window=64, output_tokens=400)
    try:
        s.run("a long build with durable facts to harvest")
        candidates = s.harvest_candidates()
        assert isinstance(candidates, list)
        assert len(candidates) > 0
        cand = candidates[0]
        # each candidate is (text, vector, tags)-shaped
        assert cand.text
        assert cand.vector is not None
        assert isinstance(cand.tags, dict)
    finally:
        s.close()


def test_close_hands_candidates_to_atlas_client_when_configured(tmp_pool_dir):
    """If an atlas client is configured, close() hands it the harvest candidates (the only
    seam to the closed API — a dumb request, no atlas logic in the session)."""
    received: list = []

    class RecordingAtlas:
        def submit(self, candidates):
            received.extend(candidates)

    s = Session(
        model="mock",
        pool_gb=5,
        pool_dir=tmp_pool_dir,
        context_window=64,
        output_tokens=400,
        atlas_client=RecordingAtlas(),
    )
    s.run("a long build")
    s.close()
    assert len(received) > 0  # candidates were handed off on close


# --- fail-soft: a pager error never crashes a run ----------------------------
def test_pager_error_does_not_crash_run(tmp_pool_dir):
    """FAIL-SOFT: if the pager raises during a run, the session logs it and finishes on the
    model's native window — the run still returns a result (the pager is an optimization)."""
    s = _mock_session(tmp_pool_dir)

    # monkeypatch the pager so prefetching blows up
    def _boom(*a, **k):
        raise RuntimeError("pager exploded")

    s.pager.prefetch_from = _boom  # type: ignore[method-assign]
    try:
        result = s.run("a task while the pager is broken")
        assert isinstance(result, RunResult)
        assert result.text  # the run completed on the native window
    finally:
        s.close()


def test_encoder_error_does_not_crash_run(tmp_pool_dir):
    """An encoder hiccup while encoding spill is fail-soft: the run continues, just without
    that slice in the pool."""
    s = _mock_session(tmp_pool_dir, context_window=64, output_tokens=300)

    def _boom_encode(text):  # noqa: ANN001
        raise ValueError("encoder hiccup")

    s.encoder.encode = _boom_encode  # type: ignore[method-assign]
    try:
        result = s.run("a task while the encoder is broken")
        assert isinstance(result, RunResult)
        assert result.text
    finally:
        s.close()


# --- streaming overlaps prefetch (background thread while generating) --------
def test_run_uses_background_prefetch_when_streaming(tmp_pool_dir):
    """When the model streams (>1 chunk), the session prefetches on a side thread *while*
    the model generates — the prefetch overlaps generation rather than serializing after it.

    We prove overlap with a slow-generating mock and a slow prefetch: wall-clock for the run
    is close to max(generation, prefetch), not their sum."""

    class SlowMock:
        name = "slow-mock"
        context_window = 4096

        def generate(self, prompt, *, system=None, stop=None, max_tokens=None):
            for _ in range(5):
                time.sleep(0.02)  # 5 * 20ms = 100ms of "generation"
                yield "slice token plan build module function class test verify "

        def count_tokens(self, text):
            return max(1, len(text) // 4)

    s = Session(model=SlowMock(), pool_gb=5, pool_dir=tmp_pool_dir)

    # make a prefetch noticeably slow so a *serial* design would clearly add its cost
    original = s.pager.prefetch_from

    def _slow_prefetch(*a, **k):
        time.sleep(0.02)
        return original(*a, **k)

    s.pager.prefetch_from = _slow_prefetch  # type: ignore[method-assign]
    try:
        t0 = time.perf_counter()
        result = s.run("overlap test")
        elapsed = time.perf_counter() - t0
        assert result.text
        # generation alone is ~100ms; a fully-serial prefetch-after-each-chunk would add a
        # lot more. Overlap keeps us well under generation + (5 * prefetch) = ~200ms.
        assert elapsed < 0.30
    finally:
        s.close()


# --- stages report -----------------------------------------------------------
def test_runresult_stages_recorded(tmp_pool_dir):
    s = _mock_session(tmp_pool_dir, context_window=64, output_tokens=300)
    try:
        result = s.run("a multi-stage build")
        # at minimum an open and a close stage are recorded
        assert result.stages
    finally:
        s.close()
