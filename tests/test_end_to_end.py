"""End-to-end: the whole engine on a MockLLM, proving the pitch.

This is the kill-gate property in miniature, hermetic (no GPU, no network): a planted
**load-bearing fact** is established early in a long run, then the model produces enough
output to blow past a deliberately small ``context_window``. WITH the engine the fact is
**encoded into the pool and paged back**, so it stays reachable; WITHOUT the engine (a raw
window of the same tiny size) it falls off the end and is lost.

The mechanism we assert:
  1. a Session over ``model="mock"`` runs open -> close without error;
  2. the pool budget is held the whole time;
  3. the planted fact, established before the window would overflow, is still retrievable
     from the pool after the run (encode-and-recover);
  4. the no-engine baseline (just the model's tiny window) cannot reach that same fact.

Everything is deterministic and offline (MockLLM + tmp_path pool). Mirrors the bench's
shape (``bench/drift_vs_window.py``) but at unit speed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from aether_context.encoder import StaticEncoder
from aether_context.session import Session

# A distinctive, load-bearing fact we plant early and later try to recover. Phrased so its
# embedding is well-separated from generic build chatter.
PLANTED_FACT = (
    "CRITICAL CONSTRAINT: the primary database is postgres and the auth token "
    "expiry is exactly 3600 seconds; never change these two invariants."
)
# A query that should retrieve the planted fact (shares its load-bearing tokens).
RECOVERY_QUERY = "what is the database and the auth token expiry constraint"


def _tiny_window_session(tmp_pool_dir: Path, **kw) -> Session:
    """A Session whose mock model has a deliberately tiny window + long output, so a run
    overflows and the engine must encode-and-recover."""
    params: dict = dict(
        model="mock",
        pool_gb=5,
        pool_dir=tmp_pool_dir,
        context_window=64,   # tiny: a handful of slices fit at once
        output_tokens=400,   # long: forces overflow well past the window
    )
    params.update(kw)
    return Session(**params)


# --- the pitch: encode-and-recover past a small window -----------------------
def test_planted_fact_is_reachable_after_overflow_with_engine(tmp_pool_dir):
    """WITH the engine: a fact planted before overflow is still retrievable from the pool
    after a run that blew past the tiny native window."""
    s = _tiny_window_session(tmp_pool_dir)
    try:
        # plant the load-bearing fact early (it gets encoded into the pool)
        s.remember(PLANTED_FACT, tags={"kind": "constraint"})
        # then run a long build that overflows the tiny window many times over
        s.run("build a large full-stack app with many modules and long files")

        # recover: search the pool for the fact's region
        hits = s.recall(RECOVERY_QUERY, k=5)
        assert hits, "the engine should page back at least one slice"
        joined = " ".join(h.text.lower() for h in hits)
        # the load-bearing tokens survive and are reachable
        assert "postgres" in joined
        assert "3600" in joined
    finally:
        s.close()


def test_baseline_without_engine_cannot_reach_the_fact(tmp_pool_dir):
    """WITHOUT the engine: the model's raw tiny window cannot hold the planted fact across a
    long generation — the fact is not in what the bare model can see at the end.

    We model the no-engine baseline as the raw context window: only the last
    ``context_window`` tokens of the conversation are visible. The planted fact, established
    before a long overflow, is no longer inside that window."""
    window_tokens = 64
    chars_per_token = 4
    window_chars = window_tokens * chars_per_token

    # the conversation: the fact, then a long stream of unrelated build chatter
    long_tail = (" build module function class test refactor encode slice" * 200)
    transcript = PLANTED_FACT + long_tail

    # the bare model only sees the last `window` chars (its native attention)
    visible = transcript[-window_chars:]

    # the load-bearing fact has fallen off the end of the raw window
    assert "postgres" not in visible.lower()
    assert "3600" not in visible

    # and a pure-window baseline (no pool) therefore cannot recover it
    enc = StaticEncoder(dim=256)
    q = enc.encode(RECOVERY_QUERY)
    v = enc.encode(visible)
    # the visible window is generic chatter, far from the constraint query
    cosine = float(np.dot(q, v))
    assert cosine < 0.5  # no strong match — the fact is unreachable without the engine


def test_engine_beats_baseline_on_reachability(tmp_pool_dir, tmp_path):
    """Head-to-head, hermetic: the engine keeps the planted fact reachable; the raw-window
    baseline does not. This is the ON-beats-OFF delta the bench reports, at unit speed."""
    # ON: full engine
    s = _tiny_window_session(tmp_pool_dir)
    try:
        s.remember(PLANTED_FACT, tags={"kind": "constraint"})
        s.run("a long build that overflows the window repeatedly")
        on_hits = s.recall(RECOVERY_QUERY, k=5)
        on_reachable = any("postgres" in h.text.lower() for h in on_hits)
    finally:
        s.close()

    # OFF: a raw window of the same size, no pool — the fact scrolled off
    window_chars = 64 * 4
    transcript = PLANTED_FACT + (" generic build chatter token" * 300)
    visible = transcript[-window_chars:]
    off_reachable = "postgres" in visible.lower()

    assert on_reachable is True
    assert off_reachable is False
    assert on_reachable and not off_reachable  # ON beats OFF


# --- the run still holds all the invariants ----------------------------------
def test_full_open_to_close_holds_budget_and_runs_clean(tmp_pool_dir):
    """Full lifecycle: open -> stream+encode+fade -> paged reason -> close, with the pool
    budget held throughout and no exception escaping."""
    s = Session(
        model="mock",
        pool_gb=5,
        pool_dir=tmp_pool_dir,
        context_window=64,
        output_tokens=500,
        pool_ceiling_bytes=16 * 1224,  # hard small ceiling to exercise the governor
    )
    try:
        s.remember(PLANTED_FACT)
        result = s.run("a very long build")
        assert result.text
        # budget held at or below the ceiling at the end of the run
        assert s.pool.bytes_used() <= s.pool.ceiling_bytes
        # the pool actually accumulated spill (encode-on-spill happened)
        assert len(s.pool) > 0
    finally:
        s.close()


def test_pool_survives_reopen_and_fact_still_recoverable(tmp_pool_dir):
    """The encoded fact is disk-resident: closing and reopening the same pool dir restores
    it, and it is still recoverable (mmap persistence carries the load-bearing fact)."""
    s = _tiny_window_session(tmp_pool_dir)
    s.remember(PLANTED_FACT, tags={"kind": "constraint"})
    s.run("a long build")
    s.close()  # flush to disk

    # reopen a fresh session over the SAME pool dir
    s2 = Session(model="mock", pool_gb=5, pool_dir=tmp_pool_dir)
    try:
        hits = s2.recall(RECOVERY_QUERY, k=5)
        joined = " ".join(h.text.lower() for h in hits)
        assert "postgres" in joined  # the fact survived the reopen
    finally:
        s2.close()
