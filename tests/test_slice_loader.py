"""Tests for the B3 slice loader — the prefetch pager.

The pager is the **"pager" of the virtual-memory-for-attention design**: it keeps a small,
budget-bounded warm set of the slices the session is about to need so a hit is an O(1) memory
lookup, not a cold ANN search. The mechanism is ported from the upstream single-threaded
``SliceLoader`` (an LRU-budgeted warm cache with ``prefetch``/``get`` + hit-rate tracking);
the trading ``SliceKey(regime, setup, symbol, timeframe)`` is generalized to a plain discrete
``SliceKey(session, topic)`` and the cold path is injected as ``retrieve_fn = pool.search``.

All tests are numpy-only and never touch the network — the pool lives under pytest's
``tmp_path`` and randomness is seeded. The pager core is single-threaded and pure; the
re-probe is idle-aware (ported from ``exploration.py``) and the depth-cap-1 grounding verdict
is ported from ``latency_budget.py``. Concurrency belongs to the caller (the session), so it
is *not* exercised here.
"""
from __future__ import annotations

import numpy as np
import pytest

from aether_context.config import PoolConfig
from aether_context.context_pool import ContextPool, Slice
from aether_context.encoder import StaticEncoder
from aether_context.slice_loader import (
    DEFAULT_WARM_BUDGET,
    Grounding,
    Pager,
    SliceKey,
    grounding_verdict,
    reprobe_probability,
    should_reprobe,
)

DIM = 256


# --- helpers -----------------------------------------------------------------
def _unit(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v if n < 1e-12 else (v / n).astype(np.float32)


def _basis(i: int, dim: int = DIM) -> np.ndarray:
    """A unit vector along axis ``i`` (axis-i and axis-j are orthogonal for i != j)."""
    v = np.zeros(dim, dtype=np.float32)
    v[i % dim] = np.float32(1.0)
    return v


def make_slice(sid: str, *, session: str = "s1", vector: np.ndarray | None = None,
               text: str | None = None, tokens: int = 512, score: float = 0.5) -> Slice:
    if vector is None:
        vector = _basis(abs(hash(sid)) % DIM)
    return Slice(id=sid, session=session, vector=_unit(vector),
                 text=text if text is not None else f"text for {sid}",
                 tokens=tokens, meta={}, score=score)


def small_config(tmp_pool_dir, **kw) -> PoolConfig:
    params = dict(pool_gb=5, dim=DIM, index="flat", dir=tmp_pool_dir)
    params.update(kw)
    return PoolConfig(**params)


def make_pager(tmp_pool_dir, *, budget: int = DEFAULT_WARM_BUDGET, encoder=None):
    """A Pager over a fresh empty pool + a real StaticEncoder (counts cold calls)."""
    pool = ContextPool(small_config(tmp_pool_dir))
    enc = encoder if encoder is not None else StaticEncoder(dim=DIM)
    return Pager(pool, enc, budget=budget), pool, enc


class CountingPool:
    """A minimal pool stand-in that counts ``search`` calls (cold path observability).

    The Pager only needs ``search(query_vec, k, session=...)`` from the pool, so a tiny
    stub lets a test prove that a warm hit makes *zero* cold calls and a miss makes one.
    """

    def __init__(self, by_topic: dict[str, list[Slice]]):
        self._by_topic = by_topic  # the SliceKey.topic -> slices it should retrieve
        self.search_calls = 0
        self.last_query: np.ndarray | None = None

    def search(self, query_vec: np.ndarray, k: int, session=None) -> list[Slice]:
        self.search_calls += 1
        self.last_query = np.asarray(query_vec, dtype=np.float32)
        # deterministic stub: the closest axis index decides the topic bucket.
        idx = int(np.argmax(np.abs(self.last_query)))
        return self._by_topic.get(str(idx), [])[:k]


class StubEncoder:
    """Deterministic encoder stub: maps a known phrase to a known basis vector."""

    dim = DIM

    def __init__(self, mapping: dict[str, int]):
        self._mapping = mapping  # phrase substring -> axis index

    def encode(self, text: str) -> np.ndarray:
        for phrase, idx in self._mapping.items():
            if phrase in text:
                return _basis(idx)
        return _basis(0)


# --- SliceKey: discrete, hashable, generalized -------------------------------
def test_slice_key_is_discrete_session_topic_and_hashable():
    k = SliceKey(session="proj", topic="auth-refactor")
    assert k.session == "proj"
    assert k.topic == "auth-refactor"
    # hashable + usable as a dict key (warm-set requires this)
    assert {k: 1}[SliceKey("proj", "auth-refactor")] == 1


def test_slice_key_is_frozen():
    k = SliceKey(session="proj", topic="t")
    with pytest.raises(Exception):
        k.topic = "other"  # type: ignore[misc]


def test_slice_key_equality_by_value():
    assert SliceKey("a", "b") == SliceKey("a", "b")
    assert SliceKey("a", "b") != SliceKey("a", "c")


def test_slice_key_rejects_eight_dim_vector_and_regime_tuple():
    # MOAT: a closed low-dim coordinate / regime tuple must never address a slice.
    with pytest.raises(TypeError):
        SliceKey(session="proj", topic=np.zeros(8))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        SliceKey(session="proj", topic=("trend", "breakout", "NQ", "5m"))  # type: ignore[arg-type]


# --- warm hit is O(1), no cold call ------------------------------------------
def test_prefetched_key_is_a_warm_hit_with_no_cold_call():
    """The whole point: a key that was prefetched is served from the warm set without
    a second cold ``pool.search`` (warm O(1) vs cold miss)."""
    s = make_slice("a", session="proj", vector=_basis(0))
    pool = CountingPool({"0": [s]})
    enc = StubEncoder({"auth": 0})
    pager = Pager(pool, enc, budget=8)

    key = SliceKey("proj", "auth")
    pager.prefetch(key, "the auth module")          # one cold search to warm
    calls_after_prefetch = pool.search_calls
    got = pager.get(key)                              # warm -> O(1), no cold call

    assert [x.id for x in got] == ["a"]
    assert pool.search_calls == calls_after_prefetch  # get() did NOT hit the cold path
    assert pager.hits == 1 and pager.misses == 0


def test_cold_key_is_a_miss_then_warms():
    s = make_slice("b", session="proj", vector=_basis(1))
    pool = CountingPool({"1": [s]})
    enc = StubEncoder({"db": 1})
    pager = Pager(pool, enc, budget=8)

    key = SliceKey("proj", "db")
    got = pager.get(key, "the db layer")             # cold -> miss + opportunistic warm
    assert [x.id for x in got] == ["b"]
    assert pager.misses == 1 and pager.hits == 0
    calls_after_first = pool.search_calls

    again = pager.get(key)                            # now warm -> hit, no new cold call
    assert [x.id for x in again] == ["b"]
    assert pager.hits == 1
    assert pool.search_calls == calls_after_first


# --- hit-rate is measured, not assumed ---------------------------------------
def test_hit_rate_is_computed_from_hits_and_misses():
    s = make_slice("a", session="proj", vector=_basis(0))
    pool = CountingPool({"0": [s]})
    enc = StubEncoder({"x": 0})
    pager = Pager(pool, enc, budget=8)

    key = SliceKey("proj", "x")
    pager.prefetch(key, "x")
    for _ in range(9):
        pager.get(key)                                # 9 hits
    pager.get(SliceKey("proj", "miss"), "nope")       # 1 miss (empty bucket)
    assert abs(pager.hit_rate() - 0.9) < 1e-9


def test_hit_rate_zero_before_any_access():
    pool = CountingPool({})
    pager = Pager(pool, StubEncoder({}), budget=4)
    assert pager.hit_rate() == 0.0


# --- prefetch_from: embed -> search -> warm the expected slices --------------
def test_prefetch_from_warms_expected_slices_on_a_seeded_pool(tmp_pool_dir):
    """``prefetch_from(reasoning_text)`` embeds the text, searches the pool, and warms the
    nearest slices — so a later ``get`` for that region is a hit. Seeded real pool."""
    pager, pool, enc = make_pager(tmp_pool_dir, budget=8)
    # Two well-separated regions in embedding space, keyed by distinct phrases.
    auth_vec = enc.encode("authentication login session token")
    db_vec = enc.encode("database migration schema sql")
    pool.add(make_slice("auth_slice", session="s1", vector=auth_vec,
                        text="the auth module handles login"))
    pool.add(make_slice("db_slice", session="s1", vector=db_vec,
                        text="the db layer runs migrations"))

    key = SliceKey("s1", "auth")
    warmed = pager.prefetch_from(key, "now refactor the authentication login flow")
    # the auth slice is the nearest to auth-flavored reasoning -> it gets warmed
    assert any(s.id == "auth_slice" for s in warmed)
    # and a subsequent get for that key is a warm hit (no cold call)
    got = pager.get(key)
    assert any(s.id == "auth_slice" for s in got)
    assert pager.hits == 1


def test_prefetch_from_returns_resident_slices_in_window(tmp_pool_dir):
    pager, pool, enc = make_pager(tmp_pool_dir, budget=8)
    pool.add(make_slice("a", session="s1", vector=enc.encode("alpha beta gamma")))
    key = SliceKey("s1", "alpha")
    pager.prefetch_from(key, "alpha beta gamma delta")
    resident_ids = {s.id for s in pager.window()}
    assert "a" in resident_ids


def test_prefetch_from_session_scopes_the_search(tmp_pool_dir):
    """The cold search a key drives is scoped to ``key.session`` so a pager for one session
    never warms another session's slices (namespace isolation rides through the pager)."""
    pager, pool, enc = make_pager(tmp_pool_dir, budget=8)
    v = enc.encode("shared topic phrase")
    pool.add(make_slice("mine", session="A", vector=v))
    pool.add(make_slice("theirs", session="B", vector=v))
    pager.prefetch_from(SliceKey("A", "shared"), "shared topic phrase")
    resident = {s.id for s in pager.window()}
    assert "mine" in resident
    assert "theirs" not in resident


# --- window() reflects the resident warm set ---------------------------------
def test_window_returns_resident_slices(tmp_pool_dir):
    pager, pool, enc = make_pager(tmp_pool_dir, budget=8)
    pool.add(make_slice("a", session="s1", vector=enc.encode("first topic")))
    pool.add(make_slice("b", session="s1", vector=enc.encode("second wholly different")))
    pager.prefetch_from(SliceKey("s1", "first"), "first topic here")
    pager.prefetch_from(SliceKey("s1", "second"), "second wholly different here")
    ids = {s.id for s in pager.window()}
    assert {"a", "b"} <= ids


def test_window_empty_before_any_prefetch(tmp_pool_dir):
    pager, _, _ = make_pager(tmp_pool_dir, budget=4)
    assert pager.window() == []


# --- budget bounds the warm set + LRU eviction -------------------------------
def test_warm_set_never_exceeds_budget(tmp_pool_dir):
    pager, pool, enc = make_pager(tmp_pool_dir, budget=2)
    for i in range(5):
        v = _basis(i)
        pool.add(make_slice(f"s{i}", session="s1", vector=v))
        pager.prefetch(SliceKey("s1", f"t{i}"), f"axis {i} phrase", query_vec=v)
    assert pager.warm_count <= 2


def test_lru_eviction_drops_least_recently_used(tmp_pool_dir):
    pager, pool, enc = make_pager(tmp_pool_dir, budget=2)
    a, b, c = SliceKey("s1", "a"), SliceKey("s1", "b"), SliceKey("s1", "c")
    pager.prefetch(a, "a", query_vec=_basis(0))
    pager.prefetch(b, "b", query_vec=_basis(1))
    pager.get(a)                                   # touch a -> b is LRU
    pager.prefetch(c, "c", query_vec=_basis(2))    # evicts b (LRU, unprotected)
    assert pager.warm_count == 2
    assert pager.is_warm(a)
    assert not pager.is_warm(b)
    assert pager.is_warm(c)


def test_default_warm_budget_is_sixteen():
    assert DEFAULT_WARM_BUDGET == 16


# --- invalidate --------------------------------------------------------------
def test_invalidate_by_predicate_drops_matching_warm_keys(tmp_pool_dir):
    pager, pool, enc = make_pager(tmp_pool_dir, budget=8)
    pager.prefetch(SliceKey("A", "x"), "x", query_vec=_basis(0))
    pager.prefetch(SliceKey("B", "y"), "y", query_vec=_basis(1))
    dropped = pager.invalidate(lambda k: k.session == "A")
    assert dropped == 1
    assert pager.warm_count == 1
    assert not pager.is_warm(SliceKey("A", "x"))


# --- re-probe: idle-aware (ported from exploration.py) -----------------------
def test_reprobe_probability_rises_with_idle():
    """Just probed -> low; long idle -> approaches 1. Monotone non-decreasing in idle."""
    p0 = reprobe_probability(0.0)
    p_mid = reprobe_probability(20.0)
    p_long = reprobe_probability(200.0)
    assert p0 < p_mid < p_long
    assert p_long <= 1.0


def test_reprobe_probability_floor_just_after_a_probe():
    # right after a probe the probability is at its floor (base epsilon), never 0
    p0 = reprobe_probability(0.0, base_eps=0.02)
    assert abs(p0 - 0.02) < 1e-9
    assert p0 > 0.0


def test_should_reprobe_uses_the_probability_and_rng_draw():
    # a draw below the probability fires; a draw above it does not
    assert should_reprobe(200.0, rng_uniform=0.0) is True       # almost-certain region
    assert should_reprobe(0.0, rng_uniform=0.99) is False       # just-probed, high draw


def test_pager_should_reprobe_grows_with_idle_steps(tmp_pool_dir):
    """The pager exposes an idle-aware re-probe: probability rises the longer a key has gone
    unaccessed (so a stale-but-recoverable region gets re-checked)."""
    pager, pool, enc = make_pager(tmp_pool_dir, budget=8)
    key = SliceKey("s1", "x")
    pager.prefetch(key, "x", query_vec=_basis(0))
    # fresh access -> low re-probe probability
    p_fresh = pager.reprobe_probability(key)
    # advance idle by getting other keys (each get is a "period")
    for i in range(1, 60):
        pager.prefetch(SliceKey("s1", f"o{i}"), f"o{i}", query_vec=_basis(i % DIM))
        pager.get(SliceKey("s1", f"o{i}"))
    p_idle = pager.reprobe_probability(key)
    assert p_idle > p_fresh


# --- depth-cap-1 grounding verdict (ported from latency_budget.py) -----------
def test_grounding_passes_with_provenance_and_no_contradiction():
    assert grounding_verdict(has_provenance=True, contradicts_hard_fact=False) == Grounding.PASS


def test_grounding_flags_fabrication_without_provenance():
    # no provenance / derivation -> fabrication -> FLAG
    assert grounding_verdict(has_provenance=False, contradicts_hard_fact=False) == Grounding.FLAG


def test_grounding_flags_contradiction_of_a_hard_fact():
    assert grounding_verdict(has_provenance=True, contradicts_hard_fact=True) == Grounding.FLAG


def test_grounding_does_not_flag_mere_context_disagreement():
    # disagreeing with recent (possibly stale) context is NOT a flag when it has provenance
    # and does not contradict a HARD fact — that is how a regime/decision shift survives.
    assert grounding_verdict(has_provenance=True, contradicts_hard_fact=False) == Grounding.PASS


def test_pager_ground_slice_passes_a_provenanced_resident_slice(tmp_pool_dir):
    """A slice that is resident (came from the pool, so it has provenance) and does not
    contradict a hard fact grounds PASS — the pager never censors a paged-back fact."""
    pager, pool, enc = make_pager(tmp_pool_dir, budget=8)
    sl = make_slice("a", session="s1", vector=enc.encode("a load bearing fact"))
    pool.add(sl)
    got = pager.prefetch_from(SliceKey("s1", "fact"), "a load bearing fact")
    assert got  # warmed at least the one slice
    verdict = pager.ground(got[0], contradicts_hard_fact=False)
    assert verdict == Grounding.PASS


def test_pager_ground_flags_a_slice_that_contradicts_a_hard_fact(tmp_pool_dir):
    pager, pool, enc = make_pager(tmp_pool_dir, budget=8)
    sl = make_slice("a", session="s1", vector=enc.encode("contradictory claim"))
    pool.add(sl)
    got = pager.prefetch_from(SliceKey("s1", "claim"), "contradictory claim")
    verdict = pager.ground(got[0], contradicts_hard_fact=True)
    assert verdict == Grounding.FLAG


# --- fail-soft: a cold-path error degrades, never raises into the run --------
def test_get_on_search_failure_degrades_to_empty_not_raise(tmp_pool_dir):
    """The pager is an optimization, never a correctness dependency: if the injected cold
    search raises, ``get`` logs and returns an empty window rather than crashing the run."""
    class BoomPool:
        def search(self, query_vec, k, session=None):
            raise RuntimeError("ann backend exploded")

    pager = Pager(BoomPool(), StubEncoder({"x": 0}), budget=4)
    out = pager.get(SliceKey("s1", "x"), "x")     # must not raise
    assert out == []
    assert pager.misses == 1


def test_prefetch_from_on_encoder_failure_degrades(tmp_pool_dir):
    """An encoder hiccup during prefetch is fail-soft too: no warm, but no raise."""
    class BoomEncoder:
        dim = DIM

        def encode(self, text):
            raise ValueError("encoder hiccup")

    pager, pool, _ = make_pager(tmp_pool_dir, budget=4)
    pager_b = Pager(pool, BoomEncoder(), budget=4)
    out = pager_b.prefetch_from(SliceKey("s1", "x"), "anything")
    assert out == []
    assert pager_b.window() == []
