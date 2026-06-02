"""Tests for the B2 context pool — session-namespaced mmap'd vector store + governor.

The context pool is the "disk" of the virtual-memory-for-attention design: encoded
:class:`Slice` payloads land here, vectors live in an mmap'd file (disk-resident), and a
budget governor evicts the lowest-retention slices (via the :class:`Witness`) so the pool
never exceeds its byte ceiling. Search is cosine nearest-neighbor with a session/namespace
filter so far-apart sessions never bleed into each other.

All tests are numpy-only and never touch the network or the user's real ``~/.aether-context``
home directory — pool state lives under pytest's ``tmp_path`` and randomness is seeded. The
``flat`` numpy index path is the one exercised in CI (hnswlib is an optional extra); a
``flat == hnsw`` fixture test runs only when hnswlib is importable.
"""
from __future__ import annotations

import numpy as np
import pytest

from aether_context.config import PoolConfig
from aether_context.context_pool import ContextPool, Slice
from aether_context.errors import PoolCorrupt


DIM = 256


# --- helpers -----------------------------------------------------------------
def _unit(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector to float32 unit length."""
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v if n < 1e-12 else (v / n).astype(np.float32)


def _basis(i: int, dim: int = DIM, *, sign: float = 1.0) -> np.ndarray:
    """A unit vector along axis ``i`` (so axis-i and axis-j are orthogonal for i != j)."""
    v = np.zeros(dim, dtype=np.float32)
    v[i % dim] = np.float32(sign)
    return v


def make_slice(
    sid: str,
    *,
    session: str = "s1",
    vector: np.ndarray | None = None,
    text: str | None = None,
    tokens: int = 512,
    score: float = 0.5,
    meta: dict | None = None,
) -> Slice:
    """Build a Slice with sensible defaults for tests."""
    if vector is None:
        vector = _basis(abs(hash(sid)) % DIM)
    return Slice(
        id=sid,
        session=session,
        vector=_unit(vector),
        text=text if text is not None else f"text for {sid}",
        tokens=tokens,
        meta=meta if meta is not None else {},
        score=score,
    )


def small_config(tmp_pool_dir, **kw) -> PoolConfig:
    """A valid PoolConfig pointed at the tmp pool dir (pool_gb at the 5 GB floor)."""
    params = dict(pool_gb=5, dim=DIM, index="flat", dir=tmp_pool_dir)
    params.update(kw)
    return PoolConfig(**params)


def _slice_cost(cfg: PoolConfig) -> int:
    """Per-slice byte cost as the pool charges it (vector bytes + a fixed payload est).

    Imported lazily from the implementation so the test charges exactly what the pool
    charges (keeps the 'never exceeds budget' assertion honest)."""
    from aether_context.context_pool import slice_cost_bytes

    return slice_cost_bytes(cfg.dim)


def _unit_matrix(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalize a matrix to float32 unit rows."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return (mat / norms).astype(np.float32)


# --- Slice dataclass ---------------------------------------------------------
def test_slice_carries_the_documented_fields():
    s = make_slice("a", session="proj", text="hello", tokens=7, score=0.9,
                   meta={"phase": "plan"})
    assert s.id == "a"
    assert s.session == "proj"
    assert s.text == "hello"
    assert s.tokens == 7
    assert s.score == 0.9
    assert s.meta == {"phase": "plan"}
    assert s.vector.shape == (DIM,)
    assert s.vector.dtype == np.float32


def test_slice_vector_is_self_contained():
    # A Slice is a plain dataclass over the 256-dim retrieval embedding and a plain meta dict —
    # nothing else. The vector is exactly `dim` wide.
    s = make_slice("a")
    assert s.vector.shape == (DIM,)
    assert isinstance(s.meta, dict)


# --- add / search basics -----------------------------------------------------
def test_add_then_search_returns_nearest_by_cosine(tmp_pool_dir):
    pool = ContextPool(small_config(tmp_pool_dir))
    near = make_slice("near", vector=_basis(0))
    far = make_slice("far", vector=_basis(1))
    pool.add(near)
    pool.add(far)
    # query points along axis 0 -> 'near' (axis 0) is the nearest by cosine
    hits = pool.search(_basis(0), k=1)
    assert len(hits) == 1
    assert hits[0].id == "near"


def test_search_orders_results_by_descending_cosine(tmp_pool_dir):
    pool = ContextPool(small_config(tmp_pool_dir))
    # three slices at increasing angle from the query along axis 0
    pool.add(make_slice("aligned", vector=_unit(np.array([1.0, 0.0, 0.0] + [0.0] * (DIM - 3)))))
    pool.add(make_slice("mid", vector=_unit(np.array([1.0, 1.0, 0.0] + [0.0] * (DIM - 3)))))
    pool.add(make_slice("off", vector=_unit(np.array([1.0, 5.0, 0.0] + [0.0] * (DIM - 3)))))
    hits = pool.search(_basis(0), k=3)
    assert [h.id for h in hits] == ["aligned", "mid", "off"]


def test_search_k_caps_result_count(tmp_pool_dir):
    pool = ContextPool(small_config(tmp_pool_dir))
    for i in range(10):
        pool.add(make_slice(f"s{i}", vector=_basis(i)))
    hits = pool.search(_basis(0), k=3)
    assert len(hits) == 3


def test_search_on_empty_pool_returns_empty_list(tmp_pool_dir):
    pool = ContextPool(small_config(tmp_pool_dir))
    assert pool.search(_basis(0), k=5) == []


def test_search_returned_slices_round_trip_text_and_meta(tmp_pool_dir):
    pool = ContextPool(small_config(tmp_pool_dir))
    pool.add(make_slice("a", vector=_basis(0), text="load-bearing fact",
                         meta={"tag": "fact"}, tokens=42))
    hit = pool.search(_basis(0), k=1)[0]
    assert hit.text == "load-bearing fact"
    assert hit.meta == {"tag": "fact"}
    assert hit.tokens == 42


# --- session / namespace isolation -------------------------------------------
def test_session_filter_isolates_far_apart_sessions(tmp_pool_dir):
    """A search scoped to one session never returns another session's slices, even when
    the other session has a vector that is a *closer* cosine match."""
    pool = ContextPool(small_config(tmp_pool_dir))
    # both sit on axis 0, but they belong to different sessions
    pool.add(make_slice("mine", session="A", vector=_basis(0)))
    pool.add(make_slice("theirs", session="B", vector=_basis(0)))
    hits = pool.search(_basis(0), k=5, session="A")
    ids = {h.id for h in hits}
    assert "mine" in ids
    assert "theirs" not in ids


def test_search_without_session_filter_spans_all_sessions(tmp_pool_dir):
    pool = ContextPool(small_config(tmp_pool_dir))
    pool.add(make_slice("a", session="A", vector=_basis(0)))
    pool.add(make_slice("b", session="B", vector=_basis(1)))
    hits = pool.search(_basis(0), k=5)  # no session -> all sessions visible
    assert {h.id for h in hits} == {"a", "b"}


def test_session_filter_empty_when_session_unknown(tmp_pool_dir):
    pool = ContextPool(small_config(tmp_pool_dir))
    pool.add(make_slice("a", session="A", vector=_basis(0)))
    assert pool.search(_basis(0), k=5, session="ghost") == []


# --- budget governor: never exceeds budget -----------------------------------
def test_pool_never_exceeds_byte_budget(tmp_pool_dir):
    """The whole point of the governor: after every add the pool sits at or below its
    byte ceiling. We use a tiny ceiling so a handful of adds forces eviction."""
    cfg = small_config(tmp_pool_dir)
    # tiny explicit ceiling so the test is fast and deterministic
    pool = ContextPool(cfg, ceiling_bytes=5 * _slice_cost(cfg))
    for i in range(50):
        pool.add(make_slice(f"s{i}", vector=_basis(i), score=(i + 1) / 50.0))
        assert pool.bytes_used() <= pool.ceiling_bytes


def test_evict_to_budget_drops_lowest_retention_first(tmp_pool_dir):
    cfg = small_config(tmp_pool_dir)
    pool = ContextPool(cfg, ceiling_bytes=2 * _slice_cost(cfg))
    pool.add(make_slice("keep_hi", vector=_basis(0), score=0.95))
    pool.add(make_slice("keep_mid", vector=_basis(1), score=0.60))
    pool.add(make_slice("drop_lo", vector=_basis(2), score=0.05))
    pool.evict_to_budget()
    remaining = {h.id for h in pool.search(_basis(0), k=10)} | \
                {h.id for h in pool.search(_basis(1), k=10)} | \
                {h.id for h in pool.search(_basis(2), k=10)}
    assert "drop_lo" not in remaining
    assert "keep_hi" in remaining


def test_evicted_slice_is_not_returned_by_search(tmp_pool_dir):
    cfg = small_config(tmp_pool_dir)
    pool = ContextPool(cfg, ceiling_bytes=1 * _slice_cost(cfg))
    pool.add(make_slice("first", vector=_basis(0), score=0.1))
    pool.add(make_slice("second", vector=_basis(0), score=0.9))  # higher retention
    # only room for one: the low-score 'first' must be gone
    hits = pool.search(_basis(0), k=5)
    assert [h.id for h in hits] == ["second"]


# --- bytes_used / stats ------------------------------------------------------
def test_bytes_used_grows_with_added_slices(tmp_pool_dir):
    pool = ContextPool(small_config(tmp_pool_dir))
    base = pool.bytes_used()
    pool.add(make_slice("a", vector=_basis(0)))
    after_one = pool.bytes_used()
    pool.add(make_slice("b", vector=_basis(1)))
    after_two = pool.bytes_used()
    assert after_one > base
    assert after_two > after_one


def test_stats_reports_count_bytes_and_sessions(tmp_pool_dir):
    pool = ContextPool(small_config(tmp_pool_dir))
    pool.add(make_slice("a", session="A", vector=_basis(0)))
    pool.add(make_slice("b", session="B", vector=_basis(1)))
    st = pool.stats()
    assert st["count"] == 2
    assert st["bytes_used"] == pool.bytes_used()
    assert set(st["sessions"]) == {"A", "B"}
    assert st["dim"] == DIM
    assert st["index"] in ("flat", "hnsw")


# --- persistence: survives reopen --------------------------------------------
def test_pool_survives_reopen_same_dir(tmp_pool_dir):
    """Create → add → close, then reopen the same dir: every slice and its payload come
    back and search returns the same nearest neighbor (mmap persistence)."""
    cfg = small_config(tmp_pool_dir)
    pool = ContextPool(cfg)
    pool.add(make_slice("alpha", session="A", vector=_basis(0), text="persisted",
                         meta={"k": "v"}, tokens=11, score=0.7))
    pool.add(make_slice("beta", session="A", vector=_basis(1), text="other", score=0.3))
    pool.close()

    reopened = ContextPool(small_config(tmp_pool_dir))
    assert reopened.stats()["count"] == 2
    hit = reopened.search(_basis(0), k=1)[0]
    assert hit.id == "alpha"
    assert hit.text == "persisted"
    assert hit.meta == {"k": "v"}
    assert hit.tokens == 11
    np.testing.assert_allclose(hit.vector, _basis(0), atol=1e-6)


def test_reopened_search_matches_pre_close_search(tmp_pool_dir):
    cfg = small_config(tmp_pool_dir)
    pool = ContextPool(cfg)
    for i in range(8):
        pool.add(make_slice(f"s{i}", vector=_basis(i), score=0.5))
    before = [h.id for h in pool.search(_basis(3), k=4)]
    pool.close()

    reopened = ContextPool(small_config(tmp_pool_dir))
    after = [h.id for h in reopened.search(_basis(3), k=4)]
    assert before == after


def test_reopen_preserves_session_isolation(tmp_pool_dir):
    cfg = small_config(tmp_pool_dir)
    pool = ContextPool(cfg)
    pool.add(make_slice("mine", session="A", vector=_basis(0)))
    pool.add(make_slice("theirs", session="B", vector=_basis(0)))
    pool.close()

    reopened = ContextPool(small_config(tmp_pool_dir))
    ids = {h.id for h in reopened.search(_basis(0), k=5, session="A")}
    assert ids == {"mine"}


def test_reopen_corrupt_metadata_raises_pool_corrupt(tmp_pool_dir):
    cfg = small_config(tmp_pool_dir)
    pool = ContextPool(cfg)
    pool.add(make_slice("a", vector=_basis(0)))
    meta_path = pool.metadata_path
    pool.close()
    # scribble garbage over the sidecar metadata
    meta_path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(PoolCorrupt):
        ContextPool(small_config(tmp_pool_dir))


# --- flat top-k matches a known fixture --------------------------------------
def test_flat_topk_matches_known_fixture(rng, tmp_path):
    """The flat numpy brute-force index must return the exact top-k a hand-computed
    cosine ranking gives — this is the ground-truth correctness fixture."""
    vectors = _unit_matrix(rng.standard_normal((6, DIM)).astype(np.float32))
    query = _unit(rng.standard_normal(DIM).astype(np.float32))
    # ground-truth cosine ranking (vectors are unit, so cosine == dot)
    cosines = vectors @ query
    expected_order = [int(i) for i in np.argsort(-cosines)]

    cfg = PoolConfig(pool_gb=5, dim=DIM, index="flat", dir=tmp_path / "pool")
    pool = ContextPool(cfg)
    for i in range(6):
        pool.add(make_slice(f"v{i}", vector=vectors[i], score=0.5))
    hits = pool.search(query, k=6)
    got_order = [int(h.id[1:]) for h in hits]
    assert got_order == expected_order


# --- flat == hnsw when hnswlib is available ----------------------------------
def test_flat_and_hnsw_agree_on_topk(rng, tmp_path):
    """When hnswlib is installed, the hnsw index must agree with the flat brute-force
    index on the top-k for a fixed seed. Skipped cleanly when hnswlib is absent."""
    pytest.importorskip("hnswlib")
    vectors = _unit_matrix(rng.standard_normal((20, DIM)).astype(np.float32))
    query = _unit(rng.standard_normal(DIM).astype(np.float32))

    flat = ContextPool(PoolConfig(pool_gb=5, dim=DIM, index="flat", dir=tmp_path / "flat"))
    hnsw = ContextPool(PoolConfig(pool_gb=5, dim=DIM, index="hnsw", dir=tmp_path / "hnsw"))
    for i in range(20):
        sl = make_slice(f"v{i}", vector=vectors[i], score=0.5)
        flat.add(sl)
        hnsw.add(Slice(id=sl.id, session=sl.session, vector=sl.vector.copy(),
                       text=sl.text, tokens=sl.tokens, meta=dict(sl.meta),
                       score=sl.score))
    flat_ids = [h.id for h in flat.search(query, k=5)]
    hnsw_ids = [h.id for h in hnsw.search(query, k=5)]
    assert flat_ids == hnsw_ids


# --- graceful fallback when hnsw requested but lib missing --------------------
def test_hnsw_request_without_lib_falls_back_to_flat(tmp_pool_dir, monkeypatch):
    """If config.index == 'hnsw' but hnswlib is not importable, the pool must fall back
    to the flat index and still work — never hard-fail on a missing optional dep."""
    import aether_context.context_pool as cp

    monkeypatch.setattr(cp, "_HNSWLIB_AVAILABLE", False, raising=False)
    pool = ContextPool(small_config(tmp_pool_dir, index="hnsw"))
    assert pool.index_kind == "flat"
    pool.add(make_slice("a", vector=_basis(0)))
    assert pool.search(_basis(0), k=1)[0].id == "a"
