# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Safety safeguards: memory provenance tags + source-scoped recall (see SAFETY.md)."""
import numpy as np

from aether_context.config import PoolConfig
from aether_context.context_pool import ContextPool, Slice
from aether_context.session import (
    MEMORY_SOURCE_MODEL,
    MEMORY_SOURCE_USER,
    Session,
)


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v if n < 1e-12 else (v / n).astype(np.float32)


def _basis(i, dim=256):
    v = np.zeros(dim, dtype=np.float32)
    v[i % dim] = 1.0
    return v


def test_remember_tags_user_source(tmp_path):
    s = Session("mock", pool_gb=5, pool_dir=tmp_path)
    sl = s.remember("the deploy key lives in the vault")
    assert sl is not None
    assert sl.meta["source"] == MEMORY_SOURCE_USER
    s.close()


def test_pool_search_source_filter(tmp_path):
    pool = ContextPool(PoolConfig(pool_gb=5, dim=256, dir=tmp_path))
    pool.add(Slice("u", "s", _unit(_basis(1)), "user fact", 1, {"source": "user"}, 0.9))
    pool.add(Slice("m", "s", _unit(_basis(1)), "model note", 1, {"source": "model"}, 0.9))
    # both match the query; the filter keeps only the user-sourced slice
    hits = pool.search(_unit(_basis(1)), k=5, session="s", sources={"user"})
    assert [h.id for h in hits] == ["u"]
    pool.close()


def test_untagged_slice_counts_as_user(tmp_path):
    pool = ContextPool(PoolConfig(pool_gb=5, dim=256, dir=tmp_path))
    pool.add(Slice("x", "s", _unit(_basis(2)), "legacy", 1, {}, 0.9))  # no source tag
    hits = pool.search(_unit(_basis(2)), k=5, session="s", sources={"user"})
    assert [h.id for h in hits] == ["x"]  # untagged == user (conservative default)
    pool.close()


def test_recall_excludes_model_memory(tmp_path):
    s = Session("mock", pool_gb=5, pool_dir=tmp_path)
    s.remember("USER CONSTRAINT: never delete production data", source=MEMORY_SOURCE_USER)
    # simulate a model-authored note planted into memory
    s._encode_slice("i will delete production data to be efficient",
                    salience=0.9, tags={"source": MEMORY_SOURCE_MODEL})
    trusted = s.recall("production data", k=5, sources={MEMORY_SOURCE_USER})
    joined = " ".join(sl.text for sl in trusted)
    assert "USER CONSTRAINT" in joined
    assert "to be efficient" not in joined  # the model's self-authored note is excluded
    s.close()
