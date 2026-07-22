# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the public, session-scoped memory management surface.

The management API intentionally exposes text metadata rather than retrieval vectors. Every
mutation is optimistic and session-scoped, and durable mutations must be visible on disk before
they return. These tests stay hermetic under pytest's temporary directory.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
import json
from pathlib import Path
import threading
from uuid import UUID

import numpy as np
import pytest

from aether_context import MemoryPage, MemoryRecord, Session
from aether_context.config import PoolConfig
from aether_context.context_pool import ContextPool, Slice
from aether_context.errors import AetherContextError, MemoryConflict, MemoryNotFound
from aether_context.session import MEMORY_SOURCE_MODEL, MEMORY_SOURCE_USER


def _session(pool_dir: Path, session_id: str = "memory-test", **kwargs: object) -> Session:
    """Open a deterministic named session over an isolated pool directory."""
    return Session(
        "mock",
        pool_gb=5,
        pool_dir=pool_dir,
        pool_mode="separate",
        session_id=session_id,
        **kwargs,
    )


def _remember(session: Session, text: str, **kwargs: object):
    """Remember a fact and make the fail-soft return concrete for the tests below."""
    stored = session.remember(text, **kwargs)
    assert stored is not None
    return stored


def test_memory_page_get_and_type_filter_expose_vector_free_records(tmp_pool_dir: Path) -> None:
    session = _session(tmp_pool_dir)
    try:
        stored = [
            _remember(session, "first constraint", tags={"kind": "constraint"}),
            _remember(session, "implementation note", tags={"kind": "note"}),
            _remember(session, "second constraint", tags={"kind": "constraint"}),
        ]

        first = session.memory_page(limit=2)
        assert isinstance(first, MemoryPage)
        assert first.total == 3
        assert len(first.items) == 2
        assert first.next_cursor

        second = session.memory_page(limit=2, cursor=first.next_cursor)
        assert second.total == 3
        assert len(second.items) == 1
        assert second.next_cursor is None

        records = [*first.items, *second.items]
        assert {record.id for record in records} == {item.id for item in stored}
        assert len({record.id for record in records}) == 3
        public_fields = {field.name for field in fields(MemoryRecord)}
        assert {
            "id",
            "text",
            "tokens",
            "meta",
            "score",
            "version",
            "created_at",
            "updated_at",
        } <= public_fields
        assert public_fields.isdisjoint({"vector", "session"})
        assert all(isinstance(record, MemoryRecord) for record in records)

        fetched = session.memory_get(stored[1].id)
        assert fetched.id == stored[1].id
        assert fetched.text == "implementation note"

        constraints = session.memory_page(limit=10, memory_type="constraint")
        assert constraints.total == 2
        assert {record.meta["kind"] for record in constraints.items} == {"constraint"}
        assert {record.id for record in constraints.items} == {stored[0].id, stored[2].id}
    finally:
        session.close()


def test_memory_replace_increments_version_and_rejects_a_stale_writer(
    tmp_pool_dir: Path,
) -> None:
    session = _session(tmp_pool_dir)
    try:
        stored = _remember(
            session,
            "deploy to staging",
            source=MEMORY_SOURCE_USER,
            tags={"kind": "constraint", "owner": "ops"},
        )
        before = session.memory_get(stored.id)

        with pytest.raises(AetherContextError):
            session.memory_replace(
                stored.id,
                text="attempted provenance rewrite",
                expected_version=before.version,
                meta_patch={"source": MEMORY_SOURCE_MODEL, "_version": 99},
            )
        assert session.memory_get(stored.id) == before

        updated = session.memory_replace(
            stored.id,
            text="deploy to production after approval",
            expected_version=before.version,
            meta_patch={"kind": "decision", "owner": "release"},
        )

        assert updated.id == before.id
        assert updated.text == "deploy to production after approval"
        assert updated.version == before.version + 1
        assert updated.created_at == before.created_at
        assert updated.updated_at >= before.updated_at
        assert updated.meta["kind"] == "decision"
        assert updated.meta["owner"] == "release"
        assert updated.meta["source"] == MEMORY_SOURCE_USER

        sidecar = json.loads(session.pool.metadata_path.read_text(encoding="utf-8"))
        persisted = next(
            item for item in sidecar["slices"] if item["id"] == stored.id
        )
        assert persisted["text"] == updated.text
        assert persisted["meta"]["_version"] == updated.version
        with pytest.raises(MemoryConflict) as caught:
            session.memory_replace(
                stored.id,
                text="stale overwrite",
                expected_version=before.version,
            )
        assert caught.value.hint
        assert session.memory_get(stored.id).text == updated.text
    finally:
        session.close()


def test_memory_delete_is_durable_before_return_and_survives_reopen(
    tmp_pool_dir: Path,
) -> None:
    session_id = "durable-delete"
    session = _session(tmp_pool_dir, session_id)
    doomed = _remember(session, "delete me", tags={"kind": "note"})
    survivor = _remember(session, "keep me", tags={"kind": "constraint"})
    current = session.memory_get(doomed.id)

    deleted = session.memory_delete(
        doomed.id,
        expected_version=current.version,
        durable=True,
    )

    assert deleted.id == doomed.id
    assert deleted.version == current.version
    with pytest.raises(MemoryNotFound):
        session.memory_get(doomed.id)
    sidecar = json.loads(session.pool.metadata_path.read_text(encoding="utf-8"))
    assert {record["id"] for record in sidecar["slices"]} == {survivor.id}
    session.close()

    reopened = _session(tmp_pool_dir, session_id)
    try:
        with pytest.raises(MemoryNotFound):
            reopened.memory_get(doomed.id)
        assert reopened.memory_get(survivor.id).text == "keep me"
    finally:
        reopened.close()


def test_memory_clear_expected_count_is_atomic_and_durable(tmp_pool_dir: Path) -> None:
    session_id = "durable-clear"
    session = _session(tmp_pool_dir, session_id)
    first = _remember(session, "one")
    second = _remember(session, "two")

    with pytest.raises(MemoryConflict) as caught:
        session.memory_clear(expected_count=1, durable=True)
    assert caught.value.hint
    assert {record.id for record in session.memory_page(limit=10).items} == {
        first.id,
        second.id,
    }

    assert session.memory_clear(expected_count=2, durable=True) == 2
    assert session.memory_page(limit=10).total == 0
    sidecar = json.loads(session.pool.metadata_path.read_text(encoding="utf-8"))
    assert sidecar["count"] == 0
    assert sidecar["slices"] == []
    session.close()

    reopened = _session(tmp_pool_dir, session_id)
    try:
        assert reopened.memory_page(limit=10).total == 0
    finally:
        reopened.close()


def test_memory_management_never_crosses_session_scope_even_in_shared_mode(
    tmp_pool_dir: Path,
) -> None:
    session_a = Session(
        "mock",
        pool_gb=5,
        pool_dir=tmp_pool_dir,
        pool_mode="shared",
        session_id="owner-A",
    )
    owned_a = _remember(session_a, "A private managed record")
    session_a.close()

    session_b = Session(
        "mock",
        pool_gb=5,
        pool_dir=tmp_pool_dir,
        pool_mode="shared",
        session_id="owner-B",
    )
    try:
        assert session_b.memory_page(limit=10).total == 0
        with pytest.raises(MemoryNotFound):
            session_b.memory_get(owned_a.id)
        with pytest.raises(MemoryNotFound):
            session_b.memory_replace(owned_a.id, text="hijacked", expected_version=1)
        with pytest.raises(MemoryNotFound):
            session_b.memory_delete(owned_a.id, expected_version=1)
        assert session_b.memory_clear(expected_count=0) == 0

        owned_b = _remember(session_b, "B private managed record")
        assert {item.id for item in session_b.memory_page(limit=10).items} == {owned_b.id}
    finally:
        session_b.close()

    reopened_a = Session(
        "mock",
        pool_gb=5,
        pool_dir=tmp_pool_dir,
        pool_mode="shared",
        session_id="owner-A",
    )
    try:
        assert reopened_a.memory_get(owned_a.id).text == "A private managed record"
        assert {item.id for item in reopened_a.memory_page(limit=10).items} == {owned_a.id}
        with pytest.raises(MemoryNotFound):
            reopened_a.memory_get(owned_b.id)
    finally:
        reopened_a.close()


def test_separate_mode_recall_does_not_fall_back_to_global_pool(tmp_pool_dir: Path) -> None:
    owner = _session(tmp_pool_dir, "recall-owner")
    _remember(owner, "OMEGA private deployment secret")
    owner.close()

    stranger = _session(tmp_pool_dir, "recall-stranger")
    try:
        assert stranger.recall("OMEGA private deployment secret", k=4) == []
    finally:
        stranger.close()


def test_same_named_session_uses_uuid_ids_without_overwriting_after_restart(
    tmp_pool_dir: Path,
) -> None:
    session_id = "stable-session"
    first_session = _session(tmp_pool_dir, session_id)
    first = _remember(first_session, "fact from the first process")
    first_session.close()

    second_session = _session(tmp_pool_dir, session_id)
    second = _remember(second_session, "fact from the second process")
    try:
        assert first.id != second.id
        UUID(first.id.rsplit(":", 1)[-1])
        UUID(second.id.rsplit(":", 1)[-1])
        page = second_session.memory_page(limit=10)
        assert page.total == 2
        assert {item.text for item in page.items} == {
            "fact from the first process",
            "fact from the second process",
        }
    finally:
        second_session.close()


def test_remember_tags_cannot_spoof_provenance_or_version(tmp_pool_dir: Path) -> None:
    session = _session(tmp_pool_dir, "provenance")
    try:
        stored = _remember(
            session,
            "model-authored production claim",
            source=MEMORY_SOURCE_MODEL,
            tags={
                "kind": "constraint",
                "source": MEMORY_SOURCE_USER,
                "_version": 500,
                "_created_at": "attacker-controlled",
                "_updated_at": "attacker-controlled",
            },
        )
        record = session.memory_get(stored.id)

        assert stored.meta["source"] == MEMORY_SOURCE_MODEL
        assert record.meta["source"] == MEMORY_SOURCE_MODEL
        assert record.version == 1
        assert record.created_at != "attacker-controlled"
        assert record.updated_at != "attacker-controlled"
        assert {
            item.id
            for item in session.recall(
                "model-authored production claim",
                k=4,
                sources={MEMORY_SOURCE_USER},
            )
        } == set()
        assert {
            item.id
            for item in session.recall(
                "model-authored production claim",
                k=4,
                sources={MEMORY_SOURCE_MODEL},
            )
        } == {stored.id}
    finally:
        session.close()


@pytest.mark.parametrize("quantize_bits", [0, 4, 8])
def test_delete_and_clear_zero_every_freed_mmap_row(
    tmp_path: Path,
    quantize_bits: int,
) -> None:
    session = _session(
        tmp_path / f"q{quantize_bits}",
        f"zero-q{quantize_bits}",
        pool_quantize=quantize_bits,
    )
    try:
        records = [_remember(session, f"nonzero vector {index}") for index in range(3)]
        old_count = len(session.pool)
        assert old_count == 3

        middle = session.memory_get(records[1].id)
        session.memory_delete(
            middle.id,
            expected_version=middle.version,
            durable=True,
        )
        assert session.pool._mmap is not None
        assert np.count_nonzero(np.asarray(session.pool._mmap[len(session.pool):old_count])) == 0

        assert session.memory_clear(expected_count=2, durable=True) == 2
        assert np.count_nonzero(np.asarray(session.pool._mmap[:old_count])) == 0
    finally:
        session.close()

    disk_dtype = np.uint8 if quantize_bits else np.float32
    raw_rows = np.fromfile(session.pool.vectors_path, dtype=disk_dtype)
    assert np.count_nonzero(raw_rows) == 0


def test_two_concurrent_replacements_have_exactly_one_winner(tmp_pool_dir: Path) -> None:
    session = _session(tmp_pool_dir, "optimistic-race")
    stored = _remember(session, "original")
    version = session.memory_get(stored.id).version
    barrier = threading.Barrier(3)

    def replace(text: str) -> tuple[str, object]:
        barrier.wait(timeout=5)
        try:
            return (
                "updated",
                session.memory_replace(stored.id, text=text, expected_version=version),
            )
        except MemoryConflict as exc:
            return "conflict", exc

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(replace, text) for text in ("writer one", "writer two")]
            barrier.wait(timeout=5)
            outcomes = [future.result(timeout=10) for future in futures]

        assert [status for status, _ in outcomes].count("updated") == 1
        assert [status for status, _ in outcomes].count("conflict") == 1
        winner = next(value for status, value in outcomes if status == "updated")
        assert isinstance(winner, MemoryRecord)
        assert winner.version == version + 1
        assert session.memory_get(stored.id).text == winner.text
    finally:
        session.close()


def test_pool_add_and_search_are_safe_under_basic_concurrency(tmp_path: Path) -> None:
    dim = 256
    pool = ContextPool(PoolConfig(pool_gb=5, dim=dim, index="flat", dir=tmp_path / "pool"))
    barrier = threading.Barrier(5)

    def writer(offset: int) -> None:
        barrier.wait(timeout=5)
        for index in range(20):
            vector = np.zeros(dim, dtype=np.float32)
            vector[(offset + index) % dim] = 1.0
            pool.add(
                Slice(
                    id=f"slice-{offset + index}",
                    session="concurrent",
                    vector=vector,
                    text=f"text {offset + index}",
                    tokens=2,
                    meta={"source": MEMORY_SOURCE_USER},
                    score=0.8,
                )
            )

    def reader(axis: int) -> None:
        query = np.zeros(dim, dtype=np.float32)
        query[axis] = 1.0
        barrier.wait(timeout=5)
        for _ in range(50):
            for hit in pool.search(query, k=5, session="concurrent"):
                assert hit.session == "concurrent"
                assert hit.vector.shape == (dim,)

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(writer, 0),
                executor.submit(writer, 20),
                executor.submit(reader, 0),
                executor.submit(reader, 20),
            ]
            barrier.wait(timeout=5)
            for future in futures:
                future.result(timeout=15)
        assert len(pool) == 40
    finally:
        pool.close()
