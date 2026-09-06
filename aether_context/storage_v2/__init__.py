"""Encrypted append-only segments using SQLite's multiprocess WAL transaction.

FULL synchronization commits the segment, HMAC lexical postings, manifest CAS,
event and idempotency result together. No text or embedding dictionaries exist.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
import sqlite3
import threading

from ..crypto import ContextFault

SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
 id TEXT PRIMARY KEY, owner TEXT NOT NULL, project TEXT NOT NULL,
 request_key TEXT NOT NULL, request_digest TEXT NOT NULL, profile TEXT NOT NULL,
 state TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1, fence INTEGER NOT NULL DEFAULT 1,
 binding TEXT, binding_digest TEXT, authority BLOB, snapshot BLOB,
 cursor INTEGER NOT NULL DEFAULT 0, root TEXT NOT NULL,
 bytes INTEGER NOT NULL DEFAULT 0, expires INTEGER NOT NULL,
 hold INTEGER NOT NULL DEFAULT 0, checkpoint INTEGER NOT NULL DEFAULT 0,
 key_ref TEXT, key_version TEXT, retention_class TEXT NOT NULL DEFAULT 'ephemeral',
 hold_reason TEXT, cleanup_state TEXT NOT NULL DEFAULT 'NONE',
 seal_digest TEXT, checkpoint_manifest_checksum TEXT, checkpoint_object_ref_digest TEXT,
 cleanup_remote_receipt TEXT, cleanup_operation_key TEXT, cleanup_request_digest TEXT,
 deletion_receipt TEXT,
 UNIQUE(owner,project,request_key));
CREATE TABLE IF NOT EXISTS segments (
 cycle TEXT NOT NULL REFERENCES cycles(id), seq INTEGER NOT NULL,
 id TEXT NOT NULL UNIQUE, plane TEXT NOT NULL, lane TEXT NOT NULL,
 metadata TEXT NOT NULL, payload BLOB NOT NULL, digest TEXT NOT NULL,
 previous TEXT NOT NULL, root TEXT NOT NULL, tokens INTEGER NOT NULL,
 expires INTEGER NOT NULL, superseded INTEGER NOT NULL DEFAULT 0,
 quarantined INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(cycle,seq));
CREATE TABLE IF NOT EXISTS postings (
 cycle TEXT NOT NULL, term TEXT NOT NULL, seq INTEGER NOT NULL,
 PRIMARY KEY(cycle,term,seq));
CREATE TABLE IF NOT EXISTS operations (
 cycle TEXT NOT NULL, operation TEXT NOT NULL, key TEXT NOT NULL,
 request_digest TEXT NOT NULL, result BLOB NOT NULL,
 PRIMARY KEY(cycle,operation,key));
CREATE TABLE IF NOT EXISTS events (
 cycle TEXT NOT NULL, cursor INTEGER NOT NULL, body TEXT NOT NULL,
 PRIMARY KEY(cycle,cursor));
CREATE TABLE IF NOT EXISTS fences (
 cycle TEXT NOT NULL, lane TEXT NOT NULL, fence INTEGER NOT NULL,
 PRIMARY KEY(cycle,lane));
CREATE TABLE IF NOT EXISTS call_budgets (
 cycle TEXT NOT NULL, minute INTEGER NOT NULL, calls INTEGER NOT NULL,
 PRIMARY KEY(cycle,minute));
CREATE TABLE IF NOT EXISTS retention_operations (
 cycle TEXT NOT NULL, operation TEXT NOT NULL, key TEXT NOT NULL,
 request_digest TEXT NOT NULL, result TEXT,
 PRIMARY KEY(cycle,operation,key));
CREATE INDEX IF NOT EXISTS segment_visibility ON segments(cycle,plane,lane,seq);
"""

_CYCLE_COLUMNS = {
    "key_ref": "TEXT",
    "key_version": "TEXT",
    "retention_class": "TEXT NOT NULL DEFAULT 'ephemeral'",
    "hold_reason": "TEXT",
    "cleanup_state": "TEXT NOT NULL DEFAULT 'NONE'",
    "seal_digest": "TEXT",
    "checkpoint_manifest_checksum": "TEXT",
    "checkpoint_object_ref_digest": "TEXT",
    "cleanup_remote_receipt": "TEXT",
    "cleanup_operation_key": "TEXT",
    "cleanup_request_digest": "TEXT",
    "deletion_receipt": "TEXT",
}


class SegmentStoreV2:
    def __init__(self, path: str | Path, cache_bytes: int):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.cache_kib = max(1024, cache_bytes // 1024)
        self.writer = threading.RLock()
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            columns = {row[1] for row in db.execute("PRAGMA table_info(cycles)")}
            if "snapshot" not in columns:
                db.execute("ALTER TABLE cycles ADD COLUMN snapshot BLOB")
                columns.add("snapshot")
            # SQLite has no transactional ALTER COLUMN operation. Additive,
            # constant-default columns keep databases created by 0.3.1 readable.
            for name, declaration in _CYCLE_COLUMNS.items():
                if name not in columns:
                    db.execute(f"ALTER TABLE cycles ADD COLUMN {name} {declaration}")
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ContextFault("context_storage_corrupt")
        if os.name != "nt":
            os.chmod(self.path, 0o600)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA trusted_schema=OFF")
        db.execute(f"PRAGMA cache_size=-{self.cache_kib}")
        try:
            yield db
        except sqlite3.Error as exc:
            raise ContextFault("context_storage_unavailable") from exc
        finally:
            db.close()

    @contextmanager
    def transaction(self):
        with self.writer, self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
