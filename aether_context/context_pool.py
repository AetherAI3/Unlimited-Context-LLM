# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""B2 context pool — session-namespaced, mmap'd vector store + budget governor.

This is the **"disk" of the virtual-memory-for-attention design**. Encoded
:class:`Slice` payloads (256-dim retrieval vector + text + tokens + meta) land here; the
vectors live in an **mmap'd file** so the pool is disk-resident and survives a reopen, and
a **budget governor** evicts the lowest-retention slices (ranked by the :class:`Witness`)
so the pool never exceeds its byte ceiling.

What it is / is not
-------------------
  * It *is* a cosine nearest-neighbor store with a session/namespace filter, so two
    far-apart sessions never bleed into each other's results.
  * It *is* fail-soft about the index: the ``flat`` numpy brute-force index **always
    works**; if ``hnswlib`` is importable and ``config.index == "hnsw"`` it uses HNSW for
    speed, otherwise it transparently falls back to flat. A missing optional dependency is
    never a hard failure.
  * A :class:`Slice` is a self-contained dataclass over the 256-dim retrieval embedding.

On-disk layout
--------------
Two files inside the pool dir (``config.dir``):

  * ``vectors.f32`` — a flat ``float32`` mmap of ``[capacity, dim]`` rows; each live slice
    owns one row by its ``row`` index. Grown by re-allocating to a larger capacity.
  * ``pool.json`` — the sidecar header + per-slice metadata records. The vectors file is
    the bulk store; the sidecar is the index of record (id → row + payload). A malformed
    sidecar raises :class:`~aether_context.errors.PoolCorrupt`.

The in-RAM ANN index is rebuilt from the persisted vectors on open, so reopening never
requires the optional ``hnswlib`` to have been present when the pool was written.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from aether_context._log import get_logger
from aether_context.config import PoolConfig, TOKENS_PER_GB
from aether_context.errors import PoolBudgetError, PoolCorrupt
from aether_context.witness import Witness

logger = get_logger(__name__)

# --- optional fast ANN backend (never a hard dependency) ---------------------
try:  # pragma: no cover - import availability is environment-dependent
    import hnswlib as _hnswlib  # type: ignore[import-untyped, import-not-found]

    _HNSWLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - the common CI path (flat fallback)
    _hnswlib = None
    _HNSWLIB_AVAILABLE = False

# --- persistence constants ---------------------------------------------------
#: On-disk format version for the sidecar/header. Bump on any layout change.
POOL_FORMAT_VERSION: int = 1
#: mmap'd vectors filename inside the pool dir.
VECTORS_FILENAME: str = "vectors.f32"
#: Sidecar metadata filename inside the pool dir.
METADATA_FILENAME: str = "pool.json"
#: Initial row capacity for a fresh vectors mmap (grown geometrically).
_INITIAL_CAPACITY: int = 64

# --- budget accounting -------------------------------------------------------
#: Fixed per-slice payload overhead charged on top of the vector bytes. Mirrors the
#: README pool math (~2.2 KB/slice at 512 tok/slice, 256-dim) so ``reach`` lines up:
#: 256 float32 = 1024 B of vector, leaving ~1.2 KB for text + meta + index bookkeeping.
SLICE_PAYLOAD_BYTES: int = 1200


def slice_cost_bytes(dim: int) -> int:
    """Bytes the pool charges per resident slice: vector bytes + fixed payload overhead.

    ``dim * 4`` (float32 vector) plus :data:`SLICE_PAYLOAD_BYTES` for the text/meta/index
    bookkeeping. The governor uses this to translate the GB ceiling into a max slice count
    (it stays in lockstep with :meth:`ContextPool.bytes_used`).
    """
    return dim * 4 + SLICE_PAYLOAD_BYTES


def _default_ceiling_bytes(pool_gb: int, dim: int) -> int:
    """Byte ceiling for a pool sized ``pool_gb`` of reach.

    Reach is ``pool_gb * TOKENS_PER_GB`` tokens; at 512 tok/slice that is a slice count,
    and each slice costs :func:`slice_cost_bytes`. We derive the ceiling from *reach* (not
    raw GB) so "pool size = reach" stays the honest mental model.
    """
    slices = (pool_gb * TOKENS_PER_GB) // 512
    return int(slices) * slice_cost_bytes(dim)


@dataclass
class Slice:
    """A self-contained encoded chunk of context — the pool's unit of storage.

    Fields (all carried verbatim through persistence and search):

      id       stable identifier, unique within the pool
      session  namespace; ``search(session=...)`` isolates one session from the rest
      vector   the 256-dim float32 retrieval embedding
      text     the original text the vector encodes (paged back to the model on a hit)
      tokens   token count of ``text`` (for window/budget math)
      meta     arbitrary JSON-serializable tags (phase, source, etc.)
      score    retention salience in ``[0,1]`` — the witness uses this to rank for eviction
    """

    id: str
    session: str
    vector: np.ndarray
    text: str
    tokens: int
    meta: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


class _FlatIndex:
    """Brute-force numpy cosine index — the always-available fallback.

    Search is a single ``matrix @ query`` dot product (vectors are unit, so dot == cosine).
    O(N) but correct on every platform with zero extra dependencies.
    """

    kind = "flat"

    def __init__(self, dim: int) -> None:
        self._dim = dim

    def search(
        self, matrix: np.ndarray, query: np.ndarray, k: int
    ) -> list[tuple[int, float]]:
        """Top-``k`` ``(row, cosine)`` pairs over ``matrix`` rows, highest cosine first.

        ``matrix`` is ``(n, dim)`` unit rows; ``query`` is a ``(dim,)`` unit vector. Returns
        at most ``k`` pairs. An empty matrix yields ``[]``.
        """
        n = matrix.shape[0]
        if n == 0 or k <= 0:
            return []
        cosines = matrix @ query  # unit rows -> dot product is cosine similarity
        kk = min(k, n)
        # argpartition for the top-kk, then sort just those descending (stable on ties).
        top = np.argpartition(-cosines, kk - 1)[:kk]
        top = top[np.argsort(-cosines[top], kind="stable")]
        return [(int(r), float(cosines[r])) for r in top]


class _HnswIndex:
    """Thin wrapper over ``hnswlib`` for fast approximate cosine search.

    Used only when ``hnswlib`` is importable and ``config.index == "hnsw"``. Rebuilt from
    the pool's vector matrix on add/open (cheap relative to retrieval over a long run).
    """

    kind = "hnsw"

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._index: Any = None
        self._rows: list[int] = []

    def rebuild(self, matrix: np.ndarray, rows: list[int]) -> None:
        """(Re)build the HNSW graph from ``matrix`` rows labelled by ``rows``."""
        n = matrix.shape[0]
        index = _hnswlib.Index(space="cosine", dim=self._dim)
        index.init_index(max_elements=max(1, n), ef_construction=200, M=16)
        if n:
            index.add_items(matrix, np.asarray(rows, dtype=np.int64))
        index.set_ef(max(16, min(200, n)))
        self._index = index
        self._rows = list(rows)

    def search(
        self, matrix: np.ndarray, query: np.ndarray, k: int
    ) -> list[tuple[int, float]]:
        """Top-``k`` ``(row, cosine)`` pairs via HNSW; builds lazily if not yet built."""
        n = matrix.shape[0]
        if n == 0 or k <= 0:
            return []
        if self._index is None:
            self.rebuild(matrix, list(range(n)))
        kk = min(k, n)
        labels, distances = self._index.knn_query(query, k=kk)
        # hnswlib 'cosine' returns distance = 1 - cosine; recover the similarity.
        pairs = [
            (int(lbl), float(1.0 - dist))
            for lbl, dist in zip(labels[0], distances[0])
        ]
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs


class ContextPool:
    """Session-namespaced, mmap'd, budget-governed vector store of :class:`Slice`.

    Construct with a :class:`~aether_context.config.PoolConfig`. If the config's ``dir``
    already holds a pool, it is reopened (vectors + sidecar restored, in-RAM index rebuilt);
    otherwise a fresh empty pool is created lazily on first :meth:`add`.

    Public surface:
      * :meth:`add` — store a slice, then enforce the byte budget via the witness.
      * :meth:`search` — cosine top-``k``, optionally filtered to one ``session``.
      * :meth:`evict_to_budget` — drop lowest-retention slices until under the ceiling.
      * :meth:`bytes_used`, :meth:`stats` — accounting.
      * :meth:`close` — flush vectors + sidecar to disk (also called on GC).
    """

    def __init__(self, config: PoolConfig, *, ceiling_bytes: int | None = None) -> None:
        self._config = config
        self._dim = config.dim
        self._dir = Path(config.dir)
        # Resolve the index kind: honor 'hnsw' only when the lib is importable.
        requested = config.index
        if requested == "hnsw" and _HNSWLIB_AVAILABLE:
            self._index: _FlatIndex | _HnswIndex = _HnswIndex(self._dim)
        else:
            if requested == "hnsw":
                logger.debug(
                    "index='hnsw' requested but hnswlib unavailable; using flat fallback"
                )
            self._index = _FlatIndex(self._dim)
        # Byte ceiling: explicit override (tests) else derived from pool reach.
        self._ceiling_bytes = (
            int(ceiling_bytes)
            if ceiling_bytes is not None
            else _default_ceiling_bytes(config.pool_gb, self._dim)
        )
        if self._ceiling_bytes <= 0:
            raise PoolBudgetError(
                f"computed pool ceiling is {self._ceiling_bytes} bytes (non-positive)",
                hint="Raise pool_gb (floor is 5 GB) or pass a positive ceiling_bytes.",
            )
        # In-RAM state. The mmap is the bulk vector store; these mirror live slices.
        self._slices: dict[str, Slice] = {}        # id -> Slice (stored unit vector)
        self._row_of: dict[str, int] = {}          # id -> row index in the mmap
        self._order: list[str] = []                # insertion order of live ids
        self._witness = Witness()                  # retention scores for eviction
        self._mmap: np.memmap | None = None
        self._capacity = 0
        self._dirty = False
        self._index_dirty = True
        # Restore from disk if a pool already exists in this dir.
        if self._metadata_file().exists():
            self._load()

    # -- paths -----------------------------------------------------------------
    def _vectors_file(self) -> Path:
        return self._dir / VECTORS_FILENAME

    def _metadata_file(self) -> Path:
        return self._dir / METADATA_FILENAME

    @property
    def metadata_path(self) -> Path:
        """Path to the sidecar metadata file (``pool.json``) inside the pool dir."""
        return self._metadata_file()

    @property
    def vectors_path(self) -> Path:
        """Path to the mmap'd vectors file (``vectors.f32``) inside the pool dir."""
        return self._vectors_file()

    # -- introspection ---------------------------------------------------------
    @property
    def index_kind(self) -> str:
        """The resolved index kind actually in use (``"flat"`` or ``"hnsw"``)."""
        return self._index.kind

    @property
    def ceiling_bytes(self) -> int:
        """The byte ceiling the governor holds the pool at or below."""
        return self._ceiling_bytes

    def __len__(self) -> int:
        return len(self._slices)

    # -- write -----------------------------------------------------------------
    def add(self, sl: Slice) -> None:
        """Store ``sl`` (overwriting any slice with the same id), then enforce the budget.

        The vector is validated to be ``(dim,)`` and a normalized copy is written into the
        mmap's row for this id. The slice's ``score`` is fed to the witness as its retention
        salience. After the write the governor runs (:meth:`evict_to_budget`) so the pool is
        *always* at or below its ceiling the instant ``add`` returns.
        """
        vec = np.asarray(sl.vector, dtype=np.float32)
        if vec.shape != (self._dim,):
            raise PoolBudgetError(
                f"slice {sl.id!r} vector has shape {vec.shape}, expected {(self._dim,)}",
                hint=f"Encode with the pool's dim ({self._dim}); the 256-dim retrieval "
                f"embedding is the only vector the pool stores.",
            )
        row = self._row_of.get(sl.id)
        if row is None:
            row = len(self._order)
            self._ensure_capacity(row + 1)
            self._order.append(sl.id)
            self._row_of[sl.id] = row
        stored_vec = self._unit(vec)
        assert self._mmap is not None
        self._mmap[row] = stored_vec
        self._slices[sl.id] = Slice(
            id=sl.id,
            session=sl.session,
            vector=stored_vec.copy(),
            text=sl.text,
            tokens=int(sl.tokens),
            meta=dict(sl.meta),
            score=float(sl.score),
        )
        self._witness.touch(sl.id, salience=float(sl.score), now=0.0)
        self._dirty = True
        self._index_dirty = True
        self.evict_to_budget()

    # -- read ------------------------------------------------------------------
    def search(
        self, query_vec: np.ndarray, k: int, session: str | None = None
    ) -> list[Slice]:
        """Top-``k`` slices by cosine similarity to ``query_vec``, highest first.

        If ``session`` is given the search is **scoped to that namespace** — slices from
        other sessions are invisible, so far-apart sessions never cross-contaminate (even
        when another session has a closer vector). Returns ``[]`` for an empty pool, an
        unknown session, or ``k <= 0``.
        """
        if k <= 0 or not self._order:
            return []
        query = self._unit(np.asarray(query_vec, dtype=np.float32))
        if query.shape != (self._dim,):
            raise PoolBudgetError(
                f"query vector has shape {query.shape}, expected {(self._dim,)}",
                hint=f"Search with a {self._dim}-dim vector (the encoder's output dim).",
            )
        if session is None:
            return self._search_global(query, k)
        return self._search_session(query, k, session)

    def _search_global(self, query: np.ndarray, k: int) -> list[Slice]:
        """Unfiltered top-``k`` over every live slice (uses the resolved ANN index)."""
        matrix = self._live_matrix()
        self._refresh_index(matrix)
        pairs = self._index.search(matrix, query, k)
        return [self._slices[self._order[row]] for row, _ in pairs]

    def _search_session(self, query: np.ndarray, k: int, session: str) -> list[Slice]:
        """Top-``k`` restricted to one session via a brute-force masked dot product.

        The session filter is exact (a numpy mask over the live matrix), so isolation holds
        regardless of which index backend is active — correctness never rides on the ANN.
        """
        rows = [i for i, sid in enumerate(self._order)
                if self._slices[sid].session == session]
        if not rows:
            return []
        matrix = self._live_matrix()
        sub = matrix[rows]
        cosines = sub @ query
        kk = min(k, len(rows))
        top = np.argpartition(-cosines, kk - 1)[:kk]
        top = top[np.argsort(-cosines[top], kind="stable")]
        return [self._slices[self._order[rows[int(j)]]] for j in top]

    # -- governor --------------------------------------------------------------
    def evict_to_budget(self) -> list[str]:
        """Evict the lowest-retention slices until the pool fits under its byte ceiling.

        Ranking is by the :class:`Witness` (lowest score first), exactly as the witness's
        own ``budget_evict`` does — so the survivors are always the most-retained slices.
        Returns the evicted ids (``[]`` when the pool already fits). The governor is called
        automatically after every :meth:`add`, so "never exceeds budget" is literally true.
        """
        cost = slice_cost_bytes(self._dim)
        max_slices = max(0, self._ceiling_bytes // cost)
        if len(self._order) <= max_slices:
            return []
        # Lowest-retention first = the tail of the witness ranking (highest first).
        ranked = self._witness.rank()  # highest score first
        known = set(ranked)
        unranked = [sid for sid in self._order if sid not in known]
        full_ranking = ranked + unranked
        evict_ids = full_ranking[max_slices:]
        for sid in evict_ids:
            self._remove(sid)
        if evict_ids:
            self._compact_rows()
            self._dirty = True
            self._index_dirty = True
            logger.debug(
                "evicted %d slice(s) to hold pool at <=%d bytes",
                len(evict_ids), self._ceiling_bytes,
            )
        return evict_ids

    def clear_session(self, session_id: str | None) -> int:
        """Remove every live slice belonging to ``session_id`` and return the count removed.

        This is the storage half of the engine's *clear* semantics: dropping the slices a
        single session externalized into the pool. When ``session_id`` is ``None`` the whole
        pool is emptied (the shared/global clear), so a ``shared`` pool clears all sessions.

        Rows are compacted after removal so ``_live_matrix`` stays a dense prefix and the
        sidecar row indices remain consistent; the byte accounting and stats follow the
        surviving slices exactly. Returns ``0`` when nothing matched (idempotent, safe).
        """
        if session_id is None:
            removed = list(self._order)
        else:
            removed = [
                sid for sid in self._order
                if self._slices[sid].session == session_id
            ]
        if not removed:
            return 0
        for sid in removed:
            self._remove(sid)
        self._compact_rows()
        self._dirty = True
        self._index_dirty = True
        logger.debug(
            "cleared %d slice(s) for session %r", len(removed), session_id
        )
        return len(removed)

    def _remove(self, sid: str) -> None:
        """Drop a single slice id from the in-RAM structures (mmap row reclaimed on compact)."""
        self._slices.pop(sid, None)
        self._row_of.pop(sid, None)
        self._witness.forget(sid)
        try:
            self._order.remove(sid)
        except ValueError:
            pass

    def _compact_rows(self) -> None:
        """Re-pack live slices into contiguous rows ``[0..len)`` after eviction.

        Keeps the mmap dense so ``_live_matrix`` is a simple prefix slice and row indices
        stay stable for the sidecar. The stored ``Slice.vector`` is the source of truth for
        each row, so packing never depends on a stale mmap position.
        """
        if self._mmap is None:
            return
        n = len(self._order)
        if n == 0:
            self._row_of = {}
            return
        packed = np.empty((n, self._dim), dtype=np.float32)
        for new_row, sid in enumerate(self._order):
            packed[new_row] = self._slices[sid].vector
            self._row_of[sid] = new_row
        self._mmap[:n] = packed

    # -- accounting ------------------------------------------------------------
    def bytes_used(self) -> int:
        """Total bytes the live slices occupy, by the same accounting the governor uses."""
        return len(self._slices) * slice_cost_bytes(self._dim)

    def stats(self) -> dict[str, Any]:
        """A small dict snapshot: count, bytes, ceiling, dim, index kind, sessions."""
        return {
            "count": len(self._slices),
            "bytes_used": self.bytes_used(),
            "ceiling_bytes": self._ceiling_bytes,
            "dim": self._dim,
            "index": self.index_kind,
            "sessions": sorted({s.session for s in self._slices.values()}),
        }

    # -- persistence -----------------------------------------------------------
    def close(self) -> None:
        """Flush vectors + sidecar to disk and release the mmap. Idempotent.

        Releasing the mmap matters on Windows: a still-mapped file cannot be reopened, so
        reopening the same pool dir would fail with ``[Errno 22]`` unless the prior handle
        is dropped here.
        """
        self._flush()
        self._release_mmap()

    def __del__(self) -> None:  # pragma: no cover - best-effort flush on GC
        try:
            self._flush()
            self._release_mmap()
        except Exception:  # noqa: BLE001 - never raise from a finalizer
            pass

    def _flush(self) -> None:
        """Write the mmap and sidecar metadata if anything changed since the last flush."""
        if not self._dirty:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        if self._mmap is not None:
            self._mmap.flush()
        records = [
            {
                "id": sid,
                "session": self._slices[sid].session,
                "row": self._row_of[sid],
                "text": self._slices[sid].text,
                "tokens": self._slices[sid].tokens,
                "meta": self._slices[sid].meta,
                "score": self._slices[sid].score,
            }
            for sid in self._order
        ]
        header = {
            "version": POOL_FORMAT_VERSION,
            "dim": self._dim,
            "count": len(self._order),
            "capacity": self._capacity,
            "index": self._config.index,
            "ceiling_bytes": self._ceiling_bytes,
            "slices": records,
        }
        tmp = self._metadata_file().with_suffix(".json.tmp")
        tmp.write_text(json.dumps(header), encoding="utf-8")
        os.replace(tmp, self._metadata_file())
        self._dirty = False

    def _load(self) -> None:
        """Restore a pool from ``pool.json`` + ``vectors.f32``; raise PoolCorrupt on bad data."""
        path = self._metadata_file()
        try:
            header = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            raise PoolCorrupt(f"could not read pool metadata at {path}: {exc}") from exc
        try:
            disk_dim = int(header["dim"])
            records = header["slices"]
            capacity = int(header.get("capacity", len(records)))
        except (KeyError, TypeError, ValueError) as exc:
            raise PoolCorrupt(
                f"pool metadata at {path} is missing required fields: {exc}"
            ) from exc
        if disk_dim != self._dim:
            raise PoolCorrupt(
                f"pool at {self._dir} was written with dim={disk_dim}, "
                f"but this pool expects dim={self._dim}",
            )
        vfile = self._vectors_file()
        if not vfile.exists():
            raise PoolCorrupt(
                f"pool metadata at {path} present but vectors file {vfile} is missing"
            )
        self._capacity = max(capacity, _INITIAL_CAPACITY)
        self._open_mmap(self._capacity)
        assert self._mmap is not None
        try:
            for rec in records:
                sid = rec["id"]
                row = int(rec["row"])
                vec = np.asarray(self._mmap[row], dtype=np.float32).copy()
                sl = Slice(
                    id=sid,
                    session=rec["session"],
                    vector=vec,
                    text=rec["text"],
                    tokens=int(rec["tokens"]),
                    meta=dict(rec.get("meta", {})),
                    score=float(rec.get("score", 0.0)),
                )
                self._order.append(sid)
                self._row_of[sid] = row
                self._slices[sid] = sl
                self._witness.touch(sid, salience=sl.score, now=0.0)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise PoolCorrupt(
                f"pool metadata at {path} has a malformed slice record: {exc}"
            ) from exc
        self._index_dirty = True

    # -- mmap management -------------------------------------------------------
    def _ensure_capacity(self, n_rows: int) -> None:
        """Make sure the mmap can hold ``n_rows`` rows, growing geometrically if needed.

        Windows cannot reopen (``w+``) a file that is still memory-mapped, so the old
        mapping is copied into RAM and **fully released** before the larger mmap is opened.
        """
        if self._mmap is not None and n_rows <= self._capacity:
            return
        new_cap = max(_INITIAL_CAPACITY, self._capacity)
        while new_cap < n_rows:
            new_cap *= 2
        carry: np.ndarray | None = None
        if self._mmap is not None:
            keep = min(self._capacity, len(self._order))
            if keep > 0:
                carry = np.asarray(self._mmap[:keep], dtype=np.float32).copy()
        self._release_mmap()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._open_mmap(new_cap, carry=carry)

    def _release_mmap(self) -> None:
        """Flush and drop the current mmap so the OS handle is released (Windows-safe)."""
        if self._mmap is not None:
            try:
                self._mmap.flush()
            except (ValueError, OSError):  # already closed / detached
                pass
            self._mmap = None

    def _open_mmap(self, capacity: int, *, carry: np.ndarray | None = None) -> None:
        """Open the vectors mmap at ``capacity`` rows, optionally seeding it with ``carry``.

        On a reopen (no ``carry`` provided, file already on disk) the existing rows are read
        with :func:`numpy.fromfile` — which opens and closes the file cleanly, leaving no
        lingering mapping before the ``w+`` open (the cause of an [Errno 22] on Windows).
        """
        vfile = self._vectors_file()
        seed = carry
        if seed is None and vfile.exists() and capacity > 0:
            seed = self._read_existing_rows(vfile, capacity)
        mm = np.memmap(vfile, dtype=np.float32, mode="w+", shape=(capacity, self._dim))
        if seed is not None and seed.shape[0] > 0:
            rows = min(seed.shape[0], capacity)
            mm[:rows] = seed[:rows]
        self._mmap = mm
        self._capacity = capacity

    def _read_existing_rows(self, vfile: Path, capacity: int) -> np.ndarray | None:
        """Read existing on-disk vector rows (up to ``capacity``) into RAM for a reopen.

        Uses :func:`numpy.fromfile` (plain read, no mmap) so no file handle survives the
        call — required so the subsequent ``w+`` mmap open succeeds on Windows.
        """
        size = vfile.stat().st_size
        row_bytes = self._dim * 4
        if row_bytes == 0:
            return None
        n_rows = size // row_bytes
        if n_rows == 0:
            return None
        take = min(n_rows, capacity)
        flat = np.fromfile(vfile, dtype=np.float32, count=take * self._dim)
        if flat.size < take * self._dim:
            return None
        return flat.reshape(take, self._dim)

    # -- helpers ---------------------------------------------------------------
    def _live_matrix(self) -> np.ndarray:
        """The ``(n, dim)`` matrix of live slice vectors in current row order."""
        n = len(self._order)
        if n == 0 or self._mmap is None:
            return np.empty((0, self._dim), dtype=np.float32)
        return np.asarray(self._mmap[:n])

    def _refresh_index(self, matrix: np.ndarray) -> None:
        """Rebuild the ANN index from ``matrix`` if it is stale (flat index is a no-op)."""
        if isinstance(self._index, _HnswIndex) and self._index_dirty:
            self._index.rebuild(matrix, list(range(matrix.shape[0])))
        self._index_dirty = False

    @staticmethod
    def _unit(vec: np.ndarray) -> np.ndarray:
        """L2-normalize ``vec`` to a float32 unit vector (zero vector passes through)."""
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12 or math.isnan(norm):
            return vec.astype(np.float32)
        return (vec / norm).astype(np.float32)


__all__ = [
    "ContextPool",
    "Slice",
    "slice_cost_bytes",
    "POOL_FORMAT_VERSION",
    "VECTORS_FILENAME",
    "METADATA_FILENAME",
    "SLICE_PAYLOAD_BYTES",
]
