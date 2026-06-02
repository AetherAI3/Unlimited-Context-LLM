# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI - Brandon Barrante
# SPDX-License-Identifier: Apache-2.0
"""B3 slice loader — the prefetch **pager** for virtual-memory-for-attention.

Make context-slice retrieval fast by PRE-LOADING the slices the session is about to need,
instead of retrieving cold on every turn. In a coding run the next slice is highly
predictable from what the model is reasoning about *now*: embed the current reasoning text,
search the pool, and keep the nearest slices in a small **warm set**. When the model needs
that context, the slice is an O(1) memory lookup, not an ANN search.

Expected retrieval latency is then ``E[t] = h·t_warm + (1−h)·t_cold`` where ``h`` is the hit
rate; the pager's whole job is to push ``h → 1`` by prefetching from current state. ``t_warm``
is a dict lookup (~µs); ``t_cold`` is the classical pool search (~ms). The hit rate is
**measured**, not assumed.

Design
------
A single-threaded, LRU-budgeted warm cache (``prefetch``/``get``/``invalidate`` + hit-rate),
plus two small disciplines:

  * the **idle-aware ε re-probe** (:func:`reprobe_probability` / :func:`should_reprobe`) — a
    key that has gone idle gets a rising probability of being re-checked, so a
    stale-but-recoverable region never stays dark forever; and
  * the **depth-cap-1 provenance grounding verdict** (:func:`grounding_verdict`, capped at
    :data:`MAX_CORRECTION_DEPTH`) — a paged-back slice is flagged only if it has *no
    provenance* or contradicts a *hard fact*; merely disagreeing with recent (possibly
    stale) context is **not** a flag.

The key
-------
A :class:`SliceKey(session, topic)` is a plain discrete coordinate: ``session`` is the
namespace and ``topic`` is a coarse phase/subject label. The cold path is injected as
``retrieve_fn`` defaulting to ``context_pool.search`` (scoped to ``key.session``). Keys are
**discrete strings only** — :class:`SliceKey` rejects non-string coordinates with
``TypeError`` (vectors address slices only *inside* :meth:`Pager.get`, never as a key).

Single-threaded by design
-------------------------
The pager core is **single-threaded and pure**. Concurrency belongs to the caller: the session
runs :meth:`Pager.prefetch_from` on a background thread *while the model generates* (the backend
HTTP/subprocess call releases the GIL, so a prefetch thread genuinely overlaps generation). No
thread, lock, or queue lives in this module.

Fail-soft
---------
The pager is an *optimization*, never a correctness dependency. A cold-search error or an
encoder hiccup is logged and degrades to an empty window — it never raises into a long run.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Protocol

import numpy as np

from aether_context._log import get_logger
from aether_context.context_pool import Slice

logger = get_logger(__name__)

#: Number of pre-assembled slice *keys* kept warm at once (the pager's working-set budget).
#: Mirrors the upstream ``DEFAULT_WARM_BUDGET``; 16 keys is plenty of reach for one turn while
#: staying tiny in RAM (the slice payloads themselves live in the pool, not here).
DEFAULT_WARM_BUDGET: int = 16

# --- idle-aware re-probe constants (ported from exploration.py) --------------
#: Floor re-probe probability right after a key was accessed (never 0 — nothing stays dark).
BASE_EPS: float = 0.02
#: Idle periods at which the gap-to-certain halves (so probability rises smoothly toward 1).
HALF_LIFE_PERIODS: float = 20.0

# --- depth-cap grounding constants (ported from latency_budget.py) -----------
#: One correction attempt, then abstain — the grounding check never recurses unboundedly.
MAX_CORRECTION_DEPTH: int = 1


class Grounding(str, Enum):
    """Provenance-first grounding verdict for a paged-back slice."""

    PASS = "pass"
    FLAG = "flag"


# --- the key (generalized; discrete; no trading/8-dim coordinate) ------------
@dataclass(frozen=True)
class SliceKey:
    """A discrete, hashable coordinate that addresses a region of the pool.

    A plain ``(session, topic)`` pair: ``session`` is the namespace (scopes the cold search)
    and ``topic`` is a coarse phase/subject label the session's own state machine assigns.

    The key is **discrete strings only**. A vector (the 256-dim retrieval embedding) is
    rejected with ``TypeError`` — vectors address slices only *inside* :meth:`Pager.get`,
    never as a key.
    """

    session: str
    topic: str

    def __post_init__(self) -> None:
        if not isinstance(self.session, str):
            raise TypeError(
                f"SliceKey.session must be a str, got {type(self.session).__name__}; "
                "a SliceKey is a discrete (session, topic) coordinate, not a vector."
            )
        if not isinstance(self.topic, str):
            raise TypeError(
                f"SliceKey.topic must be a str, got {type(self.topic).__name__}; a key is a "
                "discrete string coordinate, not a vector. Use a topic label; query vectors "
                "address slices only inside Pager.get/search."
            )


# --- minimal structural contracts the pager depends on -----------------------
class _PoolLike(Protocol):
    """The slice of :class:`~aether_context.context_pool.ContextPool` the pager uses."""

    def search(
        self, query_vec: np.ndarray, k: int, session: str | None = ...
    ) -> list[Slice]: ...


class _EncoderLike(Protocol):
    """The slice of :class:`~aether_context.encoder.StaticEncoder` the pager uses."""

    def encode(self, text: str) -> np.ndarray: ...


#: Cold path signature: a key + query vector + k -> the slices for that region.
RetrieveFn = Callable[["SliceKey", np.ndarray, int], list[Slice]]


# --- idle-aware ε re-probe (ported from exploration.py) ----------------------
def reprobe_probability(
    periods_since_probe: float,
    base_eps: float = BASE_EPS,
    half_life: float = HALF_LIFE_PERIODS,
) -> float:
    """Re-probe probability for a region idle ``periods_since_probe`` periods.

    Rises from ``base_eps`` (just accessed) toward ``1.0`` (long idle), so no suppressed /
    stale region stays dark forever. The gap-to-1 halves every ``half_life`` idle periods.
    Monotone non-decreasing in idle time and clamped to ``[base_eps, 1.0]``.
    """
    if half_life <= 0:
        return 1.0
    gap = (1.0 - base_eps) * (0.5 ** (max(0.0, periods_since_probe) / half_life))
    return max(base_eps, min(1.0, 1.0 - gap))


def should_reprobe(
    periods_since_probe: float,
    rng_uniform: float,
    base_eps: float = BASE_EPS,
    half_life: float = HALF_LIFE_PERIODS,
) -> bool:
    """Decide whether to re-probe an idle region; ``rng_uniform`` is a draw in ``[0,1)``."""
    return rng_uniform < reprobe_probability(periods_since_probe, base_eps, half_life)


# --- depth-cap-1 provenance grounding (ported from latency_budget.py) --------
def grounding_verdict(has_provenance: bool, contradicts_hard_fact: bool) -> Grounding:
    """Provenance-first grounding for a paged-back slice (recursion capped at depth 1).

    A claim is flagged only if it **contradicts a hard fact** or has **no provenance** at all.
    Disagreeing with recent (possibly stale) *context* is explicitly NOT a flag — that is how
    a correct new decision survives the catcher during a shift. A slice that came from the pool
    inherently has provenance (it was encoded and externalized from real prior context), so the
    common pager case (resident slice, no hard-fact contradiction) is :attr:`Grounding.PASS`.
    """
    if contradicts_hard_fact:
        return Grounding.FLAG
    if not has_provenance:
        return Grounding.FLAG
    return Grounding.PASS


# --- the pager ---------------------------------------------------------------
class Pager:
    """Single-threaded, budget-bounded warm cache of pool slices — the B3 pager.

    Wraps a :class:`~aether_context.context_pool.ContextPool` and a
    :class:`~aether_context.encoder.StaticEncoder`. Construct with a warm-key budget; the
    pager keeps at most ``budget`` :class:`SliceKey` regions warm, evicting the least-recently
    used when full. The slice payloads live in the pool — the warm set holds only the small
    ``key -> [Slice]`` mapping and per-key LRU / idle bookkeeping.

    Public surface:
      * :meth:`prefetch_from` — embed reasoning text, search the pool, warm the result (the
        method the session runs on a side thread while the model generates).
      * :meth:`prefetch` — warm a key with explicit text and/or an explicit query vector.
      * :meth:`get` — hot path: warm ``O(1)`` hit, else a cold search that warms opportunistically.
      * :meth:`window` — the resident slices (the working set the model can be handed this turn).
      * :meth:`hit_rate` — measured hits / (hits + misses).
      * :meth:`invalidate` — drop warm keys matching a predicate (stale entry / topic change).
      * :meth:`reprobe_probability` — idle-aware re-probe probability for a warm key.
      * :meth:`ground` — depth-cap-1 grounding verdict for a paged-back slice.
    """

    def __init__(
        self,
        pool: _PoolLike,
        encoder: _EncoderLike,
        budget: int = DEFAULT_WARM_BUDGET,
        *,
        retrieve_fn: RetrieveFn | None = None,
        default_k: int = 8,
    ) -> None:
        self._pool = pool
        self._encoder = encoder
        self.budget = max(1, int(budget))
        self._default_k = max(1, int(default_k))
        # Cold path: SliceKey + query vector -> slices. Defaults to a session-scoped pool
        # search so namespace isolation rides through the pager for free.
        self._retrieve: RetrieveFn = (
            retrieve_fn if retrieve_fn is not None else self._pool_retrieve
        )
        # Warm state. The slice payloads stay in the pool; here we keep only the mapping
        # and the LRU / idle counters (pure bookkeeping, single-threaded).
        self._warm: dict[SliceKey, list[Slice]] = {}
        self._lastused: dict[SliceKey, int] = {}
        self._seq = 0
        self.hits = 0
        self.misses = 0
        self.prefetched = 0

    # -- cold path (injected; defaults to a session-scoped pool search) --------
    def _pool_retrieve(self, key: SliceKey, query_vec: np.ndarray, k: int) -> list[Slice]:
        """Default cold path: a session-scoped ``pool.search`` for ``key``'s region."""
        return self._pool.search(query_vec, k, session=key.session)

    # -- LRU bookkeeping (single-threaded) ------------------------------------
    def _touch(self, key: SliceKey) -> None:
        """Mark ``key`` as most-recently-used (monotone sequence counter)."""
        self._seq += 1
        self._lastused[key] = self._seq

    def _evict_one(self, protect: set[SliceKey]) -> None:
        """Evict the least-recently-used warm key not in ``protect`` (no-op if all protected)."""
        cand = [k for k in self._warm if k not in protect]
        if not cand:
            return
        victim = min(cand, key=lambda k: self._lastused.get(k, 0))
        self._warm.pop(victim, None)
        self._lastused.pop(victim, None)

    def _store_warm(self, key: SliceKey, slices: list[Slice], protect: set[SliceKey]) -> None:
        """Insert/refresh ``key`` in the warm set, evicting LRU to stay within budget."""
        if key not in self._warm and len(self._warm) >= self.budget:
            self._evict_one(protect)
        if key not in self._warm and len(self._warm) >= self.budget:
            # Every other warm key is protected this pass; drop the incoming one.
            return
        self._warm[key] = slices
        self._touch(key)

    # -- embedding (fail-soft) ------------------------------------------------
    def _embed(self, text: str) -> np.ndarray | None:
        """Encode ``text`` to a query vector; on any encoder error degrade to ``None``."""
        try:
            return np.asarray(self._encoder.encode(text), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 - fail-soft: pager is an optimization
            logger.warning("encoder failed during prefetch (%s); skipping warm", exc)
            return None

    def _retrieve_safe(
        self, key: SliceKey, query_vec: np.ndarray, k: int
    ) -> list[Slice] | None:
        """Run the injected cold path; on any error degrade to ``None`` (never raise)."""
        try:
            return list(self._retrieve(key, query_vec, k))
        except Exception as exc:  # noqa: BLE001 - fail-soft: never crash a long run
            logger.warning("cold retrieve failed for %r (%s); degrading", key, exc)
            return None

    # -- write: warm the predicted-next region --------------------------------
    def prefetch_from(
        self, key: SliceKey, reasoning_text: str, *, k: int | None = None
    ) -> list[Slice]:
        """Embed ``reasoning_text``, search the pool, and warm the nearest slices under ``key``.

        This is the pager's headline move and the one the session runs on a side thread *while
        the model generates*: from what the model is reasoning about *now*, predict and warm the
        slices it is about to need. Returns the warmed slices (``[]`` on an encoder/search hiccup
        — fail-soft, never raises). Off the hot path; protects the prefetched key from eviction.
        """
        query = self._embed(reasoning_text)
        if query is None:
            return []
        return self.prefetch(key, reasoning_text, query_vec=query, k=k)

    def prefetch(
        self,
        key: SliceKey,
        reasoning_text: str | None = None,
        *,
        query_vec: np.ndarray | None = None,
        k: int | None = None,
    ) -> list[Slice]:
        """Warm ``key`` from an explicit ``query_vec`` (or embed ``reasoning_text``).

        Idempotent-ish: an already-warm key is left in place and returned. Bounded by the warm
        budget (LRU eviction of an unprotected key when full). Returns the slices now warm for
        ``key`` (``[]`` on a degraded embed/search). Re-warming refreshes the key's idle clock.
        """
        if key in self._warm:
            self._touch(key)
            return self._warm[key]
        query = query_vec
        if query is None:
            if reasoning_text is None:
                return []
            query = self._embed(reasoning_text)
            if query is None:
                return []
        query = np.asarray(query, dtype=np.float32)
        slices = self._retrieve_safe(key, query, k if k is not None else self._default_k)
        if slices is None:
            return []
        self._store_warm(key, slices, protect={key})
        self.prefetched += 1
        return self._warm.get(key, [])

    # -- read: the hot path ----------------------------------------------------
    def get(
        self,
        key: SliceKey,
        reasoning_text: str | None = None,
        *,
        query_vec: np.ndarray | None = None,
        k: int | None = None,
    ) -> list[Slice]:
        """Hot path. Warm ``key`` -> ``O(1)`` hit; cold -> a search that warms opportunistically.

        On a warm hit the slices come straight from the warm set (no cold call) and ``hits`` is
        incremented. On a miss the cold path runs (embedding ``reasoning_text`` or using
        ``query_vec``), the result warms the key (within budget), and ``misses`` is incremented —
        so :meth:`hit_rate` reflects reality. A degraded cold path returns ``[]`` (still a miss).
        """
        warm = self._warm.get(key)
        if warm is not None:
            self.hits += 1
            self._touch(key)
            return warm
        self.misses += 1
        query = query_vec
        if query is None and reasoning_text is not None:
            query = self._embed(reasoning_text)
        if query is None:
            # No way to address the region without a query vector -> empty window (fail-soft).
            return []
        query = np.asarray(query, dtype=np.float32)
        slices = self._retrieve_safe(key, query, k if k is not None else self._default_k)
        if slices is None:
            return []
        self._store_warm(key, slices, protect={key})
        return self._warm.get(key, [])

    # -- invalidation ----------------------------------------------------------
    def invalidate(self, predicate: Callable[[SliceKey], bool]) -> int:
        """Drop warm keys matching ``predicate`` (stale entry / topic / session change).

        Returns how many warm keys were dropped. The pool is untouched — invalidation only
        cools the warm set so the next :meth:`get` re-reads fresh from the pool.
        """
        drop = [k for k in self._warm if predicate(k)]
        for k in drop:
            self._warm.pop(k, None)
            self._lastused.pop(k, None)
        return len(drop)

    @property
    def default_k(self) -> int:
        """How many slices the cold path pulls per region by default (the resident width)."""
        return self._default_k

    @default_k.setter
    def default_k(self, value: int) -> None:
        """Set the resident width (floored at 1). Used by the Extended-Thinking toggle."""
        self._default_k = max(1, int(value))

    def reset(self) -> int:
        """Drop the entire resident window (every warm key); return how many were dropped.

        This is the *resident* half of the engine's clear semantics: it empties the working
        set the model would be handed this turn, leaving the pool (the reachable slices on
        disk) completely untouched. The next :meth:`get` / :meth:`prefetch_from` re-reads
        fresh from the pool. Hit/miss counters are left intact so a measured hit rate is not
        forged by a clear. Returns the number of warm keys evicted (``0`` when already empty).
        """
        dropped = len(self._warm)
        self._warm.clear()
        self._lastused.clear()
        return dropped

    def is_warm(self, key: SliceKey) -> bool:
        """Whether ``key`` currently has a resident warm entry."""
        return key in self._warm

    # -- resident window -------------------------------------------------------
    def window(self) -> list[Slice]:
        """The resident slices across all warm keys — the working set for this turn.

        De-duplicated by slice id (a slice may be warmed under more than one key), ordered by
        warm-key recency (most-recently-used keys first) so the freshest context leads.
        """
        seen: set[str] = set()
        out: list[Slice] = []
        for key in sorted(self._warm, key=lambda k: self._lastused.get(k, 0), reverse=True):
            for sl in self._warm[key]:
                if sl.id not in seen:
                    seen.add(sl.id)
                    out.append(sl)
        return out

    @property
    def warm_count(self) -> int:
        """Number of warm keys currently resident (``<= budget``)."""
        return len(self._warm)

    # -- measured hit rate + latency math -------------------------------------
    def hit_rate(self) -> float:
        """Measured hit rate ``hits / (hits + misses)`` (``0.0`` before any access)."""
        n = self.hits + self.misses
        return self.hits / n if n else 0.0

    def expected_latency(self, t_warm: float, t_cold: float) -> float:
        """``E[t] = h·t_warm + (1−h)·t_cold`` at the current measured hit rate ``h``."""
        h = self.hit_rate()
        return h * t_warm + (1.0 - h) * t_cold

    def speedup(self, t_warm: float, t_cold: float) -> float:
        """How many times faster than always-cold at the current hit rate (``inf`` if free)."""
        e = self.expected_latency(t_warm, t_cold)
        return (t_cold / e) if e > 0 else float("inf")

    # -- idle-aware re-probe ---------------------------------------------------
    def reprobe_probability(
        self, key: SliceKey, base_eps: float = BASE_EPS, half_life: float = HALF_LIFE_PERIODS
    ) -> float:
        """Idle-aware re-probe probability for warm ``key``.

        ``periods_since_probe`` is the number of pager accesses since ``key`` was last touched
        (a never-warmed key reads as maximally idle). Rises toward ``1.0`` the longer ``key`` has
        gone unaccessed, so a stale-but-recoverable region gets re-checked. See
        :func:`reprobe_probability`.
        """
        last = self._lastused.get(key)
        idle = float(self._seq - last) if last is not None else float(self._seq)
        return reprobe_probability(idle, base_eps=base_eps, half_life=half_life)

    def should_reprobe(
        self,
        key: SliceKey,
        rng_uniform: float,
        base_eps: float = BASE_EPS,
        half_life: float = HALF_LIFE_PERIODS,
    ) -> bool:
        """Whether to re-probe warm ``key`` now; ``rng_uniform`` is a draw in ``[0,1)``."""
        return rng_uniform < self.reprobe_probability(
            key, base_eps=base_eps, half_life=half_life
        )

    # -- depth-cap-1 grounding -------------------------------------------------
    def ground(self, slice_: Slice, *, contradicts_hard_fact: bool = False) -> Grounding:
        """Depth-cap-1 grounding verdict for a paged-back ``slice_``.

        A slice that came from the pool has provenance (it was encoded and externalized from real
        prior context), so it PASSes unless it contradicts a *hard fact*. Merely disagreeing with
        recent context is not a flag. See :func:`grounding_verdict`.
        """
        has_provenance = bool(slice_.text) or bool(slice_.meta)
        return grounding_verdict(
            has_provenance=has_provenance, contradicts_hard_fact=contradicts_hard_fact
        )

    # -- convenience -----------------------------------------------------------
    def prefetch_many(self, items: Iterable[tuple[SliceKey, str]]) -> int:
        """Warm several ``(key, reasoning_text)`` pairs; return how many keys ended up warm."""
        warmed = 0
        for key, text in items:
            if self.prefetch_from(key, text):
                warmed += 1
        return warmed


__all__ = [
    "Pager",
    "SliceKey",
    "Grounding",
    "grounding_verdict",
    "reprobe_probability",
    "should_reprobe",
    "RetrieveFn",
    "DEFAULT_WARM_BUDGET",
    "BASE_EPS",
    "HALF_LIFE_PERIODS",
    "MAX_CORRECTION_DEPTH",
]
