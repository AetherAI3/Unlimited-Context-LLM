"""+/- retention witness — page-replacement scoring + budget eviction.

This is the *fidelity field* that decides which encoded slices stay resident in the
context pool and which fade out. It is the local-first generalization of the
AetherCloud context engine's eviction policy: a **pure scoring function over access
events**, with no atlas-cell, ground-truth, tiering, or promotion coupling.

Why score on salience, not recency
-----------------------------------
The naive cache evicts on recency/frequency (LRU/LFU). For a long coding run that is
exactly backwards: the load-bearing fact established an hour ago is rare and old, so an
LRU policy throws it out first. Two rules fix it (ported from the upstream retention
policy, atlas coupling stripped):

  1. **Score retention on SURPRISE x IMPACT x UNIQUENESS, not frequency.** The geometric
     mean (see :func:`retention_score`) means one weak driver can't be masked by a strong
     one — a slice has to be salient on every axis to score high. In coding terms:
     surprise ~ content density, impact ~ query relevance, uniqueness ~ 1/(1+similar).
  2. **A salient slice fades by *idle time*, never by raw frequency**, and **re-hardens**
     the instant it is relevant again.

The lifecycle of a slice id
---------------------------
  * :meth:`Witness.touch` — **harden** (or re-harden): register the slice and lift its
    score toward the access salience. A re-touch never demotes a still-strong slice.
  * :meth:`Witness.decay` — **fade**: recompute every slice's live score from how long it
    has been idle. Monotone non-increasing in elapsed time, never negative.
  * :meth:`Witness.rank` — order ids by current score, highest (most retained) first.
  * :meth:`Witness.budget_evict` — drop the lowest-score slices first until the pool fits
    under its byte ceiling, then stop. Returns the evicted ids.

Fail-soft: the witness is an *optimization* over the pool, never a correctness gate. It
only ever returns ids; the pool is the single source of truth for slice payloads.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from aether_context._log import get_logger

_log = get_logger(__name__)

# --- retention math constants (ported from upstream, atlas coupling stripped) ---
#: At or above this retention score a slice is considered salient ("hardened"): it is
#: the last thing the budget governor will evict. Mirrors the upstream SALIENT_THRESHOLD.
SALIENT_THRESHOLD: float = 0.60
#: Exponential fade rate per unit of idle time. Tuned so a unit-salience slice retains
#: ~37% of its score after ~20 idle units (1 / DEFAULT_DECAY_RATE), i.e. a slow, steady
#: fade rather than a cliff — old-but-salient slices survive a long run.
DEFAULT_DECAY_RATE: float = 0.05


def _clamp_unit(x: float) -> float:
    """Clamp a float into the closed unit interval ``[0.0, 1.0]``."""
    return max(0.0, min(1.0, float(x)))


def retention_score(surprise: float, impact: float, uniqueness: float) -> float:
    """Retention in ``[0,1]`` = geometric mean of the three drivers.

    The geometric mean is deliberate: one weak driver can't be masked by a strong one
    (unlike a sum), so a slice must be salient on every axis to score high. All inputs
    are clamped to ``[0,1]`` first.

      surprise    content density / novelty of the slice, normalized to [0,1]
      impact      query relevance / magnitude of the slice, normalized to [0,1]
      uniqueness  ``1 / (1 + neighbors_in_embedding_space)`` -> rarer = higher
    """
    s = _clamp_unit(surprise)
    i = _clamp_unit(impact)
    u = _clamp_unit(uniqueness)
    return (s * i * u) ** (1.0 / 3.0)


def squash(x: float, scale: float) -> float:
    """``tanh`` squash of a raw magnitude to ``[0,1]`` at a given scale.

    Used to normalize an unbounded magnitude (e.g. a raw relevance distance) into a
    driver for :func:`retention_score`. A non-positive ``scale`` is a safe no-op (0.0).
    """
    if scale <= 0:
        return 0.0
    return math.tanh(abs(x) / scale)


def uniqueness_from_neighbors(neighbor_count: int) -> float:
    """``1 / (1 + max(0, neighbor_count))`` — rarer slices (fewer neighbors) score higher."""
    return 1.0 / (1.0 + max(0, neighbor_count))


@dataclass
class _Entry:
    """Per-slice retention state: a base score anchored at its last touch time.

    The *live* score is ``base * exp(-rate * (now - last_touch))`` — recomputed lazily
    so a slice that has been idle longer has faded further. Only the base score and the
    anchor time are stored; the decayed value is always derived.
    """

    base: float        # score at last_touch (in [0,1])
    last_touch: float  # the ``now`` at which ``base`` was set


class Witness:
    """The +/- fidelity field over slice ids: harden, fade, re-harden, rank, evict.

    Pure bookkeeping over access events. Holds only ``{slice_id -> _Entry}``; never the
    slice payloads (those live in the context pool). Stateless with respect to the pool:
    callers feed it ids + saliences and read back rankings / eviction lists.
    """

    def __init__(self, decay_rate: float = DEFAULT_DECAY_RATE) -> None:
        """Create an empty witness.

        ``decay_rate`` (> 0) sets how fast idle slices fade; the default gives a slow,
        steady fade so old-but-salient slices survive a long run. A non-positive value
        falls back to :data:`DEFAULT_DECAY_RATE`.
        """
        self._decay_rate: float = decay_rate if decay_rate > 0 else DEFAULT_DECAY_RATE
        self._entries: dict[str, _Entry] = {}

    # -- harden / re-harden ----------------------------------------------------
    def touch(self, slice_id: str, salience: float, now: float = 0.0) -> float:
        """**Harden** (or re-harden) ``slice_id`` toward ``salience`` at time ``now``.

        On first touch the slice is registered with ``salience`` (clamped to ``[0,1]``).
        On re-touch the new base is ``max(decayed_current_score, salience)`` so a strong
        re-touch lifts a faded slice back up and a *weak* re-touch never demotes a still-
        strong slice. The anchor time is reset to ``now`` either way (the slice is fresh).

        Returns the slice's new (base) score.
        """
        s = _clamp_unit(salience)
        existing = self._entries.get(slice_id)
        if existing is not None:
            decayed = self._decayed_score(existing, now)
            s = max(decayed, s)
        self._entries[slice_id] = _Entry(base=s, last_touch=float(now))
        return s

    # -- fade ------------------------------------------------------------------
    def decay(self, now: float) -> None:
        """**Fade** every slice: collapse each live (decayed) score into its base at ``now``.

        After this call each slice's stored base equals its decayed value as of ``now`` and
        its anchor is ``now``. The result is monotone non-increasing in elapsed time and
        never negative. Calling repeatedly with non-decreasing ``now`` keeps fading.
        """
        for slice_id, entry in self._entries.items():
            faded = self._decayed_score(entry, now)
            self._entries[slice_id] = _Entry(base=faded, last_touch=float(now))

    def _decayed_score(self, entry: _Entry, now: float) -> float:
        """Live score for an entry as of ``now``: ``base * exp(-rate * max(0, elapsed))``."""
        elapsed = float(now) - entry.last_touch
        if elapsed <= 0.0:
            return entry.base  # no negative-time lift; un-aged slices keep their base
        return entry.base * math.exp(-self._decay_rate * elapsed)

    # -- read ------------------------------------------------------------------
    def score(self, slice_id: str, now: float | None = None) -> float:
        """Current retention score of ``slice_id`` (0.0 if unknown).

        If ``now`` is given, returns the decayed-as-of-``now`` score without mutating
        state; otherwise returns the stored base score (the value as of its last touch
        or the last :meth:`decay`).
        """
        entry = self._entries.get(slice_id)
        if entry is None:
            return 0.0
        if now is None:
            return entry.base
        return self._decayed_score(entry, now)

    def ids(self) -> list[str]:
        """All known slice ids, ordered highest-score first (alias of :meth:`rank`)."""
        return self.rank()

    def rank(self, now: float | None = None) -> list[str]:
        """Slice ids ordered by score, **highest (most retained) first**.

        Ties break on insertion order (Python dict order) for determinism. If ``now`` is
        given, ranks by the decayed-as-of-``now`` score without mutating state.
        """
        scored = [(sid, self.score(sid, now=now)) for sid in self._entries]
        # stable sort by descending score; ties keep dict insertion order
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [sid for sid, _ in scored]

    def forget(self, slice_id: str) -> None:
        """Drop a slice id from the witness. A no-op if it is unknown (safe to call)."""
        self._entries.pop(slice_id, None)

    # -- budget eviction -------------------------------------------------------
    def budget_evict(
        self, ceiling_bytes: int, bytes_per_slice: int, now: float | None = None
    ) -> list[str]:
        """Evict the **lowest-score** slices first until the pool fits under ``ceiling_bytes``.

        Each retained slice costs ``bytes_per_slice``. Slices are dropped from the bottom
        of the ranking (least retained first) and eviction **stops the instant** the
        remaining count fits — so the survivors are always the highest-score slices and
        the pool is exactly at or below the ceiling. The witness drops the evicted ids
        from its own bookkeeping.

        Returns the evicted ids in eviction order (lowest score first). A no-op (``[]``)
        when the pool already fits or ``bytes_per_slice`` is non-positive.
        """
        if bytes_per_slice <= 0:
            return []
        max_slices = max(0, ceiling_bytes // bytes_per_slice)
        ranked = self.rank(now=now)  # highest score first
        if len(ranked) <= max_slices:
            return []
        # keep the top `max_slices`; evict the rest (these are the lowest scores)
        evicted = ranked[max_slices:]
        for sid in evicted:
            del self._entries[sid]
        _log.debug(
            "budget_evict dropped %d slice(s) to fit %d bytes (%d per slice)",
            len(evicted), ceiling_bytes, bytes_per_slice,
        )
        return evicted


__all__ = [
    "Witness",
    "retention_score",
    "squash",
    "uniqueness_from_neighbors",
    "SALIENT_THRESHOLD",
    "DEFAULT_DECAY_RATE",
]
