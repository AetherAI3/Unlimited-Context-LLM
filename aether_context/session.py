"""B5 lifecycle controller — :class:`Session`, the process lifecycle of the engine.

This is the part a user drives:

    from aether_context import Session
    s = Session(model="ollama/qwen2.5", pool_gb=5)
    print(s.run("Build me a full-stack weightlifting tracker app.").text)

It ties the four other parts together into the virtual-memory-for-attention lifecycle:

  * **open** a fresh working window for the task;
  * **stream + encode + fade** — as the model emits (and as input arrives), encode the spill
    into the :class:`~aether_context.context_pool.ContextPool` while the
    :class:`~aether_context.witness.Witness` fades the cold (the pool's governor holds the
    byte budget after every write);
  * **paged reason** — the :class:`~aether_context.slice_loader.Pager` keeps the right slices
    resident, prefetched on a **background thread while the model generates** (the backend's
    HTTP/subprocess call releases the GIL, so the prefetch genuinely overlaps generation);
  * **close** — flush the pool and retain the run's abstracted **artifacts** (text + vector +
    tags) locally for inspection. The session is fully local — nothing ever leaves your machine.

Lifecycle cadence
-----------------
The loop is a simple **CONTINUOUS** (perceive + encode every step) / **EVENT** (act on a real
event) / **VERIFY** (occasional bookkeeping) cadence. The only vector the engine ever stores or
compares is the 256-dim retrieval embedding.

Fail-soft (design law 3)
------------------------
The pager, encoder, and pool are *optimizations*, never correctness dependencies. Any error
inside encode/prefetch/recall is logged and the run continues on the model's native window — a
retrieval hiccup never crashes a long build.
"""
from __future__ import annotations

import threading
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from aether_context._log import get_logger
from aether_context.config import PoolConfig, SessionConfig, reach_tokens
from aether_context.context_pool import ContextPool, Slice, slice_cost_bytes
from aether_context.encoder import StaticEncoder
from aether_context.errors import (
    AetherContextError,
    BackendUnavailable,
    ModelNotPulled,
    OllamaNotRunning,
)
from aether_context.local_llm import LocalLLM, load_model
from aether_context.slice_loader import Pager, SliceKey
from aether_context.tokenizer import from_backend
from aether_context.witness import Witness, retention_score, uniqueness_from_neighbors

logger = get_logger(__name__)

#: Default number of slices the pager keeps resident as the working set this turn.
DEFAULT_RESIDENT_K: int = 8
#: Salience floor for an encoded spill slice (so cold-but-real context is never zero-weighted).
_SPILL_SALIENCE_FLOOR: float = 0.30
#: Salience for an explicitly remembered fact — high, so the witness hardens it strongly.
_REMEMBER_SALIENCE: float = 0.95
#: Topic label assigned to the running reasoning region (the pager's working-set key).
_REASONING_TOPIC: str = "reasoning"
#: Resident in-RAM index estimate, MB per GB of reach (mirrors the README table / CLI's
#: _INDEX_MB_PER_GB: ~145 MB at 5 GB). Used only for the honest status RAM estimate.
_RESIDENT_RAM_MB_PER_GB: int = 29
#: Default transcript export filename (timestamp-free, written under cwd).
_DEFAULT_TRANSCRIPT_NAME: str = "aether-transcript.txt"
#: How much the Extended-Thinking toggle widens the resident working set (mock-honest).
_EXTENDED_RESIDENT_BONUS: int = 8


# ---------------------------------------------------------------------------
# Results & harvest candidates
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HarvestCandidate:
    """An abstracted, durable artifact of a run, retained locally.

    Deliberately tiny and self-contained: ``text`` + its 256-dim retrieval ``vector`` + plain
    ``tags`` — nothing else, no external state.
    """

    text: str
    vector: np.ndarray
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    """The outcome of :meth:`Session.run`.

    Fields:
      text        the model's full generated text for the task
      stages      ordered list of lifecycle stage records (open / stream / page / complete)
      hit_rate    the pager's measured retrieval hit rate over the run
      spilled     number of slices encoded into the pool during the run (encode-on-spill)
      resident    number of slices resident in the pager window at the end
      overflowed  whether the run exceeded the model's native ``context_window``
    """

    text: str
    stages: list[dict[str, Any]]
    hit_rate: float
    spilled: int = 0
    resident: int = 0
    overflowed: bool = False


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
class Session:
    """The engine lifecycle controller: open -> stream+encode+fade -> page -> close.

    Construct with a model spec (or a :class:`~aether_context.local_llm.LocalLLM` object) and
    a pool size; the session builds the local LLM (via
    :func:`~aether_context.local_llm.load_model`), a :class:`ContextPool`, a :class:`Witness`,
    and a :class:`Pager`. The session is fully local and offline — nothing leaves your machine.

    Public surface:
      * :meth:`run` ``(task) -> RunResult`` — the full lifecycle for one task.
      * :meth:`ask` ``(msg) -> str`` — a convenience wrapper returning just the text.
      * :meth:`stream` ``(task) -> Iterator[str]`` — stream the model's chunks, encoding spill.
      * :meth:`remember` / :meth:`recall` — plant and retrieve a load-bearing fact.
      * :meth:`harvest_candidates` — the run's durable artifacts.
      * :meth:`close` + context-manager (``with Session(...) as s:``).
    """

    def __init__(
        self,
        model: "str | LocalLLM",
        pool_gb: int = 5,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        pull: bool = False,
        fallback_to_mock: bool = True,
        pool_dir: "str | Path | None" = None,
        pool_index: str = "flat",
        pool_mode: str = "separate",
        context_window: int | None = None,
        output_tokens: int | None = None,
        resident_k: int = DEFAULT_RESIDENT_K,
        pool_ceiling_bytes: int | None = None,
        session_id: str | None = None,
        **cfg: Any,
    ) -> None:
        self.id: str = session_id or f"sess-{uuid.uuid4().hex[:12]}"
        self._closed: bool = False
        self._resident_k: int = max(1, int(resident_k))
        # Pool sharing discipline. "separate" (default) keeps every search/encode scoped to
        # THIS session's namespace, so two sessions over one dir never see each other's
        # slices. "shared" makes reach global (search/encode with session=None) so a named or
        # persistent pool can be read across sessions. The PoolConfig records the same mode.
        self.pool_mode: str = pool_mode
        # Extended-Thinking toggle (honest): in the mock it only widens the resident set and
        # surfaces in status; it is never a silent capability claim.
        self.extended: bool = False
        # Conversation transcript: one (role, text) tuple per ask()/run() turn, used by export().
        self._transcript: list[tuple[str, str]] = []
        # -- the model (THE WRAPPER) ------------------------------------------
        self.local_llm: LocalLLM = self._build_model(
            model,
            pull=pull,
            context_window=context_window,
            output_tokens=output_tokens,
            fallback_to_mock=fallback_to_mock,
            **cfg,
        )
        self._count_tokens = from_backend(self.local_llm)

        # -- session config (window fractions etc.) ---------------------------
        self.config = SessionConfig(
            model=model, system=system, max_tokens=max_tokens
        )

        # -- the pool ("disk") ------------------------------------------------
        pool_config = PoolConfig(
            pool_gb=pool_gb,
            mode=pool_mode,
            index=pool_index,
            dir=Path(pool_dir) if pool_dir is not None else PoolConfig().dir,
        )
        self.pool: ContextPool = ContextPool(
            pool_config, ceiling_bytes=pool_ceiling_bytes
        )
        # Retain the reach (GB) for honest status reporting without reaching into pool internals.
        self.pool_gb: int = int(pool_config.pool_gb)

        # -- encoder ("encode-on-spill") --------------------------------------
        self.encoder: StaticEncoder = StaticEncoder(dim=pool_config.dim)

        # -- witness (page-replacement) + pager (the pager) -------------------
        # In "shared" mode the cold path searches globally (session=None) so the resident
        # window can draw on slices from any session; "separate" keeps the default
        # session-scoped cold path (key.session) so namespaces never bleed.
        self.witness: Witness = Witness()
        self.pager: Pager = Pager(
            self.pool,
            self.encoder,
            default_k=self._resident_k,
            retrieve_fn=self._cold_retrieve,
        )

        # -- run state --------------------------------------------------------
        self._harvest: list[HarvestCandidate] = []
        self._spill_seq: int = 0
        self._clock: float = 0.0

    # -- construction helpers -------------------------------------------------
    @staticmethod
    def _build_model(
        model: "str | LocalLLM",
        *,
        pull: bool,
        context_window: int | None,
        output_tokens: int | None,
        fallback_to_mock: bool = True,
        **cfg: Any,
    ) -> LocalLLM:
        """Resolve ``model`` to a :class:`LocalLLM`, forwarding the relevant kwargs.

        A bare :class:`LocalLLM` object is returned unchanged. A spec string is dispatched to
        :func:`~aether_context.local_llm.load_model`; ``context_window`` / ``output_tokens`` /
        ``pull`` are forwarded only when meaningful (so the mock honors a tiny window and a
        long output, and Ollama honors ``pull``).

        When ``fallback_to_mock`` is true (the default) and the requested backend cannot be
        loaded (daemon down, model not pulled, optional backend not installed), the engine
        degrades to the deterministic mock model so a clean-clone / offline run never crashes.
        The fallback is announced via :mod:`warnings` (visible on stderr regardless of logging
        config) so it is never silent. Pass ``fallback_to_mock=False`` to fail loudly instead.
        """
        if not isinstance(model, str):
            return model  # bring-your-own backend
        kw: dict[str, Any] = dict(cfg)
        if pull:
            kw["pull"] = True
        if context_window is not None:
            kw["context_window"] = context_window
        if output_tokens is not None:
            # only the mock backend accepts output_tokens; forward it conditionally.
            if model.strip() == "mock":
                kw["output_tokens"] = output_tokens
        try:
            return load_model(model, **kw)
        except (OllamaNotRunning, ModelNotPulled, BackendUnavailable) as exc:
            if not fallback_to_mock or model.strip() == "mock":
                raise
            warnings.warn(
                f"Could not load model {model!r} ({exc}); falling back to the deterministic "
                "mock model. Output will be SYNTHETIC. Pass fallback_to_mock=False to fail "
                "instead, or start the backend (e.g. `ollama serve` + `ollama pull <model>`).",
                RuntimeWarning,
                stacklevel=2,
            )
            mock_kw: dict[str, Any] = {}
            if context_window is not None:
                mock_kw["context_window"] = context_window
            if output_tokens is not None:
                mock_kw["output_tokens"] = output_tokens
            return load_model("mock", **mock_kw)

    # -- properties -----------------------------------------------------------
    @property
    def closed(self) -> bool:
        """Whether the session has been closed (pool flushed, artifacts retained)."""
        return self._closed

    @property
    def context_window(self) -> int:
        """The model's native token window (the size the engine pages *around*)."""
        try:
            return int(self.local_llm.context_window)
        except (AttributeError, ValueError, TypeError):
            return 8192

    def _key(self, topic: str = _REASONING_TOPIC) -> SliceKey:
        """The pager key for this session's region under ``topic``."""
        return SliceKey(session=self.id, topic=topic)

    def _scope(self) -> str | None:
        """The session id this session's searches are scoped to, or ``None`` when shared.

        ``"separate"`` (default) scopes every search/encode to :attr:`id` so two sessions
        over one pool dir stay isolated. ``"shared"`` returns ``None`` so the search spans
        every session's slices (global reach across sessions).
        """
        return None if self.pool_mode == "shared" else self.id

    def _cold_retrieve(self, key: SliceKey, query_vec: np.ndarray, k: int) -> list[Slice]:
        """Pager cold path honoring the session's pool mode (shared -> global search)."""
        return self.pool.search(query_vec, k, session=self._scope())

    # -- plant / recover a fact -----------------------------------------------
    def remember(self, text: str, *, tags: dict[str, Any] | None = None) -> Slice | None:
        """Encode ``text`` as a high-salience slice into the pool (a load-bearing fact).

        This is how a durable constraint is established before a long run so it survives the
        overflow: it is encoded into the pool immediately and hardened in the witness. Returns
        the stored :class:`Slice` (or ``None`` if encoding fails — fail-soft).
        """
        return self._encode_slice(text, salience=_REMEMBER_SALIENCE, tags=tags or {})

    def recall(self, query: str, k: int = DEFAULT_RESIDENT_K) -> list[Slice]:
        """Retrieve the slices nearest to ``query`` from this session's pool region.

        The hot path of recovery: embed ``query``, search the pool scoped to this session, and
        return the nearest slices. Fail-soft — an encoder/search error yields ``[]`` (the run
        continues on the model's native window).
        """
        return self._recall_local(query, k)

    def _recall_local(self, query: str, k: int) -> list[Slice]:
        try:
            qvec = self.encoder.encode(query)
        except AetherContextError as exc:
            logger.warning("recall encode failed (%s); returning no local hits", exc)
            return []
        try:
            scoped = self.pool.search(qvec, k, session=self._scope())
            if scoped:
                return scoped
            # Reopen case: a fresh Session over an existing pool dir has a new session id, so
            # the prior run's slices live under a different namespace. Fall back to a global
            # search so a disk-resident fact is still recoverable after a close + reopen.
            # (In shared mode the scoped search is already global, so this is a harmless re-run.)
            return self.pool.search(qvec, k, session=None)
        except AetherContextError as exc:
            logger.warning("recall pool search failed (%s); returning no local hits", exc)
            return []

    # -- encode-on-spill ------------------------------------------------------
    def _encode_slice(
        self, text: str, *, salience: float, tags: dict[str, Any]
    ) -> Slice | None:
        """Encode ``text`` into a pool slice and add it (fail-soft).

        Returns the stored slice, or ``None`` if the encoder or the pool add fails — in which
        case the run simply proceeds without that slice (the pager is an optimization). The
        witness is touched so the slice participates in retention ranking.
        """
        text = text.strip()
        if not text:
            return None
        try:
            vec = self.encoder.encode(text)
        except Exception as exc:  # noqa: BLE001 - fail-soft: encoder is an optimization
            logger.warning("encode-on-spill failed (%s); slice dropped, run continues", exc)
            return None
        self._spill_seq += 1
        sid = f"{self.id}:slice:{self._spill_seq}"
        tokens = self._count_tokens(text)
        score = self._salience(text, salience)
        meta = dict(tags)
        sl = Slice(
            id=sid, session=self.id, vector=np.asarray(vec, dtype=np.float32),
            text=text, tokens=int(tokens), meta=meta, score=score,
        )
        try:
            self.pool.add(sl)
        except AetherContextError as exc:
            logger.warning("pool add failed (%s); slice dropped, run continues", exc)
            return None
        self._clock += 1.0
        self.witness.touch(sid, salience=score, now=self._clock)
        # Every encoded durable artifact (planted fact or spill) is a run artifact:
        # text + its 256-dim retrieval vector + tags, retained locally.
        self._harvest.append(
            HarvestCandidate(text=text, vector=sl.vector.copy(), tags=dict(meta))
        )
        return sl

    def _salience(self, text: str, base: float) -> float:
        """Retention salience for a slice: SURPRISE x IMPACT x UNIQUENESS, floored.

        Retention math: surprise ~ content density (token variety),
        impact ~ the caller's base weight, uniqueness ~ 1/(1+near-neighbors already in pool).
        Floored at :data:`_SPILL_SALIENCE_FLOOR` so real-but-cold context is never zeroed.
        """
        words = text.split()
        density = min(1.0, len(set(words)) / 64.0) if words else 0.0
        neighbors = self._approx_neighbor_count(text)
        uniqueness = uniqueness_from_neighbors(neighbors)
        score = retention_score(surprise=density, impact=base, uniqueness=uniqueness)
        return max(_SPILL_SALIENCE_FLOOR, float(score))

    def _approx_neighbor_count(self, text: str) -> int:
        """Cheap near-neighbor count for uniqueness scoring (fail-soft, best-effort)."""
        try:
            qvec = self.encoder.encode(text)
            hits = self.pool.search(qvec, k=4, session=self._scope())
        except AetherContextError:
            return 0
        # count strongly-similar resident slices (cosine > 0.9)
        return sum(1 for h in hits if float(np.dot(qvec, h.vector)) > 0.9)

    # -- the lifecycle: run ---------------------------------------------------
    def run(self, task: str) -> RunResult:
        """Run the full lifecycle for ``task`` and return a :class:`RunResult`.

        Stages: **open** a fresh window, **stream** the model while encoding spill and paging
        on a side thread, **page** the resident working set, then return. The pool budget is
        held after every encoded write. Any pager/encoder/pool error is logged and the run
        continues on the model's native window (fail-soft).
        """
        if self._closed:
            raise AetherContextError(
                "run() called on a closed Session.",
                hint="Open a new Session; a closed one has flushed its pool.",
            )
        stages: list[dict[str, Any]] = [
            {"stage": "open", "task_tokens": self._count_tokens(task)}
        ]

        text, spilled, overflowed = self._stream_and_encode(task, stages)

        # paged reason: warm the working set from the final reasoning text (fail-soft).
        self._page_working_set(task + "\n" + text, stages)

        # Record the turn for export(): the user task then the model's reply (ask() funnels
        # through run(), so logging here covers both without double-counting).
        self._transcript.append(("user", task))
        self._transcript.append(("assistant", text))

        stages.append({"stage": "complete", "out_tokens": self._count_tokens(text)})
        return RunResult(
            text=text,
            stages=stages,
            hit_rate=self.pager.hit_rate(),
            spilled=spilled,
            resident=self.pager.warm_count,
            overflowed=overflowed,
        )

    def _stream_and_encode(
        self, task: str, stages: list[dict[str, Any]]
    ) -> tuple[str, int, bool]:
        """Stream the model for ``task`` while encoding spill and prefetching concurrently.

        Returns ``(full_text, slices_spilled, overflowed)``. The prefetch runs on a background
        thread so it overlaps generation (the backend call releases the GIL). Spill is encoded
        whenever the running transcript crosses the window's trigger fraction.
        """
        window = self.context_window
        trigger_chars = int(window * self.config.trigger_fraction) * 4  # chars/4 budget
        chunks: list[str] = []
        spilled = 0
        overflowed = False
        pending = ""  # text accumulated since the last spill
        prefetch_thread: threading.Thread | None = None

        try:
            stream = self.local_llm.generate(
                task, system=self.config.system, max_tokens=self.config.max_tokens
            )
        except AetherContextError as exc:
            # a backend that fails to even start streaming: surface the typed error.
            logger.error("model generate failed to start: %s", exc)
            raise

        for chunk in stream:
            chunks.append(chunk)
            pending += chunk
            # when the running transcript crosses the trigger, encode the spill and launch a
            # background prefetch from what we are reasoning about now (overlaps generation).
            if trigger_chars > 0 and len(pending) >= trigger_chars:
                overflowed = True
                if self._encode_slice(
                    pending, salience=_SPILL_SALIENCE_FLOOR, tags={"kind": "spill"}
                ):
                    spilled += 1
                prefetch_thread = self._launch_prefetch(pending, prefetch_thread)
                pending = ""

        self._join(prefetch_thread)

        # encode the final remainder as a slice too (so the tail is recoverable).
        if pending.strip():
            if self._encode_slice(
                pending, salience=_SPILL_SALIENCE_FLOOR, tags={"kind": "spill"}
            ):
                spilled += 1

        # fade the cold after the run (witness page-replacement step).
        self._fade()
        stages.append({"stage": "stream", "spilled": spilled, "overflowed": overflowed})
        return "".join(chunks), spilled, overflowed

    def _launch_prefetch(
        self, reasoning_text: str, prior: threading.Thread | None
    ) -> threading.Thread | None:
        """Start a background prefetch from ``reasoning_text`` (fail-soft); join any prior.

        The pager core is single-threaded; concurrency is the caller's job. We join the prior
        thread first (so prefetches never pile up) then launch a fresh daemon thread. Any error
        inside the thread is swallowed there — the pager is an optimization.
        """
        self._join(prior)

        def _work() -> None:
            try:
                self.pager.prefetch_from(self._key(), reasoning_text)
            except Exception as exc:  # noqa: BLE001 - fail-soft: never crash a run
                logger.warning("background prefetch failed (%s); run continues", exc)

        thread = threading.Thread(target=_work, daemon=True, name=f"prefetch-{self.id}")
        thread.start()
        return thread

    @staticmethod
    def _join(thread: threading.Thread | None) -> None:
        """Join a prefetch thread if present (bounded; it is pure CPU/IO over the pool)."""
        if thread is not None and thread.is_alive():
            thread.join(timeout=10.0)

    def _page_working_set(self, reasoning_text: str, stages: list[dict[str, Any]]) -> None:
        """Warm the resident working set from the final reasoning (fail-soft, synchronous)."""
        try:
            self.pager.prefetch_from(self._key(), reasoning_text)
        except Exception as exc:  # noqa: BLE001 - fail-soft
            logger.warning("final page-in failed (%s); run continues on native window", exc)
        stages.append(
            {
                "stage": "page",
                "resident": self.pager.warm_count,
                "hit_rate": self.pager.hit_rate(),
            }
        )

    def _fade(self) -> None:
        """Advance the witness clock and fade cold slices (page-replacement). Fail-soft."""
        try:
            self._clock += 1.0
            self.witness.decay(self._clock)
        except (ValueError, ArithmeticError) as exc:  # defensive; decay is pure math
            logger.debug("witness decay skipped (%s)", exc)

    # -- convenience surface --------------------------------------------------
    def ask(self, msg: str) -> str:
        """Run ``msg`` through the full lifecycle and return just the model's text."""
        return self.run(msg).text

    def stream(self, task: str) -> Iterator[str]:
        """Stream the model's chunks for ``task``, encoding spill as it flows.

        Yields each text chunk as it arrives (so a UI can render live), while the same
        encode-on-spill discipline runs underneath. The pool budget is held throughout. After
        the stream is exhausted the witness fades the cold.
        """
        if self._closed:
            raise AetherContextError(
                "stream() called on a closed Session.",
                hint="Open a new Session; a closed one has flushed its pool.",
            )
        window = self.context_window
        trigger_chars = int(window * self.config.trigger_fraction) * 4
        pending = ""
        prefetch_thread: threading.Thread | None = None
        try:
            stream = self.local_llm.generate(
                task, system=self.config.system, max_tokens=self.config.max_tokens
            )
        except AetherContextError as exc:
            logger.error("model generate failed to start: %s", exc)
            raise
        for chunk in stream:
            pending += chunk
            if trigger_chars > 0 and len(pending) >= trigger_chars:
                self._encode_slice(
                    pending, salience=_SPILL_SALIENCE_FLOOR, tags={"kind": "spill"}
                )
                prefetch_thread = self._launch_prefetch(pending, prefetch_thread)
                pending = ""
            yield chunk
        self._join(prefetch_thread)
        if pending.strip():
            self._encode_slice(
                pending, salience=_SPILL_SALIENCE_FLOOR, tags={"kind": "spill"}
            )
        self._fade()

    def harvest_candidates(self) -> list[HarvestCandidate]:
        """The run's abstracted durable artifacts (text + 256-dim vector + tags).

        Retained locally; returned as a shallow copy so the caller cannot mutate session state.
        """
        return list(self._harvest)

    # -- clear / export / extended-thinking / status --------------------------
    def clear(self, scope: str = "session") -> int:
        """Clear resident and (optionally) externalized state. Returns slices removed.

        Two honest scopes:

        * ``"resident"`` — empty only the in-memory resident window (the pager's warm set).
          The reachable pool on disk is untouched, so this is :meth:`/new`: a fresh window
          over the same reach. Always returns ``0`` (no pool slices were dropped).
        * ``"session"`` (default) — also drop the slices THIS session externalized into the
          pool. In ``"shared"`` mode the pool is global, so this empties the whole pool
          (every session's slices). Returns the count of pool slices removed.

        Resetting the witness keeps retention ranking consistent with the now-smaller pool.
        Fail-soft: an empty/absent pool simply removes nothing.
        """
        if scope not in ("resident", "session"):
            raise AetherContextError(
                f"clear scope must be 'resident' or 'session', got {scope!r}.",
                hint="Use scope='resident' (window only) or scope='session' (window + slices).",
            )
        self.pager.reset()
        if scope == "resident":
            return 0
        # session scope: drop this session's pool slices (or all of them when shared).
        return self.pool.clear_session(self._scope())

    def export(self, path: str | None = None) -> str:
        """Write the conversation transcript to ``path`` and return the path written.

        With no ``path`` the transcript is written to ``aether-transcript.txt`` under the
        current working directory (timestamp-free, so re-export overwrites in place). Each
        turn is rendered as ``role: text`` on its own block. Returns the resolved path as a
        string so a caller (the REPL ``/export``) can report exactly where it landed.
        """
        target = Path(path) if path else Path.cwd() / _DEFAULT_TRANSCRIPT_NAME
        lines = [f"{role}: {text}" for role, text in self._transcript]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return str(target)

    def toggle_extended(self) -> bool:
        """Flip the Extended-Thinking toggle and return its new value (honest, mock-safe).

        In the mock backend this only widens the resident working set (more slices paged in
        per turn) and surfaces in :meth:`status_dict`; it never silently changes the model's
        real capability. Returns the toggle's new state so a REPL can report ``on``/``off``.
        """
        self.extended = not self.extended
        bonus = _EXTENDED_RESIDENT_BONUS if self.extended else 0
        self.pager.default_k = self._resident_k + bonus
        return self.extended

    def status_dict(self) -> dict[str, Any]:
        """A snapshot of the session's state for the ``status`` command (all honest).

        Fields: ``pool_gb`` (reach in GB), ``slices_used`` / ``capacity`` (governor budget),
        ``reach_tokens``, ``hit_rate`` (the pager's measured rate), ``resident_ram_mb``
        (estimate ~29 MB/GB of reach), ``pool_mode``, ``index`` (the resolved kind actually
        in use), ``model`` (the backend name), and ``extended`` (the toggle state).
        """
        stats = self.pool.stats()
        cost = slice_cost_bytes(int(stats["dim"]))
        capacity = self.pool.ceiling_bytes // cost if cost else 0
        return {
            "pool_gb": self.pool_gb,
            "slices_used": int(stats["count"]),
            "capacity": int(capacity),
            "reach_tokens": reach_tokens(self.pool_gb),
            "hit_rate": self.pager.hit_rate(),
            "resident_ram_mb": self.pool_gb * _RESIDENT_RAM_MB_PER_GB,
            "pool_mode": self.pool_mode,
            "index": stats["index"],
            "model": getattr(self.local_llm, "name", str(self.config.model)),
            "extended": self.extended,
        }

    # -- close + context manager ----------------------------------------------
    def close(self) -> None:
        """Flush the pool and release its mmap. Idempotent.

        Closing twice is a no-op. After close the session cannot ``run``/``stream`` again; the
        run's artifacts remain available via :meth:`harvest_candidates` until then.
        """
        if self._closed:
            return
        self._closed = True
        self._flush_pool()

    def _flush_pool(self) -> None:
        """Flush + release the pool's mmap (fail-soft; close must never raise)."""
        try:
            self.pool.close()
        except AetherContextError as exc:
            logger.warning("pool close failed (%s); state may be partially flushed", exc)

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


__all__ = [
    "Session",
    "RunResult",
    "HarvestCandidate",
    "DEFAULT_RESIDENT_K",
]
