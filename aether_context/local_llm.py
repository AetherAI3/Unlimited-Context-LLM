# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""THE WRAPPER — backend-agnostic local-LLM adapters.

This is the surface a user actually touches. The whole engine talks to *one* protocol
(:class:`LocalLLM`); every backend (Ollama, llama.cpp, HF, Mock) satisfies it so the
engine never special-cases a backend.

Design laws honored here:
  * **Simple** — one spec string picks a backend (``"ollama/qwen2.5"``, ``"mock"``…).
  * **Light core** — the primary Ollama path uses the Python standard library only
    (``urllib``); no extra dependency for ``pip install aether-context``.
  * **Forgiving** — daemon down / model missing raise *typed* errors from
    :mod:`aether_context.errors`, each with an actionable ``.hint``. Metadata probes are
    fail-soft (a failed ``/api/show`` degrades ``context_window`` to a fallback, never
    raises into a run).
  * **Streaming** — ``generate`` yields text chunks so the pager can prefetch the next
    slices *while* the model is still talking.

No ``print()`` (use the logging seam), no bare ``except`` (catch specific, re-wrap as a
typed error).
"""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Iterator, Optional, Protocol, runtime_checkable

from aether_context._log import get_logger
from aether_context.errors import (
    AetherContextError,
    BackendUnavailable,
    ModelNotPulled,
    OllamaNotRunning,
)
from aether_context.tokenizer import estimate

_log = get_logger(__name__)

#: Fallback context window (tokens) when a backend cannot report its own.
DEFAULT_CONTEXT_WINDOW = 8192
#: Default Ollama daemon host.
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
#: Recognised backends in a spec string.
_BACKENDS = ("ollama", "llamacpp", "hf", "mock")
#: HTTP timeout (seconds) for Ollama requests. Generation can be slow → generous.
_OLLAMA_TIMEOUT = 600
#: Best-effort metadata probe timeout (seconds) — short; failure is non-fatal.
_OLLAMA_PROBE_TIMEOUT = 5


# ---------------------------------------------------------------------------
# The protocol every adapter satisfies.
# ---------------------------------------------------------------------------
@runtime_checkable
class LocalLLM(Protocol):
    """Backend-agnostic local-model contract.

    ``generate`` **streams** text chunks (yield once with the full text if a backend
    cannot stream — it still works, you just lose the free prefetch concurrency).
    """

    name: str

    @property
    def context_window(self) -> int:
        """Token window the model exposes this turn (read-only; may be lazily probed)."""
        ...

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        stop: Optional[list[str]] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Stream model output for ``prompt`` as a sequence of text chunks."""
        ...

    def count_tokens(self, text: str) -> int:
        """Return the token count of ``text`` for budget math."""
        ...


# ---------------------------------------------------------------------------
# Spec parsing.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """Parsed model spec: a backend, a model ref, and pass-through options."""

    backend: str
    ref: str
    options: dict = field(default_factory=dict)


_SPEC_HINT = (
    "Use one of: 'ollama/qwen2.5' (or a bare name -> ollama), "
    "'llamacpp:/path/to/model.gguf', 'hf/org/model', or 'mock'."
)


def parse_spec(spec: str) -> ModelSpec:
    """Parse a spec string into a :class:`ModelSpec`.

    Grammar (one obvious format)::

        ollama/<name>            -> Ollama (tags ok: ollama/llama3.1:8b)
        <name>                   -> bare name assumed Ollama
        llamacpp:<path>          -> llama.cpp over a .gguf (first ':' splits; drive ':' kept)
        hf/<org>/<model>         -> HF transformers
        mock                     -> built-in deterministic model

    Raises :class:`~aether_context.errors.BackendUnavailable` (carrying a ``.hint`` with
    the format) for anything unparseable or an unknown backend.
    """
    if not isinstance(spec, str):
        raise BackendUnavailable(
            f"Model spec must be a string (backend/ref). Got: {type(spec).__name__}",
            hint=_SPEC_HINT,
        )
    text = spec.strip()
    if not text:
        raise BackendUnavailable(
            "Model spec is empty. Must be backend/ref "
            "(e.g. ollama/qwen2.5, llamacpp:/path/model.gguf, hf/org/model, mock).",
            hint=_SPEC_HINT,
        )

    # mock — exact bare token.
    if text == "mock":
        return ModelSpec(backend="mock", ref="mock")

    # llamacpp:<path> — split on the FIRST colon only so a Windows drive ':' survives.
    if text.startswith("llamacpp:"):
        ref = text[len("llamacpp:"):]
        if not ref:
            raise BackendUnavailable(
                "llamacpp spec needs a path: 'llamacpp:/path/to/model.gguf'",
                hint=_SPEC_HINT,
            )
        return ModelSpec(backend="llamacpp", ref=ref)

    # slash-prefixed backends: ollama/<ref>, hf/<org/model>.
    if "/" in text:
        head, ref = text.split("/", 1)
        if head == "ollama":
            return ModelSpec(backend="ollama", ref=ref)
        if head == "hf":
            return ModelSpec(backend="hf", ref=ref)
        if head in _BACKENDS:
            return ModelSpec(backend=head, ref=ref)
        raise BackendUnavailable(
            f"Unknown backend '{head}' in spec '{spec}'. "
            "Must be backend/ref (e.g. ollama/qwen2.5, llamacpp:/path/model.gguf, "
            "hf/org/model, mock).",
            hint=_SPEC_HINT,
        )

    # Bare name -> assumed Ollama (documented convenience; see docs/local-models.md).
    return ModelSpec(backend="ollama", ref=text)


# ---------------------------------------------------------------------------
# load_model — the one public entry point.
# ---------------------------------------------------------------------------
def load_model(spec: "str | LocalLLM", **kw: object) -> LocalLLM:
    """Resolve ``spec`` to a :class:`LocalLLM`.

    * A :class:`LocalLLM` object is returned unchanged (bring-your-own backend).
    * A spec string is parsed and dispatched to the matching adapter; extra ``**kw`` are
      forwarded to the adapter constructor.

    Raises :class:`~aether_context.errors.BackendUnavailable` for an unknown backend or a
    backend whose optional dependency is not installed.
    """
    # Already a backend object? Pass it straight through (duck-typed protocol).
    if not isinstance(spec, str) and isinstance(spec, LocalLLM):
        return spec

    if not isinstance(spec, str):
        raise BackendUnavailable(
            f"model must be a spec string or a LocalLLM object. Got: {type(spec).__name__}",
            hint=_SPEC_HINT,
        )

    parsed = parse_spec(spec)
    backend = parsed.backend
    if backend == "mock":
        return MockLLM(name=parsed.ref, **kw)  # type: ignore[arg-type]
    if backend == "ollama":
        return OllamaLLM(parsed.ref, **kw)  # type: ignore[arg-type]
    if backend == "llamacpp":
        return LlamaCppLLM(parsed.ref, **kw)  # type: ignore[arg-type]
    if backend == "hf":
        return HFLLM(parsed.ref, **kw)  # type: ignore[arg-type]
    raise BackendUnavailable(
        f"Unknown backend '{backend}'.", hint=_SPEC_HINT
    )


# ---------------------------------------------------------------------------
# MockLLM — deterministic, dependency-free.
# ---------------------------------------------------------------------------
#: Vocabulary the mock draws from (deterministically) to build pseudo-output.
_MOCK_WORDS = (
    "plan step build module function class test verify refactor encode slice pool "
    "window context retrieve prefetch witness fade harden session token vector index "
    "the and then so we now next done note check ok yes done finally indeed thus"
).split()
#: Mock chunk width in characters (>1 chunk for any non-trivial output → streaming).
_MOCK_CHUNK_CHARS = 32


@dataclass
class MockLLM:
    """Deterministic, offline, zero-dependency model.

    Output is derived from a hash of ``(prompt, system)`` so the same input always
    produces the same text — letting tests assert properties and the bench run a hermetic
    baseline. ``output_tokens`` controls *length* independently of ``context_window`` so a
    test/bench can force overflow with a tiny window and a long generation.
    """

    name: str = "mock"
    context_window: int = DEFAULT_CONTEXT_WINDOW
    output_tokens: int = 64

    def _build_text(self, prompt: str, system: Optional[str]) -> str:
        seed_src = f"{system or ''}\x00{prompt}".encode("utf-8")
        digest = hashlib.sha256(seed_src).digest()
        # Expand the 32-byte digest deterministically to as many words as we need.
        words: list[str] = []
        i = 0
        # ~6 chars per word incl. space; target enough words for output_tokens (chars/4).
        target_words = max(1, (self.output_tokens * 4) // 6)
        while len(words) < target_words:
            # rehash with a counter so we never run out of entropy
            block = hashlib.sha256(digest + i.to_bytes(4, "big")).digest()
            for b in block:
                words.append(_MOCK_WORDS[b % len(_MOCK_WORDS)])
                if len(words) >= target_words:
                    break
            i += 1
        return " ".join(words)

    @property
    def is_streaming(self) -> bool:
        """True iff a representative ``generate`` yields more than one chunk."""
        return len(self._build_text("probe", None)) > _MOCK_CHUNK_CHARS

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        stop: Optional[list[str]] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Yield deterministic pseudo-output in fixed-width chunks (streaming)."""
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
        text = self._build_text(prompt, system)

        # Apply max_tokens cap (chars/4 estimate → char budget).
        if max_tokens is not None and max_tokens >= 0:
            char_budget = max_tokens * 4
            text = text[:char_budget]

        # Apply stop sequences: truncate at the earliest occurrence of any stop string.
        if stop:
            cut = len(text)
            for s in stop:
                if s:
                    idx = text.find(s)
                    if idx != -1:
                        cut = min(cut, idx)
            text = text[:cut]

        for start in range(0, len(text), _MOCK_CHUNK_CHARS):
            yield text[start:start + _MOCK_CHUNK_CHARS]

    def count_tokens(self, text: str) -> int:
        """Estimate token count (chars/4) — the backend-agnostic budget rule."""
        return estimate(text)


# ---------------------------------------------------------------------------
# OllamaLLM — primary path, stdlib urllib only.
# ---------------------------------------------------------------------------
class OllamaLLM:
    """Ollama adapter over HTTP using only the standard library (``urllib``).

    Talks to ``<host>/api/chat`` with ``stream=true``. ``context_window`` is auto-detected
    from ``<host>/api/show`` metadata (``model_info.*context_length`` / ``num_ctx``),
    falling back to :data:`DEFAULT_CONTEXT_WINDOW` if the probe fails (fail-soft). Token
    counts use the chars/4 estimate (Ollama exposes no count endpoint).
    """

    def __init__(
        self,
        ref: str,
        *,
        host: str = DEFAULT_OLLAMA_HOST,
        context_window: Optional[int] = None,
        pull: bool = False,
        model_options: Optional[dict] = None,
    ) -> None:
        self.name: str = ref
        self.host: str = host.rstrip("/")
        self._pull: bool = pull
        self._model_options: dict = dict(model_options or {})
        # Lazily resolved; an explicit value wins, else probe (fail-soft), else fallback.
        self._context_window: Optional[int] = context_window

    # -- metadata (fail-soft) ------------------------------------------------
    @property
    def context_window(self) -> int:
        """Token window, auto-detected via ``/api/show`` (fallback on any failure)."""
        if self._context_window is None:
            self._context_window = self._probe_context_window()
        return self._context_window

    def _probe_context_window(self) -> int:
        """Best-effort ``/api/show`` probe. Never raises — degrades to the fallback."""
        try:
            payload = json.dumps({"model": self.name}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.host}/api/show",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=_OLLAMA_PROBE_TIMEOUT) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return self._extract_context_length(body)
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as exc:
            _log.debug("ollama /api/show probe failed, using fallback window: %s", exc)
            return DEFAULT_CONTEXT_WINDOW

    @staticmethod
    def _extract_context_length(body: dict) -> int:
        """Pull a context length out of /api/show metadata; fallback if absent."""
        info = body.get("model_info") or {}
        for key, value in info.items():
            if key.endswith("context_length") and isinstance(value, int) and value > 0:
                return value
        params = body.get("parameters")
        if isinstance(params, str):
            for line in params.splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[0] == "num_ctx" and parts[1].isdigit():
                    return int(parts[1])
        return DEFAULT_CONTEXT_WINDOW

    # -- generation ----------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        stop: Optional[list[str]] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Stream chunks from ``/api/chat``. Typed, forgiving errors on failure."""
        if self._pull:
            self._ensure_pulled()
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        options = dict(self._model_options)
        if stop:
            options["stop"] = list(stop)
        if max_tokens is not None:
            options["num_predict"] = int(max_tokens)

        body: dict = {"model": self.name, "messages": messages, "stream": True}
        if options:
            body["options"] = options

        yield from self._stream_chat(body)

    def _stream_chat(self, body: dict) -> Iterator[str]:
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=_OLLAMA_TIMEOUT)
        except urllib.error.HTTPError as exc:
            raise self._http_error_to_typed(exc) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise OllamaNotRunning(
                f"Could not reach the Ollama daemon at {self.host}: {exc}"
            ) from exc

        with resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    _log.debug("skipping non-JSON ollama stream line: %s", exc)
                    continue
                if obj.get("error"):
                    raise ModelNotPulled(
                        f"Ollama error for model '{self.name}': {obj['error']}"
                    )
                chunk = (obj.get("message") or {}).get("content")
                if chunk:
                    yield chunk
                if obj.get("done"):
                    break

    def _http_error_to_typed(self, exc: urllib.error.HTTPError) -> AetherContextError:
        """Map an Ollama HTTP error to a typed, hinted aether-context error."""
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except OSError:
            detail = ""
        if exc.code == 404 or "not found" in detail.lower():
            return ModelNotPulled(
                f"Ollama model '{self.name}' is not pulled: {detail or exc}",
                hint=f"Pull it first: `ollama pull {self.name}` (or pass pull=True).",
            )
        return OllamaNotRunning(
            f"Ollama returned HTTP {exc.code} from {self.host}: {detail or exc}"
        )

    def _ensure_pulled(self) -> None:
        """Best-effort ``/api/pull`` when ``pull=True``. Typed error on failure."""
        payload = json.dumps({"model": self.name, "stream": False}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/pull",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_OLLAMA_TIMEOUT) as resp:
                resp.read()
        except urllib.error.HTTPError as exc:
            raise self._http_error_to_typed(exc) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise OllamaNotRunning(
                f"Could not reach the Ollama daemon at {self.host} to pull "
                f"'{self.name}': {exc}"
            ) from exc

    # -- token counting ------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        """Estimate token count (chars/4) — Ollama has no count endpoint."""
        return estimate(text)


# ---------------------------------------------------------------------------
# LlamaCppLLM — guarded import of llama_cpp.
# ---------------------------------------------------------------------------
class LlamaCppLLM:
    """llama.cpp adapter (``llama-cpp-python``). Import-guarded.

    Requires the ``[llamacpp]`` extra. Constructing it without the dependency raises a
    typed :class:`~aether_context.errors.BackendUnavailable` with the install hint.
    """

    def __init__(
        self,
        model_path: str,
        *,
        context_window: Optional[int] = None,
        model_options: Optional[dict] = None,
    ) -> None:
        try:
            from llama_cpp import Llama  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BackendUnavailable(
                f"The llama.cpp backend needs 'llama-cpp-python': {exc}",
                hint="Install it: pip install \"aether-context[llamacpp]\"",
            ) from exc

        opts = dict(model_options or {})
        n_ctx = opts.pop("n_ctx", context_window or 0)  # 0 -> llama.cpp auto-detects
        try:
            self._llama = Llama(model_path=model_path, n_ctx=n_ctx, **opts)
        except (OSError, ValueError) as exc:
            raise BackendUnavailable(
                f"Could not load gguf at '{model_path}': {exc}",
                hint="Check the .gguf path and that model_options are valid for llama.cpp.",
            ) from exc

        self.name: str = model_path
        detected = getattr(self._llama, "n_ctx", None)
        try:
            ctx = int(detected()) if callable(detected) else int(detected or 0)
        except (TypeError, ValueError):
            ctx = 0
        self.context_window: int = context_window or ctx or DEFAULT_CONTEXT_WINDOW

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        stop: Optional[list[str]] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Stream chunks via ``create_chat_completion(stream=True)``."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict = {"messages": messages, "stream": True}
        if stop:
            kwargs["stop"] = list(stop)
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)
        try:
            for part in self._llama.create_chat_completion(**kwargs):
                delta = (part.get("choices") or [{}])[0].get("delta") or {}
                chunk = delta.get("content")
                if chunk:
                    yield chunk
        except (RuntimeError, ValueError, KeyError) as exc:
            raise BackendUnavailable(
                f"llama.cpp generation failed: {exc}",
                hint="Verify the model and n_ctx; reduce max_tokens if out of memory.",
            ) from exc

    def count_tokens(self, text: str) -> int:
        """Token count via llama.cpp's real tokenizer; estimate on failure."""
        try:
            return len(self._llama.tokenize(text.encode("utf-8")))
        except (RuntimeError, ValueError, AttributeError) as exc:
            _log.debug("llama.cpp tokenize failed, using estimate: %s", exc)
            return estimate(text)


# ---------------------------------------------------------------------------
# HFLLM — guarded import of transformers.
# ---------------------------------------------------------------------------
class HFLLM:
    """Hugging Face transformers adapter. Import-guarded.

    Requires the ``[hf]`` extra (transformers + torch). Uses a ``TextIteratorStreamer`` so
    generation still streams. ``context_window`` comes from the model config; token counts
    use the model's own tokenizer.
    """

    def __init__(
        self,
        model_ref: str,
        *,
        context_window: Optional[int] = None,
        model_options: Optional[dict] = None,
    ) -> None:
        try:
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise BackendUnavailable(
                f"The HF backend needs 'transformers' (and torch): {exc}",
                hint="Install it: pip install \"aether-context[hf]\"",
            ) from exc

        opts = dict(model_options or {})
        opts.setdefault("device_map", "auto")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_ref)
            self._model = AutoModelForCausalLM.from_pretrained(model_ref, **opts)
        except (OSError, ValueError) as exc:
            raise BackendUnavailable(
                f"Could not load HF model '{model_ref}': {exc}",
                hint="Check the org/model id and your network/cache for the download.",
            ) from exc

        self.name: str = model_ref
        cfg = getattr(self._model, "config", None)
        cfg_ctx = getattr(cfg, "max_position_embeddings", None)
        self.context_window: int = (
            context_window
            or (cfg_ctx if isinstance(cfg_ctx, int) and cfg_ctx > 0 else DEFAULT_CONTEXT_WINDOW)
        )

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        stop: Optional[list[str]] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Stream chunks via a background generate + ``TextIteratorStreamer``."""
        import threading

        from transformers import TextIteratorStreamer  # type: ignore[import-not-found]

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            inputs = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self._model.device)
        except (ValueError, AttributeError) as exc:
            raise BackendUnavailable(
                f"HF chat templating failed for '{self.name}': {exc}",
                hint="The model may lack a chat template; try a chat/instruct variant.",
            ) from exc

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs: dict = {
            "input_ids": inputs,
            "streamer": streamer,
            "max_new_tokens": int(max_tokens) if max_tokens is not None else 512,
        }
        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()
        emitted = ""
        for chunk in streamer:
            if not chunk:
                continue
            if stop:
                emitted += chunk
                cut = self._first_stop(emitted, stop)
                if cut is not None:
                    remainder = emitted[:cut][len(emitted) - len(chunk):]
                    if remainder:
                        yield remainder
                    break
            yield chunk
        thread.join()

    @staticmethod
    def _first_stop(text: str, stop: list[str]) -> Optional[int]:
        cut: Optional[int] = None
        for s in stop:
            if s:
                idx = text.find(s)
                if idx != -1:
                    cut = idx if cut is None else min(cut, idx)
        return cut

    def count_tokens(self, text: str) -> int:
        """Token count via the model's tokenizer; estimate on failure."""
        try:
            return len(self._tokenizer.encode(text))
        except (RuntimeError, ValueError, AttributeError) as exc:
            _log.debug("HF tokenizer encode failed, using estimate: %s", exc)
            return estimate(text)


__all__ = [
    "LocalLLM",
    "ModelSpec",
    "parse_spec",
    "load_model",
    "MockLLM",
    "OllamaLLM",
    "LlamaCppLLM",
    "HFLLM",
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_OLLAMA_HOST",
]
