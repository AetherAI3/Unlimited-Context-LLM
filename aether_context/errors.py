"""Typed errors for aether-context.

Design intent: every failure in the library re-wraps the underlying cause into one of
these typed errors. No module in the package may use a bare ``except:`` — catch the
specific exception and re-raise as one of these, so callers always get an actionable
``.hint`` with the exact fix command.

Fail-soft discipline lives in the call sites (the pager/retrieval degrades and logs
rather than raising into a long run); these types exist so that when something *is*
worth surfacing, it surfaces with a fix.
"""
from __future__ import annotations


class AetherContextError(Exception):
    """Base class for all aether-context errors.

    Carries a human-actionable ``.hint`` describing how to fix the condition. Subclasses
    provide a sensible default hint; callers may override it per-instance.
    """

    #: Default fix hint; subclasses override. Always a non-empty string.
    default_hint: str = "See the docs at https://github.com/aethersystems/unlimited-context"

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message: str = message
        self.hint: str = hint if hint is not None else self.default_hint

    def __str__(self) -> str:
        return f"{self.message}\nhint: {self.hint}"


class PoolBudgetError(AetherContextError):
    """The context pool exceeded (or cannot satisfy) its GB budget ceiling."""

    default_hint = (
        "Raise the pool size (e.g. `aether-context --pool 10`) or let the witness evict "
        "stale slices; the pool budget floor is 5 GB."
    )


class OllamaNotRunning(AetherContextError):
    """The Ollama daemon is not reachable on localhost:11434."""

    default_hint = "Start the Ollama daemon: `ollama serve`"


class ModelNotPulled(AetherContextError):
    """The requested Ollama model has not been downloaded yet."""

    default_hint = "Pull the model first: `ollama pull <model>` (or pass `pull=True`)"


class BackendUnavailable(AetherContextError):
    """A backend's optional dependency is missing or the backend cannot be loaded."""

    default_hint = (
        "Install the backend extra, e.g. `pip install \"aether-context[llamacpp]\"` or "
        "`pip install \"aether-context[hf]\"`; or use `model=\"mock\"` to run offline."
    )


class EncoderError(AetherContextError):
    """The static encoder failed to produce an embedding."""

    default_hint = (
        "Check the input text is a non-empty string; if a custom encoder table was "
        "provided, verify its shape is (vocab, dim)."
    )


class PoolCorrupt(AetherContextError):
    """The on-disk context pool (mmap index or payloads) is malformed."""

    default_hint = (
        "Re-initialize the pool with `aether-context init`, or delete the pool dir "
        "under ~/.aether-context to rebuild it (non-destructive to your models)."
    )


__all__ = [
    "AetherContextError",
    "PoolBudgetError",
    "OllamaNotRunning",
    "ModelNotPulled",
    "BackendUnavailable",
    "EncoderError",
    "PoolCorrupt",
]
