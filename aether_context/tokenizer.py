# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI - Brandon Barrante
# SPDX-License-Identifier: Apache-2.0
"""Token-count seam.

Budget math across the engine is backend-agnostic, so by default we estimate token
counts with the ``CHARS_PER_TOKEN = 4`` rule (``len(text) // 4``). When a
backend exposes a real tokenizer (llama.cpp, HF), ``from_backend`` prefers it for more
accurate budgeting and falls back — fail-soft — to the estimate if the backend has no
counter or its counter raises.
"""
from __future__ import annotations

from typing import Callable, Optional, Protocol, runtime_checkable

from aether_context._log import get_logger

_log = get_logger(__name__)

#: Average characters per token, used for backend-agnostic budgeting.
CHARS_PER_TOKEN = 4


def estimate(text: str) -> int:
    """Estimate token count as ``len(text) // CHARS_PER_TOKEN``.

    Empty string → 0. Any non-empty string costs at least 1 token so short fragments are
    never undercounted to zero in budget math. Monotonic non-decreasing in length.
    """
    if not isinstance(text, str):
        raise TypeError(f"estimate() expects str, got {type(text).__name__}")
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


@runtime_checkable
class _Counter(Protocol):
    def count_tokens(self, text: str) -> int: ...


def from_backend(model: Optional[object]) -> Callable[[str], int]:
    """Return a ``count(text) -> int`` callable, preferring the backend's tokenizer.

    If ``model`` exposes a working ``count_tokens(text)`` method, that is used. Otherwise
    (no method, ``None`` model, or the backend counter raising) we fall back to
    :func:`estimate` so budget math always works.
    """
    if isinstance(model, _Counter):
        counter = model  # narrowed by the Protocol check

        def _count(text: str) -> int:
            try:
                return int(counter.count_tokens(text))
            except Exception as exc:  # fail-soft: never let budgeting break the run
                _log.debug("backend count_tokens failed, using estimate: %s", exc)
                return estimate(text)

        return _count

    return estimate


__all__ = ["estimate", "from_backend", "CHARS_PER_TOKEN"]
