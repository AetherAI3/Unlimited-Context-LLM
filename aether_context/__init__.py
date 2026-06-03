# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""aether-context — virtual memory for an LLM's attention (local-first, numpy-only core).

"Unlimited" means *reach, not attention*: the model keeps its native window; the engine
externalizes overflow to a local pool and pages the right slices back in via retrieval.

The whole product is three lines:

    from aether_context import Session

    s = Session(model="ollama/qwen2.5", pool_gb=5)
    s.run("Build me a full-stack weightlifting tracker app.")

No Ollama? ``Session(model="mock", pool_gb=5)`` runs the engine end-to-end with a built-in
deterministic model (offline, zero deps) — great for trying the API, tests, and CI.

Public surface (intentionally tiny):
  * :class:`Session` — the lifecycle controller you drive.
  * :func:`load_model` — resolve a model spec / bring-your-own backend to a ``LocalLLM``.
  * :data:`__version__` — the package version.
"""
from aether_context.local_llm import load_model
from aether_context.session import Session

__version__ = "0.1.0"

__all__ = ["Session", "load_model", "__version__"]
