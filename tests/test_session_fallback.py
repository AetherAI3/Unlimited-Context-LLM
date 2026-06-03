# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Session fallback-to-mock behavior (hermetic — load_model is monkeypatched, no network)."""

from __future__ import annotations

import pytest

import aether_context.session as session_mod
from aether_context.errors import OllamaNotRunning
from aether_context.local_llm import MockLLM
from aether_context.session import Session


def _patch_load(monkeypatch, *, fail: bool) -> None:
    real = session_mod.load_model

    def fake_load(spec, **kw):
        if spec == "mock":
            return real("mock", **kw)
        if fail:
            raise OllamaNotRunning("daemon not reachable at localhost:11434")
        return real(spec, **kw)

    monkeypatch.setattr(session_mod, "load_model", fake_load)


def test_fallback_to_mock_warns_and_uses_mock(tmp_pool_dir, monkeypatch):
    _patch_load(monkeypatch, fail=True)
    with pytest.warns(RuntimeWarning):
        s = Session(
            model="ollama/qwen2.5",
            pool_gb=5,
            pool_dir=str(tmp_pool_dir),
            fallback_to_mock=True,
        )
    assert isinstance(s.local_llm, MockLLM)
    s.close()


def test_no_fallback_raises_the_real_error(tmp_pool_dir, monkeypatch):
    _patch_load(monkeypatch, fail=True)
    with pytest.raises(OllamaNotRunning):
        Session(
            model="ollama/qwen2.5",
            pool_gb=5,
            pool_dir=str(tmp_pool_dir),
            fallback_to_mock=False,
        )


def test_explicit_mock_never_triggers_fallback_warning(tmp_pool_dir, monkeypatch, recwarn):
    # model="mock" must load directly, no RuntimeWarning about falling back.
    _patch_load(monkeypatch, fail=False)
    s = Session(model="mock", pool_gb=5, pool_dir=str(tmp_pool_dir))
    assert isinstance(s.local_llm, MockLLM)
    assert not any(issubclass(w.category, RuntimeWarning) for w in recwarn.list)
    s.close()
