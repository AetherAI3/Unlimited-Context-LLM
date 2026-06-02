# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI - Brandon Barrante
# SPDX-License-Identifier: Apache-2.0
"""Tests for THE WRAPPER — local_llm.py (network-free).

Covers:
  * spec-string parsing into ModelSpec (ollama/, bare name, llamacpp:, hf/, mock)
  * MockLLM determinism, streaming (>1 chunk), count_tokens, configurable window
  * load_model() dispatch (spec string + passthrough of a LocalLLM object)
  * OllamaLLM raising the typed error WITH a .hint when the daemon is unreachable
    (urllib monkeypatched to raise — NO real network)
  * guarded backends raise BackendUnavailable (not ImportError) when deps missing

These tests import the submodule directly (the package __init__ surface is finalized
in a later stage) and never hit the network.
"""
from __future__ import annotations

import io
import urllib.error

import pytest

from aether_context.errors import (
    BackendUnavailable,
    ModelNotPulled,
    OllamaNotRunning,
)
from aether_context.local_llm import (
    DEFAULT_CONTEXT_WINDOW,
    HFLLM,
    LlamaCppLLM,
    LocalLLM,
    MockLLM,
    ModelSpec,
    OllamaLLM,
    load_model,
    parse_spec,
)


# ---- spec-string parsing ----------------------------------------------------
def test_parse_ollama_prefixed():
    spec = parse_spec("ollama/qwen2.5")
    assert spec.backend == "ollama"
    assert spec.ref == "qwen2.5"
    assert spec.options == {}


def test_parse_ollama_tag_preserved():
    # a tag uses ':' but the leading 'ollama/' fixes the backend, so the colon stays in ref
    spec = parse_spec("ollama/llama3.1:8b")
    assert spec.backend == "ollama"
    assert spec.ref == "llama3.1:8b"


def test_parse_bare_name_defaults_to_ollama():
    spec = parse_spec("qwen2.5")
    assert spec.backend == "ollama"
    assert spec.ref == "qwen2.5"


def test_parse_llamacpp_path():
    spec = parse_spec("llamacpp:/models/qwen2.5-7b.gguf")
    assert spec.backend == "llamacpp"
    assert spec.ref == "/models/qwen2.5-7b.gguf"


def test_parse_llamacpp_windows_path_keeps_drive_colon():
    # a Windows path has its own colon; only the FIRST colon splits backend from ref
    spec = parse_spec("llamacpp:C:/models/qwen2.5.gguf")
    assert spec.backend == "llamacpp"
    assert spec.ref == "C:/models/qwen2.5.gguf"


def test_parse_hf_org_model():
    spec = parse_spec("hf/Qwen/Qwen2.5-7B-Instruct")
    assert spec.backend == "hf"
    # the org/model path after 'hf/' is preserved verbatim
    assert spec.ref == "Qwen/Qwen2.5-7B-Instruct"


def test_parse_mock():
    spec = parse_spec("mock")
    assert spec.backend == "mock"
    assert spec.ref == "mock"


def test_parse_empty_spec_raises_with_actionable_message():
    with pytest.raises(BackendUnavailable) as ei:
        parse_spec("   ")
    # the message must teach the one obvious format
    assert "backend/ref" in str(ei.value)
    assert ei.value.hint.strip() != ""


def test_parse_unknown_backend_raises():
    with pytest.raises(BackendUnavailable) as ei:
        parse_spec("totally-not-a-backend/foo")
    assert "totally-not-a-backend" in str(ei.value)


def test_parse_non_string_raises():
    with pytest.raises(BackendUnavailable):
        parse_spec(1234)  # type: ignore[arg-type]


def test_modelspec_is_frozen():
    spec = ModelSpec(backend="mock", ref="mock", options={})
    with pytest.raises(Exception):
        spec.backend = "ollama"  # type: ignore[misc]


# ---- MockLLM: deterministic, streaming, counts ------------------------------
def test_mock_satisfies_protocol():
    m = MockLLM()
    assert isinstance(m, LocalLLM)
    assert isinstance(m.name, str)
    assert isinstance(m.context_window, int)


def test_mock_default_context_window():
    assert MockLLM().context_window == DEFAULT_CONTEXT_WINDOW


def test_mock_configurable_context_window():
    assert MockLLM(context_window=2048).context_window == 2048


def test_mock_generate_is_deterministic_for_same_prompt():
    a = "".join(MockLLM().generate("hello world"))
    b = "".join(MockLLM().generate("hello world"))
    assert a == b
    assert a != ""


def test_mock_generate_differs_for_different_prompts():
    a = "".join(MockLLM().generate("alpha"))
    b = "".join(MockLLM().generate("beta"))
    assert a != b


def test_mock_system_changes_output_deterministically():
    base = "".join(MockLLM().generate("p"))
    withsys1 = "".join(MockLLM().generate("p", system="be terse"))
    withsys2 = "".join(MockLLM().generate("p", system="be terse"))
    assert withsys1 == withsys2
    assert withsys1 != base


def test_mock_streams_more_than_one_chunk():
    m = MockLLM(output_tokens=400)
    chunks = list(m.generate("a long prompt that forces several chunks"))
    assert len(chunks) > 1  # streaming: pager can overlap prefetch with generation
    assert all(isinstance(c, str) for c in chunks)


def test_mock_is_streaming_property_true():
    assert MockLLM(output_tokens=400).is_streaming is True


def test_mock_output_tokens_independent_of_context_window():
    # output length is controlled by output_tokens, NOT context_window — lets the bench
    # force overflow with a tiny window but a long generation
    small = MockLLM(context_window=128, output_tokens=400)
    text = "".join(small.generate("force overflow"))
    assert small.count_tokens(text) > small.context_window


def test_mock_respects_max_tokens():
    m = MockLLM(output_tokens=400)
    full = list(m.generate("prompt"))
    capped = list(m.generate("prompt", max_tokens=5))
    assert m.count_tokens("".join(capped)) <= m.count_tokens("".join(full))
    assert m.count_tokens("".join(capped)) <= 5 + 8  # small slack for chunk granularity


def test_mock_count_tokens_matches_estimate():
    from aether_context.tokenizer import estimate

    m = MockLLM()
    assert m.count_tokens("a" * 40) == estimate("a" * 40)
    assert m.count_tokens("") == 0


def test_mock_stop_truncates_output():
    m = MockLLM(output_tokens=200)
    full = "".join(m.generate("prompt"))
    # pick a substring that actually appears, then assert stop cuts at/before it
    token = full[10:14]
    stopped = "".join(m.generate("prompt", stop=[token]))
    assert len(stopped) < len(full)            # the stop genuinely truncated
    assert full.startswith(stopped)            # output is a clean prefix of the full run
    assert token not in stopped                # nothing past the stop point survives


# ---- load_model dispatch ----------------------------------------------------
def test_load_model_passthrough_localllm_object():
    m = MockLLM()
    assert load_model(m) is m


def test_load_model_from_mock_spec():
    m = load_model("mock")
    assert isinstance(m, MockLLM)


def test_load_model_mock_forwards_kwargs():
    m = load_model("mock", context_window=4096)
    assert m.context_window == 4096


def test_load_model_ollama_spec_builds_ollama_without_network():
    # constructing the adapter must NOT touch the network (lazy connection)
    m = load_model("ollama/qwen2.5")
    assert isinstance(m, OllamaLLM)
    assert m.name == "qwen2.5"


def test_load_model_unknown_backend_raises_backend_unavailable():
    with pytest.raises(BackendUnavailable):
        load_model("nope/model")


# ---- OllamaLLM typed errors when the daemon is unreachable (NO network) ------
def _patch_urlopen_raise(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    def _boom(*args: object, **kwargs: object) -> object:
        raise exc

    monkeypatch.setattr("aether_context.local_llm.urllib.request.urlopen", _boom)


def test_ollama_generate_daemon_down_raises_typed_with_hint(monkeypatch):
    # simulate "connection refused" — daemon not running
    _patch_urlopen_raise(monkeypatch, urllib.error.URLError("Connection refused"))
    m = OllamaLLM("qwen2.5")
    with pytest.raises(OllamaNotRunning) as ei:
        list(m.generate("hi"))
    assert "ollama serve" in ei.value.hint.lower()


def test_ollama_context_window_falls_back_when_daemon_down(monkeypatch):
    # /api/show is best-effort: if it fails, context_window degrades to the fallback,
    # it does NOT raise (fail-soft for an optional metadata probe)
    _patch_urlopen_raise(monkeypatch, urllib.error.URLError("Connection refused"))
    m = OllamaLLM("qwen2.5")
    assert m.context_window == DEFAULT_CONTEXT_WINDOW


def test_ollama_count_tokens_uses_estimate_offline():
    from aether_context.tokenizer import estimate

    m = OllamaLLM("qwen2.5")
    assert m.count_tokens("a" * 40) == estimate("a" * 40)


def test_ollama_model_not_pulled_raises_typed_with_hint(monkeypatch):
    # Ollama returns HTTP 404 with a "model not found" body when the model isn't pulled
    fp = io.BytesIO(b'{"error":"model \'ghost\' not found"}')
    err = urllib.error.HTTPError(
        url="http://localhost:11434/api/chat",
        code=404,
        msg="Not Found",
        hdrs=None,  # type: ignore[arg-type]
        fp=fp,
    )
    _patch_urlopen_raise(monkeypatch, err)
    m = OllamaLLM("ghost")
    with pytest.raises(ModelNotPulled) as ei:
        list(m.generate("hi"))
    assert "ollama pull" in ei.value.hint.lower()


def test_ollama_name_and_host_defaults():
    m = OllamaLLM("qwen2.5")
    assert m.name == "qwen2.5"
    assert "11434" in m.host


def test_ollama_custom_host():
    m = OllamaLLM("qwen2.5", host="http://192.168.1.2:11434")
    assert m.host == "http://192.168.1.2:11434"


# ---- guarded backends: friendly error when the optional dep is missing ------
def test_llamacpp_missing_dep_raises_backend_unavailable(monkeypatch):
    # force the import to fail so we exercise the guard, regardless of what's installed
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if name == "llama_cpp" or name.startswith("llama_cpp."):
            raise ImportError("no llama_cpp")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(BackendUnavailable) as ei:
        LlamaCppLLM("/models/x.gguf")
    assert "llamacpp" in ei.value.hint.lower() or "llama" in ei.value.hint.lower()


def test_hf_missing_dep_raises_backend_unavailable(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("no transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(BackendUnavailable) as ei:
        HFLLM("Qwen/Qwen2.5-7B-Instruct")
    assert "hf" in ei.value.hint.lower() or "transformers" in ei.value.hint.lower()
