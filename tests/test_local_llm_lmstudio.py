# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Tests for the LM Studio adapter (mocked HTTP, no network)."""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from aether_context.errors import BackendUnavailable
from aether_context.local_llm import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_LMSTUDIO_BASE,
    LMStudioLLM,
    LocalLLM,
    OpenAICompatLLM,
    load_model,
    parse_spec,
)
from aether_context.tokenizer import estimate


def _sse(*objs: object) -> io.BytesIO:
    body = "".join(f"data: {json.dumps(o)}\n\n" for o in objs) + "data: [DONE]\n\n"
    return io.BytesIO(body.encode("utf-8"))


def _capture_open(llm: OpenAICompatLLM, monkeypatch: pytest.MonkeyPatch, response: object) -> dict:
    captured: dict = {}

    def _open(req: urllib.request.Request) -> object:
        captured["req"] = req
        captured["body"] = json.loads(req.data.decode("utf-8") if req.data else b"{}")
        return response

    monkeypatch.setattr(llm, "_open", _open)
    return captured


def test_parse_lmstudio_spec():
    spec = parse_spec("lmstudio/qwen2.5-7b-instruct")
    assert spec.backend == "lmstudio"
    assert spec.ref == "qwen2.5-7b-instruct"


def test_parse_lmstudio_model_id_with_slash():
    spec = parse_spec("lmstudio/lmstudio-community/Qwen2.5-7B-Instruct")
    assert spec.backend == "lmstudio"
    assert spec.ref == "lmstudio-community/Qwen2.5-7B-Instruct"


def test_bare_lmstudio_stays_ollama():
    spec = parse_spec("lmstudio")
    assert spec.backend == "ollama"
    assert spec.ref == "lmstudio"


def test_load_model_dispatches_lmstudio():
    llm = load_model("lmstudio/qwen2.5-7b-instruct")
    assert isinstance(llm, LMStudioLLM)
    assert isinstance(llm, LocalLLM)
    assert llm.name == "qwen2.5-7b-instruct"


def test_default_base_url_and_no_key():
    llm = LMStudioLLM("qwen2.5-7b-instruct")
    assert llm.base_url == DEFAULT_LMSTUDIO_BASE
    assert llm.api_key == ""


def test_does_not_inherit_openai_or_openrouter_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-remote")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    llm = LMStudioLLM("qwen")
    assert llm.base_url == DEFAULT_LMSTUDIO_BASE
    assert llm.api_key == ""


def test_default_request_url_and_no_authorization(monkeypatch):
    llm = LMStudioLLM("qwen2.5-7b-instruct")
    captured = _capture_open(llm, monkeypatch, _sse({"choices": [{"delta": {"content": "ok"}}]}))
    assert "".join(llm.generate("hi")) == "ok"
    assert captured["req"].full_url == "http://localhost:1234/v1/chat/completions"
    assert captured["req"].get_header("Authorization") is None


def test_explicit_api_key_sends_bearer(monkeypatch):
    llm = LMStudioLLM("qwen", api_key="tok")
    captured = _capture_open(llm, monkeypatch, _sse({"choices": [{"delta": {"content": "ok"}}]}))
    list(llm.generate("hi"))
    assert captured["req"].get_header("Authorization") == "Bearer tok"


def test_generate_is_inherited_sse_parser():
    assert LMStudioLLM.generate is OpenAICompatLLM.generate


def test_generate_streams_content(monkeypatch):
    llm = LMStudioLLM("qwen")
    chunks = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"reasoning": "(thinking)"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
    ]
    monkeypatch.setattr(llm, "_open", lambda req: _sse(*chunks))
    assert "".join(llm.generate("hi")) == "Hello"


def test_stop_and_max_tokens_in_payload(monkeypatch):
    llm = LMStudioLLM("qwen")
    captured = _capture_open(llm, monkeypatch, _sse({"choices": [{"delta": {"content": "x"}}]}))
    list(llm.generate("hi", stop=["END"], max_tokens=16))
    assert captured["body"]["stop"] == ["END"]
    assert captured["body"]["max_tokens"] == 16
    assert captured["body"]["model"] == "qwen"


def test_count_tokens_uses_estimate():
    llm = LMStudioLLM("qwen")
    assert llm.count_tokens("a" * 40) == estimate("a" * 40)


def test_default_context_window_fallback():
    assert LMStudioLLM("qwen").context_window == DEFAULT_CONTEXT_WINDOW


def test_explicit_context_window():
    assert LMStudioLLM("qwen", context_window=2048).context_window == 2048


def test_connection_refused_is_actionable(monkeypatch):
    llm = LMStudioLLM("qwen")
    monkeypatch.setattr(
        llm, "_open", lambda req: (_ for _ in ()).throw(urllib.error.URLError("Connection refused"))
    )
    with pytest.raises(BackendUnavailable) as ei:
        list(llm.generate("hi"))
    hint = ei.value.hint.lower()
    assert "lm studio" in hint
    assert "local server" in hint
    assert "1234" in hint


def test_model_http_error_is_actionable(monkeypatch):
    fp = io.BytesIO(b'{"error":{"message":"Model not found"}}')
    err = urllib.error.HTTPError(
        url="http://localhost:1234/v1/chat/completions",
        code=404,
        msg="Not Found",
        hdrs=None,  # type: ignore[arg-type]
        fp=fp,
    )
    llm = LMStudioLLM("ghost")
    monkeypatch.setattr(llm, "_open", lambda req: (_ for _ in ()).throw(err))
    with pytest.raises(BackendUnavailable) as ei:
        list(llm.generate("hi"))
    text = str(ei.value).lower()
    assert "ghost" in text
    assert "load" in ei.value.hint.lower()
    assert "lm studio" in ei.value.hint.lower()


def test_openai_still_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(BackendUnavailable) as ei:
        OpenAICompatLLM("m", base_url="https://x/v1")
    assert "api key" in str(ei.value).lower()
