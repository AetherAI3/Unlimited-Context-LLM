# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Tests for the OpenAI-compatible adapter (mocked HTTP, no network)."""
import io
import json

import pytest

from aether_context.errors import BackendUnavailable
from aether_context.local_llm import OpenAICompatLLM, load_model, parse_spec


def test_spec_parses_openai_backend():
    spec = parse_spec("openai/deepseek/deepseek-chat")
    assert spec.backend == "openai"
    assert spec.ref == "deepseek/deepseek-chat"


def test_openrouter_default_base_url(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    llm = OpenAICompatLLM("deepseek/deepseek-chat")
    assert llm.base_url.startswith("https://openrouter.ai/api/v1")
    assert llm.api_key == "sk-test"


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(BackendUnavailable):
        OpenAICompatLLM("m", base_url="https://x/v1")


def _sse(*objs):
    body = "".join(f"data: {json.dumps(o)}\n\n" for o in objs) + "data: [DONE]\n\n"
    return io.BytesIO(body.encode("utf-8"))


def test_generate_streams_content(monkeypatch):
    llm = OpenAICompatLLM("m", base_url="https://x/v1", api_key="k")
    chunks = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"reasoning": "(thinking)"}}]},  # ignored
        {"choices": [{"delta": {"content": "lo"}}]},
    ]
    monkeypatch.setattr(llm, "_open", lambda req: _sse(*chunks))
    assert "".join(llm.generate("hi")) == "Hello"


def test_chat_returns_tool_calls_and_usage(monkeypatch):
    llm = OpenAICompatLLM("m", base_url="https://x/v1", api_key="k")
    resp = {
        "choices": [{"message": {
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "lookup_issue", "arguments": "{\"number\": 7}"}}],
        }}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
    }
    monkeypatch.setattr(llm, "_open", lambda req: io.BytesIO(json.dumps(resp).encode()))
    out = llm.chat([{"role": "user", "content": "go"}], tools=[{"type": "function"}])
    assert out["tool_calls"][0]["function"]["name"] == "lookup_issue"
    assert out["usage"]["prompt_tokens"] == 11


def test_count_tokens_estimate():
    llm = OpenAICompatLLM("m", base_url="https://x/v1", api_key="k")
    assert llm.count_tokens("abcd" * 4) > 0


def test_load_model_dispatches_openai(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    llm = load_model("openai/deepseek/deepseek-chat")
    assert isinstance(llm, OpenAICompatLLM)
