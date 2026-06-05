"""Tool-call recovery from message content (the #1-risk small-model hardening)."""

from __future__ import annotations

import json

from aether_agent.toolparse import extract_tool_calls


def _names(calls):
    return [c["function"]["name"] for c in calls]


def test_bare_json_object_in_prose():
    # the exact shape qwen2.5-coder:7b emitted via Ollama
    c = 'Let\'s start.\n\n{"name": "repo_search", "arguments": {"query": "FAIL"}}'
    calls = extract_tool_calls(c)
    assert _names(calls) == ["repo_search"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "FAIL"}
    assert calls[0]["id"] == "call-1"


def test_qwen_tool_call_tags():
    c = '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
    assert _names(extract_tool_calls(c)) == ["read_file"]


def test_json_fence():
    c = "I'll write it:\n```json\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"x\", \"content\": \"y\"}}\n```"
    calls = extract_tool_calls(c)
    assert _names(calls) == ["write_file"]
    assert json.loads(calls[0]["function"]["arguments"])["path"] == "x"


def test_function_wrapper_and_string_arguments():
    c = '{"function": {"name": "run_tests", "arguments": "{\\"command\\": \\"pytest\\"}"}}'
    calls = extract_tool_calls(c)
    assert _names(calls) == ["run_tests"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "pytest"}


def test_two_calls_deduped_and_ordered():
    c = ('{"name":"read_file","arguments":{"path":"a"}} then '
         '{"name":"read_file","arguments":{"path":"a"}} and '
         '{"name":"write_file","arguments":{"path":"b","content":"c"}}')
    calls = extract_tool_calls(c)
    assert _names(calls) == ["read_file", "write_file"]  # dup dropped
    assert [c["id"] for c in calls] == ["call-1", "call-2"]


def test_prose_only_returns_empty():
    assert extract_tool_calls("just thinking out loud, no call here") == []
    assert extract_tool_calls("") == []
    assert extract_tool_calls(None) == []
