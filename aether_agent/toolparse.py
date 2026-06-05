"""
Tool-call recovery — the #1-risk hardening from the bridge spec.

Many local models (qwen2.5-coder, gemma, ...) emit a tool call as JSON TEXT in
the message `content` instead of in the structured OpenAI `tool_calls` field —
Ollama's chat template doesn't always parse it out. The brain only acts on
structured `tool_calls`, so without this it sees "no tool call" and stalls.

extract_tool_calls() recovers them from content and returns OpenAI-shaped
tool_calls ({id, type, function:{name, arguments}}), so the rest of the loop is
unchanged. Handles the common shapes:
  - <tool_call>{...}</tool_call>            (Qwen native)
  - ```json\n{...}\n```  /  ``` {...} ```   (fenced)
  - bare  {"name": "...", "arguments": {...}}  objects in prose
  - {"tool_call": {...}} / {"function": {...}} wrappers
"""

from __future__ import annotations

import json
import re
from typing import Any


def _balanced_json_objects(text: str) -> list[str]:
    """Return every top-level balanced {...} substring (string-aware)."""
    out: list[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start : i + 1])
                    start = -1
    return out


def _as_call(obj: Any) -> tuple[str, dict] | None:
    """Normalize a parsed object to (name, args) if it looks like a tool call."""
    if not isinstance(obj, dict):
        return None
    # unwrap common wrappers
    if "tool_call" in obj and isinstance(obj["tool_call"], dict):
        obj = obj["tool_call"]
    if "function" in obj and isinstance(obj["function"], dict):
        obj = obj["function"]
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None
    if "arguments" not in obj and "parameters" not in obj:
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            args = {}
    if not isinstance(args, dict):
        args = {}
    return name, args


def _candidates(content: str) -> list[str]:
    """Pull the most-likely JSON-bearing fragments, best signal first."""
    frags: list[str] = []
    frags += re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.S)
    frags += re.findall(r"```(?:json|tool_code)?\s*(\{.*?\})\s*```", content, re.S)
    # bare balanced objects (catches the no-fence case)
    frags += _balanced_json_objects(content)
    return frags


def extract_tool_calls(content: str | None) -> list[dict]:
    """Recover OpenAI-shaped tool_calls from message content. Empty if none.

    Deduplicates identical (name,args) and assigns stable ids call-1, call-2, ...
    `arguments` is a JSON STRING (matches the OpenAI tool_call contract the brain
    decodes with json.loads)."""
    if not content:
        return []
    seen: set[str] = set()
    calls: list[dict] = []
    for frag in _candidates(content):
        try:
            obj = json.loads(frag)
        except (json.JSONDecodeError, TypeError):
            continue
        nc = _as_call(obj)
        if nc is None:
            continue
        name, args = nc
        key = name + " " + json.dumps(args, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        calls.append(
            {
                "id": f"call-{len(calls) + 1}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        )
    return calls
