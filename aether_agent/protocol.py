"""
Bridge protocol — the FROZEN event seam between the headless brain (this Python
package) and the AetherCode TS host. One schema, two transports (local NDJSON
over stdio · cloud SSE). Canonical spec: aether-code/docs/BRIDGE_PROTOCOL.md.

Design (per specs/aethercode_bridge.md):
- The brain DECIDES and emits events; it never renders ANSI and never touches
  the filesystem. The host RENDERS every event and EXECUTES every tool_call,
  returning a tool_result. One tool implementation, one path-guard, identical
  for local and cloud — so local/cloud UX is identical by construction.

Wire format: one JSON object per line (NDJSON). Every message has a "type".
"""

from __future__ import annotations

import json
from typing import Any, Iterable

# Bump on ANY breaking change to the message shapes below. The TS mirror
# (src/core/brain_protocol.ts) MUST carry the same number; the conformance
# fixture (tests/fixtures/bridge_conformance.json) pins both. Canonical:
# aether-code/docs/CONTRACTS.md.
PROTOCOL_VERSION = 2

# --- brain -> host events (the brain emits these) -------------------------
EV_STAGE = "stage"          # {name, face}  staged lifecycle marker
EV_MONOLOGUE = "monologue"  # {text, depth} nested reasoning-tree line
EV_SKILL = "skill"          # {name, reason}  a procedure packet was pinned
EV_TURN = "turn"            # {n, tool_calls, malformed, invented, no_call, fail_count}  per-turn diag
EV_TOOL_CALL = "tool_call"  # {id, name, args}  host must execute + reply
EV_TELEMETRY = "telemetry"  # {tokens, tps, ctx_used, ctx_cap, vram}
EV_STATUS = "status"        # {phase, pool_used, pool_cap}  drives the pool bar
EV_CHECKPOINT = "checkpoint"  # {git_sha}  a verified step was committed
EV_DONE = "done"            # {ok, result, remaining, reason}
EV_ERROR = "error"          # {msg}

BRAIN_EVENTS = frozenset(
    {EV_STAGE, EV_MONOLOGUE, EV_SKILL, EV_TURN, EV_TOOL_CALL, EV_TELEMETRY, EV_STATUS, EV_CHECKPOINT, EV_DONE, EV_ERROR}
)

# --- host -> brain commands (the host sends these) ------------------------
CMD_TASK = "task"            # {text, cwd, pool_gb, effort, model, test_cmd}  starts a run
CMD_TOOL_RESULT = "tool_result"  # {id, output, exit_code}  reply to a tool_call
CMD_CONTROL = "control"     # {action: pause|resume|steer, note}

HOST_COMMANDS = frozenset({CMD_TASK, CMD_TOOL_RESULT, CMD_CONTROL})

# Canonical tool names — the ONE implementation lives in the host.
TOOLS = frozenset({"read_file", "write_file", "run_shell", "run_tests", "repo_search", "git_commit"})


# --- event constructors (brain side) --------------------------------------
def stage(name: str, face: str = "") -> dict[str, Any]:
    return {"type": EV_STAGE, "name": name, "face": face}


def monologue(text: str, depth: int = 0) -> dict[str, Any]:
    return {"type": EV_MONOLOGUE, "text": text, "depth": depth}


def skill(name: str, reason: str = "") -> dict[str, Any]:
    return {"type": EV_SKILL, "name": name, "reason": reason}


def turn(
    n: int, tool_calls: int, malformed: int, invented: int, no_call: bool, fail_count: int | None
) -> dict[str, Any]:
    """Per-assistant-turn diagnostics — the §8 emission curve feed."""
    return {
        "type": EV_TURN,
        "n": n,
        "tool_calls": tool_calls,
        "malformed": malformed,
        "invented": invented,
        "no_call": no_call,
        "fail_count": fail_count,
    }


def tool_call(call_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"type": EV_TOOL_CALL, "id": call_id, "name": name, "args": args}


def telemetry(
    tokens: int = 0, tps: float = 0.0, ctx_used: int = 0, ctx_cap: int = 0, vram: float = 0.0
) -> dict[str, Any]:
    return {
        "type": EV_TELEMETRY,
        "tokens": tokens,
        "tps": tps,
        "ctx_used": ctx_used,
        "ctx_cap": ctx_cap,
        "vram": vram,
    }


def status(phase: str, pool_used: int, pool_cap: int) -> dict[str, Any]:
    return {"type": EV_STATUS, "phase": phase, "pool_used": pool_used, "pool_cap": pool_cap}


def checkpoint(git_sha: str) -> dict[str, Any]:
    return {"type": EV_CHECKPOINT, "git_sha": git_sha}


def done(ok: bool, result: str, remaining: int = 0, reason: str = "") -> dict[str, Any]:
    """Terminal event. `ok` is derived from a real final test run, never from the
    loop-exit reason. `remaining` = failing tests when not ok; `reason` ∈
    {"", "stalled", "no-progress", "max-turns", "unverified"}."""
    return {"type": EV_DONE, "ok": ok, "result": result, "remaining": remaining, "reason": reason}


def error(msg: str) -> dict[str, Any]:
    return {"type": EV_ERROR, "msg": msg}


# --- NDJSON codec ----------------------------------------------------------
def encode(message: dict[str, Any]) -> str:
    """One message -> one NDJSON line (newline included).

    ASCII-safe on the wire (kaomoji faces become \\uXXXX escapes) so it never
    trips a Windows cp1252 console/pipe; the TS host's JSON.parse decodes them
    back to the original characters.
    """
    return json.dumps(message, ensure_ascii=True, separators=(",", ":")) + "\n"


def decode(line: str) -> dict[str, Any] | None:
    """One NDJSON line -> a message dict, or None for blank/malformed lines."""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("type"), str):
        return None
    return obj


def decode_many(lines: Iterable[str]) -> list[dict[str, Any]]:
    out = []
    for ln in lines:
        msg = decode(ln)
        if msg is not None:
            out.append(msg)
    return out
