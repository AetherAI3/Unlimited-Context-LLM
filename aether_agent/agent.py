"""
The agent loop — persona + tools + Unlimited Context as working memory + the
autonomy/checkpoint kernel.

Each turn: render the pool-fill status bar, ask the model (with tools), execute
any tool calls, remember the results into the pool, and gate on tests. On green
the kernel checkpoints; on stalled failures it nudges a strategy change (or
escalates if a frontier hook is wired).

The chat+tool loop is factored into `run_agent_events` — a generator that yields
the lightweight event dicts the REPL / bridge render (monologue/tool_call/
tool_result/done). `run_agent` consumes it to build an `AgentResult`, and
`brains.LocalBrain` re-yields the same events (one loop, two surfaces). The
kernel test-gate + git-checkpoint semantics stay opt-in: `run_agent` enables
them; a plain chat-loop brain need not git-checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

from aether_agent.adapter import OllamaChat, DEFAULT_MODEL
from aether_agent.persona import SYSTEM_PROMPT
from aether_agent.tools import Tools, tool_schema
from aether_agent import kernel, statusbar
from aether_context import Session

_PHASE_BY_TOOL = {
    "repo_search": "scanning",
    "read_file": "scanning",
    "write_file": "anchoring",
    "run_tests": "grounding",
    "run_shell": "grounding",
    "git_commit": "paging",
    "web_search": "scanning",
    "web_fetch": "scanning",
}


@dataclass
class AgentResult:
    ok: bool
    steps: int
    summary: str


def _used_tokens(sess: Optional[Session]) -> int:
    if sess is None:
        return 0
    try:
        return int(sess.status_dict().get("slices_used", 0)) * 512
    except Exception:
        return 0


def _parse_args(call: dict) -> dict:
    try:
        return json.loads(call.get("function", {}).get("arguments") or "{}")
    except Exception:
        return {}


def run_agent_events(
    task: str,
    *,
    llm: Any,
    tools: Tools,
    cwd: str = ".",
    pool_gb: int = 5,
    max_steps: int = 40,
    sess: Optional[Session] = None,
    stuck: Optional[kernel.StuckDetector] = None,
    escalate: Optional[kernel.EscalateFn] = None,
    schema: Optional[list[dict]] = None,
    git_checkpoint: bool = True,
    verify_finish: bool = True,
    on_status: Optional[Callable[[str], None]] = None,
) -> Iterator[dict[str, Any]]:
    """Drive the chat+tool loop, yielding render-ready event dicts.

    Events: {'type':'monologue','text'} · {'type':'tool_call','name','args'} ·
    {'type':'tool_result','name','output'} · {'type':'done','text'[,'ok']}.

    Shared by `run_agent` (full kernel: Session memory + git-checkpoints +
    escalation + a final verifying test run) and `brains.LocalBrain` (chat loop;
    no git-checkpoint). The injected `llm` exposes `.chat(messages, tools=...)`
    returning the assistant message dict; `tools` is a real `Tools` executor.

    `verify_finish` (parity with headless.py's stronger finish): on a no-tool-call
    turn, if the model never ran the gate since its last edit, run it ONCE and
    only finish clean on a green exit — otherwise nudge with the failure and keep
    going. Guarded: a no-op when `tools` carries no test command, so a pure
    read/answer task and the existing unit tests are unaffected.
    """
    stuck = stuck or kernel.StuckDetector()
    esc = escalate or kernel.default_escalate
    schema = schema if schema is not None else tool_schema()
    emit = on_status or (lambda s: None)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    phase = "reasoning"
    tested_since_edit = False  # did the model run the gate since its last edit?

    for steps in range(1, max_steps + 1):
        emit(statusbar.render(_used_tokens(sess), pool_gb, phase=phase))
        msg = llm.chat(messages, tools=schema)
        calls = msg.get("tool_calls") or []
        content = (msg.get("content") or "").strip()

        if not calls:
            final = content or "(done)"
            verify = _maybe_verify(
                verify_finish, tools, tested_since_edit, steps, messages, git_checkpoint, cwd
            )
            if verify is not None:
                for ev in verify["events"]:
                    yield ev
                if not verify["green"]:
                    tested_since_edit = True
                    continue  # nudge already appended — keep working toward green
            if content:
                yield {"type": "monologue", "text": content}
            yield {"type": "done", "text": final}
            return

        if content:
            yield {"type": "monologue", "text": content}
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})

        for call in calls:
            name = call.get("function", {}).get("name", "")
            args = _parse_args(call)
            phase = _PHASE_BY_TOOL.get(name, "reasoning")
            yield {"type": "tool_call", "name": name, "args": args}

            result = tools.execute(name, args)
            if sess is not None:
                try:
                    sess.remember(f"{name}({args}) ->\n{result}", tags={"kind": "tool", "tool": name})
                except Exception:
                    pass
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result})
            yield {"type": "tool_result", "name": name, "output": result}

            if name == "write_file":
                tested_since_edit = False
            if name == "run_tests":
                tested_since_edit = True
                if kernel.tests_pass(result):
                    if git_checkpoint:
                        kernel.git_checkpoint(cwd, f"aether: step {steps} green")
                elif stuck.update(kernel.parse_fail_count(result)):
                    hint = esc(f"{task}\n\nStuck. Last test output:\n{result}")
                    nudge = (
                        f"Escalation hint:\n{hint}"
                        if hint
                        else "You are stuck — change strategy; do not repeat the failing approach."
                    )
                    messages.append({"role": "user", "content": nudge})

    yield {"type": "done", "text": "max steps reached without a clean finish", "ok": False}


def _maybe_verify(
    enabled: bool,
    tools: Tools,
    tested_since_edit: bool,
    steps: int,
    messages: list[dict],
    git_checkpoint: bool,
    cwd: str,
) -> Optional[dict[str, Any]]:
    """A final verifying gate on a no-tool-call turn (parity with headless).

    Returns None to skip (verification disabled, no gate configured, or the model
    already finished on a verified run since its last edit). Otherwise runs the
    gate ONCE and returns {'green': bool, 'events': [...]}; on red, appends a
    reality nudge to `messages` so the caller can keep working.
    """
    if not enabled:
        return None
    test_cmd = (getattr(tools, "test_cmd", "") or "").strip()
    if not test_cmd:
        return None
    if tested_since_edit:  # model already verified since its last edit -> trust it
        return None
    events: list[dict[str, Any]] = [{"type": "tool_call", "name": "run_tests", "args": {}}]
    output = tools.run_tests()
    events.append({"type": "tool_result", "name": "run_tests", "output": output})
    green = kernel.tests_pass(output)
    if green:
        if git_checkpoint:
            kernel.git_checkpoint(cwd, f"aether: step {steps} verified green")
    else:
        fails = kernel.parse_fail_count(output)
        messages.append({
            "role": "user",
            "content": (
                f"{fails if fails is not None else 'Some'} tests still failing. Read the SOURCE "
                "module the failing test imports (not the test file), fix the bug there, then "
                "run_tests once. Do not declare success until the gate exits 0."
            ),
        })
    return {"green": green, "events": events}


def run_agent(
    task: str,
    *,
    model: str = DEFAULT_MODEL,
    pool_gb: int = 5,
    cwd: str = ".",
    max_steps: int = 40,
    test_cmd: str = "pytest -q",
    escalate: Optional[kernel.EscalateFn] = None,
    on_status: Optional[Callable[[str], None]] = None,
) -> AgentResult:
    llm = OllamaChat(model=model)
    tools = Tools(cwd, test_cmd=test_cmd)
    emit = on_status or (lambda s: print(s, flush=True))

    sess = Session(model=f"ollama/{model}", pool_gb=pool_gb, pull=False, fallback_to_mock=True)
    sess.remember(f"TASK: {task}", tags={"kind": "task"})

    tool_calls = 0
    ok = False
    summary = "max steps reached without a clean finish"
    try:
        for ev in run_agent_events(
            task,
            llm=llm,
            tools=tools,
            cwd=cwd,
            pool_gb=pool_gb,
            max_steps=max_steps,
            sess=sess,
            escalate=escalate,
            git_checkpoint=True,
            verify_finish=True,
            on_status=emit,
        ):
            if ev["type"] == "tool_call":
                tool_calls += 1
            elif ev["type"] == "done":
                summary = ev.get("text", "(done)")
                ok = ev.get("ok", True)
        return AgentResult(ok=ok, steps=max(tool_calls, 1), summary=summary)
    finally:
        try:
            sess.close()
        except Exception:
            pass
