"""
The agent loop — persona + tools + Unlimited Context as working memory + the
autonomy/checkpoint kernel.

Each turn: render the pool-fill status bar, ask the model (with tools), execute
any tool calls, remember the results into the pool, and gate on tests. On green
the kernel checkpoints; on stalled failures it nudges a strategy change (or
escalates if a frontier hook is wired).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

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
}


@dataclass
class AgentResult:
    ok: bool
    steps: int
    summary: str


def _used_tokens(sess: Session) -> int:
    try:
        return int(sess.status_dict().get("slices_used", 0)) * 512
    except Exception:
        return 0


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
    stuck = kernel.StuckDetector()
    esc = escalate or kernel.default_escalate
    emit = on_status or (lambda s: print(s, flush=True))
    schema = tool_schema()

    sess = Session(model=f"ollama/{model}", pool_gb=pool_gb, pull=False, fallback_to_mock=True)
    sess.remember(f"TASK: {task}", tags={"kind": "task"})

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    phase = "reasoning"
    steps = 0
    try:
        for steps in range(1, max_steps + 1):
            emit(statusbar.render(_used_tokens(sess), pool_gb, phase=phase))
            msg = llm.chat(messages, tools=schema)
            calls = msg.get("tool_calls") or []
            if not calls:
                return AgentResult(ok=True, steps=steps, summary=(msg.get("content") or "(done)").strip())

            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for call in calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                phase = _PHASE_BY_TOOL.get(name, "reasoning")
                result = tools.execute(name, args)
                sess.remember(f"{name}({args}) ->\n{result}", tags={"kind": "tool", "tool": name})
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result})

                if name == "run_tests":
                    if kernel.tests_pass(result):
                        kernel.git_checkpoint(cwd, f"aether: step {steps} green")
                    elif stuck.update(kernel.parse_fail_count(result)):
                        hint = esc(f"{task}\n\nStuck. Last test output:\n{result}")
                        nudge = (
                            f"Escalation hint:\n{hint}"
                            if hint
                            else "You are stuck — change strategy; do not repeat the failing approach."
                        )
                        messages.append({"role": "user", "content": nudge})
        return AgentResult(ok=False, steps=steps, summary="max steps reached without a clean finish")
    finally:
        try:
            sess.close()
        except Exception:
            pass
