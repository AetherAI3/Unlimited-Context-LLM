# aether_agent/multi_runner.py
# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Run several agents at once (sub-project D).

A daemon thread per ``(agent, task)`` runs ``agent_runner.run`` (Ollama is HTTP
I/O-bound, so the GIL releases during the request -> real parallelism). Every
event is pushed to a single ``queue.Queue`` that the MAIN thread drains and hands
to ``emit(label, event)`` -- so all terminal writes happen on one thread (no
garbled output). A shared ``write_lock`` (passed into each runner) serializes
destructive tools across agents. Bounded to 8 concurrent.
"""
from __future__ import annotations

import queue
import threading
from typing import Callable

from aether_agent import agent_runner

MAX_CONCURRENT = 8
Emit = Callable[[str, dict], None]
ConfirmFn = Callable[[str, dict], bool]


def _deny(_name: str, _args: dict) -> bool:
    return False


def run_many(jobs, *, emit: Emit, confirm: "ConfirmFn | None" = None, cwd: str = ".") -> list[dict]:
    """Run ``jobs`` (list of ``(Agent, task)``) concurrently; ``emit(label, ev)``
    is called on the main thread per event. Returns a summary dict per agent."""
    jobs = list(jobs)[:MAX_CONCURRENT]
    write_lock = threading.Lock()
    q: "queue.Queue[tuple[str, dict]]" = queue.Queue()
    summaries: dict[str, dict] = {
        a.name: {"name": a.name, "ok": False, "summary": "", "tool_calls": 0} for a, _ in jobs
    }

    def worker(agent, task) -> None:
        try:
            for ev in agent_runner.run(
                agent, task, cwd=cwd, confirm=confirm or _deny, write_lock=write_lock
            ):
                q.put((agent.name, ev))
        except Exception as e:  # noqa: BLE001 — surface as an error event, never crash the thread
            q.put((agent.name, {"type": "error", "msg": str(e)}))
        finally:
            q.put((agent.name, {"type": "_thread_done"}))

    threads = [threading.Thread(target=worker, args=(a, t), daemon=True) for a, t in jobs]
    for th in threads:
        th.start()

    remaining = len(jobs)
    while remaining > 0:
        label, ev = q.get()
        etype = ev.get("type")
        if etype == "_thread_done":
            remaining -= 1
            continue
        if etype == "tool_call":
            summaries[label]["tool_calls"] += 1
        elif etype == "done":
            summaries[label]["ok"] = bool(ev.get("ok", True))
            summaries[label]["summary"] = str(ev.get("text", ""))
        emit(label, ev)

    return [summaries[a.name] for a, _ in jobs]


__all__ = ["run_many", "MAX_CONCURRENT"]
