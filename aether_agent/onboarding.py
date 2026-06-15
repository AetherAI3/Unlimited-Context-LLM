# aether_agent/onboarding.py
# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""First-run / preflight state machine for the Ollama-managed terminal.

``preflight(ctl, ...)`` runs detect -> (install?) -> serve -> pick -> pull and
returns a typed :class:`Preflight`. Silent and instant when everything is already
healthy. ``prompt``/``resources``/``emit`` are injected so the flow is testable
with no TTY, no network, no real Ollama. ``emit`` defaults to a no-op (quiet
tests); ``repl.py`` passes a real writer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from aether_agent import hardware, progress

Prompt = Callable[[str], str]
Emit = Callable[[str], None]
ResourcePick = Callable[[], "tuple[str, str]"]


@dataclass
class Preflight:
    ok: bool
    chosen_model: str | None = None
    message: str = ""
    steps: list[str] = field(default_factory=list)


def _default_pick() -> "tuple[str, str]":
    return hardware.best_fit(hardware.detect_resources())


def preflight(
    ctl: Any,
    *,
    resources: ResourcePick | None = None,
    prompt: Prompt = lambda q: "",
    emit: Emit = lambda s: None,
) -> Preflight:
    pick = resources or _default_pick
    steps: list[str] = []

    if not ctl.detect().get("installed"):
        ans = prompt("Ollama not found. Install it now? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            return Preflight(False, None, "Install Ollama manually: https://ollama.com/download", steps)
        ok, detail = ctl.install(consent=True)
        steps.append(progress.step("install ollama", ok))
        emit(steps[-1] + "\n")
        if not ok:
            return Preflight(False, None, detail, steps)

    if not ctl.detect().get("daemon_up"):
        ok = ctl.ensure_serve()
        steps.append(progress.step("start ollama serve", ok))
        emit(steps[-1] + "\n")
        if not ok:
            return Preflight(False, None, "could not start ollama serve", steps)

    installed = {m["tag"] for m in ctl.list_models()}
    if installed:
        return Preflight(True, sorted(installed)[0], "", steps)

    tag, reason = pick()
    emit(f"  recommended: {tag}  ({reason})\n")
    ans = prompt(f"Pull {tag}? [Enter=yes / tag <x>] ").strip()
    if ans.lower().startswith("tag "):
        tag = ans[4:].strip() or tag
    ok, detail = ctl.pull(tag, on_progress=lambda s, c, t: emit("\r" + progress.pull_line(tag, c, t)))
    emit("\n" + progress.step(f"pull {tag}", ok) + "\n")
    if not ok:
        return Preflight(False, None, f"pull failed: {detail}", steps)
    return Preflight(True, tag, "", steps)


def doctor(ctl: Any, *, resources: ResourcePick | None = None) -> str:
    """Report-only health check with exact fixes. Never starts/installs anything."""
    pick = resources or _default_pick
    det = ctl.detect()
    models = [m["tag"] for m in ctl.list_models()] if det.get("daemon_up") else []
    tag, reason = pick()
    lines = [
        "aether doctor",
        progress.step("ollama installed", det.get("installed", False)),
        progress.step("ollama daemon up", det.get("daemon_up", False)),
        progress.step(f"a model is pulled ({len(models)})", bool(models)),
        f"  recommended for this machine: {tag}  ({reason})",
    ]
    if not det.get("installed"):
        lines.append("  fix: install -> https://ollama.com/download (or run /setup)")
    elif not det.get("daemon_up"):
        lines.append("  fix: start it -> run /setup (or `ollama serve`)")
    elif not models:
        lines.append(f"  fix: pull a model -> /pull {tag}")
    return "\n".join(lines)


__all__ = ["Preflight", "preflight", "doctor"]
