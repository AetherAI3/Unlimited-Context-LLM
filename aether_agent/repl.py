# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Native interactive REPL — the ``aether`` front door with no subcommand.

Mirror of aether-code ``src/commands/chat.ts`` ``repl()``. Prints the splash,
then loops on ``input()``: a line starting with ``/`` goes to
``slash.dispatch``; anything else is a chat turn driven by ``brains.select_brain``
whose events are rendered with the shared vocabulary (monologue / tool_call /
tool_result / done / error).

Policy: ``FileTokenStore`` has a token -> cloud brain; else local Ollama. The
``backend`` config knob (``auto`` / ``local`` / ``cloud``) is honored by
``select_brain``. Ctrl-C aborts the current turn (prints ``(interrupted)``);
Ctrl-C at an empty prompt (or EOF) exits. A non-TTY stdin uses the same plain
``input()`` loop (no raw-mode key handling — that lives in the TS host).

``readline`` is imported best-effort for arrow-key history on POSIX; on Windows
(no GNU readline) the plain ``input()`` loop is used unchanged.
"""
from __future__ import annotations

import sys
from typing import Any, Optional

from aether_agent.auth import FileTokenStore, auth_status
from aether_agent.brains import select_brain
from aether_agent.config import load_config
from aether_agent.ollama_ctl import OllamaCtl
from aether_agent.onboarding import preflight as _preflight_impl
from aether_agent.slash import SlashContext, dispatch
from aether_agent.splash import render_splash
from aether_agent.transport import ApiClient

#: Package version for the splash (best-effort; falls back if metadata absent).
try:  # pragma: no cover - trivial import guard
    from importlib.metadata import version as _pkg_version

    VERSION = _pkg_version("aether-context")
except Exception:  # noqa: BLE001
    VERSION = "0.1.0"

_PROMPT = "aether › "


def _backend_label(backend: str, authed: bool) -> str:
    """Human label for the brain that will serve turns this session."""
    b = (backend or "auto").strip().lower()
    if b == "cloud" or (b == "auto" and authed):
        return "cloud (Aether API)"
    return "local Ollama (offline)"


def _render_event(ev: dict[str, Any], out: Any) -> None:
    """Render one brain event to ``out`` (codec-safe)."""
    etype = ev.get("type")
    if etype == "monologue":
        text = str(ev.get("text", ""))
        if text:
            _safe_write(out, text + "\n")
    elif etype == "tool_call":
        _safe_write(out, f"  - {ev.get('name', '')}({_fmt_args(ev.get('args', {}))})\n")
    elif etype == "tool_result":
        _safe_write(out, f"  - {ev.get('name', '')} -> {_first_line(str(ev.get('output', '')))}\n")
    elif etype == "error":
        _safe_write(out, f"\n[x] {ev.get('msg', 'error')}\n")
    elif etype == "done":
        text = str(ev.get("text", ""))
        if text:
            _safe_write(out, text + "\n")


def _fmt_args(args: Any) -> str:
    if not isinstance(args, dict) or not args:
        return ""
    return ", ".join(f"{k}={_short(str(v))}" for k, v in args.items())


def _short(s: str, n: int = 40) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _first_line(s: str, n: int = 80) -> str:
    line = s.strip().splitlines()[0] if s.strip() else ""
    return _short(line, n)


def _run_turn(brain: Any, line: str, out: Any) -> None:
    """Drive one chat turn, rendering events. Ctrl-C aborts just this turn."""
    try:
        for ev in brain.run(line):
            _render_event(ev, out)
    except KeyboardInterrupt:
        _safe_write(out, "\n(interrupted)\n")


def _build_ollama(backend: str, authed: bool) -> Any:
    """The local Ollama controller, only when this session will serve locally."""
    b = (backend or "auto").strip().lower()
    if b == "cloud" or (b == "auto" and authed):
        return None  # cloud session — no local daemon to manage
    return OllamaCtl()


def _preflight(ctl: Any, **kw: Any) -> Any:
    return _preflight_impl(ctl, **kw)


def _make_ctx(authed: bool, api: Any, model: str, ollama: Any = None) -> SlashContext:
    return SlashContext(api=api, authed=authed, model=model, ollama=ollama)


def main(argv: Optional[list[str]] = None) -> int:
    """Run the interactive REPL. Returns a process exit code (0 on clean exit).
    ``argv`` is accepted for signature parity with other entry points but the
    REPL takes no positional arguments."""
    out = sys.stdout
    cfg = load_config()
    base_url = cfg.get("baseUrl", "")
    backend = str(cfg.get("backend", "auto"))
    model = str(cfg.get("defaultModel", "") or "")

    store = FileTokenStore()
    authed = store.get() is not None
    api = ApiClient(base_url, store)

    ollama = _build_ollama(backend, authed)
    chosen_model = model
    if ollama is not None:
        pf = _preflight(ollama, emit=lambda s: _safe_write(out, s))
        if not pf.ok:
            _safe_write(out, f"\n{pf.message}\n")
            ollama.stop_owned()
            return 1
        chosen_model = pf.chosen_model or model

    label = _backend_label(backend, authed)
    short_backend = "cloud" if "cloud" in label else "local"
    _safe_write(out, render_splash(VERSION, chosen_model or "auto", short_backend) + "\n\n")
    _safe_write(out, "Type a prompt, or /help for commands. /exit to quit.\n\n")

    ctx = _make_ctx(authed, api, chosen_model, ollama)
    is_tty = bool(getattr(sys.stdin, "isatty", lambda: False)())
    if is_tty:
        _enable_readline_history()

    try:
        while True:
            try:
                line = input(_PROMPT) if is_tty else _read_line()
            except KeyboardInterrupt:
                out.write("\n")
                return 0
            except EOFError:
                out.write("\n")
                return 0
            if line is None:
                return 0
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                res = dispatch(ctx, line)
                if res.get("exit"):
                    return 0
                text = res.get("text")
                if text:
                    _safe_write(out, text + "\n")
                if res.get("setup") and ollama is not None:
                    pf = _preflight(ollama, emit=lambda s: _safe_write(out, s))
                    if pf.ok:
                        ctx.model = pf.chosen_model or ctx.model
                    else:
                        _safe_write(out, f"\n{pf.message}\n")
                if res.get("restart"):
                    ctx.authed = store.get() is not None
                    _safe_write(out, "(session restarted — context cleared)\n")
                continue
            brain = select_brain(
                authed=store.get() is not None,
                backend=backend,
                api=api,
                model=ctx.model or chosen_model or "",
            )
            _run_turn(brain, line, out)
            out.write("\n")
    finally:
        if ollama is not None:
            ollama.stop_owned()


def _safe_write(out: Any, text: str) -> None:
    """Write text, surviving a terminal codec (e.g. cp1252) that can't encode a
    character — re-encode replacing the offender rather than crashing the REPL."""
    try:
        out.write(text)
    except UnicodeEncodeError:
        enc = getattr(out, "encoding", None) or "ascii"
        out.write(text.encode(enc, errors="replace").decode(enc, errors="replace"))


def _read_line() -> Optional[str]:
    """Read one line from a non-TTY stdin; None on EOF."""
    data = sys.stdin.readline()
    if data == "":
        return None
    return data.rstrip("\n")


def _enable_readline_history() -> None:
    """Best-effort: enable in-memory line history + editing on POSIX. A no-op on
    Windows (no GNU readline) — the plain input() loop still works."""
    try:  # pragma: no cover - depends on platform readline availability
        import readline  # noqa: F401
    except Exception:  # noqa: BLE001
        return


def _auth_summary(base_url: str, store: FileTokenStore) -> str:
    """One-line auth summary (kept here so the REPL and CLI ``auth status`` agree)."""
    st = auth_status(base_url, store)
    if not st["logged_in"]:
        return "not logged in"
    return f"logged in ({st['token_type']})"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "VERSION"]
