# aether-context (npm)

**Unlimited Context — virtual memory for an LLM's attention.** Local-first, numpy-only core.

This is the npm launcher. The engine itself is Python; this package exists so that
`npx aether-context` works without you thinking about `pip` first.

```bash
npx aether-context setup
```

That single command sizes the local pool, checks for a local model, and verifies the engine
end to end. It works with no daemon, no network and no model pulled — the verification runs
against the built-in mock model.

## Install

```bash
npm install -g aether-context
```

Or run it without installing:

```bash
npx aether-context --help
```

## Requirements

**Python 3.10 or newer** on your PATH. The launcher does not bundle an interpreter; it finds
one, then creates a private virtualenv in your OS cache directory and installs the matching
`aether-context` release from PyPI into it. Your global `site-packages` is never touched and
nothing needs `sudo`.

If Python is missing or too old, the launcher says so and points at the fix rather than failing
with a stack trace.

## Commands

```bash
aether-context setup                 # guided first run
aether-context chat --model mock     # interactive REPL, works offline
aether-context run "your task"       # one-shot
aether-context status                # pool, reach, hit rate
aether-context doctor                # diagnose Ollama / disk / RAM
```

## Environment

| Variable | Effect |
| --- | --- |
| `AETHER_CONTEXT_PYTHON` | Interpreter to use, skipping the PATH search. |
| `AETHER_CONTEXT_VERSION` | Release to install (`latest` for the newest). Defaults to this package's own version. |
| `AETHER_CONTEXT_HOME` | Where the private virtualenv lives. |

## Prefer pip?

The Python package is the same software, published from the same commit:

```bash
pip install aether-context
```

## Links

- Source: <https://github.com/AetherAI3/Unlimited-Context-LLM>
- PyPI: <https://pypi.org/project/aether-context/>
- License: Apache-2.0
