# Contributing to Unlimited Context

`aether-context` is published **as-is** under Apache-2.0 by **Aether AI** ([Aether AI](https://aethersystems.net)).

**Bug reports, issues, and pull requests are welcome.** A small, focused PR with a passing test is the fastest way in — keep the offline / clean-clone path green (see below). You're also free to **fork** and take it wherever you need; the Apache-2.0 license makes that easy.

**Security vulnerabilities:** please do **not** open a public issue. Report them privately as described in [SECURITY.md](SECURITY.md) (private advisory).

> **The one promise:** the core is **numpy-only** and runs **offline**. A clean clone with zero
> models installed still passes `pytest` (the built-in `MockLLM` makes that true). If you fork and
> change something, keep that path green — see [Project ethos](#project-ethos).

## Dev setup (for forkers)

You need **Python 3.10–3.13** and `git`. That's it. No GPU, no API key, no model download.

```bash
git clone https://github.com/DBarr3/Unlimited-Context.git
cd Unlimited-Context
python -m venv .venv

# activate the venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\Activate.ps1       # Windows PowerShell

pip install -e ".[dev]"            # editable install + pytest, ruff, mypy
pytest -q                          # should be green, offline, in well under 10 minutes
```

That's the whole loop. `pip install -e ".[dev]"` pulls `numpy` plus the three dev tools
(`pytest`, `ruff`, `mypy`) and nothing else — no torch, no compile, no network.

### The inner loop

```bash
ruff check .                       # lint
ruff format .                      # auto-format
mypy aether_context                # type-check the package
pytest -q                          # run the suite (offline, fast)
pytest tests/test_encoder.py -q    # run one file while iterating
```

CI runs exactly these four steps on Python 3.10, 3.11, 3.12, and 3.13. If they pass locally, they
should pass in CI.

## Testing

Tests live in `tests/` and use **pytest only** — no network, no GPU, no real model.

- **No network, ever.** Use the built-in `MockLLM` (a deterministic, dependency-free model) and stub
  servers. A test that reaches the network doesn't belong here.
- **Use the fixtures.** `tests/conftest.py` provides a temp pool directory, a seeded RNG, and a
  `MockLLM`. Prefer those over hand-rolled setup so tests stay hermetic and reproducible.
- **Import submodules directly** in tests (e.g. `from aether_context.encoder import StaticEncoder`),
  not via the package surface — the `__init__` export surface is intentionally finalized late.
- **Arrange–Act–Assert.** Keep each test focused on one behavior with a descriptive name, e.g.
  `test_returns_empty_window_when_pool_is_cold`.
- **Property tests** suit the numeric modules (encoder, witness, pool): shape, unit-norm,
  determinism, monotonicity, budget ceilings.

```bash
pytest -q                  # whole suite
pytest -k encoder -q       # only tests matching "encoder"
pytest -x -q               # stop on first failure while debugging
```

## Code style

The toolchain is deliberately boring.

| Tool | What it enforces | Command |
|---|---|---|
| **ruff** | lint + import order + formatting | `ruff check .` / `ruff format .` |
| **mypy** | static types on the package | `mypy aether_context` |
| **pytest** | the suite, offline | `pytest -q` |

House rules baked into the code:

- **Type hints on every public function.** mypy runs over `aether_context/` in CI.
- **No bare `except:`.** Catch the specific exception and re-wrap it as a typed error from
  `aether_context.errors` (each carries a `.hint` with the fix). Never silently swallow an error.
- **No `print()` in library code.** Use `aether_context._log.get_logger(__name__)`. The library is
  silent by default (a `NullHandler`); the host app decides whether to surface logs.
- **Small, single-purpose modules.** Aim for 200–400 lines; 800 is a hard ceiling. Many small files
  beat a few large ones.
- **Fail-soft.** The pager and retrieval are an *optimization*, never a correctness dependency. A
  retrieval miss, encoder hiccup, or backend stall must degrade, log, and continue — never crash a
  long run.

## Project ethos

A few non-negotiables that shaped the project:

- **numpy-only core.** The only runtime dependency of the core is `numpy`. The Ollama path uses the
  Python standard library (`urllib`) — no extra dependency. Everything else (`llama.cpp`, HF
  `transformers`/`torch`, `hnswlib`) is an **opt-in extra**.
- **dataclasses, not pydantic.** Config and data carriers are `@dataclass`es. This keeps the install
  light — fewer deps, more installs.
- **Local-first, offline-first.** No network, no API key, and no account are required for the engine
  to work.
- **Honest naming.** "Unlimited" means **reach, not attention**. Names, docstrings, and metrics say
  so. We don't oversell.

## Reporting bugs

Open an issue with your OS, Python version, and the smallest repro you can manage (ideally on
`MockLLM` so it reproduces offline). Running `aether-context doctor` and pasting its output helps.

For security vulnerabilities, use the private advisory in [SECURITY.md](SECURITY.md) — not a public
issue.

## License

This project is licensed under [Apache-2.0](LICENSE).
