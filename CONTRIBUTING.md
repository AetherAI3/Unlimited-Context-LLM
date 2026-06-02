# Contributing to Unlimited Context

Thanks for being here. `aether-context` is an open project from Brandon Barrante ([Aether Ai](https://aethersystems.net)),
Apache-2.0, and we want it to be genuinely pleasant to hack on. This page gets you from a fresh
clone to a green test run in under ten minutes, and points you at good first things to work on.

> **The one promise:** the core is **numpy-only** and runs **offline**. A clean clone with zero
> models installed must still pass `pytest` (the built-in `MockLLM` makes that true). If a change
> breaks the offline/clean-clone path, it's wrong — see [Project ethos](#project-ethos).

## Quickstart for contributors

You need **Python 3.10–3.13** and `git`. That's it. No GPU, no API key, no model download.

```bash
git clone https://github.com/aether/unlimited-context.git
cd unlimited-context
python -m venv .venv

# activate the venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\Activate.ps1       # Windows PowerShell

pip install -e ".[dev]"            # editable install + pytest, ruff, mypy
pytest -q                          # should be green, offline, in well under 10 minutes
```

That's the whole loop. `pip install -e ".[dev]"` pulls `numpy` plus the three dev tools
(`pytest`, `ruff`, `mypy`) and nothing else — no torch, no compile, no network. If `pytest` isn't
green on a fresh clone, that's a bug; please open an issue.

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

## How to make a change

1. **Open or claim an issue** so we don't duplicate work. For anything non-trivial, a quick comment
   describing your approach saves everyone time.
2. **Branch** off `main`: `git checkout -b fix/short-description`.
3. **Write the test first.** This repo is test-driven (see [Testing](#testing)). The test should fail
   (red), then your change makes it pass (green), then you refactor.
4. **Keep the change small and single-purpose.** One concern per PR is much easier to review and
   merge.
5. **Run the inner loop** (ruff + mypy + pytest) until it's all green.
6. **Open a PR** against `main`. The PR template will ask you for a short "why", a test plan, and a
   confirmation that the offline clean-clone path still works.

We aim to give a first response on PRs quickly. Small, well-scoped, well-tested PRs merge fastest.

## Testing

Tests live in `tests/` and use **pytest only** — no network, no GPU, no real model.

- **No network, ever.** Use the built-in `MockLLM` (a deterministic, dependency-free model) and stub
  servers. A test that reaches the network will be rejected.
- **Use the fixtures.** `tests/conftest.py` provides a temp pool directory, a seeded RNG, and a
  `MockLLM`. Prefer those over hand-rolled setup so tests stay hermetic and reproducible.
- **Import submodules directly** in tests (e.g. `from aether_context.encoder import StaticEncoder`),
  not via the package surface — the `__init__` export surface is intentionally finalized late.
- **Arrange–Act–Assert.** Keep each test focused on one behavior with a descriptive name, e.g.
  `test_returns_empty_window_when_pool_is_cold`.
- **Property tests are welcome** for the numeric modules (encoder, witness, pool): shape, unit-norm,
  determinism, monotonicity, budget ceilings.

```bash
pytest -q                  # whole suite
pytest -k encoder -q       # only tests matching "encoder"
pytest -x -q               # stop on first failure while debugging
```

## Code style

We keep the toolchain boring on purpose so contributions are easy to review.

| Tool | What it enforces | Command |
|---|---|---|
| **ruff** | lint + import order + formatting | `ruff check .` / `ruff format .` |
| **mypy** | static types on the package | `mypy aether_context` |
| **pytest** | the suite, offline | `pytest -q` |

House rules (these mirror the build plan and are enforced in review):

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

A few non-negotiables that shape what we accept:

- **numpy-only core.** The only runtime dependency of the core is `numpy`. The Ollama path uses the
  Python standard library (`urllib`) — no extra dependency. Everything else (`llama.cpp`, HF
  `transformers`/`torch`, `hnswlib`) is an **opt-in extra**. Adding a core dependency is a big deal
  and needs a strong, discussed reason.
- **dataclasses, not pydantic.** Config and data carriers are `@dataclass`es. This keeps the install
  light and is a deliberate choice for OSS credibility — fewer deps, more installs.
- **Local-first, offline-first.** No network, no API key, and no account are required for the engine
  to work. Honor that in every change.
- **Honest naming.** "Unlimited" means **reach, not attention**. Names, docstrings, and metrics say
  so. We don't oversell.

## Good first issues

Friendly, well-scoped places to start. Look for the `good first issue` label on the tracker, or pick
from these:

- **Docs polish.** Tighten `docs/local-models.md`, fix a confusing sentence, add a missing example,
  or improve a docstring. Always welcome.
- **More encoder property tests.** Add labeled similar/dissimilar string pairs to the encoder
  similarity-margin test, or add throughput/edge-case coverage (empty string, unicode, very long
  input).
- **`aether-context doctor` checks.** Add a friendly diagnostic for a common "it didn't work" case
  (free disk vs pool size, RAM vs index, Ollama reachable) with an exact fix command in the output.
- **A new backend adapter.** Implement the `LocalLLM` protocol for a backend we don't ship yet
  (vLLM, text-generation-inference, an OpenAI-compatible local server). Keep it behind an opt-in
  extra and guard the import.
- **Bench scenarios.** Add a scripted long-build scenario to `bench/drift_vs_window.py`, or a new
  metric (e.g. a sharper drift/contradiction detector) for the engine-on-vs-off comparison.
- **Windows / macOS / Linux quirks.** The mmap pool layout must reopen identically across OSes —
  reproduce and fix any platform-specific issue you hit.
- **Example apps.** Add a small, self-contained example under `examples/` that runs on `MockLLM` so
  it works on a clean clone.

If you're not sure whether an idea fits, open an issue and ask. We'd rather talk early.

## Reporting bugs and requesting features

- **Bugs:** use the bug report template. Include your OS, Python version, and the smallest repro you
  can manage (ideally on `MockLLM` so we can reproduce offline). Running `aether-context doctor` and
  pasting its output helps a lot.
- **Features:** use the feature request template. Tell us the problem first, then the proposed
  solution. Features that keep the core numpy-only and offline-first are easiest to land.

## Code of Conduct

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Be kind; assume good
faith.

## License

By contributing, you agree that your contributions are licensed under the project's
[Apache-2.0](LICENSE) license.
