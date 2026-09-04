# Changelog

All notable changes to `aether-context` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] — 2026-09-04

### Fixed
- `publish-npm.yml` was not valid YAML — `run: echo "dry run: packed ..."` put `": "` inside a
  plain scalar, so the file did not parse and GitHub could not see its `on:` block. Dispatching
  it failed with `Workflow does not have 'workflow_dispatch' trigger` while the trigger was
  present in the source. Both `run:` steps are block scalars now.
- The PyPI and npm project descriptions rendered with broken images and dead links.
  `pyproject.toml` sets `readme = "README.md"`, so the README *is* the long description, and
  neither registry resolves relative links: the live page showed three broken images and eight
  dead links (`SAFETY.md`, `USE_POLICY.md`, `CONTRIBUTING.md`, `LICENSE`, `CITATION.cff`,
  `docs/`, `examples/`). Every link and image is absolute now, and the three images that lived
  on GitHub `user-attachments` are committed under `assets/` and served from
  `raw.githubusercontent.com`, so they are versioned with the repo. A registry only re-reads the
  description on upload, which is the reason this release exists.

### Added
- `tests/test_workflow_yaml.py` — every workflow file must parse *and* still declare at least
  one trigger, with both publish workflows pinned as dispatchable. Nothing in CI parsed workflow
  YAML before, which is how the above reached `main`. `pyyaml` joins the `dev` extra for it; the
  runtime dependency set is unchanged (numpy only).
- PyPI and npm version badges, now that both registries carry the package.
- `assets/demo.gif` — a real captured session (`npx aether-context setup`, `doctor`, `status`)
  against a live Ollama, now the README hero in place of generated artwork.
- `docs/index.html` — the project landing page, published from `main` `/docs` at
  <https://aetherai3.github.io/Unlimited-Context-LLM/>, with Open Graph and Twitter card tags so
  a shared link unfurls instead of showing a bare URL.

### Changed
- `RELEASING.md` rewritten. It had documented tag-triggered publishing (removed in #63), owner
  `DBarr3` (a deleted account), environment `pypi` (the workflow uses `pypi-production`), and a
  `version` literal in `pyproject.toml` (now dynamic) — and it now records the trap that failed
  the first real publish: the pending publisher's *project name* field is the distribution name,
  not the repository name. A mismatch passes the OIDC exchange and then fails the upload with
  `400 Non-user identities cannot create new projects`.
- **README rebuilt.** Install moved to the top (it had been at line 310 of 386, below five
  separate tables about pool size and RAM); those five tables are now one **Sizing** section —
  the RAM material had been split in half by the MPO section, with the resident-index table and
  the shared-vs-separate summary sitting under a heading about the context chain. The
  `1,000×+` and `~9,000×` figures, four lines apart, are now one number. Every caveat is kept,
  and the benchmark caveat now leads with the fact that the headline run used a hosted model
  rather than a local one.
- `CITATION.cff` tracks the release version again.

### Removed
- `docs/superpowers/` and `docs/plans/` — 5,241 lines of internal build plans and specs, written
  for an agent workflow rather than for readers, several of them for work that has not shipped.
  History keeps them.

## [0.3.0] — 2026-09-03

First release published to a package registry: PyPI as `aether-context`, npm as `aether-context`.

### Added
- **`aether-context setup`** — the guided first run, in three steps: size the pool, check for a
  local model, then verify the engine with a real encode/retrieve round trip. The verification
  runs against the mock model in a throwaway directory, so it passes with no daemon, no network
  and no model pulled, and leaves the pool you just configured empty. `--pool N --yes` makes the
  whole command non-interactive, so it is safe in a Dockerfile or a CI step.
- **npm launcher** (`packages/npm-cli`, published as `aether-context`). `npx aether-context`
  finds a Python 3.10+ interpreter, builds a private virtualenv under the OS cache directory,
  installs the matching PyPI release into it, and forwards every argument through. Nothing is
  written to global `site-packages` and nothing needs `sudo`. Zero npm dependencies.
- `aether_context.ui` — the CLI's presentation seam (stdlib only; the core stays numpy-only).
  Color is emitted only to a real tty and obeys `NO_COLOR`/`FORCE_COLOR`/`TERM=dumb`; VT100 mode
  is enabled explicitly on Windows and styling is dropped if that fails. Box-drawing and glyphs
  degrade to ASCII — prose included — when the stream's encoding cannot represent them, so a
  `cp1252` console gets readable text instead of `?` characters or a `UnicodeEncodeError`.
- `publish (npm)` workflow, and a preflight that refuses to publish a launcher whose pinned
  release is not yet on PyPI.

### Changed
- **The distribution now ships `aether_context` only.** `aether_agent/` stays in the repo and in
  the test suite, but is no longer packaged, and the `aether` / `aether-smoke` console scripts
  are gone from this distribution. PyPI's `aether-agent` already owns that import path and the
  `aether` command; pip does not detect file conflicts across distributions, so shipping a second
  copy would have silently overwritten the other package's files wherever both were installed.
  Install the terminal from its own package: `pip install aether-agent` or
  `npm install -g aether-agents`.
- The version is declared once, in `aether_context.__version__`, and read dynamically by the
  build. A release-parity test pins the npm launcher's version to it.
- `init`, `doctor` and `status` render through the new presentation seam. The `status` column
  layout and the doctor's `[ok]`/`[warn]`/`[fail]`/`[skip]` bracket text are unchanged — both are
  scraped by scripts — and the slice meter is appended after the counts rather than replacing them.
- `publish (pypi)` builds an sdist and a wheel, asserts the wheel contains no `aether_agent`
  files, and runs `twine check` before uploading. Its environment is now `pypi-production`,
  matching the sibling `aether-agent` and `agent-browser` repos.
- README: corrected the install instructions, and the claim that this package ships the `aether`
  command.

## [0.2.0] — 2026-08-13

### Added
- **Permanently retained slices.** `Session.pin(text, tags=...)` encodes a slice the witness
  never fades and the byte governor never evicts, at any pool pressure (equivalent to
  `remember(..., pinned=True)`). Permanent slices are **re-pinned from the pool when a session
  opens**: permanence is re-derived from the slices' own tags, so a pinned fact survives a
  restart instead of coming back as ordinary content and fading on the first long run. Use it
  for the handful of load-bearing constraints a long-running agent must never lose.
- **MPO context chain (on by default).** Links the session's slices into one connected
  structure and assists retrieval: when cosine pulls an entry slice, the MPO (Matrix Product
  Operator) chain pulls in the slices most coupled to it — widening the working set with the
  *connected thread* instead of isolated nearest-neighbors. Cosine stays the retrieval
  mechanism; the chain improves selection accuracy. Deterministic, numpy-only, fail-soft
  (degrades to plain cosine). In a planted-thread bench, connected-context recall@8 rises
  0.15 → 0.78. Disable with `Session(mpo_chain=False)` / `--no-mpo-chain`.
- `Witness` **temporal lock-in (anti-thrash)** — a freshly touched (just paged-in) slice carries
  a short-lived eviction bonus (`pin_periods` / `pin_bonus`) so the byte governor cannot evict it
  straight back to disk on the next turn, breaking the evict→cold-miss→re-page window flap. The
  bonus is small by design: it beats comparable-salience churn but never overrides a genuinely
  load-bearing slice, and it affects eviction ordering only (retrieval ranking is unchanged).
  `ContextPool` now drives eviction through `Witness.eviction_order` at a monotone write tick.

### Changed
- `ContextPool` HNSW index now **adds rows incrementally** (`O(new)`) instead of rebuilding the
  whole graph (`O(N)`) on every search-after-add; a full rebuild happens only when eviction or a
  reopen renumbers rows. Removes the quadratic insert cost on long, high-write runs.
- `--index tiered` is no longer a silent capability claim: it now **warns and runs the flat index**
  (it was always falling back to flat). README/CLI wording updated to match until a real paged-graph
  index ships.

## [0.1.0] — 2026-06-02

Initial public engine. Give any local LLM a billion-token *reach* via an encode-and-page context
pool — local-first, numpy-only core.

### Added
- `Session` — open → stream+encode+fade → paged reason → close lifecycle for local models.
- Local-model wrapper (`local_llm.py`): Ollama (stdlib `urllib`, no extra dep), llama.cpp, Hugging
  Face, and a deterministic offline `MockLLM`. One spec string: `ollama/qwen2.5`,
  `llamacpp:/path.gguf`, `hf/org/model`, `mock`.
- `StaticEncoder` — numpy-only, generate-on-import 256-dim embedder (`ENCODER_VERSION = static_v1`),
  validated by a supervised similarity-margin test.
- `ContextPool` — session-namespaced, budget-governed vector store (flat numpy index always; optional
  `hnswlib` via the `[fast]` extra); persists and reopens.
- `Witness` — +/- retention (harden / fade / re-harden) and the pool budget governor.
- `Pager` (`slice_loader`) — predictive prefetch + hit-rate; concurrency driven by the streaming
  session loop.
- `Session(fallback_to_mock=True)` — degrades to the mock model with a visible warning when a backend
  can't be loaded, so a clean-clone / offline run never crashes.
- `aether-context` CLI: `init`, `run "<task>"`, `chat` (REPL slash-commands `/clear` `/cls`
  `/new` `/status` `/pool` `/model` `/think` `/export` `/help` `/quit`), `status`, `clear` /
  `clear --all` (honest resident-vs-pool semantics, confirm on shared/persistent), `doctor`,
  `bench`, plus `--pool` / `--pool-mode {separate,shared}` / `--index {flat,hnsw,tiered}` / `--model`.
- `bench/drift_vs_window.py` — head-to-head engine ON vs OFF (drift / correctness / hit rate /
  completion), hermetic via `MockLLM`.
- Docs (`how-it-works`, `local-models`), examples, CI, Apache-2.0 license.

[Unreleased]: https://github.com/AetherAI3/Unlimited-Context-LLM/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/AetherAI3/Unlimited-Context-LLM/releases/tag/v0.3.1
[0.3.0]: https://github.com/AetherAI3/Unlimited-Context-LLM/releases/tag/v0.3.0
[0.2.0]: https://github.com/AetherAI3/Unlimited-Context-LLM/releases/tag/v0.2.0
