# Changelog

All notable changes to `aether-context` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

[Unreleased]: https://github.com/DBarr3/Unlimited-Context/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DBarr3/Unlimited-Context/releases/tag/v0.1.0
