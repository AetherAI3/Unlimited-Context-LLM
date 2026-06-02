# Changelog

All notable changes to `aether-context` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- `atlas_client.py` — thin, optional, off-by-default client of the hosted AetherCloud API. Calls it;
  contains none of it.
- Docs (`how-it-works`, `local-models`, `aethercloud`), examples, CI, Apache-2.0 license.

[Unreleased]: https://github.com/aether/unlimited-context/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aether/unlimited-context/releases/tag/v0.1.0
