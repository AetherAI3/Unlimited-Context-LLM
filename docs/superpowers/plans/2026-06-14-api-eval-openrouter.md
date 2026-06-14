# API Eval (OpenRouter) Implementation Plan

> Implemented in this session. See `docs/superpowers/specs/2026-06-14-api-eval-openrouter-design.md`.

**Goal:** Autonomous harness proving the engine on a real DeepSeek/OpenRouter model: a lean agent loop works 50–100 real GitHub issues with tools, measuring cost, tool usage (+redundant), coherence drift, and work-outcome across the session — ON vs OFF — and emits a start→finish line graph.

**Architecture:** New `OpenAICompatLLM` backend (stdlib urllib; `generate` streaming + `chat(messages,tools)`). New `bench/api_eval.py`: fetch+cache issues → lean agent loop with 2 synthetic tools → arms (off / on / on_chain) → per-turn metrics → JSON+CSV → optional matplotlib PNG.

## Tasks
1. `OpenAICompatLLM` adapter in `aether_context/local_llm.py` + `tests/test_local_llm_openai.py` (mocked HTTP): spec `openai/<ref>`, env/base_url resolution (OpenRouter default), streaming `generate` (reasoning delta ignored), `chat` returning `{content,tool_calls,usage}`, typed errors, key never logged.
2. `bench/api_eval.py` + `tests/test_api_eval.py` (dry-run): GitHub issue fetch+cache, `IssueTools` (lookup_issue/search_issues + redundant count), lean agent loop, arms off/on/on_chain (ON retrieves via `session._cold_retrieve`), metrics (cost from `usage`, tools, redundant, coherence per turn, work-outcome vs real label), JSON+CSV, `--plot` (guarded matplotlib), `--dry-run` (`_MockChat`). `.gitignore` artifacts.
3. Full suite + moat-seal guard + ruff/mypy + push + PR.

## Expected results (hypotheses)
- **Cost:** OFF curves up super-linearly (re-sends growing transcript); ON/ON+CHAIN near-flat; DeepSeek caching makes ON savings clear by the final third.
- **Coherence:** OFF drifts down; ON holds; ON+CHAIN highest; gap widens over turns.
- **Tools:** OFF redundant lookups climb; ON/ON+CHAIN far fewer.
- **Outcome:** correct-triage ON+CHAIN > ON > OFF, gap growing late-session.
- **Graph:** `api_eval_plot.png` — cumulative cost + coherence vs turn, one line per arm.

Full task-by-task code is implemented directly in this session (adapter, harness, tests).
