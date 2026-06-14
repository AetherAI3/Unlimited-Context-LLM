# API Eval (OpenRouter) — Design

**Date:** 2026-06-14
**Status:** Approved-for-planning
**Repo:** Unlimited-Context-LLM (`aether-context`)

## 1. Summary

A simple, autonomous test harness that proves the engine on a real hosted reasoning model
(DeepSeek via OpenRouter). A lean agent loop works through 50–100 real GitHub issues using
tools; the harness measures, across the whole session: **cost, efficiency, tools called
(incl. redundant), coherence, and the gain** from the engine + MPO chain.

Pure OpenRouter — no AETHER-CLOUD, no aether infra. Hand over a burner `OPENROUTER_API_KEY`
and it runs end-to-end with no intervention.

Keep-it-simple budget: **one adapter + one bench file + their unit tests.** No reuse of the
full headless brain/kernel/protocol — a tiny purpose-built loop in the bench is simpler and
fully controlled.

## 2. Goals / Non-Goals

### Goals
- Vendor-neutral `OpenAICompatLLM` (`openai/<model>` spec) — any OpenAI-compatible API. stdlib
  `urllib`, no new dependency. Used for both Session streaming and the loop's tool-calling chat.
- A lean agent loop (in the bench) with 2 synthetic tools over the fetched issues; counts tool
  calls and **redundant** tool calls.
- Measure over the session: cost ($), efficiency (tokens/latency), tools called + redundant,
  coherence per turn, and ON-vs-OFF gain.
- Autonomous: `OPENROUTER_API_KEY` + `--model`; default a cheap DeepSeek reasoning slug.

### Non-Goals
- No AETHER-CLOUD / aether infra / private namespaces (moat-seal guard must pass).
- Not in the CI matrix (paid + needs a key). CI runs the adapter unit tests (mocked HTTP) and
  the bench `--dry-run` (MockLLM) only.
- No real filesystem/tool execution — tools return synthetic results from the issue data.
- No new runtime dependency.

## 3. Adapter — `OpenAICompatLLM` (`aether_context/local_llm.py`)

Implements the `LocalLLM` protocol; one place for OpenRouter HTTP.

- **Spec** `openai/<model>` → `OpenAICompatLLM(ref, base_url=…, api_key=…, context_window=…)`;
  wired into `parse_spec` + `load_model`.
- **Config:** kwargs else env. `OPENROUTER_API_KEY` set + no base_url ⇒ default
  `https://openrouter.ai/api/v1`. Generic fallback `OPENAI_BASE_URL`/`OPENAI_API_KEY`.
- **`generate`** (LocalLLM, streaming) — POST `chat/completions` `stream:true`; yield
  `choices[0].delta.content`; reasoning `delta.reasoning` ignored. For the Session arms.
- **`chat(messages, tools)`** (bench helper, non-stream) — POST `chat/completions` with
  `tools`; return `{content, tool_calls, usage}` (real prompt/completion token counts from the
  API `usage` field — drives the cost metric). For the agent-loop arms.
- **`context_window`** caller-provided (eval forces it small); **`count_tokens`** chars/4.
- **Errors** typed/hinted (401/429/5xx → `BackendUnavailable`); key never logged.

## 4. The harness — `bench/api_eval.py`

Autonomous, paid. **No key ⇒ print skip + exit 0.** Flags: `--model`, `--repo` (default a big
public repo), `--issues` (50–100), `--turns`, `--window` (small, force overflow), `--arms`,
`--max-calls` (hard cap), `--dry-run`.

### 4.1 Corpus
Fetch issues via GitHub REST (stdlib urllib, unauthenticated, `per_page=100`), cache to
`bench/.cache/issues-<owner>-<repo>.json`. Each issue's real fields (number, title, body,
labels) are ground truth.

### 4.2 Tools (synthetic, from the corpus)
- `search_issues(query)` → list of `{number, title}` whose title/labels match.
- `lookup_issue(number)` → that issue's body + labels.
The harness answers these from the cached data and records each call.

### 4.3 Agent loop (lean, in the bench)
A task that forces the model to revisit earlier issues (e.g. "triage these issues: group by
component, then for each early issue confirm its label" — the early ones overflow the window).
Per turn: call `adapter.chat(messages, tools)`; if it returns tool_calls → answer them
(synthetic), append, continue; else it's the answer → score. Cap by `--turns` / `--max-calls`.

**Engine integration / arms:**
- **OFF** — no engine: messages = full growing transcript **truncated to `--window`** (early
  issues fall off → the model must re-`lookup_issue` them).
- **ON+CHAIN** (default on) — tool results + turns are encoded into a `Session` pool; each
  turn's prompt is built from the engine's recalled working set (chain-expanded) instead of the
  full transcript, so earlier issues stay reachable without re-fetching.
- Optional middle arm **ON** (`--arms off,on,on_chain`) — engine, chain off — to isolate the
  chain's marginal value.

### 4.4 Metrics (per turn + session aggregate, per arm)
- **Cost** — prompt+completion tokens from API `usage` → $ via a `--price-in/--price-out`
  (per-1M) flag. OFF balloons each turn; ON stays flat → Δ$ = the headline.
- **Tools called** — total, and **redundant** (same `lookup_issue(n)` issued more than once =
  the model forgot). The engine should cut redundancy.
- **Coherence** — per turn: does the answer use the correct issue fields (exact-match against
  ground truth)? Plotted vs turn index → drift curve.
- **Efficiency** — tokens/turn, latency/turn, turns-to-done.
- **Gained** — ON / ON+CHAIN vs OFF deltas on all the above.

### 4.5 Output / safety
Prints a table (per arm: cost, tools, redundant, coherence, latency) + writes
`api_eval_results.json` (**gitignored**) with per-turn detail + model + counts + est. cost.
`--max-calls` hard-caps spend. Key from env only.

## 5. Testing
- `tests/test_local_llm_openai.py` (CI, mocked HTTP, no network): spec parse; streaming parse +
  `[DONE]`; reasoning-delta ignored; `chat` tool_calls parse + usage; error mapping; env
  base_url resolution; `count_tokens`.
- `bench/api_eval.py --dry-run` (MockLLM + a scripted tool-calling mock): exercises fetch-cache
  parse, the loop, tool counting, scoring, and the table — no key/network. CI runs the dry-run.

## 6. Reversibility
Pure addition: one backend (+spec wiring), one bench, one test. Default engine behavior
unchanged. moat-seal guard scans the additions and stays green (vendor-neutral).
