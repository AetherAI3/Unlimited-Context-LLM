# SWE-bench CodePro Eval — Design

**Date:** 2026-06-23 · **Repo:** Unlimited-Context · **Branch:** `feat/turbovec-bench`
**Status:** approved design, pre-implementation

## Goal

Measure dsv4-pro **coding** capability **before vs after** the "codepro full effort" engine,
on real SWE-bench-lite, the **same way** as the 2026-06-14 DeepSeek-v4-pro recall eval
(`bench/api_eval.py` — OpenRouter dsv4-pro adapter, agentic tool-loop, off-vs-on arms, hard
spend cap). The baseline eval measured *memory recall*; this measures *resolved-rate* (do the
repo's own tests pass after the agent's patch). Produces a headline off-vs-codepro number, then
feeds a live-tuning loop.

Non-goal: this does **not** route through the AETHER-CLOUD cloud `/agent` codepro endpoint or
the VPS5 atlas oracle. "codepro" here = the local Unlimited-Context `Session` engine at max
settings — the same engine that powers the cloud codepro lane. Atlas `caps`/turbovec wiring on
VPS5 is tracked separately and is irrelevant to this measurement.

## Two-phase architecture

Don't reinvent test execution. Split generation (ours) from scoring (official `swebench`).

### Phase A — generation (`bench/swe_eval.py`, ours)
Per instance, per arm, run an agentic tool-loop that drives dsv4-pro to produce a candidate
patch. Emit SWE-bench predictions JSONL: `{instance_id, model_name_or_path, model_patch}`.

- **Adapter:** reuse `OpenAICompatLLM` (`openai/deepseek/deepseek-v4-pro` via OpenRouter) — same
  as the baseline; streaming + tool-calling.
- **Loop:** reuse the `api_eval.py` agentic loop verbatim; swap the toolset + corpus.
- **Repo access:** per instance, shallow-clone the instance repo and `git checkout base_commit`
  into a read-only per-instance workdir on the run host. The agent's file tools read from it.
  (Phase B does its own isolated checkout inside Docker; this clone is only for the agent's
  reading during generation.)

### Phase B — evaluation (official `swebench`, on VPS5)
`python -m swebench.harness.run_evaluation --predictions_path predictions_<arm>.jsonl
--dataset princeton-nlp/SWE-bench_Lite --run_id <arm>` builds each instance's test Docker image
and reports resolved/unresolved. **Dep:** `pip install swebench` on VPS5 (not currently
installed). Docker required (Linux). Host = **VPS5** (154 GiB disk; runs the live atlas oracle
too, but Phase B is Docker CPU/disk only and the codepro arm uses the local engine, so the
oracle is untouched).

## Arms (Engine MAX)

| arm | context handling |
|---|---|
| `off` (baseline) | raw dsv4-pro; repo context the agent has pulled is truncated to the model's real window each turn (the "forgets early files" failure) |
| `codepro` | Unlimited-Context `Session`: large overpool pool, **TurboVec 8-bit** quantized encode, **MPO chain ON**, higher `max_steps` ("max reasoning"). File reads are encoded into the engine; each turn the engine recalls the relevant code slices instead of re-dumping whole files → holds far more of the repo in reach past the window |

Expected per the baseline (recall-phase cost −54%, coherence 0.15→1.0): the `codepro` arm sends a
**compact recalled** context → **cheaper per turn** than `off`, which drags a large truncated
transcript. The `off` arm is the cost driver. $25 hard cap is very likely sufficient for the
full 600 runs (baseline never hit $0.20); a halt would be `off` bulk and the run resumes.

## Agent tools (harness-hosted over the instance checkout)

- `read_file(path)` — returns file contents (codepro: also encoded into the engine pool)
- `grep(pattern, path_glob?)` — ripgrep over the checkout
- `list_dir(path)` — tree listing
- `edit_file(path, ...)` / emit final unified diff — captured as `model_patch`
- `run_tests(node_id?)` — optional pre-submit self-check (best-effort; final truth is Phase B)

Identical loop to `api_eval.py`; only the tool host + corpus differ.

## Reproducibility / resume

- Fixed, sorted 300-instance SWE-bench-lite list (seeded; reproducible).
- Per-`(instance, arm)` result cached to disk (predictions + per-instance status JSONL); the run
  **skips already-done** pairs on restart → survives crash and cap-halt; resumable to finish.
- **Shared $25 hard cap** persisted across arms; halt-on-hit the instant it is reached, leaving a
  clean partial. Mirrors the baseline's global-cap discipline.

## Outputs

- `runs/swe_eval/predictions_off.jsonl`, `predictions_codepro.jsonl`
- `runs/swe_eval/results_<arm>.json` (per-instance resolved + cost + tool counts)
- `docs/benchmarks/2026-06-23-swe-codepro/RESULTS.md` mirroring the baseline doc:
  headline resolved-rate **off vs codepro**, total + per-phase cost, **per-instance flips**
  (instances codepro resolves that raw can't), redundant-tool counts, context-recall signal.

## Live-tuning knobs (CLI flags)

`--pool-gb`, `--turbovec-bits`, `--mpo-chain {on,off}`, `--recall-k`, `--max-steps`, `--window`,
`--instances N` (subset), `--arms off,codepro`. Tuning = re-run a subset with a different config
and diff the resolved-rate / cost.

## Components & boundaries

- `bench/swe_eval.py` — orchestrator: dataset load, per-instance workdir, arm runner, cap, resume,
  predictions writer. Depends on: `api_eval` loop primitives, `aether_context` (`Session`,
  `quantize`), `OpenAICompatLLM`.
- `bench/swe_tools.py` — the repo file toolset (read_file/grep/list_dir/edit_file/run_tests) over
  a checkout dir. Pure, unit-testable with a fixture repo; no network.
- `bench/swe_scoring.py` — thin wrapper to invoke + parse the official `swebench` evaluation
  output into `results_<arm>.json`. Isolates the Docker/official-harness seam.
- Reuse (no change): `OpenAICompatLLM`, `Session`, `quantize`, the agentic loop in `api_eval.py`
  (extract the loop into a shared helper if needed so both evals call it — minimal, behavior-
  preserving).

## Testing

- `swe_tools` unit tests against a tiny fixture repo (read/grep/edit/diff capture) — deterministic.
- `swe_eval` dry-run mode (`--dry-run`) with a mock chat that emits a canned patch → asserts
  predictions JSONL shape, resume/skip, cap-halt, both arms wire the engine vs truncation. No API,
  no Docker.
- `swe_scoring` parser test against a captured sample `swebench` report JSON.
- Phase B itself (Docker) validated by a 1-instance live smoke before the overnight 300.

## Risks

1. **dsv4-pro tool-calling reliability on coding** — baseline proved the adapter works; coding
   patches are larger. Mitigate: `max_steps` budget + a final "emit unified diff" forcing turn.
2. **Patch applies but is empty/garbage** — Phase B marks unresolved (correct). Track "patch
   non-empty & applies" as a secondary metric to separate generation failure from test failure.
3. **$25 halts mid-run** — accepted; partial headline on completed subset, resumable.
4. **VPS5 Docker disk** — lite images fit in 154 GiB; prune images between batches if needed.
5. **swebench dataset/version drift** — pin `princeton-nlp/SWE-bench_Lite` + record the
   `swebench` package version in RESULTS.md.
