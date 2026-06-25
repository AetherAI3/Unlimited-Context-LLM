# SWE-bench CodePro Eval — Results (2026-06-24)

**What:** does the Unlimited-Context engine make a model fix more real bugs? Same model
(`gpt-4.1-mini`), same SWE-bench-lite instances, with vs without the engine. Patches scored by
the **official** `swebench` Docker harness (the repo's own tests must pass). 0 scoring errors.

- **off** — raw model; the repo context the agent pulls is window-truncated (it forgets early reads).
- **codepro** — Unlimited-Context engine: overpool + TurboVec 8-bit + MPO chain (recall keeps the
  whole investigation in reach past the native window).

Interactive charts: [`charts.html`](charts.html). Runbook: [`RUNBOOK.md`](RUNBOOK.md).

## Headline

**~2× more real bugs fixed.** At the largest sample (N=180): **codepro 27 vs off 14** resolved
(15.0% vs 7.7%).

| N (instances) | off resolved | codepro resolved | lift | off % | codepro % |
|---|---|---|---|---|---|
| 27  | 3 | 7  | 2.33× | 11.1 | 25.9 |
| 46  | 4 | 11 | 2.75× | 8.7  | 23.9 |
| 89  | 6 | 16 | 2.67× | 6.7  | 18.0 |
| **180** | **14** | **27** | **1.93×** | **7.7** | **15.0** |

**Honest read:** the lift peaked at 2.67× on easier early instances and settled to ~1.9× as
harder cases came in. The defensible large-sample claim is **~2×**, not 2.7× — the engine's edge
is biggest exactly where the model would otherwise forget context. codepro leads at every
checkpoint.

## Tuning (knob sweep, N=18, directional)

| config | resolved / 18 |
|---|---|
| baseline (turbovec 8-bit, recall-k 8, chain ON) | 4 |
| turbovec 4-bit | 3 |
| recall-k 16 | 2 |
| MPO chain OFF | 1 |

- **MPO chain is the load-bearing piece** — off → near the no-engine floor.
- Don't over-compress (8-bit > 4-bit) or over-retrieve (k=8 > k=16).
- Baseline config is already near-optimal. N=18 is noisy — re-tune at N≥50 for significance.

## Next lever: empty-patch rate

codepro left **~27% of turns with no patch at all** (off ~18%) — each is a guaranteed zero,
mostly a SEARCH block whose whitespace didn't match the file. Fix shipped (whitespace-tolerant
**unique** match + re-feed the file on a miss). Cutting empties is pure headroom **above** the ~2×.
**Validation pending** (see blockers).

## Model projection — THEORY (not measured)

Only `gpt-4.1-mini` is real. Hypothesis: the engine multiplier tapers on stronger frontier models
(which already manage long context better) while absolute resolved rises. Turn the projection into
real bars by re-running with `--model` swapped (Opus / GPT-4.1 / GLM / DeepSeek-v4-pro). See the
clearly-labelled simulation chart in `charts.html`.

## Caps

- **Context (engine):** overpool **300 MiB/session** (`CODEPRO_SESSION_CAP_BYTES`, 50 MiB–2 GiB),
  TurboVec 8-bit, OpenMythos + UnlimitedContext. **Atlas grounding** wired (`--atlas-ground`):
  VPS5 atlas holds astropy/sympy/sklearn/scipy/pandas API docstrings = SWE-lite's libs — promising
  lever, score still infra-stuck.
- **Run:** output tokens capped at 4096 (was defaulting to 65k → cost + 402s); HTTP timeout 180s +
  per-instance deadline (a hung request had stalled the run).

## Blockers (external, not code)

1. **OpenRouter key drained** — the benchmark run spent its credit; codepro's large-context calls
   now 402. **Top up the key** before: empty-patch validation, full-300, atlas score.
2. **Atlas probe score** infra-stuck on VPS5 Docker image churn — re-score `atlas18.jsonl` on a
   quiet box (one run, don't interrupt).

After a top-up the remaining work needs no new code: rerun the empty-fix probe, resume to 300,
re-score atlas.

## Reproduce

```
# generation (any host w/ OpenRouter key + git)
python -m bench.swe_eval --instances 0 --arms off,codepro --max-output-tokens 4096 --out runs/full
# scoring (Linux + Docker)
python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path runs/full/predictions_<arm>.jsonl --run_id <arm> --max_workers 8
```
