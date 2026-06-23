# SWE-bench CodePro Eval — Runbook

## Phase A — generation (any host with OpenRouter key + git)
    export OPENROUTER_API_KEY=...
    # 1-instance smoke first:
    python -m bench.swe_eval --instances 1 --arms off,codepro --max-usd 25
    # full lite overnight:
    python -m bench.swe_eval --instances 0 --arms off,codepro --max-usd 25
    # -> runs/swe_eval/predictions_off.jsonl + predictions_codepro.jsonl (resumable; re-run to continue)

## Phase B — scoring (VPS5, Docker, Linux)
    ssh root@aether-vps5
    pip install swebench
    cd <repo>
    python -m swebench.harness.run_evaluation \
      --dataset_name princeton-nlp/SWE-bench_Lite \
      --predictions_path runs/swe_eval/predictions_off.jsonl --run_id off --max_workers 4
    python -m swebench.harness.run_evaluation \
      --dataset_name princeton-nlp/SWE-bench_Lite \
      --predictions_path runs/swe_eval/predictions_codepro.jsonl --run_id codepro --max_workers 4
    # -> <model>.off.json / <model>.codepro.json reports

## Headline
    python - <<'PY'
    from pathlib import Path
    from bench.swe_scoring import parse_report
    for arm in ("off", "codepro"):
        r = parse_report(next(Path('.').glob(f'*.{arm}.json')))
        print(arm, r["resolved"], "/", r["total"], "=", r["resolved_rate"])
    PY

## Live tuning knobs (re-run a subset, diff resolved-rate/cost)
    --pool-gb --turbovec-bits {0,4,8} --no-mpo-chain --recall-k --max-steps --window --instances N

## Notes
- 1-instance smoke MUST pass Phase B (Docker build + test run) before the overnight 300.
- $25 hard cap shared across arms+instances; halts clean + resumes (re-run same command).
- codepro arm uses the LOCAL engine (no VPS5 atlas oracle) -> Docker CPU/disk only.
- pool floor is 5 GB (engine guard); --pool-gb below 5 is rejected.
