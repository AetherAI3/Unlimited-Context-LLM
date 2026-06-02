# The long-build task spec (bench task)

This is the scripted build that `bench/drift_vs_window.py` runs **twice** — once with the
Unlimited Context engine **ON**, once **OFF** — to measure the difference. It is also the task
`examples/coding_agent.py` walks. The point is a build long enough that the load-bearing
**constraints**, stated once up front, scroll off a small model's native window before the build
finishes. The engine's job is to keep them reachable; the bench measures whether it does.

## The load-bearing constraints (stated once, in the spec)

Four invariants are declared in the **spec** stage and must hold across every later stage:

| Constraint | Value |
|---|---|
| database | `postgres` |
| auth token expiry | exactly `3600` seconds |
| api framework | `fastapi` |
| package manager | `uv` |

A stage "drifts" if it cannot reach a constraint it depends on. A build "completes" if all four
constraints are still reachable at the final review stage with no human re-priming.

## The stages

| # | Stage | Instruction | Depends on |
|---|---|---|---|
| 0 | spec | State the four constraints above. | — (plants them) |
| 1 | schema | Design the database schema and migrations for the user model. | postgres |
| 2 | auth | Implement the auth module: login, refresh, token expiry handling. | 3600 |
| 3 | api | Build the REST API endpoints for users and sessions. | fastapi |
| 4 | tests | Write the integration test suite for the auth and api modules. | 3600, fastapi |
| 5 | tooling | Set up the project tooling, lockfile, and CI pipeline. | uv |
| 6 | review | Review the whole build for consistency against the original spec. | postgres, 3600, fastapi, uv |

`--quick` runs stages 0–3 (a CI smoke); the full run walks all seven.

## How the two arms differ

- **OFF (raw window).** Only the model's native window is visible — modeled as the last
  `context_window` tokens of the transcript. As each stage appends output, the spec scrolls off
  the end. By the later stages the constraints are gone, so dependent stages drift and the build
  does not complete.

- **ON (engine).** The spec is **encoded into the local pool** at stage 0 and **paged back** by
  retrieval whenever a later stage needs it. The constraints stay reachable past the window, so
  dependent stages stay correct and the build completes unattended.

## Metrics (what the bench reports)

- **drift** — count of dependent stages that could not reach a constraint they depend on.
- **correctness** — fraction of dependent stages whose constraints were reachable.
- **hit_rate** — the pager's retrieval hit rate over the run (ON only).
- **completion** — did all four constraints survive to the final review stage?

The **delta** (ON minus OFF) is the pitch. In the hermetic mock configuration the model has no
real intelligence, so the bench measures pure **reachability of the planted facts** — exactly
the mechanism the engine provides — and proves ON beats OFF with no GPU and no network.

## Running it

```bash
python bench/drift_vs_window.py            # hermetic mock, full build
python bench/drift_vs_window.py --quick    # CI smoke
python bench/drift_vs_window.py --json     # machine-readable report
python bench/drift_vs_window.py --model ollama/qwen2.5   # run it for real (local model)
```

The default (`--model mock`) touches no network and no GPU. A real backend is used only when you
name one explicitly; if it cannot be reached the ON arm degrades to the mock and the report says
so — the bench never crashes on a missing daemon.
