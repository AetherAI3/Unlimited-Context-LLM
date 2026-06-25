# Complexity-Gated Propose↔Critique Reasoning Escalation — Design

**Date:** 2026-06-25 · **Repo:** Unlimited-Context (harness/engine) · **Status:** approved direction, pre-plan

## Goal

Lift SWE-bench resolved-rate beyond the measured ~2× (codepro 27 vs off 14 @ N=180) by making the
agent **think harder only when it needs to**. Today the bench agent is a flat read→grep→submit
loop with no escalation — it drops exactly the hard instances (vulns, workarounds, multi-file
fixes) where both `off` and current `codepro` fail. Add a **complexity gate** that, on hard
instances, escalates into a **propose↔critique debate** grounded by OpenMythos and locked into the
MPO chain so the win compounds.

Two wins, by design:
- **Time:** easy/medium instances stay on the cheap flat loop; only *heavy* ones pay the debate cost.
- **Solve:** the debate cracks hard instances a standard agent abandons.

## Mental-model correction (grounding this design)

From reading the live systems:
- **Serena** (`lib/serena_intake.py`) = intake/clarity gate. **Arbiter** (`lib/orchestrator/core/arbiter.py`)
  = pre-synthesis judge that already carries a `complexity_label: light|medium|heavy` + risk flags
  and escalates (sonnet→opus). The "two sides arguing" = Serena (proposer/intake) vs Arbiter
  (critic/judge) inside the orchestrator.
- **OpenMythos** (`lib.orchestrator.atlas.openmythos`) is **retrieval, not reasoning** — single-pass
  dual-retrieve `ground()`. It *feeds* a reasoner; it isn't one.
- The bench agent uses **neither** today. This design brings the Serena/Arbiter *pattern* (propose
  vs critique) into the agent loop, with OpenMythos as the per-round grounding, and the MPO chain as
  the memory that compounds wins.

## Scope

**This spec (one implementation plan):** complexity gate + propose↔critique debate + per-round
OpenMythos grounding + MPO-lock of wins + a `codepro_debate` bench arm + measurement on the heavy
subset.

**Follow-on (separate specs, noted not built here):**
- **GLM-5.2 weight-tune:** host GLM-5.2 (open/hostable) on the VPS; LoRA on engine+debate traces
  (recall → escalate → correct-patch) so the model natively escalates and exploits recall.
- **Atlas write-back:** on a resolve, record the winning fix/pattern into the VPS5 atlas →
  cross-run compounding (next session/problem easier), beyond the within-run MPO lock.

## Components & boundaries

New file **`bench/swe_debate.py`** (keeps `swe_eval.py` lean; one clear purpose):

1. **`classify_complexity(problem, signals) -> "light"|"medium"|"heavy"`**
   Cheap classifier. Heuristic first (problem-statement length; keywords: "security/vuln/race/
   deadlock/edge case/regression/intermittent/fails when"; # candidate files surfaced during the
   first K investigation steps), with an optional single LLM self-assessment for ambiguous cases.
   Mirrors Arbiter's light/medium/heavy. Threshold tunable; default: escalate on `heavy` only.

2. **`debate_patch(chat, tools, ground_fn, problem, *, rounds, max_tokens) -> (patch, trace)`**
   The propose↔critique loop (both roles = the agent model; GLM-5.2 later):
   - **Propose (Serena-role):** draft a SEARCH/REPLACE fix + 1-line rationale, given the
     OpenMythos-grounded context for the current focus.
   - **Critique (Arbiter-role):** attack it — "does this fix the *failing behavior*? wrong
     location? missed edge case? breaks tests?" → `{accept|reject, objections[]}`.
   - **Revise:** on reject, proposer revises against objections; `ground_fn` re-grounds on the
     objection focus (ties in iterative retrieve). Loop ≤ `rounds` (default 2). Stop on accept or
     exhaustion → return best candidate (apply via the empty-patch-hardened `edit_file`).

3. **Per-round OpenMythos grounding** — `ground_fn(query)` calls `openmythos.ground(query=...)`
   with the round's focus (the critic's objection / proposer's gap), not just the static problem.
   `ground()` already takes `query`; the only change is calling it per round with a focused query.

4. **MPO-lock the win** — on convergence/resolve, `session.remember()` the
   (problem → winning fix + key reasoning) so later instances in the run recall it (within-run
   compounding). Cross-run = atlas write-back (follow-on).

5. **`swe_eval` wiring** — new arm `codepro_debate` (or `--debate` on codepro): build the engine as
   today; in `run_instance`, after a short investigation, call `classify_complexity`; if `heavy`,
   route to `debate_patch`; else flat loop (byte-identical to current codepro). Flag-gated → off =
   identical behavior.

## Data flow

```
instance → flat investigate (K steps, populate engine + signals)
        → classify_complexity
            light/medium → flat submit  (cheap, unchanged)
            heavy        → debate_patch:
                             propose (grounded) → critique → [reject → reground+revise]* → submit
        → on resolve: session.remember(win)   # MPO lock, compounds within run
```

## Measurement

Bench arms: `off`, `codepro`, **`codepro_debate`**. Same instances. Report:
- Overall resolved-rate codepro vs codepro_debate.
- **Heavy-subset** resolve delta (where the lever must bite — the headline for this feature).
- Debate cost (extra $/heavy instance) + escalation rate (% classified heavy) → the time/solve trade.
- Empty-patch rate (should also fall — critique catches non-fixes).

Success = codepro_debate > codepro on the heavy subset at acceptable extra cost, with overall
resolved-rate up.

## Testing

- `classify_complexity` unit tests (keyword/length/file-count → label; ambiguous → escalate).
- `debate_patch` dry-run with a scripted mock: proposer emits a bad patch → critic rejects with an
  objection → proposer fixes → critic accepts → patch applied. Assert round cap, best-candidate
  fallback on no-accept.
- `swe_eval` flag-off path byte-identical (existing tests); `--debate` light path = flat loop.
- No live API in unit tests (mock chat + stub ground_fn).

## Risks

- **Debate cost** — gated to heavy only; escalation-rate metric guards runaway spend; round cap.
- **Critic false-accept / false-reject** — cap rounds, always submit best candidate; track
  accept-but-unresolved to tune the critic prompt.
- **Misclassification** — tunable threshold; default conservative (escalate only clear-heavy) to
  protect cost; can flip to escalate-on-ambiguous if heavy-subset wins justify it.
- **Blocked on credit** — like all live runs, needs the OpenRouter top-up; build + dry-run now,
  measure after.

## Out of scope (here)

GLM-5.2 hosting/LoRA; atlas write-back ingest; cross-run compounding; multi-model (separate
proposer/critic models) — all follow-on specs once this measures positive.
