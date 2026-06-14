# API Eval — Results (DeepSeek-v4-pro via OpenRouter)

**Date:** 2026-06-14 · **Model:** `deepseek/deepseek-v4-pro` · **Corpus:** 60 real
`microsoft/vscode` issues · **Task:** read N issues, then recall an early issue's label *from
memory, no re-lookup* · **Window:** 2000 tok (forces overflow) · **Turns:** 40 (20 read → 20
recall) per arm · **Spend:** $0.185 of $25 cap · no cost spikes, no halts.

Arms: **off** = no engine (transcript truncated to the window) — *this is the before baseline*.
**on** = engine (chain off). **on_chain** = engine + MPO context chain.

## Headline

| arm | recall coherence | work outcome | total cost | tools (redundant) |
|---|---|---|---|---|
| **off** (baseline) | **0.15** | **3 / 20** | $0.0711 | 26 (1) |
| **on** | **1.00** | **20 / 20** | $0.0542 | 39 (3) |
| **on_chain** | **1.00** | **20 / 20** | $0.0600 | 43 (5) |

**Engine vs baseline: coherence 0.15 → 1.00 (6.7×); cost −24% overall, −54% in the recall
phase; work outcome 3/20 → 20/20.**

## Coherence over the session (the drift curve)

Recall-phase coherence by turn (turns 21–40). Graph: `runs/api_eval_plot.png`.

- **off:** `0 0 0 0 0 0 0 0 0 .10 .09 .08 .15 .14 .13 .12 .12 .11 .10 .15` — the baseline
  **forgets**: the oldest reads have fallen out of the window (flat 0), and it only scores when
  a recall happens to target a *recent* read still in-window. Ends at **0.15**.
- **on / on_chain:** `1.0` every recall turn — the engine keeps every early read reachable.
  **No drift.**

This is the core claim, on a real reasoning model: without the engine the model loses early
facts as the session grows; with it, coherence stays flat at 1.0.

## Cost & caching

| phase | off $/turn | on $/turn | reduction |
|---|---|---|---|
| read (turns 1–20) | 0.00224 | 0.00219 | ~0% (both reading) |
| **recall (turns 21–40)** | 0.00117 | **0.00053** | **−54%** |

The reduction shows up exactly where the hypothesis predicted — **in the back half of the
session**, once the baseline is dragging a large (truncated) transcript into every call while
the engine sends a compact recalled window. DeepSeek prompt-caching is active in both arms
(cumulative cached prompt tokens: off 52.9k, on 12.2k, on_chain 10.8k) — the baseline re-sends
a big repeated prefix so it *caches more*, yet still costs more in absolute terms because it
ships far more tokens overall. Net: **−24% total cost, −54% recall-phase cost** for the engine.

## Tool usage / work

- All arms complete the work; **only the engine arms get it right** (20/20 vs 3/20).
- Redundant lookups stayed low on all arms because the recall instruction said "do not look it
  up" and the model obeyed — so the baseline **failed quietly from faded memory** rather than
  re-fetching. (Redundant-tool pressure would surface on a task that *permits* re-lookup; here
  the memory miss shows up as the coherence collapse instead.)

## on vs on_chain (honest)

On this **single-fact** recall task the MPO chain shows **no marginal gain** over plain engine
recall (both 1.00) and costs slightly more (more connected slices pulled → more tool/looks).
That's expected: the chain's value is **multi-slice connected threads**, not single-needle
recall (the synthetic planted-thread bench showed chain lifting connected-context recall
0.15 → 0.78). A future eval with a thread task would isolate the chain's contribution.

## Verdict

On a real hosted reasoning model, over a session that overflows the window, the engine
delivers: **coherence 0.15 → 1.00**, **work outcome 3/20 → 20/20**, **cost −24% overall /
−54% late-session**. The baseline (off) matches the expected "forgets and drifts" failure;
the engine holds the thread for a fraction of a cent.

Artifacts: `runs/api_eval_results.json` (per-turn detail), `runs/api_eval_series.csv`,
`runs/api_eval_plot.png` (cumulative cost + coherence vs turn, per arm).
