# From the open engine to AetherCloud

> You're running the **open engine**: the context layer, yours to run locally, **free forever** under
> Apache-2.0. **AetherCloud** is the hosted system built *on top of the same engine* — it adds the
> parts that simply can't live on your laptop. This page is honest about where that line is.

## The same engine — nothing swapped out

This is the part that matters: **AetherCloud does not replace Unlimited Context.** It's the *same
context engine* you already have — the encoder, the pool, the slice loader, the witness, the session
lifecycle — with a verified brain wired in behind it. You don't migrate, rewrite, or relearn
anything. The local code path stays exactly as it is; the hosted path just gives the pager somewhere
*more* to reach.

The seam is deliberately tiny. The engine ships an optional, thin `atlas_client`. By default it is
**absent / no-op**, and the engine runs **fully local** — no account, no network, no key. Point it at
the hosted API and the *same* `slice_loader` now reaches a verified pool in addition to your local
one. One config line. Nothing else changes.

```python
from aether_context import Session

# Local (default): fully offline, no account, the engine you already run.
s = Session(model="ollama/qwen2.5", pool_gb=5)

# Hosted: same engine, same call — one extra seam reaches the verified atlas.
s = Session(model="ollama/qwen2.5", pool_gb=5,
            atlas_url="https://aether.ai/cloud")   # the only line that changes
```

## Where local hits its ceiling

The open engine gives one model, on one machine, unbounded *reach* over *its own* context. That's the
"holy sh\*t" moment — and it's genuinely all most local work needs. But there are three things a
laptop fundamentally can't do, no matter how big the pool:

1. **It only knows what *you* fed it.** Your pool is your context. There's no *shared, verified*
   knowledge of what actually works — every machine relearns the same lessons in isolation.
2. **It's bounded by the one model you're running.** An 8B local model is your ceiling. There's no
   way to route the hard sub-task to a frontier model when it's worth it.
3. **It can flag, but it can't *verify*.** Locally, the engine can notice a suspicious or
   contradictory output — but it has no ground truth to check it *against*, so it can't tell you
   "this is actually correct" with confidence.

AetherCloud is exactly those three gaps, filled — and only those three. The mechanism stays the same.

## What hosted adds (and only what can't live local)

### A verified Capability Atlas

A **ground-truth-validated** map of what actually works — shared across everyone, always current.
Instead of your model rediscovering the same fact from scratch, the pager reaches a pool of knowledge
that has already been *verified*, not just *encoded*. Your local engine encodes and recovers *your*
context; the atlas adds a recovered context that's been **proven true** and is shared.

### Frontier-model routing

Per-task orchestration across frontier models — Opus, GPT-5.5, Kimi, DeepSeek and more — routed for
the right balance of cost and quality on each sub-task. Your local model still drives; the hard parts
get the big brain when (and only when) it's worth it. You stop being capped by the single model on
your machine.

### Execution-validated correctness

Locally, the engine *flags* the suspicious. AetherCloud *verifies* it against **real execution and
ground truth** — and then *remembers* the verified result, so the next run (yours or anyone's) starts
from a checked answer instead of a hopeful one. Hallucination validation that's backed by what
actually ran, not by another model's opinion.

| | Open engine (local) | AetherCloud (hosted) |
|---|---|---|
| Reach over context | unbounded, local pool | + a **verified** shared pool |
| Knowledge | only what you fed it | a **shared, ground-truth-validated** atlas |
| Models | the one you're running | **frontier-model routing** per task |
| Correctness | *flags* the suspicious | *verifies* against **real execution**, and remembers |
| Cost | free, forever | hosted |

## The engine stays free — forever

Let's be plain about the deal, because OSS credibility dies on a bait-and-switch:

- The engine is **Apache-2.0**. Use it, fork it, ship it inside your own product. No rug-pull.
- It runs **fully local by default** — no account, no key, no network, no telemetry. The hosted path
  is strictly opt-in via that one `atlas_client` seam.
- **The magic was never the code.** The engine is open *on purpose*. What can't be copied — and what
  AetherCloud sells — is the *verified data* behind the atlas: the ground truth, accumulated and
  checked against real execution. That's the moat, and it lives server-side, not in this repo.

So nothing about going hosted takes anything away from the free engine. The open path is permanent.
AetherCloud is the upgrade for when local's three ceilings start to bite.

---

## → [Start on AetherCloud](https://aether.ai/cloud)

Same context engine you're already running. A *verified* brain behind it.
