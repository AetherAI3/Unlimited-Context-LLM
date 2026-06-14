# Live Session Eval — DeepSeek-v4-pro over OpenRouter

**What this is:** a real, paid, end-to-end test of the Unlimited Context engine driving a
reasoning model through a long agent session that **overflows the model's window** — measuring
what the engine buys you against a no-engine baseline.

| | |
|---|---|
| **Model** | `deepseek/deepseek-v4-pro` (reasoning), via **OpenRouter** |
| **Adapter** | `OpenAICompatLLM` (`openai/<model>`) — streaming + tool-calling, stdlib only |
| **Corpus** | 60 **real GitHub issues** from `microsoft/vscode` (fetched live, cached) |
| **Harness** | `bench/api_eval.py` — lean agentic tool-loop, the harness acts as the tool host |
| **Date** | 2026-06-14 |
| **Spend** | **$0.185** of a $25 hard cap · no cost spikes · no halts |

## The arms

- **off** — *baseline.* No engine; the growing transcript is truncated to the window each call.
- **on** — engine (encode-on-spill + recall), MPO context chain off.
- **on_chain** — engine + the MPO context chain.

## The tasks

The model works the issues with two tools (`lookup_issue`, `search_issues`); the harness
answers them from the real issue data. Two task modes:

1. **recall** (run live here) — **read** N issues, then on later turns **recall an early
   issue's primary label from memory, no re-lookup.** Isolates *memory*: by the time it's
   asked, the early read has fallen out of the window. Ground truth = the issue's real label.
2. **thread** (`--task thread`, built; isolates the **chain**) — read issues, then **list ALL
   issues you read with label X.** Scored as set-recall. The chain's connected-context pull
   should surface more of the thread than plain recall. *Pending its own live run.*

Setup for the live recall run: **40 turns (20 read → 20 recall), window 2000 tokens**
(small, to force overflow), 60 issues available.

## Results (recall task)

| arm | recall coherence | work outcome | total cost | recall-phase $/turn | tools (redundant) |
|---|---|---|---|---|---|
| **off** (baseline) | **0.15** | **3 / 20** | $0.0711 | $0.00117 | 26 (1) |
| **on** | **1.00** | **20 / 20** | $0.0542 | **$0.00053** | 39 (3) |
| **on_chain** | **1.00** | **20 / 20** | $0.0600 | $0.00055 | 43 (5) |

### Baseline improvements the system proved
- **Coherence: 0.15 → 1.00 (6.7×).** The baseline forgets; the engine holds every early fact.
- **Work outcome: 3/20 → 20/20.** The task is only *completed correctly* with the engine.
- **Cost: −24% overall, −54% in the recall phase.** The win lands in the back half of the
  session (matches the "reduction by the final stages" expectation).

## Coherence over the session (drift)

Recall-phase coherence by turn (21→40):

- **off:** `0 0 0 0 0 0 0 0 0 .10 .09 .08 .15 .14 .13 .12 .12 .11 .10 .15` — flat 0 on the oldest
  reads (out of window), only scoring when a recall happens to hit a *recent* read. The classic
  forget-and-drift failure of a long agent run.
- **on / on_chain:** `1.0` on every recall turn — **no drift.**

See `runs/api_eval_plot.png` (cumulative cost + coherence vs turn, per arm).

## Cost & caching

| phase | off $/turn | on $/turn | reduction |
|---|---|---|---|
| read (1–20) | 0.00224 | 0.00219 | ~0% |
| **recall (21–40)** | 0.00117 | **0.00053** | **−54%** |

DeepSeek prompt-caching is active in both arms. The baseline re-sends a large repeated prefix
each call, so it actually *caches more* (cumulative cached prompt tokens: off 52.9k vs on 12.2k)
— but it still costs **more** because it ships far more total tokens. The engine sends a small
recalled working set, so even with less to cache its absolute cost is lower.

## on vs on_chain — honest

On the **single-fact recall** task the chain shows **no marginal gain** over plain engine
recall (both 1.00) and costs a little more (it pulls extra connected slices). That's expected:
the chain's value is **multi-slice connected threads**, not single-needle recall. The
synthetic planted-thread bench (`bench/chain_recall.py`) showed the chain lifting
connected-context recall **0.15 → 0.78**; the **thread** task above is built to confirm that on
the live model. The headline win on *this* task is the engine memory itself.

## Observability / safety during the run

The run was supervised with hard backstops: a **$25 global spend cap**, a **per-call
cost-spike abort** ($0.50), and **coherence flags** — plus staged checkpoints (a $1 smoke and a
$5 health check) before the full run was authorized. Nothing tripped; total spend was $0.185.

## Reproduce

```bash
export OPENROUTER_API_KEY=...   # a low-cap burner; revoke after
python -m bench.api_eval --model deepseek/deepseek-v4-pro --repo microsoft/vscode \
  --issues 60 --turns 40 --window 2000 --arms off,on,on_chain \
  --price-in 0.435 --price-out 0.87 --max-usd 25 --plot --out runs/
# thread task (isolate the chain):
python -m bench.api_eval --task thread --arms on,on_chain ...
```

Raw artifacts from the live run live under `runs/` (git-ignored): `api_eval_results.json`
(per-turn detail), `api_eval_series.csv`, `api_eval_plot.png`.
