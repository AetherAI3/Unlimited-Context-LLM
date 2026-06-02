<div align="center">

# ⚡ Unlimited Context

### Unlimited context for **Ollama** — give any local LLM a **billion-token memory**. Local-first, on your own machine, free.

**An open project from [Aether](https://aethersystems.net)** · Apache-2.0 · `pip install aether-context`

[![license](https://img.shields.io/badge/license-Apache--2.0-06b6d4)]() [![python](https://img.shields.io/badge/python-3.10+-14b8a6)]() [![built by](https://img.shields.io/badge/built%20by-Aether-7c3aed)]()

</div>

---

> **Your context window didn't get bigger. Its *reach* became unbounded.**
> Unlimited Context is virtual memory for an LLM's attention — a small, fast resident window over a vast local store, paged in and out *while the model reasons*. So an 8B model on your laptop stays coherent across a build that would blow past any context window on earth.

## The problem everyone hits

Long agentic runs all die the same way. The model fills its window, starts **compressing** its own history, silently drops the one detail that mattered three steps ago — and drifts. You've seen it: the runaway PR, the agent that confidently rewrites a function it already wrote, the build that falls apart at hour two. Bigger windows just delay it, and a crammed 1M-token window **rots in the middle** anyway.

## The fix: don't compress the overflow — *encode* it

Unlimited Context fixes the overflow, not the window. Instead of blindly summarizing what spills over, it **encodes and externalizes** it to a local pool on your disk, and pages the *right slice* back in exactly when the model needs it. Nothing load-bearing is silently lost — it's filed, and recoverable.

**Compress & forget ✗  →  Encode & recover ✓**

## What you get

- 🧠 **Unbounded reach** — ~1B tokens of encoded context in ~5GB on disk; the model reaches it in slices.
- ⚡ **Zero added latency** — the pager runs *concurrently* with generation, hidden behind the model's own thinking, so reaching the pool costs you no extra wall-clock.
- 🪟 **Curated beats crammed** — a small, relevant resident window outperforms a stuffed one (no lost-in-the-middle) — and costs less.
- 🔒 **Local-first** — your context never leaves your machine. Free storage, full privacy, works offline.
- 🤖 **Any model** — Llama, Qwen, Mistral, Phi — via Ollama, llama.cpp, or HF. Bring your own brain.
- 📉 **Coherence you can *measure*** — ship the head-to-head: same model, engine on vs off, watch the drift rate fall off a cliff.

## What that buys you in coding time
- The real win isn't the token count — it's that the wall disappears. A typical ~128K context window fills after well under an hour of active agent work, then starts compacting and forgetting. A 5 GB pool is ~9,000× bigger.
  
Vast-ballpark — assuming a busy coding agent encodes ~300K–1M keep-worthy tokens/hour (chatty swarms burn more, careful single agents less):
  -PoolReach≈ active autonomous coding before it even fills*5 GB~1.16B~1,200–3,900 hrs (weeks of nonstop building)10 GB~2.33B~2,300–7,800 hrs15 GB~3.49B~3,500–11,600 hrs20 GB~4.65B~4,700–15,500 hrs
  -For color: 5 GB of reach ≈ ~100M lines of code, or a shelf of ~8,000 books — you won't fill it in one sitting. <sub>Rough order of magnitude. And because the witnesses fade stale slices, the pool never hard-stops anyway — it just keeps        -what's relevant. Translation: run a build as long as you want; it won't lose the plot.</sub>
🧮 RAM & running many sessions
Built to stay light: vectors live on disk (mmap'd) — only the small index graph and a hot working set are resident.
  RAM  ≈  ~180 MB base (engine + encoder)  +  ~29 MB per GB of pool  +  ~30 MB per session
  Shared pool vs separate pools — the biggest RAM lever when you run many sessions:

--pool-mode shared — one pool, one index, all sessions reach the same memory. Index paid once; each session adds ~30 MB. RAM barely moves — spin up dozens. Great for related work or agent swarms. (No isolation between sessions.)
--pool-mode separate (default) — each session its own pool + index, fully isolated/private. One index each, so RAM scales with N × pool.

Pool8 GB · shared8 GB · separate16 GB · shared16 GB · separate5 GBdozens~13dozens~3310 GBdozens~7dozens~1815 GBdozens~4dozens~1220 GBdozens~3dozens~9
TL;DR: shared pool → RAM isn't your limit. Separate pools → ~3–13 sessions on 8 GB, double on 16 GB. A bigger pool always buys reach, never more sessions. Need headroom? Smaller pool, or --index tiered.

## Quickstart

```bash
pip install aether-context
```

```python
from aether_context import Session

s = Session(model="ollama/qwen2.5", pool_gb=5)
s.run("Build me a full-stack weightlifting tracker app.")
# runs long. stays coherent. walk away.
```

That's the whole thing. One small model, one command, a billion tokens of reach behind it.

## Pick your memory size

First run drops you into a slider — pick how much your model gets to remember:

```text
$ aether-context init
──────────────────────────────────────────────────────────────────
  ⚡ choose your context pool          encoded reach · not a window
──────────────────────────────────────────────────────────────────
  ▸  5 GB   ████░░░░░░░░░░░░   ~1.16B tokens   a big project   (floor)
     10 GB  ████████░░░░░░░░   ~2.33B tokens   a large monorepo + docs
     15 GB  ████████████░░░░   ~3.49B tokens   multiple repos / long runs
     20 GB  ████████████████   ~4.65B tokens   massive corpus / power user
──────────────────────────────────────────────────────────────────
  reach ≈ pool_GB × 233M tokens     custom: --pool 12  (any size ≥ 5 GB)
  ↑/↓ slide      ↵ confirm

  pool [5]: 10
  ✓ 10 GB  →  your model can now reach ~2.33 billion tokens
```

**The math, per tier** (derived, not vibes):

| Pool | Slices | Encoded reach | Slider |
|------|--------|---------------|--------|
| **5 GB** *(floor)* | 2.27M | **~1.16B tokens** | `████░░░░░░░░░░░░` |
| 10 GB | 4.55M | **~2.33B tokens** | `████████░░░░░░░░` |
| 15 GB | 6.82M | **~3.49B tokens** | `████████████░░░░` |
| 20 GB | 9.09M | **~4.65B tokens** | `████████████████` |

How those numbers come out: ~2.2 KB per slice (a 256-dim vector + compressed text + metadata) ÷ 512 tokens per slice → **~455K slices/GB → ~233M tokens of reach per GB**. So `reach ≈ pool_GB × 233M`. 5 GB is the floor; bump anytime with `aether-context --pool 20`.

> **Honest:** that's encoded **reach**, retrieved in slices — not a bigger attention window, and it rides on retrieval hit rate. A bigger pool buys more reachable codebase/corpus *per session* — not more concurrent sessions (those are RAM-bound, ~30 on 8 GB either way).

## RAM & running many sessions

The engine is built to stay light: **vectors live on disk (mmap'd)** — only the small HNSW index graph and a hot working set are ever resident. So RAM is a predictable formula, not a mystery:

```
RAM  ≈  ~180 MB   base (engine + shared static encoder)
      +  ~29 MB   per GB of pool   (resident index)
      +  ~30 MB   per active session
```

**Resident index cost by pool size:**

| Pool | Index RAM (resident) |
|------|----------------------|
| 5 GB | ~146 MB |
| 10 GB | ~291 MB |
| 15 GB | ~436 MB |
| 20 GB | ~582 MB |

### Shared pool vs separated pools

Running more than one session? You pick how the pool is shared — and it's the single biggest RAM lever:

- **`--pool-mode shared`** — one pool, one index, all sessions reach the same memory. The index is paid **once**; each extra session adds only ~30 MB, so RAM barely moves as you add sessions. Best for related work (same project) or max concurrency on a small machine. Trade-off: sessions can see each other's context (no isolation).
- **`--pool-mode separate`** *(default)* — each session gets its own pool + index, fully **isolated and private**. Clean, but you pay one index **per session**, so RAM scales with `N × pool`. Best for unrelated tasks or when isolation matters.

(The encoder is always shared — it's stateless, ~31 MB, loaded once. Only the pool/index differs.)

### How many sessions actually fit

| Pool | 8 GB · shared | 8 GB · separate | 16 GB · shared | 16 GB · separate |
|------|---------------|-----------------|----------------|------------------|
| 5 GB  | dozens¹ | **~13** | dozens¹ | **~33** |
| 10 GB | dozens¹ | **~7**  | dozens¹ | **~18** |
| 15 GB | dozens¹ | **~4**  | dozens¹ | **~12** |
| 20 GB | dozens¹ | **~3**  | dozens¹ | **~9**  |

<sub>Reserves: ~2.5 GB held back on an 8 GB machine, ~6 GB on 16 GB — the rest stays for your OS and editor. ¹ With a shared pool, RAM stops being the limit (50–70+ sessions fit); you're bounded by CPU and good sense, not memory.</sub>

> **TL;DR.** **Shared pool → RAM is not your limit**, spin up as many sessions as your CPU allows. **Separate pools → one index each**, so plan on ~3 (20 GB) to ~13 (5 GB) sessions on 8 GB, roughly double at 16 GB. A bigger pool always buys **reach**, never more sessions. Need more headroom? Shrink the pool, or run `--index tiered` to keep only warm graph nodes resident.

## How it works (60 seconds)

It's **virtual memory, for attention.** Map it to an OS and it clicks:

| OS | Unlimited Context |
|---|---|
| RAM | the **resident window** the model sees now (small, fast) |
| Disk | the **context pool** (~5GB, ~1B tokens, encoded) |
| Pager | the **slice loader** — prefetches the next slice from what the model is reasoning about *right now* |
| Page-replacement | the **witnesses (+/−)** — salient slices *harden*, stale ones *fade*, anything relevant again *re-hardens* |

All of it runs while the model generates, so the reach costs you nothing in wall-clock. → full explainer in [`docs/how-it-works.md`](docs/how-it-works.md).

## Honest about the word "unlimited"

"Unlimited" means **reach, not attention.** Your model keeps its native window — we make it *reach* a billion-token pool in slices, via fast retrieval. The whole thing rides on retrieval **hit rate**; when it's high (and the loader is built to keep it high), the pool feels like one seamless context.

## License

**Apache-2.0.** Use it, fork it, ship it in your product. Built by [Aether](https://aethersystems.net).

---

<div align="center">

Built by Brandon Barrante, Aether Ai  **[Aether](https://aethersystems.net)**
<img width="2752" height="1536" alt="image" src="https://github.com/user-attachments/assets/4b7eef9a-8b1c-4dc7-b926-771ce53ed04d" />

*Unbounded reach for the model you already run.*

</div>
