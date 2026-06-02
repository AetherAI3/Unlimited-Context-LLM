<div align="center">

# ⚡ Unlimited Context LLM

Give your Ai superpowers with **Unlimited context for [Ollama](https://ollama.com)** — give any local LLM a **billion-token memory**. Local-first, on your own machine, free.

<img width="537" height="405" alt="Unlimited Context" src="https://github.com/user-attachments/assets/79758729-ead7-42ca-9784-831cae68ef06" />

[![License](https://img.shields.io/badge/license-Apache--2.0-06b6d4)](LICENSE) [![Python](https://img.shields.io/badge/python-3.10%2B-14b8a6)](https://www.python.org) [![Built by Aether](https://img.shields.io/badge/built%20by-Aether-7c3aed)](https://aethersystems.net)

**An open project from [Aether](https://aethersystems.net)** · Apache-2.0 · `pip install aether-context`

</div>

---

> **Your context window didn't get bigger. Its *reach* became unbounded.**
> Unlimited Context is virtual memory for an LLM's attention — a small, fast resident window over a vast local store, paged in and out *while the model reasons*. So an 8B model on your laptop stays coherent across a build that would blow past any context window on earth.

## The problem everyone hits

Long agentic runs all die the same way. The model fills its window, starts **compressing** its own history, silently drops the one detail that mattered three steps ago — and drifts. You've seen it: the runaway PR, the agent that confidently rewrites a function it already wrote, the build that falls apart at hour two. Bigger windows just delay it, and a crammed 1M-token window **rots in the middle** anyway.

## The fix: don't compress the overflow — *encode* it

Unlimited Context fixes the overflow, not the window. Instead of blindly summarizing what spills over, it **encodes and externalizes** it to a local pool on your disk, and pages the *right slice* back in exactly when the model needs it. Nothing load-bearing is silently lost — it's filed, and recoverable.

<p align="center"><strong>Compress &amp; forget ✗ &nbsp;→&nbsp; Encode &amp; recover ✓</strong></p>

<div align="center">
  <img width="880" alt="Compress and forget vs encode and recover" src="https://github.com/user-attachments/assets/dadae038-5e1a-45c6-b16c-4763da4238a8" />
</div>

## How it works (60 seconds)

It's **virtual memory, for attention.** Map it to an OS and it clicks:

| OS | Unlimited Context |
|---|---|
| RAM | the **resident window** the model sees now (small, fast) |
| Disk | the **context pool** (~5 GB, ~1B tokens, encoded) |
| Pager | the **slice loader** — prefetches the next slice from what the model is reasoning about *right now* |
| Page-replacement | the **witnesses (+/−)** — salient slices *harden*, stale ones *fade*, anything relevant again *re-hardens* |

All of it runs while the model generates, so reaching the pool costs you no extra wall-clock. → full explainer in [`docs/how-it-works.md`](docs/how-it-works.md).

## What you get

- 🧠 **Unbounded reach** — ~1B tokens of encoded context in ~5 GB on disk; the model reaches it in slices.
- ⚡ **Zero added latency** — the pager runs *concurrently* with generation, hidden behind the model's own thinking, so reaching the pool costs you no extra wall-clock.
- 🪟 **Curated beats crammed** — a small, relevant resident window outperforms a stuffed one (no lost-in-the-middle) — and costs less.
- 🔒 **Local-first** — your context never leaves your machine. Free storage, full privacy, works offline.
- 🤖 **Any model** — Llama, Qwen, Mistral, Phi — via Ollama, llama.cpp, or HF. Bring your own brain.
- 📉 **Coherence you can *measure*** — ship the head-to-head: same model, engine on vs off, watch the drift rate fall off a cliff.

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

## What that buys you in coding time

The real win isn't the token count — it's that the wall disappears. A typical ~128K context window fills after well under an hour of active agent work, then starts compacting and forgetting. A 5 GB pool is **~9,000× bigger**.

Rough ballpark — assuming a busy coding agent encodes ~300K–1M keep-worthy tokens/hour (chatty swarms burn more, careful single agents less):

| Pool | Reach | Active autonomous coding before it even fills\* |
|:----:|:-----:|:------------------------------------------------|
| **5 GB** | ~1.16B | **~1,200–3,900 hrs** — weeks of nonstop building |
| 10 GB | ~2.33B | ~2,300–7,800 hrs |
| 15 GB | ~3.49B | ~3,500–11,600 hrs |
| 20 GB | ~4.65B | ~4,700–15,500 hrs |

For color: 5 GB of reach ≈ ~100M lines of code, or a shelf of ~8,000 books — you won't fill it in one sitting.

<sub>\* Rough order of magnitude. Because the witnesses fade stale slices, the pool never hard-stops anyway — it just keeps what's relevant. Run a build as long as you want; it won't lose the plot. The per-session RAM math is in [RAM footprint](#ram-footprint) below.</sub>

<div align="center">
  <img width="880" alt="Coding time per pool size" src="https://github.com/user-attachments/assets/af626850-96b1-43a2-91fd-b5162bc21e5a" />
</div>

## Common commands

A friendly cheat-sheet — the handful of commands you'll actually reach for:

| Command | What it's for |
|---|---|
| `aether-context init` | Pick your pool size — the on-disk storage slider — on first run. |
| `aether-context run "<task>"` | One-shot a task with full reach, then print the result. |
| `aether-context chat` | Open an interactive session; type `/status` anytime, `/clear` to reset. |
| `aether-context status` | See pool size, slices used, reach, and hit rate at a glance. |
| `aether-context doctor` | Check Ollama, your model, disk, and RAM before a long run. |
| `aether-context --pool 20` | Resize the pool anytime (non-destructive re-index). |

> **Tip:** run `aether-context doctor` first — it catches the three things that ever go wrong (Ollama down, model not pulled, not enough disk) and prints the exact fix.

## Running many sessions

Running more than one agent? How the pool is shared is the single biggest RAM lever:

- **`--pool-mode shared`** — one pool, one index, all sessions reach the same memory. The index is paid **once**; each extra session adds only ~30 MB, so RAM barely moves as you add sessions. Best for related work (same project) or max concurrency on a small machine. Trade-off: sessions can see each other's context (no isolation).
- **`--pool-mode separate`** *(default)* — each session gets its own pool + index, fully **isolated and private**. Clean, but you pay one index **per session**, so RAM scales with `N × pool`. Best for unrelated tasks or when isolation matters.

**How many actually fit:**

| Pool | 8 GB · shared | 8 GB · separate | 16 GB · shared | 16 GB · separate |
|------|---------------|-----------------|----------------|------------------|
| 5 GB  | dozens¹ | **~13** | dozens¹ | **~33** |
| 10 GB | dozens¹ | **~7**  | dozens¹ | **~18** |
| 15 GB | dozens¹ | **~4**  | dozens¹ | **~12** |
| 20 GB | dozens¹ | **~3**  | dozens¹ | **~9**  |

<sub>Reserves: ~2.5 GB held back on an 8 GB machine, ~6 GB on 16 GB — the rest stays for your OS and editor. ¹ With a shared pool, RAM stops being the limit (50–70+ sessions fit); you're bounded by CPU and good sense, not memory.</sub>

## RAM footprint

The engine stays light: **vectors live on disk (mmap'd)** — only the small HNSW index graph and a hot working set are ever resident. So RAM is a predictable formula, not a mystery:

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

(The encoder is always shared — stateless, ~31 MB, loaded once. Only the pool/index differs.)

> **TL;DR.** **Shared pool → RAM is not your limit** — spin up as many sessions as your CPU allows. **Separate pools → one index each**, so plan on ~3 (20 GB) to ~13 (5 GB) sessions on 8 GB, roughly double at 16 GB. A bigger pool always buys **reach**, never more sessions. Need more headroom? Shrink the pool, or run `--index tiered` to keep only warm graph nodes resident.

## Honest about the word "unlimited"

"Unlimited" means **reach, not attention.** Your model keeps its native window — we make it *reach* a billion-token pool in slices, via fast retrieval. The whole thing rides on retrieval **hit rate**; when it's high (and the loader is built to keep it high), the pool feels like one seamless context.

## Status & community

Published **as-is** under Apache-2.0. Bug reports are welcome via [Issues](https://github.com/DBarr3/Unlimited-Context/issues); pull requests aren't actively reviewed. If it's useful, a GitHub star helps others find it. Runnable examples live in [`examples/`](examples/) — start with [`quickstart.py`](examples/quickstart.py), then [`coding_agent.py`](examples/coding_agent.py).

## License

**Apache-2.0.** Use it, fork it, ship it in your product.

---

<div align="center">

Built by **Brandon Barrante** · [Aether](https://aethersystems.net)

<img width="880" alt="Unlimited Context" src="https://github.com/user-attachments/assets/4b7eef9a-8b1c-4dc7-b926-771ce53ed04d" />

*Unbounded reach for the model you already run.*

</div>
