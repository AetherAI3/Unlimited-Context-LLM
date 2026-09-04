<div align="center">

# ⚡ Unlimited Context

**Virtual memory for an LLM's attention.** Keep a billion-token pool on your own disk; the model
reaches it in slices, one small window at a time. Local-first, offline, free.

<img alt="npx aether-context: guided setup, a clean doctor check, and pool status - real terminal output" width="716" src="https://raw.githubusercontent.com/AetherAI3/Unlimited-Context-LLM/main/assets/demo.gif">

[![PyPI](https://img.shields.io/pypi/v/aether-context?style=flat-square&logo=pypi&logoColor=white&color=06b6d4)](https://pypi.org/project/aether-context/)
[![npm](https://img.shields.io/npm/v/aether-context?style=flat-square&logo=npm&logoColor=white&color=cb3837)](https://www.npmjs.com/package/aether-context)
[![License](https://img.shields.io/badge/License-Apache_2.0-06b6d4?style=flat-square)](https://github.com/AetherAI3/Unlimited-Context-LLM/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-14b8a6?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![Built by Aether](https://img.shields.io/badge/Built_by-Aether-7c3aed?style=flat-square)](https://aethersystems.net)
[![Stars](https://img.shields.io/github/stars/AetherAI3/Unlimited-Context-LLM?style=flat-square&logo=github&color=eab308)](https://github.com/AetherAI3/Unlimited-Context-LLM/stargazers)

[Install](https://github.com/AetherAI3/Unlimited-Context-LLM#install) ·
[How it works](https://github.com/AetherAI3/Unlimited-Context-LLM#how-it-works) ·
[The proof](https://github.com/AetherAI3/Unlimited-Context-LLM#the-proof) ·
[Sizing](https://github.com/AetherAI3/Unlimited-Context-LLM#sizing-disk-reach-and-ram) ·
[Safety](https://github.com/AetherAI3/Unlimited-Context-LLM/blob/main/SAFETY.md)

</div>

---

> **Your context window didn't get bigger. Its *reach* did.**
> The model keeps its small window. The engine keeps a vast store on your disk and pulls the
> *right slice* back in while the model reasons. A small local model stays coherent across runs
> that would blow past any context window.

## Install

```bash
pip install aether-context
aether-context setup
```

`setup` sizes the pool, checks for a local model, and verifies the engine end to end. It works
with no daemon, no network and no model pulled — the check runs against the built-in mock model.

```python
from aether_context import Session

s = Session(model="ollama/qwen2.5", pool_gb=5)
s.run("Build me a full-stack weightlifting tracker app.")
# runs long. stays coherent. walk away.
```

Prefer npm? Same software, same release. The npm package is a launcher that installs the Python
engine into a private virtualenv for you (it needs Python 3.10+ on your PATH):

```bash
npx aether-context setup
```

<details>
<summary>Other install routes</summary>

```bash
# Straight from source, always the latest main:
pip install git+https://github.com/AetherAI3/Unlimited-Context-LLM.git

# Isolated, if you only want the CLI:
pipx install aether-context
```

The distribution name is **`aether-context`** — `pip install unlimited-context` is not this
package.
</details>

That's the whole thing. One small model, one command, a billion tokens of reach behind it.

## The problem

Long agentic runs all die the same way. The model fills its window, starts **compressing** its own
history, silently drops the one detail that mattered three steps ago — and drifts. You've seen it:
the runaway PR, the agent that rewrites a function it already wrote, the build that falls apart at
hour two. Bigger windows just delay it, and a crammed 1M-token window **rots in the middle** anyway.

The fix isn't a bigger window. It's to stop throwing the overflow away. Instead of summarizing what
spills over, Unlimited Context **encodes** it to a local pool on your disk and **recovers** the
right slice exactly when it's needed. Nothing load-bearing is silently lost.

<p align="center"><strong>Compress &amp; forget ✗ &nbsp;→&nbsp; Encode &amp; recover ✓</strong></p>

## How it works

It's **virtual memory, for attention.** Map it to an OS and it clicks:

| OS | Unlimited Context |
|---|---|
| RAM | the **resident window** the model sees now (small, fast) |
| Disk | the **context pool** — your encoded memory (~5 GB ≈ ~1B tokens) |
| Pager | the **slice loader** — prefetches the next slice from what the model is reasoning about *right now* |
| Page-replacement | the **retention policy** — useful slices *stay*, stale ones *fade*, anything relevant again comes back |

The pager runs concurrently with generation, so most of the fetch hides behind the model's own
thinking. Full explainer: [`docs/how-it-works.md`](https://github.com/AetherAI3/Unlimited-Context-LLM/blob/main/docs/how-it-works.md).

## What you get

- 🧠 **Unbounded reach** — ~1B tokens of encoded context in ~5 GB on disk; the model reaches it in slices.
- 🧩 **MPO context chain** — recall pulls the whole connected thread, not isolated nearest-neighbors.
- 🪟 **Curated beats crammed** — a small, relevant resident window outperforms a stuffed one (no lost-in-the-middle) — and costs less.
- 🔒 **Local-first** — your context never leaves your machine. Free storage, full privacy, works offline.
- 🤖 **Any model** — Llama, Qwen, Mistral, Phi — via Ollama, llama.cpp, or Hugging Face, or your own API-backed model.
- 📉 **Coherence you can measure** — the head-to-head is committed: same model, engine on vs off.

## The proof

Not a synthetic micro-benchmark — a **real, paid, end-to-end run.** A reasoning model
(`deepseek-v4-pro`, via OpenRouter) driven through a **40-turn agent session that overflows its
window** (2,000-token window, 60 real `microsoft/vscode` issues), measured **engine off vs on** —
one live run, **$0.19**, 2026-06-14.

- **The model stops forgetting.** Recall of early facts after they fall out of the window:
  **0.15 → 1.00.** The baseline drifts and forgets; the engine holds every early fact — zero drift.
- **Failure turns into success on the real work.** Tasks completed correctly: **3 / 20 → 20 / 20.**
- **Cheaper, not just better.** **−24%** total cost, **−54%** in the back half — the engine sends a
  compact recalled slice instead of dragging the whole transcript into every call.

| Metric | Off (baseline) | On (engine) | Change |
|---|:---:|:---:|:---:|
| **Recall coherence** (early facts still correct) | 0.15 | **1.00** | **6.7×** |
| **Work outcome** (tasks done right) | 3 / 20 | **20 / 20** | **3 → 20** |
| **Cost — full session** | $0.0711 | **$0.0542** | **−24%** |
| **Cost — back half (recall phase)** | $0.00117/turn | **$0.00053/turn** | **−54%** |

<p align="center">
  <img alt="Cumulative cost and recall coherence vs turn — engine off vs on" width="780"
       src="https://raw.githubusercontent.com/AetherAI3/Unlimited-Context-LLM/main/docs/benchmarks/artifacts/2026-06-14-deepseek-v4-pro/api_eval_plot.png">
</p>

**Committed data:** [full write-up](https://github.com/AetherAI3/Unlimited-Context-LLM/blob/main/docs/benchmarks/2026-06-14-deepseek-v4-pro-session-eval.md) ·
[raw artifacts](https://github.com/AetherAI3/Unlimited-Context-LLM/tree/main/docs/benchmarks/artifacts/2026-06-14-deepseek-v4-pro)
(`api_eval_results.json`, `api_eval_series.csv`, `api_eval_plot.png`, `RESULTS.md`) · reproduce with
`python -m bench.api_eval --model deepseek/deepseek-v4-pro --repo microsoft/vscode --arms off,on,on_chain --plot`

<sub>**Scope, honestly:** this run used a hosted reasoning model, not a local one — the mechanism is
backend-agnostic, but the headline number is not a local number. It measures the **engine**
(retrieve-on-overflow memory), not the MPO chain: on this single-fact recall task the chain
**ties** plain recall (both 1.00), and its multi-slice edge is **synthetic-only so far**
(`bench/chain_recall.py`: connected-context recall 0.15 → 0.78), with the live `thread` run
**pending**, not yet claimed. The 2,000-token window is deliberately tiny to force overflow, so a
realistic window shows a smaller (still real) gain. N = 20 recall turns, single run.</sub>

## Sizing: disk, reach and RAM

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

One table for the whole trade-off — disk in, reach out, RAM cost, and how many isolated sessions
fit on a small machine:

| Pool | Slices | Encoded reach | Index RAM | Sessions on 8 GB (separate pools) |
|:----:|:------:|:-------------:|:---------:|:---------------------------------:|
| **5 GB** *(floor)* | 2.27M | **~1.16B tokens** | ~146 MB | ~13 |
| 10 GB | 4.55M | **~2.33B tokens** | ~291 MB | ~7 |
| 15 GB | 6.82M | **~3.49B tokens** | ~436 MB | ~4 |
| 20 GB | 9.09M | **~4.65B tokens** | ~582 MB | ~3 |

Roughly double the session counts on a 16 GB machine. Where the numbers come from: ~2.2 KB per
slice (a 256-dim vector + compressed text + metadata) ÷ 512 tokens per slice → **~455K slices/GB →
~233M tokens of reach per GB**. So `reach ≈ pool_GB × 233M`. At the 5 GB floor that's about
**9,000×** a 128K window. Bump the pool anytime with `aether-context --pool 20`.

**RAM is a formula, not a mystery.** Vectors live on disk (mmap'd) — only the small index graph and
a hot working set are ever resident:

```
RAM  ≈  ~180 MB   base (engine + shared static encoder)
      +  ~29 MB   per GB of pool   (resident index)
      +  ~30 MB   per active session
```

**Sharing the pool is the biggest RAM lever.** `--pool-mode separate` *(default)* gives every
session its own pool and index — fully isolated and private, but you pay one index per session, so
RAM scales with `N × pool` (that's the last column above). `--pool-mode shared` pays for the index
**once**; each extra session adds only ~30 MB, so 50–70+ sessions fit and CPU becomes the limit
instead of memory. The trade-off is that sessions can see each other's context. Use shared for
related work on one project, separate for unrelated tasks.

**How much building is that?** A ~128K window fills after well under an hour of active agent work,
then starts compacting and forgetting. Assuming a busy coding agent encodes ~300K–1M keep-worthy
tokens an hour, a 5 GB pool covers on the order of **1,200–3,900 hours** before it even fills —
weeks of nonstop building. Because the retention policy fades stale slices, the pool never
hard-stops anyway; it just keeps what's relevant.

<div align="center">
  <img width="880" alt="Coding time per pool size" src="https://raw.githubusercontent.com/AetherAI3/Unlimited-Context-LLM/main/assets/coding-time-per-pool.png">
</div>

> **Honest:** that's encoded **reach**, retrieved in slices — not a bigger attention window, and it
> rides on retrieval hit rate. A bigger pool buys more reachable codebase or corpus *per session*,
> never more concurrent sessions (those are RAM-bound). `--index tiered` is reserved for a future
> paged-graph index and currently runs the flat index — it does not yet reduce resident RAM.

## Commands

| Command | What it's for |
|---|---|
| `aether-context setup` | **Start here.** Guided first run: size the pool, check your model, verify the engine. |
| `aether-context init` | Pick your pool size — the on-disk storage slider — on first run. |
| `aether-context run "<task>"` | One-shot a task with full reach, then print the result. |
| `aether-context run "<task>" --no-mpo-chain` | Same, with the MPO context chain disabled (plain cosine). |
| `aether-context chat` | Open an interactive session; type `/status` anytime, `/clear` to reset. |
| `aether-context status` | See pool size, slices used, reach, and hit rate at a glance. |
| `aether-context doctor` | Check Ollama, your model, disk, and RAM before a long run. |
| `aether-context --pool 20` | Resize the pool anytime (non-destructive re-index). |

> **Tip:** run `aether-context doctor` first — it catches the three things that ever go wrong
> (Ollama down, model not pulled, not enough disk) and prints the exact fix.

## MPO: the context chain

Plain semantic search returns isolated nearest-neighbors — the single closest slices, ripped out of
the thread they belonged to. Recall a fact and you often miss the three slices around it that made
it make sense.

The **MPO context chain** links the session's slices into one connected structure, so when cosine
pulls an entry slice, the chain pulls in the slices most coupled to it — widening the working set
with the *connected thread*, not stray hits. Cosine is still the retrieval mechanism; the chain
assists it.

The chain is Aether-tuned, deterministic and fully local — no training, no network. It is purely
**additive**: it only ever *adds* connected context, never blocks or replaces a hit, and on any
hiccup it falls back cleanly to plain cosine. In a planted-thread benchmark it lifts
connected-context recall from **0.15 (cosine alone) to 0.78** — over 5× more of the right thread in
the window. That result is synthetic so far; see the caveat under [The proof](https://github.com/AetherAI3/Unlimited-Context-LLM#the-proof).

On by default:

```python
Session(model="ollama/qwen2.5", pool_gb=10)                  # chain on by default
Session(model="ollama/qwen2.5", pool_gb=10, mpo_chain=False) # plain cosine
```
```bash
aether-context run "..." --no-mpo-chain                      # disable for one run
```

## The `aether` coding terminal

**`aether`** is an open-source agentic coding terminal that runs on this engine. Turns run on your
local [Ollama](https://ollama.com) by default — no account, no network; sign in and they switch to
the Aether cloud API. It ships as its own package:

```bash
pip install aether-agent      # or: npm install -g aether-agents
```

`aether` opens the REPL, `aether "<prompt>"` is a one-shot turn, and `aether code "<task>"` is an
autonomous coding run on the Unlimited Context brain (test-gated, git-checkpointed). Full command
list, slash commands and backend settings live at
[AetherAI3/aether-agent](https://github.com/AetherAI3/aether-agent).

<sub>The `aether_agent/` directory in *this* repo is the Python-native twin — same commands, same
backend, same tools — kept here for development and deliberately **not** published from this
package: PyPI's `aether-agent` already owns that import path and the `aether` command, so shipping
a second copy would silently overwrite it wherever both are installed. From a clone with Ollama up,
`python -m aether_agent.smoke` runs the SSRF guard, a real local turn, a web search and fetch, and
the cloud path when signed in.</sub>

## Safety and use policy

Giving a model durable memory is powerful, and the failure modes are real: runaway agents,
grounding drift, an agent's own notes hardening into its rules. What those are and what we do about
them is written up in
**[Ethical & Safety Measures](https://github.com/AetherAI3/Unlimited-Context-LLM/blob/main/SAFETY.md)**.
Use of the project is governed by the
**[Acceptable Use Policy](https://github.com/AetherAI3/Unlimited-Context-LLM/blob/main/USE_POLICY.md)**
— by using the project you agree to and are bound by its terms.

## Honest about the word "unlimited"

"Unlimited" means **reach, not attention.** Your model keeps its native window; the engine makes it
*reach* a billion-token pool in slices, via fast retrieval. The whole thing rides on retrieval hit
rate — when that's high, and the loader is built to keep it high, the pool feels like one seamless
context. When it isn't, you get a miss, and a miss looks like forgetting. The measured evidence for
all of this is in [The proof](https://github.com/AetherAI3/Unlimited-Context-LLM#the-proof), caveats included.

## Contributing

**PRs and issues are welcome** — start with
[CONTRIBUTING.md](https://github.com/AetherAI3/Unlimited-Context-LLM/blob/main/CONTRIBUTING.md).
There are open issues tagged
[good first issue](https://github.com/AetherAI3/Unlimited-Context-LLM/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
and [help wanted](https://github.com/AetherAI3/Unlimited-Context-LLM/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
right now, including an LM Studio backend, a Windows quickstart, and a recall-quality benchmark at
100K / 1M / 10M tokens.

Runnable examples live in
[`examples/`](https://github.com/AetherAI3/Unlimited-Context-LLM/tree/main/examples) — start with
[`quickstart.py`](https://github.com/AetherAI3/Unlimited-Context-LLM/blob/main/examples/quickstart.py),
then [`coding_agent.py`](https://github.com/AetherAI3/Unlimited-Context-LLM/blob/main/examples/coding_agent.py).

If the engine earns its place in your setup, **a star helps other people find it.**

## Citation

If Unlimited Context helps your work, please cite it. Built and maintained by **Aether AI**.

```bibtex
@software{unlimited_context_2026,
  title        = {Unlimited Context (aether-context): virtual memory for LLM attention},
  author       = {Barrante, Brandon},
  organization = {Aether AI},
  year         = {2026},
  url          = {https://github.com/AetherAI3/Unlimited-Context-LLM},
  license      = {Apache-2.0}
}
```

GitHub's "Cite this repository" button reads
[`CITATION.cff`](https://github.com/AetherAI3/Unlimited-Context-LLM/blob/main/CITATION.cff) directly.

## License

**Apache-2.0.** Use it, fork it, ship it in your product.

---

<div align="center">

Built by **Aether AI** · [aethersystems.net](https://aethersystems.net)

<img width="880" alt="Aether" src="https://raw.githubusercontent.com/AetherAI3/Unlimited-Context-LLM/main/assets/aether-footer.jpg">

*Unbounded reach for the model you already run.*

</div>
