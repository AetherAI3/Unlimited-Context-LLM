# How it works — virtual memory, for attention

> The one-line model: **Unlimited Context is virtual memory for an LLM's attention.** A small, fast
> *resident window* over a vast *local pool*, paged in and out **while the model reasons**. Your
> context window didn't get bigger — its *reach* became unbounded.

If you've ever written an operating system, you already understand this engine. We just took the
oldest trick in systems programming — *don't keep everything in fast memory; keep the right thing,
and page the rest* — and pointed it at the one resource a language model is starved for: attention.

## The thesis: encode & recover, not compress & forget

Every long agentic run dies the same way. The model fills its window, starts **compressing** its own
history to make room, and silently drops the one detail that mattered three steps ago. Then it
drifts: it rewrites a function it already wrote, contradicts a decision it made an hour ago, and the
build falls apart. A bigger window only delays the failure — and a crammed million-token window
*rots in the middle* anyway (the lost-in-the-middle effect).

The fix is not a bigger window. The fix is to stop *throwing the overflow away*.

| | What happens to overflow | What you lose |
|---|---|---|
| **Compress & forget** ✗ | summarized in place, the detail is gone | the one load-bearing fact, silently |
| **Encode & recover** ✓ | encoded to a vector + filed to a local pool | *nothing* — it's recoverable on demand |

When a turn's worth of context spills past the working window, Unlimited Context does **not**
summarize it into a lossy blob. It **encodes** that text into a 256-dimensional vector, writes the
vector plus the original text to a pool on your disk, and remembers it. Later, when the model starts
reasoning about something that *touches* that filed content, the pager retrieves the exact slice and
pages it back into the working window. Nothing load-bearing is silently lost. It's filed, and it's
recoverable. **That is the whole pitch — and it lives in the code, not the marketing.**

## The OS mapping (this is the part that clicks)

Map every module to a piece of an operating system and the design explains itself:

| OS concept | The module | What it does here |
|---|---|---|
| **RAM** — small, fast, resident | the **working window** | the context the model actually sees this turn |
| **Disk** — vast, cheap | `context_pool` | mmap'd 256-dim vector index + slice payloads (~5 GB, ~1B tokens) |
| **Pager** | `slice_loader` | prefetches the right slice from what the model is reasoning about *now* |
| **Page-replacement** | `witness` | salient slices **harden**, stale ones **fade**, relevant-again **re-hardens** |
| **Encode-on-spill** | `encoder` | static numpy embedder: tokenize → lookup → mean-pool → 256-dim unit vector |
| **Process lifecycle** | `session` | open → stream + encode + fade → paged reason → close |
| **Device driver** | `local_llm` | the wrapper around Ollama / llama.cpp / HF — the part you actually touch |

Each section below walks one row.

### RAM → the working window

The working window is the model's native context — 8K, 32K, 128K tokens, whatever your model ships
with. We do not pretend it's bigger. It is *RAM*: small, fast, and the only thing the model can
directly "see." Everything else is paged.

The engine governs this window with three fractions (mirrored from AetherCloud, in `config.py`):

- **`trigger_fraction` (0.75)** — once the window is 75% full, overflow encoding kicks in.
- **`target_fraction` (0.50)** — a paged compaction drains the window back down to ~50% occupancy,
  leaving headroom so we're not re-triggering every token.
- **`verbatim_fraction` (0.30)** — the most recent ~30% of the window is kept *verbatim*, never
  encoded away. The immediate present is sacred; only the cooling tail spills to the pool.

### Disk → the context pool

The pool is *disk*: vast, cheap, and not resident. Vectors live in an **mmap'd file** on your disk,
so the OS pages them in on demand and only the small in-RAM index graph plus a hot working set are
ever truly resident. The pool is **session-namespaced** — far-apart sessions stay in separate
regions so they don't pollute each other's retrieval.

Each entry is a `Slice`: `(id, session, vector(256), text, tokens, meta, score)`. The vector is the
retrieval key; the text is the recoverable payload. A budget governor enforces a hard GB ceiling by
evicting the lowest-scoring slices (see *page-replacement*, below). **The pool never grows past its
budget** — eviction runs after each add, so "never exceeds budget" is literally true.

### Pager → the slice loader

The pager is the part that makes reach feel free. As the model generates, `slice_loader` looks at
what the model is reasoning about *right now*, predicts which slices it will need next, and
**prefetches** them into a warm, LRU-budgeted cache. A warm hit is O(1); a cold miss falls back to a
pool search. It tracks its own **hit rate** — the single number that determines whether the pool
*feels* like one seamless context (high hit rate) or like a model that keeps forgetting (low).

The pager core is deliberately **single-threaded and pure** — concurrency is the caller's job.
`session.py` runs `prefetch` on a background thread **while the model is generating**. Because the
generate call (the Ollama HTTP request, the llama.cpp call, the subprocess) releases the GIL, that
prefetch thread genuinely overlaps with generation. The retrieval happens *behind* the model's own
thinking, so reach is effectively free in wall-clock terms. There's also an ε re-probe: as the
loader goes idle it occasionally re-probes the pool, so a slice that became relevant again gets
pulled back even if the prediction missed it.

### Page-replacement → the witness

The witness is the page-replacement policy — the OS clock hand, generalized into a **fidelity
field**. Every slice carries a retention `score`:

- **harden** — on access or when a slice proves relevant, its score rises (it earns its place in
  memory);
- **fade** — with elapsed time, scores **decay** monotonically (cold slices sink);
- **re-harden** — a faded slice that becomes relevant *again* is lifted back up.

Under budget pressure the governor evicts **lowest-score-first**. So the slices that keep mattering
stay reachable, and the ones that stopped mattering quietly make room. In coding-context terms the
scoring weighs *surprise* (content density), *impact* (query relevance), and *uniqueness*
(`1 / (1 + similar)`) — pure scoring over access events, with **no** atlas-cell, ground-truth, or
trading coupling.

### Encode-on-spill → the encoder

When a slice spills, the `encoder` turns its text into a 256-dim unit vector. It's a static,
Model2Vec-style numpy embedder: a regex tokenizer, a `(vocab, 256)` float32 lookup table, a mean-pool
of the token rows, and an L2-normalize. It's **stateless, shared, and tiny** (~31 MB), so it loads
once and runs on every session at well over a million tokens/second/core — no GPU, no network, no
model download.

Strings that share tokens share rows in the table, so the mean-pool gives you real lexical cosine
structure: similar strings land closer together than dissimilar ones (we pin `ENCODER_VERSION` and
test that the similar-pair margin beats the dissimilar-pair margin). The encoder produces the
**256-dim retrieval embedding only** — it is *not* an attention mechanism and it is *not* the closed
8-dim capability coordinate. Retrieval embedding in, slices out. That's the contract.

### Process lifecycle → the session

`session.py` is the process: it owns one run from start to finish.

1. **open** — a fresh working window.
2. **stream + encode + fade** — as the model emits tokens and as new input arrives, the cooling tail
   is encoded and spilled to the pool, and the witness fades the cold slices.
3. **paged reason** — the pager keeps the right slices resident, prefetching on a side thread so the
   model never stalls waiting for memory.
4. **close** — extract the durable take-aways, emit abstracted harvest candidates `(text, vector,
   tags)`, and flush.

Crucially, **every step is fail-soft**. The pager and retrieval are an *optimization*, never a
correctness dependency. A retrieval miss, an encoder hiccup, or a backend stall degrades to the
model's native window — it logs and continues. You never lose a two-hour build to a pager glitch.

### Device driver → the local LLM wrapper

`local_llm.py` is the device driver: one `LocalLLM` protocol that Ollama, llama.cpp, HF, and the
built-in `MockLLM` all satisfy. The engine never special-cases a backend; it just calls `generate`
(which streams) and `count_tokens`. Full guide: [`local-models.md`](local-models.md).

## The pool + RAM math (where the numbers come from)

The headline — *"~1B tokens of reach in ~5 GB"* — is derived, not vibes.

**Reach.** A slice is ~2.2 KB on disk (a 256-dim vector + compressed text + metadata) and holds 512
tokens. So:

```
2.2 KB / slice, 512 tokens / slice
  →  ~455K slices per GB
  →  ~233M tokens of reach per GB
  →  reach ≈ pool_gb × 233M tokens
```

| Pool | Slices | Encoded reach | Index RAM (resident) |
|------|--------|---------------|----------------------|
| **5 GB** *(floor)* | 2.27M | **~1.16B tokens** | ~146 MB |
| 10 GB | 4.55M | **~2.33B tokens** | ~291 MB |
| 15 GB | 6.82M | **~3.49B tokens** | ~436 MB |
| 20 GB | 9.09M | **~4.65B tokens** | ~582 MB |

5 GB is the floor — below it the reach is too small to matter and the witness budget governor has no
headroom to avoid thrashing. (`config.py` enforces this: `pool_gb < 5` is rejected with the reason.)

**RAM.** Vectors live on disk (mmap'd); only the index graph and a hot working set are ever resident.
So RAM is a predictable formula, not a mystery:

```
RAM  ≈  ~180 MB   base       (engine + the one shared static encoder)
      +  ~29 MB   per GB of pool   (the resident index)
      +  ~30 MB   per active session
```

The encoder is **always shared** — stateless, ~31 MB, loaded once regardless of how many sessions
run. Only the pool/index differs between modes:

- **`--pool-mode shared`** — one pool, one index, all sessions reach the same memory. The index is
  paid **once**; each extra session adds only ~30 MB, so RAM barely moves as you add sessions
  (dozens fit — 50–70+, you're CPU-bound, not RAM-bound). Trade-off: no isolation between sessions.
- **`--pool-mode separate`** *(default)* — each session gets its own pool + index, fully isolated and
  private. You pay one index per session, so RAM scales with `N × pool`: roughly **~3 sessions
  (20 GB) to ~13 (5 GB) on an 8 GB machine**, about double that at 16 GB.

A bigger pool always buys more **reach** per session — never more concurrent sessions. Those are
RAM-bound either way.

## Honest about "unlimited": reach, not attention

We will not sell you a fairy tale, because in OSS a fake "infinite window" is fatal to credibility
and a real mechanism is rewarded.

**"Unlimited" means reach, not attention.** Your model keeps its native attention window — 8K stays
8K. What becomes unbounded is what that window can *reach*: a billion-token local pool, retrieved in
slices, paged in exactly when the reasoning calls for it. The whole thing rides on retrieval **hit
rate**. When it's high — and the loader is built to keep it high — the pool feels like one seamless,
enormous context. When it's low, you fall back to the native window and lose no correctness, just
the reach. We hand you the real mechanism precisely because the real mechanism is the part that holds
up over a long run.

Don't take the claim on faith — measure it:

```bash
python bench/drift_vs_window.py --model ollama/qwen2.5 --task examples/long_build.md
```

Same model, engine **on vs off**, one long build. It reports cross-stage contradictions (drift),
per-stage correctness, retrieval hit rate, and whether it finished unattended. **The delta is the
pitch** — and it generates on your own hardware.

## Where to go next

- [`local-models.md`](local-models.md) — point the engine at your model (Ollama, llama.cpp, HF, mock).
- [`aethercloud.md`](aethercloud.md) — the same engine, upgraded: a *verified* brain behind it.
