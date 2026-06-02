# Local models — the wrapper (Ollama, llama.cpp, HF)

> Unlimited Context wraps **the model you already run**. This page is the whole surface: how to point
> it at a local model, what's supported, and how to fix the three things that ever go wrong.

## TL;DR

```bash
pip install aether-context          # core, numpy-only
ollama serve                        # start Ollama (if you use it)
ollama pull qwen2.5                 # grab a small model
```
```python
from aether_context import Session

s = Session(model="ollama/qwen2.5", pool_gb=5)
print(s.run("Summarize this repo, then refactor the auth module.").text)
```

No Ollama? It still runs — pass `model="mock"` and the engine works end-to-end with a built-in
deterministic model (great for trying the API, tests, and CI offline).

## How to name a model

One string. The part before the slash/colon is the backend; the rest is the model.

| You write | Backend | Notes |
|---|---|---|
| `"ollama/qwen2.5"` | Ollama | recommended; talks to `localhost:11434` |
| `"qwen2.5"` | Ollama | bare name → assumed Ollama |
| `"ollama/llama3.1:8b"` | Ollama | tags work too |
| `"llamacpp:/models/qwen2.5-7b.gguf"` | llama.cpp | needs `pip install "aether-context[llamacpp]"` |
| `"hf/Qwen/Qwen2.5-7B-Instruct"` | HF transformers | needs `pip install "aether-context[hf]"` |
| `"mock"` | built-in | deterministic, offline, zero deps |

You can also pass an object that satisfies the `LocalLLM` protocol (see below) for any backend we
don't ship.

## The backends

### Ollama (the easy path) — no extra dependency

The primary adapter speaks HTTP to the Ollama daemon using only the Python **standard library**
(`urllib`). So `pip install aether-context` is all you need on the Python side; Ollama itself is the
one external program.

- **Context window** is auto-detected from `ollama show` metadata; falls back to 8192 if unknown.
- **Streaming** is on — tokens arrive as the model generates, which is what lets the pager fetch the
  next slices *while* the model is still talking.
- **Auto-pull** (opt-in): `Session(model="ollama/qwen2.5", pull=True)` pulls the model if it's
  missing instead of erroring.

```python
s = Session(model="ollama/llama3.1:8b", pool_gb=10)        # bigger pool = more reach
for chunk in s.stream("Walk the codebase and write a design doc."):
    print(chunk, end="", flush=True)
```

### llama.cpp — bring a `.gguf`

```bash
pip install "aether-context[llamacpp]"
```
```python
s = Session(model="llamacpp:/models/qwen2.5-7b-instruct-q4.gguf", pool_gb=5,
            model_options={"n_ctx": 8192, "n_gpu_layers": 35})
```

`model_options` pass straight through to `llama_cpp.Llama(...)`. Token counts use llama.cpp's real
tokenizer (more accurate budgeting than the chars/4 estimate).

### Hugging Face transformers

```bash
pip install "aether-context[hf]"     # pulls transformers + torch
```
```python
s = Session(model="hf/Qwen/Qwen2.5-7B-Instruct", pool_gb=5,
            model_options={"device_map": "auto", "torch_dtype": "auto"})
```

Uses a `TextIteratorStreamer` so generation still streams; `context_window` comes from the model
config; token counts use the model's own tokenizer.

### Mock (offline / tests / CI)

```python
s = Session(model="mock", pool_gb=5)        # or Session(model="mock", context_window=2048)
```

Deterministic output derived from the prompt. Set a small `context_window` to *force* overflow and
watch the engine page context back in — that's exactly what `bench/drift_vs_window.py` does to prove
the mechanism without a GPU.

## The contract (write your own backend)

Any object with this shape works as `model=`:

```python
class LocalLLM(Protocol):
    name: str
    context_window: int                       # tokens
    def generate(self, prompt: str, *, system: str | None = None,
                 stop: list[str] | None = None, max_tokens: int | None = None
                 ) -> Iterator[str]: ...       # yield text chunks (streaming)
    def count_tokens(self, text: str) -> int: ...
```

That's it. `generate` **streams** (yields chunks) so the pager overlaps retrieval with generation.
If your backend can't stream, yield once with the full text — it still works, you just lose the
free concurrency.

```python
from aether_context import Session

class MyBackend:
    name = "my-llm"
    context_window = 8192
    def generate(self, prompt, *, system=None, stop=None, max_tokens=None):
        yield my_model.complete(prompt, system=system)      # one chunk is fine
    def count_tokens(self, text):
        return len(my_model.tokenize(text))

s = Session(model=MyBackend(), pool_gb=5)
```

## Picking pool size (reach)

Pool size is **reach**, not window. `reach ≈ pool_gb × 233M tokens`.

| `pool_gb` | Reach | Index RAM | Good for |
|---|---|---|---|
| 5 (floor) | ~1.16B | ~145 MB | a big project |
| 10 | ~2.33B | ~291 MB | a large monorepo + docs |
| 15 | ~3.49B | ~436 MB | multiple repos / long runs |
| 20 | ~4.65B | ~582 MB | massive corpus / power user |

```bash
aether-context init          # interactive slider, writes ~/.aether-context/config.json
aether-context --pool 20     # resize anytime (re-index, non-destructive)
```

Running many sessions? `--pool-mode shared` pays the index **once** (RAM barely moves per session);
`--pool-mode separate` (default) isolates each session but pays one index each. Bigger pool always
buys reach, never more concurrent sessions (those are RAM-bound).

## When it doesn't work (the only three things)

`aether-context doctor` checks all of these and prints the exact fix:

| Symptom | Cause | Fix |
|---|---|---|
| `OllamaNotRunning` | daemon not up | `ollama serve` |
| `ModelNotPulled: qwen2.5` | model not downloaded | `ollama pull qwen2.5` (or `pull=True`) |
| slow / low hit rate | pool too small or cold | bigger `pool_gb`, or let the session warm up |
| RAM warning at init | index won't fit | smaller pool, or `--index tiered` |

Everything fails **soft**: if retrieval misses or a backend stalls, the run logs it and continues on
the model's native window — you never lose the whole build to a pager hiccup.

## Honest note

"Unlimited" means **reach, not attention.** Your model keeps its native window; we make it *reach* a
billion-token local pool in slices via fast retrieval. Quality rides on retrieval hit rate — high hit
rate feels like one seamless context. Benchmark it yourself: `python bench/drift_vs_window.py`.
