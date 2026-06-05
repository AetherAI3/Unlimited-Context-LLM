"""
Per-model profiles — the one piece of real per-model engineering. The harness is
model-agnostic, but each model wants its own sampling and has its own tool-call
quirks. A profile maps a model tag to (tier, sampling, tool-call style) so the
adapter applies the right knobs and the roster can show tiers.

This is the #1-risk surface from the Gemma brainstorm: Gemma 4 tool-calling is
real (~86%) but wants Google's sampling (temp 1.0 / top_k 64 / top_p 0.95) and
thinks-before-the-call; Qwen coding wants low-temp determinism. Same cycle, same
harness — only these knobs differ. Adding a model = adding a profile (model-agnostic).
"""

from __future__ import annotations

from dataclasses import dataclass

# Tiers (from the brainstorm roster). light = runs where a 30B won't.
TIER_LIGHT = "light"
TIER_STRONG = "strong-local"
TIER_CLOUD = "cloud"

# Recommended CONCRETE Ollama tags (real pulls, sized for a typical small box —
# ~8GB VRAM / 16GB RAM). The universal small default is qwen2.5-coder:7b: it fits,
# it has the best small-model tool-calling (criterion #1), and it's Apache-2.0.
# GEMMA_LIGHT is the working Gemma option (official tag, fits 8GB). 'gemma4' is
# NOT a real Ollama model — never pull it.
LIGHT_DEFAULT = "qwen2.5-coder:7b"   # universal small — runs ~anywhere, strong tools
GEMMA_LIGHT = "gemma3:4b"            # the Gemma option (fits 8GB; weaker tool-calling)
GEMMA_E4B = "gemma3n:e4b"            # the efficient 'e4b' (the real one; ~7GB)
DEPTH_DEFAULT = "qwen3-coder:30b"    # depth build — needs ~24GB RAM/VRAM


@dataclass(frozen=True)
class ModelProfile:
    """How to drive one model. `match` is the lowercased substring that selects it."""

    match: str
    tier: str
    sampling: dict  # temperature / top_p / top_k passed to the inference endpoint
    notes: str = ""


# Order matters: first substring match wins. Specific before generic.
PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile(
        "gemma",
        TIER_LIGHT,
        {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "Gemma family (Google sampling: temp 1.0 / top_k 64 / top_p 0.95). REAL Ollama "
        "tags: gemma3:4b (~3.3GB, fits 8GB GPU) · gemma3n:e4b (~7GB, the efficient 'e4b') · "
        "gemma3:12b (stronger). NOTE: 'gemma4' is NOT an Ollama model — do not use it. "
        "Gemma tool-calling is weaker than Qwen; for a small box prefer qwen2.5-coder.",
    ),
    ModelProfile(
        "qwen",
        TIER_STRONG,
        {"temperature": 0.2, "top_p": 0.9, "top_k": 40},
        "Qwen-Coder (qwen2.5-coder:7b = the universal small default, ~4.7GB, fits 8GB GPU, "
        "best small-model tool-calling, Apache-2.0; qwen3-coder:30b = depth, needs ~24GB). "
        "Low-temp deterministic coding; most reliable tool-calling.",
    ),
    ModelProfile(
        "devstral",
        TIER_STRONG,
        {"temperature": 0.2, "top_p": 0.9, "top_k": 40},
        "Devstral Small 24B — drop-in agentic alternative.",
    ),
    ModelProfile(
        "deepseek",
        TIER_CLOUD,
        {"temperature": 0.3, "top_p": 0.95, "top_k": 40},
        "DeepSeek V4 Flash — cloud tier (via the CLI API models), not a local Ollama pull.",
    ),
)

# Safe default for an unknown tag: deterministic, like Qwen coding.
_DEFAULT = ModelProfile(
    "", TIER_STRONG, {"temperature": 0.2, "top_p": 0.9, "top_k": 40}, "unknown model — safe deterministic default"
)


def for_model(model: str) -> ModelProfile:
    """The profile for a model tag (first substring match; safe default otherwise)."""
    low = (model or "").lower()
    for p in PROFILES:
        if p.match in low:
            return p
    return _DEFAULT


def validate_tool_calls(calls: list, allowed: frozenset[str]) -> tuple[list, int]:
    """Split emitted tool_calls into (valid, invalid_count). A call is valid when
    its function name is a known tool. Invalid = the model invented a tool — a
    small-model emission-fray signal, distinct from malformed JSON args. Used by
    the diag/StageRecord and the stress measurement."""
    valid, invalid = [], 0
    for c in calls:
        name = (c.get("function") or {}).get("name", "")
        if name in allowed:
            valid.append(c)
        else:
            invalid += 1
    return valid, invalid
