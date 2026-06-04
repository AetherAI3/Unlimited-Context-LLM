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
        "Gemma 4 (E4B light / 26B-MoE / 31B dense). Apache-2.0. Tool-calling ~86%; "
        "Google sampling; plans better with thinking-before-the-call. Do NOT use Gemma 3 "
        "(custom terms + ~7% tool-calling).",
    ),
    ModelProfile(
        "qwen",
        TIER_STRONG,
        {"temperature": 0.2, "top_p": 0.9, "top_k": 40},
        "Qwen3-Coder (30B depth / -next light-MoE). Low-temp deterministic coding; "
        "most reliable tool-calling. Default strong-local.",
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
