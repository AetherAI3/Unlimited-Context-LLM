"""Per-model profile selection + tool-call validation (the Gemma #1-risk surface)."""

from __future__ import annotations

from aether_agent import profiles
from aether_agent.adapter import OllamaChat


def test_gemma_gets_google_sampling_and_light_tier():
    p = profiles.for_model("gemma4:e4b")
    assert p.tier == profiles.TIER_LIGHT
    assert p.sampling == {"temperature": 1.0, "top_p": 0.95, "top_k": 64}


def test_qwen_gets_deterministic_strong_tier():
    p = profiles.for_model("qwen3-coder:30b")
    assert p.tier == profiles.TIER_STRONG
    assert p.sampling["temperature"] == 0.2


def test_unknown_model_falls_back_to_safe_default():
    p = profiles.for_model("some-random-model:latest")
    assert p.tier == profiles.TIER_STRONG
    assert p.sampling["temperature"] == 0.2  # deterministic default


def test_adapter_binds_the_profile():
    # No network — just confirm the adapter selects the right sampling per model.
    assert OllamaChat(model="gemma4:e4b").profile.sampling["top_k"] == 64
    assert OllamaChat(model="qwen3-coder:30b").profile.sampling["top_k"] == 40


def test_validate_tool_calls_splits_valid_from_invented():
    allowed = frozenset({"read_file", "write_file"})
    calls = [
        {"function": {"name": "read_file"}},
        {"function": {"name": "make_coffee"}},  # invented
        {"function": {"name": "write_file"}},
    ]
    valid, invalid = profiles.validate_tool_calls(calls, allowed)
    assert [c["function"]["name"] for c in valid] == ["read_file", "write_file"]
    assert invalid == 1
