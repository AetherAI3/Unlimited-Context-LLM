# tests/test_hardware.py
from aether_agent import hardware
from aether_agent.hardware import Resources


def _fit(ram, vram=None, gpu=None):
    res = hardware.detect_resources(
        ram_probe=lambda: ram,
        vram_probe=lambda: (vram, gpu),
    )
    return hardware.best_fit(res)[0]


def test_best_fit_matrix():
    assert _fit(16) == "qwen2.5-coder:7b"          # 16*0.7=11.2 -> 7b (min 8)
    assert _fit(6) == "qwen2.5-coder:3b"           # 6*0.7=4.2 -> 3b (min 4)
    assert _fit(48) == "qwen3-coder:30b"           # 48*0.7=33.6 -> 30b (min 24)
    assert _fit(8, vram=24, gpu="NVIDIA") == "qwen3-coder:30b"  # VRAM wins


def test_low_resources_falls_back_to_permissive_floor():
    # 2 GB fits nothing -> the permissive floor, never the license-flagged Gemma.
    assert _fit(2) == hardware.DEFAULT_FLOOR == "qwen2.5-coder:3b"


def test_flagged_model_never_auto_selected():
    # Even where Gemma (min 6) would "fit", best_fit must skip license-flagged.
    tag = _fit(7)  # 7*0.7=4.9 -> 3b (min4); gemma(min6) excluded anyway
    assert tag != "gemma3n:e4b"


def test_detect_resources_failsoft_default():
    def boom():
        raise OSError("no probe")
    res = hardware.detect_resources(ram_probe=boom, vram_probe=boom)
    assert res == Resources(ram_gb=8.0, vram_gb=None, gpu=None)
