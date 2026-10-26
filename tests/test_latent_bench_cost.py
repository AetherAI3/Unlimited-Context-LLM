# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Latent-transfer bench cost helpers: FLOP formulas, device info, peak RSS."""
from __future__ import annotations

from bench.latent_transfer import cost


def test_forward_flops_formula():
    assert cost.forward_flops(blocks=10, head=7, tokens=100, head_tokens=3) == 2 * 10 * 100 + 2 * 7 * 3


def test_forward_flops_zero_head_tokens_is_sender_only():
    assert cost.forward_flops(10, 7, 100, 0) == 2000


def test_train_flops_is_twice_forward():
    assert cost.train_flops(10, 7, 100, 3) == 2 * cost.forward_flops(10, 7, 100, 3)


def test_device_info_keys():
    info = cost.device_info()
    for key in ("platform", "processor", "cpu_count", "python", "torch", "torch_threads", "cuda_used",
                "device", "transformers", "numpy"):
        assert key in info
    assert info["device"] == "cpu" and info["cuda_used"] is False
    assert info["cpu_count"] == __import__("os").cpu_count()


def test_peak_rss_bytes_is_int_or_none():
    rss = cost.peak_rss_bytes()
    assert rss is None or (isinstance(rss, int) and rss > 0)
