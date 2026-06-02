# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI - Brandon Barrante
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures for the aether-context test suite.

All tests are numpy-only and never touch the network or the real ``~/.aether-context``
home directory — pool state lives under pytest's ``tmp_path`` and randomness is seeded.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

#: Fixed seed so every property test is deterministic and reproducible.
SEED = 1234


@pytest.fixture
def tmp_pool_dir(tmp_path: Path) -> Path:
    """An isolated, empty pool directory under pytest's tmp_path.

    Used wherever a test needs to read/write pool state (config, mmap index) without
    touching the user's real ``~/.aether-context`` directory.
    """
    d = tmp_path / ".aether-context"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded numpy random generator for deterministic vector/property tests."""
    return np.random.default_rng(SEED)
