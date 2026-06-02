"""Tests for PoolConfig / SessionConfig dataclasses + JSON persistence + reach math."""
import json
from pathlib import Path

import pytest

from aether_context.config import (
    PoolConfig,
    SessionConfig,
    reach_tokens,
    POOL_GB_FLOOR,
    TRIGGER_FRACTION,
    TARGET_FRACTION,
    VERBATIM_FRACTION,
    TOKENS_PER_GB,
)
from aether_context.errors import AetherContextError


# ---- PoolConfig defaults ----------------------------------------------------
def test_pool_config_defaults():
    cfg = PoolConfig()
    assert cfg.pool_gb == 5
    assert cfg.mode == "separate"
    assert cfg.index == "flat"
    assert cfg.dim == 256
    assert cfg.slice_tokens == 512


def test_pool_dir_defaults_under_aether_context(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    cfg = PoolConfig()
    assert cfg.dir.name == ".aether-context"


# ---- PoolConfig validation: 5 GB floor --------------------------------------
def test_pool_gb_below_floor_is_rejected_with_reason():
    with pytest.raises(AetherContextError) as exc_info:
        PoolConfig(pool_gb=3)
    msg = str(exc_info.value).lower()
    assert "5" in msg  # the floor is surfaced in the reason
    assert "pool" in msg


def test_pool_gb_at_floor_is_accepted():
    cfg = PoolConfig(pool_gb=POOL_GB_FLOOR)
    assert cfg.pool_gb == 5


def test_pool_gb_above_floor_is_accepted():
    assert PoolConfig(pool_gb=20).pool_gb == 20


def test_bad_index_kind_is_rejected():
    with pytest.raises(AetherContextError):
        PoolConfig(index="quantum")  # not one of flat/hnsw/tiered


def test_bad_mode_is_rejected():
    with pytest.raises(AetherContextError):
        PoolConfig(mode="telepathic")


# ---- reach math matches the README table ------------------------------------
def test_tokens_per_gb_constant():
    assert TOKENS_PER_GB == 233_000_000


@pytest.mark.parametrize(
    "gb,expected_billion",
    [(5, 1.165), (10, 2.33), (15, 3.495), (20, 4.66)],
)
def test_reach_matches_readme_table(gb, expected_billion):
    reach = reach_tokens(gb)
    assert reach == gb * 233_000_000
    assert abs(reach / 1e9 - expected_billion) < 1e-6


# ---- SessionConfig defaults + fractions -------------------------------------
def test_session_config_defaults():
    cfg = SessionConfig(model="mock")
    assert cfg.model == "mock"
    assert cfg.system is None
    assert cfg.max_tokens is None
    assert cfg.verbatim_fraction == 0.30
    assert cfg.trigger_fraction == 0.75
    assert cfg.target_fraction == 0.50


def test_window_fraction_constants_mirror_aethercloud():
    assert TRIGGER_FRACTION == 0.75
    assert TARGET_FRACTION == 0.50
    assert VERBATIM_FRACTION == 0.30


def test_session_config_requires_model():
    with pytest.raises(TypeError):
        SessionConfig()  # type: ignore[call-arg]


# ---- save / load round-trip -------------------------------------------------
def test_pool_config_round_trips_through_json(tmp_pool_dir):
    cfg = PoolConfig(pool_gb=10, mode="shared", index="flat", dir=tmp_pool_dir)
    path = cfg.save()
    assert path.exists()
    assert path.name == "config.json"

    loaded = PoolConfig.load(dir=tmp_pool_dir)
    assert loaded.pool_gb == 10
    assert loaded.mode == "shared"
    assert loaded.index == "flat"
    assert loaded.dim == cfg.dim
    assert loaded.slice_tokens == cfg.slice_tokens


def test_saved_json_is_human_readable(tmp_pool_dir):
    cfg = PoolConfig(pool_gb=15, dir=tmp_pool_dir)
    path = cfg.save()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["pool_gb"] == 15
    assert data["index"] == "flat"


def test_load_missing_config_returns_defaults(tmp_pool_dir):
    # no file written yet -> defaults (dir pinned to the tmp dir)
    cfg = PoolConfig.load(dir=tmp_pool_dir)
    assert cfg.pool_gb == 5
    assert cfg.dir == tmp_pool_dir


def test_load_corrupt_config_raises_typed_error(tmp_pool_dir):
    (tmp_pool_dir / "config.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(AetherContextError):
        PoolConfig.load(dir=tmp_pool_dir)
