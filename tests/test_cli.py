"""CLI surface tests — ``aether-context init / --pool / doctor / bench``.

All tests are hermetic: no network, no real ``~/.aether-context`` (pool state lives under
pytest's ``tmp_path``), no Ollama. ``doctor`` must run fully offline and report its checks
with exact fix commands; ``init`` must be non-tty safe (take ``--pool N`` or env, never block
on a prompt under pytest); ``--pool 3`` must be rejected with the 5 GB floor reason.

Mirrors the AAA pytest style used across the suite. We drive ``cli.main(argv)`` directly so
we never spawn a subprocess.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aether_context import cli
from aether_context.config import POOL_GB_FLOOR, PoolConfig


# --- argument parsing --------------------------------------------------------
def test_build_parser_exposes_the_documented_subcommands() -> None:
    """The parser wires up init / doctor / bench and the top-level --pool resize."""
    parser = cli.build_parser()
    # parse each documented form without error
    init_ns = parser.parse_args(["init", "--pool", "10"])
    assert init_ns.command == "init"
    assert init_ns.pool == 10

    doctor_ns = parser.parse_args(["doctor"])
    assert doctor_ns.command == "doctor"

    bench_ns = parser.parse_args(["bench", "--quick"])
    assert bench_ns.command == "bench"

    resize_ns = parser.parse_args(["--pool", "12"])
    assert resize_ns.pool == 12
    assert resize_ns.command is None  # top-level resize, no subcommand


def test_no_command_prints_help_and_exits_clean(capsys: pytest.CaptureFixture[str]) -> None:
    """Bare invocation is friendly: prints usage, exits 0, never raises."""
    code = cli.main([])
    out = capsys.readouterr().out
    assert code == 0
    assert "aether-context" in out
    assert "doctor" in out


# --- init: non-tty safe, default 5, reject < 5 -------------------------------
def test_init_with_explicit_pool_writes_config(tmp_pool_dir: Path) -> None:
    """`init --pool 10 --dir <tmp>` persists a PoolConfig with the chosen reach (no prompt)."""
    code = cli.main(["init", "--pool", "10", "--dir", str(tmp_pool_dir)])
    assert code == 0
    cfg = PoolConfig.load(tmp_pool_dir)
    assert cfg.pool_gb == 10


def test_init_default_is_five_gb_when_non_tty(
    tmp_pool_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no --pool and a non-tty stdin, init takes the default 5 GB (never blocks)."""
    monkeypatch.delenv("AETHER_POOL_GB", raising=False)
    code = cli.main(["init", "--dir", str(tmp_pool_dir)])
    assert code == 0
    cfg = PoolConfig.load(tmp_pool_dir)
    assert cfg.pool_gb == POOL_GB_FLOOR  # == 5


def test_init_reads_pool_from_env_when_no_flag(
    tmp_pool_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AETHER_POOL_GB is honored when --pool is absent (the documented non-tty path)."""
    monkeypatch.setenv("AETHER_POOL_GB", "15")
    code = cli.main(["init", "--dir", str(tmp_pool_dir)])
    assert code == 0
    cfg = PoolConfig.load(tmp_pool_dir)
    assert cfg.pool_gb == 15


def test_init_pool_3_is_rejected_with_the_floor_reason(
    tmp_pool_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--pool 3` is rejected (non-zero exit) and the reason names the 5 GB floor."""
    code = cli.main(["init", "--pool", "3", "--dir", str(tmp_pool_dir)])
    assert code != 0
    captured = capsys.readouterr()
    blob = (captured.err + captured.out).lower()
    assert "5" in blob  # the floor is named
    assert "pool" in blob
    # no config should have been written for an invalid request
    assert not (tmp_pool_dir / "config.json").exists()


def test_top_level_pool_3_is_rejected_with_reason(
    tmp_pool_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The top-level resize form `--pool 3` is rejected with the same floor reason."""
    code = cli.main(["--pool", "3", "--dir", str(tmp_pool_dir)])
    assert code != 0
    captured = capsys.readouterr()
    blob = (captured.err + captured.out).lower()
    assert "5" in blob and "pool" in blob


def test_pool_resize_is_non_destructive_reindex(tmp_pool_dir: Path) -> None:
    """Resizing changes pool_gb in place without deleting the dir (non-destructive)."""
    cli.main(["init", "--pool", "5", "--dir", str(tmp_pool_dir)])
    # a marker file simulates existing pool payloads that resize must not delete
    marker = tmp_pool_dir / "vectors.bin"
    marker.write_bytes(b"\x00" * 8)
    code = cli.main(["--pool", "20", "--dir", str(tmp_pool_dir)])
    assert code == 0
    assert PoolConfig.load(tmp_pool_dir).pool_gb == 20
    assert marker.exists()  # payloads untouched


# --- doctor: runs offline, prints exact fixes --------------------------------
def test_doctor_runs_offline_and_returns_a_code(
    tmp_pool_dir: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`doctor` must complete fully offline (Ollama unreachable) and never raise.

    We point it at an unroutable host so the reachability probe fails fast and is reported
    as a fixable condition, not an exception.
    """
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:1")  # nothing listening here
    code = cli.main(["doctor", "--dir", str(tmp_pool_dir)])
    out = capsys.readouterr().out
    assert isinstance(code, int)  # returned a status, did not raise
    # it reports on each of the three documented failure modes
    low = out.lower()
    assert "ollama" in low
    # and prints the exact fix command for a down daemon
    assert "ollama serve" in low


def test_doctor_prints_model_pull_fix(
    tmp_pool_dir: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When asked about a specific model, doctor prints the exact `ollama pull` command."""
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:1")
    code = cli.main(["doctor", "--model", "qwen2.5", "--dir", str(tmp_pool_dir)])
    out = capsys.readouterr().out.lower()
    assert isinstance(code, int)
    assert "ollama pull qwen2.5" in out


def test_doctor_reports_ram_vs_index(
    tmp_pool_dir: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Doctor estimates index RAM for the configured pool and reports free RAM vs index."""
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:1")
    cli.main(["init", "--pool", "5", "--dir", str(tmp_pool_dir)])
    capsys.readouterr()  # drain init output
    cli.main(["doctor", "--dir", str(tmp_pool_dir)])
    out = capsys.readouterr().out.lower()
    assert "ram" in out or "memory" in out
    assert "index" in out


# --- bench delegation --------------------------------------------------------
def test_bench_quick_delegates_and_runs_hermetic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`bench --quick` delegates to the bench script and runs a hermetic mock comparison."""
    code = cli.main(["bench", "--quick"])
    out = capsys.readouterr().out.lower()
    assert code == 0
    # the bench reports an ON-vs-OFF comparison
    assert "on" in out and "off" in out
