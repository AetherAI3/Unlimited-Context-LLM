"""CLI-surface tests — ``run`` / ``status`` / ``clear`` / ``chat`` + the slash dispatcher.

Every test is hermetic: numpy-only, NO network, NO tty, NO ``input()``. We drive
``aether_context.cli.main([...])`` directly (never a subprocess), exercise the PURE
``dispatch_slash`` function, and call the ``Session`` API for the pool-mode / clear semantics.
Pool state lives under pytest's ``tmp_path`` so the real ``~/.aether-context`` is never touched.

Mirrors the AAA pytest style used across the suite.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aether_context import cli
from aether_context.cli import ReplState, dispatch_slash
from aether_context.config import PoolConfig
from aether_context.session import Session


# --- run ---------------------------------------------------------------------
def test_run_returns_zero_and_produces_output(
    tmp_pool_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`run "hello" --model mock --dir <tmp>` exits 0 and prints the model text + a status line."""
    # Act
    code = cli.main(["run", "hello", "--model", "mock", "--dir", str(tmp_pool_dir)])
    out = capsys.readouterr().out

    # Assert
    assert code == 0
    assert out.strip()  # the mock produced text
    assert "pool" in out.lower()  # the one-line status tail rode along


# --- status ------------------------------------------------------------------
def test_status_prints_fields_after_a_run(
    tmp_pool_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """After a run, `status --dir <tmp>` reports pool GB / slices / reach / pool-mode."""
    # Arrange: a run encodes at least one slice into the pool on disk.
    cli.main(["run", "build a small app", "--model", "mock", "--dir", str(tmp_pool_dir)])
    capsys.readouterr()  # drain the run output

    # Act
    code = cli.main(["status", "--dir", str(tmp_pool_dir)])
    out = capsys.readouterr().out.lower()

    # Assert
    assert code == 0
    assert "pool" in out and "gb" in out
    assert "slices" in out
    assert "reach" in out
    assert "pool-mode" in out


def test_status_hit_rate_is_honestly_na_without_a_live_session(
    tmp_pool_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`status` never fabricates a hit rate when no pager is running — it says N/A."""
    cli.main(["status", "--dir", str(tmp_pool_dir)])
    out = capsys.readouterr().out.lower()
    assert "hit rate" in out
    assert "n/a" in out


# --- clear -------------------------------------------------------------------
def test_clear_empties_the_pool_and_status_shows_zero_slices(
    tmp_pool_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`clear --dir <tmp> --yes` empties the pool; a following status shows 0 slices."""
    # Arrange: a run leaves slices in the pool.
    cli.main(["run", "encode some spill", "--model", "mock", "--dir", str(tmp_pool_dir)])
    capsys.readouterr()

    # Act: clear the whole pool (non-tty needs --yes to proceed).
    code = cli.main(["clear", "--dir", str(tmp_pool_dir), "--yes"])
    out = capsys.readouterr().out.lower()

    # Assert: the clear reported, and status now counts zero slices.
    assert code == 0
    assert "cleared" in out
    cli.main(["status", "--dir", str(tmp_pool_dir)])
    status_out = capsys.readouterr().out.lower()
    assert "slices:      0 /" in status_out


def test_clear_non_tty_without_yes_refuses_a_persistent_dir(
    tmp_pool_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-default (persistent) dir requires --yes off a tty; without it, clear refuses."""
    cli.main(["run", "spill", "--model", "mock", "--dir", str(tmp_pool_dir)])
    capsys.readouterr()
    code = cli.main(["clear", "--dir", str(tmp_pool_dir)])  # no --yes, non-tty
    captured = capsys.readouterr()
    blob = (captured.out + captured.err).lower()
    assert code != 0
    assert "yes" in blob  # told the user how to proceed


def test_clear_all_removes_the_pool_dir(
    tmp_pool_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`clear --all --dir <tmp> --yes` removes the entire pool directory."""
    # Arrange: a run materializes the pool dir + files.
    cli.main(["run", "spill", "--model", "mock", "--dir", str(tmp_pool_dir)])
    capsys.readouterr()
    assert tmp_pool_dir.exists()

    # Act
    code = cli.main(["clear", "--all", "--dir", str(tmp_pool_dir), "--yes"])
    out = capsys.readouterr().out.lower()

    # Assert
    assert code == 0
    assert "removed" in out
    assert not tmp_pool_dir.exists()


def test_clear_all_non_tty_without_yes_refuses(
    tmp_pool_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`clear --all` ALWAYS confirms; off a tty without --yes it refuses (dir survives)."""
    cli.main(["run", "spill", "--model", "mock", "--dir", str(tmp_pool_dir)])
    capsys.readouterr()
    code = cli.main(["clear", "--all", "--dir", str(tmp_pool_dir)])  # no --yes
    captured = capsys.readouterr()
    assert code != 0
    assert tmp_pool_dir.exists()  # nothing was removed
    assert "yes" in (captured.out + captured.err).lower()


# --- dispatch_slash (pure) ---------------------------------------------------
@pytest.mark.parametrize(
    "line, expected_action",
    [
        ("/help", "help"),
        ("/status", "status"),
        ("/new", "new"),
        ("/clear", "clear"),
        ("/cls", "clear"),
        ("/pool 10", "pool"),
        ("/model mock", "model"),
        ("/think", "think"),
        ("/export", "export"),
        ("/quit", "quit"),
        ("/bogus", "unknown"),
    ],
)
def test_dispatch_slash_maps_each_command_to_its_action(
    line: str, expected_action: str
) -> None:
    """The PURE dispatcher returns the documented action for each slash-command."""
    state = ReplState()
    action, _message = dispatch_slash(state, line)
    assert action == expected_action


def test_dispatch_slash_passes_argument_payload() -> None:
    """Parameterized commands carry their argument through as the message payload."""
    state = ReplState()
    assert dispatch_slash(state, "/pool 10") == ("pool", "10")
    assert dispatch_slash(state, "/model qwen2.5") == ("model", "qwen2.5")
    assert dispatch_slash(state, "/export out.txt") == ("export", "out.txt")


def test_dispatch_slash_non_slash_line_is_a_continue() -> None:
    """A plain (non-slash) line dispatches to 'continue' carrying the verbatim text."""
    state = ReplState()
    action, message = dispatch_slash(state, "build me an app")
    assert action == "continue"
    assert message == "build me an app"


@pytest.mark.parametrize("prefix", ["﻿", "\xef\xbb\xbf"])
def test_dispatch_slash_strips_a_leading_bom(prefix: str) -> None:
    """A BOM some shells prepend to piped input must not hide a slash-command."""
    state = ReplState()
    action, _message = dispatch_slash(state, f"{prefix}/status")
    assert action == "status"


def test_chat_non_tty_handles_a_single_line_and_exits(
    tmp_pool_dir: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`chat` over a non-tty reads exactly one piped line, prints, and exits 0 (never blocks)."""
    # Arrange: feed a single '/help' line via a fake input(); stdin is non-tty under pytest.
    lines = iter(["/help"])
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: next(lines))

    # Act
    code = cli.main(["chat", "--model", "mock", "--dir", str(tmp_pool_dir)])
    out = capsys.readouterr().out.lower()

    # Assert
    assert code == 0
    assert "slash-commands" in out  # the /help payload was printed


def test_chat_non_tty_eof_exits_cleanly(
    tmp_pool_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An immediate EOF (empty piped stdin) ends the REPL with a clean 0."""
    def _raise_eof(*_a: object, **_k: object) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise_eof)
    assert cli.main(["chat", "--model", "mock", "--dir", str(tmp_pool_dir)]) == 0


# --- pool_mode: separate isolates, shared spans sessions ---------------------
def test_separate_mode_isolates_sessions(tmp_pool_dir: Path) -> None:
    """In `separate` mode a session never sees another session's slices (namespace isolation)."""
    # Arrange: session A plants a fact into a shared dir under separate mode.
    sess_a = Session(model="mock", pool_dir=str(tmp_pool_dir), pool_mode="separate")
    sess_a.remember("ALPHA-load-bearing-fact about the deployment target")
    sess_a.close()

    # A brand-new separate session B over the same dir.
    sess_b = Session(
        model="mock", pool_dir=str(tmp_pool_dir), pool_mode="separate",
        session_id="sess-B-isolated",
    )
    try:
        # B's own scope is empty; a scoped search for B finds nothing of A's.
        qvec = sess_b.encoder.encode("ALPHA-load-bearing-fact")
        scoped = sess_b.pool.search(qvec, k=4, session=sess_b.id)
        assert scoped == []
    finally:
        sess_b.close()


def test_shared_mode_spans_sessions(tmp_pool_dir: Path) -> None:
    """In `shared` mode a session searches globally (session=None) across all sessions."""
    # Arrange: session A plants a fact.
    sess_a = Session(model="mock", pool_dir=str(tmp_pool_dir), pool_mode="shared")
    sess_a.remember("BETA-shared-fact visible across sessions")
    sess_a.close()

    # A new shared session B over the same dir should recall A's fact (global reach).
    sess_b = Session(
        model="mock", pool_dir=str(tmp_pool_dir), pool_mode="shared",
        session_id="sess-B-shared",
    )
    try:
        hits = sess_b.recall("BETA-shared-fact", k=4)
        assert any("BETA-shared-fact" in h.text for h in hits)
        # _scope() resolves to None (global) in shared mode.
        assert sess_b._scope() is None
    finally:
        sess_b.close()


# --- Session.clear scope semantics -------------------------------------------
def test_session_clear_session_scope_drops_this_sessions_slices(tmp_pool_dir: Path) -> None:
    """`Session.clear(scope='session')` removes this session's slices; status reflects 0."""
    # Arrange: a session encodes slices.
    sess = Session(model="mock", pool_dir=str(tmp_pool_dir), pool_mode="separate")
    try:
        sess.remember("a durable fact")
        sess.remember("another durable fact")
        assert sess.status_dict()["slices_used"] >= 2

        # Act
        removed = sess.clear(scope="session")

        # Assert: slices gone, status now reads zero.
        assert removed >= 2
        assert sess.status_dict()["slices_used"] == 0
    finally:
        sess.close()


def test_session_clear_resident_scope_keeps_the_pool(tmp_pool_dir: Path) -> None:
    """`Session.clear(scope='resident')` clears only the window and returns 0; pool survives."""
    sess = Session(model="mock", pool_dir=str(tmp_pool_dir), pool_mode="separate")
    try:
        sess.remember("a durable fact that must survive a resident clear")
        before = sess.status_dict()["slices_used"]
        assert before >= 1

        removed = sess.clear(scope="resident")

        assert removed == 0  # no pool slices were dropped
        assert sess.status_dict()["slices_used"] == before  # pool untouched
    finally:
        sess.close()


def test_session_clear_rejects_an_unknown_scope(tmp_pool_dir: Path) -> None:
    """An unknown clear scope is a typed, hinted error (never a silent no-op)."""
    from aether_context.errors import AetherContextError

    sess = Session(model="mock", pool_dir=str(tmp_pool_dir))
    try:
        with pytest.raises(AetherContextError):
            sess.clear(scope="everything")
    finally:
        sess.close()


# --- Session.export / toggle_extended ----------------------------------------
def test_session_export_writes_the_transcript(tmp_path: Path) -> None:
    """`Session.export(path)` writes the (role, text) transcript and returns the path."""
    sess = Session(model="mock", pool_dir=str(tmp_path / "pool"))
    try:
        sess.ask("first question")
        sess.ask("second question")
        target = tmp_path / "transcript.txt"

        written = sess.export(str(target))

        assert Path(written) == target
        body = target.read_text(encoding="utf-8")
        assert "user: first question" in body
        assert "user: second question" in body
        assert "assistant:" in body
    finally:
        sess.close()


def test_session_toggle_extended_flips_and_widens_resident(tmp_path: Path) -> None:
    """`toggle_extended()` flips the flag, surfaces in status, and widens the resident width."""
    sess = Session(model="mock", pool_dir=str(tmp_path / "pool"))
    try:
        assert sess.extended is False
        base_k = sess.pager.default_k

        on = sess.toggle_extended()
        assert on is True
        assert sess.status_dict()["extended"] is True
        assert sess.pager.default_k > base_k  # honestly widened

        off = sess.toggle_extended()
        assert off is False
        assert sess.pager.default_k == base_k  # restored
    finally:
        sess.close()


# --- ContextPool.clear_session (storage half) --------------------------------
def test_context_pool_clear_session_removes_only_that_session(
    tmp_pool_dir: Path, rng: np.random.Generator
) -> None:
    """`ContextPool.clear_session(id)` drops one session's rows and keeps the rest consistent."""
    from aether_context.context_pool import ContextPool, Slice

    cfg = PoolConfig(pool_gb=5, dir=tmp_pool_dir)
    pool = ContextPool(cfg)
    try:
        for i in range(3):
            vec = rng.standard_normal(cfg.dim).astype(np.float32)
            pool.add(Slice(id=f"A:{i}", session="A", vector=vec, text=f"a{i}", tokens=3))
        for i in range(2):
            vec = rng.standard_normal(cfg.dim).astype(np.float32)
            pool.add(Slice(id=f"B:{i}", session="B", vector=vec, text=f"b{i}", tokens=3))
        assert len(pool) == 5

        removed = pool.clear_session("A")

        assert removed == 3
        assert len(pool) == 2
        # surviving slices are all session B, and stats agree with the live count.
        assert pool.stats()["sessions"] == ["B"]
        assert pool.stats()["count"] == 2
        # a clear of an absent session is a safe no-op.
        assert pool.clear_session("ZZZ") == 0
    finally:
        pool.close()


def test_context_pool_clear_session_none_empties_everything(
    tmp_pool_dir: Path, rng: np.random.Generator
) -> None:
    """`clear_session(None)` empties the whole pool (the shared/global clear)."""
    from aether_context.context_pool import ContextPool, Slice

    cfg = PoolConfig(pool_gb=5, dir=tmp_pool_dir)
    pool = ContextPool(cfg)
    try:
        for i in range(4):
            vec = rng.standard_normal(cfg.dim).astype(np.float32)
            pool.add(Slice(id=f"X:{i}", session=f"S{i % 2}", vector=vec, text=f"x{i}", tokens=3))
        assert len(pool) == 4

        removed = pool.clear_session(None)

        assert removed == 4
        assert len(pool) == 0
        assert pool.stats()["count"] == 0
    finally:
        pool.close()
