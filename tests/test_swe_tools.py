import subprocess
from pathlib import Path

import pytest

from bench.swe_tools import RepoTools, prepare_checkout


def _git(args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A tiny git repo at a known commit, for tools to read/edit."""
    d = tmp_path / "repo"
    d.mkdir()
    _git(["init", "-q"], d)
    _git(["config", "user.email", "t@t"], d)
    _git(["config", "user.name", "t"], d)
    (d / "a.py").write_text("def add(x, y):\n    return x - y  # bug\n", encoding="utf-8")
    (d / "pkg").mkdir()
    (d / "pkg" / "b.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(["add", "-A"], d)
    _git(["commit", "-qm", "base"], d)
    return d


def test_read_file_returns_contents(repo):
    t = RepoTools(repo)
    out = t.read_file("a.py")
    assert "def add" in out["content"]
    assert out.get("error") is None


def test_read_file_missing_is_error_not_raise(repo):
    t = RepoTools(repo)
    out = t.read_file("nope.py")
    assert "error" in out


def test_grep_finds_match(repo):
    t = RepoTools(repo)
    out = t.grep("return")
    hits = " ".join(h["line"] for h in out["matches"])
    assert "return" in hits
    assert any(h["path"] == "a.py" for h in out["matches"])


def test_list_dir(repo):
    t = RepoTools(repo)
    out = t.list_dir(".")
    assert "a.py" in out["entries"]
    assert "pkg/" in out["entries"]


def test_edit_file_then_patch_is_unified_diff(repo):
    t = RepoTools(repo)
    r = t.edit_file("a.py", "    return x - y  # bug", "    return x + y")
    assert r.get("error") is None
    patch = t.current_patch()
    assert patch.startswith("diff --git") or "--- a/a.py" in patch
    assert "+    return x + y" in patch


def test_edit_missing_oldstring_is_error(repo):
    t = RepoTools(repo)
    r = t.edit_file("a.py", "NONEXISTENT", "x")
    assert "error" in r


def test_prepare_checkout_at_base_commit(tmp_path, repo):
    # `repo` is a real git repo; clone it to a base commit via file:// (no network).
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                          capture_output=True, text=True).stdout.strip()
    work = tmp_path / "work"
    out = prepare_checkout(f"file://{repo.as_posix()}", base, work)
    assert (out / "a.py").is_file()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=out,
                          capture_output=True, text=True).stdout.strip()
    assert head == base
