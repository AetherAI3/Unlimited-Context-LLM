"""Filesystem tool contract shared with the Aether Agent host."""

from __future__ import annotations

import json
import os

from aether_agent.tools import Tools, tool_schema


def test_schema_and_paginated_listing(tmp_path):
    names = [entry["function"]["name"] for entry in tool_schema()]
    assert names[:4] == ["read_file", "list_directory", "patch_file", "write_file"]
    (tmp_path / "empty").mkdir()
    (tmp_path / "sub dir").mkdir()
    (tmp_path / "é space.txt").write_text("é", encoding="utf-8")
    tools = Tools(str(tmp_path))
    assert json.loads(tools.list_directory("empty"))["entries"] == []
    first = json.loads(tools.list_directory(".", limit=1))
    second = json.loads(tools.list_directory(".", first["next_cursor"], limit=2))
    paths = [item["path"] for item in first["entries"] + second["entries"]]
    assert "./é space.txt" in paths
    assert "./sub dir" in paths
    (tmp_path / "new.txt").write_text("new")
    assert "conflict" in tools.execute("list_directory", {"path": ".", "cursor": first["next_cursor"]})


def test_ranged_read_and_targeted_patch_conflict(tmp_path):
    target = tmp_path / "file.txt"
    target.write_bytes(b"first\nsecond\n")
    tools = Tools(str(tmp_path))
    read = json.loads(tools.read_file("file.txt", start_line=2, max_lines=1))
    assert read["content"] == "second"
    patch = {"path": "file.txt", "expected_sha256": read["sha256"],
             "old_text": "second", "new_text": "SECOND"}
    assert "patched" in tools.execute("patch_file", patch)
    assert target.read_text(encoding="utf-8") == "first\nSECOND\n"
    assert "conflict" in tools.execute("patch_file", patch)
    assert target.read_text(encoding="utf-8") == "first\nSECOND\n"


def test_boundary_insertion_and_mode_preservation(tmp_path):
    target = tmp_path / "bounds.txt"
    target.write_bytes(b"middle\n")
    os.chmod(target, 0o640)
    mode = target.stat().st_mode & 0o777
    tools = Tools(str(tmp_path))
    digest = json.loads(tools.read_file("bounds.txt"))["sha256"]
    assert "patched" in tools.execute("patch_file", {"path": "bounds.txt", "expected_sha256": digest,
                                                   "old_text": "", "new_text": "first\n", "start_line": 1})
    assert target.read_text(encoding="utf-8") == "first\nmiddle\n"
    assert target.stat().st_mode & 0o777 == mode


def test_ambiguous_and_nonmatching_hunks_do_not_write(tmp_path):
    target = tmp_path / "same.txt"
    target.write_bytes(b"same\nsame\n")
    tools = Tools(str(tmp_path))
    digest = json.loads(tools.read_file("same.txt"))["sha256"]
    for old_text, reason in [("same", "ambiguous"), ("missing", "does not match")]:
        result = tools.execute("patch_file", {"path": "same.txt", "expected_sha256": digest,
                                              "old_text": old_text, "new_text": "new"})
        assert reason in result
        assert target.read_bytes() == b"same\nsame\n"
