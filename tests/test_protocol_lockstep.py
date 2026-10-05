# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Protocol lockstep — pins PROTOCOL_VERSION and the canonical 10-tool tuple.

The TS host (aether-code/src/core/brain_protocol.ts) mirrors these EXACT values.
If either drifts, local and cloud paths advertise different capabilities — these
tests are the drift tripwire on the Python side.
"""
from __future__ import annotations

from aether_agent import protocol

# The canonical order, identical in BOTH repos. Filesystem tools precede the
# original write/shell tools; web_search/web_fetch remain last.
CANONICAL_TOOLS = (
    "read_file",
    "list_directory",
    "patch_file",
    "write_file",
    "run_shell",
    "run_tests",
    "repo_search",
    "git_commit",
    "web_search",
    "web_fetch",
)


def test_protocol_version_is_four():
    assert protocol.PROTOCOL_VERSION == 4


def test_tools_is_the_ten_name_tuple_in_order():
    # Must be an ordered tuple (not a set) so the mirror's order is pinned.
    assert isinstance(protocol.TOOLS, tuple)
    assert protocol.TOOLS == CANONICAL_TOOLS


def test_web_tools_are_present():
    assert "web_search" in protocol.TOOLS
    assert "web_fetch" in protocol.TOOLS


def test_original_tools_keep_their_relative_order():
    assert tuple(name for name in protocol.TOOLS if name not in {"list_directory", "patch_file"}) == CANONICAL_TOOLS[0:1] + CANONICAL_TOOLS[3:]
