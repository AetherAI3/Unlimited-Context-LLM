# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Protocol lockstep — pins PROTOCOL_VERSION and the canonical 8-tool tuple.

The TS host (aether-code/src/core/brain_protocol.ts) mirrors these EXACT values.
If either drifts, local and cloud paths advertise different capabilities — these
tests are the drift tripwire on the Python side.
"""
from __future__ import annotations

from aether_agent import protocol

# The canonical order, identical in BOTH repos. The first six are the original
# coding tools; web_search/web_fetch were appended (network tools, not path-jailed).
CANONICAL_TOOLS = (
    "read_file",
    "write_file",
    "run_shell",
    "run_tests",
    "repo_search",
    "git_commit",
    "web_search",
    "web_fetch",
)


def test_protocol_version_is_three():
    assert protocol.PROTOCOL_VERSION == 3


def test_tools_is_the_eight_name_tuple_in_order():
    # Must be an ordered tuple (not a set) so the mirror's order is pinned.
    assert isinstance(protocol.TOOLS, tuple)
    assert protocol.TOOLS == CANONICAL_TOOLS


def test_web_tools_are_present():
    assert "web_search" in protocol.TOOLS
    assert "web_fetch" in protocol.TOOLS


def test_original_six_tools_unchanged_and_first():
    assert protocol.TOOLS[:6] == CANONICAL_TOOLS[:6]
