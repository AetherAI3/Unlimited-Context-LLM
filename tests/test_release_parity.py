# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Release-parity checks across the two registries this repo publishes to.

`aether-context` ships twice from one commit: the Python package on PyPI, and an npm launcher
that installs *that exact release* into a private virtualenv. The launcher pins the version it
installs to its own ``package.json`` version, so a drift between the two would mean
``npx aether-context`` silently installing a different release than ``pip install`` gives you —
or, once the pin points at a version that was never published, failing outright on first run.

That is not a hypothetical: the sibling `aether-agent` launcher shipped pinned to a version its
npm counterpart had moved past, and installed an older agent than the documented route did.
These tests are the cheap detector.
"""
from __future__ import annotations

import json
import re
from importlib.metadata import distribution
from pathlib import Path

import pytest

import aether_context

try:  # tomllib landed in 3.11; the package floor is 3.10, so this import is optional here.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only taken on the 3.10 leg of the matrix
    tomllib = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent
NPM_DIR = REPO_ROOT / "packages" / "npm-cli"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _npm_manifest() -> dict[str, object]:
    return json.loads((NPM_DIR / "package.json").read_text(encoding="utf-8"))


# --- version parity ----------------------------------------------------------
def test_npm_launcher_version_matches_the_python_package() -> None:
    """The npm launcher and the Python package are one release, so they carry one version."""
    # Act
    npm_version = _npm_manifest()["version"]

    # Assert
    assert npm_version == aether_context.__version__, (
        "packages/npm-cli/package.json and aether_context.__version__ disagree; the launcher "
        "would install a different release than `pip install aether-context` gives you"
    )


def test_pyproject_takes_its_version_from_the_package() -> None:
    """`pyproject.toml` must stay dynamic, so the package attribute is the single source."""
    # Act
    text = PYPROJECT.read_text(encoding="utf-8")

    # Assert
    assert 'dynamic = ["version"]' in text
    assert 'version = { attr = "aether_context.__version__" }' in text
    assert not re.search(r'^version\s*=\s*"', text, re.M), "a literal version would drift"


# --- what the wheel is allowed to contain ------------------------------------
# Read from the build configuration and the installed metadata rather than by scanning the
# TOML for a substring: the prose comment above the setting mentions `aether_agent` too, and a
# test that a comment can satisfy is not a test.
def test_the_installed_distribution_owns_only_aether_context() -> None:
    """`aether_agent` must stay out of the wheel: PyPI's `aether-agent` owns that import path.

    Both distributions installed together would write the same ``aether_agent/`` files, and pip
    does not detect conflicts across distributions — the second install silently wins.
    """
    # Act
    top_level = distribution("aether-context").read_text("top_level.txt")

    # Assert
    assert top_level is not None
    assert top_level.split() == ["aether_context"]


@pytest.mark.skipif(tomllib is None, reason="tomllib requires Python 3.11+")
def test_the_build_configuration_declares_only_aether_context() -> None:
    """The declared package list is the thing the wheel is actually built from."""
    # Act
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    # Assert
    assert config["tool"]["setuptools"]["packages"] == ["aether_context"]
    assert config["project"]["scripts"] == {"aether-context": "aether_context.cli:main"}


def test_the_distribution_declares_only_its_own_console_script() -> None:
    """`aether` and `aether-smoke` belong to the `aether-agent` distribution, not this one."""
    # Act
    installed = {entry.name: entry.value for entry in distribution("aether-context").entry_points}

    # Assert
    assert installed == {"aether-context": "aether_context.cli:main"}


# --- the launcher's own contract ---------------------------------------------
def test_launcher_bin_is_listed_and_present() -> None:
    """The published tarball must actually contain the file `bin` points at."""
    # Act
    manifest = _npm_manifest()
    bin_path = manifest["bin"]["aether-context"]  # type: ignore[index]

    # Assert
    assert (NPM_DIR / bin_path).is_file()
    assert bin_path in manifest["files"]  # type: ignore[operator]


def test_launcher_pins_the_matching_pypi_release() -> None:
    """The launcher installs `aether-context==<its own version>`, not a floating range."""
    # Act
    source = (NPM_DIR / "bin" / "aether-context.js").read_text(encoding="utf-8")

    # Assert
    assert "aether-context==${version}" in source
    assert "PACKAGE_VERSION" in source


def test_launcher_minimum_python_matches_requires_python() -> None:
    """A launcher that accepts an interpreter the package rejects fails after the install."""
    # Arrange
    floor = re.search(r'requires-python\s*=\s*">=(\d+)\.(\d+)', PYPROJECT.read_text(encoding="utf-8"))
    assert floor is not None
    expected = [int(floor.group(1)), int(floor.group(2))]

    # Act
    source = (NPM_DIR / "bin" / "aether-context.js").read_text(encoding="utf-8")
    declared = re.search(r"MIN_PYTHON\s*=\s*\[(\d+),\s*(\d+)\]", source)

    # Assert
    assert declared is not None
    assert [int(declared.group(1)), int(declared.group(2))] == expected
