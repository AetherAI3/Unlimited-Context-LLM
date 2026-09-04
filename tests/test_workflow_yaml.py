# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Every workflow file must be parseable YAML that still declares its triggers.

This exists because of a real escape. `publish-npm.yml` merged to `main` containing

    run: echo "dry run: packed ... without publishing"

which is **not valid YAML** — a plain scalar cannot contain ``": "``, and the quotes there
belong to the shell string, not to YAML. GitHub could not read the file's `on:` block, so
dispatching it failed with the thoroughly misleading
``Workflow does not have 'workflow_dispatch' trigger`` while the trigger was sitting right
there in the source. Nothing in CI parsed workflow YAML, so nothing caught it.

The trigger assertion is the half that matters: a syntactically valid file whose `on:` block
got swallowed by a mis-indented key is silently un-runnable, which is exactly the failure that
a publish workflow cannot afford — you find out at release time.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml  # dev-only dependency; declared in the `dev` extra, never at runtime

WORKFLOW_DIR = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def _workflow_files() -> list[Path]:
    return sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))


def test_there_are_workflows_to_check() -> None:
    """Guard the guard: a glob that matches nothing would make every test below vacuous."""
    # Assert
    assert _workflow_files(), f"no workflow files found under {WORKFLOW_DIR}"


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_workflow_is_valid_yaml(path: Path) -> None:
    """The file parses. An unparseable workflow is invisible to GitHub, not merely broken."""
    # Act / Assert
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - only on a regression
        pytest.fail(f"{path.name} is not valid YAML: {exc}")

    assert isinstance(document, dict), f"{path.name} did not parse to a mapping"


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_workflow_declares_triggers(path: Path) -> None:
    """`on:` survives the parse with at least one trigger.

    YAML 1.1 reads a bare ``on`` as the boolean ``True``, which is why the key is looked up
    both ways — GitHub accepts the file either way, so the test must too.
    """
    # Act
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    triggers = document.get("on", document.get(True))

    # Assert
    assert triggers, f"{path.name} declares no triggers; GitHub can never run it"
    if isinstance(triggers, dict):
        assert list(triggers.keys()), f"{path.name} has an empty `on:` mapping"


def test_publish_workflows_are_dispatchable() -> None:
    """Both publish workflows must be manually dispatchable — that is how releases are cut.

    Pinned explicitly because the failure is asymmetric: losing this trigger does not break
    any test or any push, it just makes the release button disappear until someone needs it.
    """
    # Arrange
    expected = {"publish.yml", "publish-npm.yml"}

    # Act / Assert
    for name in expected:
        path = WORKFLOW_DIR / name
        assert path.is_file(), f"{name} is missing"
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        triggers = document.get("on", document.get(True))
        assert isinstance(triggers, dict), f"{name} has no trigger mapping"
        assert "workflow_dispatch" in triggers, f"{name} is not manually dispatchable"
