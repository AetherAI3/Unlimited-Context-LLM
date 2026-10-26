"""The NumPy-only core never imports torch, transformers or the latent package."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CORE = ["aether_context", "aether_context.session", "aether_context.local_llm", "aether_context.encoder",
        "aether_context.engine", "aether_context.context_pool"]


def _modules_after(imports: list[str]) -> set[str]:
    code = "import sys\n" + "".join(f"import {m}\n" for m in imports) + "print('\\n'.join(sorted(sys.modules)))"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    return set(out.stdout.split())


def test_core_never_imports_latent_or_torch():
    loaded = _modules_after(CORE)
    assert "torch" not in loaded
    assert "transformers" not in loaded
    assert not any(m == "aether_context.latent" or m.startswith("aether_context.latent.") for m in loaded)


def test_latent_package_import_is_torch_free():
    loaded = _modules_after(["aether_context.latent", "aether_context.latent.protocol"])
    assert "torch" not in loaded and "transformers" not in loaded


def test_latent_extra_declared_and_core_stays_numpy_only():
    tomllib = pytest.importorskip("tomllib")  # 3.11+; the 3.10 CI leg skips this one test
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    latent = " ".join(project["optional-dependencies"]["latent"])
    for dep in ("torch", "transformers", "pydantic"):
        assert dep in latent
    assert project["dependencies"] == ["numpy>=1.24,<3.0"]
