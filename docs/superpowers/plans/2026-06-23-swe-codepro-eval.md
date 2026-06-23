# SWE-bench CodePro Eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure dsv4-pro coding capability before (raw, window-truncated) vs after (Unlimited-Context engine at max: overpool + TurboVec 8-bit + MPO chain) on real SWE-bench-lite, scored by the official `swebench` resolved-rate.

**Architecture:** Two phases. Phase A (ours, `bench/swe_eval.py` + `bench/swe_tools.py`): an agentic tool-loop drives dsv4-pro over a per-instance git checkout to produce a patch → SWE-bench predictions JSONL. Phase B (`bench/swe_scoring.py` → official `swebench`): builds per-instance test Docker images on VPS5, computes resolved-rate. Reuses `api_eval.py` cost primitives + the `Session` engine.

**Tech Stack:** Python stdlib + numpy, `aether_context` (`Session`, `StaticEncoder`, `quantize`), `OpenAICompatLLM` (OpenRouter dsv4-pro), `datasets` (load SWE-bench-lite), `swebench` (Phase B, VPS5+Docker), `git` CLI.

**Branch:** `feat/turbovec-bench` (Unlimited-Context). Tests flat in `tests/`.

**Reused symbols (do not redefine):**
- `bench/api_eval.py`: `cost_usd(usage, *, price_in, price_out) -> float`, `cached_tokens(usage) -> int`.
- `aether_context/session.py`: `Session(model, pool_gb=5, pool_quantize=0, mpo_chain=True, context_window=...)`; methods `.remember(text)`, `._key(topic=...) -> SliceKey`, `._cold_retrieve(key, query_vec, k) -> list[Slice]` (slice has `.text`), `.close()`. **TurboVec = `pool_quantize=8`.**
- `aether_context/encoder.py`: `StaticEncoder(dim=256).encode(text) -> np.ndarray`.
- `aether_context/local_llm.py`: `OpenAICompatLLM(model, context_window).chat(messages, tools=None, *, max_tokens=None) -> {"content","usage","tool_calls"}`.

---

## Task 0: Setup — deps + dirs

**Files:**
- Modify: `requirements-bench.txt` (create if absent)

- [ ] **Step 1: Record bench deps**

Create/append `requirements-bench.txt`:

```
datasets>=2.19
swebench>=2.1.0
```

- [ ] **Step 2: Install locally (generation only; swebench Phase B runs on VPS5)**

Run: `python -m pip install datasets`
Expected: installs; `python -c "import datasets; print('ok')"` prints `ok`.
(Do NOT need `swebench` locally — Phase B is VPS5/Docker. It's listed for the VPS5 host.)

- [ ] **Step 3: Commit**

```bash
git add requirements-bench.txt
git commit -m "chore(bench): pin SWE-bench eval deps (datasets, swebench)"
```

---

## Task 1: `swe_tools.py` — repo file tools (read_file)

**Files:**
- Create: `bench/swe_tools.py`
- Test: `tests/test_swe_tools.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_swe_tools.py
import subprocess
from pathlib import Path
import pytest
from bench.swe_tools import RepoTools


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_swe_tools.py -o addopts="" -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'bench.swe_tools'`.

- [ ] **Step 3: Write minimal implementation**

```python
# bench/swe_tools.py
"""Repo file tools for the SWE-bench codepro eval — read/grep/list/edit over a single
git checkout, plus the unified-diff capture used as the SWE-bench model_patch. Pure
(no network); the agent loop in swe_eval.py hosts these over a per-instance checkout."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


class RepoTools:
    """Tools answered from a checked-out repo dir. Edits write into the working tree;
    `current_patch()` returns `git diff` against the base commit = the model_patch."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.calls = 0
        self.redundant = 0
        self._read: set[str] = set()

    def _safe(self, rel: str) -> Path | None:
        """Resolve rel under root; None if it escapes (path-traversal guard)."""
        p = (self.root / rel).resolve()
        try:
            p.relative_to(self.root.resolve())
        except ValueError:
            return None
        return p

    def read_file(self, path: str) -> dict:
        self.calls += 1
        if path in self._read:
            self.redundant += 1
        self._read.add(path)
        p = self._safe(path or "")
        if p is None or not p.is_file():
            return {"error": f"no file {path}"}
        try:
            return {"path": path, "content": p.read_text(encoding="utf-8", errors="replace")[:20000]}
        except OSError as e:
            return {"error": str(e)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_swe_tools.py -o addopts="" -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add bench/swe_tools.py tests/test_swe_tools.py
git commit -m "feat(bench): swe_tools RepoTools.read_file with path-traversal guard"
```

---

## Task 2: `swe_tools.py` — grep, list_dir, edit_file, current_patch

**Files:**
- Modify: `bench/swe_tools.py`
- Test: `tests/test_swe_tools.py`

- [ ] **Step 1: Add failing tests**

```python
# append to tests/test_swe_tools.py
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
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_swe_tools.py -o addopts="" -q`
Expected: FAIL — `AttributeError: 'RepoTools' object has no attribute 'grep'`.

- [ ] **Step 3: Implement**

Append to `bench/swe_tools.py`:

```python
    def grep(self, pattern: str, path_glob: str = "") -> dict:
        self.calls += 1
        cmd = ["git", "grep", "-n", "-I", "-e", pattern or ""]
        if path_glob:
            cmd += ["--", path_glob]
        proc = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True)
        matches = []
        for ln in proc.stdout.splitlines()[:100]:
            parts = ln.split(":", 2)
            if len(parts) == 3:
                matches.append({"path": parts[0], "lineno": parts[1], "line": parts[2]})
        return {"matches": matches}

    def list_dir(self, path: str = ".") -> dict:
        self.calls += 1
        p = self._safe(path or ".")
        if p is None or not p.is_dir():
            return {"error": f"no dir {path}"}
        entries = sorted(
            (c.name + "/" if c.is_dir() else c.name) for c in p.iterdir()
            if c.name != ".git")
        return {"entries": entries}

    def edit_file(self, path: str, old: str, new: str) -> dict:
        self.calls += 1
        p = self._safe(path or "")
        if p is None or not p.is_file():
            return {"error": f"no file {path}"}
        src = p.read_text(encoding="utf-8", errors="replace")
        if old not in src:
            return {"error": f"old string not found in {path}"}
        p.write_text(src.replace(old, new, 1), encoding="utf-8")
        return {"ok": True, "path": path}

    def current_patch(self) -> str:
        """Unified diff of the working tree vs HEAD (the base commit) = model_patch."""
        proc = subprocess.run(["git", "diff"], cwd=self.root, capture_output=True, text=True)
        return proc.stdout

    TOOLS_SCHEMA = [
        {"type": "function", "function": {
            "name": "read_file",
            "description": "Read a repo file's contents by path.",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {
            "name": "grep",
            "description": "Search the repo for a regex/string (git grep). Optional path glob.",
            "parameters": {"type": "object",
                           "properties": {"pattern": {"type": "string"},
                                          "path_glob": {"type": "string"}},
                           "required": ["pattern"]}}},
        {"type": "function", "function": {
            "name": "list_dir",
            "description": "List entries in a repo directory.",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {
            "name": "edit_file",
            "description": "Replace the first occurrence of `old` with `new` in a file.",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"},
                                          "old": {"type": "string"}, "new": {"type": "string"}},
                           "required": ["path", "old", "new"]}}},
    ]

    def dispatch(self, name: str, args: dict) -> Any:
        if name == "read_file":
            return self.read_file(args.get("path", ""))
        if name == "grep":
            return self.grep(args.get("pattern", ""), args.get("path_glob", ""))
        if name == "list_dir":
            return self.list_dir(args.get("path", "."))
        if name == "edit_file":
            return self.edit_file(args.get("path", ""), args.get("old", ""), args.get("new", ""))
        return {"error": f"unknown tool {name}"}
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_swe_tools.py -o addopts="" -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add bench/swe_tools.py tests/test_swe_tools.py
git commit -m "feat(bench): swe_tools grep/list_dir/edit_file + unified-diff capture + schema"
```

---

## Task 3: `swe_tools.py` — `prepare_checkout`

**Files:**
- Modify: `bench/swe_tools.py`
- Test: `tests/test_swe_tools.py`

- [ ] **Step 1: Add failing test (local file:// repo, no network)**

```python
# append to tests/test_swe_tools.py
from bench.swe_tools import prepare_checkout


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
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_swe_tools.py::test_prepare_checkout_at_base_commit -o addopts="" -q`
Expected: FAIL — `ImportError: cannot import name 'prepare_checkout'`.

- [ ] **Step 3: Implement**

Append to `bench/swe_tools.py`:

```python
def prepare_checkout(repo_url: str, base_commit: str, workdir: Path) -> Path:
    """Clone `repo_url` into `workdir` and checkout `base_commit`. Idempotent: if workdir
    already holds that commit, reuse it. Returns the checkout path. Used only so the agent's
    file tools can READ the repo during generation; Phase B does its own isolated checkout."""
    workdir = Path(workdir)
    if (workdir / ".git").is_dir():
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=workdir,
                              capture_output=True, text=True).stdout.strip()
        if head == base_commit:
            return workdir
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)
    workdir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", repo_url, str(workdir)], check=True,
                   capture_output=True)
    subprocess.run(["git", "checkout", "-q", base_commit], cwd=workdir, check=True,
                   capture_output=True)
    return workdir
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_swe_tools.py -o addopts="" -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add bench/swe_tools.py tests/test_swe_tools.py
git commit -m "feat(bench): swe_tools prepare_checkout (clone + checkout base_commit, idempotent)"
```

---

## Task 4: `swe_eval.py` — config + dataset/instance selection

**Files:**
- Create: `bench/swe_eval.py`
- Test: `tests/test_swe_eval.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_swe_eval.py
from pathlib import Path
from bench.swe_eval import SweConfig, select_instances


_FAKE = [
    {"instance_id": "p__c-3", "repo": "p/c", "base_commit": "c3", "problem_statement": "s3"},
    {"instance_id": "p__c-1", "repo": "p/c", "base_commit": "c1", "problem_statement": "s1"},
    {"instance_id": "p__c-2", "repo": "p/c", "base_commit": "c2", "problem_statement": "s2"},
]


def test_select_instances_sorted_and_capped():
    got = select_instances(_FAKE, n=2)
    assert [i["instance_id"] for i in got] == ["p__c-1", "p__c-2"]  # sorted, first 2


def test_select_instances_all_when_n_zero():
    got = select_instances(_FAKE, n=0)
    assert len(got) == 3
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_swe_eval.py -o addopts="" -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'bench.swe_eval'`.

- [ ] **Step 3: Implement**

```python
# bench/swe_eval.py
"""swe_eval — SWE-bench codepro eval (Phase A: generation). Drives dsv4-pro through an
agentic file-tool loop over a per-instance checkout, off vs codepro-engine arms, and writes
SWE-bench predictions JSONL. Phase B scoring is bench/swe_scoring.py (official swebench).

Run (dry-run, no key): python -m bench.swe_eval --dry-run
Run (live): OPENROUTER_API_KEY=... python -m bench.swe_eval --instances 1 --arms off,codepro
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from aether_context.encoder import StaticEncoder
from aether_context.session import Session
from bench.api_eval import cached_tokens, cost_usd
from bench.swe_tools import RepoTools, prepare_checkout

DEFAULT_MODEL = "deepseek/deepseek-v4-pro"
DATASET = "princeton-nlp/SWE-bench_Lite"
MODEL_NAME = "dsv4pro-codepro"   # stamped into predictions model_name_or_path


@dataclass
class SweConfig:
    model: str = DEFAULT_MODEL
    arms: tuple[str, ...] = ("off", "codepro")
    instances: int = 0            # 0 = all of lite (300); N = first N (sorted)
    window: int = 8192            # off-arm truncation window
    max_steps: int = 30           # tool steps per instance ("max reasoning" budget)
    pool_gb: int = 50             # codepro overpool reach
    turbovec_bits: int = 8        # codepro TurboVec quant (0/4/8)
    mpo_chain: bool = True        # codepro MPO chain
    recall_k: int = 8             # slices recalled per turn (codepro)
    price_in: float = 0.3
    price_out: float = 1.2
    max_usd: float = 25.0         # hard global spend cap (shared across arms+instances)
    cost_spike_usd: float = 0.50
    dry_run: bool = False
    out_dir: Path = field(default_factory=lambda: Path("runs/swe_eval"))
    work_dir: Path = field(default_factory=lambda: Path("runs/swe_eval/checkouts"))


def select_instances(rows: list[dict], n: int) -> list[dict]:
    """Deterministic instance set: sort by instance_id, take first n (n=0 -> all)."""
    ordered = sorted(rows, key=lambda r: r["instance_id"])
    return ordered if n <= 0 else ordered[:n]


def load_lite(cfg: SweConfig) -> list[dict]:
    """Load SWE-bench-lite test split. Dry-run -> a synthetic 2-instance stand-in."""
    if cfg.dry_run:
        return [
            {"instance_id": "syn__repo-1", "repo": "syn/repo", "base_commit": "HEAD",
             "problem_statement": "Fix add() to add instead of subtract."},
            {"instance_id": "syn__repo-2", "repo": "syn/repo", "base_commit": "HEAD",
             "problem_statement": "Fix VALUE to 2."},
        ]
    from datasets import load_dataset
    ds = load_dataset(DATASET, split="test")
    return [dict(r) for r in ds]
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_swe_eval.py -o addopts="" -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add bench/swe_eval.py tests/test_swe_eval.py
git commit -m "feat(bench): swe_eval SweConfig + deterministic instance selection + lite loader"
```

---

## Task 5: `swe_eval.py` — single-instance arm runner (the agent loop)

**Files:**
- Modify: `bench/swe_eval.py`
- Test: `tests/test_swe_eval.py`

- [ ] **Step 1: Failing test (mock chat that edits then stops; both arms)**

```python
# append to tests/test_swe_eval.py
import subprocess, json as _json
from bench.swe_eval import run_instance


class _PatchChat:
    """Two steps: (1) call edit_file to fix the bug, (2) emit a final message."""
    def __init__(self, *_a, **_k): self._t = 0
    def chat(self, messages, tools=None, *, max_tokens=None):
        self._t += 1
        usage = {"prompt_tokens": 50, "completion_tokens": 10}
        if self._t == 1:
            return {"content": None, "usage": usage, "tool_calls": [
                {"id": "c1", "type": "function", "function": {
                    "name": "edit_file",
                    "arguments": _json.dumps({"path": "a.py",
                                              "old": "return x - y  # bug",
                                              "new": "return x + y"})}}]}
        return {"content": "done", "usage": usage, "tool_calls": []}


def _make_repo(tmp_path):
    d = tmp_path / "repo"; d.mkdir()
    for a in (["init","-q"],["config","user.email","t@t"],["config","user.name","t"]):
        subprocess.run(["git",*a], cwd=d, check=True, capture_output=True)
    (d/"a.py").write_text("def add(x, y):\n    return x - y  # bug\n", encoding="utf-8")
    subprocess.run(["git","add","-A"], cwd=d, check=True, capture_output=True)
    subprocess.run(["git","commit","-qm","base"], cwd=d, check=True, capture_output=True)
    return d


def _base(repo):
    return subprocess.run(["git","rev-parse","HEAD"], cwd=repo,
                          capture_output=True, text=True).stdout.strip()


def test_run_instance_produces_patch_off_arm(tmp_path):
    repo = _make_repo(tmp_path)
    cfg = SweConfig(dry_run=True, work_dir=tmp_path/"wd")
    inst = {"instance_id": "syn__repo-1", "repo": "syn/repo",
            "base_commit": _base(repo), "problem_statement": "fix add"}
    budget = {"spent": 0.0}
    rec = run_instance("off", inst, cfg, _PatchChat(), budget,
                       repo_url=f"file://{repo.as_posix()}")
    assert rec["instance_id"] == "syn__repo-1"
    assert "+    return x + y" in rec["model_patch"]
    assert rec["arm"] == "off"


def test_run_instance_codepro_arm_uses_engine(tmp_path):
    repo = _make_repo(tmp_path)
    cfg = SweConfig(dry_run=True, work_dir=tmp_path/"wd", pool_gb=1)
    inst = {"instance_id": "syn__repo-1", "repo": "syn/repo",
            "base_commit": _base(repo), "problem_statement": "fix add"}
    budget = {"spent": 0.0}
    rec = run_instance("codepro", inst, cfg, _PatchChat(), budget,
                       repo_url=f"file://{repo.as_posix()}")
    assert "+    return x + y" in rec["model_patch"]
    assert rec["arm"] == "codepro"
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_swe_eval.py -k run_instance -o addopts="" -q`
Expected: FAIL — `ImportError: cannot import name 'run_instance'`.

- [ ] **Step 3: Implement**

Append to `bench/swe_eval.py`:

```python
def _truncate(messages: list[dict], window_tokens: int) -> list[dict]:
    """OFF arm: keep system + a tail of messages fitting ~window (chars/4 estimate)."""
    from aether_context.tokenizer import estimate
    budget = window_tokens
    head, tail = messages[:1], []
    for m in reversed(messages[1:]):
        c = estimate(m.get("content") or "")
        if budget - c < 0:
            break
        budget -= c
        tail.insert(0, m)
    return head + tail


_SYS = ("You are an expert software engineer fixing a real bug in a repository. Use read_file, "
        "grep, and list_dir to investigate, then edit_file to apply a minimal fix. When the fix "
        "is complete, reply with a one-line summary and STOP (no more tool calls). Make the repo's "
        "own tests pass; change as little as possible.")


def _empty_record(arm: str, inst: dict, halted: str) -> dict:
    return {
        "instance_id": inst["instance_id"],
        "model_name_or_path": MODEL_NAME if arm == "codepro" else f"{MODEL_NAME}-off",
        "model_patch": "", "arm": arm, "cost_usd": 0.0, "cached_tokens": 0,
        "tool_calls": 0, "redundant_tool_calls": 0, "patch_nonempty": False,
        "halted": halted,
    }


def run_instance(arm: str, inst: dict, cfg: SweConfig, chat, budget: dict,
                 *, repo_url: Optional[str] = None) -> dict:
    """Run ONE SWE instance under ONE arm. Returns a predictions record incl. model_patch.
    OFF: window-truncated transcript. CODEPRO: Session(overpool+turbovec+chain) recall."""
    url = repo_url or f"https://github.com/{inst['repo']}.git"
    workdir = cfg.work_dir / arm / inst["instance_id"]
    try:
        checkout = prepare_checkout(url, inst["base_commit"], workdir)
    except Exception as e:  # clone/checkout failed — record empty patch, keep the batch alive
        return _empty_record(arm, inst, f"checkout_error: {type(e).__name__}")
    tools = RepoTools(checkout)

    session: Optional[Session] = None
    encoder = StaticEncoder(dim=256)
    if arm == "codepro":
        session = Session("swe", pool_gb=cfg.pool_gb,
                          pool_dir=cfg.out_dir / f"pool_{inst['instance_id']}_{arm}",
                          context_window=cfg.window, mpo_chain=cfg.mpo_chain,
                          pool_quantize=cfg.turbovec_bits)

    user = (f"Repository: {inst['repo']} @ {inst['base_commit'][:10]}\n\n"
            f"Problem:\n{inst['problem_statement']}\n\nInvestigate and fix it.")
    transcript: list[dict] = [{"role": "system", "content": _SYS},
                              {"role": "user", "content": user}]
    cost = 0.0
    cached = 0
    halted: Optional[str] = None

    for _step in range(cfg.max_steps):
        if budget["spent"] >= cfg.max_usd:
            halted = "budget_cap"
            break
        if session is not None:
            qvec = encoder.encode(inst["problem_statement"])
            recalled = session._cold_retrieve(session._key(), qvec, cfg.recall_k)
            mem = "\n".join(f"[mem] {s.text}" for s in recalled) or "(empty)"
            convo = _truncate([transcript[0],
                               {"role": "system", "content": f"Working memory:\n{mem}"}]
                              + transcript[1:], cfg.window)
        else:
            convo = _truncate(transcript, cfg.window)

        out = chat.chat(convo, tools=RepoTools.TOOLS_SCHEMA)
        usage = out.get("usage", {}) or {}
        call_cost = cost_usd(usage, price_in=cfg.price_in, price_out=cfg.price_out)
        cost += call_cost
        budget["spent"] += call_cost
        cached += cached_tokens(usage)
        if call_cost > cfg.cost_spike_usd:
            halted = "cost_spike"
            break

        tcs = out.get("tool_calls") or []
        if tcs:
            transcript.append({"role": "assistant", "content": out.get("content"),
                               "tool_calls": tcs})
            for tc in tcs:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = tools.dispatch(fn.get("name", ""), args)
                rjson = json.dumps(result)[:1500]
                transcript.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                                   "content": rjson})
                if session is not None:
                    session.remember(f"{fn.get('name')}({args}): {rjson}")
            continue
        transcript.append({"role": "assistant", "content": out.get("content") or "done"})
        break

    patch = tools.current_patch()
    if session is not None:
        session.close()
    return {
        "instance_id": inst["instance_id"],
        "model_name_or_path": MODEL_NAME if arm == "codepro" else f"{MODEL_NAME}-off",
        "model_patch": patch,
        "arm": arm,
        "cost_usd": round(cost, 6),
        "cached_tokens": cached,
        "tool_calls": tools.calls,
        "redundant_tool_calls": tools.redundant,
        "patch_nonempty": bool(patch.strip()),
        "halted": halted,
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_swe_eval.py -k run_instance -o addopts="" -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add bench/swe_eval.py tests/test_swe_eval.py
git commit -m "feat(bench): swe_eval run_instance — off truncation vs codepro engine recall, patch capture"
```

---

## Task 6: `swe_eval.py` — orchestrator (cap + resume + predictions JSONL)

**Files:**
- Modify: `bench/swe_eval.py`
- Test: `tests/test_swe_eval.py`

- [ ] **Step 1: Failing test (resume-skip + cap-halt)**

```python
# append to tests/test_swe_eval.py
from bench.swe_eval import run_swe_eval


def test_resume_skips_done_and_writes_predictions(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    insts = [{"instance_id":"syn__repo-1","repo":"syn/repo","base_commit":_base(repo),
              "problem_statement":"fix add"}]
    cfg = SweConfig(dry_run=True, arms=("off",), out_dir=tmp_path/"out",
                    work_dir=tmp_path/"wd")
    monkeypatch.setattr("bench.swe_eval._repo_url",
                        lambda inst: f"file://{repo.as_posix()}")
    run_swe_eval(cfg, instances=insts, chat_factory=lambda cfg: _PatchChat())
    pred = (tmp_path/"out"/"predictions_off.jsonl").read_text(encoding="utf-8").strip()
    assert "syn__repo-1" in pred
    r2 = run_swe_eval(cfg, instances=insts, chat_factory=lambda cfg: _PatchChat())
    assert r2["skipped"] >= 1


def test_global_cap_halts(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    insts = [{"instance_id":f"syn__repo-{i}","repo":"syn/repo","base_commit":_base(repo),
              "problem_statement":"fix add"} for i in range(1,4)]
    cfg = SweConfig(dry_run=True, arms=("off",), out_dir=tmp_path/"out2",
                    work_dir=tmp_path/"wd2", max_usd=0.0)  # cap already reached
    monkeypatch.setattr("bench.swe_eval._repo_url",
                        lambda inst: f"file://{repo.as_posix()}")
    r = run_swe_eval(cfg, instances=insts, chat_factory=lambda cfg: _PatchChat())
    assert r["halted"] == "budget_cap"
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_swe_eval.py -k "resume or cap" -o addopts="" -q`
Expected: FAIL — `ImportError: cannot import name 'run_swe_eval'`.

- [ ] **Step 3: Implement**

Append to `bench/swe_eval.py`:

```python
def _repo_url(inst: dict) -> str:
    """GitHub clone URL for an instance (monkeypatched to file:// in tests)."""
    return f"https://github.com/{inst['repo']}.git"


def _done_ids(path: Path) -> set[str]:
    """instance_ids already written to a predictions JSONL (resume support)."""
    if not path.exists():
        return set()
    ids = set()
    for ln in path.read_text(encoding="utf-8").splitlines():
        try:
            ids.add(json.loads(ln)["instance_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return ids


def _make_live_chat(cfg: SweConfig):
    from aether_context.local_llm import OpenAICompatLLM
    return OpenAICompatLLM(cfg.model, context_window=cfg.window)


def run_swe_eval(cfg: SweConfig, instances: Optional[list[dict]] = None,
                 chat_factory=None) -> dict:
    """Drive every (instance, arm), honoring the shared $ cap and skipping done work.
    Appends each record to runs/swe_eval/predictions_<arm>.jsonl. Resumable."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rows = instances if instances is not None else select_instances(load_lite(cfg), cfg.instances)
    chat_factory = chat_factory or (_make_live_chat if not cfg.dry_run else (lambda c: None))
    budget = {"spent": 0.0}
    summary: dict[str, Any] = {"arms": {a: {"done": 0, "patched": 0, "cost": 0.0}
                                        for a in cfg.arms},
                               "skipped": 0, "halted": None}

    for arm in cfg.arms:
        pred_path = cfg.out_dir / f"predictions_{arm}.jsonl"
        done = _done_ids(pred_path)
        chat = chat_factory(cfg)
        for inst in rows:
            if inst["instance_id"] in done:
                summary["skipped"] += 1
                continue
            if budget["spent"] >= cfg.max_usd:
                summary["halted"] = "budget_cap"
                break
            rec = run_instance(arm, inst, cfg, chat, budget, repo_url=_repo_url(inst))
            with pred_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            summary["arms"][arm]["done"] += 1
            summary["arms"][arm]["patched"] += int(rec["patch_nonempty"])
            summary["arms"][arm]["cost"] = round(
                summary["arms"][arm]["cost"] + rec["cost_usd"], 6)
            if rec.get("halted") in ("budget_cap", "cost_spike"):
                summary["halted"] = rec["halted"]
                break
        if summary["halted"]:
            break

    summary["total_cost_usd"] = round(budget["spent"], 6)
    (cfg.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_swe_eval.py -o addopts="" -q`
Expected: PASS (all swe_eval tests).

- [ ] **Step 5: Commit**

```bash
git add bench/swe_eval.py tests/test_swe_eval.py
git commit -m "feat(bench): swe_eval orchestrator — shared cap, resume-skip, predictions JSONL"
```

---

## Task 7: `swe_eval.py` — CLI

**Files:**
- Modify: `bench/swe_eval.py`
- Test: `tests/test_swe_eval.py`

- [ ] **Step 1: Failing test**

```python
# append to tests/test_swe_eval.py
from bench.swe_eval import _build_config


def test_cli_flags_map_to_config():
    cfg = _build_config(["--instances","5","--arms","off,codepro","--pool-gb","20",
                         "--turbovec-bits","8","--no-mpo-chain","--max-usd","25",
                         "--window","4096","--max-steps","12","--dry-run"])
    assert cfg.instances == 5
    assert cfg.arms == ("off","codepro")
    assert cfg.pool_gb == 20 and cfg.turbovec_bits == 8
    assert cfg.mpo_chain is False
    assert cfg.max_usd == 25.0 and cfg.window == 4096 and cfg.max_steps == 12
    assert cfg.dry_run is True
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_swe_eval.py -k cli_flags -o addopts="" -q`
Expected: FAIL — `ImportError: cannot import name '_build_config'`.

- [ ] **Step 3: Implement**

Append to `bench/swe_eval.py`:

```python
def _build_config(argv: Optional[list[str]] = None) -> SweConfig:
    p = argparse.ArgumentParser(prog="swe_eval",
                                description="SWE-bench codepro eval (generation phase).")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--arms", default="off,codepro")
    p.add_argument("--instances", type=int, default=0, help="0 = all lite (300); N = first N")
    p.add_argument("--window", type=int, default=8192)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--pool-gb", type=int, default=50)
    p.add_argument("--turbovec-bits", type=int, default=8, choices=(0, 4, 8))
    mpo = p.add_mutually_exclusive_group()
    mpo.add_argument("--mpo-chain", dest="mpo_chain", action="store_true", default=True)
    mpo.add_argument("--no-mpo-chain", dest="mpo_chain", action="store_false")
    p.add_argument("--recall-k", type=int, default=8)
    p.add_argument("--max-usd", type=float, default=25.0)
    p.add_argument("--out", default="runs/swe_eval")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    return SweConfig(
        model=a.model, arms=tuple(x.strip() for x in a.arms.split(",") if x.strip()),
        instances=a.instances, window=a.window, max_steps=a.max_steps, pool_gb=a.pool_gb,
        turbovec_bits=a.turbovec_bits, mpo_chain=a.mpo_chain, recall_k=a.recall_k,
        max_usd=a.max_usd, dry_run=a.dry_run, out_dir=Path(a.out),
        work_dir=Path(a.out) / "checkouts")


def main(argv: Optional[list[str]] = None) -> int:
    cfg = _build_config(argv)
    if not cfg.dry_run and not (os.environ.get("OPENROUTER_API_KEY")
                                or os.environ.get("OPENAI_API_KEY")):
        print("No OPENROUTER_API_KEY — use --dry-run to exercise the harness.")
        return 0
    summary = run_swe_eval(cfg)
    print(json.dumps(summary, indent=2))
    print(f"predictions -> {cfg.out_dir}/predictions_<arm>.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass + crash-proof dry-run**

Run: `python -m pytest tests/test_swe_eval.py -o addopts="" -q`
Expected: PASS (all).
Run: `python -m bench.swe_eval --dry-run --arms off`
Expected: prints a summary JSON; NO traceback. The synthetic `syn/repo` clone fails, so each
record carries `halted: checkout_error...` and an empty patch (handled in Task 5's
`run_instance` try/except). This proves the batch survives per-instance checkout failures.

- [ ] **Step 5: Commit**

```bash
git add bench/swe_eval.py tests/test_swe_eval.py
git commit -m "feat(bench): swe_eval CLI (off-vs-codepro, tuning flags, dry-run)"
```

---

## Task 8: `swe_scoring.py` — official swebench wrapper + report parse

**Files:**
- Create: `bench/swe_scoring.py`
- Test: `tests/test_swe_scoring.py`

- [ ] **Step 1: Failing test (parse a captured report; no Docker)**

```python
# tests/test_swe_scoring.py
import json
from pathlib import Path
from bench.swe_scoring import parse_report


def test_parse_report_extracts_resolved(tmp_path):
    report = {"total_instances": 3, "resolved_instances": 2,
              "resolved_ids": ["a-1", "a-2"], "unresolved_ids": ["a-3"]}
    p = tmp_path / "report.json"
    p.write_text(json.dumps(report), encoding="utf-8")
    out = parse_report(p)
    assert out["resolved"] == 2
    assert out["total"] == 3
    assert out["resolved_rate"] == round(2 / 3, 4)
    assert set(out["resolved_ids"]) == {"a-1", "a-2"}
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_swe_scoring.py -o addopts="" -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'bench.swe_scoring'`.

- [ ] **Step 3: Implement**

```python
# bench/swe_scoring.py
"""swe_scoring — Phase B wrapper. Runs the OFFICIAL swebench evaluation harness (Docker,
VPS5) over a predictions JSONL, then parses the resolved-rate report. Generation is
bench/swe_eval.py; this only scores. `parse_report` is pure + unit-tested (no Docker)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

DATASET = "princeton-nlp/SWE-bench_Lite"


def parse_report(report_path: Path) -> dict[str, Any]:
    """Normalize a swebench run report into {total, resolved, resolved_rate, ids}."""
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    total = int(data.get("total_instances") or data.get("total") or 0)
    resolved_ids = list(data.get("resolved_ids") or [])
    resolved = int(data.get("resolved_instances") or len(resolved_ids))
    rate = round(resolved / total, 4) if total else 0.0
    return {"total": total, "resolved": resolved, "resolved_rate": rate,
            "resolved_ids": resolved_ids,
            "unresolved_ids": list(data.get("unresolved_ids") or [])}


def run_evaluation(predictions_path: Path, run_id: str, *,
                   max_workers: int = 4, dataset: str = DATASET) -> list[str]:
    """Invoke the official swebench harness (Docker required). Returns the command run.
    On VPS5: `pip install swebench` first. The report JSON lands in the CWD as
    `<model>.<run_id>.json` per swebench's convention."""
    cmd = ["python", "-m", "swebench.harness.run_evaluation",
           "--dataset_name", dataset,
           "--predictions_path", str(predictions_path),
           "--run_id", run_id,
           "--max_workers", str(max_workers)]
    subprocess.run(cmd, check=True)
    return cmd
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_swe_scoring.py -o addopts="" -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add bench/swe_scoring.py tests/test_swe_scoring.py
git commit -m "feat(bench): swe_scoring — official swebench wrapper + pure report parser"
```

---

## Task 9: Full dry-run gate + runbook doc

**Files:**
- Create: `docs/benchmarks/2026-06-23-swe-codepro/RUNBOOK.md`

- [ ] **Step 1: Whole-suite green**

Run: `python -m pytest tests/test_swe_tools.py tests/test_swe_eval.py tests/test_swe_scoring.py -o addopts="" -q`
Expected: PASS (all).

- [ ] **Step 2: Write the runbook**

Create `docs/benchmarks/2026-06-23-swe-codepro/RUNBOOK.md`:

```markdown
# SWE-bench CodePro Eval — Runbook

## Phase A — generation (any host with OpenRouter key + git)
    export OPENROUTER_API_KEY=...
    # 1-instance smoke first:
    python -m bench.swe_eval --instances 1 --arms off,codepro --max-usd 25
    # full lite overnight:
    python -m bench.swe_eval --instances 0 --arms off,codepro --max-usd 25
    # -> runs/swe_eval/predictions_off.jsonl + predictions_codepro.jsonl (resumable; re-run to continue)

## Phase B — scoring (VPS5, Docker, Linux)
    ssh root@aether-vps5
    pip install swebench
    cd <repo>
    python -m swebench.harness.run_evaluation \
      --dataset_name princeton-nlp/SWE-bench_Lite \
      --predictions_path runs/swe_eval/predictions_off.jsonl --run_id off --max_workers 4
    python -m swebench.harness.run_evaluation \
      --dataset_name princeton-nlp/SWE-bench_Lite \
      --predictions_path runs/swe_eval/predictions_codepro.jsonl --run_id codepro --max_workers 4
    # -> <model>.off.json / <model>.codepro.json reports

## Headline
    python - <<'PY'
    from pathlib import Path
    from bench.swe_scoring import parse_report
    for arm in ("off", "codepro"):
        r = parse_report(next(Path('.').glob(f'*.{arm}.json')))
        print(arm, r["resolved"], "/", r["total"], "=", r["resolved_rate"])
    PY

## Live tuning knobs (re-run a subset, diff resolved-rate/cost)
    --pool-gb --turbovec-bits {0,4,8} --no-mpo-chain --recall-k --max-steps --window --instances N

## Notes
- 1-instance smoke MUST pass Phase B (Docker build + test run) before the overnight 300.
- $25 hard cap shared across arms+instances; halts clean + resumes (re-run same command).
- codepro arm uses the LOCAL engine (no VPS5 atlas oracle) -> Docker CPU/disk only.
```

- [ ] **Step 3: Commit**

```bash
git add docs/benchmarks/2026-06-23-swe-codepro/RUNBOOK.md
git commit -m "docs(bench): SWE-bench codepro eval runbook (generation + VPS5 scoring + tuning)"
```

---

## Self-Review notes (spec coverage)

- Two-phase (gen ours / score official): Tasks 5–8. ✓
- off vs codepro engine-MAX (overpool+turbovec8+chain): `run_instance` Task 5 (`pool_quantize`, `mpo_chain`). ✓
- Reuse api_eval loop primitives: imports `cost_usd`/`cached_tokens`; loop pattern mirrored. ✓
- Repo file tools + diff capture: Tasks 1–3. ✓
- $25 cap + resume: Task 6. ✓
- dry-run + unit tests: Tasks 1–8; full gate Task 9. ✓
- 1-instance smoke before 300 + VPS5 Docker: RUNBOOK Task 9. ✓
- Live-tuning knobs: CLI Task 7 + RUNBOOK. ✓
- RESULTS.md: generated post-run from reports (RUNBOOK headline snippet); the formatted doc
  mirrors `docs/benchmarks/.../2026-06-14-deepseek-v4-pro/RESULTS.md` after the run lands.
