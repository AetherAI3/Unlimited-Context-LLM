# Complexity-Gated Propose↔Critique Reasoning Escalation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On hard SWE-bench instances, escalate the agent from its flat read→submit loop into a propose↔critique debate (grounded by the engine, win locked into memory) to lift resolved-rate beyond ~2×.

**Architecture:** New `bench/swe_debate.py` holds a pure complexity gate + a propose↔critique loop. `bench/swe_eval.py` gets a flag-gated `codepro_debate` arm: investigate as today, classify the instance, and on `heavy` produce the patch via the debate (else the normal flat submit). Flag-off ⇒ byte-identical to current codepro.

**Tech Stack:** Python stdlib + the existing bench (`swe_eval`, `swe_tools`, `aether_context.Session`). Same OpenAI-compat `chat(messages, tools, max_tokens)` adapter. No new deps. Dry-run testable with mock chat.

**Branch:** `feat/reasoning-escalation`.

**Reused symbols (do NOT redefine):**
- `bench/swe_eval.py`: `_parse_blocks(text) -> list[(path,search,replace)]`, `SweConfig`, `run_instance`, `_make_live_chat`.
- `bench/swe_tools.py`: `RepoTools.edit_file(path,old,new) -> dict` (whitespace-tolerant), `.current_patch() -> str`, `.read_file(path)`, `._read` (set of read paths).
- Chat contract: `chat.chat(messages, tools=None, *, max_tokens=None) -> {"content","tool_calls","usage"}`.

**Import discipline (avoid circular import):** `swe_debate.py` may `from bench.swe_eval import _parse_blocks` at top. `swe_eval.py` imports `swe_debate` **lazily inside `run_instance`** (function-level), never at module top.

---

## Task 1: complexity gate

**Files:**
- Create: `bench/swe_debate.py`
- Test: `tests/test_swe_debate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_swe_debate.py
from bench.swe_debate import classify_complexity


def test_heavy_on_hard_keywords():
    assert classify_complexity("There is a security vulnerability in the auth check") == "heavy"
    assert classify_complexity("intermittent race condition causes a deadlock") == "heavy"


def test_heavy_on_long_problem():
    assert classify_complexity("x " * 400) == "heavy"  # very long statement


def test_light_on_short_simple():
    assert classify_complexity("Fix typo in the docstring of add()") == "light"


def test_medium_default():
    assert classify_complexity("Update the parser to handle empty input lists") == "medium"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_swe_debate.py -o addopts="" -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'bench.swe_debate'`.

- [ ] **Step 3: Implement**

```python
# bench/swe_debate.py
"""Complexity-gated propose↔critique reasoning escalation for the SWE-bench agent.

classify_complexity: cheap, deterministic gate (keywords + length). debate_patch: a
propose↔critique loop that drafts a fix, has a critic attack it, revises against the
objections (re-grounding each round), and applies the best candidate. Both reusable +
unit-testable with a mock chat; no live API required.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

# Words that mark a genuinely hard instance — the kind a flat agent drops.
_HEAVY_RE = re.compile(
    r"\b(security|vulnerab|exploit|race condition|deadlock|concurren|"
    r"intermittent|edge case|regression|segfault|corrupt|workaround|"
    r"thread.?safe|memory leak|undefined behaviou?r)\b", re.I)
_HEAVY_LEN = 600   # problem statements this long (chars) are treated as heavy
_LIGHT_LEN = 120   # short + no hard keyword => light


def classify_complexity(problem: str, files_seen: int = 0) -> str:
    """Return 'light' | 'medium' | 'heavy'. Deterministic: hard keyword or a long/broad
    statement => heavy; short + simple => light; otherwise medium. `files_seen` (distinct
    files touched during investigation) nudges toward heavy when many are involved."""
    p = problem or ""
    if _HEAVY_RE.search(p) or len(p) >= _HEAVY_LEN or files_seen >= 4:
        return "heavy"
    if len(p) <= _LIGHT_LEN and files_seen <= 1:
        return "light"
    return "medium"
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_swe_debate.py -o addopts="" -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add bench/swe_debate.py tests/test_swe_debate.py
git commit -m "feat(bench): complexity gate (classify_complexity) for reasoning escalation"
```

---

## Task 2: propose↔critique debate loop

**Files:**
- Modify: `bench/swe_debate.py`
- Test: `tests/test_swe_debate.py`

- [ ] **Step 1: Add failing tests**

```python
# append to tests/test_swe_debate.py
import json
import subprocess
from bench.swe_debate import debate_patch, _parse_verdict


def test_parse_verdict_tolerant():
    assert _parse_verdict('{"accept": true, "objections": []}')["accept"] is True
    v = _parse_verdict('junk {"accept": false, "objections": ["wrong file"]} tail')
    assert v["accept"] is False and v["objections"] == ["wrong file"]
    assert _parse_verdict("not json")["accept"] is False  # unparseable -> reject (keep iterating)


def _repo(tmp_path):
    d = tmp_path / "r"; d.mkdir()
    for a in (["init","-q"],["config","user.email","t@t"],["config","user.name","t"]):
        subprocess.run(["git",*a], cwd=d, check=True, capture_output=True)
    (d/"a.py").write_text("def add(x, y):\n    return x - y  # bug\n", encoding="utf-8")
    subprocess.run(["git","add","-A"], cwd=d, check=True, capture_output=True)
    subprocess.run(["git","commit","-qm","b"], cwd=d, check=True, capture_output=True)
    return d


_SR_BAD = ("a.py\n<<<<<<< SEARCH\n    return x - y  # bug\n=======\n    return x * y\n>>>>>>> REPLACE\n")
_SR_GOOD = ("a.py\n<<<<<<< SEARCH\n    return x - y  # bug\n=======\n    return x + y\n>>>>>>> REPLACE\n")


class _DebateChat:
    """propose#1 -> bad fix; critic#1 -> reject; propose#2 -> good fix; critic#2 -> accept.
    Distinguishes propose vs critique by a marker in the last user message."""
    def __init__(self):
        self.props = 0
    def chat(self, messages, tools=None, *, max_tokens=None):
        u = messages[-1]["content"]
        usage = {"prompt_tokens": 20, "completion_tokens": 10}
        if "CRITIQUE" in u:
            ok = "return x + y" in u  # the proposed patch text is echoed into the critique prompt
            verdict = {"accept": bool(ok), "objections": [] if ok else ["wrong operator"]}
            return {"content": json.dumps(verdict), "usage": usage, "tool_calls": []}
        self.props += 1
        return {"content": _SR_BAD if self.props == 1 else _SR_GOOD, "usage": usage, "tool_calls": []}


def test_debate_revises_until_accept_and_applies(tmp_path):
    from bench.swe_tools import RepoTools
    t = RepoTools(_repo(tmp_path))
    r = debate_patch(_DebateChat(), t, ground_fn=lambda q: "", problem="fix add",
                     rounds=3, max_output_tokens=512)
    assert r["accepted"] is True
    assert r["applied"] >= 1
    assert "+    return x + y" in t.current_patch()


def test_debate_round_cap_submits_best(tmp_path):
    from bench.swe_tools import RepoTools
    class _AlwaysReject(_DebateChat):
        def chat(self, messages, tools=None, *, max_tokens=None):
            u = messages[-1]["content"]; usage = {"prompt_tokens":20,"completion_tokens":10}
            if "CRITIQUE" in u:
                return {"content": json.dumps({"accept": False, "objections": ["nope"]}),
                        "usage": usage, "tool_calls": []}
            return {"content": _SR_GOOD, "usage": usage, "tool_calls": []}
    t = RepoTools(_repo(tmp_path))
    r = debate_patch(_AlwaysReject(), t, ground_fn=lambda q: "", problem="fix add", rounds=2)
    assert r["accepted"] is False
    assert r["applied"] >= 1  # best candidate still applied
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_swe_debate.py -k "verdict or debate" -o addopts="" -q`
Expected: FAIL — `ImportError: cannot import name 'debate_patch'`.

- [ ] **Step 3: Implement**

Append to `bench/swe_debate.py`:

```python
from bench.swe_eval import _parse_blocks  # safe: swe_eval imports swe_debate only lazily

_PROPOSE_SYS = (
    "You are the PROPOSER. Produce a minimal fix as SEARCH/REPLACE edit blocks, EXACTLY:\n"
    "<path>\n<<<<<<< SEARCH\n<exact current lines>\n=======\n<replacement>\n>>>>>>> REPLACE\n"
    "Output ONLY the blocks. Use the grounded context to get SEARCH text right.")
_CRITIQUE_SYS = (
    "You are the CRITIC. Attack the proposed fix: does it address the FAILING behaviour? "
    "Wrong file/location? Missed edge case? Will the repo's tests still fail? "
    "Reply JSON ONLY: {\"accept\": true|false, \"objections\": [\"...\"]}. Accept only if the "
    "fix is correct and complete.")


def _parse_verdict(text: str) -> dict:
    """Tolerant parse of the critic's JSON verdict; unparseable => reject (keep iterating)."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return {"accept": bool(d.get("accept")), "objections": list(d.get("objections") or [])}
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return {"accept": False, "objections": ["unparseable verdict"]}


def debate_patch(chat: Any, tools: Any, ground_fn: Callable[[str], str], problem: str, *,
                 rounds: int = 2, max_output_tokens: int = 4096,
                 account: Optional[Callable[[dict], bool]] = None) -> dict:
    """Propose↔critique loop. The proposer drafts SEARCH/REPLACE edits (grounded per round on
    the current focus); the critic accepts/rejects with objections; on reject the proposer
    revises against them and re-grounds. Applies the latest candidate via tools.edit_file each
    round (so current_patch() reflects the best attempt). Returns
    {accepted, applied, rounds_used}. `account(out)` (optional) tallies cost; if it returns
    False (cost spike) the loop stops."""
    focus = problem
    applied_total = 0
    accepted = False
    used = 0
    for used in range(1, rounds + 1):
        ground = ground_fn(focus) or ""
        prop_msgs = [{"role": "system", "content": _PROPOSE_SYS},
                     {"role": "user", "content": f"Problem:\n{problem}\n\nContext:\n{ground}\n\nPropose the fix."}]
        out = chat.chat(prop_msgs, tools=None, max_tokens=max_output_tokens)
        if account is not None and not account(out):
            break
        applied_here = 0
        for path, search, replace in _parse_blocks(out.get("content") or ""):
            res = tools.edit_file(path, search, replace)
            if isinstance(res, dict) and res.get("ok"):
                applied_here += 1
                applied_total += 1
        patch = tools.current_patch()
        crit_msgs = [{"role": "system", "content": _CRITIQUE_SYS},
                     {"role": "user", "content": f"CRITIQUE this fix for:\n{problem}\n\nProposed patch:\n{patch}"}]
        out = chat.chat(crit_msgs, tools=None, max_tokens=max_output_tokens)
        if account is not None and not account(out):
            break
        verdict = _parse_verdict(out.get("content") or "")
        if verdict["accept"] and applied_here:
            accepted = True
            break
        focus = problem + "\n\nFix these objections: " + "; ".join(verdict["objections"])
    return {"accepted": accepted, "applied": applied_total, "rounds_used": used}
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_swe_debate.py -o addopts="" -q`
Expected: PASS (all swe_debate tests).

- [ ] **Step 5: Commit**

```bash
git add bench/swe_debate.py tests/test_swe_debate.py
git commit -m "feat(bench): propose↔critique debate loop (per-round ground, apply best candidate)"
```

---

## Task 3: wire the `codepro_debate` arm into swe_eval

**Files:**
- Modify: `bench/swe_eval.py` (SweConfig ~line 44; session build ~line 204; gate replaces `patch = tools.current_patch()` ~line 336; return dict)
- Test: `tests/test_swe_eval.py`

- [ ] **Step 1: Add failing test**

```python
# append to tests/test_swe_eval.py
def test_codepro_debate_heavy_routes_to_debate(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    monkeypatch.setattr("bench.swe_debate.classify_complexity", lambda *a, **k: "heavy")
    def _fake_debate(chat, tools, ground_fn, problem, **kw):
        tools.edit_file("a.py", "    return x - y  # bug", "    return x + y")
        return {"accepted": True, "applied": 1, "rounds_used": 1}
    monkeypatch.setattr("bench.swe_debate.debate_patch", _fake_debate)
    cfg = SweConfig(dry_run=True, arms=("codepro_debate",), out_dir=tmp_path/"o",
                    work_dir=tmp_path/"w", pool_gb=5, debate=True)
    rec = run_instance("codepro_debate", _inst(repo), cfg, _SRChat(), {"spent":0.0},
                       repo_url=f"file://{repo.as_posix()}")
    assert rec["arm"] == "codepro_debate"
    assert rec["patch_nonempty"] is True
    assert rec.get("complexity") == "heavy"
    assert "+    return x + y" in rec["model_patch"]


def test_codepro_debate_flag_off_no_debate(tmp_path):
    # debate=False -> codepro_debate behaves like plain codepro (no debate call), no crash
    repo = _make_repo(tmp_path)
    cfg = SweConfig(dry_run=True, arms=("codepro_debate",), out_dir=tmp_path/"o2",
                    work_dir=tmp_path/"w2", pool_gb=5, debate=False)
    rec = run_instance("codepro_debate", _inst(repo), cfg, _SRChat(), {"spent":0.0},
                       repo_url=f"file://{repo.as_posix()}")
    assert rec["arm"] == "codepro_debate"
    assert rec.get("complexity") is None  # gate not run when debate flag off
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_swe_eval.py -k codepro_debate -o addopts="" -q`
Expected: FAIL — `TypeError: SweConfig.__init__() got an unexpected keyword argument 'debate'`.

- [ ] **Step 3: Implement — SweConfig flags**

In `bench/swe_eval.py`, after the `max_output_tokens` field add:

```python
    debate: bool = False          # codepro_debate arm: gate hard instances into propose↔critique
    debate_rounds: int = 2        # max propose↔critique rounds on a heavy instance
```

- [ ] **Step 4: Implement — build the engine for the debate arm**

Change the session-build condition (line ~204) from `if arm == "codepro":` to:

```python
    if arm in ("codepro", "codepro_debate"):
```

- [ ] **Step 5: Implement — gate + debate**

Replace the single line `patch = tools.current_patch()` (line ~336, just before the return dict) with:

```python
    # ── Complexity gate: heavy + codepro_debate -> propose↔critique escalation ──
    complexity = None
    if arm == "codepro_debate" and cfg.debate:
        from bench import swe_debate as _dbg  # lazy import (no circular dependency)
        complexity = _dbg.classify_complexity(inst["problem_statement"], len(tools._read))
        if complexity == "heavy" and not tools.current_patch().strip() and budget["spent"] < cfg.max_usd:
            def _ground(q: str) -> str:
                if session is None:
                    return ""
                qv = encoder.encode(q)
                hits = session._cold_retrieve(session._key(), qv, cfg.recall_k)
                return "\n".join(f"[mem] {s.text}" for s in hits)
            _dbg.debate_patch(chat, tools, _ground, inst["problem_statement"],
                              rounds=cfg.debate_rounds, max_output_tokens=cfg.max_output_tokens,
                              account=_account)
    patch = tools.current_patch()
```

- [ ] **Step 6: Implement — record complexity**

In the `run_instance` return dict (the `return { "instance_id": ... }` block), add this key after `"halted": halted,`:

```python
        "complexity": complexity,
```

- [ ] **Step 7: Run to verify pass**

Run: `python -m pytest tests/test_swe_eval.py -o addopts="" -q`
Expected: PASS (all, incl new codepro_debate tests).

- [ ] **Step 8: Commit**

```bash
git add bench/swe_eval.py tests/test_swe_eval.py
git commit -m "feat(bench): codepro_debate arm — gate heavy instances into the debate (flag-gated)"
```

---

## Task 4: CLI flags + arm

**Files:**
- Modify: `bench/swe_eval.py` (`_build_config`)
- Test: `tests/test_swe_eval.py`

- [ ] **Step 1: Add failing test**

```python
# append to tests/test_swe_eval.py
def test_cli_debate_flags():
    cfg = _build_config(["--arms","codepro_debate","--debate","--debate-rounds","3","--dry-run"])
    assert cfg.arms == ("codepro_debate",)
    assert cfg.debate is True
    assert cfg.debate_rounds == 3
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_swe_eval.py -k cli_debate -o addopts="" -q`
Expected: FAIL — argparse rejects `--debate`.

- [ ] **Step 3: Implement**

In `_build_config`, after the `--atlas-ground` argument add:

```python
    p.add_argument("--debate", action="store_true",
                   help="codepro_debate arm: gate heavy instances into a propose↔critique loop")
    p.add_argument("--debate-rounds", type=int, default=2)
```

And in the returned `SweConfig(...)`, add (alongside `atlas_ground=a.atlas_ground,`):

```python
        debate=a.debate, debate_rounds=a.debate_rounds,
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_swe_eval.py -o addopts="" -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add bench/swe_eval.py tests/test_swe_eval.py
git commit -m "feat(bench): --debate / --debate-rounds CLI for codepro_debate"
```

---

## Task 5: full gate + runbook note

**Files:**
- Modify: `docs/benchmarks/2026-06-23-swe-codepro/RUNBOOK.md`

- [ ] **Step 1: Whole suite green**

Run: `python -m pytest tests/test_swe_debate.py tests/test_swe_eval.py tests/test_swe_tools.py tests/test_swe_scoring.py -o addopts="" -q`
Expected: PASS (all).

- [ ] **Step 2: Dry-run the new arm end to end (no API)**

Run: `python -m bench.swe_eval --dry-run --arms codepro_debate --debate`
Expected: prints a summary JSON, no traceback (synthetic instances; checkout-error records are fine).

- [ ] **Step 3: Append runbook section**

Add to `docs/benchmarks/2026-06-23-swe-codepro/RUNBOOK.md`:

```markdown
## Reasoning escalation (codepro_debate)

    # 3-arm comparison; heavy instances escalate into propose↔critique
    python -m bench.swe_eval --instances 0 --arms off,codepro,codepro_debate \
      --debate --debate-rounds 2 --max-output-tokens 4096 --max-usd 35 --out runs/debate

Score all three arms (VPS5) as usual. Headline metric: resolved-rate on the **heavy** subset
(`complexity == "heavy"` in predictions) — codepro_debate vs codepro. Also track escalation rate
(% heavy) and extra $/heavy instance.
```

- [ ] **Step 4: Commit**

```bash
git add docs/benchmarks/2026-06-23-swe-codepro/RUNBOOK.md
git commit -m "docs(bench): runbook — codepro_debate 3-arm run + heavy-subset metric"
```

---

## Self-Review (spec coverage)

- Complexity gate (Arbiter light/medium/heavy parity): Task 1. ✓
- Propose↔critique debate, per-round grounding, apply best candidate, round cap: Task 2. ✓
- MPO-lock: debate runs against the live `Session` (built for codepro_debate); read results were
  already `session.remember`'d during the flat investigation, and `_ground` recalls them →
  within-run compounding. (Explicit post-resolve win-record is a follow-on; not needed for v1.) ✓
- `codepro_debate` arm + flag-off byte-identical: Task 3 (gated on `arm=="codepro_debate" and
  cfg.debate`; the session-build change is inert for other arms; `complexity` is None when off). ✓
- CLI + record `complexity` for the heavy-subset metric: Tasks 3, 4. ✓
- Dry-run + unit tests, no live API: Tasks 1–5. ✓
- GLM-5.2 / atlas write-back OUT of scope: not in any task. ✓

**Type consistency:** `classify_complexity(problem, files_seen=0)->str`;
`debate_patch(chat,tools,ground_fn,problem,*,rounds,max_output_tokens,account)->{accepted,applied,rounds_used}`;
`_parse_verdict(text)->{accept,objections}`; SweConfig `debate`/`debate_rounds`; record key
`complexity`. Consistent across tasks.
