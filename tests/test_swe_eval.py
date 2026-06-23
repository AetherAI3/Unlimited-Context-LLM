import json as _json
import subprocess
from pathlib import Path

from bench.swe_eval import (SweConfig, _build_config, run_instance,
                            run_swe_eval, select_instances)


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


class _PatchChat:
    """Two steps: (1) call edit_file to fix the bug, (2) emit a final message."""
    def __init__(self, *_a, **_k):
        self._t = 0

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
    d = tmp_path / "repo"
    d.mkdir()
    for a in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", *a], cwd=d, check=True, capture_output=True)
    (d / "a.py").write_text("def add(x, y):\n    return x - y  # bug\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=d, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=d, check=True, capture_output=True)
    return d


def _base(repo):
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                          capture_output=True, text=True).stdout.strip()


class _LazyChat:
    """Step 1: 'done' with NO tool call (empty patch). After pushback, step 2 edits."""
    def __init__(self, *_a, **_k):
        self._t = 0

    def chat(self, messages, tools=None, *, max_tokens=None):
        self._t += 1
        usage = {"prompt_tokens": 50, "completion_tokens": 10}
        if self._t == 1:
            return {"content": "I analyzed it, the fix is obvious. Done.",
                    "usage": usage, "tool_calls": []}
        if self._t == 2:
            return {"content": None, "usage": usage, "tool_calls": [
                {"id": "c1", "type": "function", "function": {
                    "name": "edit_file",
                    "arguments": _json.dumps({"path": "a.py",
                                              "old": "return x - y  # bug",
                                              "new": "return x + y"})}}]}
        return {"content": "done", "usage": usage, "tool_calls": []}


def test_finish_without_edit_is_rejected_until_patch_exists(tmp_path):
    repo = _make_repo(tmp_path)
    cfg = SweConfig(dry_run=True, work_dir=tmp_path / "wd", out_dir=tmp_path / "out",
                    max_steps=10)
    inst = {"instance_id": "syn__repo-1", "repo": "syn/repo",
            "base_commit": _base(repo), "problem_statement": "fix add"}
    rec = run_instance("off", inst, cfg, _LazyChat(), {"spent": 0.0},
                       repo_url=f"file://{repo.as_posix()}")
    # The first 'done' (no edit) was rejected; the model then edited -> patch lands.
    assert rec["patch_nonempty"] is True
    assert "+    return x + y" in rec["model_patch"]


def test_run_instance_produces_patch_off_arm(tmp_path):
    repo = _make_repo(tmp_path)
    cfg = SweConfig(dry_run=True, work_dir=tmp_path / "wd")
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
    cfg = SweConfig(dry_run=True, work_dir=tmp_path / "wd", out_dir=tmp_path / "out",
                    pool_gb=5)  # 5 GB = engine pool floor
    inst = {"instance_id": "syn__repo-1", "repo": "syn/repo",
            "base_commit": _base(repo), "problem_statement": "fix add"}
    budget = {"spent": 0.0}
    rec = run_instance("codepro", inst, cfg, _PatchChat(), budget,
                       repo_url=f"file://{repo.as_posix()}")
    assert "+    return x + y" in rec["model_patch"]
    assert rec["arm"] == "codepro"


def test_resume_skips_done_and_writes_predictions(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    insts = [{"instance_id": "syn__repo-1", "repo": "syn/repo", "base_commit": _base(repo),
              "problem_statement": "fix add"}]
    cfg = SweConfig(dry_run=True, arms=("off",), out_dir=tmp_path / "out",
                    work_dir=tmp_path / "wd")
    monkeypatch.setattr("bench.swe_eval._repo_url",
                        lambda inst: f"file://{repo.as_posix()}")
    run_swe_eval(cfg, instances=insts, chat_factory=lambda cfg: _PatchChat())
    pred = (tmp_path / "out" / "predictions_off.jsonl").read_text(encoding="utf-8").strip()
    assert "syn__repo-1" in pred
    r2 = run_swe_eval(cfg, instances=insts, chat_factory=lambda cfg: _PatchChat())
    assert r2["skipped"] >= 1


def test_global_cap_halts(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    insts = [{"instance_id": f"syn__repo-{i}", "repo": "syn/repo", "base_commit": _base(repo),
              "problem_statement": "fix add"} for i in range(1, 4)]
    cfg = SweConfig(dry_run=True, arms=("off",), out_dir=tmp_path / "out2",
                    work_dir=tmp_path / "wd2", max_usd=0.0)  # cap already reached
    monkeypatch.setattr("bench.swe_eval._repo_url",
                        lambda inst: f"file://{repo.as_posix()}")
    r = run_swe_eval(cfg, instances=insts, chat_factory=lambda cfg: _PatchChat())
    assert r["halted"] == "budget_cap"


def test_cli_flags_map_to_config():
    cfg = _build_config(["--instances", "5", "--arms", "off,codepro", "--pool-gb", "20",
                         "--turbovec-bits", "8", "--no-mpo-chain", "--max-usd", "25",
                         "--window", "4096", "--max-steps", "12", "--dry-run"])
    assert cfg.instances == 5
    assert cfg.arms == ("off", "codepro")
    assert cfg.pool_gb == 20 and cfg.turbovec_bits == 8
    assert cfg.mpo_chain is False
    assert cfg.max_usd == 25.0 and cfg.window == 4096 and cfg.max_steps == 12
    assert cfg.dry_run is True
