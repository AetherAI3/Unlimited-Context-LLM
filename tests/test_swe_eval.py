import subprocess
from pathlib import Path

from bench.swe_eval import (SweConfig, _build_config, _extract_diff, run_instance,
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


_DIFF = ("diff --git a/a.py b/a.py\n"
         "--- a/a.py\n+++ b/a.py\n"
         "@@ -1,2 +1,2 @@\n def add(x, y):\n"
         "-    return x - y  # bug\n+    return x + y\n")


class _DiffChat:
    """Investigation turns (tools set) -> 'ready' no tool call; patch turn (tools=None)
    -> a fenced unified diff. Mirrors how dsv4-pro is driven in the two-phase loop."""
    def __init__(self, *_a, **_k):
        pass

    def chat(self, messages, tools=None, *, max_tokens=None):
        usage = {"prompt_tokens": 50, "completion_tokens": 10}
        if tools is None:  # forced patch turn
            return {"content": "```diff\n" + _DIFF + "```", "usage": usage, "tool_calls": []}
        return {"content": "ready", "usage": usage, "tool_calls": []}  # stop investigating


class _NoDiffChat:
    """Never emits a diff (prose only) -> empty patch."""
    def __init__(self, *_a, **_k):
        pass

    def chat(self, messages, tools=None, *, max_tokens=None):
        usage = {"prompt_tokens": 50, "completion_tokens": 10}
        return {"content": "I think the bug is in a.py somewhere.", "usage": usage,
                "tool_calls": []}


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


def _inst(repo):
    return {"instance_id": "syn__repo-1", "repo": "syn/repo",
            "base_commit": _base(repo), "problem_statement": "fix add"}


def test_extract_diff_from_fenced_block():
    out = _extract_diff("Here is the fix:\n```diff\n" + _DIFF + "```\nDone.")
    assert out.startswith("diff --git a/a.py")
    assert "+    return x + y" in out


def test_extract_diff_empty_when_no_diff():
    assert _extract_diff("I could not find the bug.") == ""


def test_run_instance_captures_diff_off_arm(tmp_path):
    repo = _make_repo(tmp_path)
    cfg = SweConfig(dry_run=True, work_dir=tmp_path / "wd", out_dir=tmp_path / "out")
    rec = run_instance("off", _inst(repo), cfg, _DiffChat(), {"spent": 0.0},
                       repo_url=f"file://{repo.as_posix()}")
    assert rec["arm"] == "off"
    assert rec["patch_nonempty"] is True
    assert "diff --git a/a.py" in rec["model_patch"]
    assert "+    return x + y" in rec["model_patch"]


def test_run_instance_codepro_arm_uses_engine(tmp_path):
    repo = _make_repo(tmp_path)
    cfg = SweConfig(dry_run=True, work_dir=tmp_path / "wd", out_dir=tmp_path / "out",
                    pool_gb=5)  # 5 GB = engine pool floor
    rec = run_instance("codepro", _inst(repo), cfg, _DiffChat(), {"spent": 0.0},
                       repo_url=f"file://{repo.as_posix()}")
    assert rec["arm"] == "codepro"
    assert rec["patch_nonempty"] is True
    assert "+    return x + y" in rec["model_patch"]


def test_run_instance_no_diff_is_empty_patch(tmp_path):
    repo = _make_repo(tmp_path)
    cfg = SweConfig(dry_run=True, work_dir=tmp_path / "wd", out_dir=tmp_path / "out")
    rec = run_instance("off", _inst(repo), cfg, _NoDiffChat(), {"spent": 0.0},
                       repo_url=f"file://{repo.as_posix()}")
    assert rec["patch_nonempty"] is False
    assert rec["model_patch"] == ""


def test_resume_skips_done_and_writes_predictions(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    insts = [_inst(repo)]
    cfg = SweConfig(dry_run=True, arms=("off",), out_dir=tmp_path / "out",
                    work_dir=tmp_path / "wd")
    monkeypatch.setattr("bench.swe_eval._repo_url",
                        lambda inst: f"file://{repo.as_posix()}")
    run_swe_eval(cfg, instances=insts, chat_factory=lambda cfg: _DiffChat())
    pred = (tmp_path / "out" / "predictions_off.jsonl").read_text(encoding="utf-8").strip()
    assert "syn__repo-1" in pred
    r2 = run_swe_eval(cfg, instances=insts, chat_factory=lambda cfg: _DiffChat())
    assert r2["skipped"] >= 1


def test_global_cap_halts(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    insts = [{"instance_id": f"syn__repo-{i}", "repo": "syn/repo", "base_commit": _base(repo),
              "problem_statement": "fix add"} for i in range(1, 4)]
    cfg = SweConfig(dry_run=True, arms=("off",), out_dir=tmp_path / "out2",
                    work_dir=tmp_path / "wd2", max_usd=0.0)  # cap already reached
    monkeypatch.setattr("bench.swe_eval._repo_url",
                        lambda inst: f"file://{repo.as_posix()}")
    r = run_swe_eval(cfg, instances=insts, chat_factory=lambda cfg: _DiffChat())
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
