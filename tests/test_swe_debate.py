from bench.swe_debate import classify_complexity


def test_heavy_on_hard_keywords():
    assert classify_complexity("There is a security vulnerability in the auth check") == "heavy"
    assert classify_complexity("intermittent race condition causes a deadlock") == "heavy"


def test_heavy_on_long_problem():
    assert classify_complexity("x " * 400) == "heavy"


def test_light_on_short_simple():
    assert classify_complexity("Fix typo in the docstring of add()") == "light"


def test_medium_default():
    assert classify_complexity("Update the parser to handle empty input lists") == "medium"


import json
import subprocess
from bench.swe_debate import debate_patch, _parse_verdict


def test_parse_verdict_tolerant():
    assert _parse_verdict('{"accept": true, "objections": []}')["accept"] is True
    v = _parse_verdict('junk {"accept": false, "objections": ["wrong file"]} tail')
    assert v["accept"] is False and v["objections"] == ["wrong file"]
    assert _parse_verdict("not json")["accept"] is False


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
    def __init__(self):
        self.props = 0
    def chat(self, messages, tools=None, *, max_tokens=None):
        u = messages[-1]["content"]
        usage = {"prompt_tokens": 20, "completion_tokens": 10}
        if "CRITIQUE" in u:
            ok = "return x + y" in u
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
    assert r["applied"] >= 1
