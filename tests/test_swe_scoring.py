import json

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


def test_parse_report_zero_total_no_div_by_zero(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"total_instances": 0}), encoding="utf-8")
    out = parse_report(p)
    assert out["resolved_rate"] == 0.0
    assert out["resolved"] == 0
