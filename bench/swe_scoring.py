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
