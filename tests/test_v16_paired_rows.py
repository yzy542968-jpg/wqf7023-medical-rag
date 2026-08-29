from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_rows(path: Path, offset: float) -> None:
    rows = []
    for case_id in ("CXR1", "CXR2"):
        for question_type in ("findings", "impression"):
            for condition in ("no_history", "retrieved_history", "random_history"):
                rows.append({
                    "case_id": case_id,
                    "question_type": question_type,
                    "condition": condition,
                    "token_f1": 0.2 + offset,
                    "answer_only_contract_valid": 1.0,
                    "evidence_provenance_valid": 1.0,
                    "hit_token_ceiling": 0.0,
                    "input_tokens": 10,
                    "output_tokens": 5,
                })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_confirmation_scope_and_spectrum_manifest(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "summary.json"
    write_rows(left, 0.1)
    write_rows(right, 0.0)
    manifest.write_text(
        json.dumps({"case_id": "CXR1", "spectrum": "normal"}) + "\n"
        + json.dumps({"case_id": "CXR2", "spectrum": "abnormal"}) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluate_v16_paired_rows.py"),
            "--left-rows", str(left),
            "--right-rows", str(right),
            "--left-label", "route",
            "--right-label", "base",
            "--manifest", str(manifest),
            "--evaluation-scope", "confirmation",
            "--bootstrap-iterations", "20",
            "--output", str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["evaluation_scope"] == "confirmation"
    assert summary["spectrum_sensitivity"]["normal"]["case_count"] == 1
    assert summary["route_minus_base"]["token_f1"]["retrieved_history"]["mean_difference"] > 0
