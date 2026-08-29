from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_confirmation_manifest_requires_complete_two_question_matrix(tmp_path: Path) -> None:
    rankings = tmp_path / "rankings.jsonl"
    output = tmp_path / "manifest.jsonl"
    rows = [
        {
            "case_id": "CXR1",
            "question_type": question_type,
            "spectrum": "normal",
            "rankings": {"rrf_lambdamart": ["CXR2", "CXR3", "CXR4"]},
        }
        for question_type in ("findings", "impression")
    ]
    rankings.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_v16_confirmation_manifest.py"),
            "--ranking-rows",
            str(rankings),
            "--output",
            str(output),
            "--expected-cases",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    manifest = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert manifest[0]["case_id"] == "CXR1"
    assert manifest[0]["selection_policy"] == "all_technically_eligible_v10_test_cases"
    summary = json.loads(output.with_suffix(".jsonl.summary.json").read_text(encoding="utf-8"))
    assert summary["case_count"] == 1


def test_confirmation_manifest_rejects_missing_question_type(tmp_path: Path) -> None:
    rankings = tmp_path / "rankings.jsonl"
    output = tmp_path / "manifest.jsonl"
    rankings.write_text(
        json.dumps(
            {
                "case_id": "CXR1",
                "question_type": "findings",
                "rankings": {"rrf_lambdamart": ["CXR2", "CXR3", "CXR4"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_v16_confirmation_manifest.py"),
            "--ranking-rows",
            str(rankings),
            "--output",
            str(output),
            "--expected-cases",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "Incomplete question matrix" in completed.stderr
