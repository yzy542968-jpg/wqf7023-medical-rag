from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_v15_retrieval_transfer import combined_rows, select_scope


def make_rows(condition: str) -> list[dict]:
    rows = []
    for index in range(48):
        for question_type in ("findings", "impression", "acute"):
            row = {
                "case_id": str(index),
                "question_type": question_type,
                "reference_is_proxy": question_type == "acute",
                "answer": "x",
                "reference_answer": "x",
            }
            if condition == "default_17":
                row.update({"policy": "whole_report", "max_new_tokens": 96})
            rows.append(row)
    return rows


def test_v15_combines_complete_condition_matrices() -> None:
    rows = combined_rows(make_rows("default_17"), make_rows("deeper_17"))
    assert len(rows) == 288
    assert len(select_scope(rows, "primary")) == 192
    assert len(select_scope(rows, "findings")) == 96

