from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from merge_v16_question_type_route import build_routed_rows, load_rows  # noqa: E402


def write_rows(path: Path) -> None:
    rows = []
    for question_type in ("findings", "impression"):
        for condition in ("no_history", "retrieved_history", "random_history"):
            rows.append(
                {
                    "case_id": "case-1",
                    "question_type": question_type,
                    "condition": condition,
                    "answer": f"{question_type}-{condition}",
                }
            )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_v16_route_inputs_have_unique_complete_keys(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    write_rows(path)
    rows = load_rows(path)

    assert len(rows) == 6
    assert rows[("case-1", "impression", "retrieved_history")]["answer"] == (
        "impression-retrieved_history"
    )


def test_v16_condition_gate_uses_specialist_only_for_matching_rows() -> None:
    rows = {
        ("case-1", question_type, condition): {
            "case_id": "case-1",
            "question_type": question_type,
            "condition": condition,
            "answer": f"base-{question_type}-{condition}",
        }
        for question_type in ("findings", "impression")
        for condition in ("no_history", "retrieved_history", "random_history")
    }
    specialist = {
        key: {**value, "answer": value["answer"].replace("base-", "specialist-")}
        for key, value in rows.items()
    }

    routed, specialist_count = build_routed_rows(
        rows,
        specialist,
        specialist_question_type="impression",
        specialist_condition="retrieved_history",
    )
    by_key = {
        (row["case_id"], row["question_type"], row["condition"]): row
        for row in routed
    }

    assert specialist_count == 1
    assert by_key[("case-1", "impression", "retrieved_history")]["answer"] == (
        "specialist-impression-retrieved_history"
    )
    assert by_key[("case-1", "impression", "no_history")]["answer"] == (
        "base-impression-no_history"
    )
