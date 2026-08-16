from __future__ import annotations

import json
from pathlib import Path

from scripts.build_v23_hybrid_preregistration import (
    POLICY_ID,
    build_records,
    second_transfer_question,
)
from scripts.evaluate_v21_template_transfer import transfer_question
from scripts.evaluate_v23_hybrid_planner import (
    _git_introduction_commit,
    _paired_case_bootstrap,
)


ROOT = Path(__file__).resolve().parents[1]


def _benchmark() -> dict:
    return json.loads(
        (ROOT / "data" / "processed" / "openi_case_scoped_hard_v21.json").read_text(
            encoding="utf-8"
        )
    )


def test_second_transfer_pack_is_complete_and_reserved() -> None:
    benchmark = _benchmark()
    source = [row for row in benchmark["questions"] if row["split"] == "test"]
    records = build_records(benchmark)
    assert len(records) == 432
    assert len({row["qid"] for row in records}) == 432
    by_qid = {row["source_qid"]: row for row in records}
    for row in source:
        generated = by_qid[row["qid"]]["question"]
        assert generated == second_transfer_question(row)
        assert generated != row["question"]
        assert generated != transfer_question(row)


def test_preregistered_manifest_declares_no_test_tuning() -> None:
    manifest = json.loads(
        (
            ROOT
            / "experiments"
            / "post_submission_v23"
            / "preregistration_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["policy_id"] == POLICY_ID
    assert manifest["policy_frozen_before_generation"] is True
    assert manifest["test_or_transfer_tuning"] is False
    assert manifest["post_evaluation_policy_changes_permitted"] is False
    assert manifest["record_count"] == 432
    assert manifest["case_count"] == 72


def test_paired_bootstrap_preserves_case_groups() -> None:
    lexical = []
    hybrid = []
    for case_id in ("c1", "c2"):
        for index, answerable in enumerate((True, False)):
            base = {
                "scope_case_id": case_id,
                "qid": f"{case_id}_{index}",
                "is_answerable": answerable,
            }
            lexical.append({**base, "answer_probability": 0.9})
            hybrid.append(
                {**base, "answer_probability": 0.9 if answerable else 0.1}
            )
    result = _paired_case_bootstrap(
        lexical, hybrid, threshold=0.5, resamples=100, seed=7
    )
    assert result["case_count"] == 2
    assert result["macro_f1_delta_hybrid_minus_lexical"]["observed"] > 0
    assert result["false_answer_rate_delta_hybrid_minus_lexical"]["observed"] < 0


def test_preregistration_commit_is_the_manifest_introduction() -> None:
    commit = _git_introduction_commit(
        "experiments/post_submission_v23/preregistration_manifest.json"
    )
    assert len(commit) == 40
    assert commit.startswith("47a2c1a")
