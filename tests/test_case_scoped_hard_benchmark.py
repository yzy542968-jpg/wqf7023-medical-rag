from __future__ import annotations

from medical_rag.evaluation.case_scoped_hard_benchmark import (
    build_case_scoped_hard_benchmark,
)


def _case(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "indication": "Persistent cough.",
        "comparison": "No prior examination is available.",
        "findings": "There is a focal right basilar opacity without pleural effusion.",
        "impression": "Right basilar airspace opacity.",
        "problems": "Opacity",
        "images": [{"filename": f"{case_id}.png"}],
    }


def test_hard_benchmark_is_balanced_and_excludes_prior_cases() -> None:
    payload = build_case_scoped_hard_benchmark(
        [_case(f"CXR{index}") for index in range(13)],
        excluded_case_ids={"CXR0"},
        max_cases=12,
        seed=19,
    )
    assert payload["version"] == "2.1"
    assert payload["answerable_count"] == payload["unanswerable_count"] == 36
    assert "CXR0" not in {row["case_id"] for row in payload["questions"]}
    split_sets = [
        set(payload["split"][name]["case_ids"])
        for name in ("development", "calibration", "test")
    ]
    assert not split_sets[0] & split_sets[1]
    assert not split_sets[0] & split_sets[2]
    assert not split_sets[1] & split_sets[2]


def test_unanswerable_questions_have_no_gold_evidence() -> None:
    payload = build_case_scoped_hard_benchmark(
        [_case("CXR1")], set(), max_cases=1, seed=3
    )
    negatives = [row for row in payload["questions"] if not row["is_answerable"]]
    assert len(negatives) == 3
    assert all(row["relevant_chunk_ids"] == [] for row in negatives)
    assert all(row["reference_answer"] == "NOT ANSWERABLE" for row in negatives)
