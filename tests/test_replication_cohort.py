from __future__ import annotations

from medical_rag.evaluation.replication_cohort import build_replication_cohort


def _case(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "indication": "Cough.",
        "findings": "There is a focal right basilar pulmonary opacity.",
        "impression": "Right basilar airspace opacity.",
        "problems": "Opacity",
        "images": [{"filename": f"{case_id}.png"}],
    }


def test_replication_cohort_is_deterministic_and_excludes_prior_cases() -> None:
    cases = [_case(f"CXR{index}") for index in range(12)]
    first = build_replication_cohort(cases, {"CXR0", "CXR1"}, max_cases=10, seed=7)
    second = build_replication_cohort(cases, {"CXR0", "CXR1"}, max_cases=10, seed=7)
    assert first["content_fingerprint_sha256"] == second["content_fingerprint_sha256"]
    assert not {"CXR0", "CXR1"} & set(first["case_ids"])
    assert first["question_count"] == 30
