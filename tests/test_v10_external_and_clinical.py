from pathlib import Path

import pytest

from medical_rag.evaluation.v10_clinical_review import (
    build_blinded_review_rows,
    public_review_rows,
    validate_completed_review,
)
from medical_rag.similar_case.mimic_cxr_adapter import (
    MimicCxrCase,
    parse_report_sections,
    patient_disjoint_partition,
    report_path,
)


def test_mimic_report_sections_and_paths() -> None:
    findings, impression = parse_report_sections(
        "FINDINGS: The lungs are clear.\nIMPRESSION: No acute disease."
    )
    assert findings == "The lungs are clear."
    assert impression == "No acute disease."
    assert report_path(Path("reports"), "10000032", "50414267") == Path(
        "reports/p10/p10000032/s50414267.txt"
    )


def test_mimic_partition_is_patient_disjoint_and_deterministic() -> None:
    cases = [
        MimicCxrCase(str(subject), str(100 + subject), (f"i{subject}",), (f"x{subject}",), "f", "i", None)
        for subject in range(20)
    ]
    fractions = {"train": 0.65, "calibration": 0.10, "validation": 0.10, "test": 0.15}
    first = patient_disjoint_partition(cases, fractions, seed=7048)
    second = patient_disjoint_partition(cases, fractions, seed=7048)
    assert first == second
    assert sum(len(values) for values in first.values()) == 20
    assert len(set().union(*(set(values) for values in first.values()))) == 20


def test_clinical_package_is_blinded_and_cannot_fake_completion() -> None:
    cases = [
        {
            "case_id": "CXR1",
            "question": "Is there edema?",
            "indication": "Dyspnea",
            "answers": {"g0": "No.", "g2": "No edema."},
            "retrieval": {"g0": "None", "g2": "Similar normal case"},
        }
    ]
    rows = build_blinded_review_rows(cases, system_names=["g0", "g2"], case_count=1)
    assert {row["presentation_code"] for row in rows} == {"A", "B"}
    assert all("system_key_private" not in row for row in public_review_rows(rows))
    with pytest.raises(ValueError, match="incomplete"):
        validate_completed_review(rows)

