from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_v6_development_qa.py"
SPEC = importlib.util.spec_from_file_location("v6_qa_verification", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def factorial_rows() -> list[dict[str, object]]:
    return [
        {"system": system, "qid": f"q{index}", "case_id": f"c{index // 3}"}
        for system in sorted(MODULE.EXPECTED_SYSTEMS)
        for index in range(360)
    ]


def test_factorial_integrity_requires_four_matched_qid_sets() -> None:
    result = MODULE.validate_factorial_rows(factorial_rows())

    assert result["row_count"] == 1440
    assert result["qids_per_system"] == 360
    assert result["identical_qid_sets"] is True


def test_factorial_integrity_rejects_duplicate_system_qid() -> None:
    rows = factorial_rows()
    rows.append(dict(rows[0]))

    with pytest.raises(ValueError, match="Duplicate"):
        MODULE.validate_factorial_rows(rows)


def test_verifier_evidence_text_matches_frozen_v5_scope() -> None:
    actual = MODULE.evidence_text(
        {"case_id": "CXR1", "findings": "Clear.", "impression": "Normal."}
    )

    assert actual == "Case ID: CXR1\nFindings: Clear.\nImpression: Normal."
