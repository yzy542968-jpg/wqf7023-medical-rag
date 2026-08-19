from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_v6_development_confirmation_separation import (
    case_id_fingerprint,
    case_id_payload,
    is_v6_eligible_case,
    report_index_class,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    ROOT
    / "data"
    / "splits"
    / "v6"
    / "v6_development_confirmation_overlap_audit.json"
)
DEVELOPMENT_MANIFEST_PATH = (
    ROOT / "data" / "splits" / "v6" / "v6_development_case_ids.txt"
)


def test_case_id_fingerprint_is_canonical_and_has_no_trailing_newline() -> None:
    assert case_id_payload([" CXR2 ", "CXR1"]) == "CXR1\nCXR2"
    expected = hashlib.sha256(b"CXR1\nCXR2").hexdigest()
    assert case_id_fingerprint(["CXR2", "CXR1"]) == expected


def test_case_id_fingerprint_rejects_duplicates_after_normalization() -> None:
    with pytest.raises(ValueError):
        case_id_fingerprint(["CXR1", " CXR1 "])


def test_v6_eligibility_includes_normal_and_indeterminate_problem_labels() -> None:
    common = {
        "case_id": "CXR1",
        "images": [{"filename": "image.png"}],
        "findings": "A" * 40,
        "impression": "B" * 8,
        "indication": "Follow-up examination",
    }
    assert is_v6_eligible_case({**common, "problems": "normal"})
    assert is_v6_eligible_case({**common, "problems": "no indexing"})
    assert report_index_class({**common, "problems": " normal "}) == (
        "report_indexed_normal"
    )
    assert report_index_class({**common, "problems": "No Indexing"}) == (
        "report_index_indeterminate"
    )
    assert report_index_class({**common, "problems": "Cardiomegaly"}) == (
        "report_indexed_abnormal"
    )
    with pytest.raises(ValueError):
        report_index_class({**common, "problems": "unknown"})


def test_v6_frozen_input_frame_and_development_are_case_id_disjoint() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    frame = audit["confirmation_selection_frame"]
    separation = audit["separation_check"]
    composition = audit["predefined_confirmation_composition"]

    assert audit["status"] == (
        "selection_frame_audit_only_no_confirmation_case_ids_instantiated"
    )
    assert audit["development_source"]["case_count"] == 120
    assert audit["development_source"]["report_index_spectrum"] == {
        "report_indexed_normal": 0,
        "report_indexed_abnormal": 114,
        "report_index_indeterminate": 6,
    }
    assert frame["v6_eligible_case_count"] == 1479
    assert frame["v6_stratifiable_case_count"] == 1462
    assert frame["report_indexed_normal_case_count"] == 1045
    assert frame["report_indexed_abnormal_case_count"] == 417
    assert frame["report_index_indeterminate_case_count"] == 17
    assert frame["problems_field_audit"] == {
        "normalized_unique_value_count": 322,
        "known_indeterminate_values": ["no indexing"],
        "known_indeterminate_value_counts": {"no indexing": 17},
        "unexpected_administrative_values": [],
    }
    assert separation["development_confirmation_eligible_overlap_count"] == 0
    assert separation["development_confirmation_stratifiable_overlap_count"] == 0
    assert separation["case_id_disjointness_verified"] is True
    assert separation["patient_level_independence_verified"] is False
    assert composition["report_indexed_normal"] == 172
    assert composition["report_indexed_abnormal"] == 68
    assert composition["instantiated_case_ids_present"] is False

    manifest_text = DEVELOPMENT_MANIFEST_PATH.read_text(encoding="utf-8")
    manifest_ids = manifest_text.splitlines()
    assert len(manifest_ids) == 120
    assert manifest_text == "\n".join(sorted(set(manifest_ids)))
    assert not manifest_text.endswith("\n")
    assert case_id_fingerprint(manifest_ids) == (
        audit["development_source"]["case_ids_sha256"]
    )
