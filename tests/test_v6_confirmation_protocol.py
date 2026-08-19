from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_confirmation_config_freezes_design_without_case_ids() -> None:
    config = json.loads((ROOT / "config" / "v6_confirmation.json").read_text(encoding="utf-8"))
    cohort = config["cohort_generation"]

    assert cohort["case_ids_instantiated"] is False
    assert cohort["candidate_pool_case_count"] == 240
    assert cohort["report_indexed_normal"] == 172
    assert cohort["report_indexed_abnormal"] == 68
    assert cohort["target"] == {
        "case_count": 120,
        "report_indexed_normal": 86,
        "report_indexed_abnormal": 34,
    }
    assert cohort["distractor"] == {
        "case_count": 120,
        "report_indexed_normal": 86,
        "report_indexed_abnormal": 34,
    }
    assert "case_ids" not in cohort


def test_confirmation_prompt_hash_matches_frozen_implementation() -> None:
    config = json.loads((ROOT / "config" / "v6_confirmation.json").read_text(encoding="utf-8"))
    path = ROOT / config["generation"]["prompt_implementation"]

    assert hashlib.sha256(path.read_bytes()).hexdigest() == config["generation"][
        "prompt_implementation_sha256"
    ]


def test_confirmation_selection_frame_matches_audit() -> None:
    config = json.loads((ROOT / "config" / "v6_confirmation.json").read_text(encoding="utf-8"))
    audit = json.loads(
        (ROOT / config["selection_frame"]["audit_path"]).read_text(encoding="utf-8")
    )["confirmation_selection_frame"]

    assert config["selection_frame"]["eligible_case_ids_sha256"] == audit[
        "v6_eligible_case_ids_sha256"
    ]
    assert config["selection_frame"]["stratifiable_case_ids_sha256"] == audit[
        "v6_stratifiable_case_ids_sha256"
    ]
    assert config["selection_frame"]["report_indexed_normal_ids_sha256"] == audit[
        "report_indexed_normal_case_ids_sha256"
    ]
    assert config["selection_frame"]["report_indexed_abnormal_ids_sha256"] == audit[
        "report_indexed_abnormal_case_ids_sha256"
    ]
