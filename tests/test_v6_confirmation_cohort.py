from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_v6_confirmation_cohort.py"
SPEC = importlib.util.spec_from_file_location("v6_confirmation_cohort", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ROOT = Path(__file__).resolve().parents[1]


def test_hash_selection_is_deterministic_disjoint_and_balanced() -> None:
    normal = [f"N{index:03d}" for index in range(300)]
    abnormal = [f"A{index:03d}" for index in range(100)]

    first = MODULE.select_and_assign(
        normal,
        abnormal,
        seed=7026,
        selection_domain="v6-selection",
        assignment_domain="v6-assignment",
    )
    second = MODULE.select_and_assign(
        reversed(normal),
        reversed(abnormal),
        seed=7026,
        selection_domain="v6-selection",
        assignment_domain="v6-assignment",
    )

    assert first == second
    assert len(first["selected"]) == 240
    assert len(first["selected_normal"]) == 172
    assert len(first["selected_abnormal"]) == 68
    assert len(first["target_normal"]) == 86
    assert len(first["target_abnormal"]) == 34
    assert len(first["distractor_normal"]) == 86
    assert len(first["distractor_abnormal"]) == 34
    assert set(first["targets"]).isdisjoint(first["distractors"])
    assert set(first["targets"]) | set(first["distractors"]) == set(first["selected"])


def test_domain_separation_changes_hash_order_key() -> None:
    selection = MODULE.hash_order_key("v6-selection", 7026, "CXR1")
    assignment = MODULE.hash_order_key("v6-assignment", 7026, "CXR1")

    assert selection != assignment
    assert len(selection) == 64


def test_instantiated_cohort_matches_freeze_and_remains_development_disjoint() -> None:
    cohort_path = ROOT / "data" / "splits" / "v6" / "v6_confirmation_cohort.json"
    freeze = json.loads(
        (ROOT / "data" / "splits" / "v6" / "v6_confirmation_cohort_freeze.json").read_text(
            encoding="utf-8"
        )
    )
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    development = set(
        (ROOT / "data" / "splits" / "v6" / "v6_development_case_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    assert hashlib.sha256(cohort_path.read_bytes()).hexdigest() == freeze["artifacts"][
        "cohort_sha256"
    ]
    assert cohort["protocol_commit"] == "eee7405"
    assert len(cohort["case_ids"]) == 240
    assert len(cohort["target_case_ids"]) == 120
    assert len(cohort["distractor_case_ids"]) == 120
    assert set(cohort["target_case_ids"]).isdisjoint(cohort["distractor_case_ids"])
    assert set(cohort["case_ids"]).isdisjoint(development)
    assert len(cohort["questions"]) == 360
    assert {row["case_id"] for row in cohort["questions"]} == set(
        cohort["target_case_ids"]
    )


def test_instantiated_cohort_has_frozen_role_and_spectrum_counts() -> None:
    cohort = json.loads(
        (ROOT / "data" / "splits" / "v6" / "v6_confirmation_cohort.json").read_text(
            encoding="utf-8"
        )
    )
    counts: dict[tuple[str, str], int] = {}
    for row in cohort["cases"]:
        key = (row["role"], row["report_index_class"])
        counts[key] = counts.get(key, 0) + 1

    assert counts == {
        ("target", "report_indexed_normal"): 86,
        ("target", "report_indexed_abnormal"): 34,
        ("distractor", "report_indexed_normal"): 86,
        ("distractor", "report_indexed_abnormal"): 34,
    }
