from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pilot_final_qa_v2_pairing_feasibility",
    ROOT / "scripts/pilot_final_qa_v2_pairing_feasibility.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_payload_permutation_is_deterministic_and_fixed_point_free() -> None:
    case_ids = ["CXR1", "CXR2", "CXR3", "CXR4"]
    first = MODULE._fixed_point_free_payload_permutation(
        case_ids, seed=7047, replicate=0
    )
    second = MODULE._fixed_point_free_payload_permutation(
        case_ids, seed=7047, replicate=0
    )
    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == list(range(len(case_ids)))
    assert not np.any(first == np.arange(len(case_ids)))


def test_payload_permutation_changes_across_replicates() -> None:
    case_ids = [f"CXR{index}" for index in range(12)]
    first = MODULE._fixed_point_free_payload_permutation(
        case_ids, seed=7047, replicate=0
    )
    second = MODULE._fixed_point_free_payload_permutation(
        case_ids, seed=7047, replicate=1
    )
    assert not np.array_equal(first, second)
