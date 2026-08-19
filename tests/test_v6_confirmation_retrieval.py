from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_v6_confirmation_retrieval.py"
SPEC = importlib.util.spec_from_file_location("v6_confirmation_retrieval", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_shuffled_assignments_are_deterministic_unique_derangements() -> None:
    targets = [f"CXR{index}" for index in range(120)]
    first = MODULE.shuffled_image_assignments(
        targets, count=100, seed=7026, domain="v6-shuffle-order"
    )
    second = MODULE.shuffled_image_assignments(
        list(reversed(targets)), count=100, seed=7026, domain="v6-shuffle-order"
    )

    assert first == second
    assert len(first) == 100
    assert len({MODULE.stable_signature(row) for row in first}) == 100
    assert all(set(row) == set(targets) for row in first)
    assert all(set(row.values()) == set(targets) for row in first)
    assert all(source != assigned for row in first for source, assigned in row.items())
