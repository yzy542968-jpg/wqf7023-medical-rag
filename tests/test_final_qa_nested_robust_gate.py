from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_final_qa_nested_robust_gate",
    ROOT / "scripts/audit_final_qa_nested_robust_gate.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_outer_assignment_is_deterministic_and_bounded() -> None:
    first = MODULE._assignment("final-qa-robust-outer", 7055, "CXR9", 5)
    second = MODULE._assignment("final-qa-robust-outer", 7055, "CXR9", 5)
    assert first == second
    assert 0 <= first < 5


def test_utility_policy_includes_exact_margin_boundary() -> None:
    keys = [("CXR1", 0), ("CXR2", 0)]
    qids = {key: 4 for key in keys}
    source = MODULE._apply_utility_policy(
        keys,
        qids,
        {4: (5, 0.05)},
        minimum_support=5,
        minimum_margin=0.05,
    )
    assert set(source.values()) == {MODULE.B6}


def test_nested_result_preserves_test_boundary_and_go_rule() -> None:
    path = (
        ROOT
        / "experiments/final_qa_development/final_qa_nested_robust_gate.json"
    )
    if not path.exists():
        return
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["test_accessed"] is False
    assert result["new_generation_performed"] is False
    assert result["status"] == "go_for_confirmation_design"
    checks = {
        key: value
        for key, value in result["advancement"].items()
        if key != "passed"
    }
    assert result["advancement"]["passed"] is all(checks.values())
    interval = result["nested_oof"]["case_bootstrap_macro_f1_vs_b3"]
    assert interval["ci95_low"] < 0 < interval["ci95_high"]
