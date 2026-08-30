from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "develop_final_qa_real_output_gate",
    ROOT / "scripts/develop_final_qa_real_output_gate.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_case_fold_is_deterministic_and_bounded() -> None:
    first = MODULE._fold("CXR123", 7053, 5)
    second = MODULE._fold("CXR123", 7053, 5)
    assert first == second
    assert 0 <= first < 5


def test_option_macro_ignores_unsupported_options() -> None:
    rows = [
        {"option_count": 2, "gold_indices": [0], "predicted_indices": [0]},
        {"option_count": 2, "gold_indices": [0], "predicted_indices": []},
    ]
    assert MODULE._option_f1(rows) == 2 / 3


def test_real_output_gate_result_preserves_test_boundary() -> None:
    path = (
        ROOT
        / "experiments/final_qa_development/final_qa_real_output_gate.json"
    )
    if not path.exists():
        return
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["test_accessed"] is False
    assert result["new_generation_performed"] is False
    assert result["status"] == "stop"
    checks = {
        key: value
        for key, value in result["advancement"].items()
        if key != "passed"
    }
    assert result["advancement"]["passed"] is all(checks.values())
