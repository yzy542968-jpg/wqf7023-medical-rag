from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "develop_final_qa_v2_selective_gate",
    ROOT / "scripts/develop_final_qa_v2_selective_gate.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_hash_parity_is_deterministic() -> None:
    first = MODULE._hash_parity("final-qa-v2-gate-train|", "CXR123")
    second = MODULE._hash_parity("final-qa-v2-gate-train|", "CXR123")
    assert first == second


def test_gate_encoder_has_stable_dense_shape() -> None:
    encoder = MODULE.GateFeatureEncoder(4, ("single", "multi"))
    numeric = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    encoder.fit(numeric)
    transformed = encoder.transform(
        numeric,
        np.asarray([0, 3]),
        np.asarray(["single", "multi"]),
    )
    assert transformed.shape == (2, 8)
    assert np.isfinite(transformed).all()


def test_frozen_result_respects_test_and_go_boundaries() -> None:
    path = (
        ROOT
        / "experiments/final_qa_development/final_qa_v2_selective_gate.json"
    )
    if not path.exists():
        return
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["test_accessed"] is False
    assert result["new_medgemma_inference_performed"] is False
    checks = {
        key: value
        for key, value in result["go_rule"].items()
        if key != "passed"
    }
    assert result["go_rule"]["passed"] is all(checks.values())
    assert result["status"] == "stop_or_redesign"
    assert result["gate_selection"]["selected_model"] == (
        "image_similarity_threshold"
    )
