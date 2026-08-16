from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from medical_rag.multimodal.fusion import (
    aggregate_view_embeddings,
    reciprocal_rank_fusion,
    select_text_weight,
)


ROOT = Path(__file__).resolve().parents[1]


def test_aggregate_view_embeddings_normalizes_mean() -> None:
    result = aggregate_view_embeddings([np.array([3.0, 0.0]), np.array([0.0, 4.0])])
    assert np.linalg.norm(result) == pytest.approx(1.0)
    assert result.tolist() == pytest.approx([2**-0.5, 2**-0.5])


def test_rrf_endpoints_preserve_single_modality_rankings() -> None:
    text = ["CXR1", "CXR2", "CXR3"]
    image = ["CXR3", "CXR2", "CXR1"]
    assert reciprocal_rank_fusion(text, image, text_weight=1.0) == text
    assert reciprocal_rank_fusion(text, image, text_weight=0.0) == image


def test_rrf_rejects_mismatched_candidate_pools() -> None:
    with pytest.raises(ValueError, match="same case IDs"):
        reciprocal_rank_fusion(["CXR1"], ["CXR2"], text_weight=0.5)


def test_weight_selection_uses_registered_tie_break() -> None:
    text = {"q1": ["A", "B"], "q2": ["A", "B"]}
    image = {"q1": ["B", "A"], "q2": ["B", "A"]}
    relevant = {"q1": "A", "q2": "B"}
    selected = select_text_weight(text, image, relevant, [0.0, 0.4, 0.6, 1.0])
    assert selected["selected_text_weight"] == 0.4


def test_preregistered_cohorts_match_frozen_benchmarks() -> None:
    config = json.loads((ROOT / "config" / "multimodal_v4.json").read_text(encoding="utf-8"))
    for split in ("development", "confirmation"):
        registered = config["cohorts"][split]
        benchmark = json.loads((ROOT / registered["benchmark_path"]).read_text(encoding="utf-8"))
        assert benchmark["case_count"] == registered["case_count"]
        assert benchmark["question_count"] == registered["question_count"]
        assert benchmark["case_id_fingerprint_sha256"] == registered["case_id_fingerprint_sha256"]
        assert benchmark["content_fingerprint_sha256"] == registered["content_fingerprint_sha256"]
