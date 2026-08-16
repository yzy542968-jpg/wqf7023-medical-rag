from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from medical_rag.multimodal.fusion import minmax_normalize, shortlist_score_fusion


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v42_manifest_freezes_fixed_reranking_policy() -> None:
    config_path = ROOT / "config" / "multimodal_v42.json"
    manifest_path = ROOT / "experiments" / "post_submission_v42" / "preregistration_manifest.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["config_sha256"] == sha256(config_path)
    assert manifest["fixed_reranking_policy"] == config["reranking"]
    assert manifest["fixed_reranking_policy"]["shortlist_size"] == 100
    assert manifest["fixed_reranking_policy"]["text_weight"] == 0.5
    assert manifest["fixed_reranking_policy"]["parameter_tuning_after_preregistration"] is False
    assert manifest["confirmation_run_limit"] == 1


def test_v42_source_hashes_are_current() -> None:
    manifest = json.loads(
        (ROOT / "experiments" / "post_submission_v42" / "preregistration_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for source in manifest["source_files"]:
        assert source["sha256"] == sha256(ROOT / source["path"])


def test_minmax_normalize_uses_registered_zero_range_policy() -> None:
    assert minmax_normalize([2.0, 4.0, 6.0]).tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert minmax_normalize([3.0, 3.0]).tolist() == [0.0, 0.0]


def test_shortlist_fusion_reranks_only_fixed_candidate_prefix() -> None:
    ranking = ["A", "B", "C", "D"]
    fused = shortlist_score_fusion(
        ranking,
        text_scores=[4.0, 3.0, 2.0, 1.0],
        image_scores={"A": 0.0, "B": 0.9, "C": 1.0, "D": 2.0},
        shortlist_size=3,
        text_weight=0.5,
    )
    assert fused == ["B", "A", "C", "D"]


def test_shortlist_fusion_rejects_candidate_mismatch() -> None:
    with pytest.raises(ValueError, match="same case IDs"):
        shortlist_score_fusion(["A", "B"], [2.0, 1.0], {"A": 0.1}, 1, 0.5)
