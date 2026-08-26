from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from medical_rag.multimodal.evaluation import evaluate_confirmation_gate


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256_variants(path: Path) -> set[str]:
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    return {hashlib.sha256(value).hexdigest() for value in (raw, lf, crlf)}


def test_v41_manifest_freezes_config_and_sources() -> None:
    config_path = ROOT / "config" / "multimodal_v41.json"
    manifest_path = ROOT / "experiments" / "post_submission_v41" / "preregistration_manifest.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["config_sha256"] == sha256(config_path)
    assert manifest["confirmation_run_limit"] == 1
    assert manifest["confirmation_tuning"] is False
    assert config["retrieval"]["joint_embedding_dimension"] == 128
    assert config["retrieval"]["image_weights_md5"] == "a83080e2f23aa584a4f2b24c39b1bb64"
    missing = [source["path"] for source in manifest["source_files"] if not (ROOT / source["path"]).is_file()]
    if missing:
        pytest.skip(f"requires local source artifacts excluded from Git: {missing}")
    for source in manifest["source_files"]:
        assert source["sha256"] in text_sha256_variants(ROOT / source["path"])


def test_v41_confirmation_gate_rejects_text_only_selection() -> None:
    config = json.loads((ROOT / "config" / "multimodal_v41.json").read_text(encoding="utf-8"))
    gate = config["confirmation_gate"]
    assert gate["selected_text_weight_must_be_less_than"] == 1.0
    assert gate["fusion_mrr_must_exceed_report_only_mrr"] is True
    assert gate["on_failure"].startswith("Do not evaluate confirmation")

    metrics = {
        "image_only_biovil_t": {"mrr": 0.05},
        "report_only_bm25": {"mrr": 0.55},
        "paired_biovil_t_rrf": {"mrr": 0.55},
    }
    result = evaluate_confirmation_gate(config, metrics, selected_text_weight=1.0)
    assert result["passed"] is False
    assert result["checks"] == {
        "image_mrr_exceeds_v4": True,
        "fusion_mrr_exceeds_report_only": False,
        "selected_text_weight_below_limit": False,
    }


def test_v41_confirmation_gate_accepts_strict_improvement() -> None:
    config = json.loads((ROOT / "config" / "multimodal_v41.json").read_text(encoding="utf-8"))
    metrics = {
        "image_only_biovil_t": {"mrr": 0.05},
        "report_only_bm25": {"mrr": 0.55},
        "paired_biovil_t_rrf": {"mrr": 0.56},
    }
    result = evaluate_confirmation_gate(config, metrics, selected_text_weight=0.9)
    assert result["passed"] is True
