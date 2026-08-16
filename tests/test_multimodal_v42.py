from __future__ import annotations

import hashlib
import json
from pathlib import Path


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
