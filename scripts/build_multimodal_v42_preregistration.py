from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "multimodal_v42.json"
OUTPUT = ROOT / "experiments" / "post_submission_v42" / "preregistration_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source_paths = [
        ROOT / config["source"]["cases_path"],
        ROOT / config["cohorts"]["development"]["benchmark_path"],
        ROOT / config["cohorts"]["confirmation"]["benchmark_path"],
        ROOT / "data" / "processed" / "openi_multimodal_source_manifest.json",
    ]
    manifest = {
        "experiment": config["experiment"],
        "policy_frozen_before_v4_2_formal_outcomes": True,
        "config_sha256": sha256(CONFIG),
        "source_files": [
            {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path)}
            for path in source_paths
        ],
        "fixed_reranking_policy": config["reranking"],
        "development_gate": config["development_gate"],
        "confirmation_tuning": False,
        "confirmation_run_limit": 1,
        "protected_submission_artifacts_unchanged": True,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
