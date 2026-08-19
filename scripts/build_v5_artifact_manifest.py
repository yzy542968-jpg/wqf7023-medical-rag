from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("experiments/post_submission_v5/artifact_manifest.json")

ARTIFACTS = [
    Path("config/multimodal_v5.json"),
    Path("data/processed/openi_multimodal_v5_cohort.json"),
    Path("experiments/post_submission_v5/confirmation_multimodal_runtime.json"),
    Path("experiments/post_submission_v5/confirmation_report_only_runtime.json"),
    Path("experiments/post_submission_v5/confirmation_retrieval_summary.json"),
    Path("experiments/post_submission_v5/qa_multimodal/final_optimized_test_summary.json"),
    Path("experiments/post_submission_v5/qa_report_only/final_optimized_test_summary.json"),
    Path("experiments/post_submission_v5/v5_statistics.json"),
    Path("scripts/analyze_multimodal_v5_statistics.py"),
    Path("scripts/build_multimodal_v5_cohort.py"),
    Path("scripts/build_multimodal_v5_prompt_packs.py"),
    Path("scripts/run_multimodal_v5_retrieval.py"),
    Path("src/medical_rag/dashboard/multimodal_runtime.py"),
    Path("tests/test_multimodal_dashboard.py"),
    Path("tests/test_multimodal_v5.py"),
]


def portable_sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen V5 artifact manifest.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    missing = [path.as_posix() for path in ARTIFACTS if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing V5 artifacts: {', '.join(missing)}")

    cohort = json.loads(
        (ROOT / "data/processed/openi_multimodal_v5_cohort.json").read_text(encoding="utf-8")
    )
    config = json.loads((ROOT / "config/multimodal_v5.json").read_text(encoding="utf-8"))
    manifest = {
        "experiment": "v5_end_to_end_multimodal_qa",
        "freeze_version": "5.0",
        "provenance_status": config["status"],
        "cohort": {
            "case_count": cohort["case_count"],
            "question_count": cohort["question_count"],
            "case_id_fingerprint_sha256": cohort["case_id_fingerprint_sha256"],
            "development_case_count": len(cohort["split"]["development"]["case_ids"]),
            "confirmation_case_count": len(cohort["split"]["confirmation"]["case_ids"]),
        },
        "artifacts": [
            {
                "path": relative.as_posix(),
                "bytes": (ROOT / relative).stat().st_size,
                "sha256_lf_normalized": portable_sha256(ROOT / relative),
            }
            for relative in ARTIFACTS
        ],
        "local_only_artifacts": [
            "experiments/post_submission_v5/**/*.jsonl",
            "experiments/post_submission_v5/**/*.log",
            "data/processed/prompt_packs/**",
        ],
        "claim_limits": config["claim_limits"],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": args.output.as_posix(), "artifact_count": len(ARTIFACTS)}, indent=2))


if __name__ == "__main__":
    main()
