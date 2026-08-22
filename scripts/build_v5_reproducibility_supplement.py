from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments/post_submission_v5/v5_reproducibility_supplement_manifest.json"

# This is intentionally separate from artifact_manifest.json. It documents the
# transitive runtime inputs without changing the original V5 freeze record.
DEPENDENCIES = [
    "config/multimodal_v5.json",
    "config/multimodal_v41.json",
    "data/processed/openi_cases.jsonl",
    "data/processed/openi_multimodal_v5_cohort.json",
    "experiments/final_optimized/semantic_agent/semantic_agent_selection.json",
    "scripts/build_multimodal_v5_cohort.py",
    "scripts/build_multimodal_v5_prompt_packs.py",
    "scripts/run_multimodal_v4_retrieval.py",
    "scripts/run_multimodal_v41_retrieval.py",
    "scripts/run_multimodal_v5_retrieval.py",
    "scripts/run_hf_generation.py",
    "scripts/evaluate_final_optimized_test.py",
    "scripts/analyze_multimodal_v5_statistics.py",
    "scripts/run_grouped_statistical_analysis.py",
    "src/medical_rag/multimodal/biovilt.py",
    "src/medical_rag/multimodal/evaluation.py",
    "src/medical_rag/multimodal/fusion.py",
    "src/medical_rag/retrieval/bm25_retriever.py",
    "src/medical_rag/retrieval/tfidf_retriever.py",
    "src/medical_rag/evaluation/answer_metrics.py",
    "src/medical_rag/evaluation/metrics.py",
    "src/medical_rag/agentic/evidence_checker.py",
    "src/medical_rag/agentic/semantic_evidence_checker.py",
]


def normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a supplemental V5 dependency manifest.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    missing = [relative for relative in DEPENDENCIES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError("Missing supplemental dependencies: " + ", ".join(missing))

    payload = {
        "experiment": "v5_end_to_end_multimodal_qa",
        "manifest_kind": "post_freeze_reproducibility_supplement",
        "primary_v5_results_modified": False,
        "purpose": "Record transitive runtime dependencies omitted from the original lightweight freeze manifest.",
        "model_and_cache_note": "Local BioViL-T, Qwen, MedNLI weights, image pixels, and generated JSONL rows remain environment- or policy-controlled inputs.",
        "dependencies": [
            {
                "path": relative,
                "bytes": (ROOT / relative).stat().st_size,
                "sha256_lf_normalized": normalized_sha256(ROOT / relative),
            }
            for relative in DEPENDENCIES
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": args.output.as_posix(), "dependency_count": len(DEPENDENCIES)}, indent=2))


if __name__ == "__main__":
    main()
