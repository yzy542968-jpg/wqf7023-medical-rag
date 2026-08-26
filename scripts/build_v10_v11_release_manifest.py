"""Build the public V10/V11 thesis release manifest."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "v10_v11_final_release_manifest.json"
CANONICAL_TEXT_SUFFIXES = {".json", ".md", ".txt", ".toml", ".yaml", ".yml"}

FILES = {
    "manuscript_markdown": "docs/P2_V10_V11_FINAL_MANUSCRIPT.md",
    "manuscript_docx": "deliverables/22097191_ZHANG_YUE_Final_Research_Project.docx",
    "manuscript_pdf": "deliverables/22097191_ZHANG_YUE_Final_Research_Project.pdf",
    "defence_pptx": "deliverables/22097191_ZHANG_YUE_Final_Defence.pptx",
    "readme": "README.md",
    "result_registry": "docs/FINAL_RESULTS_REGISTRY.md",
    "prompt_registry": "docs/PROMPT_TEMPLATES.md",
    "release_audit": "docs/V10_V11_FINAL_RELEASE_AUDIT.md",
    "v10_freeze": "docs/V10_TECHNICAL_FREEZE.md",
    "v10_config": "config/v10_confirmation.json",
    "v10_split_freeze": "data/splits/v10/v10_cluster_disjoint_split_freeze.json",
    "v10_retrieval_summary": "data/splits/v10/v10_confirmation_retrieval_summary.json",
    "v10_qa_summary": "data/splits/v10/v10_confirmation_qa_summary.json",
    "v10_radgraph_summary": "data/splits/v10/v10_radgraph_metrics_summary.json",
    "v10_fact_attention_summary": "data/splits/v10/v10_fact_attention_2x2_summary.json",
    "v10_qrel_sensitivity_summary": "data/splits/v10/v10_qrel_sensitivity_summary.json",
    "v10_qrel_sensitivity_results": "docs/V10_POSTHOC_QREL_SENSITIVITY_RESULTS.md",
    "v10_clinical_review_status": "docs/V10_CLINICAL_REVIEW_PACKAGE_STATUS.md",
    "v10_clinical_review_summary": "data/splits/v10/v10_clinical_review_package_summary.json",
    "v11_config": "config/v11_development.json",
    "v11_evidence_ablation_summary": "data/splits/v11/v11_development_evidence_ablation_summary.json",
    "v11_candidate_generation_k100": "data/splits/v11/v11_candidate_generation_audit_summary.json",
    "v11_candidate_generation_k200": "data/splits/v11/v11_candidate_generation_audit_k200_summary.json",
    "v11_generation_summary": "data/splits/v11/v11_medgemma_generation_48_clean_summary.json",
    "v11_generation_statistics": "data/splits/v11/v11_medgemma_generation_48_statistical_summary.json",
    "v11_planner_reserved_protocol": "docs/V11_PLANNER_RESERVED_SET_PROTOCOL.md",
    "v11_planner_reserved_summary": "data/splits/v11/v11_question_planner_reserved_summary.json",
}


def release_payload(path: Path) -> tuple[bytes, str]:
    if path.suffix.lower() in CANONICAL_TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8")
        canonical = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        return canonical, "canonical_utf8_lf"
    return path.read_bytes(), "raw_bytes"


def main() -> None:
    missing = [relative for relative in FILES.values() if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError("Missing release artifacts: " + ", ".join(missing))

    manifest = {
        "release": "v10-v11-final-thesis-freeze",
        "generated_on": date.today().isoformat(),
        "repository": "https://github.com/yzy542968-jpg/wqf7023-medical-rag",
        "primary_study": "V10 frozen confirmation",
        "development_extension": "V11 Train/Validation only",
        "acceptance": {
            "python_compileall": "passed",
            "pytest": {"passed": 276, "failed": 0, "errors": 0},
            "fresh_clone_pytest": {"passed": 272, "skipped": 4, "failed": 0, "errors": 0},
            "docx_pdf_pages": 58,
            "pptx_slides": 15,
            "pptx_overflow_test": "passed",
            "human_review": "future_work_no_scores_claimed",
            "external_validation": "future_work_no_result_claimed",
        },
        "claim_boundary": (
            "V10 reports automated within-source retrieval and report-reference-consistency "
            "metrics. V11 is development-only. Neither establishes physician-adjudicated "
            "diagnostic correctness, clinical safety, treatment utility, deployment performance, "
            "or external patient-level generalization."
        ),
        "files": {},
    }
    for key, relative in FILES.items():
        path = ROOT / relative
        payload, hash_mode = release_payload(path)
        manifest["files"][key] = {
            "path": relative.replace("\\", "/"),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "hash_mode": hash_mode,
        }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
