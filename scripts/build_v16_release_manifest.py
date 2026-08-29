"""Build the public V16 final thesis release manifest."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "v16_final_release_manifest.json"
CANONICAL_TEXT_SUFFIXES = {".json", ".md", ".txt", ".toml", ".yaml", ".yml"}

FILES = {
    "manuscript_markdown": "docs/P2_FINAL_MANUSCRIPT.md",
    "manuscript_docx": "deliverables/22097191_ZHANG_YUE_Final_Research_Project.docx",
    "manuscript_pdf": "deliverables/22097191_ZHANG_YUE_Final_Research_Project.pdf",
    "defence_slide_outline": "docs/P2_FINAL_DEFENCE_SLIDE_OUTLINE.md",
    "readme": "README.md",
    "dashboard": "app.py",
    "result_registry": "docs/FINAL_RESULTS_REGISTRY.md",
    "release_audit": "docs/V16_FINAL_RELEASE_AUDIT.md",
    "v10_freeze": "docs/V10_TECHNICAL_FREEZE.md",
    "v16_development_protocol": "docs/V16_DEVELOPMENT_PROTOCOL.md",
    "v16_confirmation_protocol": "docs/V16_CONFIRMATION_PROTOCOL.md",
    "v16_retrieval_results": "docs/V16_RETRIEVAL_CONFIRMATION_RESULTS.md",
    "v16_generation_results": "docs/V16_GENERATION_CONFIRMATION_RESULTS.md",
    "v16_reference_deviation": "docs/V16_PROTOCOL_DEVIATION_REFERENCE_COMPLETENESS.md",
    "v16_technical_freeze": "docs/V16_FINAL_TECHNICAL_FREEZE.md",
    "v16_retrieval_summary": "experiments/v16_confirmation/v16_test_rankings.json",
    "v16_generation_summary": "data/splits/v16/v16_impression_gate_vs_base_confirmation.json",
    "v16_standard_nlg": "data/splits/v16/v16_impression_gate_standard_nlg_confirmation.json",
    "v16_clinical_metrics": "data/splits/v16/v16_impression_gate_clinical_metrics_confirmation.json",
    "manuscript_builder": "scripts/build_final_v16_manuscript.py",
    "docx_builder": "scripts/build_final_docx.py",
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
        "release": "v16-final-thesis-freeze",
        "generated_on": date.today().isoformat(),
        "repository": "https://github.com/yzy542968-jpg/wqf7023-medical-rag",
        "methodological_foundation": "V10 frozen alignment-controlled study",
        "final_retrieval_method": "V12 RRF Top-200 plus LambdaMART",
        "final_integrated_confirmation": "V16 section-aware MedGemma/QLoRA route",
        "acceptance": {
            "python_compileall": "passed",
            "pytest": {"passed": 315, "failed": 0, "errors": 0},
            "docx_pdf_pages": 56,
            "docx_all_pages_visually_inspected": True,
            "final_pptx_generated": False,
            "defence_slide_outline": "complete",
            "human_review": "future_work_no_independent_scores_claimed",
            "external_validation": "future_work_no_result_claimed",
        },
        "claim_boundary": (
            "Automated within-source retrieval and report-reference-consistency results only. "
            "Patient-level independence was not identifier-verified. The release does not "
            "establish physician-adjudicated similarity, diagnostic correctness, clinical "
            "safety, treatment utility, deployment performance, or external generalization."
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
