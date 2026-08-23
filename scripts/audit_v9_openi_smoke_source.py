from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.similar_case.openi_adapter import read_openi_paired_cases


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit OpenI for V9 engineering smoke use.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/processed/openi_cases.jsonl"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = read_openi_paired_cases(args.cases)
    index_counts = {
        label: sum(case.metadata.get("report_index_class") == label for case in cases)
        for label in ("normal", "abnormal", "indeterminate")
    }
    payload = {
        "status": "engineering_smoke_only_not_v9_confirmation_eligible",
        "source_path": str(args.cases),
        "source_sha256": sha256_file(args.cases),
        "case_count": len(cases),
        "case_with_images_count": sum(bool(case.image_paths) for case in cases),
        "case_with_indication_count": sum(bool(case.indication) for case in cases),
        "case_with_findings_count": sum(bool(case.findings) for case in cases),
        "case_with_impression_count": sum(bool(case.impression) for case in cases),
        "case_with_active_problem_labels_count": sum(bool(case.labels) for case in cases),
        "report_index_class_counts": index_counts,
        "graded_qrels_exclude_report_index_indeterminate": True,
        "reliable_patient_ids_available": False,
        "patient_level_confirmation_eligible": False,
        "confirmation_ids_instantiated": False,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
