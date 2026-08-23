from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.similar_case.chexpert_plus_adapter import (  # noqa: E402
    read_chexpert_plus_cases,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_identifiers(values: list[str]) -> str:
    payload = "\n".join(sorted(set(values))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit an official CheXpert Plus source before V9 splitting."
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--chexbert-labels", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = read_chexpert_plus_cases(
        args.csv,
        image_root=args.image_root,
        chexbert_labels_path=args.chexbert_labels,
        require_image_files=False,
    )
    image_paths = [Path(path) for case in cases for path in case.image_paths]
    missing_images = sum(not path.is_file() for path in image_paths)
    patients = [str(case.patient_id) for case in cases if case.patient_id is not None]
    labels_available = sum(
        case.metadata.get("label_annotation_available") is True for case in cases
    )
    radgraph_available = sum(
        case.metadata.get("radgraph_annotation_available") is True for case in cases
    )
    payload = {
        "status": "source_readiness_audit_not_confirmation_cohort",
        "confirmation_ids_instantiated": False,
        "csv_path": str(args.csv),
        "csv_sha256": sha256_file(args.csv),
        "chexbert_labels_path": (
            str(args.chexbert_labels) if args.chexbert_labels is not None else None
        ),
        "chexbert_labels_sha256": (
            sha256_file(args.chexbert_labels)
            if args.chexbert_labels is not None
            else None
        ),
        "study_count": len(cases),
        "patient_count": len(set(patients)),
        "image_view_count": len(image_paths),
        "missing_image_count": missing_images,
        "study_ids_sha256": sha256_identifiers(
            [case.study_id for case in cases]
        ),
        "patient_ids_sha256": sha256_identifiers(patients),
        "study_with_chexbert_labels_count": labels_available,
        "study_with_radgraph_facts_count": radgraph_available,
        "patient_id_complete": len(patients) == len(cases),
        "graded_qrels_ready": (
            bool(cases)
            and labels_available == len(cases)
            and radgraph_available == len(cases)
        ),
        "image_files_complete": missing_images == 0,
    }
    output = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
