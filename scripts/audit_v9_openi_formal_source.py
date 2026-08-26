from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.similar_case.openi_adapter import read_openi_paired_cases  # noqa: E402


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
        description="Audit OpenI as the provenance-patient-unique V9 source."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/processed/openi_cases.jsonl"),
    )
    parser.add_argument(
        "--prior-use-audit",
        type=Path,
        default=Path("data/splits/v8/v8_reuse_audit.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = read_openi_paired_cases(args.cases, source_unique_patient=True)
    prior_use = json.loads(args.prior_use_audit.read_text(encoding="utf-8"))
    source_sha256 = sha256_file(args.cases)
    if prior_use["source"]["sha256"] != source_sha256:
        raise RuntimeError("Prior-use audit and V9 source SHA-256 do not match.")

    patient_keys = [str(case.patient_id) for case in cases]
    report_index_counts = {
        label: sum(case.metadata.get("report_index_class") == label for case in cases)
        for label in ("normal", "abnormal", "indeterminate")
    }
    qrel_frame = [
        case
        for case in cases
        if case.metadata.get("label_annotation_available") is True
    ]
    payload = {
        "status": "formal_source_eligible_before_v9_outcomes",
        "source_path": str(args.cases),
        "source_sha256": source_sha256,
        "case_count": len(cases),
        "case_ids_sha256": sha256_identifiers([case.study_id for case in cases]),
        "provenance_patient_key_count": len(set(patient_keys)),
        "provenance_patient_keys_sha256": sha256_identifiers(patient_keys),
        "released_patient_identifiers_available": False,
        "source_collection_one_study_per_patient": True,
        "patient_identity_claim": "source-design patient uniqueness",
        "patient_identity_evidence_doi": "10.1093/jamia/ocv080",
        "source_design_patient_separation_supported": True,
        "identifier_verified_patient_separation": False,
        "case_disjointness_operationalizes_source_design_patient_separation": True,
        "report_index_class_counts": report_index_counts,
        "report_index_qrel_frame_count": len(qrel_frame),
        "report_index_qrel_frame_sha256": sha256_identifiers(
            [case.study_id for case in qrel_frame]
        ),
        "previously_untouched_eligible_frame_count": prior_use[
            "v8_confirmation_frame"
        ]["eligible_case_count"],
        "previously_untouched_eligible_frame_sha256": prior_use[
            "v8_confirmation_frame"
        ]["eligible_case_ids_sha256"],
        "prior_use_audit_path": str(args.prior_use_audit),
        "prior_use_audit_sha256": sha256_file(args.prior_use_audit),
        "radgraph_generation_complete": False,
        "confirmation_ids_instantiated": False,
        "v9_outcomes_inspected": False,
    }
    output = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
