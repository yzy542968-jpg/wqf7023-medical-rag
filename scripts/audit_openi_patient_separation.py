from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identifier_sha256(values: set[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the evidence boundary for OpenI patient separation."
    )
    parser.add_argument(
        "--reports", type=Path, default=Path("data/raw/indiana_reports.csv")
    )
    parser.add_argument(
        "--projections", type=Path, default=Path("data/raw/indiana_projections.csv")
    )
    parser.add_argument(
        "--cases", type=Path, default=Path("data/processed/openi_cases.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/post_freeze_audits/openi_patient_separation_audit.json"
        ),
    )
    args = parser.parse_args()

    reports = read_csv(args.reports)
    projections = read_csv(args.projections)
    with args.cases.open("r", encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle if line.strip()]

    report_fields = set(reports[0]) if reports else set()
    projection_fields = set(projections[0]) if projections else set()
    case_fields = set(cases[0]) if cases else set()
    report_uids = {str(row.get("uid", "")).strip() for row in reports}
    projection_uids = {str(row.get("uid", "")).strip() for row in projections}
    case_ids = {str(row.get("case_id", "")).strip() for row in cases}
    expected_case_ids = {f"CXR{uid}" for uid in report_uids}
    patient_field_names = {
        "patient_id",
        "patientid",
        "subject_id",
        "subjectid",
        "person_id",
        "personid",
    }
    exposed_patient_fields = sorted(
        field
        for field in report_fields | projection_fields | case_fields
        if field.lower() in patient_field_names
    )

    if len(report_uids) != len(reports):
        raise RuntimeError("Raw report uid values are not unique.")
    if projection_uids != report_uids:
        raise RuntimeError("Projection and report uid universes differ.")
    if expected_case_ids != case_ids:
        raise RuntimeError("Processed case IDs do not map one-to-one from raw report uid values.")

    payload = {
        "audit_type": "post_freeze_evidence_boundary_audit",
        "changes_frozen_results": False,
        "source_collection": "OpenI/IU-Xray",
        "source_design_evidence": {
            "claim": "The source collection included no more than one study per patient.",
            "doi": "10.1093/jamia/ocv080",
            "source_design_patient_separation_supported": True,
        },
        "local_provenance": {
            "raw_report_count": len(reports),
            "raw_report_uid_unique_count": len(report_uids),
            "projection_count": len(projections),
            "projection_uid_unique_count": len(projection_uids),
            "processed_case_count": len(cases),
            "processed_case_id_unique_count": len(case_ids),
            "raw_uid_to_processed_case_id_exact_set_match": expected_case_ids == case_ids,
            "exposed_patient_identifier_fields": exposed_patient_fields,
        },
        "fingerprints": {
            "raw_reports_sha256": file_sha256(args.reports),
            "raw_projections_sha256": file_sha256(args.projections),
            "processed_cases_sha256": file_sha256(args.cases),
            "raw_report_uids_sha256": identifier_sha256(report_uids),
            "processed_case_ids_sha256": identifier_sha256(case_ids),
        },
        "claim_boundary": {
            "case_id_disjointness_verified": True,
            "duplicate_cluster_disjointness_verified_in_v10": True,
            "source_design_patient_separation_supported": True,
            "identifier_verified_patient_separation": False,
            "external_patient_level_generalization": False,
            "recommended_wording": (
                "Patient separation was supported by the OpenI collection design, "
                "which included no more than one study per patient, but could not be "
                "independently re-verified from released subject identifiers."
            ),
        },
        "future_work": {
            "dataset": "MIMIC-CXR/MIMIC-CXR-JPG",
            "status": "future_work_only_authorized_data_not_downloaded",
            "reason": (
                "Identifier-verified external patient-level validation requires "
                "authorized subject/study identifiers and a separately frozen protocol; "
                "the multi-terabyte source is outside the completed thesis scope."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
