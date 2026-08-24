from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.evaluation.v10_clinical_review import (  # noqa: E402
    build_blinded_review_rows,
    public_review_rows,
)
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from run_v10_evidence_generation_development import read_json, read_jsonl  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "v10_clinical_review.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_QA = ROOT / "experiments" / "v10_publication" / "v10_confirmation_qa_rows.jsonl"
DEFAULT_RETRIEVAL = ROOT / "experiments" / "v10_publication" / "v10_confirmation_retrieval_rows.jsonl"
DEFAULT_QA_SUMMARY = ROOT / "data" / "splits" / "v10" / "v10_confirmation_qa_summary.json"
DEFAULT_RETRIEVAL_SUMMARY = (
    ROOT / "data" / "splits" / "v10" / "v10_confirmation_retrieval_summary.json"
)
DEFAULT_PUBLIC = ROOT / "experiments" / "v10_publication" / "v10_clinical_review_public.csv"
DEFAULT_PRIVATE = ROOT / "experiments" / "v10_publication" / "v10_clinical_review_private_key.csv"
DEFAULT_METADATA = ROOT / "experiments" / "v10_publication" / "v10_clinical_reviewer_metadata.json"
DEFAULT_SUMMARY = ROOT / "data" / "splits" / "v10" / "v10_clinical_review_package_summary.json"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pending V10 independent review package.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--qa-rows", type=Path, default=DEFAULT_QA)
    parser.add_argument("--retrieval-rows", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--qa-summary", type=Path, default=DEFAULT_QA_SUMMARY)
    parser.add_argument("--retrieval-summary", type=Path, default=DEFAULT_RETRIEVAL_SUMMARY)
    parser.add_argument("--public-output", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--private-output", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    config = read_json(args.config)
    if config["review_status"] != "pending_independent_review":
        raise RuntimeError("clinical review config is not pending")
    qa_summary = read_json(args.qa_summary)
    retrieval_summary = read_json(args.retrieval_summary)
    if file_sha256(args.qa_rows) != str(qa_summary["qa_rows_sha256"]):
        raise RuntimeError("QA rows differ from their completed confirmation summary")
    if file_sha256(args.retrieval_rows) != str(retrieval_summary["retrieval_rows_sha256"]):
        raise RuntimeError("retrieval rows differ from their completed confirmation summary")
    cases = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    qa_rows = read_jsonl(args.qa_rows)
    retrieval_rows = read_jsonl(args.retrieval_rows)
    qa = {
        (str(row["case_id"]), str(row["question_type"]), str(row["system"])): row
        for row in qa_rows
    }
    retrieval = {
        (str(row["case_id"]), str(row["question_type"])): row
        for row in retrieval_rows
        if row["system"] == "r5_fact_attention"
    }
    candidate_ids = sorted(
        {case_id for case_id, _, _ in qa},
        key=lambda case_id: (
            hashlib.sha256(
                f"v10-clinical-case|{config['seed']}|{case_id}".encode("utf-8")
            ).hexdigest(),
            case_id,
        ),
    )[: int(config["case_count"])]
    review_cases = []
    for position, case_id in enumerate(candidate_ids, start=1):
        question_type = "findings" if position % 2 == 1 else "impression"
        source = cases[case_id]
        answers = {}
        evidence = {}
        for system in config["systems"]:
            row = qa[(case_id, question_type, system)]
            answers[system] = str(row["answer"])
            evidence[system] = "\n".join(
                f"[{item['case_id']}:{item['section']}] {item['statement']}"
                for item in row.get("historical_support", [])
            ) or "No historical evidence used."
        review_cases.append(
            {
                "case_id": case_id,
                "question": str(qa[(case_id, question_type, config["systems"][0])]["question"]),
                "indication": str(source.get("indication", "")),
                "target_image_reference": str(
                    qa[(case_id, question_type, config["systems"][0])]["target_image_path"]
                ),
                "answers": answers,
                "retrieval": evidence,
            }
        )
    rows = build_blinded_review_rows(
        review_cases,
        system_names=config["systems"],
        case_count=int(config["case_count"]),
        seed=int(config["seed"]),
    )
    public = public_review_rows(rows)
    private = [
        {
            "package_case_id": row["package_case_id"],
            "presentation_code": row["presentation_code"],
            "system_key_private": row["system_key_private"],
        }
        for row in rows
    ]
    write_csv(args.public_output, public)
    write_csv(args.private_output, private)
    metadata = {
        "review_status": "pending_independent_review",
        "reviewer_name_or_code": "",
        "reviewer_role": "",
        "reviewer_specialty": "",
        "years_radiology_experience": "",
        "review_date": "",
        "exclusions": "",
        "missingness_note": "",
        "independent_of_system_development": "",
    }
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    summary = {
        "study": "V10 independent clinical review package",
        "status": "pending_independent_review",
        "case_count": len(candidate_ids),
        "presentation_rows": len(rows),
        "findings_cases": (len(candidate_ids) + 1) // 2,
        "impression_cases": len(candidate_ids) // 2,
        "public_package_sha256": file_sha256(args.public_output),
        "private_key_sha256": file_sha256(args.private_output),
        "reviewer_metadata_sha256": file_sha256(args.metadata_output),
        "qa_rows_sha256": file_sha256(args.qa_rows),
        "retrieval_rows_sha256": file_sha256(args.retrieval_rows),
        "reviewer_ratings_fabricated": False,
        "claim_boundary": "Package preparation is complete; independent clinical review has not been conducted.",
    }
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
