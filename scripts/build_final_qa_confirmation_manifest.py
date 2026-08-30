"""Instantiate the withheld Final-QA Test role after protocol freeze."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def _sha256_ids(values: Iterable[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(set(values))).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    development = json.loads(args.development_manifest.read_text(encoding="utf-8"))
    with args.mapping_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = sorted(
        (row for row in rows if row["v10_partition"] == "test"),
        key=lambda row: row["case_id"],
    )
    case_ids = {row["case_id"] for row in selected}
    cluster_ids = {row["v10_cluster_id"] for row in selected}
    prior_case_ids = {
        str(case["case_id"])
        for role in development["roles"].values()
        for case in role["cases"]
    }
    prior_cluster_ids = {
        str(case["cluster_id"])
        for role in development["roles"].values()
        for case in role["cases"]
    }
    case_overlap = len(case_ids & prior_case_ids)
    cluster_overlap = len(cluster_ids & prior_cluster_ids)
    if case_overlap or cluster_overlap:
        raise RuntimeError("Final-QA Test overlaps development cases or clusters")
    cases = [
        {
            "case_id": row["case_id"],
            "source_report_id": row["source_report_id"],
            "official_split": row["official_split"],
            "cluster_id": row["v10_cluster_id"],
            "question_count": int(row["question_count"]),
            "indication_available": row["indication_available"].lower() == "true",
            "findings_available": row["findings_available"].lower() == "true",
            "impression_available": row["impression_available"].lower() == "true",
        }
        for row in selected
    ]
    role = {
        "case_count": len(cases),
        "qa_row_count": sum(case["question_count"] for case in cases),
        "case_ids_sha256": _sha256_ids(case_ids),
        "cluster_ids_sha256": _sha256_ids(cluster_ids),
        "official_split_counts": dict(
            Counter(case["official_split"] for case in cases)
        ),
        "cases": cases,
    }
    if role["case_count"] != int(config["expected_case_count"]):
        raise RuntimeError("Instantiated Test case count differs from protocol")
    if role["qa_row_count"] != int(config["expected_question_count"]):
        raise RuntimeError("Instantiated Test question count differs from protocol")
    manifest = {
        "study": config["study"],
        "status": "confirmation_manifest_instantiated_no_outcomes",
        "protocol_commit": str(config["protocol_commit"]),
        "development_manifest_sha256": _sha256_file(args.development_manifest),
        "mapping_csv_sha256": _sha256_file(args.mapping_csv),
        "final_gate_policy_sha256": _sha256_file(args.final_gate_policy),
        "roles": {
            "train": development["roles"]["train"],
            "test": role,
        },
        "overlap_checks": {
            "test_development_case_overlap": case_overlap,
            "test_development_cluster_overlap": cluster_overlap,
            "all_zero": True,
        },
        "outcomes_generated": False,
        "outcomes_inspected": False,
        "boundary": config["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/final_qa_confirmation.json"
    )
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=ROOT / "data/splits/final_qa/final_qa_development_manifest.json",
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=ROOT / "data/splits/final_qa/final_qa_case_mapping.csv",
    )
    parser.add_argument(
        "--final-gate-policy",
        type=Path,
        default=ROOT / "data/splits/final_qa/final_qa_final_gate_policy.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/splits/final_qa/final_qa_confirmation_manifest.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
