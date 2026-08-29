from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_ROLES = ("train", "calibration", "validation")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_ids(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(values))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    mapping_rows: list[dict[str, str]] = []
    with args.mapping_csv.open("r", encoding="utf-8", newline="") as handle:
        mapping_rows.extend(csv.DictReader(handle))
    split = _load_json(args.v10_split)
    historical_case_ids = tuple(split["partitions"]["train"]["case_ids"])
    historical_clusters = {
        cluster["cluster_id"]
        for cluster in split["clusters"]
        if any(case_id in set(historical_case_ids) for case_id in cluster["case_ids"])
    }

    role_payloads: dict[str, Any] = {}
    role_case_sets: dict[str, set[str]] = {}
    role_cluster_sets: dict[str, set[str]] = {}
    for role in DEVELOPMENT_ROLES:
        selected = sorted(
            (row for row in mapping_rows if row["v10_partition"] == role),
            key=lambda row: row["case_id"],
        )
        case_ids = {row["case_id"] for row in selected}
        cluster_ids = {row["v10_cluster_id"] for row in selected}
        role_case_sets[role] = case_ids
        role_cluster_sets[role] = cluster_ids
        role_payloads[role] = {
            "case_count": len(selected),
            "qa_row_count": sum(int(row["question_count"]) for row in selected),
            "case_ids_sha256": _sha256_ids(case_ids),
            "cluster_ids_sha256": _sha256_ids(cluster_ids),
            "official_split_counts": dict(
                Counter(row["official_split"] for row in selected)
            ),
            "cases": [
                {
                    "case_id": row["case_id"],
                    "source_report_id": row["source_report_id"],
                    "official_split": row["official_split"],
                    "cluster_id": row["v10_cluster_id"],
                    "question_count": int(row["question_count"]),
                    "indication_available": row["indication_available"].lower()
                    == "true",
                    "findings_available": row["findings_available"].lower() == "true",
                    "impression_available": row["impression_available"].lower()
                    == "true",
                }
                for row in selected
            ],
        }

    overlap: dict[str, int] = {}
    for index, left in enumerate(DEVELOPMENT_ROLES):
        for right in DEVELOPMENT_ROLES[index + 1 :]:
            overlap[f"{left}_{right}_case_overlap"] = len(
                role_case_sets[left] & role_case_sets[right]
            )
            overlap[f"{left}_{right}_cluster_overlap"] = len(
                role_cluster_sets[left] & role_cluster_sets[right]
            )
    if any(overlap.values()):
        raise ValueError(f"Development roles are not case/cluster disjoint: {overlap}")

    validation_history_cluster_overlap = len(
        (role_cluster_sets["calibration"] | role_cluster_sets["validation"])
        & historical_clusters
    )
    if validation_history_cluster_overlap:
        raise ValueError("Calibration/Validation clusters overlap the historical Train bank")

    test_rows = [row for row in mapping_rows if row["v10_partition"] == "test"]
    manifest = {
        "study": "Final QA development manifest",
        "status": "development_roles_only_no_test_outcomes",
        "protocol_commit": "ba5ac18",
        "mapping_csv": "data/splits/final_qa/final_qa_case_mapping.csv",
        "mapping_csv_sha256": _sha256_file(args.mapping_csv),
        "v10_split": "data/splits/v10/v10_cluster_disjoint_split.json",
        "v10_split_sha256": _sha256_file(args.v10_split),
        "historical_bank": {
            "role": "v10_train",
            "case_count": len(historical_case_ids),
            "case_ids_sha256": _sha256_ids(historical_case_ids),
            "train_target_exclusion": "exclude target case and complete target cluster",
        },
        "roles": role_payloads,
        "overlap_checks": {
            **overlap,
            "calibration_validation_cluster_overlap_with_historical_bank": validation_history_cluster_overlap,
            "all_zero": True,
        },
        "withheld_confirmation_role": {
            "case_count": len(test_rows),
            "qa_row_count": sum(int(row["question_count"]) for row in test_rows),
            "outcomes_generated": False,
            "outcomes_inspected": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=ROOT / "data/splits/final_qa/final_qa_case_mapping.csv",
    )
    parser.add_argument(
        "--v10-split",
        type=Path,
        default=ROOT / "data/splits/v10/v10_cluster_disjoint_split.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/splits/final_qa/final_qa_development_manifest.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = build(parse_args())
    print(json.dumps({key: value for key, value in result.items() if key != "roles"}, indent=2))
