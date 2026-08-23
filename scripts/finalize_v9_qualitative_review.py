from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_PACK = (
    ROOT / "experiments" / "post_submission_v9" / "v9_qualitative_review_pack.jsonl"
)
DEFAULT_PUBLIC_INDEX = (
    ROOT / "data" / "splits" / "v9" / "v9_qualitative_case_index.csv"
)
DEFAULT_SUMMARY = (
    ROOT / "data" / "splits" / "v9" / "v9_qualitative_review_summary.json"
)
REVIEW_NOTE = (
    "Accepted the assistant-proposed V9 qualitative labels without modification; "
    "this researcher review is exploratory and is not independent clinical adjudication."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def accept_proposals(
    rows: Sequence[Mapping[str, Any]], *, reviewer_initials: str, review_date: str
) -> list[dict[str, Any]]:
    if len(rows) != 24:
        raise ValueError(f"Expected the frozen 24-case review pack, found {len(rows)} rows.")
    if not reviewer_initials.strip() or not review_date.strip():
        raise ValueError("Reviewer initials and review date are required.")

    accepted: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        proposals = [str(value) for value in row.get("assistant_proposed_labels_v1_0", [])]
        if not proposals:
            raise ValueError(f"Case {row.get('case_id')} has no assistant proposal to accept.")
        row["researcher_reviewed_labels_v1_0"] = proposals
        row["review_status"] = "accepted"
        row["review_note"] = REVIEW_NOTE
        row["reviewer_initials"] = reviewer_initials.strip()
        row["review_date"] = review_date.strip()
        accepted.append(row)
    return accepted


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows
    )
    path.write_text(payload + "\n", encoding="utf-8", newline="\n")


def write_public_index(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "case_id",
        "selection_category",
        "selection_reason",
        "assistant_proposed_labels_v1_0",
        "researcher_reviewed_labels_v1_0",
        "review_status",
        "review_note",
        "reviewer_initials",
        "review_date",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "selection_category": row["selection_category"],
                    "selection_reason": row["selection_reason"],
                    "assistant_proposed_labels_v1_0": ";".join(
                        row["assistant_proposed_labels_v1_0"]
                    ),
                    "researcher_reviewed_labels_v1_0": ";".join(
                        row["researcher_reviewed_labels_v1_0"]
                    ),
                    "review_status": row["review_status"],
                    "review_note": row["review_note"],
                    "reviewer_initials": row["reviewer_initials"],
                    "review_date": row["review_date"],
                }
            )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_summary(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    local_pack: Path,
    public_index: Path,
) -> None:
    labels = Counter(
        label
        for row in rows
        for label in row["researcher_reviewed_labels_v1_0"]
    )
    categories = Counter(str(row["selection_category"]) for row in rows)
    payload = {
        "study": "v9_post_hoc_qualitative_review",
        "status": "researcher_review_complete",
        "review_type": "researcher acceptance of assistant-proposed exploratory labels",
        "independent_clinical_adjudication": False,
        "reviewed_case_count": len(rows),
        "accepted_without_modification": len(rows),
        "modified": 0,
        "excluded": 0,
        "reviewer_initials": rows[0]["reviewer_initials"],
        "review_date": rows[0]["review_date"],
        "selection_category_counts": dict(sorted(categories.items())),
        "reviewed_label_counts": dict(sorted(labels.items())),
        "local_pack_sha256": file_sha256(local_pack),
        "public_index_sha256": file_sha256(public_index),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record the researcher's acceptance of all frozen V9 proposals."
    )
    parser.add_argument("--local-pack", type=Path, default=DEFAULT_LOCAL_PACK)
    parser.add_argument("--public-index", type=Path, default=DEFAULT_PUBLIC_INDEX)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--reviewer-initials", default="ZY")
    parser.add_argument("--review-date", default="2026-08-19")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accepted = accept_proposals(
        read_jsonl(args.local_pack),
        reviewer_initials=args.reviewer_initials,
        review_date=args.review_date,
    )
    write_jsonl(args.local_pack, accepted)
    write_public_index(args.public_index, accepted)
    write_summary(
        args.summary,
        accepted,
        local_pack=args.local_pack,
        public_index=args.public_index,
    )
    print(
        json.dumps(
            {
                "accepted": len(accepted),
                "reviewer_initials": args.reviewer_initials,
                "review_date": args.review_date,
                "local_pack": str(args.local_pack),
                "public_index": str(args.public_index),
                "summary": str(args.summary),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
