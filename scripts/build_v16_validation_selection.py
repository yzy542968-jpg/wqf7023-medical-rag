"""Build a deterministic V16 Validation selection manifest.

The existing V12 generation screening manifest contains 48 cases.  This
utility derives the broader manifest from the already-frozen V12 Validation
ranking rows, retaining only cases in the V10 Validation partition.  It does
not inspect generated answers or outcomes and does not select a new model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical_problem_label(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def spectrum_for_case(case: dict[str, Any]) -> str:
    problems = canonical_problem_label(case.get("problems"))
    if problems == "normal":
        return "report_indexed_normal"
    if problems == "no indexing":
        return "report_index_indeterminate"
    return "report_indexed_abnormal"


def selection_digest(case_id: str, seed: int) -> str:
    payload = f"v16-validation-selection|{seed}|{case_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run(args: argparse.Namespace) -> None:
    split = read_json(args.split)
    validation_ids = {
        str(value) for value in split["partitions"]["validation"]["case_ids"]
    }
    cases = {
        str(row["case_id"]): row
        for row in read_jsonl(args.cases)
    }
    ranking_rows = read_jsonl(args.ranking_rows)
    ranking_cases = {
        str(row["case_id"])
        for row in ranking_rows
        if str(row.get("question_type")) in {"findings", "impression"}
    }
    unexpected = sorted(ranking_cases - validation_ids)
    missing_case_records = sorted(ranking_cases - set(cases))
    if unexpected:
        raise RuntimeError(f"Ranking rows contain non-Validation cases: {unexpected}")
    if missing_case_records:
        raise RuntimeError(f"Ranking cases missing from source cases: {missing_case_records}")
    if len(ranking_cases) != args.expected_case_count:
        raise RuntimeError(
            f"Expected {args.expected_case_count} ranking cases, found {len(ranking_cases)}"
        )
    question_counts = {
        case_id: sum(
            str(row["case_id"]) == case_id
            and str(row.get("question_type")) in {"findings", "impression"}
            for row in ranking_rows
        )
        for case_id in ranking_cases
    }
    incomplete = sorted(case_id for case_id, count in question_counts.items() if count != 2)
    if incomplete:
        raise RuntimeError(f"Ranking rows do not cover findings/impression for: {incomplete}")

    rows = [
        {
            "case_id": case_id,
            "selection_digest": selection_digest(case_id, args.seed),
            "spectrum": spectrum_for_case(cases[case_id]),
        }
        for case_id in sorted(ranking_cases)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    summary = {
        "study": "V16 broader Validation generation manifest",
        "status": "derived_from_frozen_v12_validation_rankings",
        "no_test_evaluation": True,
        "counts": {
            "v10_validation_partition_cases": len(validation_ids),
            "ranking_cases": len(ranking_cases),
            "manifest_cases": len(rows),
            "ranking_rows": len(ranking_rows),
        },
        "inputs": {
            "cases_sha256": file_sha256(args.cases),
            "split_sha256": file_sha256(args.split),
            "ranking_rows_sha256": file_sha256(args.ranking_rows),
        },
        "configuration": {
            "seed": args.seed,
            "selection_digest_domain": "v16-validation-selection",
            "question_types_required": ["findings", "impression"],
        },
        "output_sha256": file_sha256(args.output),
        "claim_boundary": (
            "Manifest construction only; no generated answer, Test outcome, or model result "
            "was inspected for case selection."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--split", type=Path, default=ROOT / "data/splits/v10/v10_cluster_disjoint_split.json")
    parser.add_argument(
        "--ranking-rows",
        type=Path,
        default=ROOT / "experiments/v12_optimization/retrieval/v12_qwen3_validation_rankings_rows.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-case-count", type=int, default=376)
    parser.add_argument("--seed", type=int, default=1616)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
