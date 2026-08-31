"""Freeze the complete-case V17 Calibration generation pilot manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.similar_case.v17_question_conditioned import select_complete_case_pilot  # noqa: E402


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _hash_ids(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(set(values))).encode("utf-8")).hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_json(args.config)
    retrieval_summary = _load_json(args.retrieval_summary)
    if not retrieval_summary.get("generation_authorized"):
        raise RuntimeError("V17 retrieval did not authorize generation")
    rows = _read_jsonl(args.retrieval_rows)
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    sampling = config["sampling"]
    selected_ids = select_complete_case_pilot(
        {case_id: len(values) for case_id, values in by_case.items()},
        domain=str(sampling["domain"]),
        seed=int(config["seed"]),
        target_questions=int(sampling["target_question_count"]),
        maximum_questions=int(sampling["maximum_question_count"]),
    )
    selected_rows = [row for case_id in selected_ids for row in by_case[case_id]]
    strata = Counter(str(row["stratum"]) for row in selected_rows)
    manifest = {
        "study": "V17 matched historical-evidence generation pilot manifest",
        "status": "frozen_before_generation",
        "data_role": "final_qa_calibration",
        "test_accessed": False,
        "selection_domain": sampling["domain"],
        "seed": int(config["seed"]),
        "case_count": len(selected_ids),
        "question_count": len(selected_rows),
        "stratum_counts": dict(sorted(strata.items())),
        "case_ids": sorted(selected_ids),
        "case_ids_sha256": _hash_ids(selected_ids),
        "retrieval_summary_sha256": hashlib.sha256(args.retrieval_summary.read_bytes()).hexdigest(),
        "retrieval_rows_sha256": hashlib.sha256(args.retrieval_rows.read_bytes()).hexdigest(),
        "selection_rule": "SHA-256 ordered complete cases until at least 2500 and at most 3000 questions",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/v17_generation_pilot.json")
    parser.add_argument("--retrieval-summary", type=Path, default=ROOT / "experiments/v17_exploratory/v17_retrieval_calibration_summary.json")
    parser.add_argument("--retrieval-rows", type=Path, default=ROOT / "experiments/v17_exploratory/v17_retrieval_calibration_rows.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/splits/final_qa/v17_generation_pilot_manifest.json")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))

