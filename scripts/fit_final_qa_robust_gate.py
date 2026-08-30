"""Fit the frozen final question policy on all Final-QA development outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_final_qa_nested_robust_gate import _fit_utilities  # noqa: E402
from develop_final_qa_real_output_gate import _question_ids  # noqa: E402
from evaluate_final_qa_full_role import (  # noqa: E402
    B3,
    B6,
    CONDITIONS,
    cases_for_role,
    keyed_full,
)
from evaluate_final_qa_qlora_pilot import read_jsonl  # noqa: E402
from medical_rag.qa.radrestruct_hierarchy import RadReStructHierarchy  # noqa: E402
from medical_rag.qa.question_vectorizer import RadReStructQuestionVectorizer  # noqa: E402


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = read_jsonl(args.rows)
    rows_by_condition = {
        condition: keyed_full(rows, condition) for condition in CONDITIONS
    }
    hierarchy = RadReStructHierarchy(args.radrestruct_root)
    vectorizer = RadReStructQuestionVectorizer(hierarchy)
    cases = cases_for_role(args.radrestruct_root, manifest, "validation")
    qids = _question_ids(cases, vectorizer)
    utilities = _fit_utilities(sorted(qids), qids, rows_by_condition)
    minimum_support = int(config["final_policy"]["minimum_support"])
    minimum_margin = float(config["final_policy"]["minimum_macro_margin"])
    records = []
    for question_id in sorted(hierarchy.indices_by_question):
        support, margin = utilities.get(question_id, (0, float("-inf")))
        use_history = support >= minimum_support and margin >= minimum_margin
        records.append(
            {
                "question_id": int(question_id),
                "development_support": int(support),
                "b6_minus_b3_option_label_macro_f1": (
                    float(margin) if support else None
                ),
                "source": B6 if use_history else B3,
            }
        )
    payload = {
        "study": config["study"],
        "status": "final_development_fit_frozen_before_test_manifest",
        "source_role": "Final-QA Validation reused as development",
        "test_accessed": False,
        "minimum_support": minimum_support,
        "minimum_macro_margin": minimum_margin,
        "question_id_count": len(records),
        "history_question_id_count": sum(row["source"] == B6 for row in records),
        "image_only_question_id_count": sum(row["source"] == B3 for row in records),
        "question_policy": records,
        "input_hashes": {
            "confirmation_config_sha256": _sha256_file(args.config),
            "development_manifest_sha256": _sha256_file(args.manifest),
            "validation_rows_sha256": _sha256_file(args.rows),
        },
        "boundary": config["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/final_qa_confirmation.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/splits/final_qa/final_qa_development_manifest.json",
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=ROOT
        / "experiments/final_qa_development/final_qa_validation_rows.jsonl",
    )
    parser.add_argument("--radrestruct-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/splits/final_qa/final_qa_final_gate_policy.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
