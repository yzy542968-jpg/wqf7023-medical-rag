from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.qa.radrestruct_hierarchy import RadReStructHierarchy  # noqa: E402
from medical_rag.qa.structured_metrics import (  # noqa: E402
    fit_label_majority,
    load_answer_vector,
    load_report_keys,
    repeat_prediction,
    stack_answer_vectors,
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _targets(
    role: dict[str, Any], rad_root: Path, report_keys: tuple[str, ...]
) -> np.ndarray:
    return stack_answer_vectors(
        load_answer_vector(
            rad_root
            / f"{case['official_split']}_vectorized_answers"
            / f"{case['source_report_id']}.json",
            report_keys,
        )
        for case in role["cases"]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    official_root = args.official_repository.resolve()
    manifest = _load_json(args.manifest)
    report_keys = load_report_keys(args.radrestruct_root)
    train = _targets(manifest["roles"]["train"], args.radrestruct_root, report_keys)
    validation = _targets(
        manifest["roles"]["validation"], args.radrestruct_root, report_keys
    )
    majority = repeat_prediction(fit_label_majority(train), len(validation))
    audit_rows = min(args.rows_per_condition, len(validation))
    candidates = {
        "validation_reference_vectors": validation[:audit_rows],
        "train_majority_predictions": majority[:audit_rows],
        "all_zero_predictions": np.zeros_like(validation[:audit_rows]),
    }

    local = RadReStructHierarchy(args.radrestruct_root)
    original_cwd = Path.cwd()
    sys.path.insert(0, str(official_root))
    try:
        os.chdir(official_root)
        from evaluation.evaluator_radrestruct import AutoregressiveEvaluator

        official = AutoregressiveEvaluator()
        comparisons: dict[str, Any] = {}
        for name, matrix in candidates.items():
            local_cleaned = local.clean(matrix)
            official_cleaned = official.clean_preds(matrix)
            different = int(np.count_nonzero(local_cleaned != official_cleaned))
            comparisons[name] = {
                "rows": int(matrix.shape[0]),
                "elements": int(matrix.size),
                "different_elements": different,
                "exact_match": different == 0,
            }
    finally:
        os.chdir(original_cwd)

    result = {
        "official_repository": str(official_root),
        "official_commit": args.official_commit,
        "rows_per_condition": audit_rows,
        "comparison": comparisons,
        "all_exact": all(item["exact_match"] for item in comparisons.values()),
        "boundary": (
            "This audit checks hierarchy-cleaning compatibility only. Final project "
            "metrics intentionally do not reproduce the official evaluator's aggregate-row "
            "averaging behavior."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not result["all_exact"]:
        raise RuntimeError("Local hierarchy cleaning differs from the official evaluator")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repository", type=Path, required=True)
    parser.add_argument("--radrestruct-root", type=Path, required=True)
    parser.add_argument(
        "--official-commit",
        default="b293158f0c5c1c5fa27dd615c28005eb54d7b1de",
    )
    parser.add_argument("--rows-per-condition", type=int, default=8)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/splits/final_qa/final_qa_development_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "experiments/final_qa_development/radrestruct_hierarchy_compatibility.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
