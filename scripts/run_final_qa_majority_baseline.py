from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.qa.structured_metrics import (  # noqa: E402
    fit_label_majority,
    load_answer_vector,
    load_report_keys,
    repeat_prediction,
    stack_answer_vectors,
    structured_qa_metrics,
)
from medical_rag.qa.radrestruct_hierarchy import RadReStructHierarchy  # noqa: E402


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _targets_for_role(
    role: dict[str, Any], rad_root: Path, report_keys: tuple[str, ...]
):
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
    manifest = _load_json(args.manifest)
    report_keys = load_report_keys(args.radrestruct_root)
    hierarchy = RadReStructHierarchy(args.radrestruct_root)
    train_targets = _targets_for_role(
        manifest["roles"]["train"], args.radrestruct_root, report_keys
    )
    majority = fit_label_majority(train_targets)
    results: dict[str, Any] = {}
    for role in ("calibration", "validation"):
        targets = _targets_for_role(
            manifest["roles"][role], args.radrestruct_root, report_keys
        )
        predictions = repeat_prediction(majority, len(targets))
        cleaned_predictions = hierarchy.clean(predictions)
        results[role] = {
            "raw": structured_qa_metrics(targets, predictions).as_dict(),
            "hierarchy_cleaned": structured_qa_metrics(
                targets, cleaned_predictions
            ).as_dict(),
            "changed_prediction_elements": int(
                np.count_nonzero(predictions != cleaned_predictions)
            ),
        }
    summary = {
        "study": "Final QA B0 Train-majority development baseline",
        "status": "development_only_no_test",
        "protocol_commit": manifest["protocol_commit"],
        "manifest": "data/splits/final_qa/final_qa_development_manifest.json",
        "prediction": "per-label majority fitted on mapped V10 Train target vectors",
        "hierarchy_cleaning": "repository implementation adapted from the official MIT-licensed evaluator",
        "train_case_count": int(train_targets.shape[0]),
        "label_count": int(train_targets.shape[1]),
        "majority_positive_label_count": int(majority.sum()),
        "results": results,
        "interpretation_boundary": (
            "Element accuracy is inflated by sparse report vectors and is not the primary "
            "endpoint. Supported-label macro-F1 is the prespecified development selector."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/splits/final_qa/final_qa_development_manifest.json",
    )
    parser.add_argument("--radrestruct-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "experiments/final_qa_development/b0_majority_baseline_summary.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
