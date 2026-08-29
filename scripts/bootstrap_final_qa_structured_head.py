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
    bootstrap_supported_macro_f1_difference,
    load_answer_vector,
    load_report_keys,
    stack_answer_vectors,
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_json(args.manifest)
    keys = load_report_keys(args.radrestruct_root)
    role = manifest["roles"]["validation"]
    targets = stack_answer_vectors(
        load_answer_vector(
            args.radrestruct_root
            / f"{case['official_split']}_vectorized_answers"
            / f"{case['source_report_id']}.json",
            keys,
        )
        for case in role["cases"]
    )
    with np.load(args.history_predictions, allow_pickle=False) as payload:
        history_ids = [str(value) for value in payload["case_ids"]]
        history = np.asarray(payload["predictions"], dtype=np.uint8)
    with np.load(args.no_history_predictions, allow_pickle=False) as payload:
        no_history_ids = [str(value) for value in payload["case_ids"]]
        no_history = np.asarray(payload["predictions"], dtype=np.uint8)
    expected_ids = [case["case_id"] for case in role["cases"]]
    if history_ids != expected_ids or no_history_ids != expected_ids:
        raise ValueError("Prediction case order does not match the Validation manifest")
    bootstrap = bootstrap_supported_macro_f1_difference(
        targets,
        history,
        no_history,
        samples=args.samples,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )
    result = {
        "study": "Final QA structured-head paired case bootstrap",
        "status": "development_validation_no_test",
        "contrast": "same_model_top1_paired_report_embedding minus same_model_no_history",
        "bootstrap": bootstrap,
        "interpretation": (
            "A confidence interval spanning zero does not establish a historical-input "
            "benefit. This secondary reconstruction result cannot replace independent QA."
        ),
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radrestruct-root", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/splits/final_qa/final_qa_development_manifest.json"
    )
    parser.add_argument(
        "--history-predictions", type=Path, default=ROOT / "experiments/final_qa_development/structured_head_predictions/same_model_top1_paired_report_embedding.npz"
    )
    parser.add_argument(
        "--no-history-predictions", type=Path, default=ROOT / "experiments/final_qa_development/structured_head_predictions/same_model_no_history.npz"
    )
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7023)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "experiments/final_qa_development/structured_head_bootstrap.json"
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
