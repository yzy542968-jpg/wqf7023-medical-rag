from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.qa.radrestruct_hierarchy import RadReStructHierarchy  # noqa: E402
from medical_rag.qa.structured_decoding import (  # noqa: E402
    decode_answer_probabilities,
    knn_answer_probabilities,
)
from medical_rag.qa.structured_metrics import (  # noqa: E402
    load_answer_vector,
    load_report_keys,
    stack_answer_vectors,
    structured_qa_metrics,
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


def _embedding_map(path: Path) -> tuple[dict[str, np.ndarray], str]:
    with np.load(path, allow_pickle=False) as payload:
        case_ids = [str(value) for value in payload["case_ids"]]
        embeddings = np.asarray(payload["case_image_embeddings"], dtype=np.float32)
        signature = str(payload["signature"].item())
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(embeddings).all():
        raise ValueError("MedSigLIP image embeddings are invalid")
    normalized = embeddings / norms
    return dict(zip(case_ids, normalized, strict=True)), signature


def _role_embeddings(
    role: dict[str, Any], embedding_by_case: dict[str, np.ndarray]
) -> np.ndarray:
    missing = [case["case_id"] for case in role["cases"] if case["case_id"] not in embedding_by_case]
    if missing:
        raise ValueError(f"Missing image embeddings for {len(missing)} cases")
    return np.stack([embedding_by_case[case["case_id"]] for case in role["cases"]])


def _metric_record(
    probabilities: np.ndarray,
    targets: np.ndarray,
    hierarchy: RadReStructHierarchy,
    *,
    multi_threshold: float,
    fixed_threshold: float,
) -> tuple[dict[str, Any], np.ndarray]:
    predictions = decode_answer_probabilities(
        probabilities,
        hierarchy,
        multi_choice_threshold=multi_threshold,
        fixed_choice_threshold=fixed_threshold,
    )
    return structured_qa_metrics(targets, predictions).as_dict(), predictions


def _selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -row["metrics"]["supported_label_macro_f1"],
        row["top_k"],
        0 if row["weighting"] == "uniform" else 1,
        abs(row["multi_choice_threshold"] - 0.5)
        + abs(row["fixed_choice_threshold"] - 0.5),
        row["multi_choice_threshold"],
        row["fixed_choice_threshold"],
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_json(args.config)
    manifest = _load_json(args.manifest)
    hierarchy = RadReStructHierarchy(args.radrestruct_root)
    report_keys = load_report_keys(args.radrestruct_root)
    embeddings, embedding_signature = _embedding_map(args.embeddings)

    train_role = manifest["roles"]["train"]
    calibration_role = manifest["roles"]["calibration"]
    validation_role = manifest["roles"]["validation"]
    train_targets = _targets(train_role, args.radrestruct_root, report_keys)
    calibration_targets = _targets(
        calibration_role, args.radrestruct_root, report_keys
    )
    validation_targets = _targets(validation_role, args.radrestruct_root, report_keys)
    train_embeddings = _role_embeddings(train_role, embeddings)
    calibration_embeddings = _role_embeddings(calibration_role, embeddings)
    validation_embeddings = _role_embeddings(validation_role, embeddings)
    calibration_similarity = calibration_embeddings @ train_embeddings.T
    validation_similarity = validation_embeddings @ train_embeddings.T

    grid = config["grid"]
    probability_cache: dict[tuple[int, str], np.ndarray] = {}
    calibration_grid: list[dict[str, Any]] = []
    for top_k in grid["top_k"]:
        for weighting in grid["weighting"]:
            cache_key = (int(top_k), str(weighting))
            probability_cache[cache_key] = knn_answer_probabilities(
                calibration_similarity,
                train_targets,
                top_k=int(top_k),
                weighting=str(weighting),
                softmax_temperature=float(grid["cosine_softmax_temperature"]),
            )
            for multi_threshold in grid["multi_choice_threshold"]:
                for fixed_threshold in grid["fixed_choice_threshold"]:
                    metrics, _ = _metric_record(
                        probability_cache[cache_key],
                        calibration_targets,
                        hierarchy,
                        multi_threshold=float(multi_threshold),
                        fixed_threshold=float(fixed_threshold),
                    )
                    calibration_grid.append(
                        {
                            "top_k": int(top_k),
                            "weighting": str(weighting),
                            "multi_choice_threshold": float(multi_threshold),
                            "fixed_choice_threshold": float(fixed_threshold),
                            "metrics": metrics,
                        }
                    )
    selected = min(calibration_grid, key=_selection_key)
    validation_probabilities = knn_answer_probabilities(
        validation_similarity,
        train_targets,
        top_k=selected["top_k"],
        weighting=selected["weighting"],
        softmax_temperature=float(grid["cosine_softmax_temperature"]),
    )
    validation_metrics, _ = _metric_record(
        validation_probabilities,
        validation_targets,
        hierarchy,
        multi_threshold=selected["multi_choice_threshold"],
        fixed_threshold=selected["fixed_choice_threshold"],
    )
    summary = {
        "study": config["study"],
        "status": "development_only_no_test",
        "config": "config/final_qa_structured_history_diagnostic.json",
        "embedding_signature": embedding_signature,
        "train_bank_case_count": len(train_role["cases"]),
        "calibration_case_count": len(calibration_role["cases"]),
        "validation_case_count": len(validation_role["cases"]),
        "calibration_grid": calibration_grid,
        "selected_on_calibration": selected,
        "single_validation_result": {
            "configuration": {
                key: selected[key]
                for key in (
                    "top_k",
                    "weighting",
                    "multi_choice_threshold",
                    "fixed_choice_threshold",
                )
            },
            "metrics": validation_metrics,
        },
        "interpretation_boundary": config["interpretation"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/final_qa_structured_history_diagnostic.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/splits/final_qa/final_qa_development_manifest.json",
    )
    parser.add_argument("--radrestruct-root", type=Path, required=True)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT / "data/processed/v10_medsiglip_embeddings.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "experiments/final_qa_development/oracle_structured_history_summary.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "selected_on_calibration": result["selected_on_calibration"],
                "single_validation_result": result["single_validation_result"],
                "interpretation_boundary": result["interpretation_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
