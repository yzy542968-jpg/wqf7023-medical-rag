from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.chexbert_pathology import (  # noqa: E402
    CHEXBERT_LABELS,
    build_case_statistics,
    case_bootstrap_intervals,
    label_texts_batched,
    metrics_from_case_statistics,
    paired_case_bootstrap,
)
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402


DEFAULT_ROWS = ROOT / "experiments/v10_publication/v10_confirmation_qa_rows.jsonl"
DEFAULT_QA_SUMMARY = ROOT / "data/splits/v10/v10_confirmation_qa_summary.json"
DEFAULT_CACHE = ROOT / "experiments/v10_publication/v10_chexbert_text_label_cache.json"
DEFAULT_PER_ROW = ROOT / "experiments/v10_publication/v10_pathology_utility_rows.csv"
DEFAULT_CONDITIONS = ROOT / "data/splits/v10/v10_pathology_utility_conditions.csv"
DEFAULT_SUMMARY = ROOT / "data/splits/v10/v10_pathology_utility_summary.json"
EXPECTED_ROWS_SHA256 = "0e82b3cf5d3913fdac82f49b6742451cf095849cad88caa2c5bedb070f793944"
EXPECTED_ROW_COUNT = 4544
EXPECTED_CASE_COUNT = 568
SYSTEMS = (
    "g0_target_image",
    "g1_whole_report",
    "g2_hierarchical",
    "g3_selective",
)
QUESTION_TYPES = ("findings", "impression")
CHECKPOINT_SHA256 = "6550703c92d640e1e04d8105a7a185d76ece0f25fcbf033d292785bf22c0fde1"
PROTOCOL_COMMIT = "5773278"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def text_sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def validate_rows(rows: Sequence[Mapping[str, Any]], source_hash: str) -> None:
    if source_hash != EXPECTED_ROWS_SHA256:
        raise RuntimeError("Frozen V10 QA row hash differs from the protocol")
    if len(rows) != EXPECTED_ROW_COUNT:
        raise RuntimeError("Frozen V10 QA row count differs from the protocol")
    if len({str(row["case_id"]) for row in rows}) != EXPECTED_CASE_COUNT:
        raise RuntimeError("Frozen V10 case count differs from the protocol")
    if tuple(sorted({str(row["system"]) for row in rows})) != tuple(sorted(SYSTEMS)):
        raise RuntimeError("Frozen V10 system set differs from the protocol")
    if tuple(sorted({str(row["question_type"]) for row in rows})) != tuple(
        sorted(QUESTION_TYPES)
    ):
        raise RuntimeError("Frozen V10 question types differ from the protocol")
    expected_pairs = {(case_id, question_type) for case_id in {str(row["case_id"]) for row in rows} for question_type in QUESTION_TYPES}
    for system in SYSTEMS:
        selected = [row for row in rows if row["system"] == system]
        pairs = {(str(row["case_id"]), str(row["question_type"])) for row in selected}
        if len(selected) != EXPECTED_CASE_COUNT * len(QUESTION_TYPES) or pairs != expected_pairs:
            raise RuntimeError(f"Incomplete case/question coverage for {system}")


def resolve_checkpoint() -> Path:
    from f1chexbert.f1chexbert import CACHE_DIR

    checkpoint = Path(CACHE_DIR) / "chexbert.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(
            "F1CheXbert checkpoint is not available at its official cache location"
        )
    observed = file_sha256(checkpoint)
    if observed != CHECKPOINT_SHA256:
        raise RuntimeError("F1CheXbert checkpoint hash differs from the protocol")
    return checkpoint


def load_cache(path: Path, checkpoint_hash: str) -> dict[str, list[int]]:
    if not path.exists():
        return {}
    payload = read_json(path)
    metadata = payload.get("metadata", {})
    if (
        metadata.get("f1chexbert_version") != "0.0.2"
        or metadata.get("checkpoint_sha256") != checkpoint_hash
        or metadata.get("label_mode") != "rrg_binary"
    ):
        raise RuntimeError("CheXbert cache metadata differs from the frozen label policy")
    labels = payload.get("labels_by_text_sha256", {})
    for text_hash, values in labels.items():
        if len(text_hash) != 64 or len(values) != len(CHEXBERT_LABELS):
            raise RuntimeError("Malformed CheXbert cache row")
    return {str(key): [int(value) for value in values] for key, values in labels.items()}


def save_cache(path: Path, labels: Mapping[str, Sequence[int]], checkpoint_hash: str) -> None:
    payload = {
        "metadata": {
            "f1chexbert_version": "0.0.2",
            "checkpoint_sha256": checkpoint_hash,
            "label_mode": "rrg_binary",
            "text_retained": False,
        },
        "labels_by_text_sha256": {key: list(labels[key]) for key in sorted(labels)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def label_unique_texts(
    rows: Sequence[Mapping[str, Any]],
    *,
    cache_path: Path,
    checkpoint_hash: str,
    device: str,
    batch_size: int,
) -> dict[str, list[int]]:
    text_by_hash: dict[str, str] = {}
    for row in rows:
        for field in ("answer", "reference_answer"):
            text = str(row.get(field) or "")
            text_by_hash[text_sha256(text)] = text
    labels = load_cache(cache_path, checkpoint_hash)
    missing_hashes = [key for key in sorted(text_by_hash) if key not in labels]
    if missing_hashes:
        from f1chexbert import F1CheXbert

        labeler = F1CheXbert(device=device)
        missing_texts = [text_by_hash[key] for key in missing_hashes]
        predictions = label_texts_batched(labeler, missing_texts, batch_size=batch_size)
        for key, values in zip(missing_hashes, predictions.tolist(), strict=True):
            labels[key] = [int(value) for value in values]
        save_cache(cache_path, labels, checkpoint_hash)
    return labels


def select_scope(rows: Sequence[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    if scope == "all":
        return list(rows)
    return [row for row in rows if row["question_type"] == scope]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen V10 answers with official F1CheXbert labels."
    )
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--qa-summary", type=Path, default=DEFAULT_QA_SUMMARY)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--per-row-output", type=Path, default=DEFAULT_PER_ROW)
    parser.add_argument("--condition-output", type=Path, default=DEFAULT_CONDITIONS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=7139)
    args = parser.parse_args()

    qa_summary = read_json(args.qa_summary)
    source_hash = file_sha256(args.rows)
    if qa_summary.get("status") != "confirmation_complete_no_retuning":
        raise RuntimeError("V10 QA confirmation is not complete")
    if qa_summary.get("qa_rows_sha256") != source_hash:
        raise RuntimeError("V10 QA rows differ from their frozen summary")
    rows = read_jsonl(args.rows)
    validate_rows(rows, source_hash)
    checkpoint = resolve_checkpoint()
    labels_by_hash = label_unique_texts(
        rows,
        cache_path=args.cache,
        checkpoint_hash=file_sha256(checkpoint),
        device=args.device,
        batch_size=args.batch_size,
    )

    enriched: list[dict[str, Any]] = []
    for row in rows:
        reference_labels = np.asarray(
            labels_by_hash[text_sha256(str(row.get("reference_answer") or ""))],
            dtype=np.int8,
        )
        prediction_labels = np.asarray(
            labels_by_hash[text_sha256(str(row.get("answer") or ""))], dtype=np.int8
        )
        intersection = int(np.logical_and(reference_labels, prediction_labels).sum())
        reference_count = int(reference_labels.sum())
        prediction_count = int(prediction_labels.sum())
        enriched.append(
            {
                "system": str(row["system"]),
                "case_id": str(row["case_id"]),
                "question_type": str(row["question_type"]),
                "reference_labels": reference_labels,
                "prediction_labels": prediction_labels,
                "reference_positive_count": reference_count,
                "prediction_positive_count": prediction_count,
                "reference_positive_recall": (
                    intersection / reference_count if reference_count else None
                ),
                "predicted_positive_precision": (
                    intersection / prediction_count if prediction_count else None
                ),
                "positive_label_hamming_agreement": float(
                    (reference_labels == prediction_labels).mean()
                ),
                "reference_positive_omission_count": int(
                    np.logical_and(reference_labels == 1, prediction_labels == 0).sum()
                ),
                "predicted_positive_addition_count": int(
                    np.logical_and(reference_labels == 0, prediction_labels == 1).sum()
                ),
            }
        )

    scopes: dict[str, Any] = {}
    condition_rows: list[dict[str, Any]] = []
    for scope_index, scope in enumerate(("all", *QUESTION_TYPES)):
        scope_rows = select_scope(enriched, scope)
        statistics = {}
        systems_summary = {}
        for system_index, system in enumerate(SYSTEMS):
            selected = [row for row in scope_rows if row["system"] == system]
            stats = build_case_statistics(
                [row["case_id"] for row in selected],
                np.stack([row["reference_labels"] for row in selected]),
                np.stack([row["prediction_labels"] for row in selected]),
            )
            statistics[system] = stats
            point = metrics_from_case_statistics(stats)
            intervals = case_bootstrap_intervals(
                stats,
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed + scope_index * 100 + system_index,
            )
            point["bootstrap_95_ci"] = intervals
            systems_summary[system] = point
            if scope == "all":
                for condition, values in point["per_condition"].items():
                    condition_rows.append(
                        {"system": system, "condition": condition, **values}
                    )
        comparisons = {}
        for comparison_index, baseline in enumerate(
            ("g0_target_image", "g1_whole_report")
        ):
            comparisons[f"g2_hierarchical_minus_{baseline}"] = paired_case_bootstrap(
                statistics["g2_hierarchical"],
                statistics[baseline],
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed + 1000 + scope_index * 100 + comparison_index,
            )
        scopes[scope] = {"systems": systems_summary, "paired_comparisons": comparisons}

    args.per_row_output.parent.mkdir(parents=True, exist_ok=True)
    per_row_fields = [
        "system",
        "case_id",
        "question_type",
        "reference_labels",
        "prediction_labels",
        "reference_positive_count",
        "prediction_positive_count",
        "reference_positive_recall",
        "predicted_positive_precision",
        "positive_label_hamming_agreement",
        "reference_positive_omission_count",
        "predicted_positive_addition_count",
    ]
    with args.per_row_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_row_fields, lineterminator="\n")
        writer.writeheader()
        for row in enriched:
            output = dict(row)
            output["reference_labels"] = json.dumps(row["reference_labels"].tolist())
            output["prediction_labels"] = json.dumps(row["prediction_labels"].tolist())
            writer.writerow(output)

    args.condition_output.parent.mkdir(parents=True, exist_ok=True)
    condition_fields = [
        "system",
        "condition",
        "precision",
        "recall",
        "f1",
        "reference_positive_support",
        "predicted_positive_count",
        "true_positive_count",
        "omission_count",
        "addition_count",
    ]
    with args.condition_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=condition_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(condition_rows)

    import torch

    summary = {
        "study": "V10 post-hoc pathology utility supplement",
        "status": "posthoc_secondary_metric_complete_no_retuning",
        "protocol_commit": PROTOCOL_COMMIT,
        "source_rows_sha256": source_hash,
        "row_count": len(rows),
        "case_count": len({row["case_id"] for row in rows}),
        "labeler": {
            "name": "F1CheXbert",
            "version": importlib.metadata.version("f1chexbert"),
            "checkpoint_sha256": file_sha256(checkpoint),
            "label_mode": "rrg_binary",
            "labels": list(CHEXBERT_LABELS),
            "device": args.device,
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "evaluator_sha256": file_sha256(Path(__file__)),
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "scopes": scopes,
        "artifacts": {
            "cache_sha256": file_sha256(args.cache),
            "cache_committed": False,
            "per_row_sha256": file_sha256(args.per_row_output),
            "per_row_committed": False,
            "condition_table_sha256": file_sha256(args.condition_output),
        },
        "claim_boundary": (
            "F1CheXbert is automated report-reference pathology-label consistency, "
            "not physician-adjudicated diagnostic accuracy, clinical safety, or patient benefit."
        ),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

