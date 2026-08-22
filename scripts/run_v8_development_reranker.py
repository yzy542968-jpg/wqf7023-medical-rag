from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


PROTOCOL_COMMIT = "b2ab5e0"
DEFAULT_CONFIG = ROOT / "config" / "v8_candidate_reranker.json"
DEFAULT_ROWS = ROOT / "experiments" / "post_submission_v7" / "development_retrieval_rows.jsonl"
DEFAULT_MANIFEST = ROOT / "data" / "splits" / "v7" / "v7_development_manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "post_submission_v8"

FEATURE_NAMES = [
    "text_score_normalized",
    "image_score_normalized",
    "text_rank_fraction",
    "image_rank_fraction",
    "text_image_score_difference",
    "text_top1_margin",
    "image_top1_margin",
    "question_type_findings",
    "question_type_impression",
    "question_type_summary",
]
QUESTION_TYPES = {
    "case_scoped_findings": "question_type_findings",
    "case_scoped_impression": "question_type_impression",
    "case_scoped_summary": "question_type_summary",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_order(case_id: str, domain: str, seed: int) -> str:
    return hashlib.sha256(f"{domain}|{seed}|{case_id}".encode("utf-8")).hexdigest()


def normalize_features(values: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    mean_values = values.mean(axis=0)
    scale_values = values.std(axis=0)
    scale_values = np.where(scale_values < 1e-8, 1.0, scale_values)
    return (values - mean_values) / scale_values, {
        "feature_names": FEATURE_NAMES,
        "mean": mean_values.astype(float).tolist(),
        "scale": scale_values.astype(float).tolist(),
    }


def rank_fraction(values: Sequence[float], candidate_ids: Sequence[str]) -> np.ndarray:
    order = sorted(
        range(len(candidate_ids)),
        key=lambda index: (-float(values[index]), str(candidate_ids[index])),
    )
    result = np.zeros(len(order), dtype=np.float32)
    denominator = max(len(order) - 1, 1)
    for rank, index in enumerate(order):
        result[index] = 1.0 - rank / denominator
    return result


def row_features(row: Mapping[str, Any]) -> np.ndarray:
    candidate_ids = [str(value) for value in row["candidate_case_ids"]]
    text = np.asarray(row["text_scores_normalized"], dtype=np.float32)
    image = np.asarray(row["image_scores_normalized"], dtype=np.float32)
    if len(candidate_ids) != len(text) or len(text) != len(image):
        raise ValueError("V8 retrieval row feature lengths do not agree.")
    text_rank = rank_fraction(text, candidate_ids)
    image_rank = rank_fraction(image, candidate_ids)
    text_margin = float(text[0] - text[1]) if len(text) > 1 else 0.0
    image_order = sorted(image.tolist(), reverse=True)
    image_margin = float(image_order[0] - image_order[1]) if len(image_order) > 1 else 0.0
    qtype = str(row["question_type"])
    one_hot = [1.0 if QUESTION_TYPES[qtype] == name else 0.0 for name in FEATURE_NAMES[-3:]]
    features = np.column_stack(
        [
            text,
            image,
            text_rank,
            image_rank,
            text - image,
            np.full(len(text), text_margin, dtype=np.float32),
            np.full(len(text), image_margin, dtype=np.float32),
            np.tile(np.asarray(one_hot, dtype=np.float32), (len(text), 1)),
        ]
    )
    if features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("V8 feature schema has an unexpected dimension.")
    return features.astype(np.float32)


def target_index(row: Mapping[str, Any]) -> int | None:
    target = str(row["target_case_id"])
    candidates = [str(value) for value in row["candidate_case_ids"]]
    try:
        return candidates.index(target)
    except ValueError:
        return None


def negative_indices(row: Mapping[str, Any], negative_count: int = 8) -> list[int]:
    candidates = [str(value) for value in row["candidate_case_ids"]]
    target = str(row["target_case_id"])
    text = np.asarray(row["text_scores_normalized"], dtype=np.float32)
    image = np.asarray(row["image_scores_normalized"], dtype=np.float32)
    text_order = sorted(range(len(candidates)), key=lambda i: (-float(text[i]), candidates[i]))
    image_order = sorted(range(len(candidates)), key=lambda i: (-float(image[i]), candidates[i]))
    selected: list[int] = []
    for index in text_order + image_order:
        if candidates[index] == target or index in selected:
            continue
        selected.append(index)
        if len(selected) >= negative_count:
            break
    return selected


def build_pair_dataset(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    positive_features: list[np.ndarray] = []
    negative_features: list[np.ndarray] = []
    outside = 0
    pair_count = 0
    for row in rows:
        features = row_features(row)
        positive = target_index(row)
        if positive is None:
            outside += 1
            continue
        negatives = negative_indices(row)
        for negative in negatives:
            positive_features.append(features[positive])
            negative_features.append(features[negative])
            pair_count += 1
    if not positive_features:
        raise RuntimeError("V8 development rows contain no in-shortlist positives.")
    return (
        np.asarray(positive_features, dtype=np.float32),
        np.asarray(negative_features, dtype=np.float32),
        {"pair_count": pair_count, "target_outside_shortlist_count": outside},
    )


def case_mrr(rows: Sequence[Mapping[str, Any]], scores: Sequence[np.ndarray]) -> float:
    by_case: dict[str, list[float]] = defaultdict(list)
    for row, values in zip(rows, scores, strict=True):
        candidates = [str(value) for value in row["candidate_case_ids"]]
        ranking = [
            candidate
            for candidate, _ in sorted(
                zip(candidates, values.tolist(), strict=True),
                key=lambda item: (-float(item[1]), item[0]),
            )
        ]
        target = str(row["target_case_id"])
        reciprocal = 1.0 / (ranking.index(target) + 1) if target in ranking else 0.0
        by_case[str(row["case_id"])].append(reciprocal)
    return float(np.mean([np.mean(values) for values in by_case.values()])) if by_case else 0.0


def scores_for_alpha(rows: Sequence[Mapping[str, Any]], alpha: float) -> list[np.ndarray]:
    return [
        float(alpha) * np.asarray(row["text_scores_normalized"], dtype=np.float32)
        + (1.0 - float(alpha)) * np.asarray(row["image_scores_normalized"], dtype=np.float32)
        for row in rows
    ]


class LinearScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Linear(len(FEATURE_NAMES), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class MLPScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def model_for(model_type: str) -> nn.Module:
    if model_type == "linear":
        return LinearScorer()
    if model_type == "mlp":
        return MLPScorer()
    raise ValueError(f"Unknown V8 scorer: {model_type}")


def train_model(
    model_type: str,
    learning_rate: float,
    weight_decay: float,
    train_pairs: tuple[np.ndarray, np.ndarray],
    holdout_rows: Sequence[Mapping[str, Any]],
    scaler: Mapping[str, list[float]],
    max_epochs: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    positive, negative = train_pairs
    mean_values = np.asarray(scaler["mean"], dtype=np.float32)
    scale_values = np.asarray(scaler["scale"], dtype=np.float32)
    positive_tensor = torch.from_numpy((positive - mean_values) / scale_values)
    negative_tensor = torch.from_numpy((negative - mean_values) / scale_values)
    model = model_for(model_type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_epoch = 1
    best_mrr = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    losses: list[float] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = nn.functional.softplus(-(model(positive_tensor) - model(negative_tensor))).mean()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            holdout_scores = []
            for row in holdout_rows:
                features = row_features(row)
                scaled = (features - mean_values) / scale_values
                holdout_scores.append(model(torch.from_numpy(scaled)).numpy())
        current_mrr = case_mrr(holdout_rows, holdout_scores)
        losses.append(float(loss.item()))
        if current_mrr > best_mrr + 1e-12:
            best_mrr = current_mrr
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("V8 training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_holdout_mrr": best_mrr,
        "final_loss": losses[-1],
        "losses": losses,
    }


def evaluate_candidates(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[dict[str, Any]],
    scaler: Mapping[str, list[float]],
) -> list[dict[str, Any]]:
    mean_values = np.asarray(scaler["mean"], dtype=np.float32)
    scale_values = np.asarray(scaler["scale"], dtype=np.float32)
    result = []
    for candidate in candidates:
        model = candidate["model"]
        model.eval()
        scores = []
        with torch.inference_mode():
            for row in rows:
                features = row_features(row)
                scaled = (features - mean_values) / scale_values
                scores.append(model(torch.from_numpy(scaled)).numpy())
        result.append(
            {
                **{key: value for key, value in candidate.items() if key != "model"},
                "mrr": case_mrr(rows, scores),
                "scores": scores,
            }
        )
    return result


def select_best(candidates: Sequence[Mapping[str, Any]], tolerance: float = 0.005) -> Mapping[str, Any]:
    ranked = sorted(candidates, key=lambda row: (-float(row["mrr"]), int(row["parameter_count"])))
    best = ranked[0]
    simple = min(candidates, key=lambda row: int(row["parameter_count"]))
    if float(best["mrr"]) - float(simple["mrr"]) <= tolerance:
        return simple
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V8 candidate-level reranker development.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    config = read_json(args.config)
    rows = read_jsonl(args.rows)
    if len(rows) != 1080:
        raise RuntimeError("V8 development requires the frozen 1080-row V7 development matrix.")
    if file_sha256(args.rows) != "c37729122f4a562727a6663da8d81d83452ea748c36f95836b72698313612f19":
        raise RuntimeError("The development retrieval rows do not match the audited V7 input.")
    block_rows = defaultdict(list)
    for row in rows:
        block_rows[str(row["block"])].append(row)
    if set(block_rows) != {"train_a", "train_b", "validation"}:
        raise RuntimeError("V8 development blocks are incomplete.")
    train_rows = block_rows["train_a"] + block_rows["train_b"]
    validation_rows = block_rows["validation"]
    if len(train_rows) != 720 or len(validation_rows) != 360:
        raise RuntimeError("Unexpected V8 development block dimensions.")

    all_train_features = np.concatenate([row_features(row) for row in train_rows], axis=0)
    _, scaler = normalize_features(all_train_features)
    internal_holdout_cases = {
        case_id
        for case_id in {str(row["case_id"]) for row in train_rows}
        if stable_order(case_id, "v8-internal-early-stop", 8026) < "3333333333333333333333333333333333333333333333333333333333333333"
    }
    if not internal_holdout_cases:
        raise RuntimeError("V8 internal early-stopping split is empty.")
    internal_train_rows = [row for row in train_rows if str(row["case_id"]) not in internal_holdout_cases]
    internal_holdout_rows = [row for row in train_rows if str(row["case_id"]) in internal_holdout_cases]
    train_positive, train_negative, train_stats = build_pair_dataset(internal_train_rows)

    candidates: list[dict[str, Any]] = []
    for model_type in ("linear", "mlp"):
        for learning_rate in (0.001, 0.0003):
            for weight_decay in (0.0, 0.0001):
                result = train_model(
                    model_type,
                    learning_rate,
                    weight_decay,
                    (train_positive, train_negative),
                    internal_holdout_rows,
                    scaler,
                    max_epochs=20,
                    seed=8026,
                )
                candidates.append(
                    {
                        "model_type": model_type,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "parameter_count": sum(parameter.numel() for parameter in result["model"].parameters()),
                        "best_epoch_internal": result["best_epoch"],
                        "internal_holdout_mrr": result["best_holdout_mrr"],
                        "final_loss_internal": result["final_loss"],
                        "model": result["model"],
                    }
                )

    final_candidates: list[dict[str, Any]] = []
    final_positive, final_negative, final_train_stats = build_pair_dataset(train_rows)
    for candidate in candidates:
        result = train_model(
            candidate["model_type"],
            float(candidate["learning_rate"]),
            float(candidate["weight_decay"]),
            (final_positive, final_negative),
            validation_rows,
            scaler,
            max_epochs=int(candidate["best_epoch_internal"]),
            seed=8026,
        )
        final_candidates.append(
            {
                **{key: value for key, value in candidate.items() if key != "model"},
                "validation_mrr": result["best_holdout_mrr"],
                "final_epoch": int(candidate["best_epoch_internal"]),
                "model": result["model"],
            }
        )

    selected = select_best(
        [
            {**candidate, "mrr": candidate["validation_mrr"]}
            for candidate in final_candidates
        ]
    )
    global_alpha_scores = []
    for alpha in np.linspace(0.0, 1.0, 101):
        global_alpha_scores.append(
            {
                "alpha": float(alpha),
                "validation_mrr": case_mrr(validation_rows, scores_for_alpha(validation_rows, float(alpha))),
            }
        )
    best_global = max(global_alpha_scores, key=lambda row: float(row["validation_mrr"]))
    tied = [
        row
        for row in global_alpha_scores
        if abs(float(row["validation_mrr"]) - float(best_global["validation_mrr"])) <= 1e-12
    ]
    best_global = sorted(tied, key=lambda row: (abs(float(row["alpha"]) - 0.5), float(row["alpha"])))[0]
    global_validation_mrr = float(best_global["validation_mrr"])
    candidate_validation_mrr = float(selected["validation_mrr"])
    point_estimate_gain = candidate_validation_mrr - global_validation_mrr
    development_status = (
        "development_candidate_selected_confirmation_ids_not_instantiated"
        if point_estimate_gain > 0.0
        else "development_no_go_confirmation_ids_not_instantiated"
    )
    selection_decision = (
        "candidate_reranker_retained_for_confirmation_protocol"
        if point_estimate_gain > 0.0
        else "retain_global_fusion_baseline_no_candidate_point_estimate_gain"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "v8_candidate_reranker_checkpoint.pt"
    scaler_path = args.output_dir / "v8_candidate_reranker_feature_scaler.json"
    summary_path = args.output_dir / "v8_development_summary.json"
    torch.save(
        {
            "protocol_commit": PROTOCOL_COMMIT,
            "model_type": selected["model_type"],
            "feature_names": FEATURE_NAMES,
            "state_dict": selected["model"].state_dict(),
            "parameter_count": selected["parameter_count"],
            "final_epoch": selected["final_epoch"],
        },
        checkpoint_path,
    )
    scaler_path.write_text(json.dumps(scaler, indent=2) + "\n", encoding="utf-8", newline="\n")
    serial_candidates = [
        {key: value for key, value in candidate.items() if key != "model"}
        for candidate in final_candidates
    ]
    summary = {
        "experiment": "V8 candidate-level multimodal reranker development",
        "status": development_status,
        "protocol_commit": PROTOCOL_COMMIT,
        "config_sha256": file_sha256(args.config),
        "development_rows_sha256": file_sha256(args.rows),
        "manifest_sha256": file_sha256(args.manifest),
        "feature_names": FEATURE_NAMES,
        "negative_policy": "highest_non_target_bm25_and_image_candidates_deduplicated_to_8",
        "train_case_count": len({str(row["case_id"]) for row in train_rows}),
        "validation_case_count": len({str(row["case_id"]) for row in validation_rows}),
        "internal_early_stop_case_count": len(internal_holdout_cases),
        "internal_train_case_count": len({str(row["case_id"]) for row in internal_train_rows}),
        "internal_train_stats": train_stats,
        "final_train_stats": final_train_stats,
        "candidate_results": serial_candidates,
        "selected_model": {
            "model_type": selected["model_type"],
            "learning_rate": selected["learning_rate"],
            "weight_decay": selected["weight_decay"],
            "parameter_count": selected["parameter_count"],
            "final_epoch": selected["final_epoch"],
            "validation_mrr": selected["validation_mrr"],
            "accepted_for_confirmation": point_estimate_gain > 0.0,
        },
        "development_decision": {
            "comparator": "validation_selected_global_fusion",
            "comparator_alpha": best_global["alpha"],
            "comparator_validation_mrr": global_validation_mrr,
            "candidate_validation_mrr": candidate_validation_mrr,
            "candidate_minus_comparator_validation_mrr": point_estimate_gain,
            "decision": selection_decision,
            "rule": "A candidate must first exceed the comparator in validation MRR; confirmation H1 still requires a positive lower bootstrap bound.",
        },
        "global_alpha": {
            "grid": "0.00_to_1.00_step_0.01",
            "selected_alpha": best_global["alpha"],
            "validation_mrr": best_global["validation_mrr"],
            "tie_break": "closest_to_0.50_then_lower",
        },
        "outputs": {
            "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "scaler": str(scaler_path.relative_to(ROOT)).replace("\\", "/"),
            "scaler_sha256": file_sha256(scaler_path),
            "summary": str(summary_path.relative_to(ROOT)).replace("\\", "/"),
        },
        "claim_boundary": "Development selection only; no V8 confirmation IDs or confirmation outcomes were inspected. A no-go decision retains global fusion and does not instantiate a confirmation cohort.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
