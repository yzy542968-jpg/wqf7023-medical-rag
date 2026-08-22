from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "v7_adaptive_fusion_development.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_MANIFEST = ROOT / "data" / "splits" / "v7" / "v7_development_manifest.json"
DEFAULT_ROWS = ROOT / "experiments" / "post_submission_v7" / "development_retrieval_rows.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "post_submission_v7"

FEATURE_NAMES = [
    "indication_present",
    "question_token_count",
    "indication_token_count",
    "bm25_top1_normalized_score",
    "bm25_top1_top2_margin",
    "bm25_score_std",
    "bm25_score_iqr",
    "image_top1_normalized_score",
    "image_top1_top2_margin",
    "image_score_std",
    "image_score_iqr",
    "text_image_top1_agreement",
    "text_image_spearman_correlation",
]


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


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def percentile(values: np.ndarray, probability: float) -> float:
    return float(np.percentile(values, probability * 100.0)) if len(values) else 0.0


def iqr(values: np.ndarray) -> float:
    return percentile(values, 0.75) - percentile(values, 0.25)


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    first_centered = first - float(first.mean())
    second_centered = second - float(second.mean())
    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(first_centered, second_centered) / denominator)


def image_order(candidate_ids: Sequence[str], image_scores: np.ndarray) -> list[str]:
    return [
        candidate_id
        for candidate_id, _ in sorted(
            zip(candidate_ids, image_scores.tolist(), strict=True),
            key=lambda item: (-float(item[1]), item[0]),
        )
    ]


def build_features(row: Mapping[str, Any], cases: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    case_id = str(row["case_id"])
    case = cases[case_id]
    candidate_ids = [str(value) for value in row["candidate_case_ids"]]
    text_scores = np.asarray(row["text_scores_normalized"], dtype=np.float64)
    image_scores = np.asarray(row["image_scores_normalized"], dtype=np.float64)
    if len(candidate_ids) != 100 or len(text_scores) != 100 or len(image_scores) != 100:
        raise ValueError("Every V7 row must contain exactly 100 shortlist candidates and scores.")
    text_order = candidate_ids
    visual_order = image_order(candidate_ids, image_scores)
    text_ranks = np.arange(1, len(candidate_ids) + 1, dtype=np.float64)
    visual_rank_map = {case_id: index for index, case_id in enumerate(visual_order, start=1)}
    visual_ranks = np.asarray([visual_rank_map[case_id] for case_id in candidate_ids], dtype=np.float64)
    visual_sorted_scores = np.asarray(
        [image_scores[candidate_ids.index(case_id)] for case_id in visual_order],
        dtype=np.float64,
    )
    feature_values = [
        float(bool(clean_text(case.get("indication", "")))),
        float(len(clean_text(row["question"]).split())),
        float(len(clean_text(case.get("indication", "")).split())),
        float(text_scores[0]),
        float(text_scores[0] - text_scores[1]),
        float(text_scores.std()),
        iqr(text_scores),
        float(visual_sorted_scores[0]),
        float(visual_sorted_scores[0] - visual_sorted_scores[1]),
        float(image_scores.std()),
        iqr(image_scores),
        float(text_order[0] == visual_order[0]),
        pearson(text_ranks, visual_ranks),
    ]
    return np.asarray(feature_values, dtype=np.float32)


class FeatureScaler:
    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, values: np.ndarray) -> "FeatureScaler":
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.scale


class LinearAlpha(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.output = nn.Linear(input_dim, 1)

    def forward(self, features: Tensor) -> Tensor:
        return torch.sigmoid(self.output(features)).squeeze(-1)


class MLPAlpha(nn.Module):
    def __init__(self, input_dim: int, hidden_units: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return torch.sigmoid(self.network(features)).squeeze(-1)


def build_model(model_type: str, input_dim: int) -> nn.Module:
    if model_type == "linear_sigmoid":
        return LinearAlpha(input_dim)
    if model_type == "mlp_32_relu_sigmoid":
        return MLPAlpha(input_dim, 32)
    raise ValueError(f"Unknown learner: {model_type}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_pairs(
    rows: Sequence[Mapping[str, Any]],
    features_by_qid: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    feature_rows: list[np.ndarray] = []
    text_differences: list[float] = []
    image_differences: list[float] = []
    skipped = 0
    for row in rows:
        candidates = [str(value) for value in row["candidate_case_ids"]]
        text_scores = np.asarray(row["text_scores_normalized"], dtype=np.float32)
        image_scores = np.asarray(row["image_scores_normalized"], dtype=np.float32)
        target = str(row["target_case_id"])
        if target not in candidates:
            skipped += 1
            continue
        target_index = candidates.index(target)
        text_order = candidates
        visual_order = image_order(candidates, image_scores)
        negative_ids: list[str] = []
        for candidate_id in text_order:
            if candidate_id != target:
                negative_ids.append(candidate_id)
            if len(negative_ids) >= 20:
                break
        for candidate_id in visual_order:
            if candidate_id != target and candidate_id not in negative_ids:
                negative_ids.append(candidate_id)
            if len(negative_ids) >= 40:
                break
        for negative_id in negative_ids:
            negative_index = candidates.index(negative_id)
            feature_rows.append(features_by_qid[str(row["qid"])])
            text_differences.append(float(text_scores[target_index] - text_scores[negative_index]))
            image_differences.append(float(image_scores[target_index] - image_scores[negative_index]))
    if not feature_rows:
        raise RuntimeError("No valid positive-negative training pairs were available.")
    return (
        np.stack(feature_rows).astype(np.float32),
        np.asarray(text_differences, dtype=np.float32),
        np.asarray(image_differences, dtype=np.float32),
        skipped,
    )


def mean_reciprocal_rank(rows: Sequence[Mapping[str, Any]], alphas: np.ndarray) -> float:
    by_case: dict[str, dict[str, float]] = defaultdict(dict)
    for row, alpha in zip(rows, alphas.tolist(), strict=True):
        candidates = [str(value) for value in row["candidate_case_ids"]]
        text_scores = np.asarray(row["text_scores_normalized"], dtype=np.float64)
        image_scores = np.asarray(row["image_scores_normalized"], dtype=np.float64)
        fused = float(alpha) * text_scores + (1.0 - float(alpha)) * image_scores
        ranking = [
            case_id
            for case_id, _ in sorted(
                zip(candidates, fused.tolist(), strict=True),
                key=lambda item: (-float(item[1]), item[0]),
            )
        ]
        target = str(row["target_case_id"])
        reciprocal = 1.0 / (ranking.index(target) + 1) if target in ranking else 0.0
        by_case[str(row["case_id"])][str(row["question_type"])] = reciprocal
    case_values = [sum(values.values()) / len(values) for values in by_case.values()]
    return float(np.mean(case_values)) if case_values else 0.0


def alpha_predictions(model: nn.Module, features: np.ndarray, scaler: FeatureScaler) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(scaler.transform(features))
    with torch.inference_mode():
        return model(tensor).cpu().numpy().astype(np.float32)


def train_one_fold(
    *,
    model_type: str,
    learning_rate: float,
    weight_decay: float,
    optimize_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    features_by_qid: Mapping[str, np.ndarray],
    seed: int,
    max_epochs: int,
    patience: int,
    minimum_delta: float,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    optimize_features = np.stack([features_by_qid[str(row["qid"])] for row in optimize_rows])
    holdout_features = np.stack([features_by_qid[str(row["qid"])] for row in holdout_rows])
    scaler = FeatureScaler.fit(optimize_features)
    pair_features, text_diffs, image_diffs, skipped = build_pairs(optimize_rows, features_by_qid)
    model = build_model(model_type, pair_features.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    x_tensor = torch.from_numpy(scaler.transform(pair_features))
    text_tensor = torch.from_numpy(text_diffs)
    image_tensor = torch.from_numpy(image_diffs)
    best_state: dict[str, Tensor] | None = None
    best_mrr = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    epoch_history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(x_tensor), generator=generator)
        for start in range(0, len(order), 64):
            indices = order[start : start + 64]
            alpha = model(x_tensor[indices])
            pair_score_difference = alpha * text_tensor[indices] + (1.0 - alpha) * image_tensor[indices]
            loss = torch.nn.functional.softplus(-pair_score_difference).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        holdout_alpha = alpha_predictions(model, holdout_features, scaler)
        holdout_mrr = mean_reciprocal_rank(holdout_rows, holdout_alpha)
        epoch_history.append({"epoch": float(epoch), "loss": float(loss.item()), "holdout_mrr": holdout_mrr})
        if holdout_mrr > best_mrr + minimum_delta:
            best_mrr = holdout_mrr
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch_history[-1]["epoch"]
        best_mrr = epoch_history[-1]["holdout_mrr"]
    model.load_state_dict(best_state)
    return {
        "model_type": model_type,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "best_epoch": int(best_epoch),
        "best_holdout_mrr": float(best_mrr),
        "pair_count": len(pair_features),
        "target_outside_shortlist_count": skipped,
        "parameter_count": parameter_count(model),
        "scaler": scaler,
        "model_state": best_state,
        "epoch_history": epoch_history,
    }


def summarize_config(results: Sequence[Mapping[str, Any]], model_type: str) -> dict[str, Any]:
    candidates = [row for row in results if row["model_type"] == model_type]
    if not candidates:
        raise RuntimeError(f"No development candidates for {model_type}.")
    return min(
        candidates,
        key=lambda row: (
            -float(row["mean_holdout_mrr"]),
            int(row["parameter_count"]),
            float(row["learning_rate"]),
            float(row["weight_decay"]),
            json.dumps(
                {
                    "model_type": row["model_type"],
                    "learning_rate": row["learning_rate"],
                    "weight_decay": row["weight_decay"],
                },
                sort_keys=True,
            ),
        ),
    )


def train_final(
    *,
    model_type: str,
    learning_rate: float,
    weight_decay: float,
    rows: Sequence[Mapping[str, Any]],
    features_by_qid: Mapping[str, np.ndarray],
    epochs: int,
    seed: int,
) -> tuple[nn.Module, FeatureScaler, int, int]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    features = np.stack([features_by_qid[str(row["qid"])] for row in rows])
    scaler = FeatureScaler.fit(features)
    pair_features, text_diffs, image_diffs, skipped = build_pairs(rows, features_by_qid)
    model = build_model(model_type, pair_features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x_tensor = torch.from_numpy(scaler.transform(pair_features))
    text_tensor = torch.from_numpy(text_diffs)
    image_tensor = torch.from_numpy(image_diffs)
    last_loss = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(x_tensor), generator=generator)
        for start in range(0, len(order), 64):
            indices = order[start : start + 64]
            alpha = model(x_tensor[indices])
            difference = alpha * text_tensor[indices] + (1.0 - alpha) * image_tensor[indices]
            loss = torch.nn.functional.softplus(-difference).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
    return model, scaler, skipped, int(epochs)


def state_to_serializable(state: Mapping[str, Tensor]) -> dict[str, list[float]]:
    return {
        key: value.detach().cpu().reshape(-1).tolist()
        for key, value in state.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and select the V7 adaptive fusion learner.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    config = read_json(args.config)
    manifest = read_json(args.manifest)
    rows = read_jsonl(args.rows)
    cases = {str(case["case_id"]): case for case in load_cases_jsonl(args.cases)}
    if len(rows) != 1080:
        raise RuntimeError(f"Expected 1080 V7 development rows, found {len(rows)}.")
    features_by_qid = {
        str(row["qid"]): build_features(row, cases)
        for row in rows
    }
    if len(features_by_qid) != len(rows):
        raise RuntimeError("V7 retrieval rows contain duplicate qids.")
    blocks = {
        name: [row for row in rows if str(row["block"]) == name]
        for name in ("train_a", "train_b", "validation")
    }
    if any(len(block_rows) != 360 for block_rows in blocks.values()):
        raise RuntimeError("Each V7 development block must contain 360 questions.")

    training_grid = [("linear_sigmoid", lr, wd) for lr in (0.001, 0.0003) for wd in (0.0001, 0.0)]
    training_grid += [("mlp_32_relu_sigmoid", lr, wd) for lr in (0.001, 0.0003) for wd in (0.0001, 0.0)]
    fold_results: list[dict[str, Any]] = []
    development_started = time.perf_counter()
    for model_type, learning_rate, weight_decay in training_grid:
        fold_a = train_one_fold(
            model_type=model_type,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimize_rows=blocks["train_a"],
            holdout_rows=blocks["train_b"],
            features_by_qid=features_by_qid,
            seed=7026,
            max_epochs=100,
            patience=10,
            minimum_delta=0.0001,
        )
        fold_b = train_one_fold(
            model_type=model_type,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimize_rows=blocks["train_b"],
            holdout_rows=blocks["train_a"],
            features_by_qid=features_by_qid,
            seed=7026,
            max_epochs=100,
            patience=10,
            minimum_delta=0.0001,
        )
        fold_results.append(
            {
                "model_type": model_type,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "parameter_count": fold_a["parameter_count"],
                "fold_a": {
                    "best_epoch": fold_a["best_epoch"],
                    "best_holdout_mrr": fold_a["best_holdout_mrr"],
                    "pair_count": fold_a["pair_count"],
                    "target_outside_shortlist_count": fold_a["target_outside_shortlist_count"],
                },
                "fold_b": {
                    "best_epoch": fold_b["best_epoch"],
                    "best_holdout_mrr": fold_b["best_holdout_mrr"],
                    "pair_count": fold_b["pair_count"],
                    "target_outside_shortlist_count": fold_b["target_outside_shortlist_count"],
                },
                "mean_holdout_mrr": float(
                    (fold_a["best_holdout_mrr"] + fold_b["best_holdout_mrr"]) / 2.0
                ),
                "mean_best_epoch": float((fold_a["best_epoch"] + fold_b["best_epoch"]) / 2.0),
            }
        )

    best_linear = summarize_config(fold_results, "linear_sigmoid")
    best_mlp = summarize_config(fold_results, "mlp_32_relu_sigmoid")
    mlp_gain_over_linear = float(best_mlp["mean_holdout_mrr"] - best_linear["mean_holdout_mrr"])
    selected = best_linear if mlp_gain_over_linear <= 0.005 else best_mlp
    selected_epoch = int(math.floor((selected["fold_a"]["best_epoch"] + selected["fold_b"]["best_epoch"]) / 2.0 + 0.5))

    train_ab_rows = blocks["train_a"] + blocks["train_b"]
    final_model, final_scaler, final_skipped, final_epochs = train_final(
        model_type=str(selected["model_type"]),
        learning_rate=float(selected["learning_rate"]),
        weight_decay=float(selected["weight_decay"]),
        rows=train_ab_rows,
        features_by_qid=features_by_qid,
        epochs=selected_epoch,
        seed=7026,
    )
    validation_features = np.stack(
        [features_by_qid[str(row["qid"])] for row in blocks["validation"]]
    )
    adaptive_validation_alpha = alpha_predictions(final_model, validation_features, final_scaler)
    adaptive_validation_mrr = mean_reciprocal_rank(blocks["validation"], adaptive_validation_alpha)

    alpha_grid = np.arange(0.0, 1.0001, 0.01, dtype=np.float64)
    global_results = [
        {"alpha": float(alpha), "mrr": mean_reciprocal_rank(blocks["validation"], np.full(len(blocks["validation"]), alpha))}
        for alpha in alpha_grid
    ]
    global_selected = sorted(
        global_results,
        key=lambda row: (-float(row["mrr"]), abs(float(row["alpha"]) - 0.5), float(row["alpha"])),
    )[0]
    gate_grid = np.arange(0.0, 1.0001, 0.05, dtype=np.float64)
    gate_results = []
    validation_features_rows = blocks["validation"]
    for threshold in gate_grid:
        gate_alphas = []
        for row, feature in zip(validation_features_rows, validation_features, strict=True):
            text_margin = float(feature[4])
            gate_alphas.append(1.0 if text_margin >= float(threshold) else float(global_selected["alpha"]))
        gate_alpha_array = np.asarray(gate_alphas, dtype=np.float32)
        gate_mrr = mean_reciprocal_rank(validation_features_rows, gate_alpha_array)
        gate_coverage = float(np.mean(gate_alpha_array < 1.0))
        gate_results.append(
            {"threshold": float(threshold), "mrr": gate_mrr, "non_text_fraction": gate_coverage}
        )
    selected_gate = sorted(
        gate_results,
        key=lambda row: (-float(row["mrr"]), -float(row["non_text_fraction"]), float(row["threshold"])),
    )[0]

    checkpoint_path = args.output_dir / "v7_adaptive_fusion_final_checkpoint.pt"
    scaler_path = args.output_dir / "v7_adaptive_fusion_feature_scaler.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": selected["model_type"],
            "state_dict": final_model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "input_dim": len(FEATURE_NAMES),
            "epochs": final_epochs,
            "seed": 7026,
        },
        checkpoint_path,
    )
    scaler_path.write_text(
        json.dumps(
            {
                "feature_names": FEATURE_NAMES,
                "mean": final_scaler.mean.tolist(),
                "scale": final_scaler.scale.tolist(),
                "fit_scope": "train_a_plus_train_b",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    decision = {
        "experiment": "V7 adaptive multimodal fusion development",
        "status": "development_decision_complete_confirmation_not_instantiated",
        "config_path": str(args.config.relative_to(ROOT).as_posix()),
        "config_sha256": file_sha256(args.config),
        "manifest_path": str(args.manifest.relative_to(ROOT).as_posix()),
        "manifest_sha256": file_sha256(args.manifest),
        "retrieval_rows_path": str(args.rows.relative_to(ROOT).as_posix()),
        "retrieval_rows_sha256": file_sha256(args.rows),
        "source_cases_sha256": file_sha256(args.cases),
        "feature_names": FEATURE_NAMES,
        "feature_leakage_policy": "target_id_reference_qrels_target_availability_answers_verifier_outputs_forbidden",
        "training_device": "cpu",
        "development_runtime_seconds": time.perf_counter() - development_started,
        "fold_candidates": fold_results,
        "best_linear": best_linear,
        "best_mlp": best_mlp,
        "mlp_gain_over_linear": mlp_gain_over_linear,
        "complexity_rule": "mlp_minus_linear_leq_0.005_select_linear",
        "selected_configuration": {
            "model_type": selected["model_type"],
            "learning_rate": selected["learning_rate"],
            "weight_decay": selected["weight_decay"],
            "fold_a_best_epoch": selected["fold_a"]["best_epoch"],
            "fold_b_best_epoch": selected["fold_b"]["best_epoch"],
            "final_epoch_round_half_up": selected_epoch,
            "parameter_count": selected["parameter_count"],
        },
        "outside_shortlist_policy": {
            "train_a_count": sum(not bool(row["target_in_shortlist"]) for row in blocks["train_a"]),
            "train_b_count": sum(not bool(row["target_in_shortlist"]) for row in blocks["train_b"]),
            "validation_count": sum(not bool(row["target_in_shortlist"]) for row in blocks["validation"]),
            "final_train_ab_count": final_skipped,
            "retained_in_evaluation": True,
        },
        "validation_selection": {
            "global_alpha_star": global_selected,
            "global_alpha_grid": global_results,
            "adaptive_validation_mrr": adaptive_validation_mrr,
            "simple_gate_selected": selected_gate,
            "simple_gate_grid": gate_results,
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path.relative_to(ROOT).as_posix()),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "feature_scaler": str(scaler_path.relative_to(ROOT).as_posix()),
            "feature_scaler_sha256": file_sha256(scaler_path),
        },
        "confirmation_case_ids_instantiated": False,
        "claim_boundary": "Development-only selection; no V7 confirmation outcome or clinical claim.",
    }
    output_path = args.output_dir / "v7_development_decision.json"
    output_path.write_text(
        json.dumps(decision, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(decision, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
