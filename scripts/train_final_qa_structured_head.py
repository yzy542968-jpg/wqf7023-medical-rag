from __future__ import annotations

import argparse
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.qa.radrestruct_hierarchy import RadReStructHierarchy  # noqa: E402
from medical_rag.qa.structured_decoding import decode_answer_probabilities  # noqa: E402
from medical_rag.qa.structured_head import (  # noqa: E402
    FeatureBlocks,
    StructuredHistoryHead,
    history_feature_block,
    retrieve_top1_history,
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _targets(role: dict[str, Any], rad_root: Path, keys: tuple[str, ...]) -> np.ndarray:
    return stack_answer_vectors(
        load_answer_vector(
            rad_root
            / f"{case['official_split']}_vectorized_answers"
            / f"{case['source_report_id']}.json",
            keys,
        )
        for case in role["cases"]
    ).astype(np.float32)


def _embedding_maps(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]:
    with np.load(path, allow_pickle=False) as payload:
        image_ids = [str(value) for value in payload["case_ids"]]
        images = np.asarray(payload["case_image_embeddings"], dtype=np.float32)
        report_ids = [str(value) for value in payload["report_ids"]]
        reports = np.asarray(payload["report_embeddings"], dtype=np.float32)
        signature = str(payload["signature"].item())
    images = normalize(images, norm="l2").astype(np.float32)
    reports = normalize(reports, norm="l2").astype(np.float32)
    return (
        dict(zip(image_ids, images, strict=True)),
        dict(zip(report_ids, reports, strict=True)),
        signature,
    )


def _role_arrays(
    role: dict[str, Any],
    image_map: dict[str, np.ndarray],
) -> tuple[list[str], list[str], np.ndarray]:
    case_ids = [case["case_id"] for case in role["cases"]]
    clusters = [case["cluster_id"] for case in role["cases"]]
    missing = [case_id for case_id in case_ids if case_id not in image_map]
    if missing:
        raise ValueError(f"Missing target image embeddings: {len(missing)}")
    return case_ids, clusters, np.stack([image_map[case_id] for case_id in case_ids])


def _indication_features(
    train_ids: list[str],
    calibration_ids: list[str],
    validation_ids: list[str],
    raw_cases: dict[str, dict[str, Any]],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    def texts(case_ids: list[str]) -> list[str]:
        return [" ".join(str(raw_cases[case_id].get("indication") or "Not provided").split()) for case_id in case_ids]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=2048,
        sublinear_tf=True,
    )
    train_sparse = vectorizer.fit_transform(texts(train_ids))
    components = min(128, max(1, train_sparse.shape[1] - 1))
    svd = TruncatedSVD(n_components=components, random_state=seed)
    train = normalize(svd.fit_transform(train_sparse), norm="l2").astype(np.float32)
    calibration = normalize(
        svd.transform(vectorizer.transform(texts(calibration_ids))), norm="l2"
    ).astype(np.float32)
    validation = normalize(
        svd.transform(vectorizer.transform(texts(validation_ids))), norm="l2"
    ).astype(np.float32)
    return train, calibration, validation, {
        "tfidf_features": len(vectorizer.vocabulary_),
        "svd_components": components,
        "svd_explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
    }


def _probabilities(
    model: nn.Module, features: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[offset : offset + batch_size]).to(device)
            outputs.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(outputs)


def _decode_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    hierarchy: RadReStructHierarchy,
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


def _threshold_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        -row["metrics"]["supported_label_macro_f1"],
        abs(row["multi_choice_threshold"] - 0.5)
        + abs(row["fixed_choice_threshold"] - 0.5),
        -row["multi_choice_threshold"],
        -row["fixed_choice_threshold"],
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_json(args.config)
    manifest = _load_json(args.manifest)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Structured-head development requires the available CUDA GPU")

    hierarchy = RadReStructHierarchy(args.radrestruct_root)
    report_keys = load_report_keys(args.radrestruct_root)
    image_map, report_map, embedding_signature = _embedding_maps(args.embeddings)
    raw_cases = {str(row["case_id"]): row for row in _read_jsonl(args.cases)}
    roles = manifest["roles"]
    train_ids, train_clusters, train_images = _role_arrays(roles["train"], image_map)
    calibration_ids, calibration_clusters, calibration_images = _role_arrays(
        roles["calibration"], image_map
    )
    validation_ids, validation_clusters, validation_images = _role_arrays(
        roles["validation"], image_map
    )
    train_targets = _targets(roles["train"], args.radrestruct_root, report_keys)
    calibration_targets = _targets(
        roles["calibration"], args.radrestruct_root, report_keys
    )
    validation_targets = _targets(
        roles["validation"], args.radrestruct_root, report_keys
    )
    train_indication, calibration_indication, validation_indication, text_audit = _indication_features(
        train_ids, calibration_ids, validation_ids, raw_cases, seed
    )

    bank_positions = [index for index, case_id in enumerate(train_ids) if case_id in report_map]
    bank_ids = [train_ids[index] for index in bank_positions]
    bank_clusters = [train_clusters[index] for index in bank_positions]
    bank_images = train_images[bank_positions]
    bank_reports = np.stack([report_map[case_id] for case_id in bank_ids])
    train_neighbor_indices, train_similarities, train_history_reports = retrieve_top1_history(
        train_images, train_clusters, bank_images, bank_clusters, bank_reports
    )
    calibration_neighbor_indices, calibration_similarities, calibration_history_reports = retrieve_top1_history(
        calibration_images, calibration_clusters, bank_images, bank_clusters, bank_reports
    )
    validation_neighbor_indices, validation_similarities, validation_history_reports = retrieve_top1_history(
        validation_images, validation_clusters, bank_images, bank_clusters, bank_reports
    )

    train_blocks = FeatureBlocks(
        np.concatenate([train_images, train_indication], axis=1),
        history_feature_block(train_history_reports, train_similarities),
    )
    calibration_blocks = FeatureBlocks(
        np.concatenate([calibration_images, calibration_indication], axis=1),
        history_feature_block(calibration_history_reports, calibration_similarities),
    )
    validation_blocks = FeatureBlocks(
        np.concatenate([validation_images, validation_indication], axis=1),
        history_feature_block(validation_history_reports, validation_similarities),
    )
    train_target_tensor = torch.from_numpy(train_blocks.target)
    train_history_tensor = torch.from_numpy(train_blocks.history)
    train_label_tensor = torch.from_numpy(train_targets)
    loader = DataLoader(
        TensorDataset(train_target_tensor, train_history_tensor, train_label_tensor),
        batch_size=int(config["model"]["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = StructuredHistoryHead(
        train_blocks.target.shape[1] + train_blocks.history.shape[1],
        train_targets.shape[1],
        hidden_features=512,
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    positive = train_targets.sum(axis=0)
    negative = len(train_targets) - positive
    positive_weight = np.sqrt((negative + 1.0) / (positive + 1.0))
    positive_weight = np.clip(positive_weight, 1.0, 10.0).astype(np.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(positive_weight).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    best_metric = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    calibration_history_features = calibration_blocks.combined(True)
    for epoch in range(1, int(config["model"]["maximum_epochs"]) + 1):
        model.train()
        losses: list[float] = []
        dropout_generator = torch.Generator(device=device).manual_seed(seed + epoch)
        for target_features, history_features, labels in loader:
            target_features = target_features.to(device)
            history_features = history_features.to(device)
            labels = labels.to(device)
            keep = (
                torch.rand(
                    (len(labels), 1), generator=dropout_generator, device=device
                )
                >= float(config["model"]["history_dropout_probability"])
            ).to(history_features.dtype)
            features = torch.cat([target_features, history_features * keep], dim=1)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        calibration_probabilities = _probabilities(
            model,
            calibration_history_features,
            device,
            int(config["model"]["batch_size"]),
        )
        calibration_metrics, _ = _decode_metrics(
            calibration_probabilities,
            calibration_targets,
            hierarchy,
            0.5,
            0.5,
        )
        selector = calibration_metrics["supported_label_macro_f1"]
        history.append(
            {
                "epoch": epoch,
                "mean_train_loss": sum(losses) / len(losses),
                "calibration_supported_label_macro_f1": selector,
            }
        )
        if selector > best_metric + 1e-12:
            best_metric = selector
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["model"]["early_stopping_patience"]):
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "input_features": train_blocks.target.shape[1] + train_blocks.history.shape[1],
            "output_labels": train_targets.shape[1],
            "embedding_signature": embedding_signature,
        },
        args.checkpoint,
    )

    calibration_probabilities = _probabilities(
        model,
        calibration_history_features,
        device,
        int(config["model"]["batch_size"]),
    )
    threshold_grid: list[dict[str, Any]] = []
    for multi_threshold in config["decoding"]["multi_choice_threshold"]:
        for fixed_threshold in config["decoding"]["fixed_choice_threshold"]:
            metrics, _ = _decode_metrics(
                calibration_probabilities,
                calibration_targets,
                hierarchy,
                float(multi_threshold),
                float(fixed_threshold),
            )
            threshold_grid.append(
                {
                    "multi_choice_threshold": float(multi_threshold),
                    "fixed_choice_threshold": float(fixed_threshold),
                    "metrics": metrics,
                }
            )
    selected_threshold = min(threshold_grid, key=_threshold_key)
    multi_threshold = selected_threshold["multi_choice_threshold"]
    fixed_threshold = selected_threshold["fixed_choice_threshold"]
    validation_conditions: dict[str, Any] = {}
    for name, history_present in (
        ("same_model_no_history", False),
        ("same_model_top1_paired_report_embedding", True),
    ):
        probabilities = _probabilities(
            model,
            validation_blocks.combined(history_present),
            device,
            int(config["model"]["batch_size"]),
        )
        metrics, predictions = _decode_metrics(
            probabilities,
            validation_targets,
            hierarchy,
            multi_threshold,
            fixed_threshold,
        )
        validation_conditions[name] = metrics
        np.savez_compressed(
            args.predictions_dir / f"{name}.npz",
            case_ids=np.asarray(validation_ids),
            probabilities=probabilities.astype(np.float16),
            predictions=predictions,
        )
    elapsed = time.perf_counter() - started
    summary = {
        "study": config["study"],
        "status": "development_complete_no_test",
        "config": "config/final_qa_structured_head_development.json",
        "device": str(device),
        "embedding_signature": embedding_signature,
        "role_case_counts": {role: len(roles[role]["cases"]) for role in ("train", "calibration", "validation")},
        "historical_bank_case_count": len(bank_ids),
        "history_cluster_exclusion": True,
        "text_representation_audit": text_audit,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "best_calibration_macro_f1_at_default_thresholds": best_metric,
        "training_history": history,
        "threshold_grid": threshold_grid,
        "selected_threshold_on_calibration": selected_threshold,
        "single_validation_conditions": validation_conditions,
        "paired_macro_f1_difference_history_minus_no_history": (
            validation_conditions["same_model_top1_paired_report_embedding"]["supported_label_macro_f1"]
            - validation_conditions["same_model_no_history"]["supported_label_macro_f1"]
        ),
        "retrieval_similarity": {
            "train_mean": float(train_similarities.mean()),
            "calibration_mean": float(calibration_similarities.mean()),
            "validation_mean": float(validation_similarities.mean()),
        },
        "elapsed_seconds": elapsed,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "local_assets": {
            "checkpoint": str(args.checkpoint.relative_to(ROOT)),
            "validation_predictions": str(args.predictions_dir.relative_to(ROOT)),
        },
        "boundary": config["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/final_qa_structured_head_development.json"
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/splits/final_qa/final_qa_development_manifest.json"
    )
    parser.add_argument("--radrestruct-root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "data/processed/v10_medsiglip_embeddings.npz")
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "experiments/final_qa_development/structured_history_head.pt"
    )
    parser.add_argument(
        "--predictions-dir", type=Path, default=ROOT / "experiments/final_qa_development/structured_head_predictions"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "experiments/final_qa_development/structured_head_summary.json"
    )
    args = parser.parse_args()
    args.predictions_dir.mkdir(parents=True, exist_ok=True)
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "best_epoch": result["best_epoch"],
                "selected_threshold_on_calibration": result["selected_threshold_on_calibration"],
                "single_validation_conditions": result["single_validation_conditions"],
                "paired_macro_f1_difference_history_minus_no_history": result["paired_macro_f1_difference_history_minus_no_history"],
                "elapsed_seconds": result["elapsed_seconds"],
                "peak_vram_mb": result["peak_vram_mb"],
                "boundary": result["boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
