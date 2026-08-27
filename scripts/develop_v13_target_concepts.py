from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_v10_pathology_utility import (  # noqa: E402
    label_unique_texts,
    resolve_checkpoint,
    text_sha256,
)
from medical_rag.evaluation.chexbert_pathology import CHEXBERT_LABELS  # noqa: E402
from medical_rag.evaluation.target_concepts import (  # noqa: E402
    logistic_probabilities,
    macro_auprc,
    multilabel_metrics,
    select_f1_thresholds,
)
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from run_v10_evidence_generation_development import read_json, read_jsonl  # noqa: E402


DEFAULT_CASES = ROOT / "data/processed/openi_cases.jsonl"
DEFAULT_SPLIT = ROOT / "data/splits/v10/v10_cluster_disjoint_split.json"
DEFAULT_EMBEDDINGS = ROOT / "data/processed/v10_medsiglip_embeddings.npz"
DEFAULT_CACHE = ROOT / "experiments/v13_target_concept/v13_train_calibration_chexbert_cache.json"
DEFAULT_MODEL_DIR = ROOT / "experiments/v13_target_concept/models"
DEFAULT_DECISION = ROOT / "data/splits/v13/v13_target_concept_decision.json"
PROTOCOL_COMMIT = "72d5af1"
LINEAR_C = (0.01, 0.1, 1.0, 10.0)


class ConceptMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(1152, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.20),
            torch.nn.Linear(256, len(CHEXBERT_LABELS)),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def spectrum(case: dict[str, Any]) -> str:
    value = str(case.get("problems") or "").strip().lower()
    if value == "normal":
        return "normal"
    if value in {"", "no indexing"}:
        return "indeterminate"
    return "abnormal"


def report_text(case: dict[str, Any]) -> str:
    return "\n".join(
        part for part in (str(case.get("findings") or ""), str(case.get("impression") or "")) if part
    )


def label_cases(
    cases: Sequence[dict[str, Any]],
    *,
    cache_path: Path,
    checkpoint_hash: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    pseudo_rows = [
        {"answer": report_text(case), "reference_answer": report_text(case)} for case in cases
    ]
    labels = label_unique_texts(
        pseudo_rows,
        cache_path=cache_path,
        checkpoint_hash=checkpoint_hash,
        device=device,
        batch_size=batch_size,
    )
    return np.asarray(
        [labels[text_sha256(report_text(case))] for case in cases], dtype=np.int8
    )


def fit_linear_heads(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    c_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.zeros((len(CHEXBERT_LABELS), train_x.shape[1]), dtype=np.float64)
    intercepts = np.zeros(len(CHEXBERT_LABELS), dtype=np.float64)
    for index in range(len(CHEXBERT_LABELS)):
        target = train_y[:, index]
        prevalence = float(np.clip(target.mean(), 1e-6, 1.0 - 1e-6))
        if len(np.unique(target)) < 2:
            intercepts[index] = np.log(prevalence / (1.0 - prevalence))
            continue
        model = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=7141,
        )
        model.fit(train_x, target)
        coefficients[index] = model.coef_[0]
        intercepts[index] = model.intercept_[0]
    return coefficients, intercepts


def mlp_probabilities(
    model: ConceptMLP, values: np.ndarray, *, device: torch.device
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(values.astype(np.float32)).to(device))
    return torch.sigmoid(logits).cpu().numpy()


def train_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    *,
    device: torch.device,
) -> tuple[ConceptMLP, dict[str, Any]]:
    random.seed(7142)
    np.random.seed(7142)
    torch.manual_seed(7142)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(7142)
    model = ConceptMLP().to(device)
    positive = train_y.sum(axis=0).astype(np.float64)
    negative = len(train_y) - positive
    pos_weight = np.divide(
        negative,
        positive,
        out=np.ones_like(negative),
        where=positive > 0,
    )
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.from_numpy(pos_weight.astype(np.float32)).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(7142)
    train_tensor = torch.from_numpy(train_x.astype(np.float32))
    label_tensor = torch.from_numpy(train_y.astype(np.float32))
    best_score = -1.0
    best_epoch = 0
    best_state: OrderedDict[str, torch.Tensor] | None = None
    patience = 0
    history = []
    for epoch in range(1, 101):
        model.train()
        permutation = torch.randperm(len(train_tensor), generator=generator)
        losses = []
        for start in range(0, len(permutation), 64):
            indices = permutation[start : start + 64]
            batch_x = train_tensor[indices].to(device)
            batch_y = label_tensor[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        probabilities = mlp_probabilities(model, calibration_x, device=device)
        score = macro_auprc(calibration_y, probabilities)
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "calibration_macro_auprc": score}
        )
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = OrderedDict(
                (name, value.detach().cpu().clone()) for name, value in model.state_dict().items()
            )
            patience = 0
        else:
            patience += 1
        if patience >= 10:
            break
    if best_state is None:
        raise RuntimeError("MLP training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_calibration_macro_auprc": best_score,
        "epochs_run": len(history),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Develop the V13 target-image concept head.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--decision-output", type=Path, default=DEFAULT_DECISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chexbert-batch-size", type=int, default=128)
    args = parser.parse_args()

    cases_by_id = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    split = read_json(args.split)
    train_ids = [str(value) for value in split["partitions"]["train"]["case_ids"]]
    calibration_ids = [
        str(value) for value in split["partitions"]["calibration"]["case_ids"]
    ]
    selected_ids = train_ids + calibration_ids
    if set(train_ids) & set(calibration_ids):
        raise RuntimeError("Train and Calibration overlap")
    if any(case_id not in cases_by_id for case_id in selected_ids):
        raise RuntimeError("Train/Calibration contains an unknown case")

    with np.load(args.embeddings, allow_pickle=False) as encoded:
        embedding_ids = [str(value) for value in encoded["case_ids"].tolist()]
        embeddings = np.asarray(encoded["case_image_embeddings"], dtype=np.float32)
    embedding_by_id = {
        case_id: embeddings[index] for index, case_id in enumerate(embedding_ids)
    }
    if any(case_id not in embedding_by_id for case_id in selected_ids):
        raise RuntimeError("Train/Calibration embedding coverage is incomplete")
    selected_cases = [cases_by_id[case_id] for case_id in selected_ids]
    checkpoint = resolve_checkpoint()
    selected_labels = label_cases(
        selected_cases,
        cache_path=args.cache,
        checkpoint_hash=file_sha256(checkpoint),
        device=args.device,
        batch_size=args.chexbert_batch_size,
    )
    train_count = len(train_ids)
    train_x = np.stack([embedding_by_id[case_id] for case_id in train_ids])
    calibration_x = np.stack([embedding_by_id[case_id] for case_id in calibration_ids])
    train_y = selected_labels[:train_count]
    calibration_y = selected_labels[train_count:]

    prevalence = train_y.mean(axis=0)
    prevalence_probabilities = np.broadcast_to(prevalence, calibration_y.shape)
    prevalence_thresholds, prevalence_threshold_records = select_f1_thresholds(
        calibration_y, prevalence_probabilities
    )
    candidates: list[dict[str, Any]] = [
        {
            "name": "train_prevalence",
            "type": "baseline",
            "calibration_macro_auprc": macro_auprc(
                calibration_y, prevalence_probabilities
            ),
        }
    ]
    linear_states: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for c_value in LINEAR_C:
        coefficients, intercepts = fit_linear_heads(train_x, train_y, c_value=c_value)
        probabilities = logistic_probabilities(calibration_x, coefficients, intercepts)
        linear_states[c_value] = (coefficients, intercepts)
        candidates.append(
            {
                "name": f"linear_c_{c_value:g}",
                "type": "linear",
                "c": c_value,
                "calibration_macro_auprc": macro_auprc(calibration_y, probabilities),
            }
        )

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    mlp, mlp_record = train_mlp(
        train_x,
        train_y,
        calibration_x,
        calibration_y,
        device=device,
    )
    mlp_probabilities_values = mlp_probabilities(mlp, calibration_x, device=device)
    candidates.append(
        {
            "name": "mlp_1152_256_14",
            "type": "mlp",
            "calibration_macro_auprc": macro_auprc(
                calibration_y, mlp_probabilities_values
            ),
            **{key: value for key, value in mlp_record.items() if key != "history"},
        }
    )
    best_linear = max(
        (row for row in candidates if row["type"] == "linear"),
        key=lambda row: (float(row["calibration_macro_auprc"]), -float(row["c"])),
    )
    mlp_candidate = next(row for row in candidates if row["type"] == "mlp")
    selected = (
        mlp_candidate
        if float(mlp_candidate["calibration_macro_auprc"])
        >= float(best_linear["calibration_macro_auprc"]) + 0.005
        else best_linear
    )
    args.model_dir.mkdir(parents=True, exist_ok=True)
    if selected["type"] == "linear":
        coefficients, intercepts = linear_states[float(selected["c"])]
        calibration_probabilities = logistic_probabilities(
            calibration_x, coefficients, intercepts
        )
        checkpoint_path = args.model_dir / "v13_selected_linear.npz"
        thresholds, threshold_records = select_f1_thresholds(
            calibration_y, calibration_probabilities
        )
        np.savez_compressed(
            checkpoint_path,
            coefficients=coefficients.astype(np.float32),
            intercepts=intercepts.astype(np.float32),
            thresholds=thresholds.astype(np.float32),
            labels=np.asarray(CHEXBERT_LABELS),
            c=np.asarray(float(selected["c"])),
        )
    else:
        calibration_probabilities = mlp_probabilities_values
        thresholds, threshold_records = select_f1_thresholds(
            calibration_y, calibration_probabilities
        )
        checkpoint_path = args.model_dir / "v13_selected_mlp.pt"
        torch.save(
            {
                "state_dict": mlp.state_dict(),
                "thresholds": torch.from_numpy(thresholds.astype(np.float32)),
                "labels": list(CHEXBERT_LABELS),
                "architecture": "1152-256-14-gelu-dropout0.20",
            },
            checkpoint_path,
        )

    decision = {
        "study": "V13 target-image pathology concept development",
        "status": "train_calibration_selection_complete_validation_not_evaluated",
        "protocol_commit": PROTOCOL_COMMIT,
        "validation_outcomes_inspected": False,
        "counts": {
            "train": len(train_ids),
            "calibration": len(calibration_ids),
            "labels": len(CHEXBERT_LABELS),
        },
        "candidate_models": candidates,
        "selection_rule": {
            "metric": "calibration_macro_auprc",
            "linear_preference_tolerance": 0.005,
            "selected": selected,
        },
        "calibration_metrics": multilabel_metrics(
            calibration_y, calibration_probabilities, thresholds
        ),
        "thresholds": threshold_records,
        "prevalence_baseline": {
            "calibration_metrics": multilabel_metrics(
                calibration_y, prevalence_probabilities, prevalence_thresholds
            ),
            "thresholds": prevalence_threshold_records,
        },
        "train_prevalence": {
            label: float(prevalence[index]) for index, label in enumerate(CHEXBERT_LABELS)
        },
        "mlp_training": mlp_record,
        "selected_checkpoint": {
            "path": str(checkpoint_path.relative_to(ROOT).as_posix()),
            "sha256": file_sha256(checkpoint_path),
            "committed": False,
        },
        "inputs": {
            "cases_sha256": file_sha256(args.cases),
            "split_sha256": file_sha256(args.split),
            "embeddings_sha256": file_sha256(args.embeddings),
            "chexbert_checkpoint_sha256": file_sha256(checkpoint),
            "chexbert_cache_sha256": file_sha256(args.cache),
            "script_sha256": file_sha256(Path(__file__)),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "f1chexbert": importlib.metadata.version("f1chexbert"),
            "device": str(device),
        },
        "claim_boundary": (
            "Train/Calibration development on automated report-derived labels; "
            "not clinical diagnostic accuracy or confirmation evidence."
        ),
    }
    args.decision_output.parent.mkdir(parents=True, exist_ok=True)
    args.decision_output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": selected,
                "baseline_macro_auprc": candidates[0]["calibration_macro_auprc"],
                "selected_calibration_metrics": decision["calibration_metrics"],
                "checkpoint_sha256": decision["selected_checkpoint"]["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
