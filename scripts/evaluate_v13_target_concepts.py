from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from develop_v13_target_concepts import (  # noqa: E402
    ConceptMLP,
    label_cases,
    mlp_probabilities,
    spectrum,
)
from evaluate_v10_pathology_utility import resolve_checkpoint  # noqa: E402
from medical_rag.evaluation.chexbert_pathology import CHEXBERT_LABELS  # noqa: E402
from medical_rag.evaluation.target_concepts import (  # noqa: E402
    case_id_fingerprint,
    logistic_probabilities,
    macro_auprc,
    multilabel_metrics,
    spectrum_metrics,
)
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from run_v10_evidence_generation_development import read_json, read_jsonl  # noqa: E402


DEFAULT_CASES = ROOT / "data/processed/openi_cases.jsonl"
DEFAULT_SPLIT = ROOT / "data/splits/v10/v10_cluster_disjoint_split.json"
DEFAULT_EMBEDDINGS = ROOT / "data/processed/v10_medsiglip_embeddings.npz"
DEFAULT_DECISION = ROOT / "data/splits/v13/v13_target_concept_decision.json"
DEFAULT_CACHE = ROOT / "experiments/v13_target_concept/v13_validation_chexbert_cache.json"
DEFAULT_OUTPUT = ROOT / "data/splits/v13/v13_target_concept_validation_summary.json"


def predict(
    checkpoint_path: Path,
    embeddings: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, str]:
    if checkpoint_path.suffix == ".npz":
        with np.load(checkpoint_path, allow_pickle=False) as encoded:
            probabilities = logistic_probabilities(
                embeddings,
                np.asarray(encoded["coefficients"], dtype=np.float64),
                np.asarray(encoded["intercepts"], dtype=np.float64),
            )
            thresholds = np.asarray(encoded["thresholds"], dtype=np.float64)
            labels = tuple(map(str, encoded["labels"].tolist()))
        model_type = "linear"
    elif checkpoint_path.suffix == ".pt":
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = ConceptMLP().to(device)
        model.load_state_dict(payload["state_dict"])
        probabilities = mlp_probabilities(model, embeddings, device=device)
        thresholds = np.asarray(payload["thresholds"].cpu(), dtype=np.float64)
        labels = tuple(map(str, payload["labels"]))
        model_type = "mlp"
    else:
        raise ValueError("Unsupported V13 checkpoint type")
    if labels != CHEXBERT_LABELS:
        raise RuntimeError("V13 checkpoint label order differs")
    return probabilities, thresholds, model_type


def bootstrap_macro_auprc_difference(
    labels: np.ndarray,
    selected_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = rng.integers(0, len(labels), len(labels))
        draws[index] = macro_auprc(
            labels[sample], selected_probabilities[sample]
        ) - macro_auprc(labels[sample], baseline_probabilities[sample])
    observed = macro_auprc(labels, selected_probabilities) - macro_auprc(
        labels, baseline_probabilities
    )
    return {
        "mean_difference": float(observed),
        "ci_95_low": float(np.quantile(draws, 0.025)),
        "ci_95_high": float(np.quantile(draws, 0.975)),
        "case_count": len(labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen V13 concept head on Validation.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chexbert-batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=7144)
    args = parser.parse_args()

    decision = read_json(args.decision)
    if decision.get("status") != "train_calibration_selection_complete_validation_not_evaluated":
        raise RuntimeError("V13 development decision is not frozen for Validation")
    if decision.get("validation_outcomes_inspected") is not False:
        raise RuntimeError("V13 decision record does not preserve the Validation boundary")
    for key, path in (
        ("cases_sha256", args.cases),
        ("split_sha256", args.split),
        ("embeddings_sha256", args.embeddings),
    ):
        if decision["inputs"][key] != file_sha256(path):
            raise RuntimeError(f"V13 input hash differs: {key}")
    checkpoint_path = ROOT / decision["selected_checkpoint"]["path"]
    if file_sha256(checkpoint_path) != decision["selected_checkpoint"]["sha256"]:
        raise RuntimeError("V13 selected checkpoint hash differs")

    cases_by_id = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    split = read_json(args.split)
    validation_ids = [
        str(value) for value in split["partitions"]["validation"]["case_ids"]
    ]
    if len(validation_ids) != 384:
        raise RuntimeError("Unexpected V10 Validation count")
    validation_cases = [cases_by_id[case_id] for case_id in validation_ids]
    with np.load(args.embeddings, allow_pickle=False) as encoded:
        embedding_ids = [str(value) for value in encoded["case_ids"].tolist()]
        embedding_matrix = np.asarray(encoded["case_image_embeddings"], dtype=np.float32)
    embedding_by_id = {
        case_id: embedding_matrix[index] for index, case_id in enumerate(embedding_ids)
    }
    validation_x = np.stack([embedding_by_id[case_id] for case_id in validation_ids])
    chexbert_checkpoint = resolve_checkpoint()
    validation_y = label_cases(
        validation_cases,
        cache_path=args.cache,
        checkpoint_hash=file_sha256(chexbert_checkpoint),
        device=args.device,
        batch_size=args.chexbert_batch_size,
    )
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    probabilities, thresholds, model_type = predict(
        checkpoint_path, validation_x, device=device
    )
    prevalence = np.asarray(
        [decision["train_prevalence"][label] for label in CHEXBERT_LABELS], dtype=np.float64
    )
    baseline_probabilities = np.broadcast_to(prevalence, validation_y.shape)
    baseline_thresholds = np.asarray(
        [row["threshold"] for row in decision["prevalence_baseline"]["thresholds"]],
        dtype=np.float64,
    )
    spectra = [spectrum(case) for case in validation_cases]
    output = {
        "study": "V13 target-image pathology concept Validation",
        "status": "validation_development_complete_test_not_evaluated",
        "protocol_commit": decision["protocol_commit"],
        "decision_record_sha256": file_sha256(args.decision),
        "test_outcomes_inspected": False,
        "counts": {
            "validation_cases": len(validation_ids),
            "normal": spectra.count("normal"),
            "abnormal": spectra.count("abnormal"),
            "indeterminate": spectra.count("indeterminate"),
        },
        "validation_case_ids_sha256": case_id_fingerprint(validation_ids),
        "selected_model": {
            "type": model_type,
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "thresholds": thresholds.tolist(),
        },
        "metrics": multilabel_metrics(validation_y, probabilities, thresholds),
        "train_prevalence_baseline": multilabel_metrics(
            validation_y, baseline_probabilities, baseline_thresholds
        ),
        "macro_auprc_difference_vs_prevalence": bootstrap_macro_auprc_difference(
            validation_y,
            probabilities,
            baseline_probabilities,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
        ),
        "spectrum_metrics": spectrum_metrics(
            validation_y, probabilities, thresholds, spectra
        ),
        "inputs": {
            "cases_sha256": file_sha256(args.cases),
            "split_sha256": file_sha256(args.split),
            "embeddings_sha256": file_sha256(args.embeddings),
            "chexbert_checkpoint_sha256": file_sha256(chexbert_checkpoint),
            "validation_label_cache_sha256": file_sha256(args.cache),
            "script_sha256": file_sha256(Path(__file__)),
        },
        "claim_boundary": (
            "Validation-only agreement with automated report-derived CheXbert labels; "
            "not clinical diagnosis, confirmation, safety, or patient benefit."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
