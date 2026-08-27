from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from develop_v13_target_concepts import ConceptMLP  # noqa: E402
from evaluate_v13_target_concepts import predict  # noqa: E402
from medical_rag.evaluation.chexbert_pathology import CHEXBERT_LABELS  # noqa: E402


def test_mlp_checkpoint_loads_in_weights_only_mode(tmp_path: Path) -> None:
    checkpoint = tmp_path / "concept.pt"
    model = ConceptMLP()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "thresholds": torch.full((14,), 0.5),
            "labels": list(CHEXBERT_LABELS),
            "architecture": "1152-256-14-gelu-dropout0.20",
        },
        checkpoint,
    )
    probabilities, thresholds, model_type = predict(
        checkpoint,
        np.zeros((2, 1152), dtype=np.float32),
        device=torch.device("cpu"),
    )
    assert model_type == "mlp"
    assert probabilities.shape == (2, 14)
    assert thresholds.tolist() == [0.5] * 14


def test_linear_checkpoint_round_trip(tmp_path: Path) -> None:
    checkpoint = tmp_path / "concept.npz"
    np.savez_compressed(
        checkpoint,
        coefficients=np.zeros((14, 1152), dtype=np.float32),
        intercepts=np.zeros(14, dtype=np.float32),
        thresholds=np.full(14, 0.5, dtype=np.float32),
        labels=np.asarray(CHEXBERT_LABELS),
        c=np.asarray(1.0),
    )
    probabilities, thresholds, model_type = predict(
        checkpoint,
        np.zeros((2, 1152), dtype=np.float32),
        device=torch.device("cpu"),
    )
    assert model_type == "linear"
    assert probabilities.tolist() == [[0.5] * 14] * 2
    assert thresholds.tolist() == [0.5] * 14
