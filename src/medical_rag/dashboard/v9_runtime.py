"""Interactive V9 other-patient similar-case retrieval helpers.

The runtime consumes the frozen V9 training bank and learned checkpoint. An
uploaded image is treated as a new query and is never added to the bank or
written into a frozen experiment artifact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from medical_rag.retrieval.bm25_retriever import BM25Retriever
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


class V9MLPScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


@dataclass(frozen=True)
class V9DashboardResources:
    cases: dict[str, dict[str, Any]]
    candidate_ids: tuple[str, ...]
    bm25: BM25Retriever
    image_embeddings: np.ndarray
    report_embeddings: np.ndarray
    model: V9MLPScorer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_v9_resources(
    *,
    cases_path: Path,
    embedding_cache_path: Path,
    checkpoint_path: Path,
    retrieval_config_path: Path,
) -> V9DashboardResources:
    config = json.loads(retrieval_config_path.read_text(encoding="utf-8"))
    expected_checkpoint = str(config["systems"]["r4"]["checkpoint_sha256"])
    if _sha256(checkpoint_path) != expected_checkpoint:
        raise RuntimeError("The V9 dashboard checkpoint differs from the frozen R4 model.")
    all_cases = {str(case["case_id"]): case for case in load_cases_jsonl(cases_path)}
    with np.load(embedding_cache_path, allow_pickle=False) as cache:
        candidate_ids = tuple(str(value) for value in cache["candidate_ids"].tolist())
        image_embeddings = np.asarray(cache["candidate_image_embeddings"], dtype=np.float32)
        report_embeddings = np.asarray(cache["report_mean_embeddings"], dtype=np.float32)
    if len(candidate_ids) != int(config["candidate_bank_count"]):
        raise RuntimeError("The V9 candidate bank count changed.")
    if image_embeddings.shape != report_embeddings.shape or image_embeddings.shape[0] != len(candidate_ids):
        raise RuntimeError("The V9 embedding cache dimensions are inconsistent.")
    missing = [case_id for case_id in candidate_ids if case_id not in all_cases]
    if missing:
        raise RuntimeError(f"V9 candidate cases are missing: {missing[:5]}")
    cases = {case_id: all_cases[case_id] for case_id in candidate_ids}
    bm25 = BM25Retriever().fit([cases[case_id] for case_id in candidate_ids])
    model = V9MLPScorer()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()
    return V9DashboardResources(
        cases=cases,
        candidate_ids=candidate_ids,
        bm25=bm25,
        image_embeddings=image_embeddings,
        report_embeddings=report_embeddings,
        model=model,
    )


def infer_question_type(question: str) -> str:
    lowered = " ".join(str(question).lower().split())
    if any(term in lowered for term in ("acute", "cardiopulmonary abnormality")):
        return "acute"
    if any(term in lowered for term in ("impression", "diagnosis", "conclusion", "most likely")):
        return "impression"
    return "findings"


def _normalized_and_reciprocal(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    normalized = np.zeros(len(values), dtype=np.float32)
    if len(values) and float(values.max() - values.min()) > 1e-12:
        normalized = ((values - values.min()) / (values.max() - values.min())).astype(np.float32)
    order = np.lexsort((np.arange(len(values)), -values))
    reciprocal = np.zeros(len(values), dtype=np.float32)
    reciprocal[order] = 1.0 / np.arange(1, len(values) + 1, dtype=np.float32)
    return normalized, reciprocal


def _feature_matrix(
    text: np.ndarray, image_image: np.ndarray, image_report: np.ndarray, question_type: str
) -> np.ndarray:
    components = [_normalized_and_reciprocal(values) for values in (text, image_image, image_report)]
    indicator = {
        "findings": (1.0, 0.0, 0.0),
        "impression": (0.0, 1.0, 0.0),
        "acute": (0.0, 0.0, 1.0),
    }[question_type]
    return np.column_stack(
        [
            components[0][0],
            components[1][0],
            components[2][0],
            components[0][1],
            components[1][1],
            components[2][1],
            np.tile(np.asarray(indicator, dtype=np.float32), (len(text), 1)),
        ]
    ).astype(np.float32)


def retrieve_v9(
    *,
    indication: str,
    question: str,
    image_embedding: np.ndarray,
    resources: V9DashboardResources,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    query = "\n".join(value for value in (str(indication).strip(), str(question).strip()) if value)
    if not str(question).strip():
        raise ValueError("A medical question is required.")
    embedding = np.asarray(image_embedding, dtype=np.float32).reshape(-1)
    if embedding.shape[0] != resources.image_embeddings.shape[1]:
        raise ValueError("Uploaded-image embedding dimension does not match the V9 bank.")
    text_rows = resources.bm25.search(query, top_k=len(resources.candidate_ids))
    text_map = {str(row["case_id"]): float(row["score"]) for row in text_rows}
    text = np.asarray([text_map[case_id] for case_id in resources.candidate_ids], dtype=np.float64)
    image_image = resources.image_embeddings @ embedding
    image_report = resources.report_embeddings @ embedding
    features = _feature_matrix(text, image_image, image_report, infer_question_type(question))
    with torch.inference_mode():
        learned = resources.model(torch.from_numpy(features)).numpy()
    order = sorted(
        range(len(learned)),
        key=lambda index: (-float(learned[index]), resources.candidate_ids[index]),
    )[: max(1, int(top_k))]
    rows = []
    for rank, index in enumerate(order, start=1):
        case_id = resources.candidate_ids[index]
        case = resources.cases[case_id]
        rows.append(
            {
                "rank": rank,
                "case_id": case_id,
                "indication": str(case.get("indication", "")),
                "findings": str(case.get("findings", "")),
                "impression": str(case.get("impression", "")),
                "bm25_score": float(text[index]),
                "image_image_similarity": float(image_image[index]),
                "image_report_similarity": float(image_report[index]),
                "learned_score": float(learned[index]),
            }
        )
    return rows
