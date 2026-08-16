from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from medical_rag.retrieval.bm25_retriever import BM25Retriever
from medical_rag.retrieval.medcpt_retriever import MedCPTRetriever, encode_queries
from medical_rag.retrieval.tfidf_retriever import _tokens


def minmax(values: np.ndarray) -> np.ndarray:
    lower = float(values.min())
    upper = float(values.max())
    if upper == lower:
        return np.zeros_like(values)
    return (values - lower) / (upper - lower)


@dataclass
class HybridBM25MedCPTRetriever:
    cases: list[dict[str, Any]]
    bm25: BM25Retriever
    medcpt: MedCPTRetriever
    alpha: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")
        self.case_position = {case["case_id"]: idx for idx, case in enumerate(self.cases)}
        self.medcpt_case_position = {
            case_id: pos for pos, case_id in enumerate(self.medcpt.case_ids)
        }

    @classmethod
    def from_components(
        cls,
        cases: list[dict[str, Any]],
        bm25: BM25Retriever,
        medcpt: MedCPTRetriever,
        alpha: float = 0.5,
    ) -> "HybridBM25MedCPTRetriever":
        return cls(cases=cases, bm25=bm25, medcpt=medcpt, alpha=alpha)

    def score_with_embedding(self, query: str, query_embedding: np.ndarray) -> dict[str, np.ndarray]:
        bm25_query_terms = _tokens(query)
        bm25_scores = np.array(
            [self.bm25._score_document(bm25_query_terms, idx) for idx in range(len(self.cases))],
            dtype="float32",
        )

        medcpt_scores_indexed = self.medcpt.embeddings @ query_embedding
        medcpt_scores = np.zeros(len(self.cases), dtype="float32")
        for case_id, med_pos in self.medcpt_case_position.items():
            case_pos = self.case_position.get(case_id)
            if case_pos is not None:
                medcpt_scores[case_pos] = medcpt_scores_indexed[med_pos]

        hybrid_scores = self.alpha * minmax(medcpt_scores) + (1 - self.alpha) * minmax(bm25_scores)
        return {
            "hybrid": hybrid_scores,
            "bm25": bm25_scores,
            "medcpt": medcpt_scores,
        }

    def search_with_embedding(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        scores = self.score_with_embedding(query, query_embedding)
        ranked_indices = scores["hybrid"].argsort()[::-1][:top_k]

        results: list[dict[str, Any]] = []
        for rank, index in enumerate(ranked_indices, start=1):
            case = self.cases[int(index)]
            results.append(
                {
                    "rank": rank,
                    "case_id": case["case_id"],
                    "score": float(scores["hybrid"][int(index)]),
                    "bm25_score": float(scores["bm25"][int(index)]),
                    "medcpt_score": float(scores["medcpt"][int(index)]),
                    "indication": case.get("indication", ""),
                    "problems": case.get("problems", ""),
                    "findings": case.get("findings", ""),
                    "impression": case.get("impression", ""),
                    "images": case.get("images", []),
                }
            )
        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        batch_size: int = 16,
        device: str | None = None,
    ) -> list[dict[str, Any]]:
        query_embedding = encode_queries([query], batch_size=batch_size, device=device)[0]
        return self.search_with_embedding(query, query_embedding, top_k=top_k)
