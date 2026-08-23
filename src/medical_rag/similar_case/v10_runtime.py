from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from medical_rag.retrieval.bm25_retriever import BM25Retriever
from medical_rag.similar_case.schema import PairedCase
from medical_rag.similar_case.v10_reranker import (
    FactAwareFeatureIndex,
    R4Scorer,
    R5Scorer,
    augment_r4_features,
)


QUESTIONS = {
    "findings": "What are the main radiographic findings?",
    "impression": "What is the most likely radiographic impression?",
    "acute": "Is there an acute cardiopulmonary abnormality? Explain briefly.",
}


def normalized_scores_and_reciprocal_ranks(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(scores)
    normalized = np.zeros(len(scores), dtype=np.float32)
    valid_scores = scores[finite]
    if valid_scores.size and float(valid_scores.max() - valid_scores.min()) > 1e-12:
        normalized[finite] = (
            (valid_scores - valid_scores.min()) / (valid_scores.max() - valid_scores.min())
        ).astype(np.float32)
    order = sorted(np.flatnonzero(finite).tolist(), key=lambda index: (-float(scores[index]), index))
    reciprocal = np.zeros(len(scores), dtype=np.float32)
    for rank, index in enumerate(order, start=1):
        reciprocal[index] = 1.0 / rank
    return normalized, reciprocal


def r4_feature_matrix(
    bm25: np.ndarray,
    image_image: np.ndarray,
    image_report: np.ndarray,
    *,
    question_type: str,
) -> np.ndarray:
    components = [
        normalized_scores_and_reciprocal_ranks(values)
        for values in (bm25, image_image, image_report)
    ]
    question = {
        "findings": (1.0, 0.0, 0.0),
        "impression": (0.0, 1.0, 0.0),
        "acute": (0.0, 0.0, 1.0),
    }[question_type]
    indicators = np.tile(np.asarray(question, dtype=np.float32), (len(bm25), 1))
    return np.column_stack(
        [
            components[0][0],
            components[1][0],
            components[2][0],
            components[0][1],
            components[1][1],
            components[2][1],
            indicators,
        ]
    ).astype(np.float32)


@dataclass
class FrozenR5Runtime:
    candidate_ids: list[str]
    candidate_cases: list[PairedCase]
    candidate_images: np.ndarray
    candidate_reports: np.ndarray
    bm25: BM25Retriever
    fact_index: FactAwareFeatureIndex
    r4_model: R4Scorer | None
    models: list[R5Scorer]

    @classmethod
    def build(
        cls,
        *,
        candidate_ids: Sequence[str],
        cases: Mapping[str, PairedCase],
        raw_cases: Mapping[str, Mapping[str, object]],
        facts_by_case: Mapping[str, Sequence[str]],
        image_by_id: Mapping[str, np.ndarray],
        report_by_id: Mapping[str, np.ndarray],
        checkpoint_states: Sequence[Mapping[str, torch.Tensor]],
        r4_checkpoint_state: Mapping[str, torch.Tensor] | None = None,
    ) -> "FrozenR5Runtime":
        identifiers = list(candidate_ids)
        bank = [cases[case_id] for case_id in identifiers]
        models = []
        for state in checkpoint_states:
            model = R5Scorer()
            model.load_state_dict(state)
            model.eval()
            models.append(model)
        r4_model = None
        if r4_checkpoint_state is not None:
            r4_model = R4Scorer()
            r4_model.load_state_dict(r4_checkpoint_state)
            r4_model.eval()
        return cls(
            candidate_ids=identifiers,
            candidate_cases=bank,
            candidate_images=np.stack([image_by_id[case_id] for case_id in identifiers]),
            candidate_reports=np.stack([report_by_id[case_id] for case_id in identifiers]),
            bm25=BM25Retriever().fit(
                [{"case_id": case.study_id, "report_text": case.report_text} for case in bank]
            ),
            fact_index=FactAwareFeatureIndex.build(identifiers, raw_cases, facts_by_case),
            r4_model=r4_model,
            models=models,
        )

    def score(
        self,
        query: PairedCase,
        query_image: np.ndarray,
        *,
        question_type: str,
    ) -> dict[str, Any]:
        question = QUESTIONS[question_type]
        query_text = "\n".join(part for part in (query.indication, question) if part)
        bm25_scores = np.asarray(self.bm25.score_all(query_text), dtype=np.float32)
        image = np.asarray(query_image, dtype=np.float32)
        image_image = self.candidate_images @ image
        image_report = self.candidate_reports @ image
        r4 = r4_feature_matrix(
            bm25_scores,
            image_image,
            image_report,
            question_type=question_type,
        )
        r5 = augment_r4_features(r4, self.fact_index.query_features(query_text))
        with torch.inference_mode():
            r4_scores = None if self.r4_model is None else self.r4_model(torch.from_numpy(r4)).numpy()
            seed_scores = np.stack(
                [model(torch.from_numpy(r5)).numpy() for model in self.models]
            )
        ensemble = seed_scores.mean(axis=0)
        ranking = np.lexsort((np.arange(len(ensemble)), -ensemble))
        return {
            "question": question,
            "query_text": query_text,
            "bm25": bm25_scores,
            "image_image": image_image,
            "image_report": image_report,
            "seed_scores": seed_scores,
            "r4_scores": r4_scores,
            "ensemble_scores": ensemble,
            "ranking": ranking,
            "top_case_ids": [self.candidate_ids[index] for index in ranking],
        }


def component_agreement(result: Mapping[str, np.ndarray], selected_index: int) -> float:
    components = ("bm25", "image_image", "image_report")
    matches = 0
    for name in components:
        values = np.asarray(result[name])
        top = int(np.lexsort((np.arange(len(values)), -values))[0])
        matches += int(top == selected_index)
    return matches / len(components)


__all__ = [
    "FrozenR5Runtime",
    "QUESTIONS",
    "component_agreement",
    "normalized_scores_and_reciprocal_ranks",
    "r4_feature_matrix",
]
