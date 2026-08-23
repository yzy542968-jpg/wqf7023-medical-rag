from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
from torch.nn import functional as F

from medical_rag.similar_case.v10_evidence import (
    fact_units,
    normalized_text,
    sentence_units,
)


R5_ADDED_FEATURES = (
    "sentence_similarity_max",
    "sentence_similarity_mean",
    "fact_similarity_max",
    "fact_similarity_mean",
    "positive_fact_fraction",
    "negative_fact_fraction",
    "uncertain_fact_fraction",
    "evidence_redundancy",
)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


@dataclass
class UnitIndex:
    vectorizer: TfidfVectorizer
    matrix: sparse.csr_matrix
    case_indices: np.ndarray
    candidate_count: int

    @classmethod
    def build(cls, texts: Sequence[str], case_indices: Sequence[int], candidate_count: int) -> "UnitIndex":
        if not texts:
            texts = ["unavailable"]
            case_indices = [0]
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
            sublinear_tf=True,
            min_df=1,
            dtype=np.float32,
        )
        matrix = vectorizer.fit_transform(texts).tocsr()
        return cls(vectorizer, matrix, np.asarray(case_indices, dtype=np.int32), candidate_count)

    def aggregate(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        query_vector = self.vectorizer.transform([normalized_text(query)])
        scores = (self.matrix @ query_vector.T).toarray().reshape(-1).astype(np.float32)
        maximum = np.zeros(self.candidate_count, dtype=np.float32)
        total = np.zeros(self.candidate_count, dtype=np.float32)
        count = np.zeros(self.candidate_count, dtype=np.float32)
        np.maximum.at(maximum, self.case_indices, scores)
        np.add.at(total, self.case_indices, scores)
        np.add.at(count, self.case_indices, 1.0)
        mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        return maximum, mean


@dataclass
class FactAwareFeatureIndex:
    sentence_index: UnitIndex
    fact_index: UnitIndex
    fixed_features: np.ndarray

    @classmethod
    def build(
        cls,
        candidate_ids: Sequence[str],
        cases: Mapping[str, Mapping[str, object]],
        facts_by_case: Mapping[str, Sequence[str]],
    ) -> "FactAwareFeatureIndex":
        sentence_texts: list[str] = []
        sentence_case_indices: list[int] = []
        fact_texts: list[str] = []
        fact_case_indices: list[int] = []
        fixed_rows = []
        for case_index, case_id in enumerate(candidate_ids):
            case = cases[case_id]
            sentences = sentence_units(case_id, "findings", case.get("findings")) + sentence_units(
                case_id, "impression", case.get("impression")
            )
            facts = fact_units(case_id, facts_by_case.get(case_id, ()))
            sentence_texts.extend(unit.text for unit in sentences)
            sentence_case_indices.extend([case_index] * len(sentences))
            fact_texts.extend(unit.text for unit in facts)
            fact_case_indices.extend([case_index] * len(facts))
            lowered = [unit.text.lower() for unit in facts]
            denominator = max(len(lowered), 1)
            all_units = [normalized_text(unit.text).lower() for unit in (*sentences, *facts)]
            redundancy = 1.0 - len(set(all_units)) / len(all_units) if all_units else 0.0
            fixed_rows.append(
                [
                    sum("definitely present" in text for text in lowered) / denominator,
                    sum("definitely absent" in text for text in lowered) / denominator,
                    sum("uncertain" in text for text in lowered) / denominator,
                    redundancy,
                ]
            )
        return cls(
            sentence_index=UnitIndex.build(sentence_texts, sentence_case_indices, len(candidate_ids)),
            fact_index=UnitIndex.build(fact_texts, fact_case_indices, len(candidate_ids)),
            fixed_features=np.asarray(fixed_rows, dtype=np.float32),
        )

    def query_features(self, query: str) -> np.ndarray:
        sentence_max, sentence_mean = self.sentence_index.aggregate(query)
        fact_max, fact_mean = self.fact_index.aggregate(query)
        return np.column_stack(
            [sentence_max, sentence_mean, fact_max, fact_mean, self.fixed_features]
        ).astype(np.float32)


def augment_r4_features(r4_features: np.ndarray, fact_features: np.ndarray) -> np.ndarray:
    left = np.asarray(r4_features, dtype=np.float32)
    right = np.asarray(fact_features, dtype=np.float32)
    if left.ndim != 2 or left.shape[1] != 9:
        raise ValueError("R4 features must have shape [n, 9]")
    if right.ndim != 2 or right.shape != (left.shape[0], 8):
        raise ValueError("fact features must have shape [n, 8]")
    return np.column_stack([left, right]).astype(np.float32)


def stable_top(values: np.ndarray, count: int, *, largest: bool, valid: np.ndarray) -> list[int]:
    indices = np.flatnonzero(valid).tolist()
    return sorted(
        indices,
        key=lambda index: ((-1 if largest else 1) * float(values[index]), index),
    )[:count]


def sample_fact_aware_pairs(
    features: np.ndarray,
    gains: np.ndarray,
    component_scores: Sequence[np.ndarray],
    *,
    config: Mapping[str, float | int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(gains)
    top_k = int(config["component_top_k"])
    pool: set[int] = set()
    for scores in component_scores:
        pool.update(stable_top(scores, top_k, largest=True, valid=valid))
    pool.update(stable_top(gains, int(config["relevance_top_k"]), largest=True, valid=valid))
    pool.update(stable_top(gains, int(config["relevance_bottom_k"]), largest=False, valid=valid))

    normalized_components = []
    for values in component_scores:
        finite = np.asarray(values, dtype=np.float64).copy()
        finite[~valid] = -np.inf
        minimum = np.min(finite[valid])
        maximum = np.max(finite[valid])
        normalized = np.zeros_like(finite)
        if maximum - minimum > 1e-12:
            normalized[valid] = (finite[valid] - minimum) / (maximum - minimum)
        normalized_components.append(normalized)
    retrieval_strength = np.max(np.stack(normalized_components), axis=0)
    hard_valid = valid & (gains <= float(config["hard_negative_max_gain"]))
    pool.update(
        stable_top(
            retrieval_strength,
            int(config["hard_negative_k"]),
            largest=True,
            valid=hard_valid,
        )
    )

    high = sorted(pool, key=lambda index: (-float(gains[index]), index))[
        : int(config["high_candidates"])
    ]
    low = sorted(pool, key=lambda index: (float(gains[index]), index))[
        : int(config["low_candidates"])
    ]
    high_rows = []
    low_rows = []
    weights = []
    minimum_difference = float(config["minimum_gain_difference"])
    for high_index in high:
        for low_index in low:
            difference = float(gains[high_index] - gains[low_index])
            if difference + 1e-12 < minimum_difference:
                continue
            high_rows.append(features[high_index])
            low_rows.append(features[low_index])
            weights.append(difference)
    width = int(features.shape[1])
    return (
        np.asarray(high_rows, dtype=np.float32).reshape(-1, width),
        np.asarray(low_rows, dtype=np.float32).reshape(-1, width),
        np.asarray(weights, dtype=np.float32),
    )


class R4Scorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(9, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


class R5Scorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(17, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    high: np.ndarray,
    low: np.ndarray,
    weights: np.ndarray,
    *,
    batch_size: int,
    seed: int,
) -> float:
    model.train()
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(high))
    losses = []
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        high_tensor = torch.from_numpy(high[indices])
        low_tensor = torch.from_numpy(low[indices])
        weight_tensor = torch.from_numpy(weights[indices])
        loss = (F.softplus(-(model(high_tensor) - model(low_tensor))) * weight_tensor).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else 0.0


__all__ = [
    "FactAwareFeatureIndex",
    "R4Scorer",
    "R5Scorer",
    "R5_ADDED_FEATURES",
    "augment_r4_features",
    "sample_fact_aware_pairs",
    "set_determinism",
    "train_epoch",
]

