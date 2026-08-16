from __future__ import annotations

import math
from collections import Counter
from typing import Any

from medical_rag.retrieval.tfidf_retriever import _tokens


class ScopedBM25ChunkRetriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[dict[str, Any]] = []
        self.term_counts: list[Counter[str]] = []
        self.lengths: list[int] = []
        self.avg_length = 0.0
        self.idf: dict[str, float] = {}

    def fit(self, chunks: list[dict[str, Any]]) -> "ScopedBM25ChunkRetriever":
        self.chunks = chunks
        tokenized = [_tokens(chunk.get("text", "")) for chunk in chunks]
        self.term_counts = [Counter(tokens) for tokens in tokenized]
        self.lengths = [len(tokens) for tokens in tokenized]
        self.avg_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        frequencies: Counter[str] = Counter()
        for tokens in tokenized:
            frequencies.update(set(tokens))
        count = len(chunks)
        self.idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in frequencies.items()
        }
        return self

    def _score(self, query_terms: list[str], index: int) -> float:
        if not self.avg_length:
            return 0.0
        score = 0.0
        length = self.lengths[index]
        counts = self.term_counts[index]
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + self.k1 * (1 - self.b + self.b * length / self.avg_length)
            score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
        return score

    def search(
        self,
        query: str,
        top_k: int = 5,
        case_id: str | None = None,
        allowed_case_ids: set[str] | None = None,
        allowed_sections: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.chunks:
            raise RuntimeError("Retriever has not been fitted.")
        if case_id is not None and allowed_case_ids is not None:
            raise ValueError("Use either case_id or allowed_case_ids, not both.")
        candidate_indices = [
            index
            for index, chunk in enumerate(self.chunks)
            if (case_id is None or chunk["case_id"] == case_id)
            and (allowed_case_ids is None or chunk["case_id"] in allowed_case_ids)
            and (allowed_sections is None or chunk["section"] in allowed_sections)
        ]
        query_terms = _tokens(query)
        ranked = sorted(
            candidate_indices,
            key=lambda index: (-self._score(query_terms, index), self.chunks[index]["chunk_id"]),
        )[:top_k]
        return [
            {
                **self.chunks[index],
                "rank": rank,
                "score": float(self._score(query_terms, index)),
            }
            for rank, index in enumerate(ranked, start=1)
        ]
