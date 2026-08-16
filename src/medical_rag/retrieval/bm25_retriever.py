from __future__ import annotations

import math
from collections import Counter
from typing import Any

from medical_rag.retrieval.tfidf_retriever import _tokens


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.cases: list[dict[str, Any]] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_term_counts: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_length = 0.0
        self.idf: dict[str, float] = {}

    def fit(self, cases: list[dict[str, Any]]) -> "BM25Retriever":
        self.cases = cases
        self.doc_tokens = [_tokens(case.get("report_text", "")) for case in cases]
        self.doc_term_counts = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0

        document_frequency: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            document_frequency.update(set(tokens))

        document_count = len(self.doc_tokens)
        self.idf = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        return self

    def _score_document(self, query_terms: list[str], index: int) -> float:
        if not self.avg_doc_length:
            return 0.0

        score = 0.0
        term_counts = self.doc_term_counts[index]
        doc_length = self.doc_lengths[index]
        for term in query_terms:
            term_frequency = term_counts.get(term, 0)
            if term_frequency == 0:
                continue
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * doc_length / self.avg_doc_length
            )
            score += self.idf.get(term, 0.0) * (
                term_frequency * (self.k1 + 1) / denominator
            )
        return score

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.cases:
            raise RuntimeError("Retriever has not been fitted.")

        query_terms = _tokens(query)
        scores = [self._score_document(query_terms, index) for index in range(len(self.cases))]
        ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]

        results: list[dict[str, Any]] = []
        for rank, index in enumerate(ranked_indices, start=1):
            case = self.cases[index]
            results.append(
                {
                    "rank": rank,
                    "case_id": case["case_id"],
                    "score": float(scores[index]),
                    "findings": case.get("findings", ""),
                    "impression": case.get("impression", ""),
                    "images": case.get("images", []),
                }
            )
        return results

