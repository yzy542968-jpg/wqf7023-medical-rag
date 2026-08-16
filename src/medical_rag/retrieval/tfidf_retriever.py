from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


def load_cases_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalize(vector: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0:
        return vector
    return {key: value / norm for key, value in vector.items()}


def _dot(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


class TfidfRetriever:
    def __init__(self) -> None:
        self.cases: list[dict[str, Any]] = []
        self.idf: dict[str, float] = {}
        self.doc_vectors: list[dict[str, float]] = []

    def fit(self, cases: list[dict[str, Any]]) -> "TfidfRetriever":
        self.cases = cases
        tokenized_documents = [_tokens(case.get("report_text", "")) for case in cases]
        document_count = len(tokenized_documents)
        document_frequency: Counter[str] = Counter()

        for tokens in tokenized_documents:
            document_frequency.update(set(tokens))

        self.idf = {
            term: math.log((1 + document_count) / (1 + frequency)) + 1
            for term, frequency in document_frequency.items()
        }

        self.doc_vectors = [
            self._vectorize_tokens(tokens)
            for tokens in tokenized_documents
        ]
        return self

    def _vectorize_tokens(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        if not counts:
            return {}
        total = sum(counts.values())
        vector = {
            term: (count / total) * self.idf.get(term, 0.0)
            for term, count in counts.items()
        }
        return _normalize(vector)

    def _vectorize_query(self, query: str) -> dict[str, float]:
        return self._vectorize_tokens(_tokens(query))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.doc_vectors:
            raise RuntimeError("Retriever has not been fitted.")

        query_vector = self._vectorize_query(query)
        scores = [_dot(query_vector, doc_vector) for doc_vector in self.doc_vectors]
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
