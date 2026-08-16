from __future__ import annotations

from typing import Any


DEFAULT_RERANKER_MODEL = "ncbi/MedCPT-Cross-Encoder"


def case_document(case: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in [
            f"Indication: {case.get('indication', '')}",
            f"Problems: {case.get('problems', '')}",
            f"Findings: {case.get('findings', '')}",
            f"Impression: {case.get('impression', '')}",
        ]
        if part.strip()
    )


def rerank_by_scores(candidates: list[dict[str, Any]], scores: list[float]) -> list[dict[str, Any]]:
    if len(candidates) != len(scores):
        raise ValueError("candidates and scores must have the same length")
    indexed = sorted(
        zip(candidates, scores, strict=True),
        key=lambda value: value[1],
        reverse=True,
    )
    return [
        {**candidate, "reranker_score": float(score), "reranked_rank": rank}
        for rank, (candidate, score) in enumerate(indexed, start=1)
    ]


class MedCPTReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 32,
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional environment
            raise RuntimeError("MedCPT reranking requires torch and transformers") from exc

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, local_files_only=local_files_only
        ).to(self.device)
        self.model.eval()

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            encoded = self.tokenizer(
                [[query, document] for query, document in batch],
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits.squeeze(dim=1)
            scores.extend(float(value) for value in logits.detach().cpu().tolist())
        return scores

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pairs = [(query, case_document(candidate)) for candidate in candidates]
        return rerank_by_scores(candidates, self.score(pairs))
