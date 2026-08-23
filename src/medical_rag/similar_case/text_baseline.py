from __future__ import annotations

from collections.abc import Sequence

from medical_rag.retrieval.bm25_retriever import BM25Retriever
from medical_rag.similar_case.retrieval import ScoredCase
from medical_rag.similar_case.schema import PairedCase


class SimilarCaseBM25Retriever:
    """BM25 over historical reports with explicit query-bank separation checks."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self._retriever = BM25Retriever(k1=k1, b=b)
        self._bank: tuple[PairedCase, ...] = ()

    def fit(
        self,
        cases: Sequence[PairedCase],
        *,
        require_patient_ids: bool = True,
    ) -> "SimilarCaseBM25Retriever":
        ordered = tuple(sorted(cases, key=lambda case: case.study_id))
        study_ids = [case.study_id for case in ordered]
        if not ordered:
            raise ValueError("The historical candidate bank cannot be empty.")
        if len(study_ids) != len(set(study_ids)):
            raise ValueError("The historical candidate bank has duplicate study IDs.")
        if require_patient_ids and any(case.patient_id is None for case in ordered):
            raise ValueError("Formal V9 candidate banks require patient IDs.")
        empty_reports = [case.study_id for case in ordered if not case.report_text]
        if empty_reports:
            raise ValueError(
                "Every BM25 candidate requires report text; missing for "
                f"{len(empty_reports)} studies."
            )

        self._bank = ordered
        self._retriever.fit(
            [
                {
                    "case_id": case.study_id,
                    "report_text": case.report_text,
                    "findings": case.findings,
                    "impression": case.impression,
                    "images": list(case.image_paths),
                }
                for case in ordered
            ]
        )
        return self

    def _assert_query_separation(self, query: PairedCase) -> None:
        if not self._bank:
            raise RuntimeError("Retriever has not been fitted.")
        if any(case.study_id == query.study_id for case in self._bank):
            raise ValueError("The target study is present in the historical bank.")
        if query.patient_id is not None and any(
            case.patient_id == query.patient_id for case in self._bank
        ):
            raise ValueError("A target-patient study is present in the historical bank.")

    def search(
        self,
        query: PairedCase,
        question: str,
        *,
        top_k: int = 10,
    ) -> list[ScoredCase]:
        self._assert_query_separation(query)
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        rows = self._retriever.search(
            query.query_text(question),
            top_k=min(top_k, len(self._bank)),
        )
        return [
            ScoredCase(
                study_id=str(row["case_id"]),
                score=float(row["score"]),
                component_scores={"bm25": float(row["score"])},
            )
            for row in rows
        ]
