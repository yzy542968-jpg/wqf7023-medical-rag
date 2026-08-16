from __future__ import annotations

import numpy as np

from medical_rag.retrieval.hybrid_retriever import HybridBM25MedCPTRetriever
from medical_rag.retrieval.medcpt_reranker import case_document


class FakeBM25:
    def _score_document(self, query_terms: list[str], index: int) -> float:
        return float(2 - index)


class FakeMedCPT:
    case_ids = ["C1", "C2"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")


def test_hybrid_results_preserve_fields_needed_by_reranker() -> None:
    cases = [
        {
            "case_id": "C1",
            "indication": "fever",
            "problems": "pneumonia",
            "findings": "left opacity",
            "impression": "left lower lobe pneumonia",
        },
        {
            "case_id": "C2",
            "indication": "screening",
            "problems": "",
            "findings": "clear lungs",
            "impression": "no acute disease",
        },
    ]
    retriever = HybridBM25MedCPTRetriever.from_components(
        cases, FakeBM25(), FakeMedCPT(), alpha=0.3
    )

    result = retriever.search_with_embedding(
        "fever pneumonia", np.array([1.0, 0.0], dtype="float32"), top_k=1
    )[0]

    assert result["indication"] == "fever"
    assert result["problems"] == "pneumonia"
    assert "Indication: fever" in case_document(result)
    assert "Problems: pneumonia" in case_document(result)
