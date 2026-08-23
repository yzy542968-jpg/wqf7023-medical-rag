import numpy as np
import torch

from medical_rag.dashboard.v9_runtime import (
    V9DashboardResources,
    V9MLPScorer,
    infer_question_type,
    retrieve_v9,
)
from medical_rag.retrieval.bm25_retriever import BM25Retriever


def resources() -> V9DashboardResources:
    cases = {
        "A": {"case_id": "A", "report_text": "clear lungs", "findings": "Clear.", "impression": "Normal."},
        "B": {"case_id": "B", "report_text": "left opacity", "findings": "Opacity.", "impression": "Pneumonia."},
    }
    model = V9MLPScorer()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[-1].bias.fill_(0.0)
    return V9DashboardResources(
        cases=cases,
        candidate_ids=("A", "B"),
        bm25=BM25Retriever().fit([cases["A"], cases["B"]]),
        image_embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        report_embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        model=model,
    )


def test_question_type_rules() -> None:
    assert infer_question_type("What is the impression?") == "impression"
    assert infer_question_type("Is there an acute abnormality?") == "acute"
    assert infer_question_type("What are the findings?") == "findings"


def test_runtime_returns_other_patient_rows() -> None:
    rows = retrieve_v9(
        indication="cough",
        question="What are the findings?",
        image_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        resources=resources(),
        top_k=2,
    )
    assert len(rows) == 2
    assert {row["case_id"] for row in rows} == {"A", "B"}
    assert all("learned_score" in row for row in rows)
