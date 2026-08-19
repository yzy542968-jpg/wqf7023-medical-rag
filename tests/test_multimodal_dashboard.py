from __future__ import annotations

import numpy as np

from medical_rag.dashboard.multimodal_runtime import (
    answer_with_evidence_agent,
    paired_shortlist_retrieve,
)
from medical_rag.retrieval.bm25_retriever import BM25Retriever


def test_dashboard_paired_retrieval_uses_image_inside_text_shortlist() -> None:
    cases = {
        "A": {"case_id": "A", "report_text": "clear lungs", "impression": "Clear lungs."},
        "B": {"case_id": "B", "report_text": "clear lungs", "impression": "Small effusion."},
        "C": {"case_id": "C", "report_text": "fracture", "impression": "Fracture."},
    }
    ids = ["A", "B", "C"]
    bm25 = BM25Retriever().fit([cases[case_id] for case_id in ids])
    results = paired_shortlist_retrieve(
        question="What is the impression?",
        indication="clear lungs",
        candidate_ids=ids,
        cases=cases,
        bm25=bm25,
        image_embedding=np.array([1.0, 0.0]),
        report_embeddings=np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]]),
        shortlist_size=2,
        text_weight=0.5,
        top_k=3,
    )
    assert results[0]["case_id"] == "B"
    assert results[-1]["case_id"] == "C"


def test_dashboard_agent_extracts_requested_field_and_checks_support() -> None:
    result = answer_with_evidence_agent(
        "What are the findings?",
        {"findings": "Small right pleural effusion.", "impression": "Pleural effusion."},
    )
    assert result["plan"]["answer_field"] == "findings"
    assert result["final_answer"] == "Small right pleural effusion."
    assert result["support_rate"] == 1.0
    assert result["abstained"] is False


def test_dashboard_agent_supports_non_oracle_generator_callback() -> None:
    def fake_generator(prompt: str, model_name: str) -> tuple[str, str]:
        assert "Selected report evidence:" in prompt
        assert model_name == "local-test-model"
        return "raw model output", "The lungs are clear."

    result = answer_with_evidence_agent(
        "What is the impression?",
        {"case_id": "A", "findings": "The lungs are clear.", "impression": "Clear lungs."},
        generator=fake_generator,
        model_name="local-test-model",
    )
    assert result["generation_mode"] == "qwen_non_oracle"
    assert result["raw_answer"] == "raw model output"
    assert result["draft_answer"] == "The lungs are clear."
