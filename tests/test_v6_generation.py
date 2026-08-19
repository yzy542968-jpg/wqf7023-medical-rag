from __future__ import annotations

from medical_rag.multimodal.v6_generation import build_v6_qa_prompt, select_preflight_qids


def test_v6_prompt_contains_same_required_evidence_fields() -> None:
    prompt = build_v6_qa_prompt(
        {"question": "What is present?"},
        {"indication": "Cough"},
        {"findings": "Small opacity.", "impression": "Possible pneumonia."},
    )

    assert "Clinical indication: Cough" in prompt
    assert "Question: What is present?" in prompt
    assert "Findings: Small opacity." in prompt
    assert "Impression: Possible pneumonia." in prompt
    assert "Insufficient evidence." in prompt


def test_v6_prompt_handles_missing_retrieval_without_fabricating_report() -> None:
    prompt = build_v6_qa_prompt(
        {"question": "What is present?"},
        {"indication": ""},
        None,
    )

    assert "Clinical indication: Not provided" in prompt
    assert prompt.count("No report was retrieved.") == 2


def test_preflight_qids_are_deterministic_and_cover_ordered_range() -> None:
    assert select_preflight_qids(["q4", "q2", "q3", "q1"], 3) == ["q1", "q3", "q4"]
