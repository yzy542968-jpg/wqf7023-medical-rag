from __future__ import annotations

from medical_rag.qa.medgemma_contract import (
    build_compact_qa_prompt,
    parse_option_indices,
)


def test_compact_prompt_has_no_gold_history() -> None:
    prompt = build_compact_qa_prompt(
        question="Is there edema?",
        options=("yes", "no"),
        indication="Dyspnea",
        image_available=True,
    )
    assert "0: yes" in prompt
    assert "1: no" in prompt
    assert "Dyspnea" in prompt
    assert "previous answer" not in prompt.lower()


def test_parser_accepts_one_compact_array() -> None:
    parsed = parse_option_indices(
        "[2,0]", option_count=3, answer_type="multi_choice"
    )
    assert parsed["indices"] == [0, 2]
    assert parsed["contract_valid"] is True


def test_parser_retains_invalid_output_as_empty() -> None:
    assert parse_option_indices(
        "The answer is [0].", option_count=2, answer_type="single_choice"
    )["contract_valid"] is False
    assert parse_option_indices(
        "[0,1]", option_count=2, answer_type="single_choice"
    )["contract_valid"] is False
    assert parse_option_indices(
        "[3]", option_count=2, answer_type="single_choice"
    )["indices"] == []
