from medical_rag.agentic.v9_historical_evidence_agent import (
    run_bounded_historical_evidence_agent,
)


class FakePredictor:
    def __init__(self, entailment: float) -> None:
        self.entailment = entailment

    def predict(self, pairs):
        return [
            {
                "entailment": self.entailment,
                "neutral": 1.0 - self.entailment,
                "contradiction": 0.0,
            }
            for _ in pairs
        ]


CASES = {
    "A": {"findings": "There is a left pleural effusion.", "impression": "Left effusion."},
    "B": {"findings": "The lungs are clear.", "impression": "Normal chest."},
}


def run(row, primary, retry, entailment):
    return run_bounded_historical_evidence_agent(
        answer_row=row,
        primary_case_ids=primary,
        retry_case_ids=retry,
        cases=CASES,
        predictor=FakePredictor(entailment),
        minimum_support_rate=0.5,
        minimum_combined_support=0.55,
        entailment_threshold=0.75,
        contradiction_threshold=0.5,
    )


def test_no_historical_claim_needs_no_retry() -> None:
    result = run({"answer": "Clear lungs.", "historical_support": ""}, ["A"], ["B"], 0.9)
    assert result["retried"] is False
    assert result["historical_evidence_abstained"] is False


def test_supported_claim_is_retained() -> None:
    row = {
        "answer": "There is an effusion.",
        "historical_support": "There is a left pleural effusion.",
        "supporting_case_ids": ["A"],
        "uncertainty": "low",
    }
    result = run(row, ["A"], ["B"], 0.9)
    assert result["retried"] is False
    assert result["agent_historical_support"] == row["historical_support"]


def test_unsupported_claim_is_removed_after_retry() -> None:
    row = {
        "answer": "There is an effusion.",
        "historical_support": "There is a left pleural effusion.",
        "supporting_case_ids": ["B"],
        "uncertainty": "low",
    }
    result = run(row, ["B"], ["B"], 0.1)
    assert result["retried"] is True
    assert result["historical_evidence_abstained"] is True
    assert result["agent_historical_support"] == ""
    assert result["agent_answer"] == row["answer"]
