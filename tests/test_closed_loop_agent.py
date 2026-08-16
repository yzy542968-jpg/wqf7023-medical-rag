from __future__ import annotations

from medical_rag.agentic.closed_loop_agent import (
    ClosedLoopEvidenceAgent,
    infer_report_intent,
)
from medical_rag.evaluation.case_scoped_benchmark import build_case_chunks
from medical_rag.retrieval.scoped_chunk_retriever import ScopedBM25ChunkRetriever


def _agent() -> ClosedLoopEvidenceAgent:
    case = {
        "case_id": "CXR1",
        "indication": "Cough.",
        "comparison": "None.",
        "findings": "A focal right lower lobe opacity is present. No effusion.",
        "impression": "Right lower lobe pneumonia.",
    }
    retriever = ScopedBM25ChunkRetriever().fit(build_case_chunks(case))
    return ClosedLoopEvidenceAgent(retriever)


def test_agent_routes_without_receiving_gold_question_type() -> None:
    result = _agent().run("What conclusion did the radiologist reach?", "CXR1")
    assert infer_report_intent(result.question) == "impression"
    assert result.recommended_action == "ANSWER"
    assert set(result.retrieved_sections) == {"impression"}


def test_agent_abstains_without_retrieval_for_out_of_scope_request() -> None:
    result = _agent().run("What was the serum troponin concentration?", "CXR1")
    assert result.answer == "NOT ANSWERABLE"
    assert result.retrieval_calls == 0
    assert result.trace[0]["action"] == "ABSTAIN_OUT_OF_SCOPE"


def test_agent_retries_ambiguous_summary_within_fixed_budget() -> None:
    result = _agent().run("Please summarize this report.", "CXR1")
    assert result.retried
    assert result.final_intent == "impression"
    assert result.retrieval_calls == 2
    assert result.retrieved_chunk_count <= 6


def test_external_semantic_plan_can_override_lexical_unknown() -> None:
    result = _agent().run(
        "State the reporting clinician's overall interpretation.",
        "CXR1",
        planned_intent="impression",
    )
    assert result.planned_intent == "impression"
    assert result.final_intent == "impression"
    assert set(result.retrieved_sections) == {"impression"}
