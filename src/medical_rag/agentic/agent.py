from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from medical_rag.agentic.evidence_checker import EvidenceCheckResult, check_evidence_support
from medical_rag.agentic.planner import QueryPlan, plan_question


@dataclass
class AgenticRAGResult:
    plan: QueryPlan
    retrieved_cases: list[dict[str, Any]]
    draft_answer: str
    final_answer: str
    evidence_check: EvidenceCheckResult
    revised: bool
    abstained: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": asdict(self.plan),
            "retrieved_cases": self.retrieved_cases,
            "draft_answer": self.draft_answer,
            "final_answer": self.final_answer,
            "evidence_check": {
                "supported_sentences": self.evidence_check.supported_sentences,
                "unsupported_sentences": self.evidence_check.unsupported_sentences,
                "sentence_checks": [asdict(check) for check in self.evidence_check.sentence_checks],
                "support_rate": self.evidence_check.support_rate,
                "revised_answer": self.evidence_check.revised_answer,
                "abstained": self.evidence_check.abstained,
            },
            "revised": self.revised,
            "abstained": self.abstained,
        }


def _draft_answer_from_case(plan: QueryPlan, case: dict[str, Any]) -> str:
    findings = case.get("findings", "") or ""
    impression = case.get("impression", "") or ""

    if plan.answer_field == "findings":
        return findings.strip()
    if plan.answer_field == "impression":
        return impression.strip()
    return (impression or findings).strip()


def _evidence_text(retrieved_cases: list[dict[str, Any]]) -> str:
    blocks = []
    for case in retrieved_cases:
        blocks.append(
            "\n".join(
                [
                    f"Case ID: {case.get('case_id', '')}",
                    f"Findings: {case.get('findings', '')}",
                    f"Impression: {case.get('impression', '')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def run_rule_based_agent(
    question: str,
    question_type: str,
    retrieved_cases: list[dict[str, Any]],
    min_sentence_support: float = 0.65,
    min_retrieval_score: float = 0.0,
) -> AgenticRAGResult:
    plan = plan_question(question, question_type)

    if not retrieved_cases:
        evidence_check = check_evidence_support("", "", min_sentence_support=min_sentence_support)
        return AgenticRAGResult(
            plan=plan,
            retrieved_cases=[],
            draft_answer="",
            final_answer=evidence_check.revised_answer,
            evidence_check=evidence_check,
            revised=True,
            abstained=True,
        )

    top_case = retrieved_cases[0]
    draft_answer = _draft_answer_from_case(plan, top_case)
    evidence_text = _evidence_text(retrieved_cases)
    evidence_check = check_evidence_support(
        draft_answer,
        evidence_text,
        min_sentence_support=min_sentence_support,
    )

    low_confidence = float(top_case.get("score", 0.0)) < min_retrieval_score
    if low_confidence:
        final_answer = "The retrieved report evidence is insufficient to answer this question."
        abstained = True
        revised = True
    else:
        final_answer = evidence_check.revised_answer
        abstained = evidence_check.abstained
        revised = final_answer.strip() != draft_answer.strip()

    return AgenticRAGResult(
        plan=plan,
        retrieved_cases=retrieved_cases,
        draft_answer=draft_answer,
        final_answer=final_answer,
        evidence_check=evidence_check,
        revised=revised,
        abstained=abstained,
    )

