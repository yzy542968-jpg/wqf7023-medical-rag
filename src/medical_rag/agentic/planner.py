from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryPlan:
    question_type: str
    answer_field: str
    retrieval_query: str
    requires_evidence: bool = True


def plan_question(question: str, question_type: str | None = None) -> QueryPlan:
    inferred_type = question_type or "unknown"
    lower_question = question.lower()

    if inferred_type == "findings_from_indication" or "findings" in lower_question:
        answer_field = "findings"
    elif inferred_type == "impression_from_indication" or "impression" in lower_question:
        answer_field = "impression"
    else:
        answer_field = "impression_or_findings"

    return QueryPlan(
        question_type=inferred_type,
        answer_field=answer_field,
        retrieval_query=question,
        requires_evidence=True,
    )

