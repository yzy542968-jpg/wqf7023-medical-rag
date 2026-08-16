from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from medical_rag.retrieval.scoped_chunk_retriever import ScopedBM25ChunkRetriever
from medical_rag.retrieval.tfidf_retriever import _tokens


OUT_OF_SCOPE_TERMS = {
    "attenuation",
    "discharge",
    "hounsfield",
    "medication",
    "oxygen saturation",
    "pathology",
    "prescribed",
    "serum",
    "specimen",
    "treatment",
    "troponin",
}


@dataclass(frozen=True)
class AgentStep:
    step: int
    action: str
    intent: str
    query: str
    allowed_sections: list[str]
    retrieved_chunk_ids: list[str]
    evidence_score: float
    reason: str


@dataclass(frozen=True)
class ClosedLoopResult:
    question: str
    scope_case_id: str
    planned_intent: str
    final_intent: str
    answer_probability: float
    recommended_action: str
    answer: str
    retrieved_chunk_ids: list[str]
    retrieved_sections: list[str]
    retrieved_texts: list[str]
    retrieved_scores: list[float]
    retrieval_calls: int
    retrieved_chunk_count: int
    retried: bool
    trace: list[dict[str, Any]]


def infer_report_intent(question: str) -> str:
    lowered = " ".join(question.lower().split())
    if any(term in lowered for term in OUT_OF_SCOPE_TERMS):
        return "unavailable"
    if any(
        term in lowered
        for term in ("conclusion", "assessment", "bottom line", "radiologist reach", "diagnostic")
    ):
        return "impression"
    if any(
        term in lowered
        for term in ("observation", "abnormalit", "images", "radiographic", "examination show")
    ):
        return "findings"
    return "unknown"


def _rewrite_query(question: str) -> tuple[str, str]:
    lowered = question.lower()
    if "summar" in lowered:
        return f"{question} final radiology impression conclusion", "impression"
    return f"{question} radiographic findings impression", "findings"


def _lexical_overlap(question: str, rows: list[dict[str, Any]]) -> float:
    query_terms = set(_tokens(question))
    evidence_terms = set(_tokens(" ".join(str(row.get("text", "")) for row in rows)))
    if not query_terms:
        return 0.0
    return len(query_terms & evidence_terms) / len(query_terms)


def _evidence_score(
    question: str, rows: list[dict[str, Any]], route_credit: float
) -> float:
    if not rows:
        return 0.0
    top_score = max(float(row.get("score", 0.0)) for row in rows)
    normalized_bm25 = top_score / (top_score + 3.0) if top_score > 0 else 0.0
    overlap = _lexical_overlap(question, rows)
    return min(0.99, route_credit + 0.25 * normalized_bm25 + 0.20 * overlap)


class ClosedLoopEvidenceAgent:
    """Deterministic, auditable report agent with at most two retrieval calls."""

    def __init__(
        self,
        retriever: ScopedBM25ChunkRetriever,
        first_pass_k: int = 3,
        retry_k: int = 3,
        retry_threshold: float = 0.50,
    ) -> None:
        self.retriever = retriever
        self.first_pass_k = first_pass_k
        self.retry_k = retry_k
        self.retry_threshold = retry_threshold

    def run(
        self,
        question: str,
        scope_case_id: str,
        planned_intent: str | None = None,
    ) -> ClosedLoopResult:
        external_plan = planned_intent is not None
        planned_intent = planned_intent or infer_report_intent(question)
        if planned_intent not in {"findings", "impression", "unavailable", "unknown"}:
            raise ValueError(f"Unsupported planned intent: {planned_intent}")
        if planned_intent == "unavailable":
            step = AgentStep(
                step=1,
                action="ABSTAIN_OUT_OF_SCOPE",
                intent=planned_intent,
                query=question,
                allowed_sections=[],
                retrieved_chunk_ids=[],
                evidence_score=0.02,
                reason="The requested information is outside a chest-radiograph report.",
            )
            return ClosedLoopResult(
                question=question,
                scope_case_id=scope_case_id,
                planned_intent=planned_intent,
                final_intent=planned_intent,
                answer_probability=0.02,
                recommended_action="ABSTAIN",
                answer="NOT ANSWERABLE",
                retrieved_chunk_ids=[],
                retrieved_sections=[],
                retrieved_texts=[],
                retrieved_scores=[],
                retrieval_calls=0,
                retrieved_chunk_count=0,
                retried=False,
                trace=[asdict(step)],
            )

        allowed = {planned_intent} if planned_intent in {"findings", "impression"} else None
        first_rows = self.retriever.search(
            question,
            top_k=self.first_pass_k,
            case_id=scope_case_id,
            allowed_sections=allowed,
        )
        first_score = _evidence_score(
            question, first_rows, route_credit=0.55 if allowed is not None else 0.15
        )
        steps = [
            AgentStep(
                step=1,
                action="RETRIEVE_TARGET_SECTION" if allowed else "RETRIEVE_REPORT",
                intent=planned_intent,
                query=question,
                allowed_sections=sorted(allowed or []),
                retrieved_chunk_ids=[str(row["chunk_id"]) for row in first_rows],
                evidence_score=first_score,
                reason=(
                    "Initial plan supplied by the constrained semantic planner."
                    if external_plan
                    else "Initial plan from question wording."
                ),
            )
        ]
        rows = list(first_rows)
        final_intent = planned_intent
        retried = False

        if first_score < self.retry_threshold:
            rewritten, retry_intent = _rewrite_query(question)
            retry_sections = {retry_intent}
            retry_rows = self.retriever.search(
                rewritten,
                top_k=self.retry_k,
                case_id=scope_case_id,
                allowed_sections=retry_sections,
            )
            retry_route_credit = 0.55 if "summar" in question.lower() else 0.15
            retry_score = _evidence_score(
                rewritten, retry_rows, route_credit=retry_route_credit
            )
            steps.append(
                AgentStep(
                    step=2,
                    action="REWRITE_AND_RETRY",
                    intent=retry_intent,
                    query=rewritten,
                    allowed_sections=sorted(retry_sections),
                    retrieved_chunk_ids=[str(row["chunk_id"]) for row in retry_rows],
                    evidence_score=retry_score,
                    reason="Initial evidence score was below the fixed retry threshold.",
                )
            )
            rows.extend(retry_rows)
            final_intent = retry_intent
            first_score = max(first_score, retry_score)
            retried = True

        unique_rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            chunk_id = str(row["chunk_id"])
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique_rows.append(row)

        recommended_action = "ANSWER" if first_score >= self.retry_threshold else "ABSTAIN"
        answer = (
            " ".join(str(row["text"]) for row in unique_rows)
            if recommended_action == "ANSWER"
            else "NOT ANSWERABLE"
        )
        return ClosedLoopResult(
            question=question,
            scope_case_id=scope_case_id,
            planned_intent=planned_intent,
            final_intent=final_intent,
            answer_probability=first_score,
            recommended_action=recommended_action,
            answer=answer,
            retrieved_chunk_ids=[str(row["chunk_id"]) for row in unique_rows],
            retrieved_sections=[str(row["section"]) for row in unique_rows],
            retrieved_texts=[str(row["text"]) for row in unique_rows],
            retrieved_scores=[float(row.get("score", 0.0)) for row in unique_rows],
            retrieval_calls=len(steps),
            retrieved_chunk_count=len(unique_rows),
            retried=retried,
            trace=[asdict(step) for step in steps],
        )
