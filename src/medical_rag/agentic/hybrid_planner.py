from __future__ import annotations

from dataclasses import dataclass

from medical_rag.agentic.closed_loop_agent import infer_report_intent


VALID_INTENTS = {"findings", "impression", "unavailable", "unknown"}
KNOWN_REPORT_FACT_PREFIX = "what does this report state about "


@dataclass(frozen=True)
class HybridPlan:
    lexical_intent: str
    semantic_intent: str
    selected_intent: str
    planner_source: str


def select_hybrid_plan(question: str, semantic_intent: str) -> HybridPlan:
    """Use semantic routing only outside frozen high-precision lexical forms."""
    if semantic_intent not in VALID_INTENTS:
        raise ValueError(f"Unsupported semantic intent: {semantic_intent}")
    lexical_intent = infer_report_intent(question)
    normalized = " ".join(question.lower().split())
    if lexical_intent != "unknown":
        selected = lexical_intent
        source = "lexical_intent"
    elif normalized.startswith(KNOWN_REPORT_FACT_PREFIX):
        selected = "unknown"
        source = "lexical_report_fact_frame"
    else:
        selected = semantic_intent
        source = "semantic_fallback"
    return HybridPlan(
        lexical_intent=lexical_intent,
        semantic_intent=semantic_intent,
        selected_intent=selected,
        planner_source=source,
    )
