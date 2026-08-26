"""Deterministic question-intent planner for V11 development.

It is intentionally small and inspectable. It is a planning component, not a
clinical diagnosis model and not evidence that a question was understood by a
human expert.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionPlan:
    intent: str
    evidence_preferences: tuple[str, ...]
    answer_style: str
    requires_target_image: bool
    abstain_without_direct_support: bool


_RULES: tuple[tuple[str, str, tuple[str, ...], str, bool], ...] = (
    (r"\b(change|changed|interval|compare|compared|comparison|improv\w*|worsen\w*|new|prior)\b", "comparison", ("comparison", "impression", "findings"), "comparison_with_uncertainty", True),
    (r"\b(insufficient|not enough|enough information|enough evidence|cannot be determined|can .* be determined|can .* be identified|unknown|unclear)\b", "insufficient_information", ("impression", "findings", "radgraph"), "evidence_sufficiency", True),
    (r"\b(device|line\w*|tube\w*|catheter|port|hardware|pacemaker)\b", "device", ("findings", "radgraph", "impression"), "device_status", True),
    (r"\b(where|location|located|distribution|side|anatomic|region|lobe|lateral|upper|lower|right|left)\b", "location", ("findings", "radgraph", "impression"), "finding_with_location", True),
    (r"\b(size|sized|severity|severe|mild|moderate|large|small|extent|extensive|degree|prominent)\b", "severity", ("findings", "radgraph", "impression"), "finding_with_severity", True),
    (r"\b(uncertain|possible|probable|likelihood|differential|cannot exclude|could|exclude\w*|suggested)\b", "uncertainty", ("impression", "findings", "radgraph"), "qualified_answer", True),
    (r"\b(overall|summarize|summary|in brief|principal .* conclusion|main .* findings)\b", "summary", ("impression", "findings", "radgraph"), "concise_summary", True),
    (r"\b(show|shows|demonstrate|demonstrates|signs of|opacity|effusion|pneumonia|fracture|edema|acute|abnormal|abnormality|present|evidence of|any)\b", "presence", ("impression", "findings", "radgraph"), "binary_finding_with_reason", True),
)


def plan_question(question: str, indication: str = "") -> QuestionPlan:
    # The question is the primary control signal. Indication is only a
    # fallback for malformed/empty questions, so unrelated indication terms
    # cannot silently change the requested evidence type.
    text = " ".join(str(question or "").lower().split())
    if not text:
        text = " ".join(str(indication or "").lower().split())
    for pattern, intent, preferences, style, requires_image in _RULES:
        if re.search(pattern, text):
            return QuestionPlan(intent, preferences, style, requires_image, True)
    return QuestionPlan("summary", ("impression", "findings", "radgraph"), "concise_summary", True, True)


def render_planner_instruction(plan: QuestionPlan) -> str:
    return (
        f"Intent={plan.intent}; prefer evidence sections={','.join(plan.evidence_preferences)}; "
        f"answer_style={plan.answer_style}; target_image_required={str(plan.requires_target_image).lower()}; "
        f"abstain_without_direct_support={str(plan.abstain_without_direct_support).lower()}."
    )


__all__ = ["QuestionPlan", "plan_question", "render_planner_instruction"]
