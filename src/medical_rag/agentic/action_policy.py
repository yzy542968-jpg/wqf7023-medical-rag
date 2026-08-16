from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


ABSTENTION_TEXT = "The retrieved report evidence is insufficient to answer this question."


@dataclass
class VerifierActionResult:
    answer: str
    accepted_sentences: list[str]
    rejected_sentences: list[str]
    abstained: bool
    revised: bool


def _value(check: Any, key: str) -> Any:
    return check[key] if isinstance(check, dict) else getattr(check, key)


def apply_verifier_action(
    draft_answer: str,
    sentence_checks: Iterable[Any],
    *,
    action_policy: str,
    contradiction_threshold: float = 0.5,
) -> VerifierActionResult:
    checks = list(sentence_checks)
    accepted = []
    rejected = []
    for check in checks:
        if action_policy == "sentence_filter":
            keep = bool(_value(check, "supported"))
        elif action_policy == "contradiction_only":
            keep = bool(_value(check, "negation_consistent")) and float(
                _value(check, "contradiction_probability")
            ) < contradiction_threshold
        elif action_policy == "audit_only":
            keep = True
        else:
            raise ValueError(f"Unknown verifier action policy: {action_policy}")
        (accepted if keep else rejected).append(str(_value(check, "sentence")))

    if action_policy == "audit_only":
        answer = draft_answer
    else:
        answer = " ".join(accepted) if accepted else ABSTENTION_TEXT
    return VerifierActionResult(
        answer=answer,
        accepted_sentences=accepted,
        rejected_sentences=rejected,
        abstained=not accepted,
        revised=answer.strip() != draft_answer.strip(),
    )
