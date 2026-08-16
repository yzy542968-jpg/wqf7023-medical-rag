from __future__ import annotations

from medical_rag.agentic.action_policy import apply_verifier_action


def check(sentence: str, *, supported: bool, contradiction: float, consistent: bool = True) -> dict:
    return {
        "sentence": sentence,
        "supported": supported,
        "contradiction_probability": contradiction,
        "negation_consistent": consistent,
    }


def test_contradiction_only_keeps_neutral_sentence_but_removes_conflict() -> None:
    checks = [
        check("Possible opacity.", supported=False, contradiction=0.2),
        check("Pleural effusion.", supported=True, contradiction=0.9),
    ]
    result = apply_verifier_action(
        "Possible opacity. Pleural effusion.",
        checks,
        action_policy="contradiction_only",
        contradiction_threshold=0.5,
    )
    assert result.answer == "Possible opacity."
    assert result.rejected_sentences == ["Pleural effusion."]


def test_audit_only_never_changes_draft() -> None:
    draft = "Free-form answer."
    result = apply_verifier_action(
        draft,
        [check(draft, supported=False, contradiction=0.9)],
        action_policy="audit_only",
    )
    assert result.answer == draft
    assert not result.revised
