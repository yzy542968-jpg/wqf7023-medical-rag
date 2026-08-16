from __future__ import annotations

from scripts.build_case_scoped_prompt_pack_v2 import build_prompt


def test_prompt_exposes_case_scope_and_only_selected_evidence() -> None:
    question = {
        "scope_case_id": "CXR9",
        "question": "What is the impression?",
    }
    evidence = [{"section": "impression", "position": 1, "text": "No acute disease."}]
    prompt = build_prompt(question, evidence)
    assert "Case scope: CXR9" in prompt
    assert "[impression 1] No acute disease." in prompt
    assert "only the retrieved evidence" in prompt
