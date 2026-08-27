from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_v13_concept_qa_pilot import (  # noqa: E402
    add_concept_instruction,
    concept_instruction,
    threshold_passing_concepts,
)


def test_concepts_are_thresholded_ranked_and_bounded() -> None:
    probabilities = [0.1] * 14
    thresholds = [0.5] * 14
    probabilities[1] = 0.9
    probabilities[2] = 0.8
    probabilities[4] = 0.7
    probabilities[7] = 0.6
    probabilities[9] = 0.55
    probabilities[12] = 0.52
    probabilities[13] = 0.95
    selected = threshold_passing_concepts(probabilities, thresholds)
    assert len(selected) == 5
    assert selected[0][0] == "Cardiomegaly"
    assert all(label != "No Finding" for label, _ in selected)


def test_empty_concept_instruction_does_not_assert_normality() -> None:
    text = concept_instruction(())
    assert "no confident concept" in text
    assert "Do not infer normality" in text


def test_concept_line_is_inserted_before_case_inputs() -> None:
    prompt = "Instruction\nIndication: cough\nQuestion: findings"
    result = add_concept_instruction(prompt, "Automated target-image hypotheses: edema")
    lines = result.splitlines()
    assert lines[1].startswith("Automated target-image")
    assert lines[2].startswith("Indication:")
