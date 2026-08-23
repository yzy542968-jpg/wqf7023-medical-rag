from __future__ import annotations

from collections.abc import Sequence

from medical_rag.similar_case.schema import PairedCase


def build_evidence_constrained_prompt(
    *,
    indication: str,
    question: str,
    retrieved_cases: Sequence[PairedCase],
) -> str:
    evidence_blocks = []
    for index, case in enumerate(retrieved_cases, start=1):
        evidence_blocks.append(
            "\n".join(
                (
                    f"Historical case {index} [{case.study_id}]",
                    f"Findings: {case.findings or 'Not available'}",
                    f"Impression: {case.impression or 'Not available'}",
                )
            )
        )
    evidence = "\n\n".join(evidence_blocks) if evidence_blocks else "No historical cases retrieved."
    normalized_indication = " ".join(indication.split()) or "Not provided"
    normalized_question = " ".join(question.split())
    if not normalized_question:
        raise ValueError("question cannot be empty.")

    return f"""You are assisting with evidence-grounded chest-radiograph question answering.

The target radiograph is the patient being assessed. Historical cases are analogies only. They are not proof that the same findings are present in the target patient. Do not transfer a historical diagnosis to the target unless it is supported by the target image. If evidence is insufficient or conflicting, state the uncertainty or abstain.

Clinical indication:
{normalized_indication}

Question:
{normalized_question}

Retrieved other-patient historical evidence:
{evidence}

Return valid JSON with exactly these keys:
answer, target_image_findings, supporting_case_ids, historical_support, uncertainty, abstain.
"""
