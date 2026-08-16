from __future__ import annotations

import re


ABSTENTION_TEXT = "The retrieved report evidence is insufficient to answer this question."


def _field(prompt: str, label: str) -> str:
    prefix = label.lower()
    for line in prompt.splitlines():
        if line.lower().startswith(prefix):
            return line[len(label) :].strip()
    return ""


def extractive_demo_answer(prompt: str) -> str:
    """Produce a deterministic answer when model artifacts are unavailable."""
    lower = prompt.lower()
    findings = _field(prompt, "Findings:")
    impression = _field(prompt, "Impression:")
    if "impression" in lower and impression:
        return impression
    if "finding" in lower and findings:
        return findings

    evidence_match = re.search(
        r"(?is)Retrieved evidence:\s*(.+?)(?:\n\n(?:Answer clearly|Final answer:)|\Z)",
        prompt,
    )
    if evidence_match:
        evidence = evidence_match.group(1).strip()
        sentences = [
            re.sub(r"^\[[^]]+\]\s*", "", line).strip()
            for line in evidence.splitlines()
            if line.strip()
        ]
        if sentences:
            return " ".join(sentences[:2])
    return impression or findings or ABSTENTION_TEXT
