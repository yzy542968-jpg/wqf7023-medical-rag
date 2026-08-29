from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any


def build_compact_qa_prompt(
    *,
    question: str,
    options: Sequence[str],
    indication: str | None,
    image_available: bool,
    historical_evidence: Sequence[str] = (),
) -> str:
    numbered = "\n".join(f"{index}: {option}" for index, option in enumerate(options))
    evidence = "\n".join(historical_evidence) if historical_evidence else "None"
    image_rule = (
        "Use the target chest radiograph as the primary patient evidence."
        if image_available
        else "No target image is provided in this shortcut-control condition."
    )
    return "\n".join(
        [
            "You are answering one independent structured radiology question.",
            image_rule,
            "Do not assume or reconstruct answers to any other question.",
            "Historical text, when present, describes other patients and is supportive evidence only.",
            "Select every correct option for the target case.",
            "Return only a JSON array of zero-based option indices, such as [0] or [0,2].",
            "Do not return words, markdown, reasoning, or any other JSON field.",
            "",
            f"Clinical indication: {' '.join(str(indication or 'Not provided').split())}",
            f"Question: {' '.join(str(question).split())}",
            "Options:",
            numbered,
            "Historical evidence:",
            evidence,
        ]
    )


def parse_option_indices(
    raw_output: str,
    *,
    option_count: int,
    answer_type: str,
) -> dict[str, Any]:
    raw = str(raw_output or "").strip()
    matches = re.findall(r"\[[^\[\]]*\]", raw)
    if len(matches) != 1:
        return {"indices": [], "contract_valid": False, "raw_output": raw}
    try:
        values = json.loads(matches[0])
    except json.JSONDecodeError:
        return {"indices": [], "contract_valid": False, "raw_output": raw}
    valid = (
        raw == matches[0]
        and isinstance(values, list)
        and all(isinstance(value, int) and not isinstance(value, bool) for value in values)
        and len(values) == len(set(values))
        and all(0 <= value < option_count for value in values)
    )
    if answer_type in {"single_choice", "fixed_choice"}:
        valid = valid and len(values) == 1
    elif answer_type == "multi_choice":
        valid = valid and len(values) >= 1
    else:
        valid = False
    return {
        "indices": sorted(values) if valid else [],
        "contract_valid": bool(valid),
        "raw_output": raw,
    }


__all__ = ["build_compact_qa_prompt", "parse_option_indices"]
