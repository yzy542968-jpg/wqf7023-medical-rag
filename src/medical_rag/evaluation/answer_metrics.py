from __future__ import annotations

import re
from collections import Counter


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def extract_final_answer(text: str) -> str:
    matches = list(re.finditer(r"(?is)\bfinal\s+answer\s*:\s*(.*)", text or ""))
    if not matches:
        return (text or "").strip()
    answer = matches[-1].group(1).strip()
    answer = re.sub(r"^\s*[-*]\s+", "", answer)
    return answer.strip()


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
