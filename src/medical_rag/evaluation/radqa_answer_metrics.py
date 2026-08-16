from __future__ import annotations

import re
from statistics import mean
from typing import Any

from medical_rag.evaluation.answer_metrics import extract_final_answer, token_f1


def normalize_answer(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return " ".join(tokens)


def is_not_answerable(text: str) -> bool:
    normalized = normalize_answer(extract_final_answer(text))
    return normalized == "not answerable" or normalized.startswith("not answerable ")


def best_reference_score(answer: str, references: list[str]) -> tuple[float, float]:
    if not references:
        abstained = is_not_answerable(answer)
        return float(abstained), float(abstained)
    normalized = normalize_answer(extract_final_answer(answer))
    exact = max(float(normalized == normalize_answer(reference)) for reference in references)
    f1 = max(token_f1(extract_final_answer(answer), reference) for reference in references)
    return exact, f1


def summarize_generation_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["is_answerable"]]
    unanswerable = [row for row in rows if not row["is_answerable"]]
    return {
        "n": len(rows),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "exact_match": mean(float(row["exact_match"]) for row in rows),
        "token_f1": mean(float(row["token_f1"]) for row in rows),
        "answerable_exact_match": mean(float(row["exact_match"]) for row in answerable)
        if answerable
        else 0.0,
        "answerable_token_f1": mean(float(row["token_f1"]) for row in answerable)
        if answerable
        else 0.0,
        "unanswerable_accuracy": mean(float(row["predicted_unanswerable"]) for row in unanswerable)
        if unanswerable
        else 0.0,
        "false_answer_rate": mean(not row["predicted_unanswerable"] for row in unanswerable)
        if unanswerable
        else 0.0,
        "overall_abstention_rate": mean(bool(row["predicted_unanswerable"]) for row in rows),
    }


def evaluate_generation_records(
    generations: list[dict[str, Any]], prompts: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for generation in generations:
        qid = str(generation["qid"])
        if qid in seen:
            raise ValueError(f"Duplicate generation qid: {qid}")
        seen.add(qid)
        if qid not in prompts:
            raise ValueError(f"Generation qid missing from prompt pack: {qid}")
        prompt = prompts[qid]
        answer = extract_final_answer(str(generation.get("answer", "")))
        references = [str(value) for value in prompt.get("reference_answers", [])]
        exact, f1 = best_reference_score(answer, references)
        output.append(
            {
                "qid": qid,
                "patient_id": prompt["patient_id"],
                "report_id": prompt["report_id"],
                "question": prompt["question"],
                "is_answerable": bool(prompt["is_answerable"]),
                "reference_answers": references,
                "agent_action": prompt["agent_action"],
                "answer": answer,
                "predicted_unanswerable": is_not_answerable(answer),
                "exact_match": exact,
                "token_f1": f1,
                "retrieved_chunk_ids": prompt["retrieved_chunk_ids"],
                "relevant_chunk_ids": prompt["relevant_chunk_ids"],
            }
        )
    missing = sorted(set(prompts) - seen)
    if missing:
        raise ValueError(f"Prompt qids missing generations: {missing[:3]}")
    return output

