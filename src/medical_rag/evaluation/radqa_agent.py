from __future__ import annotations

from statistics import mean
from typing import Any

from medical_rag.evaluation.metrics import evaluate_retrieval
from medical_rag.retrieval.scoped_chunk_retriever import ScopedBM25ChunkRetriever


SYSTEMS = ("report_scoped_bm25", "patient_scoped_bm25", "global_bm25")


def evaluate_retrieval_system(
    questions: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    retriever: ScopedBM25ChunkRetriever,
    system: str,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reports_by_patient: dict[str, set[str]] = {}
    for chunk in chunks:
        reports_by_patient.setdefault(str(chunk["patient_id"]), set()).add(
            str(chunk["report_id"])
        )

    rows: list[dict[str, Any]] = []
    qrels: dict[str, set[str]] = {}
    rankings: dict[str, list[str]] = {}
    for question in questions:
        search_kwargs: dict[str, Any] = {}
        if system == "report_scoped_bm25":
            search_kwargs["case_id"] = question["report_id"]
        elif system == "patient_scoped_bm25":
            search_kwargs["allowed_case_ids"] = reports_by_patient[question["patient_id"]]
        elif system != "global_bm25":
            raise ValueError(f"Unknown RadQA retrieval system: {system}")
        results = retriever.search(question["question"], top_k=top_k, **search_kwargs)
        retrieved_ids = [row["chunk_id"] for row in results]
        if question["is_answerable"]:
            qrels[question["qid"]] = set(question["relevant_chunk_ids"])
            rankings[question["qid"]] = retrieved_ids
        scores = [float(row["score"]) for row in results]
        rows.append(
            {
                "qid": question["qid"],
                "split": question["split"],
                "patient_id": question["patient_id"],
                "report_id": question["report_id"],
                "question": question["question"],
                "is_answerable": question["is_answerable"],
                "reference_answers": question["reference_answers"],
                "relevant_chunk_ids": question["relevant_chunk_ids"],
                "system": system,
                "retrieved_chunk_ids": retrieved_ids,
                "retrieved_report_ids": [row["report_id"] for row in results],
                "retrieved_sections": [row["section"] for row in results],
                "retrieved_texts": [row["text"] for row in results],
                "scores": scores,
                "top1_score": scores[0] if scores else 0.0,
                "top1_margin": scores[0] - scores[1] if len(scores) > 1 else 0.0,
            }
        )

    metrics = evaluate_retrieval(qrels, rankings, k_values=(1, 3, 5))
    metrics.update(
        {
            "answerable_question_count": len(qrels),
            "all_question_count": len(questions),
            "top1_zero_score_rate": mean(row["top1_score"] == 0.0 for row in rows),
        }
    )
    return metrics, rows


def answerability_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    truth = [bool(row["is_answerable"]) for row in rows]
    predictions = [float(row["top1_score"]) >= threshold for row in rows]
    tp = sum(actual and predicted for actual, predicted in zip(truth, predictions))
    tn = sum(not actual and not predicted for actual, predicted in zip(truth, predictions))
    fp = sum(not actual and predicted for actual, predicted in zip(truth, predictions))
    fn = sum(actual and not predicted for actual, predicted in zip(truth, predictions))

    def safe_div(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    positive_precision = safe_div(tp, tp + fp)
    positive_recall = safe_div(tp, tp + fn)
    positive_f1 = safe_div(
        2 * positive_precision * positive_recall, positive_precision + positive_recall
    )
    negative_precision = safe_div(tn, tn + fn)
    negative_recall = safe_div(tn, tn + fp)
    negative_f1 = safe_div(
        2 * negative_precision * negative_recall, negative_precision + negative_recall
    )
    return {
        "threshold": threshold,
        "n": len(rows),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": safe_div(tp + tn, len(rows)),
        "balanced_accuracy": (positive_recall + negative_recall) / 2,
        "answerable_precision": positive_precision,
        "answerable_recall": positive_recall,
        "answerable_f1": positive_f1,
        "unanswerable_f1": negative_f1,
        "macro_f1": (positive_f1 + negative_f1) / 2,
        "false_answer_rate": safe_div(fp, fp + tn),
        "abstention_rate": mean(not value for value in predictions),
    }


def select_answerability_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = sorted({float(row["top1_score"]) for row in rows})
    if not scores:
        raise ValueError("Cannot calibrate answerability without retrieval scores.")
    epsilon = 1e-9
    candidates = [scores[0] - epsilon]
    candidates.extend((left + right) / 2 for left, right in zip(scores, scores[1:]))
    candidates.append(scores[-1] + epsilon)
    sweep = [answerability_metrics(rows, threshold) for threshold in candidates]
    selected = max(
        sweep,
        key=lambda row: (
            row["macro_f1"],
            row["balanced_accuracy"],
            -row["false_answer_rate"],
            row["threshold"],
        ),
    )
    return {
        "selection_rule": (
            "maximize development macro-F1, then balanced accuracy, minimize false-answer "
            "rate, then choose the higher threshold"
        ),
        "selected": selected,
        "candidate_count": len(sweep),
        "sweep": sweep,
    }


def build_agent_prompt(
    question: dict[str, Any], retrieval_row: dict[str, Any], threshold: float
) -> dict[str, Any]:
    predicted_answerable = float(retrieval_row["top1_score"]) >= threshold
    evidence_lines = [
        f"[{section} {index}] {text}"
        for index, (section, text) in enumerate(
            zip(retrieval_row["retrieved_sections"], retrieval_row["retrieved_texts"]),
            start=1,
        )
    ]
    action = "ANSWER_FROM_EVIDENCE" if predicted_answerable else "ABSTAIN_LOW_EVIDENCE"
    instruction = (
        "Answer the question using only the scoped report evidence. If the evidence does not "
        "answer the question, return exactly: NOT ANSWERABLE. Do not infer absent findings."
    )
    if not predicted_answerable:
        instruction += " The calibrated retrieval action is abstention; return NOT ANSWERABLE."
    prompt = (
        f"{instruction}\n"
        f"Report scope: {question['report_id']}\n"
        f"Agent action: {action}\n"
        f"Question: {question['question']}\n\n"
        "Retrieved evidence:\n"
        + "\n".join(evidence_lines)
        + "\n\nFinal answer:"
    )
    references = question["reference_answers"]
    return {
        "qid": question["qid"],
        "case_id": question["report_id"],
        "patient_id": question["patient_id"],
        "report_id": question["report_id"],
        "question_type": "radqa_natural_question",
        "question": question["question"],
        "is_answerable": question["is_answerable"],
        "reference_answer": references[0] if references else "NOT ANSWERABLE",
        "reference_answers": references,
        "system": "v3_radqa_report_scoped_evidence_agent",
        "prompt_mode": "natural_question_evidence_abstention",
        "retriever": retrieval_row["system"],
        "agent_action": action,
        "answerability_threshold": threshold,
        "top1_score": retrieval_row["top1_score"],
        "retrieved_chunk_ids": retrieval_row["retrieved_chunk_ids"],
        "retrieved_case_ids": retrieval_row["retrieved_report_ids"],
        "relevant_case_ids": [question["report_id"]],
        "relevant_chunk_ids": question["relevant_chunk_ids"],
        "retrieved_context": "\n".join(retrieval_row["retrieved_texts"]),
        "prompt": prompt,
    }

