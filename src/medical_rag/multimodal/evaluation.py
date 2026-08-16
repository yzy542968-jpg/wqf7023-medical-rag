from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from medical_rag.evaluation.answer_metrics import token_f1
from medical_rag.evaluation.metrics import evaluate_retrieval
from medical_rag.multimodal.fusion import aggregate_view_embeddings, rank_scores


def build_text_query(case: Mapping[str, Any], question: Mapping[str, Any]) -> str:
    return f"Clinical indication: {case.get('indication', '')}\nQuestion: {question['question']}"


def build_report_embedding_text(case: Mapping[str, Any]) -> str:
    return f"Findings: {case.get('findings', '')}\nImpression: {case.get('impression', '')}"


def aggregate_case_images(
    view_embeddings: np.ndarray,
    view_case_ids: Sequence[str],
    case_ids: Sequence[str],
) -> np.ndarray:
    if len(view_embeddings) != len(view_case_ids):
        raise ValueError("Each view embedding must have one case ID.")
    grouped: dict[str, list[np.ndarray]] = {case_id: [] for case_id in case_ids}
    for embedding, case_id in zip(view_embeddings, view_case_ids, strict=True):
        if case_id in grouped:
            grouped[case_id].append(embedding)
    missing = [case_id for case_id, values in grouped.items() if not values]
    if missing:
        raise ValueError(f"Cases without image embeddings: {missing[:5]}")
    return np.stack([aggregate_view_embeddings(grouped[case_id]) for case_id in case_ids])


def cosine_ranking(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_case_ids: Sequence[str],
) -> list[str]:
    if len(candidate_embeddings) != len(candidate_case_ids):
        raise ValueError("Candidate embeddings and IDs must have equal length.")
    scores = np.asarray(candidate_embeddings) @ np.asarray(query_embedding)
    return rank_scores(candidate_case_ids, scores.tolist())


def routed_extractive_answer(case: Mapping[str, Any], question: Mapping[str, Any]) -> str:
    source = str(question.get("answer_source", "")).lower()
    if source == "findings":
        return str(case.get("findings", ""))
    if source == "impression":
        return str(case.get("impression", ""))
    return str(case.get("report_text", ""))


def evaluate_rankings_and_answers(
    questions: Sequence[Mapping[str, Any]],
    rankings: Mapping[str, Sequence[str]],
    cases: Mapping[str, Mapping[str, Any]],
    k_values: tuple[int, ...] = (1, 5, 10),
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    qrels = {str(row["qid"]): {str(row["case_id"])} for row in questions}
    retrieval = evaluate_retrieval(
        qrels,
        {qid: list(ranking) for qid, ranking in rankings.items()},
        k_values=k_values,
    )
    rows = []
    for question in questions:
        qid = str(question["qid"])
        ranking = list(rankings[qid])
        selected_case_id = ranking[0]
        prediction = routed_extractive_answer(cases[selected_case_id], question)
        score = token_f1(prediction, str(question["reference_answer"]))
        rows.append(
            {
                "qid": qid,
                "case_id": str(question["case_id"]),
                "selected_case_id": selected_case_id,
                "top_10_case_ids": ranking[:10],
                "prediction": prediction,
                "reference_answer": str(question["reference_answer"]),
                "token_f1": score,
            }
        )
    retrieval["token_f1"] = float(np.mean([row["token_f1"] for row in rows])) if rows else 0.0
    return retrieval, rows


def evaluate_confirmation_gate(
    config: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, float]],
    selected_text_weight: float,
) -> dict[str, Any]:
    gate = config["confirmation_gate"]
    image_mrr = metrics["image_only_biovil_t"]["mrr"]
    fusion_mrr = metrics["paired_biovil_t_rrf"]["mrr"]
    report_mrr = metrics["report_only_bm25"]["mrr"]
    checks = {
        "image_mrr_exceeds_v4": image_mrr
        > float(gate["biovil_t_image_mrr_must_exceed_v4_biomedclip_mrr"]),
        "fusion_mrr_exceeds_report_only": fusion_mrr > report_mrr,
        "selected_text_weight_below_limit": selected_text_weight
        < float(gate["selected_text_weight_must_be_less_than"]),
    }
    return {"passed": all(checks.values()), "checks": checks}
