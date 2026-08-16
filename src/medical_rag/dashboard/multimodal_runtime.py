from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from medical_rag.agentic.evidence_checker import check_evidence_support
from medical_rag.agentic.planner import plan_question
from medical_rag.multimodal.fusion import minmax_normalize, shortlist_score_fusion
from medical_rag.retrieval.bm25_retriever import BM25Retriever


def encode_uploaded_image(payload: bytes, encoder: Any) -> np.ndarray:
    from health_multimodal.image.data.io import remap_to_uint8

    with Image.open(BytesIO(payload)) as source:
        grayscale = np.asarray(source.convert("L"))
    image = Image.fromarray(remap_to_uint8(grayscale)).convert("L")
    tensor = encoder.image_engine.transform(image).unsqueeze(0).to(encoder.device)
    with encoder.torch.inference_mode():
        embedding = encoder.image_engine.model(tensor).projected_global_embedding
    return encoder._normalized_numpy(embedding)[0]


def paired_shortlist_retrieve(
    *,
    question: str,
    indication: str,
    candidate_ids: Sequence[str],
    cases: Mapping[str, Mapping[str, Any]],
    bm25: BM25Retriever,
    image_embedding: np.ndarray,
    report_embeddings: np.ndarray,
    shortlist_size: int = 100,
    text_weight: float = 0.5,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    query = f"Clinical indication: {indication}\nQuestion: {question}"
    text_rows = bm25.search(query, top_k=len(candidate_ids))
    text_ranking = [str(row["case_id"]) for row in text_rows]
    text_scores = [float(row["score"]) for row in text_rows]
    similarities = np.asarray(report_embeddings) @ np.asarray(image_embedding)
    image_scores = {
        case_id: float(similarities[index]) for index, case_id in enumerate(candidate_ids)
    }
    ranking = shortlist_score_fusion(
        text_ranking,
        text_scores,
        image_scores,
        shortlist_size=shortlist_size,
        text_weight=text_weight,
    )

    shortlist = text_ranking[:shortlist_size]
    normalized_text = minmax_normalize(text_scores[:shortlist_size])
    normalized_image = minmax_normalize([image_scores[case_id] for case_id in shortlist])
    fused_scores = {
        case_id: float(text_weight * text_score + (1.0 - text_weight) * image_score)
        for case_id, text_score, image_score in zip(
            shortlist, normalized_text, normalized_image, strict=True
        )
    }
    text_by_id = {str(row["case_id"]): row for row in text_rows}
    results = []
    for rank, case_id in enumerate(ranking[:top_k], start=1):
        case = cases[case_id]
        results.append(
            {
                "rank": rank,
                "case_id": case_id,
                "selected": rank == 1,
                "fused_score": fused_scores.get(case_id, 0.0),
                "bm25_score": float(text_by_id[case_id]["score"]),
                "image_similarity": image_scores[case_id],
                "findings": str(case.get("findings", "")),
                "impression": str(case.get("impression", "")),
                "images": list(case.get("images", [])),
            }
        )
    return results


def answer_with_evidence_agent(
    question: str,
    selected_case: Mapping[str, Any],
    support_threshold: float = 0.65,
) -> dict[str, Any]:
    plan = plan_question(question)
    if plan.answer_field == "findings":
        draft = str(selected_case.get("findings", ""))
    elif plan.answer_field == "impression":
        draft = str(selected_case.get("impression", ""))
    else:
        draft = str(selected_case.get("impression") or selected_case.get("findings", ""))
    evidence = "\n".join(
        [
            f"Findings: {selected_case.get('findings', '')}",
            f"Impression: {selected_case.get('impression', '')}",
        ]
    )
    checked = check_evidence_support(draft, evidence, min_sentence_support=support_threshold)
    return {
        "plan": asdict(plan),
        "draft_answer": draft,
        "final_answer": checked.revised_answer,
        "support_rate": checked.support_rate,
        "abstained": checked.abstained,
        "sentence_checks": [asdict(row) for row in checked.sentence_checks],
    }
