"""Multimodal image-report encoding and retrieval utilities."""

from medical_rag.multimodal.fusion import (
    aggregate_view_embeddings,
    rank_scores,
    reciprocal_rank_fusion,
    select_text_weight,
    shortlist_score_fusion,
)
from medical_rag.multimodal.evaluation import (
    aggregate_case_images,
    build_report_embedding_text,
    build_text_query,
    cosine_ranking,
    evaluate_confirmation_gate,
    evaluate_rankings_and_answers,
)
from medical_rag.multimodal.openi_images import official_filename_candidates, resolve_official_image

__all__ = [
    "aggregate_view_embeddings",
    "rank_scores",
    "reciprocal_rank_fusion",
    "select_text_weight",
    "shortlist_score_fusion",
    "aggregate_case_images",
    "build_report_embedding_text",
    "build_text_query",
    "cosine_ranking",
    "evaluate_confirmation_gate",
    "evaluate_rankings_and_answers",
    "official_filename_candidates",
    "resolve_official_image",
]
