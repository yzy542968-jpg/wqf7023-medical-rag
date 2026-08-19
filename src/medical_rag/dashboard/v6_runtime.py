"""Runtime helpers for the V6 confirmation dashboard demonstration.

The dashboard uses the frozen V6 confirmation candidate pool and retrieval
policy, but it is an interactive demonstration rather than a new evaluation
run. Uploaded images are encoded at request time and are never written to the
repository or added to the frozen result artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from medical_rag.multimodal.fusion import minmax_normalize, rank_scores
from medical_rag.multimodal.v6_chunking import build_report_chunks
from medical_rag.retrieval.bm25_retriever import BM25Retriever
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


@dataclass(frozen=True)
class V6DashboardResources:
    """Validated, read-only resources used by one V6 dashboard session."""

    config: dict[str, Any]
    cases: dict[str, dict[str, Any]]
    candidate_ids: tuple[str, ...]
    target_ids: frozenset[str]
    distractor_ids: frozenset[str]
    bm25: BM25Retriever
    chunk_embeddings: np.ndarray
    chunk_case_ids: tuple[str, ...]


def load_v6_resources(
    *,
    config_path: Path,
    cohort_path: Path,
    cases_path: Path,
    medsiglip_cache_path: Path,
) -> V6DashboardResources:
    """Load and validate only artifacts that are frozen for V6."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    candidate_ids = tuple(str(case_id) for case_id in cohort["case_ids"])
    if len(candidate_ids) != 240 or len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("V6 dashboard requires the frozen 240-case candidate pool.")

    all_cases = {str(case["case_id"]): case for case in load_cases_jsonl(cases_path)}
    missing = [case_id for case_id in candidate_ids if case_id not in all_cases]
    if missing:
        raise RuntimeError(f"Frozen V6 cases are missing from the source file: {missing[:5]}")
    cases = {case_id: all_cases[case_id] for case_id in candidate_ids}

    target_ids = frozenset(str(case_id) for case_id in cohort["target_case_ids"])
    distractor_ids = frozenset(str(case_id) for case_id in cohort["distractor_case_ids"])
    if target_ids | distractor_ids != set(candidate_ids) or target_ids & distractor_ids:
        raise RuntimeError("V6 target and distractor manifests do not partition the pool.")

    # The cohort JSON also contains the broader benchmark-construction chunk
    # manifest. The formal MedSigLIP runner uses the locked findings/impression
    # chunk builder, so reconstruct that exact 64-token mapping here instead of
    # silently pairing the cache with a different chunk list.
    try:
        from transformers import AutoProcessor

        encoder_config = config["multimodal_retrieval"]["primary_encoder"]
        processor = AutoProcessor.from_pretrained(
            str(encoder_config["model"]),
            revision=str(encoder_config["revision"]),
            cache_dir=str(config_path.resolve().parents[1] / ".hf_cache"),
            local_files_only=True,
            use_fast=False,
        )
    except Exception as exc:  # pragma: no cover - depends on local model cache
        raise RuntimeError(
            "V6 dashboard requires the locally cached MedSigLIP tokenizer."
        ) from exc
    chunk_rows = [
        chunk
        for case_id in candidate_ids
        for chunk in build_report_chunks(
            cases[case_id],
            processor.tokenizer,
            max_tokens=int(config["multimodal_retrieval"]["primary_encoder"]["max_text_tokens"]),
        )
    ]
    chunk_case_ids = tuple(str(row["case_id"]) for row in chunk_rows)
    if not chunk_case_ids or not set(chunk_case_ids).issubset(set(candidate_ids)):
        raise RuntimeError("V6 report chunk reconstruction does not match the candidate pool.")

    with np.load(medsiglip_cache_path, allow_pickle=False) as cache:
        chunk_embeddings = np.asarray(cache["chunks"], dtype=np.float32)
        cached_images = np.asarray(cache["images"], dtype=np.float32)
    if len(chunk_embeddings) != len(chunk_case_ids):
        raise RuntimeError(
            "V6 MedSigLIP chunk cache does not match the locked formal chunk builder."
        )
    if cached_images.shape[0] != len(candidate_ids):
        raise RuntimeError("V6 MedSigLIP image cache does not match the candidate pool size.")

    bm25 = BM25Retriever(k1=1.5, b=0.75).fit(
        [cases[case_id] for case_id in candidate_ids]
    )
    return V6DashboardResources(
        config=config,
        cases=cases,
        candidate_ids=candidate_ids,
        target_ids=target_ids,
        distractor_ids=distractor_ids,
        bm25=bm25,
        chunk_embeddings=chunk_embeddings,
        chunk_case_ids=chunk_case_ids,
    )


def encode_uploaded_image(payload: bytes, encoder: Any) -> np.ndarray:
    """Encode one uploaded RGB image with the frozen MedSigLIP image tower."""

    from PIL import Image

    if not payload:
        raise ValueError("The uploaded image is empty.")
    with Image.open(BytesIO(payload)) as image:
        rgb_image = image.convert("RGB")
        encoded = encoder.processor(images=[rgb_image], return_tensors="pt")
    pixel_values = encoded["pixel_values"].to(encoder.device, dtype=encoder.dtype)
    with encoder.torch.inference_mode():
        features = encoder.model.get_image_features(pixel_values=pixel_values)
    embedding = encoder._normalized_numpy(features)
    if embedding.shape != (1, 1152):
        raise RuntimeError(f"Unexpected MedSigLIP image embedding shape: {embedding.shape}")
    return embedding[0].astype(np.float32)


def build_v6_query(indication: str, question: str) -> str:
    indication = str(indication).strip()
    question = str(question).strip()
    if not question:
        raise ValueError("A report-grounded question is required.")
    return f"Clinical indication: {indication}\nQuestion: {question}"


def _image_case_scores(
    image_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunk_case_ids: Sequence[str],
    candidate_ids: Sequence[str],
) -> dict[str, float]:
    similarities = np.asarray(chunk_embeddings, dtype=np.float32) @ np.asarray(
        image_embedding, dtype=np.float32
    )
    scores = {str(case_id): float("-inf") for case_id in candidate_ids}
    for similarity, case_id in zip(similarities, chunk_case_ids, strict=True):
        scores[str(case_id)] = max(scores[str(case_id)], float(similarity))
    if any(not np.isfinite(value) for value in scores.values()):
        raise RuntimeError("Every V6 candidate must have a finite report chunk score.")
    return scores


def fuse_v6_shortlist(
    text_ranking: Sequence[str],
    text_scores: Sequence[float],
    image_scores: Mapping[str, float],
    *,
    shortlist_size: int = 100,
    text_weight: float = 0.5,
) -> tuple[list[str], dict[str, float], dict[str, float], dict[str, float]]:
    """Apply the frozen V6 independent-min-max 0.5/0.5 fusion policy."""

    if len(text_ranking) != len(text_scores):
        raise ValueError("Text ranking and score lengths must match.")
    if set(text_ranking) != set(image_scores):
        raise ValueError("Text and image scores must cover the same candidates.")
    shortlist = list(text_ranking[:shortlist_size])
    normalized_text = minmax_normalize(text_scores[:shortlist_size])
    normalized_image = minmax_normalize([image_scores[case_id] for case_id in shortlist])
    fused_values = text_weight * normalized_text + (1.0 - text_weight) * normalized_image
    fused_map = {case_id: float(score) for case_id, score in zip(shortlist, fused_values, strict=True)}
    text_map = {
        case_id: float(score)
        for case_id, score in zip(shortlist, normalized_text, strict=True)
    }
    image_map = {
        case_id: float(score)
        for case_id, score in zip(shortlist, normalized_image, strict=True)
    }
    ranked_shortlist = rank_scores(shortlist, fused_values.tolist())
    return ranked_shortlist + list(text_ranking[shortlist_size:]), fused_map, text_map, image_map


def retrieve_v6(
    indication: str,
    question: str,
    image_embedding: np.ndarray,
    resources: V6DashboardResources,
    *,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Retrieve top-ranked candidate reports using the frozen V6 policy."""

    query = build_v6_query(indication, question)
    text_rows = resources.bm25.search(query, top_k=len(resources.candidate_ids))
    text_ranking = [str(row["case_id"]) for row in text_rows]
    text_scores = [float(row["score"]) for row in text_rows]
    image_scores = _image_case_scores(
        image_embedding,
        resources.chunk_embeddings,
        resources.chunk_case_ids,
        resources.candidate_ids,
    )
    ranking, fused_scores, text_norm, image_norm = fuse_v6_shortlist(
        text_ranking,
        text_scores,
        image_scores,
        shortlist_size=int(resources.config["multimodal_retrieval"]["shortlist_size"]),
        text_weight=float(resources.config["multimodal_retrieval"]["text_weight"]),
    )
    text_score_map = {case_id: score for case_id, score in zip(text_ranking, text_scores, strict=True)}
    rows: list[dict[str, Any]] = []
    for rank, case_id in enumerate(ranking[: max(1, int(top_k))], start=1):
        case = resources.cases[case_id]
        rows.append(
            {
                "rank": rank,
                "case_id": case_id,
                "findings": str(case.get("findings", "")),
                "impression": str(case.get("impression", "")),
                "indication": str(case.get("indication", "")),
                "report_index_class": (
                    "report_indexed_normal"
                    if str(case.get("problems", "")).strip().lower() == "normal"
                    else "report_indexed_abnormal"
                ),
                "bm25_score": float(text_score_map[case_id]),
                "bm25_normalized": text_norm.get(case_id),
                "image_similarity": float(image_scores[case_id]),
                "image_normalized": image_norm.get(case_id),
                "fused_score": fused_scores.get(case_id),
            }
        )
    return rows


def build_v6_generation_prompt(indication: str, question: str, selected_case: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "Answer this report-grounded research question using only the selected candidate report.",
            "Do not infer patient identity, add outside clinical knowledge, or combine facts from other cases.",
            "If the report is insufficient, say so explicitly.",
            "",
            f"Clinical indication: {str(indication).strip()}",
            f"Question: {str(question).strip()}",
            "",
            f"Selected candidate case ID: {selected_case.get('case_id', '')}",
            f"Findings: {selected_case.get('findings', '')}",
            f"Impression: {selected_case.get('impression', '')}",
            "",
            "Return one concise answer paragraph.",
        ]
    )


def extractive_v6_answer(question: str, selected_case: Mapping[str, Any]) -> str:
    """Provide a transparent no-download answer for the dashboard demo."""

    lower = str(question).lower()
    if any(term in lower for term in ("impression", "conclusion", "summary", "principal")):
        return str(selected_case.get("impression", "")) or str(selected_case.get("findings", ""))
    return str(selected_case.get("findings", "")) or str(selected_case.get("impression", ""))
