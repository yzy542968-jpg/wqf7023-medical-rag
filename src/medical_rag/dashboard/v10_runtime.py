"""Interactive V10 cluster-disjoint similar-case retrieval helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from medical_rag.similar_case.v10_calibration import predict_from_payload
from medical_rag.similar_case.v10_evidence import (
    EvidenceUnit,
    evidence_diagnostics,
    select_case_evidence,
)
from medical_rag.similar_case.v10_loader import V10RuntimeAssets, load_v10_runtime_assets
from medical_rag.similar_case.v10_multiview import attention_query_embedding, l2_normalize
from medical_rag.similar_case.v10_runtime import component_agreement
from medical_rag.similar_case.v10_split import file_sha256


def infer_question_type(question: str) -> str:
    lowered = " ".join(str(question).lower().split())
    if any(term in lowered for term in ("acute", "cardiopulmonary abnormality")):
        return "acute"
    if any(term in lowered for term in ("impression", "diagnosis", "conclusion", "most likely")):
        return "impression"
    return "findings"


@dataclass
class V10DashboardResources:
    assets: V10RuntimeAssets
    calibrator: dict[str, Any]
    confidence_threshold: float
    evidence_policy: str


def load_v10_dashboard_resources(
    *,
    cases_path: Path,
    radgraph_path: Path,
    split_path: Path,
    embeddings_path: Path,
    r5_checkpoint_dir: Path,
    attention_checkpoint_dir: Path,
    calibrator_path: Path,
    confirmation_config_path: Path,
) -> V10DashboardResources:
    config = json.loads(confirmation_config_path.read_text(encoding="utf-8"))
    paths = {
        "cases": cases_path,
        "radgraph": radgraph_path,
        "split": split_path,
        "embeddings": embeddings_path,
        "calibrator": calibrator_path,
    }
    for name, path in paths.items():
        observed = file_sha256(path)
        expected = str(config["frozen_input_sha256"][name])
        if observed != expected:
            raise RuntimeError(f"V10 dashboard input changed for {name}")
    calibrator = json.loads(calibrator_path.read_text(encoding="utf-8"))
    coverage = str(config["selective_coverage_operating_point"])
    return V10DashboardResources(
        assets=load_v10_runtime_assets(
            cases_path=cases_path,
            radgraph_path=radgraph_path,
            split_path=split_path,
            embeddings_path=embeddings_path,
            r5_checkpoint_dir=r5_checkpoint_dir,
            attention_checkpoint_dir=attention_checkpoint_dir,
        ),
        calibrator=calibrator,
        confidence_threshold=float(calibrator["coverage_thresholds"][coverage]),
        evidence_policy=str(config["evidence_policy"]),
    )


def calibration_features(
    result: Mapping[str, Any],
    evidence: list[EvidenceUnit],
    *,
    question_type: str,
    view_count: int,
) -> dict[str, float]:
    ranking = np.asarray(result["ranking"], dtype=np.int64)
    top1, top2 = int(ranking[0]), int(ranking[1])
    diagnostics = evidence_diagnostics(evidence)
    return {
        "top1_score": float(result["ensemble_scores"][top1]),
        "top1_top2_margin": float(
            result["ensemble_scores"][top1] - result["ensemble_scores"][top2]
        ),
        "component_agreement": component_agreement(result, top1),
        "ensemble_variance": float(np.var(result["seed_scores"][:, top1])),
        "evidence_score": float(diagnostics["mean_score"]),
        "evidence_redundancy": float(diagnostics["redundancy"]),
        "view_count": float(view_count),
        "question_findings": float(question_type == "findings"),
        "question_impression": float(question_type == "impression"),
        "question_acute": float(question_type == "acute"),
    }


def retrieve_v10(
    *,
    indication: str,
    question: str,
    image_embeddings: np.ndarray,
    resources: V10DashboardResources,
    top_k: int = 3,
) -> dict[str, Any]:
    if not str(question).strip():
        raise ValueError("A medical question is required.")
    views = np.asarray(image_embeddings, dtype=np.float32)
    if views.ndim == 1:
        views = views[None, :]
    views = l2_normalize(views)
    query_image = attention_query_embedding(resources.assets.attention_models, views)
    question_type = infer_question_type(question)
    query_text = "\n".join(
        value for value in (str(indication).strip(), str(question).strip()) if value
    )
    runtime = resources.assets.runtime
    prepared = {
        "question": str(question).strip(),
        "query_text": query_text,
        "question_type": question_type,
        "bm25": np.asarray(runtime.bm25.score_all(query_text), dtype=np.float32),
        "fact_features": runtime.fact_index.query_features(query_text),
    }
    result = runtime.score_prepared(prepared, query_image)
    ranking = np.asarray(result["ranking"], dtype=np.int64)
    selected_ids = [runtime.candidate_ids[int(index)] for index in ranking[:top_k]]
    evidence: list[EvidenceUnit] = []
    for case_id in selected_ids:
        evidence.extend(
            select_case_evidence(
                resources.assets.raw_cases[case_id],
                query=query_text,
                facts=resources.assets.radgraph[case_id].facts,
                policy=resources.evidence_policy,
            )
        )
    features = calibration_features(
        result,
        evidence,
        question_type=question_type,
        view_count=len(views),
    )
    confidence = float(predict_from_payload([features], resources.calibrator)[0])
    no_reliable_history = confidence < resources.confidence_threshold
    rows = []
    for rank, index in enumerate(ranking[:top_k], start=1):
        case_id = runtime.candidate_ids[int(index)]
        case = resources.assets.raw_cases[case_id]
        rows.append(
            {
                "rank": rank,
                "case_id": case_id,
                "indication": str(case.get("indication", "")),
                "findings": str(case.get("findings", "")),
                "impression": str(case.get("impression", "")),
                "bm25_score": float(result["bm25"][index]),
                "image_image_similarity": float(result["image_image"][index]),
                "image_report_similarity": float(result["image_report"][index]),
                "r4_score": float(result["r4_scores"][index]),
                "r5_score": float(result["ensemble_scores"][index]),
                "r5_seed_variance": float(np.var(result["seed_scores"][:, index])),
            }
        )
    return {
        "question_type": question_type,
        "query_text": query_text,
        "retrieved_cases": rows,
        "evidence": [] if no_reliable_history else evidence,
        "retrieval_confidence": confidence,
        "confidence_threshold": resources.confidence_threshold,
        "no_reliable_history": no_reliable_history,
        "view_count": len(views),
        "candidate_bank_count": len(runtime.candidate_ids),
        "calibration_features": features,
    }


__all__ = [
    "V10DashboardResources",
    "calibration_features",
    "infer_question_type",
    "load_v10_dashboard_resources",
    "retrieve_v10",
]
