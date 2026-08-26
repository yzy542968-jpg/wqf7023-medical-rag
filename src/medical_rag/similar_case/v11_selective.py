"""Development-time retrieval confidence and selective history gate."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Sequence


@dataclass(frozen=True)
class RetrievalConfidence:
    top_score: float
    margin: float
    component_agreement: float
    evidence_coverage: float
    ensemble_dispersion: float
    confidence: float
    score_range: float = 0.0
    normalized_top_score: float = 0.0
    normalized_margin: float = 0.0


def _sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + exp(-value))


def compute_retrieval_confidence(
    scores: Sequence[float], *, component_agreement: float = 0.0,
    evidence_coverage: float = 0.0, ensemble_dispersion: float = 0.0,
) -> RetrievalConfidence:
    values = sorted((float(value) for value in scores), reverse=True)
    if not values:
        return RetrievalConfidence(0.0, 0.0, 0.0, 0.0, ensemble_dispersion, 0.0)
    top = values[0]
    second = values[1] if len(values) > 1 else 0.0
    margin = top - second
    score_range = max(values) - min(values)
    normalized_top = (top - min(values)) / score_range if score_range > 0.0 else 0.5
    normalized_margin = margin / score_range if score_range > 0.0 else 0.0
    # BM25 and fused retrieval scores are uncalibrated and can be very large.
    # Normalize within each ranked list before combining features; otherwise
    # the old sigmoid saturated near one for almost every query.
    raw = (
        1.6 * normalized_top
        + 3.0 * normalized_margin
        + 0.8 * float(component_agreement)
        + 0.8 * float(evidence_coverage)
        - 1.2 * float(ensemble_dispersion)
        - 1.4
    )
    return RetrievalConfidence(
        top, margin, float(component_agreement), float(evidence_coverage),
        float(ensemble_dispersion), _sigmoid(raw), score_range,
        normalized_top, normalized_margin,
    )


def selective_history_decision(confidence: RetrievalConfidence, *, threshold: float) -> dict[str, float | bool | str]:
    use_history = confidence.confidence >= float(threshold)
    return {
        "confidence": confidence.confidence,
        "threshold": float(threshold),
        "use_historical_evidence": use_history,
        "decision": "retrieve_and_ground" if use_history else "abstain_from_historical_evidence",
    }


def fit_proxy_threshold(confidences: Sequence[float], proxy_relevant: Sequence[bool], *, minimum_coverage: float = 0.80) -> dict[str, float | int | str]:
    if len(confidences) != len(proxy_relevant) or not confidences:
        raise ValueError("Confidence and proxy labels must be non-empty and aligned.")
    candidates = sorted({0.0, 1.0, *(float(value) for value in confidences)})
    feasible = []
    for threshold in candidates:
        accepted = [index for index, value in enumerate(confidences) if float(value) >= threshold]
        coverage = len(accepted) / len(confidences)
        if coverage < minimum_coverage:
            continue
        precision = sum(bool(proxy_relevant[index]) for index in accepted) / len(accepted) if accepted else 0.0
        feasible.append((precision, coverage, threshold))
    if not feasible:
        feasible = [(0.0, 0.0, 1.0)]
    precision, coverage, threshold = max(feasible, key=lambda row: (row[0], -row[2]))
    return {
        "threshold": float(threshold), "proxy_precision": float(precision),
        "coverage": float(coverage), "minimum_coverage": float(minimum_coverage),
        "selection_basis": "development_proxy_relevance_only",
    }


def risk_coverage_curve(confidences: Sequence[float], proxy_relevant: Sequence[bool], points: int = 11) -> list[dict[str, float]]:
    if len(confidences) != len(proxy_relevant):
        raise ValueError("Confidence and proxy labels must be aligned.")
    order = sorted(range(len(confidences)), key=lambda index: (-float(confidences[index]), index))
    result = []
    for step in range(1, points + 1):
        count = max(1, round(len(order) * step / points))
        chosen = order[:count]
        result.append({
            "coverage": count / len(order),
            "proxy_precision": sum(bool(proxy_relevant[index]) for index in chosen) / count,
        })
    return result


__all__ = ["RetrievalConfidence", "compute_retrieval_confidence", "fit_proxy_threshold", "risk_coverage_curve", "selective_history_decision"]
