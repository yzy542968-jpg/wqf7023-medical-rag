from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

from medical_rag.agentic.semantic_evidence_checker import (
    NLIPredictor,
    check_semantic_evidence_support,
)


def historical_evidence_text(
    case_ids: Sequence[str], cases: Mapping[str, Mapping[str, Any]]
) -> str:
    parts = []
    for case_id in case_ids:
        case = cases[str(case_id)]
        parts.extend(
            [
                f"Historical case ID: {case_id}",
                f"Findings: {' '.join(str(case.get('findings', '')).split())}",
                f"Impression: {' '.join(str(case.get('impression', '')).split())}",
            ]
        )
    return "\n".join(parts)


def _matched_case_ids(
    checks: Sequence[Any], route_ids: Sequence[str], cases: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    matched = []
    for case_id in route_ids:
        case = cases[str(case_id)]
        report = " ".join(
            " ".join(str(case.get(field, "")).split()) for field in ("findings", "impression")
        ).lower()
        if any(
            check.supported
            and str(check.matched_evidence).strip()
            and str(check.matched_evidence).strip().lower() in report
            for check in checks
        ):
            matched.append(str(case_id))
    return matched


def run_bounded_historical_evidence_agent(
    *,
    answer_row: Mapping[str, Any],
    primary_case_ids: Sequence[str],
    retry_case_ids: Sequence[str],
    cases: Mapping[str, Mapping[str, Any]],
    predictor: NLIPredictor,
    minimum_support_rate: float,
    minimum_combined_support: float,
    entailment_threshold: float,
    contradiction_threshold: float,
) -> dict[str, Any]:
    claim = " ".join(str(answer_row.get("historical_support", "")).split())
    cited = [str(value) for value in answer_row.get("supporting_case_ids", [])]
    trace: list[dict[str, Any]] = [
        {
            "step": 1,
            "state": "INTAKE",
            "route": "r4_learned_mlp",
            "retrieved_case_ids": list(primary_case_ids),
            "historical_claim_present": bool(claim),
        }
    ]
    if not claim:
        trace.append(
            {
                "step": 2,
                "state": "COMPLETE",
                "decision": "no_historical_claim_to_verify",
            }
        )
        return {
            "agent_answer": str(answer_row.get("answer", "")),
            "agent_historical_support": "",
            "agent_supporting_case_ids": [],
            "agent_uncertainty": str(answer_row.get("uncertainty", "high")),
            "historical_claim_present": False,
            "initial_support_rate": 1.0,
            "final_support_rate": 1.0,
            "initial_unsupported": False,
            "final_unsupported": False,
            "retried": False,
            "historical_evidence_abstained": False,
            "historical_support_revised": False,
            "retrieval_calls": 1,
            "trace": trace,
        }

    def check(route_ids: Sequence[str]) -> Any:
        return check_semantic_evidence_support(
            claim,
            historical_evidence_text(route_ids, cases),
            predictor,
            min_combined_support=minimum_combined_support,
            entailment_threshold=entailment_threshold,
            contradiction_threshold=contradiction_threshold,
        )

    primary = check(primary_case_ids)
    primary_matches = _matched_case_ids(primary.sentence_checks, primary_case_ids, cases)
    cited_valid = bool(cited) and set(cited).issubset(set(primary_case_ids))
    primary_supported = primary.support_rate >= minimum_support_rate and (
        cited_valid or bool(primary_matches)
    )
    trace.append(
        {
            "step": 2,
            "state": "VERIFY",
            "route": "r4_learned_mlp",
            "support_rate": primary.support_rate,
            "cited_ids_valid": cited_valid,
            "matched_case_ids": primary_matches,
            "decision": "accept" if primary_supported else "retry",
        }
    )
    if primary_supported:
        final_ids = cited if cited_valid else primary_matches
        return {
            "agent_answer": str(answer_row.get("answer", "")),
            "agent_historical_support": claim,
            "agent_supporting_case_ids": final_ids,
            "agent_uncertainty": str(answer_row.get("uncertainty", "high")),
            "historical_claim_present": True,
            "initial_support_rate": primary.support_rate,
            "final_support_rate": primary.support_rate,
            "initial_unsupported": False,
            "final_unsupported": False,
            "retried": False,
            "historical_evidence_abstained": False,
            "historical_support_revised": final_ids != cited,
            "retrieval_calls": 1,
            "trace": trace,
        }

    retry = check(retry_case_ids)
    retry_matches = _matched_case_ids(retry.sentence_checks, retry_case_ids, cases)
    retry_supported = retry.support_rate >= minimum_support_rate and bool(retry_matches)
    trace.append(
        {
            "step": 3,
            "state": "RETRY_VERIFY",
            "route": "r1_image_image",
            "retrieved_case_ids": list(retry_case_ids),
            "support_rate": retry.support_rate,
            "matched_case_ids": retry_matches,
            "decision": "accept_retry_evidence" if retry_supported else "evidence_abstain",
        }
    )
    if retry_supported:
        final_claim = claim
        final_ids = retry_matches
        final_rate = retry.support_rate
        evidence_abstained = False
        uncertainty = str(answer_row.get("uncertainty", "high"))
    else:
        final_claim = ""
        final_ids = []
        final_rate = 1.0
        evidence_abstained = True
        uncertainty = "high"
    trace.append(
        {
            "step": 4,
            "state": "COMPLETE",
            "decision": "answer_preserved_historical_evidence_controlled",
            "target_image_answer_verified_by_agent": False,
        }
    )
    return {
        "agent_answer": str(answer_row.get("answer", "")),
        "agent_historical_support": final_claim,
        "agent_supporting_case_ids": final_ids,
        "agent_uncertainty": uncertainty,
        "historical_claim_present": True,
        "initial_support_rate": primary.support_rate,
        "final_support_rate": final_rate,
        "initial_unsupported": True,
        "final_unsupported": False,
        "retried": True,
        "historical_evidence_abstained": evidence_abstained,
        "historical_support_revised": final_claim != claim or final_ids != cited,
        "retrieval_calls": 2,
        "initial_sentence_checks": [asdict(value) for value in primary.sentence_checks],
        "retry_sentence_checks": [asdict(value) for value in retry.sentence_checks],
        "trace": trace,
    }
