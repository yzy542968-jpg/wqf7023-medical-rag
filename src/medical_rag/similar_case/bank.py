from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from medical_rag.similar_case.schema import PairedCase


@dataclass(frozen=True)
class CandidateBankAudit:
    query_study_id: str
    query_patient_id: str | None
    source_candidate_count: int
    excluded_same_study_count: int
    excluded_same_patient_count: int
    eligible_candidate_count: int
    post_filter_same_study_count: int
    post_filter_same_patient_count: int | None
    patient_level_exclusion_verified: bool
    patient_identity_basis: str
    identifier_verified_patient_level_exclusion: bool
    source_design_patient_separation_supported: bool


def _patient_identity_basis(
    query: PairedCase, source_cases: Sequence[PairedCase]
) -> str:
    cases = (query, *source_cases)
    bases = {
        str(case.metadata.get("patient_key_basis", "provided_patient_identifier"))
        for case in cases
        if case.patient_id is not None
    }
    if not bases:
        return "unavailable"
    if len(bases) > 1:
        return "mixed"
    return bases.pop()


def _assert_unique_studies(cases: Sequence[PairedCase]) -> None:
    study_ids = [case.study_id for case in cases]
    if len(study_ids) != len(set(study_ids)):
        raise ValueError("Candidate source contains duplicate study IDs.")


def build_candidate_bank(
    query: PairedCase,
    source_cases: Sequence[PairedCase],
    *,
    require_patient_ids: bool = True,
) -> tuple[list[PairedCase], CandidateBankAudit]:
    """Exclude the target study and target patient from a historical bank."""

    _assert_unique_studies(source_cases)
    if require_patient_ids:
        if query.patient_id is None:
            raise ValueError("The query patient ID is required for patient-level exclusion.")
        missing = [case.study_id for case in source_cases if case.patient_id is None]
        if missing:
            raise ValueError(
                "Patient IDs are required for every source case; missing for "
                f"{len(missing)} studies."
            )

    eligible: list[PairedCase] = []
    same_study_count = 0
    same_patient_count = 0
    for candidate in source_cases:
        if candidate.study_id == query.study_id:
            same_study_count += 1
            continue
        if (
            query.patient_id is not None
            and candidate.patient_id is not None
            and candidate.patient_id == query.patient_id
        ):
            same_patient_count += 1
            continue
        eligible.append(candidate)

    post_same_study = sum(case.study_id == query.study_id for case in eligible)
    if query.patient_id is None:
        post_same_patient: int | None = None
    else:
        post_same_patient = sum(
            case.patient_id == query.patient_id
            for case in eligible
            if case.patient_id is not None
        )

    patient_identity_basis = _patient_identity_basis(query, source_cases)
    complete_patient_keys = query.patient_id is not None and all(
        case.patient_id is not None for case in source_cases
    )
    no_same_patient_after_filter = post_same_patient == 0
    identifier_verified = (
        complete_patient_keys
        and no_same_patient_after_filter
        and patient_identity_basis != "source_design_one_study_per_patient"
    )
    source_design_supported = (
        complete_patient_keys
        and no_same_patient_after_filter
        and patient_identity_basis == "source_design_one_study_per_patient"
    )
    audit = CandidateBankAudit(
        query_study_id=query.study_id,
        query_patient_id=query.patient_id,
        source_candidate_count=len(source_cases),
        excluded_same_study_count=same_study_count,
        excluded_same_patient_count=same_patient_count,
        eligible_candidate_count=len(eligible),
        post_filter_same_study_count=post_same_study,
        post_filter_same_patient_count=post_same_patient,
        # Retained for compatibility; this now means identifier-verified rather
        # than separation inferred from a source-design surrogate.
        patient_level_exclusion_verified=identifier_verified,
        patient_identity_basis=patient_identity_basis,
        identifier_verified_patient_level_exclusion=identifier_verified,
        source_design_patient_separation_supported=source_design_supported,
    )
    if audit.post_filter_same_study_count:
        raise AssertionError("Target study remained in the candidate bank.")
    if require_patient_ids and audit.post_filter_same_patient_count:
        raise AssertionError("Target-patient study remained in the candidate bank.")
    return eligible, audit
