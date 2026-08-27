from __future__ import annotations

from medical_rag.similar_case.random_history_control import (
    random_history_order_key,
    select_random_history,
)


def test_random_history_selection_is_deterministic_and_excludes_formal_rows() -> None:
    candidates = [f"CXR{index}" for index in range(20)]
    first = select_random_history(
        candidates,
        seed=7131,
        assignment=0,
        target_case_id="CXR99",
        question_type="Findings",
        excluded_case_ids={"CXR1", "CXR2", "CXR3"},
    )
    second = select_random_history(
        reversed(candidates),
        seed=7131,
        assignment=0,
        target_case_id="CXR99",
        question_type="findings",
        excluded_case_ids={"CXR3", "CXR2", "CXR1"},
    )
    assert first == second
    assert len(first) == len(set(first)) == 3
    assert not set(first) & {"CXR1", "CXR2", "CXR3", "CXR99"}


def test_assignment_domain_changes_the_order() -> None:
    candidates = [f"CXR{index}" for index in range(100)]
    zero = select_random_history(
        candidates,
        seed=7131,
        assignment=0,
        target_case_id="TARGET",
        question_type="impression",
    )
    one = select_random_history(
        candidates,
        seed=7131,
        assignment=1,
        target_case_id="TARGET",
        question_type="impression",
    )
    assert zero != one
    assert random_history_order_key(
        seed=7131,
        assignment=0,
        target_case_id="TARGET",
        question_type="impression",
        candidate_case_id="CXR1",
    ) != random_history_order_key(
        seed=7131,
        assignment=1,
        target_case_id="TARGET",
        question_type="impression",
        candidate_case_id="CXR1",
    )

