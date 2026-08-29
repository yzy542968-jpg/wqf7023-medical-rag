"""Dataset adapters and evaluation helpers for the final QA study."""

from .radrestruct import (
    RadReStructCase,
    RadReStructQuestion,
    canonical_openi_case_id,
    iter_radrestruct_cases,
)

__all__ = [
    "RadReStructCase",
    "RadReStructQuestion",
    "canonical_openi_case_id",
    "iter_radrestruct_cases",
]
