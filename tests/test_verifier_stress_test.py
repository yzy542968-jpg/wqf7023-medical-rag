from __future__ import annotations

from scripts.run_verifier_polarity_stress_test import flip_explicit_polarity


def test_flip_there_is_no() -> None:
    assert flip_explicit_polarity("There is no pleural effusion.") == "There is pleural effusion."


def test_flip_sentence_initial_no() -> None:
    assert flip_explicit_polarity("No focal consolidation.") == "Focal consolidation."


def test_non_polar_sentence_is_not_modified() -> None:
    assert flip_explicit_polarity("Heart size is normal.") is None
