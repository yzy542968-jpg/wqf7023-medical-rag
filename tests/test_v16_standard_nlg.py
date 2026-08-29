from __future__ import annotations

from scripts.evaluate_v16_standard_nlg import lexical_scores, paired_case_bootstrap


def test_identical_text_has_strong_lexical_scores() -> None:
    corpus, rows = lexical_scores(
        ["no focal airspace disease", "mild bibasilar atelectasis"],
        ["no focal airspace disease", "mild bibasilar atelectasis"],
    )
    assert corpus["bleu_1"] > 0.99
    assert corpus["rouge_l"] == 1.0
    assert rows[0]["meteor"] > 0.99


def test_case_grouped_bootstrap_preserves_pairing() -> None:
    keys = [
        ("CXR1", "findings", "retrieved_history"),
        ("CXR1", "impression", "retrieved_history"),
        ("CXR2", "findings", "retrieved_history"),
        ("CXR2", "impression", "retrieved_history"),
    ]
    left = {key: {"bleu_1": 1.0} for key in keys}
    right = {key: {"bleu_1": 0.0} for key in keys}
    result = paired_case_bootstrap(
        left,
        right,
        "bleu_1",
        "retrieved_history",
        iterations=100,
        seed=7,
    )
    assert result["case_count"] == 2
    assert result["mean_difference"] == 1.0
    assert result["ci_95_low"] == 1.0
