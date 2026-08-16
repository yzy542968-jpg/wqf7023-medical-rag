from __future__ import annotations

from pathlib import Path

import pandas as pd


LETTERS = "abcd"
METRICS = (
    ("correctness_1_5", "Correctness", [1, 2, 3, 4, 5]),
    ("evidence_grounding_1_5", "Evidence grounding", [1, 2, 3, 4, 5]),
    ("potentially_harmful_0_1", "Potentially harmful", [0, 1]),
)
REQUIRED_COLUMNS = [
    *[
        f"{letter}_{metric}"
        for letter in LETTERS
        for metric, _, _ in METRICS
    ],
    "best_response_A_B_C_D_or_tie",
]


def completed_mask(frame: pd.DataFrame) -> pd.Series:
    return frame[REQUIRED_COLUMNS].notna().all(axis=1) & frame[REQUIRED_COLUMNS].apply(
        lambda row: all(str(value).strip() for value in row), axis=1
    )


def existing_index(value: object, options: list[object]) -> int | None:
    text = str(value).strip()
    if not text:
        return None
    for index, option in enumerate(options):
        if text == str(option) or text == f"{option}.0":
            return index
    return None


def save_evaluation(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)

