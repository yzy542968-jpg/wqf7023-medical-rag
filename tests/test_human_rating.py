from __future__ import annotations

import pandas as pd

from medical_rag.evaluation.human_rating import (
    REQUIRED_COLUMNS,
    completed_mask,
    existing_index,
)


def test_human_evaluation_progress_requires_every_field() -> None:
    complete = {field: "1" for field in REQUIRED_COLUMNS}
    incomplete = dict(complete)
    incomplete[REQUIRED_COLUMNS[-1]] = ""
    mask = completed_mask(pd.DataFrame([complete, incomplete]))
    assert mask.tolist() == [True, False]
    assert existing_index("2.0", [1, 2, 3]) == 1
    assert existing_index("", [1, 2, 3]) is None
