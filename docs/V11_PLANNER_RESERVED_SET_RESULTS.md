# V11 Reserved Question-Planner Wording Results

## Frozen evaluation

The second wording set was committed in `919d70f` and evaluated once without
changing the planner. It contains 96 researcher-authored questions, 12 for
each of eight operational intents, with a distracting indication attached to
every non-empty question.

| Measure | Result |
|---|---:|
| Questions | 96 |
| Accuracy | 0.9167 |
| Macro-F1 | 0.9196 |
| Indication-invariance rate | 1.0000 |
| Predictions changed by indication | 0 |

Device, location and severity achieved perfect recall. Comparison, insufficient
information, presence and uncertainty each had two or fewer errors. Six errors
fell back to the default summary intent; two uncertainty questions were routed
to presence or severity. These errors were retained and the planner was not
modified after evaluation.

This result is stronger wording-robustness evidence than the original 64-item
development set, but it remains researcher-authored and non-clinical. It does
not establish independently validated natural-language understanding or
clinician-authored intent coverage.

Machine-readable artifacts:

- `data/splits/v11/v11_question_planner_reserved_benchmark.json`
- `data/splits/v11/v11_question_planner_reserved_summary.json`
- `docs/V11_PLANNER_RESERVED_SET_PROTOCOL.md`
