# P2 Evidence Checker Manual Calibration

The file `experiments/final_p2/evidence_calibration_50.csv` contains a stratified sample of 50 answer sentences. It is designed to select the final evidence threshold without manually reviewing all generated answers.

## What to Label

Open the CSV in Excel and fill only these columns:

1. `human_supported`: enter `1` if `matched_evidence` supports the factual claim in `answer_sentence`; otherwise enter `0`.
2. `human_negation_correct`: enter `1` if the automatic `negation_consistent` value is correct; otherwise enter `0`.
3. `notes`: optionally record the reason for difficult cases.

Judge support against the matched evidence, not against general medical knowledge. A sentence is unsupported when it adds a finding, diagnosis, location, severity, or polarity that the evidence does not state. Contradictory positive/negative findings must be labelled unsupported.

Do not edit the automatic columns or reorder/delete rows.

## Why These 50 Rows

The sample deliberately includes:

- explicit negation conflicts;
- scores where thresholds 0.40 and 0.65 disagree;
- high-confidence supported examples;
- low-confidence unsupported examples;
- balanced coverage of report BM25, case BM25, and case hybrid systems.

This is a calibration set, not the final answer-quality evaluation set. The existing comparative annotation sample is still needed for relevance, completeness, hallucination, and case-contamination ratings.

## Generate and Summarize

```powershell
& ".\.venv\Scripts\python.exe" "scripts\build_evidence_calibration_sample.py"
& ".\.venv\Scripts\python.exe" "scripts\summarize_evidence_calibration.py"
```

The second command writes `experiments/final_p2/evidence_calibration_metrics.json` and recommends the threshold with the best manually measured support F1. Precision, recall, F1, accuracy, and confusion counts are reported overall and by system.
