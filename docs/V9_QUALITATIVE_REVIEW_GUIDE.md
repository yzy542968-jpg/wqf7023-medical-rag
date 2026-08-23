# V9 Qualitative Review Guide

## What is ready

The deterministic 24-case pack is available locally at:

```text
experiments/post_submission_v9/v9_qualitative_review_pack.jsonl
```

The public audit index is
`data/splits/v9/v9_qualitative_case_index.csv`. All rows are currently
`pending_researcher_review`. Assistant proposals are suggestions, not final
researcher labels.

## Review procedure

For each case, inspect both question rows and compare:

1. target image and clinical indication;
2. frozen findings/impression reference;
3. G0, G1, G2, and G3 answers;
4. R4 and backup R1 historical reports;
5. G4 support decision and trace;
6. the proposed labels and metric note.

Then populate `researcher_reviewed_labels_v1_0`, set `review_status` to
`confirmed`, `refined`, or `excluded`, add a short reason, initials, and date.
Do not call a case clinically correct unless that judgment is supported by an
appropriately qualified clinical reviewer; use reference consistency and
visible evidence language instead.

## Selected cases

| Category | Cases |
|---|---|
| Largest G3 gains | CXR869, CXR980, CXR2292, CXR1601, CXR3199, CXR2693 |
| Largest G3 losses | CXR2181, CXR2441, CXR3880, CXR227, CXR1744, CXR2658 |
| Agent retry recovered | CXR1550, CXR1868, CXR2634, CXR3662, CXR796, CXR135 |
| Historical-evidence abstention | CXR102, CXR1042, CXR1046, CXR1060, CXR1097, CXR114 |

Assistant-proposal counts in this predefined set are:

```text
retrieval_relevance_gain                 15
retrieval_relevance_failure               9
reference_consistent_answer               2
reference_inconsistent_answer            13
structured_output_failure                10
historical_support_retry_recovered        5
historical_support_abstained             13
citation_repaired                         5
```

These counts describe only the selected review set and must not be used as
population frequencies. No final V9 qualitative findings will be claimed
until the student reviews the 24 rows.

