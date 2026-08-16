# Final Results Registry

Generated from locked artifacts on 2026-08-16.

## Dataset

- OpenI report cases: 3,851
- Image mapping rows: 7,466
- Modeled input: radiology report text only

## V1 Open-Corpus Stress Test

| Measure | Locked value |
|---|---:|
| Held-out cases / questions | 36 / 108 |
| Verified Token-F1 | 0.206 |
| Case-bootstrap 95% CI | [0.167, 0.246] |
| Oracle-retrieval verified Token-F1 | 0.425 |
| Ambiguous held-out queries | 23.1% |
| Retrieval abstention | 7.4% |
| Final minus Case-BM25 | +0.035 |
| Holm-adjusted p | 0.0870 |

Interpretation: retrieval-limited open-corpus failure analysis.

## V2 Controlled Case-Scoped Workflow

| Measure | Locked value |
|---|---:|
| Confirmation cases / questions | 120 / 360 |
| Locked top-k | 6 |
| Evidence recall | 99.4% |
| Extractive-context Token-F1 | 0.997 |
| Qwen Token-F1 | 0.570 |
| Qwen minus extractive | -0.427 |
| Routed candidates equal qrels | 100% |

Interpretation: controlled case-isolation and workflow-safety benchmark. Routed Hit@1 is not a semantic-retrieval claim.

## Human-Evaluation Disposition

- V1 completed blinded rows: 0/36
- V2 completed blinded rows: 0/36
- Status: not_conducted
- Decision date: 2026-08-16
- Reason: No suitable independent reviewer was available before the P2 submission deadline.

No human result is claimed or inferred from automatic metrics. The empty blinded packages are retained as an unexecuted protocol, and the absence of independent review is reported as a limitation.
