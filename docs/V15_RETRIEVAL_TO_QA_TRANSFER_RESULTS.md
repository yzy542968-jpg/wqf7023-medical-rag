# V15 Stronger-Retrieval-to-QA Transfer Results

## Status

V15 compared the original V12 default LambdaMART Top-3 with the stronger,
pre-existing deeper LambdaMART Top-3 while keeping the frozen 48-case
Validation cohort, target image, indication, question, prompt, whole-report
evidence policy, MedGemma revision, and 96-token budget fixed.

The historical Top-1 changed for 136 of 144 questions and the Top-3 set
changed for all 144 questions. The generation comparison therefore tested a
substantive retrieval change rather than a near-identical candidate list. V10
Test was not loaded or evaluated.

## Primary non-proxy result

The primary scope contains 96 Findings and Impression rows per condition.
Acute questions are excluded because their references are constructed from
the report rather than independently supplied acute-answer labels.

| Metric | Default Top-3 | Deeper Top-3 | Difference | 95% CI |
|---|---:|---:|---:|---:|
| Token-F1 | 0.19160 | **0.19653** | +0.00493 | [-0.02693, +0.03480] |
| F1RadGraph entity | 0.15029 | **0.16142** | +0.01113 | [-0.02870, +0.05136] |
| F1RadGraph entity-relation | 0.13035 | **0.13565** | +0.00531 | [-0.02843, +0.03750] |
| F1RadGraph complete | 0.10864 | **0.11030** | +0.00166 | [-0.02651, +0.02930] |
| F1CheXbert micro F1-14 | 0.69347 | **0.70647** | +0.01300 | [-0.05051, +0.07950] |
| F1CheXbert micro F1-5 | 0.48000 | **0.56250** | +0.08250 | [-0.16471, +0.27632] |
| F1CheXbert exact set-5 | **0.89583** | 0.86458 | -0.03125 | [-0.09375, +0.02083] |

All overlap, graph, and micro-F1 point estimates favored deeper retrieval, but
every interval crossed zero. Exact five-observation set agreement was lower,
also without a conclusive interval. The predeclared promotion rule was not
met.

Both conditions retained 100% answer-contract and provenance validity. Token
ceiling rate fell from 2.08% to 0% in the primary scope. Input length also
fell slightly because the deeper Top-3 happened to contain shorter reports;
this was not an optimized evidence-length policy.

## All-row and question-type observations

Across all 144 rows, Token-F1 changed from `0.19303` to `0.19654` with paired
difference `+0.00351` and CI `[-0.02337, +0.02897]`. Complete F1RadGraph was
numerically lower by `-0.00672`, with CI crossing zero. The acute proxy rows
therefore weakened the graph-level direction and reinforce the decision to
keep them outside the primary analysis.

The question-type point estimates were heterogeneous:

| Question | Default Token-F1 | Deeper Token-F1 | Difference |
|---|---:|---:|---:|
| Findings | **0.25520** | 0.25372 | -0.00148 |
| Impression | 0.12800 | **0.13934** | +0.01134 |
| Acute proxy | 0.19590 | 0.19656 | +0.00066 |

Impression also had positive F1CheXbert point estimates, while Findings had
mixed or negative pathology point estimates. None of these subgroup intervals
established superiority. The subgroups were inspected after the paired run
and cannot justify a question-specific retrieval router on this cohort.

## Interpretation

V15 supports a careful transfer conclusion:

> A substantial and statistically supported retrieval improvement produced
> small positive primary answer-consistency point estimates, but downstream
> superiority was not confirmed in the 48-case generation sample.

This is consistent with the broader project finding that better retrieval is
helpful but not sufficient. The generator can ignore, compress, or reinterpret
retrieved evidence, and a small paired generation cohort has wide uncertainty.
The result does not invalidate the deeper retriever; it limits the strength of
the downstream claim.

The final decision is:

```text
RETAIN V12 DEEPER LAMBDAMART AS THE STRONGEST RETRIEVAL DEVELOPMENT MODEL
REPORT V15 AS NUMERICAL BUT UNCONFIRMED QA TRANSFER
DO NOT TUNE A QUESTION-SPECIFIC ROUTER ON THESE V15 OUTCOMES
DO NOT REOPEN V10 TEST
```

## Claim boundary and artifacts

Token-F1, F1RadGraph, and F1CheXbert quantify automated consistency with a
same-source hidden report reference. They do not establish diagnostic
accuracy, radiologist agreement, clinical safety, or patient benefit.
Patient-level independence remains unverifiable in processed OpenI. Human
clinical review and external MIMIC-CXR evaluation remain Future Work.

Machine-readable records:

- `data/splits/v15/v15_retrieval_transfer_generation_summary.json`
- `data/splits/v15/v15_retrieval_transfer_evaluation_summary.json`

The 144 deeper generation rows and metric caches remain local; their SHA-256
values are recorded in the summaries.

