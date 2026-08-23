# V9 Full-Source Split Protocol Amendment

## 1. Status and purpose

This amendment replaces the previously planned 279-case confirmation frame
with a larger, source-wide V9 development/validation/test design. It is
committed before deterministic split instantiation and before inspection of
any V9 retrieval or QA outcome. V5-V8 configurations, cohorts, and results
remain frozen.

The amendment addresses limited precision and spectrum coverage in a
279-case primary evaluation while preserving the strongest project-history
holdout evidence as a nested sensitivity analysis.

## 2. Source frame and eligibility

The fixed source is:

```text
data/processed/openi_cases.jsonl
3,851 paired OpenI/IU-Xray studies
SHA-256: 56e367190396011d4d67f43e7e733389a8346890bf8729e82fb4326d063bbd68
```

Cases are classified before split selection from the normalized OpenI
`problems` field:

```text
report-indexed normal:        problems == "normal"                 1,379
report-indexed abnormal:      nonempty clinical labels             2,380
report-index indeterminate:   empty or "no indexing"                  92
                                                                    -----
source total                                                        3,851
```

The primary stratifiable split universe therefore contains 3,759 cases. The
92 indeterminate cases are excluded from primary graded retrieval evaluation;
they may be retained for ungraded engineering checks but cannot be silently
reclassified as normal or abnormal.

`Report-indexed` denotes a dataset annotation, not new physician clinical
adjudication.

## 3. Predefined V9 partition

The stratifiable universe is partitioned at study/case level as follows:

| Partition | Normal | Abnormal | Total | Role |
|---|---:|---:|---:|---|
| Train / historical bank | 965 | 1,666 | 2,631 | learned fusion and fixed retrieval bank |
| Validation | 138 | 238 | 376 | model/policy selection and threshold freezing |
| Test | 276 | 476 | 752 | one-shot final V9 evaluation |
| **Total** | **1,379** | **2,380** | **3,759** | |

The allocation approximates a 70/10/20 source-wide split while preserving
the exact source strata totals. Every validation or test query retrieves only
from the fixed 2,631-case training bank. Its own report is absent from the
bank. No validation or test report text, label, RadGraph fact, embedding, or
answer may enter retrieval fitting, adaptive-fusion training, prompt tuning,
or threshold selection.

Under the documented one-study-per-patient source design, case-disjointness
operationalizes patient-disjointness. This remains `source-design patient
uniqueness`, not identifier-verified patient disjointness.

## 4. Provenance-enriched test construction

The prior V8 reuse audit identified 262 previously unused stratifiable cases:

```text
report-indexed normal:    185
report-indexed abnormal:   77
total:                    262
SHA-256: d06254afb3ca75d0a3daeae0cdf34bd772b1eea347bc562609c12f7f1f1c6f90
```

All 262 are included in the V9 test partition. The test partition is completed
to its predefined 276/476 composition by deterministically selecting:

```text
additional normal cases:     91
additional abnormal cases:  399
additional test total:      490
```

Consequently, the full 752-case test is V9-held-out but is not described as
globally untouched across all prior project versions. The nested 262-case
subset is reported separately as the strict project-history-untouched
sensitivity analysis. No metric from that subset is used for model selection.

## 5. Deterministic split algorithm

Case identifiers are canonicalized as `str(case_id).strip()` and UTF-8
encoded. SHA-256 lowercase hexadecimal digests define all deterministic
rankings. The fixed split seed is `7029`, and domain-separated payloads are:

```text
v9-test-supplement|7029|{canonical_case_id}
v9-validation|7029|{canonical_case_id}
```

Within each normal/abnormal stratum:

1. include every strict project-history-untouched case in test;
2. hash-rank the remaining cases with the test-supplement domain and select
   the predefined additional test count;
3. hash-rank the remaining cases with the validation domain and select the
   predefined validation count;
4. assign every remaining case to train.

Sorting uses `(sha256_digest, canonical_case_id)` as a deterministic tie
break. Sampling is without replacement. The algorithm has no reserve
replacement pool.

Fingerprints use sorted unique canonical case IDs joined by `"\n"`, UTF-8
encoded, with no trailing newline.

## 6. QA evaluation frame

Retrieval uses the complete 3,759-case stratifiable universe. QA metrics that
require both a findings reference and an impression reference use the
pre-audited complete-reference subset (3,244 source cases before splitting).
Question-level denominators are reported for every partition and question
type. Missing-reference rows are not imputed and are never dropped without a
reported eligibility reason.

The final answer is evaluated against the hidden target report reference,
while retrieved historical reports serve only as analogous evidence. This
preserves the new-case similar-evidence task rather than turning it back into
retrieval of the target patient's own report.

## 7. Freeze, promotion, and rerun boundary

Before split instantiation, this amendment freezes:

- the 3,759-case split universe;
- the 2,631/376/752 partition sizes and class counts;
- inclusion of all 262 strict untouched cases in test;
- deterministic selection and fingerprint rules;
- fixed train-bank retrieval design;
- the nested sensitivity-analysis interpretation.

After split instantiation, no case may be replaced because of content,
difficulty, model performance, or an unfavorable result. A genuine source
integrity failure is retained as a documented protocol deviation. Technical
reruns are allowed only under the unchanged frozen configuration.

This is a protocol amendment with a repository timestamp, not a formal
external preregistration.

## 8. Temporal declaration

At the time of this amendment:

- the final train, validation, and test manifests have not been generated or
  inspected;
- no V9 retrieval, shuffled-control, QA, or agent outcome has been inspected;
- no V9 model or policy has been selected using validation or test outcomes.

