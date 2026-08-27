# V10 Post-hoc Pathology Utility Supplement Protocol

## Status and purpose

This protocol defines a post-hoc automated pathology-label analysis of the
already frozen V10 answer rows. It does not amend the V10/V11 technical
freeze, alter a model, prompt, split, retrieved case, generated answer, or
primary endpoint, and it is not a new confirmation experiment.

The analysis addresses a narrower question than clinical diagnosis:

> Do answers generated with historical similar-case evidence reproduce the
> common chest-radiograph observations extracted from the hidden target report
> more consistently than answers generated from the target image alone?

The target report remains unavailable to retrieval and generation. Its frozen
question-specific reference answer is used only for offline scoring.

## Frozen input

- Answer rows:
  `experiments/v10_publication/v10_confirmation_qa_rows.jsonl`
- Expected SHA-256:
  `0e82b3cf5d3913fdac82f49b6742451cf095849cad88caa2c5bedb070f793944`
- Expected cases: 568
- Expected rows: 4,544
- Expected question types: `findings`, `impression`
- Expected systems:
  - `g0_target_image`
  - `g1_whole_report`
  - `g2_hierarchical`
  - `g3_selective`

The evaluator must fail before scoring if the source hash, system set, row
count, case count, question types, or per-system case/question coverage differs
from these values.

## Labeler and label policy

The analysis uses the official F1-CheXbert Python package, version `0.0.2`, and
the `chexbert.pth` checkpoint distributed from
`StanfordAIMI/RRG_scorers`. The package version, checkpoint SHA-256, runtime
device, source-row SHA-256, and evaluator SHA-256 must be retained in the
summary.

The standard report-generation (`rrg`) conversion is used. CheXbert uncertain
and positive outputs are mapped to positive, while negative and blank outputs
are mapped to zero. This binary conversion is part of the published
F1-CheXbert implementation and is not selected from V10 outcomes.

The 14 observations are:

1. Enlarged Cardiomediastinum
2. Cardiomegaly
3. Lung Opacity
4. Lung Lesion
5. Edema
6. Consolidation
7. Pneumonia
8. Atelectasis
9. Pneumothorax
10. Pleural Effusion
11. Pleural Other
12. Fracture
13. Support Devices
14. No Finding

The standard five-observation subset is Cardiomegaly, Edema, Consolidation,
Atelectasis, and Pleural Effusion.

## Metrics

For each system, the evaluator reports:

- 14-observation micro F1 and macro F1;
- five-observation micro F1 and macro F1;
- five-observation exact-set accuracy;
- per-observation precision, recall, F1, and reference-positive support;
- mean reference-positive finding recall per row;
- mean predicted-positive precision per row;
- mean positive-label Hamming agreement per row;
- reference-positive omission count and predicted-positive addition count.

Metrics are reported for all frozen rows and separately for Findings and
Impression questions. Zero-support labels remain visible and are not silently
removed from the 14-observation macro average.

The primary explanatory comparison is:

```text
g2_hierarchical - g0_target_image
```

The retrieval-stack comparison is secondary:

```text
g2_hierarchical - g1_whole_report
```

`g3_selective` is retained descriptively because selective abstention changes
coverage and is not an equal-coverage superiority test.

## Statistics

Confidence intervals use 10,000 paired bootstrap resamples grouped by
`case_id`. Both questions from a sampled case are retained together. The
bootstrap seed is `7139`. Metric differences are recomputed in every bootstrap
sample; row-level observations are never resampled independently.

The analysis is post-hoc and secondary. Confidence intervals describe
uncertainty in the frozen sample but do not convert the analysis into a
prospectively confirmed endpoint. No multiple-comparison-adjusted clinical
claim is permitted.

## Integrity and output policy

- Unique answer and reference strings may be cached by SHA-256 to avoid
  repeated model inference.
- Cache reuse is allowed only when labeler version, checkpoint hash, text hash,
  and label mode match.
- The evaluator must not repair, regenerate, truncate, or otherwise alter a
  frozen answer or reference.
- Empty strings must be retained and labeled through the same deterministic
  path.
- Per-row labels remain local because they are derived from locally retained
  answer rows. Aggregate summaries and a compact per-condition table may be
  committed.
- No V10/V11 artifact may be overwritten.

## Claim boundary

This supplement measures automated agreement between generated answers and
hidden report text over a 14-observation chest-radiograph ontology. It is not
physician-adjudicated diagnostic accuracy, does not cover all radiographic
findings, does not establish patient benefit or safety, and does not resolve
the lack of verified patient-level independence in the OpenI source.

Any result must be described as `automated pathology-label consistency` or
`F1-CheXbert report-reference agreement`, not as clinical diagnosis accuracy.

