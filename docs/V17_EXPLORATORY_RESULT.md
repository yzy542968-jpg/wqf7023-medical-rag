# V17 Exploratory Result

## Executive result

V17 answered a narrower question than the frozen Final-QA confirmation: can
actual-question-conditioned historical evidence show a benefit that is specific
to evidence relevance rather than merely to adding medical text?

The answer is mixed:

- **Yes at retrieval level.** The selected V17 recipe substantially improved
  same-question report-answer agreement, especially for positive questions.
- **Yes for history versus no history in QA.** Top-1 whole-report Related RAG
  improved Exact Accuracy over no history by 3.52 percentage points with a
  case-grouped confidence interval fully above zero.
- **Not confirmed for Related versus matched controls.** Related was numerically
  above Random, but the confidence interval crossed zero; Mismatched was
  numerically slightly higher in the ungated comparison.

These are automatic same-source Calibration results, not clinical accuracy.

## Retrieval-only experiment

The full Calibration retrieval experiment used 358 cases, 17,991 questions, and
2,348 asset-complete Train-bank cases. It completed without Final-QA Test access.

| Measure | Image only | V17 `full_query` | Difference |
|---|---:|---:|---:|
| Overall Top-1 same-question exact agreement | 0.62620 | 0.63726 | +0.01106 |
| Balanced Top-1 agreement | 0.33378 | 0.39413 | +0.06035 |
| Positive Top-1 agreement | 0.19831 | 0.40132 | +0.20301 |
| Top-3 any exact agreement | 0.73965 | 0.78500 | +0.04536 |
| Mean Top-1 qrel-v2 | 0.30161 | 0.29959 | -0.00201 |

The qrel-v2 decrease is retained: the question-specific proxy improved while a
broader case-similarity proxy did not. This supports the interpretation that V17
changed question relevance rather than general case similarity.

Matched Top-3 exact agreement was Related 0.78500, Random 0.66372, and
Mismatched 0.69012. All five protocol Go checks passed.

## Generator deviation and compact-fact results

The first complete generation run accidentally omitted the previously selected
QLoRA-384 adapter. It is retained as a base-generator exploratory result:
Related 0.12020 versus Random 0.08927, difference +0.03093, 95% CI
[+0.01636, +0.04529]. It must not be compared with the approximately 85-88%
Final-QA QLoRA results.

After the generator identity was corrected, compact fact evidence produced:

| Condition | Exact Accuracy | Balanced stratum accuracy |
|---|---:|---:|
| No history | 0.84886 | 0.55248 |
| Random facts | 0.85944 | 0.51249 |
| Mismatched facts | 0.85787 | 0.52140 |
| Related facts | 0.82420 | 0.53819 |

Related facts increased positive accuracy to 0.25658 versus 0.12500 for Random,
but reduced negative accuracy to 0.90096 versus 0.95094. Related negative
transfer was 6.92% of no-history-correct questions, versus 3.41% for Random.
Thus the compact representation encouraged positive predictions but harmed the
majority negative stratum and failed the primary Exact Accuracy criterion.

The cross-fitted relevance gate selected no history in all five outer folds.
It was therefore stopped rather than tuned further.

## Whole-report extension

Restoring the generator-familiar Top-1 findings/impression format gave:

| Condition | Exact Accuracy | Option micro-F1 | Balanced stratum accuracy |
|---|---:|---:|---:|
| No history | 0.84886 | 0.84529 | 0.55248 |
| Random | 0.88175 | 0.87611 | 0.54697 |
| Mismatched | 0.88528 | 0.88144 | 0.54562 |
| Related | **0.88410** | **0.87967** | 0.53903 |

Key paired comparisons:

- Related minus no history: +0.03524, 95% CI [+0.02168, +0.04820].
- Related minus Random: +0.00235, 95% CI [-0.00676, +0.01108].
- Related minus Mismatched: -0.00117, 95% CI [-0.00927, +0.00731].

This is the best V17 absolute Related result and it numerically exceeds Random,
but relevance-specific superiority is not statistically confirmed.

## Frozen-gate sensitivity

The pre-existing Final-QA question-ID gate was frozen before V17 and selected
history for 58/718 question IDs. In this pilot it used history for 324/2,554
questions:

| Selective condition | Exact Accuracy |
|---|---:|
| No history | 0.84886 |
| Mismatched | 0.87471 |
| Random | 0.87588 |
| Related | **0.87706** |

Related was numerically above Random by +0.00117 and Mismatched by +0.00235, but
both confidence intervals crossed zero. This is supportive sensitivity evidence,
not confirmation.

## Final interpretation

V17 demonstrates an important distinction:

```text
better question-specific historical retrieval
    does not guarantee
relevance-specific downstream QA superiority
```

Historical context clearly improved automatic QA relative to no history, but
random and mismatched history also helped. Part of the gain therefore appears to
come from contextual/prompt priors rather than uniquely from clinical relevance.
The generator's evidence-use behavior and evidence format are major bottlenecks.

V17 does not advance to Validation or Test. It does not change the frozen
Final-QA confirmation, and it does not establish clinical safety, patient-level
generalization, external validity, or physician usefulness.

## Verification

- Full repository test suite: 362 passed.
- Python compile audit: passed.
- Test access flag in every V17 compact summary: false.
- Large per-question rows remain local; committed hashes identify their exact
  bytes without publishing source-derived question/answer content.
