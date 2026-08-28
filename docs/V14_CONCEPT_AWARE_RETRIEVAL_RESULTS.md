# V14 Concept-Aware Retrieval Results

## Status

V14 evaluated whether continuous target-image pathology probabilities could
improve RRF Top-200 historical-case reranking. It used five-fold,
duplicate-cluster-grouped out-of-fold probabilities for Train queries and the
frozen V13 linear checkpoint for Calibration and Validation queries. No V10
Test case or outcome was loaded or evaluated.

The experiment retained all outcomes. It did not change the V10/V11 freeze,
the V12 development results, or the negative V13 concept-prompt result.

## Default-architecture result

The predeclared comparison used the original V12 default LambdaMART
architecture. It passed the Calibration gate:

| Metric | Base 17 | Concept 23 | Difference | 95% CI |
|---|---:|---:|---:|---:|
| Calibration combined nDCG@10 | 0.59815 | 0.62132 | +0.02317 | [+0.01693, +0.02958] |
| Calibration label-component nDCG@10 | 0.05912 | 0.07114 | +0.01202 | [+0.00456, +0.01971] |
| Calibration fact-attribute nDCG@10 | 0.59453 | 0.61007 | +0.01554 | [+0.00932, +0.02200] |

The unchanged route was then evaluated once on 376 technically eligible V10
Validation cases, producing 1,128 question rows:

| Metric | Base 17 | Concept 23 | Difference | 95% CI |
|---|---:|---:|---:|---:|
| Combined nDCG@10 | 0.59827 | **0.61635** | **+0.01808** | **[+0.01220, +0.02401]** |
| Label-component nDCG@10 | 0.05686 | **0.07130** | **+0.01444** | **[+0.00759, +0.02177]** |
| Fact-attribute nDCG@10 | 0.59361 | **0.60189** | **+0.00829** | **[+0.00232, +0.01429]** |
| Hit@1 at qrel-v2 >= 0.5 | 0.05851 | **0.08333** | **+0.02482** | **[+0.00086, +0.04965]** |
| Hit@5 at qrel-v2 >= 0.5 | 0.16312 | **0.19592** | **+0.03280** | **[+0.00443, +0.06028]** |

The positive difference was present for report-indexed normal and abnormal
queries. The indeterminate subgroup changed little and is small. Candidate
generation remained a bottleneck: RRF Top-200 contained at least one
qrel-v2-relevant historical case for 64.10% of query rows and recovered only
14.97% of all thresholded relevant items on average.

## Stronger-baseline sensitivity

The comparator audit showed that the strongest existing V12 value (`0.62023`)
came from a pre-existing deeper LambdaMART configuration, whereas the
predeclared V14 run reproduced the weaker default architecture. A transparent
post-protocol sensitivity therefore applied the same deeper architecture to
both feature sets.

The deeper comparison stopped at Calibration because it failed the frozen
`+0.005` promotion threshold:

| Metric | Deeper base 17 | Deeper concept 23 | Difference | 95% CI |
|---|---:|---:|---:|---:|
| Combined nDCG@10 | **0.62157** | 0.62167 | +0.00009 | [-0.00478, +0.00500] |
| Label-component nDCG@10 | 0.06366 | **0.07435** | +0.01069 | [+0.00474, +0.01685] |
| Fact-attribute nDCG@10 | **0.61315** | 0.61016 | -0.00299 | [-0.00800, +0.00189] |
| Hit@1 | **0.08777** | 0.08688 | -0.00089 | [-0.01950, +0.01773] |
| Hit@5 | **0.18262** | 0.18174 | -0.00089 | [-0.01950, +0.01773] |

Validation was not evaluated for the deeper concept model. The result suggests
that the higher-capacity ranker already captured most of the useful signal
available from the six concept-agreement features. The isolated
label-component gain, alongside no combined or fact-attribute improvement,
also warns against interpreting the feature as independently clinical.

## Decision

The V14 decision is:

```text
RETAIN V12 DEEPER LAMBDAMART AS THE STRONGEST CURRENT RETRIEVAL DEVELOPMENT MODEL
RETAIN V13 CONCEPT HEAD AS AN INTERPRETABLE DEVELOPMENT COMPONENT
DO NOT PROMOTE V14 CONCEPT FEATURES AS A NEW BEST RETRIEVER
DO NOT REOPEN V10 TEST
```

V14 provides a useful capacity interaction result: explicit concept agreement
helps a shallow/default ranker but does not add reliable value to the stronger
deeper ranker. It therefore does not justify more Calibration-driven tuning of
the same features.

This result also complements the V13 QA pilot. The classifier contains a real
automated report-label signal, but direct concept words degraded generation
and continuous concept features did not improve the strongest retrieval
architecture. Future use should focus on independently evaluated uncertainty
or risk coverage rather than diagnostic assertion.

## Claim boundary and reproducibility

All relevance and pathology targets are automated, report-derived proxies.
The label-component and fact-attribute sensitivities used here are internal
qrel-v2 components and are not physician judgments. The result does not
establish diagnostic accuracy, clinical similarity, safety, patient benefit,
external validity, or patient-level independence.

Machine-readable summaries:

- `data/splits/v14/v14_concept_aware_retrieval_summary.json`
- `data/splits/v14/v14_concept_aware_retrieval_deeper_summary.json`

Large OOF probabilities, per-query rankings, and model checkpoints remain
local. Their SHA-256 values are recorded in the summaries.

