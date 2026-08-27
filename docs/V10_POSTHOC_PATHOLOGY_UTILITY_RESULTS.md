# V10 Post-hoc Pathology Utility Supplement Results

## Status

This analysis was executed after the protocol-only commit `5773278`. It reads
the frozen V10 QA rows and does not regenerate or alter any V10 answer. The
source row SHA-256 remained
`0e82b3cf5d3913fdac82f49b6742451cf095849cad88caa2c5bedb070f793944`.

The analysis used `f1chexbert==0.0.2` and the official
`StanfordAIMI/RRG_scorers` checkpoint with SHA-256
`6550703c92d640e1e04d8105a7a185d76ece0f25fcbf033d292785bf22c0fde1`.
All 4,544 rows from 568 cases were labeled. Confidence intervals used 10,000
paired case-level bootstrap samples with seed 7139.

## Overall pathology-label consistency

| System | Micro F1 (14) | Macro F1 (14) | Micro F1 (5) | Macro F1 (5) | Exact set (5) |
|---|---:|---:|---:|---:|---:|
| G0 target image, no history | 0.54068 | 0.24974 | 0.33798 | 0.26192 | 0.74384 |
| G1 R4 historical RAG | 0.53095 | **0.27783** | 0.35448 | **0.29579** | **0.76408** |
| G2 R5 historical RAG | **0.54071** | 0.27118 | **0.35955** | 0.29166 | 0.76056 |
| G3 selective RAG | 0.53739 | 0.26167 | 0.31939 | 0.24525 | **0.76408** |

The strongest defensible comparison remains G2 versus G0. G2 increased the
five-observation micro F1 by `+0.02157`, but the 95% case-bootstrap interval
was `[-0.04064, +0.08545]`. Fourteen-observation micro F1 was effectively
unchanged: `+0.00002`, 95% CI `[-0.02572, +0.02603]`. The five-observation
exact-set difference was `+0.01673`, CI `[-0.01144, +0.04489]`.

These results do not confirm an overall pathology-label advantage for
historical RAG over target-image generation. They do show that the previously
reported Token-F1 and F1RadGraph improvements are not equivalent to a
confirmed gain on the standard CheXbert observation set.

## R5 versus R4 historical evidence

G2 compared with G1 produced:

| Metric | Difference | 95% CI |
|---|---:|---:|
| Micro F1 (14) | +0.00975 | [-0.00379, +0.02341] |
| Macro F1 (14) | -0.00665 | [-0.02255, +0.00848] |
| Micro F1 (5) | +0.00507 | [-0.02449, +0.03385] |
| Macro F1 (5) | -0.00414 | [-0.03167, +0.02155] |
| Exact set (5) | -0.00352 | [-0.01585, +0.00880] |
| Mean predicted-positive precision | +0.01138 | [-0.00132, +0.02427] |
| Positive-label Hamming agreement | +0.00245 | [0.00000, +0.00490] |

The evidence is mixed. R5's output had slightly better global label agreement
than R4's output, but the F1 intervals crossed zero. This is consistent with
the existing conclusion that stronger retrieval did not reliably transfer to
all answer-quality metrics.

## Findings and Impression questions

For Findings questions, G2 increased 14-label micro F1 from 0.48244 to 0.49563
and mean predicted-positive precision from 0.47887 to 0.51096. The latter
difference was `+0.03209`, 95% CI `[+0.00344, +0.06131]`. Other Findings
intervals crossed zero.

For Impression questions, G2 increased five-label micro F1 from 0.35789 to
0.38636 and macro F1 from 0.27361 to 0.32595, but both intervals crossed zero.
The 14-label micro F1 point estimate was lower than G0 by `-0.01310`, also with
an interval crossing zero.

Question-type results therefore suggest a precision benefit in Findings
answers rather than a uniform pathology-label improvement across report
sections.

## Condition-level observations

The largest descriptive G2-minus-G0 F1 increases were Pleural Other,
Atelectasis, Edema, Enlarged Cardiomediastinum, and Cardiomegaly. The largest
decreases were Fracture, Support Devices, Consolidation, Pneumonia, Pleural
Effusion, and Lung Opacity. Several conditions had very low positive support,
so these unadjusted condition-level differences are descriptive only and are
not separate superiority claims.

## Interpretation

This supplement answers an important evaluation gap: generated answers were
compared with the hidden target-report answers through a standard 14-observation
chest-radiograph labeler, rather than only through lexical overlap. The result
is not uniformly positive. Historical RAG substantially improved Token-F1 and
F1RadGraph in the frozen study, while the new CheXbert analysis found a modest
five-condition point improvement but no confirmed overall F1 advantage.

The correct thesis-level conclusion is therefore:

> Historical similar-case evidence improved automated report-reference text
> and graph consistency, but a post-hoc F1CheXbert analysis did not establish a
> general pathology-label superiority over target-image generation alone.

This is automated pathology-label consistency, not physician-adjudicated
diagnostic accuracy. CheXbert covers only 14 observations, maps uncertainty to
positive in its report-generation mode, and inherits domain and labeling
errors. Clinical blind review remains Future Work.

## Reproducibility artifacts

- Protocol: `docs/V10_POSTHOC_PATHOLOGY_UTILITY_PROTOCOL.md`
- Evaluator: `scripts/evaluate_v10_pathology_utility.py`
- Aggregate summary: `data/splits/v10/v10_pathology_utility_summary.json`
- Compact condition table:
  `data/splits/v10/v10_pathology_utility_conditions.csv`
- Local-only label cache:
  `experiments/v10_publication/v10_chexbert_text_label_cache.json`
- Local-only per-row output:
  `experiments/v10_publication/v10_pathology_utility_rows.csv`

