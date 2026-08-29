# V16 Generation Confirmation Results

## Status and scope

V16 generation confirmation is complete on the frozen 568-case Test cohort. The
comparison was executed after the V16 protocol and route were fixed. No Test
result was used to change the adapter, prompt, decoding policy, section route,
retrieval system, or evaluation rules.

The evaluated task is report-reference consistency for question answering from a
target chest radiograph, indication, question, and retrieved historical cases.
The target report is hidden from the system and used only as an automated
reference. These results do **not** establish diagnostic accuracy, physician
agreement, clinical safety, or external validity.

The post-run freeze audit found a protocol-implementation deviation: 81 cases
had empty Findings references, yielding 243 empty-reference rows per arm across
the three evidence conditions. The frozen 568-case primary denominator is
retained. A post-hoc non-empty-reference sensitivity is reported below and in
`V16_PROTOCOL_DEVIATION_REFERENCE_COMPLETENESS.md`.

## Frozen generation matrix

Each of 568 cases contributes two question types (`findings` and `impression`)
under three evidence conditions (`no_history`, `random_history`, and
`retrieved_history`). Each arm therefore contains 3,408 rows.

The primary V16 route uses:

- the frozen base MedGemma generator for Findings questions and both control
  conditions;
- the frozen V16 QLoRA adapter only for Impression questions with retrieved
  historical evidence;
- greedy decoding and the same compact answer/provenance contract in every arm.

This section-aware route was fixed on non-Test development evidence because full
QLoRA improved retrieved-history answers but degraded the no-history condition.

## Primary result

| Retrieved-history arm | Token-F1 | Difference vs base | 95% case-grouped CI |
|---|---:|---:|---:|
| Base generator | 0.20570 | - | - |
| V16 impression-gated route | **0.25591** | **+0.05020** | **[+0.03973, +0.06108]** |

The primary confidence interval is wholly above zero. Relative to the base
retrieved-history arm, the absolute improvement corresponds to approximately
24.4% relative improvement. The contract-valid and provenance-valid rates were
100% in both arms.

The gain is localized as intended. Findings output is unchanged at 0.27661
Token-F1 because it uses the base route. Impression Token-F1 increases from
0.13480 to **0.23520**.

## Historical-evidence controls

| V16 route condition | Token-F1 | Difference from retrieved history | 95% CI for retrieved minus control |
|---|---:|---:|---:|
| No history | 0.16922 | +0.08668 | positive, excludes zero |
| Random history | 0.19608 | +0.05982 | [+0.04685, +0.07289] |
| Retrieved history | **0.25591** | - | - |

Retrieved history outperforms both no history and deterministically assigned
random history. The random-history control is important because it separates the
effect of relevant retrieval from the effect of merely adding more report text.

## Standard NLG metrics

All prespecified standard NLG differences for the retrieved-history arm favor
the V16 route and have case-grouped 95% confidence intervals above zero.

| Metric | Base | V16 route | Difference | 95% CI |
|---|---:|---:|---:|---:|
| BLEU-1 (row mean) | 0.09123 | **0.14406** | **+0.05284** | [+0.04262, +0.06330] |
| BLEU-4 (row mean) | 0.00653 | **0.02016** | **+0.01363** | [+0.00773, +0.02041] |
| ROUGE-L | 0.10773 | **0.16788** | **+0.06016** | [+0.04975, +0.07064] |
| METEOR | 0.13560 | **0.17049** | **+0.03489** | [+0.02522, +0.04492] |
| CIDEr | 0.07902 | **0.31849** | **+0.23947** | [+0.17606, +0.30897] |
| BERTScore F1, baseline-rescaled | -0.13898 | **-0.08802** | **+0.05096** | [+0.04179, +0.06020] |

Negative absolute BERTScore values are possible after baseline rescaling. The
243 empty Findings-reference rows were retained rather than deleted, and
BERTScore assigned raw zero scores to them as documented by its implementation.
The paired difference, not the absolute sign, is the relevant comparison.

## Reference-completeness sensitivity

Across each 3,408-row arm, 3,165 rows have non-empty references. The 243 empty
rows correspond to Findings for the same 81 cases under all three history
conditions; every Impression reference is non-empty. When only empty-reference
rows are excluded, retrieved-history Token-F1 row means are 0.22150 for base and
0.27555 for the route. The case-grouped paired difference remains positive at
**+0.04571**, 95% CI **[+0.03371, +0.05763]**. This is a post-hoc sensitivity,
not a replacement primary analysis.

## Clinical-structure metrics

| Retrieved-history metric | Base | V16 route | Difference | 95% CI |
|---|---:|---:|---:|---:|
| RadGraph entity F1 | 0.15673 | **0.18573** | **+0.02900** | [+0.01983, +0.03840] |
| RadGraph entity-relation F1 | 0.14062 | **0.16906** | **+0.02843** | [+0.01975, +0.03767] |
| RadGraph complete F1 | 0.11565 | **0.14251** | **+0.02687** | [+0.01833, +0.03562] |
| CheXbert micro-F1, 14 labels | **0.56393** | 0.55848 | -0.00545 | [-0.01389, +0.00276] |
| CheXbert exact-set accuracy, 5 labels | 0.76937 | **0.77113** | +0.00176 | [-0.00704, +0.01056] |
| CheXbert reference-positive recall | **0.65218** | 0.64138 | **-0.01081** | **[-0.02021, -0.00170]** |

RadGraph improvements agree with the lexical and semantic metrics. CheXbert is
mixed: its primary micro-F1 difference is small and statistically inconclusive,
while reference-positive recall decreases slightly with a confidence interval
below zero. This is retained as a genuine secondary limitation. The V16 positive
claim therefore concerns overall report-reference consistency and structured
fact overlap, not uniform superiority on every clinical-label metric.

## Output length and truncation

For retrieved history, the token-ceiling rate falls from 0.87852 to **0.56602**,
a difference of **-0.31250** with a fully negative 95% confidence interval
[-0.33363, -0.29137]. Mean output length falls from 91.61 to 75.75 tokens while
the compact answer contract and deterministic provenance remain valid for every
row. The improvement is concentrated in Impression generation; Findings retains
the frozen base behavior.

## Spectrum sensitivity

| Report-indexed spectrum | Cases | Base Token-F1 | V16 route | Difference | 95% CI |
|---|---:|---:|---:|---:|---:|
| Abnormal | 359 | 0.18605 | **0.21791** | **+0.03186** | [+0.02087, +0.04324] |
| Normal | 195 | 0.23946 | **0.32157** | **+0.08211** | [+0.06052, +0.10461] |
| Indeterminate | 14 | 0.23961 | 0.31563 | +0.07603 | [-0.00702, +0.16818] |

The direction is positive in all three strata, but the indeterminate subset is
too small for a supported subgroup conclusion. `Report-indexed normal`,
`abnormal`, and `indeterminate` are dataset-derived categories rather than new
clinical adjudications.

## Secondary full-QLoRA result

Applying QLoRA to all question/evidence conditions is not the final route. It
raises retrieved-history Token-F1 from 0.20570 to 0.23590 (+0.03020, confidence
interval above zero), but decreases no-history Token-F1 by 0.01616 with a
confidence interval below zero. This interaction is the reason the prespecified
section-aware gate is preferred over blanket model replacement.

## Protocol conclusion

The V16 positive-result criterion is met:

1. the primary retrieved-history Token-F1 difference is positive with a 95%
   case-grouped confidence interval above zero;
2. answer-contract and evidence-provenance validity are both 100%;
3. RadGraph improves, while the CheXbert micro-F1 interval is not wholly below
   zero;
4. the frozen V16 route outperforms both no-history and random-history controls.

The conclusion remains bounded: section-aware QLoRA adaptation improves
automated same-source report-reference consistency and substantially reduces
token-ceiling events on this case-ID- and duplicate-cluster-disjoint OpenI Test
split. Physician-reviewed clinical correctness, patient-level independence, and
external-dataset generalization remain unverified.

## Reproducibility identities

| Artifact | SHA-256 |
|---|---|
| Base generation rows | `9f026288281174ceb9ec59a219cd08f8a2841edbb6d0baef015884f2c345fb6f` |
| QLoRA generation rows | `177a7358ea95b52a32d4ada2f47bd53bf90288c2d7543a8ce0f8f37b8b039ddb3` |
| Impression-gated generation rows | `e9871a21abc381797af7dc0649a0d29c017ab9da7a7b14ef2ee93c36d56b35ce8` |
| Paired primary evaluation JSON with completeness audit | `7a8ba6f03ad514ae3d4e66476cb293ea2439a01a4418875aae02363c33e8425e` |
| Clinical-metric evaluation JSON | `95461571c2e5ddb3961d7bc91611673c89cd1b10a30ddd598c40ca8284acd462` |
| Standard-NLG evaluation JSON with completeness audit | `10f539a36f6c9459537395e798cfddf1be961d52e4ac9276a6d913604479d826` |
| BERTScore `roberta-large` weights | `047c85f0b96269cd62e6f732644f067004eebd95af5b5d35965ae2528f13bf38` |
| BERTScore rescaling baseline | `08e2248310d0c25d8e22ef65e0a6be15060269f1a9084e560c258f09a9e122ae` |

Large per-row generations, image pixels, model caches, and clinical-metric
intermediate files remain local under repository policy. Aggregate JSONs contain
no report text or image pixels and are suitable for version control.
