# Final-QA Confirmation Result

## Status

The frozen Final-QA confirmation completed on 530 duplicate-cluster-disjoint
Test cases and all 26,747 mapped Rad-ReStruct questions. All three generation
arms were complete before outcomes were evaluated. The final policy, endpoints
and decision thresholds were fixed before the Test manifest was instantiated.

The combined primary claim passed. This is a positive same-source structured-QA
confirmation, not external clinical validation.

## Audit trail

- Protocol commit: `3c4fc26ca0a8ca8062197f8064de507e475d5aa7`.
- Test manifest commit: `95eb4ee12dec94cbbaff69c9b12110bc65f3a775`.
- Test cases: 530.
- Test questions per condition: 26,747.
- Complete generation rows: 80,241.
- Test-development case overlap: 0.
- Test-development duplicate-cluster overlap: 0.
- Canonical row SHA-256: `7aab0154568c53b9ba701c5be3e4b4c2fa80152c65ca3eaed02e2adfd609e707`.
- Raw rows SHA-256: `a2463c43919feb1f98c46f0cb4d03bce8bd68b21d1ad227ba4dca5e2c1bd067d`.
- Frozen gate-policy SHA-256: `7d5c7e535c4938b706a5046da02e639fd33f3235a322e02d902d857fad7120f8`.
- Generator: frozen 384-step QLoRA adapter over MedGemma 1.5 4B.
- Peak allocated GPU memory in the completing invocation: 4,417.8 MiB.
- Completing invocation runtime: 38,448.1 seconds (10.68 hours), excluding
  816 rows completed in the documented pre-detachment invocation.

The two pre-output corrections and the unchanged-config resume are documented
in `FINAL_QA_CONFIRMATION_TECHNICAL_DEVIATIONS.md`. Neither pre-output attempt
produced a Test row. The later planned detachment preserved 816 completed rows
and resumed only missing run keys under the same frozen configuration.

## Frozen systems

| System | Historical context |
|---|---|
| B3 | No historical report |
| B4 | One deterministic random other-cluster Train report |
| B6 | Top-1 eligible MedSigLIP image-neighbour report |
| Final gate | B6 only for frozen eligible question IDs; otherwise B3 |

The final gate selected B6 for 3,276 questions and B3 for 23,471 questions.
The gate could not select B4.

## Primary results

| Endpoint | B3 | Final gate | Difference | 95% case-cluster bootstrap CI | Decision |
|---|---:|---:|---:|---:|---|
| Exact answer-set accuracy | 0.84978 | 0.87098 | +0.02120 | [+0.01821, +0.02422] | H1 superiority passed |
| Supported-label macro-F1 | 0.24077 | 0.23965 | -0.00112 | [-0.00251, +0.00060] | H2 non-inferiority passed |
| Option micro-F1 | 0.84734 | 0.86795 | +0.02062 | descriptive | Higher |

H1 used question-level exact accuracy as the estimand and case ID as the
cluster-resampling unit. H2 used the prespecified non-inferiority margin of
`-0.005`. Both criteria passed, so the combined positive confirmation criterion
passed. H2 does not establish macro-F1 superiority: its interval includes zero.

## Complete descriptive results

| System | Exact answer set | Option micro-F1 | Supported-label macro-F1 | Structured micro-F1 | Exact report vector |
|---|---:|---:|---:|---:|---:|
| B3 no history | 0.84978 | 0.84734 | 0.24077 | 0.97451 | 0.37736 |
| B4 random history | 0.87872 | 0.87565 | 0.23564 | 0.97455 | 0.36792 |
| B6 paired-image neighbour | 0.87789 | 0.87485 | 0.24950 | 0.97442 | 0.37358 |
| Final question gate | 0.87098 | 0.86795 | 0.23965 | 0.97473 | 0.37736 |

All final-gate outputs satisfied the answer contract and provenance checks.
The B6 contract-valid rate was 0.99985; the final gate avoided those invalid
outputs by retaining B3 for the affected question IDs.

## Secondary interpretation

Global B6 improved exact accuracy over B3 by 0.02812, with a case-cluster 95%
CI of `[+0.02345, +0.03262]`. Its supported-label macro-F1 point estimate also
improved by 0.00873, but the 95% CI `[-0.00243, +0.01238]` crossed zero.
Accordingly, B6 showed confirmed ordinary-answer improvement and a positive but
unconfirmed rare-label macro trend.

Random history B4 also improved exact accuracy over B3 by 0.02894, 95% CI
`[+0.02510, +0.03295]`. However, its supported-label macro-F1 decreased by
0.00513, 95% CI `[-0.00714, -0.00055]`. Therefore ordinary exact improvement
alone cannot be attributed specifically to clinically matched historical
content; part of it is consistent with a generic contextual prompting effect.

The relevance-specific contrast is clearer in the long-tail endpoint. The
final gate exceeded random history in supported-label macro-F1 by 0.00401, with
95% CI `[+0.00017, +0.00565]`. Correct pairing therefore contributed information
beyond indiscriminate context for this endpoint, although the absolute effect
was modest.

The conservative gate reduced negative transfer among questions B3 originally
answered correctly. Global B6 changed 322 of 22,729 such answers to incorrect
(1.42%), whereas the final gate did so for 97 (0.43%). This is the intended
trade-off: the gate retained most of the ordinary-answer gain while limiting
history-induced harm and meeting the macro-F1 non-inferiority criterion.

## Meaning of the metrics

Exact answer-set accuracy is the most direct accuracy measure for this
structured QA task. The final system answered 87.10% of question answer sets
exactly, compared with 84.98% for image-only B3.

Supported-label macro-F1 gives equal weight to each reference-supported label
and is much harsher under the 2,470-label long tail. Its lower absolute value is
not contradictory to the 87.10% question accuracy. Structured micro-F1 is high
because it aggregates label decisions over all cases, while exact report-vector
accuracy requires every label in a 2,470-dimensional report vector to match and
is consequently much stricter.

## Claim boundary

The confirmation supports this statement:

> Under the frozen same-source Final-QA protocol, a conservative
> question-conditional policy using correctly paired historical image-report
> evidence improved exact structured-QA accuracy over image-only generation
> while satisfying the prespecified supported-label macro-F1 non-inferiority
> criterion.

It does not establish diagnostic accuracy for unseen patients, physician-level
correctness, clinical safety, patient benefit or external generalization.
Rad-ReStruct answers are derived from source reports, and the Test data remain
from IU X-Ray/OpenI. Patient-level independence cannot be independently verified
from the available processed identifiers. Independent clinical review and an
external patient-disjoint dataset remain Future Work.
