# Final-QA Nested-OOF Robust Gate Result

## Decision

**GO for a separate confirmation design; Test remains locked.**

The nested case-level OOF gate passed every fixed development criterion. It
retained the ordinary QA improvement associated with paired report history and
produced a small positive point estimate on supported-label macro-F1. The
macro-F1 bootstrap interval crossed zero, so development does not yet confirm
superiority over image-only B3.

## Boundaries

- Final-QA Validation was previously inspected and remained development data.
- Five outer folds isolated each evaluated case from question-utility fitting
  and hyperparameter selection.
- Four inner folds selected the support and effect-margin policy separately for
  each outer fold.
- No new MedGemma generation or QLoRA training was performed.
- Final-QA Test was not generated, inspected or evaluated.
- Results are automated structured reference consistency, not clinical or
  physician-adjudicated accuracy.

## Nested design

The audit used 358 cases and 17,864 questions. Outer folds contained 63, 73,
73, 69 and 80 cases. For each outer fold, 60 fixed combinations of minimum
question support and minimum B6-over-B3 option-label macro-F1 margin were
evaluated through four-fold inner OOF predictions.

All five outer folds selected minimum support `5`. Their selected minimum
macro margins were:

```text
outer fold 0: 0.05
outer fold 1: 0.03
outer fold 2: 0.02
outer fold 3: 0.10
outer fold 4: 0.05
```

The stable median margin is `0.05`. This value is an appropriate deterministic
candidate for a later all-development fit; it was not chosen using Test.

## Main result

| Metric | B3 image-only | Nested-OOF gate | Difference |
|---|---:|---:|---:|
| Question exact answer-set accuracy | 0.84970 | **0.87136** | **+0.02166** |
| Option micro-F1 | 0.84837 | **0.86965** | **+0.02128** |
| Supported-label macro-F1 | 0.30984 | **0.31031** | **+0.00048** |
| Exact report-vector accuracy | 0.37989 | 0.37989 | 0.00000 |

The nested policy selected B6 for 2,096 question rows and for 531 genuine B3/B6
disagreements. It recovered 453 questions that B3 answered incorrectly and B6
answered exactly, while replacing 62 B3-correct answers with B6 errors.

The paired case-bootstrap macro-F1 result was:

```text
observed difference            +0.000476
bootstrap mean difference      +0.000164
95% CI                 [-0.000537, +0.000841]
P(delta > 0)                       0.695
```

The confidence interval crosses zero. Therefore the correct claim is that the
development rule passed and ordinary QA improved strongly, while rare-label
macro superiority remains unconfirmed.

## Random-history control

The random-history B4 macro-F1 was `0.30565`, below the nested gate's `0.31031`.
This closes one important gap in the earlier real-output study: the selected
nested policy has a higher primary point estimate than both B3 image-only and
B4 random history. Test confirmation is still required to determine whether
this separation generalizes.

## Advancement checks

| Requirement | Result |
|---|---|
| Nested-OOF macro-F1 exceeds B3 | Pass |
| Nested-OOF exact within -0.001 of B3 | Pass |
| Nested-OOF option micro-F1 within -0.001 of B3 | Pass |
| Nested-OOF macro-F1 exceeds random history | Pass |
| Uses B6 on at least one disagreement | Pass |

## Interpretation

This is the strongest Final-QA development result so far because it resolves
the previous trade-off at the point-estimate level:

1. paired historical reports are not forced into every answer;
2. each question type must have enough training support;
3. B6 must show a minimum rare-label advantage before replacing B3;
4. policy hyperparameters are selected without the outer evaluation case;
5. exact QA improves by more than two percentage points while macro-F1 no
   longer declines.

The result should not be overstated. The macro gain is tiny and statistically
uncertain, the policy was developed after Validation was repeatedly studied,
and the question taxonomy is specific to Rad-ReStruct. A clean Test comparison
is necessary before promoting the nested gate to a final positive thesis claim.

## Next permitted step

Fit the deterministic final question policy on all development cases with:

```text
minimum support = 5
minimum B6-over-B3 option-label macro-F1 margin = 0.05
```

Then freeze a separate Test confirmation protocol before generating or
inspecting Test identities or outcomes. The confirmation must retain B3,
random-history B4 and paired-history B6, report exact and macro metrics jointly,
and use case-grouped uncertainty. A failed Test macro result remains reportable
and must not trigger retuning.

## Reproduction

```powershell
python scripts/audit_final_qa_nested_robust_gate.py `
  --radrestruct-root <RADRESTRUCT_ROOT>
```

Machine-readable result:

`experiments/final_qa_development/final_qa_nested_robust_gate.json`
