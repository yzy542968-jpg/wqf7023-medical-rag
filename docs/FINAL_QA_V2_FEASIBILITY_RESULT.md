# Final-QA v2 Paired-History Feasibility Result

## Decision

**Conditional GO for a lightweight selective-gating development pilot.**

Do not start a full MedGemma generation run. The cached offline experiment
shows that correctly paired historical image-report cases contain useful answer
signal, but unconditional 50/50 fusion does not improve the principal
question-level accuracy measures.

## Scope and boundary

- Historical bank: 2,351 Final-QA Train cases.
- Pilot frame: the 358-case Final-QA Validation role, which had already been
  inspected in Final-QA v1 and is therefore reused only as development data.
- Deterministic SHA-256 split: 170 cases for pilot selection and 188 cases for
  the pilot holdout.
- Pilot holdout questions: 9,368.
- Target reports and target answers were unavailable to retrieval and fusion.
- Historical payload: report-derived gold Rad-ReStruct vectors. This makes the
  experiment an oracle upper-bound diagnostic rather than a deployable system.
- Cached image-only QLoRA predictions and cached MedSigLIP image embeddings were
  reused. No model inference, model training, or Test access occurred.
- The earlier aligned-history oracle diagnostic was known before this pilot;
  this is not a blinded, preregistered, or confirmatory experiment.

## Prespecified narrow gate

The pilot selected the target-image weight on the 170-case selection partition
and evaluated the selected setting once on the 188-case holdout. It used a
Top-1 historical image neighbour and the structured answer vector derived from
that image's own paired report. Twenty deterministic fixed-point-free controls
kept historical images unchanged while rotating report payload ownership.

The selected target-image weight was `alpha = 0.5`. The narrow macro-F1 gate
passed:

| Holdout result | Value |
|---|---:|
| Image-only supported-label macro-F1 | 0.36823 |
| Correctly paired fusion macro-F1 | 0.39849 |
| Absolute difference | +0.03026 |
| Case-bootstrap 95% CI | [+0.01645, +0.03601] |
| Mean shuffled-pair fusion macro-F1 | 0.37190 |
| Maximum shuffled-pair fusion macro-F1 | 0.37604 |
| Correct pairing minus shuffled mean | +0.02659 |
| Plus-one Monte Carlo p-value | 0.04762 |

Correctly paired fusion exceeded all 20 shuffled-pair controls. This supports a
real pairing-specific development signal in the oracle payload.

## Accuracy trade-off

The same fixed fusion did not improve ordinary question-level accuracy:

| Holdout result | Exact answer-set accuracy | Option micro-F1 |
|---|---:|---:|
| Image-only QLoRA | 0.85344 | 0.88578 |
| Paired history only | 0.85589 | 0.88488 |
| Fixed 50/50 fusion | 0.85034 | 0.87846 |

Compared with image-only QA, fixed fusion changed exact accuracy by `-0.00310`
and option micro-F1 by `-0.00731`. It also reduced exact report-vector accuracy.
The macro-F1 improvement therefore reflects better coverage of less frequent
supported labels at the cost of some common-answer performance. It must not be
described as a general accuracy improvement.

## Complementarity

Image-only and paired-history predictions were meaningfully complementary:

| Question outcome | Count | Share |
|---|---:|---:|
| Both correct | 7,781 | 83.06% |
| Image-only correct, history wrong | 214 | 2.28% |
| History correct, image-only wrong | 237 | 2.53% |
| Neither correct | 1,136 | 12.13% |

A perfect source selector would reach an oracle exact answer-set accuracy of
`0.87874`, compared with `0.85344` for image-only QA. The `+0.02530` potential
gain is an upper bound, not an expected result. It establishes enough headroom
to test whether an observable-feature gate can recover part of the benefit.

## Next permitted experiment

The next experiment is deliberately small and offline:

1. use Train and the pilot-selection partition only;
2. construct features available at inference, including Top-1 similarity,
   Top-1/Top-2 margin, image-report pair compatibility, question type, answer
   agreement, neighbour agreement, and retrieval dispersion;
3. compare a deterministic threshold gate, logistic regression, and a small
   MLP, preferring the simpler model under a predefined tolerance;
4. predict whether to retain the image-only answer, use paired-history evidence,
   or abstain;
5. evaluate once on the 188-case pilot holdout against image-only, history-only,
   fixed fusion, random history, and shuffled pairing;
6. stop if exact answer-set accuracy and option micro-F1 do not improve without
   materially degrading supported-label macro-F1.

Only after this gate succeeds should the project implement deployable
report-to-fact extraction or run any new long-form MedGemma condition. Final-QA
Test remains uninstantiated and inaccessible throughout development.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\pilot_final_qa_v2_pairing_feasibility.py `
  --radrestruct-root "C:\Users\yz542_dntjhas\Documents\New project\Rad-ReStruct\data\radrestruct"
```

Machine-readable results are stored in
`experiments/final_qa_development/final_qa_v2_feasibility.json`.
