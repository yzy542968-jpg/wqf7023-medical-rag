# Final-QA Real-Output Source-Gate Result

## Decision

**STOP under the frozen OOF advancement rule.**

Question-conditional source selection produced a large improvement in ordinary
question accuracy relative to image-only B3, but none of the three five-fold
out-of-fold policies restored supported-label macro-F1 to the B3 level. The
selected policy also remained slightly below the random-history control on the
primary metric. Test remains inaccessible.

## Evaluation boundary

- Inputs were cached outputs from the completed, real report-text MedGemma
  Validation run; no new generation was performed.
- All 358 Validation cases and 17,864 questions were development data.
- Every OOF prediction came from a fold-specific policy fitted without that
  case, but the experiment was designed after earlier Validation outcomes were
  known. It is not confirmation.
- Gate features excluded gold indices and target report text.
- Final-QA Test was not generated, inspected or evaluated.
- Metrics measure structured reference consistency, not clinical accuracy.

## Baselines

| System | Macro-F1 | Question exact | Option micro-F1 |
|---|---:|---:|---:|
| B3 image-only | **0.30984** | 0.84970 | 0.84837 |
| B4 random history | 0.30565 | 0.87836 | 0.87624 |
| B6 paired Top-1 history | 0.29334 | **0.87897** | **0.87720** |

This table captures the central tension. History materially improved ordinary
QA averages, but random history was nearly as strong as paired history and B6
reduced rare-label macro-F1.

## Five-fold OOF candidates

| OOF policy | Macro-F1 | Question exact | Option micro-F1 | History disagreements used |
|---|---:|---:|---:|---:|
| Question-ID exact utility | 0.30230 | **0.87875** | **0.87728** | 739 |
| Question-ID macro utility | **0.30448** | 0.87730 | 0.87548 | 722 |
| Logistic disagreement gate | 0.30404 | 0.87506 | 0.87357 | 607 |

The macro-utility policy was selected because macro-F1 was primary. Relative
to B3 it changed:

```text
question exact       +0.02760  (+2.76 percentage points)
option micro-F1      +0.02712  (+2.71 percentage points)
supported macro-F1  -0.00536
```

The selected policy recovered 588 questions that B3 missed and B6 answered
exactly, while replacing 95 B3-correct answers with B6 errors. This explains
the strong exact-accuracy gain. It does not guarantee good per-label balance:
the errors that remained or were introduced disproportionately affected sparse
supported labels.

The case-grouped bootstrap macro-F1 difference versus B3 was `-0.00536`, with
a 95% interval of `[-0.00909, +0.00227]`. The result neither confirms a macro
improvement nor supports declaring the selected gate superior at case level.

## Logistic gate audit

The five training-fold threshold selections were stable:

```text
fold 0: 0.55
fold 1: 0.55
fold 2: 0.60
fold 3: 0.60
fold 4: 0.60
```

This stability is encouraging, but the logistic gate still missed the primary
macro requirement. More nonlinear capacity is not the obvious next answer;
the preceding oracle experiment also found that its two-layer MLP did not
materially beat simpler alternatives.

## Advancement checks

| Requirement | Result |
|---|---|
| OOF macro-F1 exceeds B3 | **Fail** |
| OOF exact within -0.001 of B3 | Pass |
| OOF option micro-F1 within -0.001 of B3 | Pass |
| OOF macro-F1 exceeds random history | **Fail** |
| Uses paired history on disagreements | Pass |

The overall rule failed because two required conditions failed.

## Interpretation

The experiment supplies a valuable split conclusion:

1. A question-conditional ensemble can convert the B3/B6 complementarity into
   a substantial ordinary QA gain. Exact answer-set accuracy around 87.7–87.9%
   is a real result on the complete Validation question set, not a 64-question
   pilot.
2. The same ensemble does not yet solve relevance specificity or rare-label
   reconstruction. Random history remains competitive and macro-F1 remains
   below image-only.
3. The remaining problem is not primarily generator syntax or GPU capacity. It
   is source selection under sparse clinical labels.

A bounded next audit may add a minimum question-level support and utility
margin before B6 can replace B3. Such a sensitivity must be labelled post-hoc
development and must not be presented as a successful frozen result. If no
robust operating point restores macro-F1, this branch should close rather than
consume Test.

## Reproduction

```powershell
python scripts/develop_final_qa_real_output_gate.py `
  --radrestruct-root <RADRESTRUCT_ROOT>
```

Machine-readable output:

`experiments/final_qa_development/final_qa_real_output_gate.json`
