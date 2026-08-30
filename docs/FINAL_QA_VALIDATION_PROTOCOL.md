# Final-QA Full Validation Protocol

## Status

This protocol is committed after Train/Calibration development and before any
Final-QA Validation model output is generated. It freezes full-role policy
selection. Test remains untouched. This is a repository-timestamped protocol,
not a formal external preregistration.

## Frozen development decisions

- Generator: MedGemma 1.5 4B revision
  `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`.
- Adapter: independently trained 384-forward-step q/v QLoRA candidate. The
  576-step candidate was rejected by the prespecified Calibration stopping rule.
- Decoding: greedy, 32 new tokens, `<end_of_turn>` stop token, bounded wrapper
  repair, independent question prediction.
- Historical bank: eligible V10 Train cases, excluding the target duplicate
  cluster by construction.
- No confidence gate: the Calibration gate selected full B6 coverage.
- V12 LambdaMART history is not advanced because it underperformed image-only
  history and increased negative transfer on Calibration.

## Validation frame

All 358 mapped Final-QA Validation cases and all 17,864 Rad-ReStruct questions
are evaluated. No QA-row sampling is permitted. The target report, target
RadGraph facts, gold answers, and gold prior-answer history remain hidden from
retrieval and generation. Predictions are generated independently and then
assembled into the official 2,470-dimensional answer representation.

## Conditions

| ID | Input |
| --- | --- |
| B3 | Target image, available indication, question and options; no history |
| B4 | B3 plus one deterministic random other-cluster Train report |
| B6 | B3 plus the whole report paired with the Top-1 MedSigLIP image neighbour |
| P1 | B3 plus question-conditioned facts from Top-3 MedSigLIP image neighbours |

B4 is an interference/control condition and is never eligible for selection as
the final system, even if it has the highest score. B6 and P1 are the only
clinically meaningful historical-retrieval candidates.

## Outcomes and selection

The primary Validation metric is supported-label macro-F1 over the
hierarchy-cleaned report vectors. Required secondary metrics are option
micro-F1, exact answer-set accuracy, official-compatible F1, root-question
macro-F1, balanced accuracy, ordinary accuracy beside B0 majority, positive
recall, specificity, exact report-vector accuracy, contract validity, negative
transfer, input tokens, latency, peak VRAM and provenance completeness.

Between B6 and P1, select the condition with higher supported-label macro-F1.
An exact tie selects lower mean input tokens and then B6 as the simpler policy.
The selected condition advances only if its supported-label macro-F1 and option
micro-F1 point estimates both exceed B3 and contract validity is no more than
0.010 lower. These are model-selection rules, not confirmatory success claims.

The random-history comparison is mandatory. If the selected meaningful system
does not exceed B4, the study may still proceed to one-shot Test confirmation,
but the interpretation must explicitly state that relevance-specific benefit
was not established and that generic context effects remain plausible.

Uncertainty uses 10,000 paired case-bootstrap replicates. The case is the
resampling unit. No Validation result may change the adapter, prompt, parsing,
retrieval features, history bank, Top-K, evidence selector or metric definition.

## Next boundary

After Validation completes, a development decision record freezes exactly one
meaningful history policy. A separate Test confirmation protocol and config are
then committed before any Final-QA Test generation. Test cases cannot be
removed, replaced or used for retuning.
