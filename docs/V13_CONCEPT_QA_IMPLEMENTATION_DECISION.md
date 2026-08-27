# V13 Concept-QA Implementation Decision

## Status

This decision was written after the target-image concept head passed its
predeclared Validation gate and before concept-on/off QA generation. It does
not reopen V10 Test or amend the V10/V11 freeze.

## Frozen pilot design

- Evaluation partition: V10 Validation only.
- Cases: 96, selected deterministically after technical eligibility checks;
  48 report-indexed normal and 48 report-indexed abnormal.
- Selection: SHA-256 ordering under domain `v13-concept-qa`, seed `7143`.
- Questions: Findings and Impression only. No proxy acute question is used.
- Historical retrieval: saved V12 `rrf_lambdamart` Top-3 for the same target
  case and question.
- Evidence policy: whole-report sentence units. V12 development showed that
  whole reports were stronger than case-to-fact evidence for the two non-proxy
  questions; this choice is frozen before V13 QA outputs are generated.
- Generator: the existing pinned MedGemma 1.5 4B revision with the same target
  image in both conditions.
- Output contract: answer-only generation followed by deterministic evidence
  and provenance assembly.
- Budget: 96 new tokens, selected from the prior V12 development tie because
  it matched 128-token answer overlap at lower cost.

The paired conditions are:

1. `concept_off`: indication, question, target image, and selected historical
   evidence;
2. `concept_on`: the identical inputs plus at most five threshold-passing
   target-image observations from the frozen V13 classifier.

The concept line is explicitly described as an automated, unverified
target-image hypothesis. Observations are ordered by classifier score. If a
non-`No Finding` observation passes threshold, `No Finding` is suppressed to
avoid presenting a logically contradictory cue. If no observation remains,
the prompt states that no confident target-image concept was predicted. No
reference text or report-derived Validation label enters either prompt.

## Evaluation and stopping

The pilot reports Token-F1, complete F1RadGraph, F1CheXbert, answer-contract
validity, provenance validity, token ceilings, input length, and 10,000
case-grouped paired bootstrap intervals. Both question rows from a case remain
grouped.

Concept augmentation is a positive QA result only if `concept_on - concept_off`
is positive with a 95% interval above zero for five-observation F1CheXbert,
while Token-F1 or complete F1RadGraph does not materially degrade and contract
or provenance validity does not decline. Otherwise the classifier remains an
interpretable development component without a downstream-QA superiority claim.

No prompt, threshold, label subset, case, or output budget may be changed after
the paired outputs are inspected.

