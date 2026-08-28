# V16 Data-Integrity Amendment 1

## Purpose

This amendment records a source-field completeness rule discovered while
building the V16 Train-only supervision manifest. It was identified before the
complete dataset was generated and before any V16 Calibration or Validation
outcome was inspected.

## Observed source condition

The processed OpenI source contains Train cases whose `findings` field is
empty, while the `impression` field is present. An empty source section is not a
valid supervised answer for the corresponding question and must not be turned
into a fabricated string.

## Rule

- Keep every technically eligible Train case in the historical bank.
- Generate a Findings training example only when the target `findings` field is
  non-empty after canonical whitespace normalization.
- Generate an Impression training example only when the target `impression`
  field is non-empty after canonical whitespace normalization.
- Do not substitute Impression for missing Findings, or vice versa.
- Do not use `report_text` as a substitute target section because that would
  change the question-to-answer contract.
- Record skipped case/question pairs and their reason in the machine-readable
  manifest summary.

This is a data-completeness exclusion at the question-example level, not an
outcome-driven case exclusion. Cases with one missing section remain available
as historical evidence and can still contribute their non-missing question
example.

## Effect on the committed protocol

This amendment clarifies the V16 protocol section “Training examples”. It does
not change the model, optimizer, split, evaluation metric, promotion rule, or
any V10/V11/V12--V15 result. The full Train-only manifest is generated only
after this amendment is committed.

## Reproducibility fields

The dataset summary must include:

- amendment path and commit;
- source-section availability counts by question type;
- skipped case/question count and a fingerprint of skipped pairs;
- final example count by question type and condition;
- source, split, embedding, and output hashes.

