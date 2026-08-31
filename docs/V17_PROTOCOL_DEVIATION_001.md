# V17 Protocol Deviation 001: Generator Identity

## What happened

The first complete V17 generation pilot used the frozen base MedGemma 1.5 4B
model without the previously selected Final-QA QLoRA-384 adapter. The run was
technically valid and its 10,216 outputs are retained locally, but it was not a
like-for-like continuation of the Final-QA B3/B4/B6 generator that produced the
approximately 85-88% Validation/Test Exact Accuracy results.

The discrepancy was detected after the base-model V17 run completed. Repository
artifacts then verified that the Final-QA Validation generator was:

```text
google/medgemma-1.5-4b-it
+ experiments/final_qa_development/final_qa_qlora_384/adapter
```

The frozen Validation summary records `model_arm = qlora` and the adapter path.

## Why it matters

The base-model V17 result cannot be compared directly with the QLoRA B3/B4/B6
results. Its low absolute accuracy reflects a generator mismatch, not a sudden
change in dataset difficulty. It remains useful as an exploratory base-generator
mechanism result:

- Related Exact Accuracy: 0.12020.
- Random Exact Accuracy: 0.08927.
- Related minus random: +0.03093; case-grouped 95% CI [+0.01636, +0.04529].
- Related minus mismatched: +0.00626; 95% CI [-0.00733, +0.01985].

These values are automatic Calibration metrics, not clinical accuracy.

## Corrective action

A separate, fully labeled QLoRA-384 run will use:

- the same frozen 51-case / 2,554-question V17 manifest;
- the same related, random, mismatched, and no-history conditions;
- the same candidate rankings, fact selector, prompts, decoding, and evaluator;
- the already trained and previously selected Final-QA QLoRA-384 adapter;
- separate local row and summary artifacts.

No model is retrained and no case, prompt, threshold, or evidence policy is
changed. Final-QA Test remains sealed.

## Interpretation boundary

This amendment was written after the base-generator outcomes were observed.
Therefore the corrected QLoRA run is a transparent post-deviation exploratory
repeat, not a preregistered or result-blind confirmation. Both runs must remain
in the audit trail; the base result must not be deleted merely because its
absolute score is lower.

