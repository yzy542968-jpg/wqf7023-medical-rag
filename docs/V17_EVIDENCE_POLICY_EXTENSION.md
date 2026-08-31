# V17 Whole-Report Evidence Policy Extension

## Motivation

The V17 retrieval-only study showed that actual-question-conditioned reranking
increased report-derived same-question agreement, especially for positive
questions. The first QLoRA-384 generation pilot nevertheless reduced primary
Exact Accuracy because six compact fact/sentence units shifted the model toward
positive predictions and doubled negative transfer relative to no history.

A cross-fitted TF-IDF relevance gate selected no history in every outer fold.
That gate is therefore stopped as a negative analysis; its threshold is not
tuned further.

The selected Final-QA QLoRA-384 generator was developed and evaluated with
whole-report findings/impression evidence. The six-unit V17 representation is a
generator-input distribution change in addition to a retrieval change. This
extension isolates retrieval by restoring the previously used whole-report
evidence format while retaining V17 candidate rankings.

## Frozen comparison

The existing 51-case / 2,554-question Calibration manifest is unchanged. The
three history conditions are:

- `related`: findings and impression from the V17 selected Top-1 case;
- `random`: findings and impression from the matched random Top-1 case;
- `mismatched`: findings and impression from the matched mismatched Top-1 case.

All three use the same prompt, QLoRA-384 adapter, maximum output tokens,
decoding, parser, and provenance fields. The already generated no-history rows
are reused byte-for-byte because their prompts and model are unchanged.

Primary metric remains Exact Accuracy. Related must exceed random to support the
specific claim that relevance, rather than extra medical text, is beneficial.
Balanced stratum accuracy, positive/non-binary behavior, negative transfer, and
related-versus-mismatched remain mandatory.

## Status and boundary

This is a post-hoc Calibration development extension motivated by an observed
input-format failure. It is not preregistered, independent confirmation,
external validation, or clinical accuracy. Final-QA Test remains sealed. The
compact-fact negative result is retained and is not replaced by this extension.

