# Manual Annotation Guide

This guide defines how to fill:

```text
experiments/manual_annotation_qwen15_full360_comparative_sample50.csv
```

The file contains 50 questions across four systems, for 200 rows total. Each row should be scored independently, while using the reference answer and retrieved-case metadata as context.

## Columns to Fill

### `relevance_0_2`

Measures whether the answer addresses the question.

| Score | Meaning |
|---:|---|
| 2 | Directly answers the question with the correct type of information. |
| 1 | Partially answers the question, but misses important content or is too vague. |
| 0 | Does not answer the question or answers a different question. |

### `evidence_support_0_2`

Measures whether the answer is supported by the selected/retrieved report evidence.

| Score | Meaning |
|---:|---|
| 2 | All or nearly all clinical claims are supported by the relevant report evidence. |
| 1 | Some claims are supported, but the answer also includes weak, inferred, or unclear claims. |
| 0 | Main claims are unsupported, contradicted, or based on the wrong case. |

For LLM-only rows, use the reference answer as the evidence anchor because there is no retrieved case.

### `hallucination_control_0_2`

Measures how well the system avoids adding unsupported medical content.

| Score | Meaning |
|---:|---|
| 2 | No unsupported extra findings, diagnoses, recommendations, or invented details. |
| 1 | Minor unsupported wording or harmless over-explanation, but no major clinical distortion. |
| 0 | Major hallucination, invented abnormality, wrong diagnosis, or misleading recommendation. |

### `completeness_0_2`

Measures whether the answer includes enough of the reference content.

| Score | Meaning |
|---:|---|
| 2 | Covers the main relevant findings/impression. |
| 1 | Captures part of the answer but omits important report content. |
| 0 | Mostly incomplete or abstains when enough evidence appears available. |

### `case_contamination_yes_no`

Marks whether the answer appears to mix claims from multiple retrieved cases.

Use `yes` if the answer includes disease findings, impressions, or recommendations that are present in other retrieved cases but not in the target/relevant case.

Use `no` if the answer stays within the target case evidence, abstains, or is simply wrong without obvious cross-case mixing.

### `notes`

Use short notes. Helpful tags:

- `retrieval_miss`
- `top1_wrong`
- `unsupported_claim`
- `over_abstain`
- `case_mixing`
- `good_supported_answer`
- `reference_too_short`
- `checker_too_strict`

## Recommended Annotation Order

1. Sort or filter by `qid` so the four systems for the same question are viewed together.
2. Read the `question` and `reference_answer`.
3. Check `retrieved_case_ids`, `top1_hit`, and `retrieved_hit`.
4. Score each system row independently.
5. Use `notes` to record why a row is good or bad.

## Interpretation Rules

Do not reward verbosity. A short answer can receive a high score if it is correct and supported.

Do not punish a safe abstention automatically. If retrieval is wrong or evidence is insufficient, abstention may deserve high hallucination-control but low completeness.

Do not treat Token-F1 as the final truth. Token-F1 is an automatic overlap signal. Manual evidence support and hallucination control are more important for the thesis argument.

When unsure, prefer conservative scoring for medical claims. Unsupported extra clinical findings should reduce evidence support and hallucination-control scores.
