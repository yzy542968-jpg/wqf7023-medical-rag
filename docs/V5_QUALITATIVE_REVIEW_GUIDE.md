# V5 Qualitative Review Guide

## Status and scope

- Review guide version: `1.1`
- Frozen protocol: `docs/V5_QUALITATIVE_ANALYSIS_PROTOCOL.md`
- V5 technical freeze: tag `v5-technical-freeze`, commit `10f57ba`
- Qualitative protocol commit: `d3b0765`
- Selection status: deterministic extraction complete
- Researcher review status: **complete on 2026-08-19**
- Assistant prefill status: complete for all 24 rows
- Refined taxonomy: `docs/V5_QUALITATIVE_TAXONOMY_V1_1.md`

This guide operationalizes the post-hoc protocol without changing any V5 model, prompt, threshold, data split, or result. The 24 rows below are representative inspection cases, not a random sample and not an estimate of clinical error prevalence. Tool-assisted observations are provisional until a researcher checks the source report, generated answers, and checker decisions.

## Review files

- Public numeric index of all 360 confirmation questions: `experiments/post_submission_v5/qualitative_case_pack.csv`
- Public 24-case representative index: `experiments/post_submission_v5/qualitative_representative_cases.csv`
- Private review worksheet with report and answer text: `outputs/v5_qualitative_researcher_review.csv`
- Private source rows: `outputs/v5_qualitative_review_local.jsonl`
- Deterministic builder: `scripts/build_v5_qualitative_review_materials.py`

The `outputs/` files remain local under repository policy. They contain full report and generation text and must not be committed. The public representative index contains identifiers, metrics, provisional categories, and review status only.

The private worksheet preserves `protocol_labels_v1_0` and is prefilled with `assistant_action_vs_v1_0`, `assistant_proposed_labels_v1_1`, and `assistant_review_note`. These fields provide a complete evidence-based recommendation for each row. They do not replace the adjacent researcher-owned fields.

## Fixed representative set

The protocol yields exactly 24 unique rows: two rows per question type in each of four strata. There is no overlap between strata.

| Orders | Stratum | Findings | Impression | Summary |
|---|---|---|---|---|
| 1-6 | Retrieval improvement | `CXR2197`, `CXR2433` | `CXR2285`, `CXR343` | `CXR2285`, `CXR3784` |
| 7-12 | Retrieval degradation | `CXR257`, `CXR3174` | `CXR1508`, `CXR257` | `CXR1120`, `CXR1508` |
| 13-18 | QA gain with support loss | `CXR1120`, `CXR3005` | `CXR2553`, `CXR3506` | `CXR2818`, `CXR3018` |
| 19-24 | Correct-retrieval generation error | `CXR2505`, `CXR2702` | `CXR112`, `CXR143` | `CXR143`, `CXR1897` |

Use the full `qid` in the worksheet when recording a decision; a case can appear under more than one question type.

## How to review each row

1. Confirm the `qid`, question type, paired case, selected case, ranks, and metric deltas.
2. For retrieval rows, compare the target and selected image filenames when useful, then compare the selected report with the reference report. Do not infer clinical correctness from rank alone.
3. For QA rows, compare the reference, report-only answer, multimodal draft, multimodal final answer, and selected report evidence.
4. For filtered sentences, inspect the exact sentence and its evidence before accepting `possible_verifier_over_rejection` or `verifier_evidence_disagreement`.
5. Check the three assistant-prefilled fields and revise them if the displayed evidence supports a different interpretation.
6. Set `researcher_decision_on_v1_1` to exactly one of `accepted`, `modified`, or `excluded`.
7. If accepted, copy the assistant proposal into `researcher_reviewed_labels_v1_1`. If modified, enter the revised semicolon-separated labels. If excluded, leave the labels empty and record the reason.
8. Add a short evidence-based note that identifies the relevant report phrase, answer phrase, or rank behavior. Add initials and review date.

## Decision meanings

- `accepted`: the researcher accepts the assistant-proposed v1.1 labels without substantive change.
- `modified`: the researcher changes, adds, or removes one or more proposed v1.1 labels.
- `excluded`: the case is removed from final qualitative interpretation and an exclusion reason is retained.
- `pending`: no researcher decision has been recorded.

The review decision is about the explanatory coding, not a medical diagnosis. The reviewer may retain multiple categories if distinct retrieval, generation, and verifier behaviors coexist.

## Coding checklist

Use the three-level definitions in `docs/V5_QUALITATIVE_TAXONOMY_V1_1.md`. In particular:

- distinguish `target_rank_improvement` from `top1_retrieval_success`;
- use `post_verification_content_loss` when content exists in the draft but disappears after filtering;
- reserve `generation_omission` for content absent from the draft itself;
- treat `abstention_case` as an outcome flag and code its possible cause separately;
- use `no_substantive_answer_loss` only as a modifier based on the frozen reference, selected report, and outputs, not as a clinical correctness judgment.

## Required boundaries

Do not convert automated flags into clinical error counts. In particular, `possible_generation_unsupported_addition` is a screening label, not a confirmed hallucination. Do not use `verifier false positive`, `verifier false negative`, `clinically correct`, or `clinically safe` without independent qualified adjudication.

The extraction and initial coding remain tool-assisted. Because the researcher accepted all 24 v1.1 proposals and the audit fields are complete, the final material may be described as **researcher-reviewed qualitative analysis**. This does not imply clinical adjudication.

## Completion check

The review is complete: all 24 rows are `accepted`, the reviewed labels match the assistant proposals, and the note, initials, and date are recorded. The outcome is `24 accepted, 0 modified, 0 excluded`.
