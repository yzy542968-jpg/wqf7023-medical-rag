# V5 Refined Qualitative Taxonomy v1.1

## Status and provenance

- Taxonomy version: `1.1`
- Source protocol taxonomy: `docs/V5_QUALITATIVE_ANALYSIS_PROTOCOL.md`, version `1.0`
- Frozen V5 commit: `10f57ba`
- Qualitative protocol commit: `d3b0765`
- Introduction stage: post-hoc qualitative interpretation
- Researcher review status: complete (`24 accepted, 0 modified, 0 excluded`)

The refined taxonomy was introduced during qualitative interpretation to improve stage-specific error attribution. It did not alter case selection, the frozen V5 protocol, or any quantitative result.

Protocol labels are retained in `protocol_labels_v1_0`. Assistant-proposed refinements are stored separately in `assistant_proposed_labels_v1_1`. Following researcher review on `2026-08-19`, the accepted labels are stored in `researcher_reviewed_labels_v1_1` without modification.

## Three-level structure

### Pipeline stage

- `retrieval`
- `generation`
- `verification`
- `abstention`
- `data_ambiguity`

### Specific pattern

| Stage | Pattern | Operational meaning |
|---|---|---|
| Retrieval | `target_rank_improvement` | The aligned-image condition improves the frozen target-case rank. |
| Retrieval | `target_rank_degradation` | The aligned-image condition worsens the frozen target-case rank. |
| Retrieval | `top1_retrieval_success` | The top-ranked retrieved report is aligned with the frozen target case. |
| Retrieval | `top1_retrieval_failure` | The top-ranked report is not aligned with the frozen target case. |
| Generation | `generation_omission` | Relevant content is absent from the draft answer despite being available in selected evidence. |
| Generation | `generation_focus_error` | The draft emphasizes a secondary finding instead of the frozen reference conclusion. |
| Generation | `unsupported_addition` | The draft adds content without visible support in the selected report. |
| Generation | `generation_inconsistency` | The generated statement is internally inconsistent or conflicts with visible selected evidence. |
| Generation | `negation_or_polarity_error` | Draft polarity conflicts with visible selected evidence. |
| Verification | `possible_verifier_over_rejection` | A filtered sentence appears supported by visible selected-report evidence. |
| Verification | `verifier_evidence_disagreement` | Automated checker output and visible evidence do not align clearly. |
| Verification | `post_verification_content_loss` | Content present in the draft is absent after checker filtering or abstention. This label does not itself assert verifier error. |
| Verification | `template_prefix_filtering` | Filtering affects generic answer framing rather than report-derived answer content. |
| Abstention | `abstention_consistent_with_available_evidence` | Abstention appears consistent with the evidence displayed to the pipeline; this is not a clinical correctness judgment. |
| Abstention | `suspected_unnecessary_abstention` | The system abstains despite visible selected-report support for the removed answer. |
| Data ambiguity | `deidentification_ambiguity` | De-identification tokens prevent confident evidence attribution. |
| Data ambiguity | `report_internal_inconsistency` | Findings and impression contain an apparent internal discrepancy. |

### Selection and outcome modifiers

- `qa_gain_support_loss`: final Token-F1 increases while automated support rate decreases.
- `no_substantive_answer_loss`: filtering does not visibly remove report-derived answer content. Substantive answer degradation is a qualitative judgment based on the frozen reference, retrieved report, and model outputs; it is not a clinical correctness judgment.
- `abstention_case`: an abstention is present in at least one compared condition. This flag records an outcome and does not explain its cause.

## v1.0 to v1.1 mapping

| Protocol v1.0 label | v1.1 treatment |
|---|---|
| `retrieval_improvement` | Retained in the audit trail; refined to `target_rank_improvement` plus explicit Top-1 outcome. |
| `retrieval_degradation` | Retained in the audit trail; refined to `target_rank_degradation` plus explicit Top-1 outcome. |
| `generation_omission` | Retained only when content is already absent from the draft; content removed after generation becomes `post_verification_content_loss`. |
| `unsupported_addition` | Retained only after visible evidence inspection; internal contradiction becomes `generation_inconsistency`. |
| `possible_verifier_over_rejection` | Retained as a cautious interpretation and paired with `post_verification_content_loss` where applicable. |
| `verifier_evidence_disagreement` | Retained and may be qualified by `deidentification_ambiguity` or `report_internal_inconsistency`. |
| `abstention_case` | Reclassified as a selection/outcome flag; cause is coded separately. |
| `no_obvious_error` | Replaced with a more specific pattern or `no_substantive_answer_loss` when filtering only removes template framing. |

## Audit-trail fields

| Field | Meaning |
|---|---|
| `protocol_labels_v1_0` | Original deterministic labels produced under the frozen qualitative protocol. |
| `assistant_action_vs_v1_0` | `unchanged` or `refined`, describing whether the substantive interpretation changed relative to v1.0. Label names may still be normalized under v1.1. |
| `assistant_proposed_labels_v1_1` | Stage-specific assistant coding under this taxonomy. |
| `assistant_review_note` | Evidence-based rationale for the proposal. |
| `researcher_reviewed_labels_v1_1` | Final researcher labels; empty while review is pending. |
| `researcher_decision_on_v1_1` | `pending`, `accepted`, `modified`, or `excluded`. |
| `review_note` | Researcher rationale or exclusion reason. |
| `reviewer_initials` | Researcher initials. |
| `review_date` | Researcher review date. |

The researcher accepted all assistant proposals without changing them, giving an outcome of `24 accepted, 0 modified, 0 excluded`. The separate assistant comparison remains `9 unchanged, 15 refined` relative to the substantive v1.0 interpretations.

## Interpretation limits

Qualitative category counts describe the predefined review set only and are not used for population-level statistical inference. The qualitative analysis is exploratory and is not used to modify the frozen V5 configuration or quantitative results. Terms such as clinical correctness, verifier false positive, and verifier false negative remain unsupported without independent qualified adjudication.
