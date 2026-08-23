# V9 Post-Hoc Qualitative Analysis Protocol

## Status and boundary

This protocol is committed before systematic V9 QA/agent case extraction and
coding. The V9 retrieval configuration is already frozen, and the V9 QA
generation protocol was frozen before Test generation. This is a post-hoc,
researcher-reviewed qualitative analysis with tool-assisted deterministic
case extraction, not a blinded preregistration or clinical adjudication.

The analysis cannot alter any V9 model, prompt, threshold, case, quantitative
result, or success rule. Category frequencies describe the predefined review
set only and are not population estimates.

## Deterministic 24-case selection

Selection occurs at the case level after frozen QA and agent outputs exist.
Within each category, cases are ordered by the stated metric and then by case
ID. A selected case cannot be reused in a later category. Each category
contributes six cases:

1. largest mean `G3 - G0` Token-F1 gains;
2. largest mean `G3 - G0` Token-F1 losses;
3. agent retry cases in which R1 historical evidence is retained;
4. agent cases ending in historical-evidence abstention.

If a category has fewer than six eligible unused cases, remaining slots are
filled from unused agent-revised cases ordered by SHA-256 of
`"v9-qualitative|7033|" + case_id`. The reason for every fill is recorded.

## Review materials and taxonomy

The local review pack contains the target image path, indication, hidden
reference reports, the four frozen system answers, retrieved case IDs and
reports, automated metrics, agent trace, and a provisional coding template.
The public index excludes full report text, full generations, and image
pixels.

Provisional taxonomy v1.0:

```text
retrieval_relevance_gain
retrieval_relevance_failure
image_interpretation_alignment
reference_consistent_answer
reference_inconsistent_answer
historical_support_supported
historical_support_retry_recovered
historical_support_abstained
citation_repaired
structured_output_failure
normal_abnormal_spectrum_ambiguity
```

`assistant_proposed_labels_v1_0` must remain distinct from
`researcher_reviewed_labels_v1_0`. Until the student reviews a row, its status
is `pending_researcher_review`. The assistant must not populate reviewer
initials, review date, or researcher-confirmed labels.

Assistant proposals are deterministic: positive/negative mean `G3-G0`
Token-F1 gives a retrieval gain/failure proposal; mean G3 Token-F1 `>=0.50`
or `<0.20` gives a reference-consistent/reference-inconsistent proposal;
any invalid G3 structured output is flagged; and agent traces directly assign
retry-recovered, historical-evidence-abstained, and citation-repaired labels.
No automated rule assigns `image_interpretation_alignment`, because that
requires the researcher to inspect the image and frozen report reference.

## Interpretation boundary

The reviewer may compare outputs with frozen report references and retrieved
evidence for research interpretation. The review does not establish clinical
correctness, patient safety, diagnostic utility, or physician-validated error
rates. Human clinical evaluation remains Future Work.
