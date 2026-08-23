# V9 Researcher-Reviewed Qualitative Error Analysis

## Status and evidence boundary

The deterministic 24-case review pack was generated from the frozen V9 QA and
agent outputs under the committed post-hoc protocol. On 19 August 2026, the
named researcher reviewed the pack and accepted all 24 assistant-proposed
taxonomy v1.0 label sets without modification. No case was excluded.

This is a researcher-reviewed, tool-assisted exploratory analysis. It is not
independent radiologist adjudication, a blinded review, or a clinical error-rate
study. Counts characterize the purposively selected review pack only and must
not be extrapolated to the complete V9 Test cohort.

## Review composition

The frozen selection contains:

| Selection category | Cases |
|---|---:|
| Largest mean G3-minus-G0 Token-F1 gains | 6 |
| Largest mean G3-minus-G0 Token-F1 losses | 6 |
| Agent retry/recovery cases | 6 |
| Historical-evidence abstention cases | 6 |
| **Total** | **24** |

The accepted labels overlap because a case may expose more than one pipeline
behavior.

| Accepted exploratory label | Cases |
|---|---:|
| `retrieval_relevance_gain` | 15 |
| `retrieval_relevance_failure` | 9 |
| `reference_consistent_answer` | 2 |
| `reference_inconsistent_answer` | 13 |
| `structured_output_failure` | 10 |
| `historical_support_retry_recovered` | 5 |
| `historical_support_abstained` | 13 |
| `citation_repaired` | 5 |

These are reviewed interpretations of frozen references, retrieved reports,
model outputs, and agent traces. Terms such as `reference_consistent` describe
agreement with the available report reference and do not establish clinical
correctness for the image.

## Exploratory findings

1. Retrieval-conditioned answers could improve markedly over image-only
   answers, but the selected gains and losses were heterogeneous. A retrieval
   gain therefore did not guarantee a reference-consistent answer.
2. Thirteen reviewed rows were labeled `reference_inconsistent_answer`, showing
   that retrieval, image interpretation, generation, and formatting remained
   separable failure stages.
3. Ten reviewed rows exposed structured-output failure. This agrees with the
   aggregate result that strict JSON completeness remained a material system
   limitation rather than an isolated formatting defect.
4. The bounded agent recovered historical support in five selected cases and
   repaired citations in five. Recovery was possible, but it was not the
   dominant outcome among the deliberately selected agent-failure cases.
5. Historical-support abstention occurred in thirteen reviewed cases. Removing
   unsubstantiated historical claims improved traceability while leaving the
   target-image answer unchanged by design; it did not verify the diagnostic
   correctness of that answer.

## Audit trail

The public index retains case IDs, selection strata, assistant proposals,
researcher-reviewed labels, review status, initials, date, and a bounded note.
The local pack additionally retains the frozen source case, generation rows,
and agent traces. Full report text, image pixels, and long generations remain
local under repository policy.

The machine-readable summary is
`data/splits/v9/v9_qualitative_review_summary.json`. It records 24 accepted,
0 modified, and 0 excluded rows together with SHA-256 fingerprints for the
local review pack and public index.

Independent clinician judgments of clinical similarity, answer correctness,
harmfulness, usefulness, and abstention appropriateness remain Future Work.
