# V10 Manuscript Integration Notes

This note is a writing aid for integrating the V10 extension into a future
manuscript. It does not replace the frozen V9 submission artifacts and does not
claim clinical review or external validation.

## Study positioning

The study evaluates a report-grounded similar-case retrieval and question-
answering workflow for a new chest-radiograph case whose final report is not
available to the system. The target image, indication, and question retrieve
historical cases from a separate report-bearing bank. The historical report is
supporting evidence, not proof of a finding in the target patient.

## Methods text to preserve

V10 uses a cluster-disjoint OpenI split formed before confirmation outcomes:
2,510 Train, 383 Calibration, 384 Validation, and 574 Test cases. Exact and
near-duplicate report links were clustered before allocation. Reliable patient
identifiers were unavailable in the processed OpenI data, so the study claims
case-ID and duplicate-cluster disjointness, plus source-design-supported patient separation rather than identifier-verified patient separation.

The primary retrieval comparison is the frozen R5 fact-attention ensemble
against the R4 nine-feature reranker. Relevance is report-derived from active
label overlap and RadGraph fact overlap, with weights 0.60 and 0.40. It is an
offline evaluation construct, not a physician similarity label.

The V10 confirmation rows contain five systems. R3 fixed multimodal was
declared during development but was not included in the confirmation runner;
therefore it must not be described as a V10 Test comparator.

## Results text supported by current evidence

On the 568 technically eligible Test cases, R5 achieved nDCG@10 0.360074 and
R4 achieved 0.349049. The case-grouped difference was +0.011025 with a 95%
bootstrap interval of [+0.007698, +0.014455] under the frozen combined qrel.
Correctly aligned images also outperformed all 100 deterministic shuffled-image
controls. These findings support image-dependent, report-derived retrieval
improvement under the stated evaluation construct.

For downstream QA, R5 historical retrieval improved Token-F1 relative to the
image-only/no-history condition, but the R5-versus-R4 difference was only
numerical and its interval crossed zero. The paper should therefore separate
the evidence for retrieval improvement from the evidence for generator-level
improvement.

## Qrel sensitivity wording

The post-hoc audit found the following pattern:

```text
combined qrel:  R5 > R4 overall; abnormal subgroup unresolved
label-only:     R5 > R4 overall; R5 < R4 in abnormal subgroup
fact-only:      R5 > R4 overall and in the abnormal subgroup
```

This pattern is a validity qualification. It does not justify selecting the
fact-only result as the new primary endpoint. The combined qrel remains the
frozen primary construct, while all three variants should be presented as
sensitivity evidence.

## Terms to use carefully

Use `provenance integrity` for the deterministic check that a case/section ID
exists and belongs to retrieved evidence. Do not call this answer entailment or
clinical citation faithfulness.

Use `retrieval confidence` for the calibrator that predicts the offline qrel
threshold. Do not call it diagnostic uncertainty, clinical risk, or a safety
calibrator.

Use `question-role-conditioned` for the fixed findings/impression/acute query
roles. Do not claim broad natural-language question robustness without a
separate natural-question benchmark.

Use `report-indexed normal`, `report-indexed abnormal`, and `report-indexed
indeterminate`. These are dataset-index categories, not independent clinical
labels.

## Limitations that must remain explicit

- Relevance is report-derived and the normal/indeterminate qrel has broad
  empty-label agreement; physician-adjudicated similarity is unavailable.
- OpenI patient separation is supported by the published one-study-per-patient source design, but cannot be independently re-verified from released subject identifiers.
- The R5 Test gain combines fact-aware features and learned multiview attention.
  A separate frozen-checkpoint 2x2 audit on Validation provides descriptive
  component contrasts, but it is not a causal attribution and does not inspect
  Test outcomes.
- R5 did not show confirmed downstream QA superiority over R4.
- Raw answer token-ceiling rates remained high even though deterministic final
  assembly produced valid records.
- The six technical exclusions were not replaced, and the retrieval/QA
  confirmation uses 568 eligible Test cases.
- No independent clinical reviewer has supplied ratings.
- No authorized external MIMIC-CXR result is available; the multi-terabyte source remains a Future Work extension rather than a thesis requirement.

## Future Work

Independent clinical review and external patient-disjoint validation remain
Future Work. A corrected follow-up study should freeze a new relevance protocol
before confirmation, define explicit normal-case relevance, and use pooled
expert judgments. The current V10 frozen outputs must not be retroactively
retuned to obtain those properties.

## V11 development-only extension text

If the V11 extension is mentioned in a revised manuscript, it should be
described as an engineering development study after the V10 freeze, not as a
new confirmation experiment. The system retains case-level retrieval, then
selects question-relevant sentence/RadGraph-fact units within each retrieved
case. On 384 Validation cases and 1,152 fixed-question rows, mean evidence
length fell from 790.5 to 316.3 characters while deterministic provenance
completeness remained 100%.

The corrected qrel-v2 audit computed relevance against the full 2,510-case
Train bank before evaluating Top-100. Full-bank nDCG@10 was 0.5537, qrel>=0.5
relevant-item recall@100 was 12.00%, and 79.69% of rows had at least one
qrel-relevant item outside Top-100. These figures make the limitation explicit:
fact selection improves context efficiency, but it cannot rescue a relevant
case that was never retrieved.

The V11 question planner is deterministic and question-led; indication text is
only a fallback when a question is empty. The selective gate uses within-list
score normalization to avoid the previous near-one confidence saturation. Its
threshold remains an offline report-derived proxy threshold, not a clinical
risk calibration.

A one-case MedGemma preflight found that compact JSON generation had 0% raw
JSON validity. The preferred answer-only generation plus deterministic
provenance path had 100% answer-contract usability on 12 smoke rows, while
whole-report inputs showed greater token-ceiling pressure than case-to-fact
inputs. This is a development diagnostic and must not be presented as a
general model-accuracy result.
