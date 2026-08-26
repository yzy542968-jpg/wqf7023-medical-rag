# V11 Remaining Optimization Audit

## Scope

This document records the additional development-only work completed after
the initial V11 evidence-selection audit. V10 frozen checkpoints, V10 Test
rows, V10 confirmation metrics, and the V10 protocol were not changed. No V11
confirmation cohort was instantiated.

The audit answers four engineering questions:

1. Does a multimodal candidate generator recover relevant cases that BM25
   misses before reranking?
2. Are the V10 facts-by-attention effects supported by case-grouped
   uncertainty estimates?
3. Does the deterministic question planner cover varied wording?
4. Does case-to-fact evidence remain useful under a larger generation
   development diagnostic?

All relevance labels remain report-derived proxies. The results are not
clinical correctness, safety, physician adjudication, or external validation.

## Candidate Generation

The complete V10 Validation partition was evaluated with the V10 Train cases
as the candidate bank. Each query used the fixed indication plus one of the
three existing question roles. The audit compared BM25, MedCPT text, MedSigLIP
image, and a deterministic reciprocal-rank-fusion union. The union used the
first 100 candidates from each source and then retained a fixed output budget.

| System | nDCG@10, K=100 | relevant recall@100 | rows with relevant item in pool |
|---|---:|---:|---:|
| BM25 | 0.5537 | 0.1200 | 55.12% |
| MedCPT text | 0.5414 | 0.0989 | 47.74% |
| MedSigLIP image | 0.5190 | 0.0804 | 58.07% |
| RRF union | 0.5867 | 0.1182 | 57.99% |

The K=100 RRF union improves nDCG and the probability that at least one
proxy-relevant case enters the pool, but its relevant-item recall is slightly
below BM25. It is therefore not promoted as a superior replacement.

At K=200, the same fixed RRF ranking gives nDCG@10 0.5867, relevant-item
recall 0.1911, and relevant-item presence 65.71%. BM25 at K=200 gives recall
0.1858 and presence 61.46%. This supports a development hypothesis that a
larger candidate budget may improve recall, but it introduces additional
latency and does not prove downstream QA improvement.

The complete machine-readable outputs are:

- `data/splits/v11/v11_candidate_generation_audit_summary.json`
- `data/splits/v11/v11_candidate_generation_audit_k200_summary.json`

The correct next experiment, if a new confirmation study is approved, is to
freeze a candidate budget and rerun downstream evidence selection under the
same budget. No V10 result is retroactively changed.

## V10 2x2 Uncertainty

Case-level bootstrap resampling was applied to the 376 Validation cases. Each
case first averages its three fixed question rows; cases, not question rows,
are resampled. With 10,000 repetitions and seed 2026:

| Effect | Estimate | 95% CI |
|---|---:|---:|
| Fact-aware reranker main effect | +0.01299 | [0.00883, 0.01729] |
| Attention-view main effect | +0.00529 | [-0.00100, 0.01163] |
| Fact x attention interaction | -0.00132 | [-0.00484, 0.00221] |

The fact-aware component has a positive Validation contrast under this fixed
checkpoint comparison. The attention component is numerically positive but
its CI includes zero. The interaction does not support a claim of synergistic
amplification.

Output: `data/splits/v10/v10_fact_attention_2x2_bootstrap_summary.json`.

## Question Planner

The deterministic planner was tested on 64 author-defined wording examples,
eight per intent. After correcting coverage for comparison verbs, device
plurals, uncertainty phrasing, and summary expressions, the diagnostic
accuracy and Macro-F1 were both 1.00 on this set.

This is not an independent or blinded benchmark: the set was created during
planner development and its labels are author-defined. It demonstrates rule
coverage and reproducibility only. A future clinician-authored paraphrase set
is still needed before making a natural-language robustness claim.

Output: `data/splits/v11/v11_question_planner_benchmark_summary.json`.

## Generation Diagnostic

A clean deterministic MedGemma development run was completed on 48 V10
Validation cases, balanced by the report-indexed normal/abnormal spectrum
(24/24). It contains 432 rows across three fixed evidence policies and uses a
separate output path from the earlier interrupted four-policy trace. The
machine-readable summary is
`data/splits/v11/v11_medgemma_generation_48_clean_summary.json`; the detailed
interpretation is in `docs/V11_MEDGEMMA_GENERATION_RESULTS.md`.

| Policy | Token-F1, all rows | Mean input tokens | Mean evidence characters | Token-ceiling rate |
|---|---:|---:|---:|---:|
| Whole report | 0.1312 | 798.2 | 672.3 | 11.81% |
| Sentence only | 0.1451 | 604.1 | 351.9 | 9.72% |
| Case-to-fact | 0.1531 | 539.3 | 245.9 | 11.81% |

All three policies had 100% answer-only contract validity and 100%
deterministic provenance validity. Raw JSON validity was 0% by design because
the preferred run asked MedGemma for an answer only and attached provenance in
deterministic code. Case-to-fact reduced context length substantially and had
the highest all-row Token-F1 in this diagnostic, but it is not an independently
labeled or clinically adjudicated accuracy estimate. It is therefore retained
as an engineering efficiency/auditability result, not as confirmed medical
answer superiority.

The previous 17-case completed-subset summary remains preserved as an
interrupted-run trace and is not combined with this clean result.

## Remaining Boundaries

The largest unresolved issue remains first-stage case retrieval. A relevant
case outside the candidate budget cannot be recovered by fact selection or
generation. The current RRF result is mixed rather than a confirmed
improvement.

Selective history gating also remains non-confirmatory: the existing proxy
gate accepts nearly all rows and has weak risk-coverage discrimination. It is
retained as a diagnostic, not advertised as clinical confidence calibration.

Human review, patient-level independence verification, and authorized
external validation remain Future Work. No additional model version or Agent
framework is required to close the current thesis; the next scientifically
meaningful extension would be one new, protocol-first candidate-generation
study with downstream QA transfer and case-grouped uncertainty estimates.
