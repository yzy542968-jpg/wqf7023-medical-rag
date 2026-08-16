# Benchmark V2: Patient-Known Evidence QA

Updated: 2026-08-14

## Research Question

Can explicit patient scoping plus deterministic report-section routing enforce case isolation and support a reproducible report-grounded QA workflow?

Benchmark V2 addresses the identifiability problem found in V1. A user-provided case identifier is used as a metadata filter, and retrieval ranks sentence-level evidence only inside that case. The prototype does not implement authentication or clinical access control. V1 remains a frozen open-corpus stress test and is not replaced.

## Data Design

| Cohort | Cases | Questions | Role |
|---|---:|---:|---|
| V2 development | 360 | 1,080 | Method development |
| V2 calibration | 120 | 360 | Top-k and verifier action selection |
| V2 test | 120 | 360 | Diagnostic evaluation; inspected before verifier-action calibration |
| V2 confirmation | 120 | 360 | Disjoint outcome-independent confirmation; primary final automated cohort |

All 720 V2 cases are disjoint from the 120 V1 cases. The confirmation cohort is also disjoint from all 600 cases used in V2 development, calibration, and initial testing. At construction time, 1,094 additional clean eligible cases remained unused.

Each case contributes three questions: findings, impression, and report conclusion. Reports are segmented into sentence-level chunks with stable IDs such as `CXR1004::findings::001`. The scoped key `(case_id, question)` is unique for all questions.

## Retrieval Conditions

1. `global_bm25`: searches sentence chunks without patient scope.
2. `case_scoped_bm25`: filters to the known case, then ranks all report sections.
3. `case_scoped_agent_routed_bm25`: a deterministic planner filters to the known case and the section required by the question type.

The routed system is deliberately described as a deterministic rule, not as learned autonomous reasoning. Because routing selects exactly the section used to define relevance, the routed candidate pool equals the qrels for every query. Routed Hit@1 is therefore a workflow sanity check, not evidence of semantic ranking ability.

### Initial Test Cohort

| Retrieval system | Hit@1 | Recall@5 | MRR | Correct case at rank 1 |
|---|---:|---:|---:|---:|
| Global BM25 | 0.000 | 0.000 | 0.001 | 0.003 |
| Case-scoped BM25 | 0.289 | 0.421 | 0.408 | 1.000 |
| Case-scoped routed BM25 | 1.000 | 0.962 | 1.000 | 1.000 |

### Confirmation Cohort

| Retrieval system | Hit@1 | Recall@5 | MRR | Correct case at rank 1 |
|---|---:|---:|---:|---:|
| Global BM25 | 0.003 | 0.002 | 0.005 | 0.008 |
| Case-scoped BM25 | 0.292 | 0.462 | 0.416 | 1.000 |
| Case-scoped routed BM25 | 1.000 | 0.978 | 1.000 | 1.000 |

Global retrieval is expected to fail because V2 intentionally uses generic questions and defines the patient through scope rather than query wording. This condition illustrates why patient identity must not be inferred from clinical-text similarity, but it is not a competitive retrieval benchmark.

## Locked Top-k

Only the 120-case calibration cohort was used to select top-k. The predeclared rule selected the smallest k reaching mean evidence recall of at least 0.95.

| k | Calibration mean recall | Complete evidence coverage |
|---:|---:|---:|
| 4 | 0.891 | 0.672 |
| 5 | 0.946 | 0.789 |
| **6** | **0.977** | **0.908** |

The locked `k=6` achieved mean evidence recall of `0.987` on the initial test and `0.994` on confirmation.

## Answer Baselines and Generation

The same local `Qwen/Qwen2.5-1.5B-Instruct` model and direct evidence-only prompt were used without test-specific tuning.

| Evaluation cohort | Extractive retrieved context F1 | Qwen Token-F1 | Qwen 95% CI | Qwen minus extractive | Evidence recall |
|---|---:|---:|---:|---:|---:|
| Diagnostic V2 test | 0.993 | 0.566 | [0.551, 0.581] | -0.427 | 0.987 |
| Primary confirmation | 0.997 | 0.570 | [0.556, 0.584] | -0.427 | 0.994 |

Confirmation Token-F1 by question type:

| Question type | Token-F1 | Evidence support audit |
|---|---:|---:|
| Findings | 0.680 | 0.729 |
| Impression | 0.612 | 0.879 |
| Summary | 0.418 | 0.882 |

The close diagnostic and confirmation Qwen results show numerical stability across disjoint cohorts. However, returning the retrieved section nearly reproduces the reference answer and outperforms Qwen by `0.427` F1. V2 is therefore primarily a section-extraction and workflow-control benchmark; it does not demonstrate a generation gain. V1's `0.206` and V2's `0.570` must not be treated as a paired model comparison because the task definition changed.

## Verifier Action Calibration

The V1 sentence-filtering verifier did not transfer safely to V2. On the V2 calibration cohort:

| Action policy | Final Token-F1 | Abstention | Interpretation |
|---|---:|---:|---|
| No rewriting / advisory audit | 0.538 | 0.000 | Preserves generated answer |
| Best sentence filter | 0.493 | 0.089 | Over-removes supported paraphrases |
| Best contradiction-only filter | 0.491 | 0.067 | NLI contradiction false positives remain |

The locked V2 policy is therefore `audit_only`. The verifier reports evidence support and flags risk for review but does not automatically alter the answer. This is a calibration-supported safety decision, not evidence that the verifier is clinically validated.

## Defensible Contribution

V2 supports a more defensible system architecture:

1. Patient identity is an explicit scope, not a semantic retrieval target.
2. A deterministic rule routes the question type to the corresponding report section.
3. Retrieval selects a compact evidence set using calibration-locked top-k.
4. The LLM answers only from that evidence.
5. Medical NLI provides an advisory grounding audit; uncertain cases remain reviewable.

## Structural Validity Audit

- Only three unique question strings occur across 1,800 main-benchmark questions.
- The routed candidate pool equals the qrels for `100%` of test and confirmation queries.
- On confirmation, `57.2%` of routed queries have all-zero BM25 scores and `77.6%` of returned scores are zero.
- Extractive context reaches `0.997` confirmation Token-F1 versus `0.570` for Qwen.
- The blinded V2 human evaluation is `0/36` complete.

These measurements are generated by `scripts/run_benchmark_v2_validity_audit.py` and stored in `experiments/benchmark_v2/validity_audit/benchmark_v2_validity_audit.json`.

## Limitations

- Section routing is deterministic and derives from three known question types; it is not a learned planner.
- Relevant chunks are inherited from report section membership rather than independent clinical annotation.
- Routed retrieval has no hard-negative chunks after section filtering, so perfect Hit@1 is structurally guaranteed.
- The extractive baseline substantially outperforms Qwen on this synthetic task.
- Generic V2 questions require an external patient identifier, matching a patient-known workflow but not open-domain QA.
- The application accepts a case ID but implements no authentication, authorization, or clinical record access control.
- Token-F1 rewards lexical overlap and is not a complete clinical correctness measure.
- Medical NLI support scores require human calibration before clinical use.
- OpenI reports are retrospective, de-identified, and text-only in the modeled pipeline.
- The 36-question blinded V1 and V2 evaluations remain external validation gates; neither should be used for further tuning.

## Locked Artifacts

- `data/processed/openi_case_scoped_benchmark_v2.json`
- `data/processed/openi_case_scoped_confirmation_v2.json`
- `experiments/benchmark_v2/calibration/locked_top_k.json`
- `experiments/benchmark_v2/calibration/semantic_verifier/semantic_agent_selection.json`
- `experiments/benchmark_v2/final_test_evaluation/test_generation_summary.json`
- `experiments/benchmark_v2/confirmation_evaluation/test_generation_summary.json`
- `experiments/benchmark_v2/confirmation_retrieval/confirmation_retrieval_summary.json`
- `experiments/benchmark_v2/validity_audit/benchmark_v2_validity_audit.json`
