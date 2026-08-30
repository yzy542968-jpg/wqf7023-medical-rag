# Final Results Registry

Generated from locked artifacts and development summaries on 2026-08-30.

## Dataset

- OpenI report cases: 3,851
- Image mapping rows: 7,466
- Final V10 modeled input: target chest image(s), indication and question; retrieved evidence comes from other-case historical image-report pairs

## V1 Open-Corpus Stress Test

| Measure | Locked value |
|---|---:|
| Held-out cases / questions | 36 / 108 |
| Verified Token-F1 | 0.206 |
| Case-bootstrap 95% CI | [0.167, 0.246] |
| Oracle-retrieval verified Token-F1 | 0.425 |
| Ambiguous held-out queries | 23.1% |
| Retrieval abstention | 7.4% |
| Final minus Case-BM25 | +0.035 |
| Holm-adjusted p | 0.0870 |

Interpretation: retrieval-limited open-corpus failure analysis.

## V2 Controlled Case-Scoped Workflow

| Measure | Locked value |
|---|---:|
| Confirmation cases / questions | 120 / 360 |
| Locked top-k | 6 |
| Evidence recall | 99.4% |
| Extractive-context Token-F1 | 0.997 |
| Qwen Token-F1 | 0.570 |
| Qwen minus extractive | -0.427 |
| Routed candidates equal qrels | 100% |

Interpretation: controlled case-isolation and workflow-safety benchmark. Routed Hit@1 is not a semantic-retrieval claim.

## Human-Evaluation Disposition

- V1 completed blinded rows: 0/36
- V2 completed blinded rows: 0/36
- Status: not_conducted
- Decision date: 2026-08-16
- Reason: No suitable independent reviewer was available before the P2 submission deadline.

No human result is claimed or inferred from automatic metrics. The empty blinded packages are retained as an unexecuted protocol, and the absence of independent review is reported as a limitation.

## V9 Historical Final Study

| Measure | Frozen value |
|---|---:|
| Source cases | 3,851 |
| Train / Validation / Test | 2,631 / 376 / 752 |
| Learned reranker nDCG@10 | 0.3279 |
| Image-only nDCG@10 | 0.3156 |
| Learned minus image-only, 95% CI | +0.01238 [0.00923, 0.01558] |
| Learned multimodal RAG Token-F1 | 0.1848 |
| Learned RAG minus no retrieval, 95% CI | +0.03924 [0.03257, 0.04574] |

V9 is retained as historical evidence. Its post-hoc similarity audit motivated duplicate clustering before the V10 split.

## V10 Final Primary Study

| Measure | Frozen value |
|---|---:|
| Source cases / duplicate clusters | 3,851 / 3,013 |
| Train / Calibration / Validation / Test | 2,510 / 383 / 384 / 574 |
| Technically eligible Test cases | 568 |
| R4 nDCG@10 | 0.34905 |
| R5 nDCG@10 | 0.36007 |
| R5 minus R4, case-bootstrap 95% CI | +0.01103 [0.00770, 0.01441] |
| Post-hoc abnormal combined-qrel R5 minus R4 | +0.00215 [-0.00129, 0.00560] |
| Post-hoc abnormal label-only R5 minus R4 | -0.00733 [-0.01092, -0.00381] |
| Correct-image / shuffled mean nDCG@10 | 0.36007 / 0.24963 |
| Shuffled-image plus-one Monte Carlo p | 0.00990 |
| Retrieval confidence Brier / ECE / AUROC | 0.16739 / 0.04579 / 0.70546 |
| No-history Token-F1 | 0.14942 |
| R4 whole-report RAG Token-F1 | 0.20752 |
| R5 whole-report historical RAG Token-F1 | 0.20919 |
| R5 minus no history, case-bootstrap 95% CI | +0.05978 [0.05114, 0.06860] |
| R5 minus R4, case-bootstrap 95% CI | +0.00167 [-0.00347, 0.00683] |
| R5 complete F1RadGraph | 0.11053 |
| Schema / provenance integrity | 100% / 100% |

Interpretation: correctly aligned images improved report-derived similar-case retrieval, fact-aware reranking produced a small confirmed aggregate gain, and historical retrieval transferred to report-reference-consistent generation. Post-hoc qrel sensitivity showed spectrum-dependent behavior and feature-metric coupling; it does not support a uniform clinical-similarity claim. These are automated within-source results, not physician-adjudicated diagnostic accuracy.

## V11 Development Extension

| Measure | Development-only value |
|---|---:|
| Hierarchical evidence audit cases | 384 |
| Mean context reduction | 59.99% |
| Provenance completeness | 100% |
| RRF K=100 nDCG@10 / relevant presence | 0.5867 / 57.99% |
| RRF K=200 relevant-item recall | 19.11% |
| Original planner examples / accuracy | 64 / 1.0000 |
| Reserved planner examples / accuracy | 96 / 0.9167 |
| Reserved planner macro-F1 / indication invariance | 0.9196 / 1.0000 |
| Clean MedGemma cases / generations | 48 / 432 |
| Whole-report Token-F1 | 0.13117 |
| Sentence-only Token-F1 | 0.14511 |
| Case-to-fact Token-F1 | 0.15312 |
| Case-to-fact minus whole report, 95% CI | +0.02195 [-0.00026, 0.04302] |
| Case-to-fact minus whole report complete F1RadGraph, 95% CI | +0.01395 [-0.00691, 0.03442] |
| Case-to-fact mean input tokens / evidence characters | 539.3 / 245.9 |
| Whole-report mean input tokens / evidence characters | 798.2 / 672.3 |

Interpretation: V11 supports efficiency, auditability and planner wording robustness. Both primary 48-case quality intervals cross zero, so V11 does not claim confirmed generation superiority. It did not instantiate a confirmation cohort and did not modify V10.

## V12 Retrieval Method, Confirmed in V16

| Measure | Frozen value |
|---|---:|
| Test queries / Train historical cases | 568 / 2,506 |
| V10 R5 combined-qrel nDCG@10 | 0.55313 |
| V12 RRF Top-200 plus LambdaMART nDCG@10 | **0.61590** |
| V12 minus R5, 95% CI | **+0.06277 [0.05460, 0.07082]** |
| Label-only V12 minus R5, 95% CI | +0.03928 [0.02450, 0.05443] |
| Fact-only V12 minus R5, 95% CI | +0.01326 [0.00405, 0.02243] |

Interpretation: the Validation-selected multi-source candidate frame plus learned reranking transferred to the existing cluster-disjoint Test partition under three automated qrel definitions. Candidate RRF alone and full-bank LambdaMART were negative controls. The result concerns automated report-derived relevance, not physician-rated clinical similarity.

## V16 Final Integrated Generation Study

| Measure | Frozen value |
|---|---:|
| Confirmation cases / rows per arm | 568 / 3,408 |
| Base retrieved-history Token-F1 | 0.20570 |
| Impression-gated QLoRA route Token-F1 | **0.25591** |
| Route minus base, 95% CI | **+0.05020 [0.03973, 0.06108]** |
| No-history / random-history Token-F1 | 0.16922 / 0.19608 |
| Route RadGraph complete F1 | **0.14251** |
| RadGraph complete difference, 95% CI | **+0.02687 [0.01833, 0.03562]** |
| CheXbert micro-F1 difference, 95% CI | -0.00545 [-0.01389, 0.00276] |
| CheXbert reference-positive recall difference, 95% CI | **-0.01081 [-0.02021, -0.00170]** |
| Base / routed token-ceiling rate | 0.87852 / **0.56602** |
| Contract / provenance validity | 100% / 100% |
| Empty Findings-reference rows / affected cases | 243 / 81 |
| Non-empty-reference route minus base, 95% CI | +0.04571 [0.03371, 0.05763] |

All six prespecified standard NLG metric intervals favored the V16 route in the retrieved-history condition. The routed method also exceeded no history and random history. CheXbert remained mixed, including a small supported decline in reference-positive recall; this negative secondary result is retained. A post-run protocol audit found 81 cases with empty Findings references. The frozen primary denominator was retained, and a non-empty-reference sensitivity remained positive. Interpretation is limited to automated same-source report-reference consistency and structured fact overlap.

## Post-Primary Structured Final-QA Validation Extension

| Measure | Validation-only value |
|---|---:|
| Cases / questions per condition | 358 / 17,864 |
| Generated rows across four conditions | 71,456 |
| No-history exact answer-set accuracy | 0.84970 |
| Random-history exact answer-set accuracy | 0.87836 |
| Top-1 image-history exact answer-set accuracy | **0.87897** |
| Question-conditioned Top-3 exact answer-set accuracy | 0.86324 |
| No-history supported-label macro-F1 | **0.30984** |
| Random-history supported-label macro-F1 | 0.30565 |
| Top-1 image-history supported-label macro-F1 | 0.29334 |
| Question-conditioned Top-3 supported-label macro-F1 | 0.29226 |
| Top-1 minus no history, 95% CI | -0.01650 [-0.02049, -0.00235] |
| Question-conditioned minus no history, 95% CI | -0.01758 [-0.02172, -0.00305] |
| Final-QA Test access | **Not run** |

Interpretation: QLoRA produced strong structured question-level accuracy, but
the prespecified historical-RAG advancement rule failed. Meaningful history
degraded report-level supported-label macro-F1, and deterministic random
history remained competitive. The extension is retained as a negative/mixed
Validation result; it does not replace V12/V16 or support a relevance-specific
RAG claim. See `docs/FINAL_QA_VALIDATION_DECISION_RECORD.md`.

## Final Evidence Boundary

- Independent radiologist review: Future Work; no scores are reported.
- Authorized external patient-level validation: Future Work; the MIMIC-CXR adapter/runbook exists, but the multi-terabyte source was not downloaded and no result is claimed.
- Retrieval confidence: report-derived research signal, not diagnostic confidence.
- OpenI patient separation: case-ID and duplicate-cluster disjointness were verified, but identifier-based patient independence could not be verified from the processed release and is not claimed.
- Clinical safety, treatment utility and deployment performance: outside the completed evidence.
