# V10 Publication-Extension Technical Freeze

## Scope

V10 is a publication-oriented extension of the preserved V9 thesis study. It does not alter any frozen V9 model, parameter, result, manuscript claim, or release artifact. Its purpose is to address the strongest remaining internal-validity and traceability limitations:

1. exact and near-duplicate reports are clustered before partitioning;
2. the learned reranker adds question-conditioned RadGraph fact features and multiview attention;
3. retrieved cases retain case, section, unit-type, and source-hash provenance;
4. a calibrated no-reliable-history state replaces unconditional historical support;
5. concise target-image answers are separated from deterministic historical citations; and
6. automated, clinical-review, and external-validation evidence boundaries are explicit.

The release tag is `v10-publication-freeze`. The confirmation protocol was committed before Test execution at `3f04eff`. Confirmation results were recorded without Test-driven retuning.

## Data partition and leakage control

The source contained 3,851 OpenI/IU-Xray cases. Exact and near-duplicate report clustering produced 3,013 clusters before case assignment. The deterministic cluster-disjoint partition contains:

| Partition | Cases |
|---|---:|
| Train | 2,510 |
| Calibration | 383 |
| Validation | 384 |
| Test | 574 |

No duplicate cluster crosses partitions. The retrieval bank contains 2,506 technically eligible Train cases. The final Test evaluation contains 568 technically eligible cases. Six frozen Test identities (`CXR894`, `CXR1293`, `CXR1297`, `CXR1615`, `CXR2601`, and `CXR2765`) had empty-report RadGraph records and were excluded under the frozen data-integrity rule without replacement.

The processed OpenI source lacks a reliable released patient identifier. The original collection design states that no more than one study was included per patient. V10 therefore combines source-design-supported patient separation with verified case-ID and duplicate-cluster disjointness, but does not claim identifier-verified patient separation.

## Frozen systems

Retrieval conditions are:

- `R0`: BM25 text retrieval;
- `R1`: MedSigLIP image-image retrieval;
- `R2`: MedSigLIP image-report retrieval;
- `R4`: frozen nine-feature candidate-level reranker; and
- `R5`: five-seed fact-aware reranker ensemble with the frozen multiview-attention query representation.

The target report is not available at inference and is never inserted into historical evidence. R5 ranks other-case reports from the Train bank. The primary comparison is R5 minus R4 on case-grouped nDCG@10.

The evidence policy is the development-selected whole-report `E0` condition. Although V10 implements sentence and RadGraph-fact units and preserves their provenance, development results did not justify replacing whole-report context for the primary QA confirmation. This negative selection result is retained.

QA conditions are:

- `G0`: target image and indication, without historical evidence;
- `G1`: R4 Top-3 whole-report evidence;
- `G2`: R5 Top-3 whole-report evidence with hierarchical provenance handling; and
- `G3`: G2 with historical evidence suppressed below the frozen retrieval-confidence threshold.

All conditions use the same frozen local MedGemma 1.5 generator. The final pipeline asks for a concise answer first, bounds it to at most two complete sentences, and attaches historical citations deterministically. Historical reports remain analogies rather than proof about the target patient.

## Retrieval confirmation

| System | nDCG@10 | MRR | Hit@1 | Hit@10 |
|---|---:|---:|---:|---:|
| R0 BM25 | 0.14076 | 0.07676 | 0.03228 | 0.16608 |
| R1 image-image | 0.33485 | 0.30864 | 0.22535 | 0.44014 |
| R2 image-report | 0.31760 | 0.25746 | 0.17077 | 0.41549 |
| R4 nine-feature | 0.34905 | 0.31115 | 0.23005 | 0.44131 |
| R5 fact + attention | 0.36007 | 0.31360 | 0.23826 | 0.44425 |

R5 minus R4 nDCG@10 was `+0.01103`, with a 95% case-bootstrap interval of `[+0.00770,+0.01441]` over 568 Test cases.

The correctly aligned R5 system achieved nDCG@10 `0.36007`. Across 100 deterministic fixed-point-free shuffled-image assignments, the mean was `0.24963`, the range was `[0.23621,0.26404]`, and the plus-one Monte Carlo p-value was `0.00990`. This supports alignment-specific image contribution within the same OpenI source; it does not establish diagnosis from pixels.

## Retrieval calibration

The retrieval calibrator was fitted only on the Calibration partition. Test performance was:

| Metric | Value |
|---|---:|
| Brier score | 0.16739 |
| ECE, 10 bins | 0.04579 |
| AUROC | 0.70546 |
| Frozen threshold | 0.176112 |
| Target coverage | 80.00% |
| Observed Test coverage | 81.51% |

Confidence estimates retrieval reliability under the operational label. They are not probabilities of answer correctness, diagnostic correctness, or clinical safety.

## QA confirmation

| System | Token-F1 | Complete F1RadGraph | Schema valid | Citation valid | Evidence abstention |
|---|---:|---:|---:|---:|---:|
| G0 no history | 0.14942 | 0.08265 | 100% | 100% | 100.00% |
| G1 R4 whole report | 0.20752 | 0.10507 | 100% | 100% | 0.00% |
| G2 R5 hierarchical | 0.20919 | 0.11053 | 100% | 100% | 0.00% |
| G3 calibrated selective | 0.20498 | 0.11041 | 100% | 100% | 17.08% |

G2 minus G0 Token-F1 was `+0.05978`, 95% CI `[+0.05114,+0.06860]`. G2 minus G1 was `+0.00167`, 95% CI `[-0.00347,+0.00683]`, so the R5 retrieval improvement did not produce a confirmed Token-F1 advantage over R4 whole-report RAG.

For complete F1RadGraph, G2 minus G0 was `+0.02788`, 95% CI `[+0.01977,+0.03639]`. G2 minus G1 was `+0.00546`, 95% CI `[+0.00005,+0.01089]`; the entity and entity-relation variants crossed zero. This is automated graph overlap, not physician-adjudicated correctness. No compatible local F1CheXbert implementation was available, and no proxy was substituted.

The G3 selective policy withheld historical evidence as designed but did not improve aggregate Token-F1. This negative result is retained without threshold retuning.

## Structured output and truncation

The initial one-pass JSON prompt failed during development and was stopped after 144 rows. The failure and amendment were committed before the replacement experiment. The selected answer-first design achieved 100% assembled-schema and citation validity by separating bounded answer generation from deterministic provenance assembly.

Raw answer token ceilings nevertheless remained frequent: `91.55%` for G0, `64.79%` for G1, and `68.31%` for G2/G3. Deterministic assembly fixes machine-readable structure; it does not prove that the underlying generator completed every intended answer. This limitation must remain visible in publication reporting.

## Clinical and external evidence status

The blinded clinical package contains 100 cases and 400 presentation rows, balanced between findings and impression questions. Rating fields are blank, system identities are stored separately, and `reviewer_ratings_fabricated=false`. Status is `pending_independent_review`.

The MIMIC-CXR adapter and patient-level split utility are implemented and tested, but authorized MIMIC-CXR inputs were absent and the multi-terabyte source was not downloaded. External-validation status is `adapter_ready_authorized_data_absent`; this remains Future Work and no external metric or generalization claim is reported.

## Post-freeze patient-separation clarification

The post-freeze audit in `docs/OPENI_PATIENT_SEPARATION_AUDIT.md` distinguishes
source-design support from identifier verification. It changes no V10 model,
split, output, metric or conclusion.

## Reproducibility fingerprints

| Artifact | SHA-256 |
|---|---|
| `config/v10_confirmation.json` | `d8533b46e7791da9dce6ab250121495abac13e1fe390d0415c0960a1f9edcb25` |
| `v10_cluster_disjoint_split.json` | `b4c1b091c3dbff0399d07c8350f4d8d68ce8ce52e0157dcc96f46af8c8baa7b3` |
| `v10_confirmation_retrieval_summary.json` | `1779745c26eaee0256abc01b02d822a44557820eb05c7493c6e5ae2364c537b9` |
| `v10_confirmation_qa_summary.json` | `46efd887c7eeba5ef5a0b7aa447bcb4ad8fcebd7fbee4537d0d97144a11529f2` |
| `v10_radgraph_metrics_summary.json` | `a7f014f7f142806f6ecb18219399b7a63f25f57110a13b77daed28bc24411b7e` |
| `v10_clinical_review_package_summary.json` | `ac4b425fcc0359972f555bc8787fce040a1e455496549c95ab7ffd6b91a70d81` |
| `retrieval_calibrator.json` | `e88d85e911c9214baa814d478754e34923c51568091686d63a7bebe6796a762e` |

Large per-row outputs, protected source data, image pixels, prompt packs, model weights, and private clinical-review keys remain local under repository policy.

## Software verification

The final V10-integrated test suite contains **252 passing tests**. Python compilation, Dashboard unit tests, a hash-gated live-asset retrieval smoke test, and whitespace checks passed. The cache warning caused by local `.pytest_cache` permissions does not represent a test failure.

## Claim boundary

V10 supports the following bounded conclusion:

> On a duplicate-cluster-disjoint, same-source OpenI confirmation set, fact-aware multiview reranking improved graded similar-case retrieval over the frozen nine-feature reranker, correctly aligned images outperformed shuffled-image controls, and retrieved historical context improved automated report-reference consistency over no-history generation. The incremental QA advantage of R5 over R4 whole-report RAG was not confirmed, calibrated evidence withholding did not improve aggregate Token-F1, raw generation truncation remained common, and neither independent clinical correctness nor external generalization was established.

No stronger diagnostic, safety, treatment, patient-identification, or deployment claim is permitted by the frozen evidence.
