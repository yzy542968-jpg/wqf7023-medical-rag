# V10 Cluster-Disjoint Hierarchical Multimodal RAG Development Protocol

## 1. Status and objective

V10 is a new publication extension whose design is frozen before V10
partition instantiation and before any V10 retrieval or QA outcome is run.
V9 is an immutable historical baseline. Because V9 data and outcomes were
already inspected, V10 is not described as result-blind preregistration.

The objective is to test whether cluster-disjoint partitioning, hierarchical
case-to-fact retrieval, and calibrated refusal produce a more valid and
reliable new-patient similar-case QA system.

## 2. Source and data boundary

The internal source is the 3,851-case processed OpenI/IU-Xray collection. The
target report remains hidden from every inference component and is used only
for offline qrels and answer evaluation. Source-design case uniqueness is not
upgraded to independently verified patient identity.

No V10 Test identity may be previewed before this protocol and the matching
configuration are committed. The deterministic rule mathematically determines
the split; the defensible claim is only that the manifest was not generated or
inspected before protocol freeze.

## 3. Duplicate-cluster construction

Cases are connected into the same leakage-control cluster when either rule is
satisfied:

1. non-empty normalized findings-plus-impression text is exactly equal; or
2. character 3-5 gram TF-IDF cosine similarity is at least `0.95`.

Exact image-byte SHA-256 equality across any view is an additional union rule.
Perceptual dHash is recorded diagnostically but cannot join clusters because
similar chest layouts produce excessive collisions.

Text normalization is NFKC, lowercase, whitespace collapse, and trim. Empty
reports remain singleton cases. Connected components are resolved with
deterministic union-find and represented by the lexicographically smallest
canonical case ID.

## 4. Cluster-disjoint partitions

Clusters, never individual cases, are assigned to:

```text
Train        65%
Calibration  10%
Validation   10%
Test         15%
```

Assignment uses deterministic SHA-256 ordering under domain
`v10-cluster-split|7040|<cluster_id>` and a greedy deficit-minimization rule.
The rule balances total cases and report-indexed normal/abnormal/indeterminate
counts without splitting a cluster. Exact target counts are therefore not
guaranteed. Ties use canonical partition order
`Train, Calibration, Validation, Test`.

All cluster and partition manifests, source hashes, implementation hashes,
overlap assertions, and class counts are saved. Pairwise cluster overlap and
case-ID overlap must both be zero.

## 5. Development roles

Train alone supplies gradient updates and hard-negative mining. A
deterministic cluster-disjoint internal early-stop subset is carved from Train.
Validation selects one frozen retrieval architecture, evidence-selection
policy, prompt format, decoding budget, and seed/ensemble rule. Calibration is
used only after architecture freeze to fit retrieval confidence and choose
coverage thresholds. Test is executed once after a confirmation protocol and
frozen configuration commit.

All questions from one case remain in one role.

## 6. Retrieval systems

All systems search the same Train-only historical bank:

```text
R0  BM25 indication + question
R1  frozen MedSigLIP multi-view image-image
R2  frozen MedSigLIP multi-view image-report
R3  frozen fixed multimodal fusion
R4  V9-equivalent nine-feature MLP retrained on V10 Train
R5  fact-aware hierarchical reranker with question/fact features
```

R5 may use candidate-report facts and sentences because historical reports are
available archive evidence. It may not use target-report text, facts, labels,
answers, or metrics at inference. Foundation encoders remain frozen.

Primary retrieval metric is case-averaged nDCG@10 under the same report-derived
0.60 active-label plus 0.40 RadGraph-fact qrel. Label-only and fact-only qrels
are prespecified sensitivity analyses. Recall@1/5/10, MRR, query latency,
memory, and candidate-bank size are secondary.

## 7. Fact-aware training and hard negatives

R5 extends R4 with only prespecified inference-available features:

- maximum and mean query-to-candidate-sentence similarity;
- maximum and mean query-to-candidate-RadGraph-fact similarity;
- number and proportion of selected positive, negative, and uncertain facts;
- selected-evidence coverage and redundancy;
- V9 component scores, reciprocal ranks, and question-role indicators.

Hard negatives are candidates ranked highly by an inference component but
having low offline relevance. Offline relevance determines training pairs only
and never enters an inference feature. Pairwise fit uses weighted softplus
ranking loss. Seeds are `7041, 7042, 7043, 7044, 7045`; the ensemble mean is
primary unless Validation shows a prespecified material degradation of at
least `0.005` nDCG@10 relative to the best single seed, in which case the best
Validation seed is frozen. Test cannot choose this rule.

R5 promotion requires its Validation nDCG@10 to exceed R4 by at least `0.005`.
Failure is reported and R4 remains the retrieval model for downstream QA.

## 8. Hierarchical evidence selection

After case-level Top-3 retrieval, evidence is selected independently inside
each case. Sentences retain `case_id`, report section, sentence index, source
text hash, and extracted RadGraph facts. Cross-case sentence pooling is not
allowed before provenance is attached.

Validation compares only these frozen policies:

```text
E0  whole findings + impression
E1  Top-3 question-relevant sentences per case
E2  Top-2 sentences plus Top-5 RadGraph facts per case
```

Selection maximizes Validation QA Token-F1 subject to structured-valid rate
and historical-citation validity. If gains are below `0.005`, prefer the
shorter context policy in order E2, E1, E0. Test cannot change the policy.

## 9. Compact generation

Generation is separated into two bounded stages with the same frozen local
MedGemma revision:

1. target answer stage: image, indication, question, and compact historical
   evidence produce a concise answer and uncertainty only;
2. support stage: selected evidence and the answer produce zero or more valid
   cited support statements.

The final JSON object is assembled deterministically by Python. The model is
not asked to reproduce the complete nested schema. Citation IDs are filtered
against retrieved case IDs, unsupported support is removed, and missing
support becomes an explicit evidence abstention rather than malformed output.

Primary engineering success criterion is at least `99%` structurally complete
assembled rows on Validation and Test. No truncated object may be repaired by
fabricating content.

## 10. Multi-view policy

Retrieval uses normalized mean aggregation of all available image views as the
reproducible baseline. The publication extension additionally evaluates a
prespecified per-view maximum and a learned view-attention head on Validation.
If neither improves nDCG@10 by `0.005`, mean aggregation remains frozen.

Under the 8 GB generation constraint, each target view may be processed in a
bounded observation pass and the observations fused textually. No historical
image is presented as target-patient evidence.

## 11. Retrieval calibration and selective answering

After the retrieval architecture is frozen, Calibration fits confidence from:

- Top-1 score;
- Top-1 minus Top-2 margin;
- component rank agreement;
- five-seed ensemble variance when R5 is used;
- selected-evidence score and redundancy;
- query role and number of available views.

The calibrator is logistic regression with L2 regularization and deterministic
seed `7046`. The positive event is Top-1 gain at least `0.50`. Reliability,
Brier score, expected calibration error, AUROC, and risk-coverage curves are
reported. Prespecified operating points target 100%, 90%, 80%, 70%, and 50%
coverage. The system may return `no_reliable_history` while still answering
from the target image with high uncertainty.

Test reports the complete curve; no Test-selected threshold is permitted.

## 12. QA and statistical plan

QA conditions are:

```text
G0  target image only, no historical retrieval
G1  R4/R5 whole-report RAG comparator
G2  promoted hierarchical evidence RAG
G3  G2 plus calibrated no-history policy
```

Primary QA comparisons are G2-G0 and G2-G1. G3 evaluates selective utility
rather than unconditional superiority. Token-F1 and F1-RadGraph are required;
F1CheXbert, GREEN, or another metric is reported only when an official pinned
local implementation is installed. Automated metrics are not clinical gold.

All confidence intervals use 10,000 case-grouped bootstrap samples. The
aligned-image system is compared with 100 deterministic fixed-point-free
shuffled-image assignments using a plus-one Monte Carlo p-value. Findings,
impression, and all-question results are reported separately and together.

## 13. Independent clinical review

A blinded package of 100 Test cases is generated only after Test outputs are
frozen. It contains randomized system order and asks a qualified reviewer to
score retrieval similarity, target-answer consistency, historical usefulness,
potential harm, and preference. The package excludes model names and automatic
scores. Reviewer role, specialty, experience, date, exclusions, and missing
ratings are retained.

No independent clinical result is claimed until a real reviewer completes the
package. Researcher-only review is labeled as such.

## 14. External MIMIC-CXR confirmation

The external adapter requires authorized MIMIC-CXR images/reports and uses
official `subject_id`, `study_id`, and image identifiers. Patient-level
Train/Calibration/Validation/Test disjointness is mandatory. A deterministic
prespecified subset may be used; sampling may not depend on V10 outcomes.

External confirmation repeats frozen R0, the promoted retrieval system,
calibration evaluation, and compact QA without OpenI-specific retuning.
Absence of authorized data is reported as not conducted, never as a negative
or positive result.

## 15. Stopping and claim rules

V10 stops after one frozen Test execution plus unchanged technical reruns.
Technical reruns may recover crashes or file errors but may not change data,
features, prompts, thresholds, seeds, or metrics.

Claims are limited as follows:

- cluster-disjoint internal confirmation supports within-source robustness;
- aligned-versus-shuffled supports image-alignment dependence;
- calibration supports selective retrieval behavior, not clinical safety;
- automated QA metrics support reference consistency, not diagnosis;
- clinical utility requires real independent review;
- external generalization requires actual patient-disjoint external results.

