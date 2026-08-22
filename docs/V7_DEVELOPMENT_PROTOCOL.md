# V7 Adaptive Multimodal Fusion: Development Protocol

## 1. Status and scope

This document defines the development stage for a proposed V7 extension to the
completed V6 model-modernized confirmation study. It is intended to be committed
before any V7 training output, validation outcome, or final confirmation case
identity is inspected. It is a version-controlled protocol, not a formal or
externally timestamped preregistration.

At protocol-freeze time:

- V6 models, parameters, cohorts, results, and claim boundaries remain frozen;
- V7 has not produced a result and must not be described as successful;
- the final V7 confirmation case IDs have not been instantiated;
- development outputs are permitted only on the explicitly assigned Train A,
  Train B, and Validation blocks;
- no model, feature, threshold, prompt, or metric may be changed after the
  confirmation protocol is frozen in response to confirmation outcomes.

The V7 research question is:

> Does a learned query-conditional multimodal fusion policy improve case-grouped
> retrieval MRR over a validation-tuned global fusion weight on a new, case-ID
> disjoint confirmation cohort, and does any improvement depend on correctly
> aligned images?

The primary endpoint is retrieval. Downstream report-grounded QA with frozen
MedGemma is secondary and cannot be used to select the retrieval policy.

## 2. Hypotheses and result promotion rules

### H1: adaptive fusion versus global fusion

For each question `q`, the adaptive system predicts `alpha_q` and ranks the
BM25 Top-100 shortlist with:

```text
S_i(q) = alpha_q * T_i(q) + (1 - alpha_q) * V_i(q)
```

where `T_i` and `V_i` are independently min-max-normalized BM25 and MedSigLIP
scores within the same shortlist. The global comparator uses a single
validation-selected `alpha*` for every question.

The H1 success criterion is:

```text
MRR(adaptive) - MRR(global alpha*)
95% paired case-grouped bootstrap CI lower bound > 0
```

The primary endpoint averages the three deterministic question types equally
within each case and then averages cases. The two-sided interval uses 5,000
case-grouped paired resamples with seed `7026`.

### H2: aligned-image specificity

The final frozen adaptive model is evaluated with the correct image and with
100 deterministic, unique, fixed-point-free shuffled-image assignments. For
each shuffled assignment, all image-derived candidate scores and all runtime
features used to predict `alpha_q` are recomputed. The aligned image is never
allowed to reuse its aligned `alpha_q` under a shuffled image.

Let `k` be the number of shuffled MRR values greater than or equal to the
aligned MRR. The plus-one Monte Carlo value is:

```text
p = (1 + k) / 101
```

H2 is supported when `p <= 0.05`. Zero of 100 shuffled controls is stronger
than the minimum required criterion, but the protocol does not require every
future run to obtain zero.

### Result interpretation matrix

| H1 | H2 | Interpretation permitted in the thesis |
|---|---|---|
| Pass | Pass | Positive evidence for a learned adaptive policy and alignment-specific visual contribution |
| Pass | Fail | Adaptive retrieval improved over global fusion, but alignment-specific visual contribution is not confirmed |
| Fail | Pass | Correct image information is useful under the frozen adaptive system, but adaptive fusion is not superior to global fusion |
| Fail | Fail | V7 adaptive-fusion claim is unsupported; retain V6 as the principal completed result |

If the point estimate is higher but the H1 interval crosses zero, report a
numerical improvement only. If adaptive fusion is worse than global fusion,
report that negative extension without changing V6.

## 3. Source, eligibility, and cohort separation

V7 uses the processed OpenI/IU-Xray source already used by V6. It is not an
external validation dataset. Only case-ID disjointness can be verified because
the processed records do not contain a reliable stable patient identifier.
The protocol must therefore use the following exact boundary statement:

> V7 verifies case-ID disjointness from prior project cohorts and records that
> patient-level independence cannot be verified from the processed source.

The prior-use audit must enumerate and hash case IDs from all available prior
sources, including V1/V2 development and test manifests, V4.2, V5, V6
development and confirmation manifests, V5/V6 qualitative review packs, and any
case IDs used for prior outcome inspection. The union is removed before V7
stratification. No case may be “reserved” informally without appearing in the
audit manifest.

The expected post-V6 frame, pending machine verification, is:

| Frame | Normal | Abnormal | Indeterminate | Total |
|---|---:|---:|---:|---:|
| Expected eligible remaining frame | 873 | 349 | 17 | 1,239 |
| Expected stratifiable frame | 873 | 349 | 0 | 1,222 |

`report-indexed normal` means normalized `problems == "normal"`.
`report-indexed abnormal` means a non-empty label other than `normal` and
`no indexing`. `report-index indeterminate` means `no indexing`; it is excluded
from the primary stratified frame. These are dataset-index categories, not new
clinical adjudications.

The expected confirmation-generation design is four equal-sized case blocks:

```text
Train A:       240 cases = 172 normal + 68 abnormal
Train B:       240 cases = 172 normal + 68 abnormal
Validation:    240 cases = 172 normal + 68 abnormal
Confirmation:  240 cases = 172 normal + 68 abnormal
Unused frame:  remaining eligible stratifiable cases
```

Within every block, target/distractor roles are balanced:

```text
Targets:       86 normal + 34 abnormal = 120
Distractors:   86 normal + 34 abnormal = 120
```

These counts are a frozen composition rule, not a claim that the actual IDs
have already been selected. If the machine prior-use audit changes the source
counts, the audit and protocol must be resolved before training begins; the
confirmation IDs must not be silently substituted.

## 4. Deterministic V7 cohort-generation design

The source and composition rules are fixed before case identities are
instantiated. Canonicalization is:

```text
canonical_case_id = str(case_id).strip()
UTF-8 encode the canonical string
SHA-256 digest in lowercase hexadecimal
```

For block selection, each stratum is sorted by:

```text
SHA256("v7-selection|7026|" + canonical_case_id)
```

The first required cases are selected from the normal and abnormal strata. A
second domain separates role assignment:

```text
SHA256("v7-assignment|7026|" + canonical_case_id)
```

Within every selected block, the first 86 normal and 34 abnormal IDs under the
assignment ordering are targets; the remainder are distractors. The exact
block ordering is recorded in the config and must be implemented by a single
deterministic builder.

Every collection fingerprint is computed from sorted unique canonical IDs,
joined with LF, UTF-8 encoded, and written with no trailing LF before SHA-256.
The builder must record source, prior-exclusion, eligible, stratifiable,
block, target, and distractor fingerprints. It must fail on unreadable images,
missing report fields, duplicate IDs, wrong composition, overlap, or hash
mismatch. There is no silent replacement pool.

Final confirmation IDs must not be generated before:

1. the V7 development protocol is committed;
2. development model choices and fixed epoch are recorded;
3. the V7 confirmation protocol and frozen config are committed.

## 5. Development split and information flow

The development blocks are used as follows:

1. Fold A: fit candidate learners on Train A and evaluate the predeclared
   candidate configuration on Train B.
2. Fold B: fit candidate learners on Train B and evaluate on Train A.
3. Use the mean of the two case-averaged holdout MRR values to select the
   learner hyperparameters and early-stopping epoch from the predeclared
   candidate grid. The feature family and hard-negative policy are fixed by
   this protocol and are not selected from outcomes.
4. Resolve ties by simpler model, fewer features, lower parameter count, and
   then the lexicographically smaller serialized configuration.
5. Choose the final epoch as deterministic round-half-up of the two selected
   fold epochs: `floor((E_A + E_B) / 2 + 0.5)`.
6. Refit the selected learner on Train A plus Train B using the fixed epoch and
   a scaler fitted only on the merged Train A/B data.
7. Only after this development decision is frozen, use Validation to select
   `alpha*` and the optional simple-gate threshold. The linear-versus-MLP
   complexity decision has already been resolved by the development rule.

The Validation block is not used to train foundation models, change the feature
family after looking at confirmation, or select a QA generator. Confirmation is
never used for development, threshold selection, or model selection.

All split operations are by case ID. The three questions produced by a target
case remain together in every split and in every bootstrap resample.

## 6. Retrieval inputs and frozen baselines

### 6.1 Text baseline

BM25 is fit over the 240-case candidate pool for each block using `k1=1.5` and
`b=0.75`. The query is:

```text
Clinical indication: {indication}
Question: {question}
```

The candidate ranking uses descending BM25 score and ascending canonical case
ID as the deterministic tie break. The first 100 cases form the fixed shortlist
for every image-assisted condition.

### 6.2 Image representation

MedSigLIP `google/medsiglip-448` remains frozen at the V6 revision. Findings and
impression are processed with the V6 sentence-aware chunks of at most 64
MedSigLIP tokens, without overlap or silent truncation. Each image view is
normalized, averaged by case, and normalized again. Candidate report score is
the maximum cosine similarity between the query image and its report chunks.

The V7 config must preserve the V6 revision, chunk policy, view aggregation,
candidate shortlist size, and independent within-shortlist min-max
normalization. No chunk policy may be selected after confirmation outcomes.

### 6.3 Required retrieval conditions

The same BM25 Top-100 shortlist is used for all conditions:

1. **BM25 text-only:** `alpha=1.00`.
2. **Image-only diagnostic:** `alpha=0.00` within the BM25 shortlist; this is
   not corpus-wide image retrieval.
3. **V6 fixed fusion:** `alpha=0.50`.
4. **Global fusion:** one `alpha*` selected on Validation.
5. **Simple gate:** a prespecified secondary policy that chooses text-only or
   global fusion from a validation-selected confidence threshold.
6. **Adaptive fusion:** the primary V7 learned policy producing `alpha_q`.

The target-outside-shortlist rate is reported for every block and condition.
Image-assisted reranking cannot recover a target absent from the text shortlist.

## 7. Adaptive fusion model

### 7.1 Output and score function

The learner receives one feature vector per query and outputs:

```text
alpha_q = sigmoid(g_theta(x_q))
```

The candidate ranking score is:

```text
fused_score(q, i) = alpha_q * normalized_text_score(q, i)
                    + (1 - alpha_q) * normalized_image_score(q, i)
```

The linear candidate is `g_theta(x)=w^T x+b`. The MLP candidate has one hidden
layer of 32 ReLU units followed by a linear output and sigmoid. No dropout,
batch normalization, or stochastic inference is used. This keeps the learner
small, deterministic, and auditable.

### 7.2 Permitted query-state features

The initial feature family is frozen as the following runtime-available
statistics:

- indication-present flag;
- question token count and indication token count;
- BM25 Top-1 normalized score;
- BM25 Top-1 minus Top-2 normalized-score margin;
- standard deviation and interquartile range of normalized BM25 shortlist
  scores;
- MedSigLIP Top-1 normalized score;
- MedSigLIP Top-1 minus Top-2 normalized-score margin;
- standard deviation and interquartile range of normalized image scores;
- whether the text and image Top-1 candidate IDs agree;
- Spearman rank correlation between text and image rankings within the
  shortlist.

The feature vector must not contain target ID, reference text, qrels,
target-in-shortlist status, answer source, generated answer, verifier output,
or any confirmation outcome. Feature scaling is fitted only on the training
fold, applied unchanged to its holdout, and refitted on Train A plus Train B
only for the final learner.

### 7.3 Pairwise training objective

For a query whose target is present in the BM25 Top-100 shortlist, the target
candidate is the positive item. Hard negatives are the union of the top 20
non-target candidates by BM25 rank and the top 20 non-target candidates by
image score, capped at 40 unique negatives. This candidate rule is fixed and
is not tuned from outcomes.

For each positive-negative pair, the loss is:

```text
L(theta) = softplus(-(S_positive - S_negative))
```

The loss is averaged over available pairs. Queries whose target is outside the
Top-100 shortlist produce no valid positive-negative pair and are excluded from
gradient optimization only. They remain in all validation and confirmation
retrieval metrics and are counted in the outside-shortlist diagnostic.

### 7.4 Development candidates and optimization

The predeclared candidate grid is:

```text
learner:        linear, one-hidden-layer MLP(32 ReLU)
learning_rate:  0.001, 0.0003
weight_decay:   0.0001, 0.0
batch_size:     64 pairwise examples
maximum_epochs: 100
early_stop_patience: 10 epochs
minimum_delta:  0.0001 holdout primary MRR
optimizer:      AdamW
seed:           7026
```

The seed, hard-negative policy, feature family, loss, and optimizer are not
chosen by inspecting the confirmation cohort. If several candidate settings
are tied within the development selection tolerance, choose the simpler
learner first. The linear-vs-MLP complexity rule is:

```text
if MRR_MLP - MRR_linear <= 0.005: select linear
otherwise: select MLP
```

The `0.005` boundary is inclusive. A candidate model cannot be promoted merely
because it has more parameters.

## 8. Validation selection rules

### 8.1 Global alpha

Search exactly:

```text
alpha in {0.00, 0.01, ..., 1.00}
```

Select the value with maximum Validation primary case-averaged MRR. If values
are exactly tied, choose the value nearest `0.50`; if still tied, choose the
smaller alpha. The selected value is `alpha*` and is frozen before confirmation.

### 8.2 Simple gate

The optional secondary gate compares the normalized BM25 Top-1 margin with a
single threshold from the fixed grid `{0.00, 0.05, 0.10, ..., 1.00}`. If the
margin is at least the threshold, it uses text-only `alpha=1.00`; otherwise it
uses the frozen global `alpha*`. Validation selects the threshold by primary
MRR, then higher coverage, then the smaller threshold. The gate is not used to
select the adaptive learner and is not a primary success criterion.

### 8.3 Source-balanced sensitivity

The primary metric gives the three question types equal weight. Because both
impression and summary use the frozen impression reference, the protocol also
reports the following prespecified secondary sensitivity metric:

```text
M_source-balanced = 0.50 * M_findings
                    + 0.25 * M_impression
                    + 0.25 * M_summary
```

This sensitivity metric is not used for training, model selection, or H1/H2
pass/fail decisions.

## 9. Shuffled-image implementation

For each of 100 control indices, sort target case IDs by:

```text
SHA256("v7-shuffle-order|7026|" + str(control_index) + "|" + case_id)
```

Map each source target to the next image in the ordered cycle. The builder must
verify uniqueness, no fixed points, and one-cycle coverage. Under each
assignment, recompute image embeddings or use a cache whose signature includes
the assignment, all image-derived candidate score distributions, dispersion,
rank correlation, Top-1 agreement, and the resulting `alpha_q`. The frozen
adaptive learner is then applied to those recomputed features. No aligned
feature or alpha may be reused in a shuffled condition.

## 10. Secondary MedGemma QA transfer

Only after primary retrieval outcomes and the adaptive model are frozen may the
secondary QA transfer be executed. It compares BM25, global `alpha*`, and
adaptive `alpha_q` Top-1 reports under the frozen V6 MedGemma 1.5 text generator
and the unchanged V6 prompt, decoding, and verifier. The generator receives the
indication, question, and selected report findings/impression; image pixels are
not passed to the generator.

The QA analysis reports raw Token-F1, verified Token-F1, support, abstention,
revision, and exact match as descriptive secondary outcomes. It does not alter
the retrieval selection, H1/H2 decision, or V6 conclusion. Automatic verifier
signals are not physician-adjudicated correctness.

## 11. Statistical plan and compute accounting

The primary unit is the case. All three questions from a case remain together
in paired bootstrap resampling. Report, at minimum, MRR, Hit@1/5/10,
target-outside-shortlist rate, case count, question count, and the H1 paired
case-grouped bootstrap interval. H2 reports aligned MRR, shuffled mean, median,
range, `k`, and plus-one `p`.

The V7 statistics adapter must report the primary equal-question/case-grouped
metric and the source-balanced sensitivity metric without replacing the
existing V6 statistical artifacts. It must also report candidate-pair counts,
queries excluded from gradient optimization because the target was outside
Top-100, training wall time, inference latency per query, peak GPU memory, and
the number of text/image score computations. These are engineering costs, not
clinical utility measurements.

## 12. Failure, deviation, and stopping rules

Technical failures such as OOM, process termination, or transient cache access
may be rerun with the identical frozen configuration and recorded. A malformed
or unreadable frozen case is a data-integrity deviation; it must not be silently
replaced by the next hash-ranked ID.

No confirmation outcome may trigger a new model, feature, prompt, threshold,
cohort, or generator. If the adaptive learner cannot be trained because no
valid in-shortlist positive pairs exist in a development fold, record V7 as a
failed development extension and retain V6 as the principal study. Do not
relax the shortlist rule after observing this failure.

The V7 development cycle stops after the final learner, global alpha, gate
threshold, frozen feature scaler, confirmation protocol, and confirmation
cohort-generation rule are recorded. There is no V7.1 tuning cycle inside this
protocol.

## 13. Reproducibility and artifact requirements

Before confirmation execution, commit:

- this protocol;
- the matching machine-readable config;
- the V7 technology reuse audit;
- the prior-use audit and development manifest;
- the development decision record;
- the V7 confirmation protocol;
- the final frozen learner checkpoint, feature scaler, and serialized feature
  schema;
- the final 240-case manifest and SHA-256 fingerprints.

The public repository may contain lightweight manifests, configuration,
metrics, and code. Raw radiology text, image pixels, model weights, generated
rows, and restricted caches remain subject to the existing repository policy.
No secrets or online-service credentials may enter Git.

## 14. Claim limits

V7 can support only a bounded methodological statement about adaptive fusion on
a same-source, case-ID-disjoint, closed-set paired-report benchmark. It cannot
support claims of clinical correctness, diagnostic ability, patient-level
independence, external validity, or deployment safety. Independent human
evaluation remains future work and must not be fabricated.
