# V7 Adaptive Multimodal Fusion: Confirmation Protocol

## 1. Protocol status

This protocol freezes the V7 confirmation configuration after the V7
development decision record was completed and before the deterministic
confirmation case IDs are instantiated. It is version controlled but is not a
formal or externally timestamped preregistration.

The development protocol, prior-use audit, development manifest, and
development decision record are frozen. The final learner, feature scaler,
global fusion weight, gate threshold, retrieval runtime, statistical plan, and
failure policy cannot change in response to confirmation outcomes.

At the time of this protocol:

- V6 artifacts remain immutable;
- V7 confirmation case IDs have not been generated or inspected;
- no confirmation retrieval, QA, shuffled-image, or outcome file exists;
- the next authorized operation is one deterministic cohort-builder run;
- technical reruns are allowed only under the identical configuration.

## 2. Objective and hypotheses

V7 tests whether the frozen adaptive fusion policy improves closed-set paired
report retrieval over the validation-selected global fusion weight on a new
case-ID-disjoint confirmation cohort.

### H1: adaptive fusion benefit

For question `q` and candidate report `i` in the BM25 Top-100 shortlist:

```text
S_i(q) = alpha_q * T_i(q) + (1 - alpha_q) * V_i(q)
```

`T_i` and `V_i` are the independent within-shortlist min-max normalized BM25
and MedSigLIP scores. `alpha_q` is produced by the frozen linear learner trained
on Train A plus Train B. The comparator is the frozen validation-selected
global `alpha*=0.52`.

H1 passes only if the lower bound of the 95% paired case-grouped bootstrap
interval for:

```text
MRR(adaptive) - MRR(global alpha*)
```

is greater than zero. The bootstrap uses 5,000 resamples and seed `7026`.

### H2: alignment-specificity control

The same frozen adaptive learner is evaluated with the correctly aligned image
and with 100 deterministic unique fixed-point-free shuffled-image assignments.
For each shuffled assignment, all image-derived candidate scores, score
distributions, cross-modal features, and `alpha_q` values are recomputed. The
aligned `alpha_q` is never reused for a shuffled image.

If `k` shuffled MRR values are greater than or equal to the aligned MRR:

```text
p = (1 + k) / 101
```

H2 passes when `p <= 0.05`.

### Interpretation matrix

| H1 | H2 | Permitted interpretation |
|---|---|---|
| Pass | Pass | Positive evidence for adaptive fusion with alignment-specific visual contribution |
| Pass | Fail | Adaptive improvement observed, but alignment-specificity is not confirmed |
| Fail | Pass | Correctly aligned visual information is useful, but adaptive fusion is not superior to global fusion |
| Fail | Fail | V7 adaptive-fusion claim is unsupported; V6 remains the principal completed result |

If the adaptive point estimate exceeds global fusion but its interval crosses
zero, report a numerical difference only. No result is promoted because of its
point estimate alone.

## 3. Confirmation source frame

The source is the same OpenI/IU-Xray processed file used in development:

```text
data/processed/openi_cases.jsonl
SHA-256: 56e367190396011d4d67f43e7e733389a8346890bf8729e82fb4326d063bbd68
```

After removing formal prior-use cases, the V5 and V6 cohorts, and the three V7
development blocks, the audited pre-confirmation frame contains:

| Category | Count | Case-ID fingerprint |
|---|---:|---|
| Eligible remaining | 519 | `98f52612031a9ae6efcd5e75610b2a4a44baa67fbabcc1a91f12a89e7af737a7` |
| Report-indexed normal | 357 | `1877873d15a272aeb3852b72fbefb5e97544f90d1363ab4a217af37be0b06a71` |
| Report-indexed abnormal | 145 | `e6c25b2e1f6eacf1ebdb410c9f51bbe9ceeabfe4f7664a07e7f462406dfe6bc9` |
| Report-index indeterminate | 17 | `3a51b0809b46becafd3fe9786e12acadc9af60e4b255b9fc73568793bd7f1211` |
| Stratifiable normal + abnormal | 502 | `800bb97469b1715f6e17c3c22b583ed3f74f5ebc5cd75b8328b14014b639d318` |

`report-indexed normal` means normalized `problems == "normal"`.
`report-indexed abnormal` means a non-empty value other than `normal` and
`no indexing`. `report-index indeterminate` means `no indexing` and is excluded
from primary stratification. These are dataset-index labels, not clinical
adjudications.

Only case-ID disjointness is asserted. Patient-level independence cannot be
verified because the processed source lacks a reliable stable patient or
subject identifier.

## 4. Deterministic confirmation cohort generation

The confirmation candidate pool is generated only after this protocol and its
matching config are committed. From the audited stratifiable frame, select:

```text
172 report-indexed normal
 68 report-indexed abnormal
240 total cases
```

Assign roles within the selected strata:

```text
Targets:       86 normal + 34 abnormal = 120
Distractors:   86 normal + 34 abnormal = 120
```

Case IDs are canonicalized as `str(value).strip()`, encoded as UTF-8, and
ordered by lowercase hexadecimal SHA-256:

```text
SHA256("v7-selection|7026|" + canonical_case_id)
```

Role assignment uses the domain-separated ordering:

```text
SHA256("v7-assignment|7026|" + canonical_case_id)
```

Within each selected stratum, the first 86 normal and 34 abnormal IDs become
targets; the remainder become distractors. Collection fingerprints are based
on sorted unique canonical IDs joined by LF with no trailing LF before SHA-256.

The builder must verify source hashes, frame hashes, image readability, report
fields, counts, unique IDs, role coverage, target-question construction, and
zero overlap with development and prior-use manifests. A malformed case is a
protocol deviation, not a reason for silent replacement.

## 5. Frozen retrieval runtime

### 5.1 BM25 and shortlist

BM25 uses `k1=1.5` and `b=0.75` over the 240-case candidate pool. The query is:

```text
Clinical indication: {indication}
Question: {question}
```

Scores are ordered by descending score and ascending canonical case ID. Only
the first 100 candidates are eligible for image reranking. A target outside
this shortlist remains a true retrieval failure and is reported; it is not
removed from confirmation evaluation.

### 5.2 MedSigLIP and score construction

MedSigLIP is frozen at:

```text
model:    google/medsiglip-448
revision: 9cea28a1a1195f665105faa6e8544c112fd960a4
```

Findings and impression are processed with the V6 sentence-aware chunks of at
most 64 tokenizer tokens, without overlap or silent truncation. Each image
view is L2-normalized, averaged by case, and L2-normalized again. The candidate
report score is the maximum image-to-report-chunk cosine similarity.

BM25 and image scores are independently min-max normalized within the Top-100
shortlist. The fixed historical comparator is `alpha=0.50`; the primary V7
comparator is the frozen global `alpha*=0.52`.

### 5.3 Frozen adaptive learner

The final learner is a 14-parameter linear sigmoid model trained with:

```text
learner:       linear sigmoid
learning rate: 0.001
weight decay:  0.0
epochs:        6
optimizer:     AdamW
loss:          pairwise logistic ranking loss
seed:          7026
```

Its feature scaler was fitted on Train A plus Train B only. The local checkpoint
and scaler hashes are recorded in `docs/V7_DEVELOPMENT_DECISION_RECORD.md` and
the machine-readable development decision file. No foundation-model parameter
is updated in confirmation.

The optional secondary gate uses threshold `0.85`: if the normalized BM25
Top-1 margin is at least the threshold it uses text-only `alpha=1.00`, otherwise
it uses global `alpha*=0.52`. The gate is diagnostic only and is not used for
the H1 decision.

## 6. Confirmation questions and metrics

Each of 120 target cases produces the same three deterministic report-derived
questions used in V6: findings, impression, and summary. This yields 360
questions. The candidate pool contains all 240 reports for every target query.

The primary retrieval metric is case-averaged MRR with equal question-type
weight within each case. Secondary metrics are Hit@1, Hit@5, Hit@10, and target
outside-shortlist rate. All three questions from one case remain together in
paired resampling.

Because impression and summary use the same frozen impression reference, report
the prespecified source-balanced sensitivity metric:

```text
M_source-balanced = 0.50 * M_findings
                    + 0.25 * M_impression
                    + 0.25 * M_summary
```

It is secondary and cannot change H1 or H2.

## 7. Shuffled-image controls

For control index `j` from 0 to 99, order the 120 target IDs by:

```text
SHA256("v7-shuffle-order|7026|" + str(j) + "|" + canonical_case_id)
```

Map each target to the image of the next ID in the cyclic order. The mapping
must be a unique fixed-point-free derangement. Under every mapping, recompute
image scores, image dispersion, image rank statistics, text/image agreement,
and adaptive alpha. The same frozen learner and all other retrieval settings
are then reused.

## 8. Secondary MedGemma QA transfer

After the primary retrieval run is frozen, execute a secondary QA transfer using
the frozen V6 MedGemma 1.5 generator and unchanged V6 prompt, decoding, and
verifier. Compare the Top-1 report selected by BM25, global fusion, and adaptive
fusion. The generator receives indication, question, and selected report
findings/impression, not image pixels.

Report raw Token-F1, verified Token-F1, support rate, abstention rate, revision
rate, and exact match descriptively. These outcomes cannot select a retrieval
model or change H1/H2. Automated verification and reference consistency are not
clinical correctness.

## 9. Deviations and stopping

OOM, process termination, and transient cache failures may be rerun under the
identical frozen configuration and recorded. No case may be replaced because
of a low score, an unfavorable qualitative output, or an outcome pattern.

No new model, feature, prompt, threshold, generator, evaluator, or data source
may be added. If the confirmation execution fails technically, record the
deviation and rerun without changing the configuration. If V7 does not pass
H1 or H2, report the negative or mixed result and retain V6 as the principal
completed evidence.

## 10. Claim limits

V7 can support only a bounded methodological claim about adaptive fusion on a
same-source, case-ID-disjoint, closed-set paired-report benchmark. It cannot
support clinical correctness, diagnosis, patient-level independence, external
validity, clinical utility, or deployment safety claims. Independent human
evaluation remains future work and is not represented by automated metrics.
