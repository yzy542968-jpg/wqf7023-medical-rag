# V9 Retrieval Confirmation Protocol

## 1. Status

This protocol freezes the V9 retrieval confirmation before test-image
encoding, ranking, metric calculation, or shuffled-control execution. The
752-case test manifest was instantiated earlier under the full-source split
protocol but has not been used for V9 outcomes.

## 2. Fixed data

```text
historical candidate bank: 2,608 report-bearing train cases
test queries:                 752 report-bearing cases
questions per test case:        3 fixed forms
strict history-untouched subset: 262 cases nested in test
```

The candidate bank contains no test study. Under documented source-design
patient uniqueness, it also contains no test patient, although released
patient identifiers were unavailable for independent verification.

Every test query remains in the analysis unless a documented source-integrity
failure makes execution impossible. No reserve or outcome-driven replacement
is permitted.

## 3. Frozen systems

```text
R0 BM25 text
R1 MedSigLIP image-image
R2 MedSigLIP image-report normalized mean
R3 fixed multimodal: BM25 0.25 / image-image 0.50 / image-report 0.25
R4 learned MLP: 9 -> 32 -> 16 -> 1, 865 parameters
```

R4 checkpoint SHA-256:

```text
8afa68a48de9d6c9128d190f1368d0d45d41a958e5eb12787d7e725e7eb09efa
```

All encoders and the checkpoint are frozen. No test metric can change a model,
feature, chunk policy, fusion weight, question, threshold, or tie rule.

## 4. Outcomes

Primary outcome:

```text
case-grouped equal-question nDCG@10
```

Secondary outcomes are nDCG@1/5, Recall@1/5/10, MRR at gain `>= 0.50`, results
by question type, zero-binary-relevant-case count, strict-subset sensitivity,
latency, and peak GPU memory.

Continuous nDCG remains primary because binary relevance can be absent for a
query under the frozen threshold.

## 5. Primary hypothesis and inference

Primary hypothesis:

> R4 learned paired reranking improves test nDCG@10 over R1 image-image
> retrieval.

The paired difference is averaged within case across the three questions.
A 10,000-iteration case bootstrap with seed `7031` produces the percentile
95% confidence interval. Confirmed superiority requires the lower confidence
bound to be greater than zero. A positive point estimate with an interval
crossing zero is described only as a numerical improvement.

R4 versus R3 and all secondary metrics are reported without replacing the
primary comparison.

## 6. Shuffled-image alignment control

Test cases are ordered by:

```text
SHA256("v9-shuffle-order|7031|" + canonical_case_id), canonical_case_id
```

One hundred unique wrong-image assignments use cyclic shifts 1 through 100 of
that ordered list. Since test size is 752, every assignment is fixed-point
free and maps each query to exactly one other test case's complete image-view
set.

For every assignment, the complete frozen R4 visual state is recomputed:

- image-image scores;
- image-report scores;
- score normalization;
- ranks/reciprocal-rank features;
- MLP output and ranking.

BM25 query scores remain attached to the original indication/question. Aligned
features or alpha/visual state cannot be reused.

The plus-one randomization p-value is:

```text
(1 + count(shuffled nDCG@10 >= aligned nDCG@10)) / 101
```

Alignment dependence requires aligned R4 nDCG@10 to exceed the shuffled
distribution with `p <= 0.05`. The full shuffled distribution, mean, standard
deviation, and 2.5/97.5 percentiles are reported.

## 7. Sensitivity subset

The 262 project-history-untouched test cases are evaluated as a predefined
sensitivity subset. Their estimates are not used to select or retune the
system and are not promoted to a separate confirmatory hypothesis family.

## 8. Failure and rerun policy

Technical OOM, process crash, or transient file-access failure may be rerun
under the identical frozen configuration. A data-integrity failure is reported
as a protocol deviation and cannot trigger case replacement. Confirmation
results cannot trigger development changes or a second test run with altered
parameters.

Large vectors, score matrices, qrels, per-question rows, report text, and image
pixels remain local. Public artifacts contain aggregate results, fingerprints,
model hashes, and documented deviations only.

## 9. Claim boundary

Confirmation supports only retrospective report-derived similarity and
retrieval claims within OpenI/IU-Xray chest radiographs. It does not establish
physician-adjudicated clinical similarity, diagnostic safety, prospective
utility, or external generalization.

