# V9 Learned Paired Reranker Development Protocol

## 1. Status and objective

This protocol freezes R4 learned-reranker development after completion of the
R0-R3 validation matrix and before train-role instantiation, gradient updates,
or learned-reranker outcomes. V9 test queries remain unencoded and unexecuted.

The objective is not to fine-tune MedSigLIP. It is to train a small scoring
model that combines frozen BM25, image-image, and image-report retrieval state
using report-derived pairwise ranking supervision.

## 2. Training roles

The 2,608 report-bearing train cases are stratified by report-indexed spectrum
and assigned deterministically with:

```text
SHA256("v9-reranker-role|7030|" + canonical_case_id)
```

Predefined roles are:

| Role | Normal | Abnormal | Total |
|---|---:|---:|---:|
| Pairwise fit queries | 578 | 1,022 | 1,600 |
| Internal early-stop queries | 181 | 319 | 500 |
| Bank-only cases | 183 | 325 | 508 |
| **Total** | **942** | **1,666** | **2,608** |

Assignment occurs within each stratum by `(digest, canonical_case_id)`. The
role manifest and fingerprints are generated only after this protocol commit.

All 2,608 reports remain the final historical bank for V9 Validation and Test.
For fit/internal queries, the query study is masked from candidates and from
BM25 corpus statistics with exact leave-one-out document count, document
frequency, and average-document-length calculations.

## 3. Frozen inputs and features

The selected report policy is normalized mean chunk embedding. For each query,
the candidate feature vector contains:

```text
min-max normalized BM25 score
min-max normalized image-image score
min-max normalized image-report score
BM25 normalized reciprocal rank
image-image normalized reciprocal rank
image-report normalized reciprocal rank
findings-question indicator
impression-question indicator
acute-question indicator
```

Scores/ranks are calculated after query-study masking. Features may not contain
case IDs, filenames, raw report text, target labels/facts, target answers,
reference metrics, or QA outcomes. Target report-derived relevance is used
only as the training label.

## 4. Pair construction

For each fit query and fixed question:

1. form a deterministic union of the Top-32 candidates under BM25,
   image-image, image-report, and offline relevance gain;
2. add the Bottom-32 candidates under offline relevance gain;
3. choose the eight highest-gain and eight lowest-gain members of that union;
4. construct all ordered high-low pairs with gain difference at least 0.05;
5. weight each pair by its continuous gain difference.

Canonical case ID resolves every score/gain tie. Queries producing no valid
pair are retained in the audit but contribute no gradient update.

## 5. Models and optimization

Two prespecified CPU-trained scorers are compared:

```text
Linear:  Linear(9, 1)
MLP:     Linear(9, 32) -> ReLU -> Linear(32, 16) -> ReLU -> Linear(16, 1)
```

Both use:

```text
loss: weighted pairwise softplus(-(score_high - score_low))
optimizer: AdamW
learning rate: 0.001
weight decay: 0.0001
batch size: 4,096 pairs
maximum epochs: 30
seed: 7030
```

Early stopping is based only on the 500 internal queries' equal-question
nDCG@10, with patience 5 and minimum improvement 0.0005. The best checkpoint
for each architecture is retained. MedSigLIP, RadGraph, and BM25 have no
trainable parameters.

## 6. Architecture selection and promotion

The two frozen best checkpoints are evaluated once on the 374-case V9
Validation qrel frame. Architecture selection maximizes validation nDCG@10;
if the absolute linear/MLP difference is below 0.005, Linear is selected.

R4 is promoted over the selected fixed multimodal comparator only if:

```text
R4 validation nDCG@10 - fixed multimodal validation nDCG@10 >= 0.005
```

A separate stronger diagnostic records whether R4 also exceeds the best
single component, currently image-image, by at least 0.005. Failure of either
rule is reported as a no-go and does not trigger feature, pair, architecture,
or hyperparameter changes.

## 7. Data and outcome boundary

- Fit and internal roles use only the train partition.
- V9 Validation may select architecture and promotion status but cannot update
  parameters.
- QA outcomes cannot select the reranker.
- V9 Test is not encoded, ranked, or inspected in this stage.
- Checkpoints, pair tensors, score matrices, and per-query rows remain local.
- Public artifacts contain only role fingerprints, aggregate metrics, hashes,
  and training diagnostics.

This is a repository-timestamped development protocol, not a formal external
preregistration.

