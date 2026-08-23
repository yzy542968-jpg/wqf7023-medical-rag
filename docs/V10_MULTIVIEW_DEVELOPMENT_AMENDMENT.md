# V10 Multi-view Development Amendment

Status: frozen before multi-view training and Validation outcome inspection.

This amendment operationalizes Section 10 of the frozen V10 development
protocol. It does not alter the V9 freeze, the V10 cluster split, the retrieval
qrel, or the Test boundary.

## Candidate bank and data roles

- The candidate bank is the eligible V10 Train partition.
- Gradient updates use only the existing `pairwise_fit` Train role.
- Early stopping uses only the existing `internal_early_stop` Train role.
- V10 Validation is used once to compare the three frozen policies below.
- Calibration and Test are not read during multi-view development.

## Frozen policies

All view embeddings are the L2-normalized MedSigLIP embeddings from the frozen
V10 embedding artifact. Candidate image embeddings use the normalized mean of
all available candidate views. Candidate report embeddings use the frozen
chunk-mean report representation.

For each query case, two component score vectors are formed: query-image to
candidate-image and query-image to candidate-report. Each component is
independently min-max normalized over the candidate bank and the final score is
their unweighted mean.

1. `mean`: L2-normalize the mean of all query-view embeddings before scoring.
2. `per_view_max`: score every query view separately and take the per-candidate
   maximum within each component before component normalization and fusion.
3. `learned_attention`: a single linear `1152 -> 1` attention head produces a
   softmax weight over the views of a query case. The weighted query embedding
   is L2-normalized before component scoring.

The attention head is trained with weighted pairwise softplus ranking loss.
Pairs are selected from the eight highest- and eight lowest-gain candidates
under the frozen V10 qrel, with a minimum gain difference of `0.05`. The
foundation encoder and candidate embeddings remain frozen. Five deterministic
seeds (`7051` through `7055`) are trained with AdamW, internal early stopping,
and the hyperparameters in `config/v10_multiview_development.json`.

The mean of the five attention score vectors is the learned-attention result.
A single seed is not selected from Validation. This removes a seed-selection
degree of freedom.

## Decision rule

Primary metric is case-averaged nDCG@10 under the existing V10 combined qrel.
The learned-attention policy is promoted only if it exceeds `mean` by at least
`0.005`. Otherwise `per_view_max` is promoted only if it exceeds `mean` by at
least `0.005`. If neither condition is met, `mean` remains frozen. This order is
prespecified and Test cannot alter it.

The selected view policy is subsequently used by the frozen retrieval model,
confidence calibrator, shuffled-image control, and compact QA pipeline.

