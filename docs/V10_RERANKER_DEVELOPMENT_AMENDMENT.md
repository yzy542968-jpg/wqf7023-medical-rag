# V10 Fact-Aware Reranker Development Amendment

## Status

This amendment freezes implementation details left open by the parent V10
development protocol. It is committed before V10 embedding outcomes, role
instantiation, gradient updates, or Validation rankings are inspected.

## Query roles

Train clusters are assigned by the first 64 bits of
`SHA256("v10-reranker-role|7041|" + cluster_id)`:

```text
[0.00, 0.70)  pairwise_fit
[0.70, 0.85)  internal_early_stop
[0.85, 1.00)  bank_only
```

Clusters remain intact. All report-bearing Train cases remain candidates;
only cases with readable images, formal RadGraph status `ok`, and a non-empty
report may act as fit/internal queries.

## R4 reproduction

R4 uses the V9 nine-feature order and architecture `9 -> 32 -> 16 -> 1`.
It is retrained from scratch on the cluster-disjoint V10 Train partition.

## R5 fact-aware model

R5 appends eight inference-available features:

```text
sentence_similarity_max
sentence_similarity_mean
fact_similarity_max
fact_similarity_mean
positive_fact_fraction
negative_fact_fraction
uncertain_fact_fraction
evidence_redundancy
```

The architecture is `17 -> 64 -> 32 -> 1`. Five deterministic seeds are
trained. Pair construction retains component-top and relevance-top/bottom
candidates and adds the 32 candidates with highest inference-component rank
but relevance gain below `0.25` as hard negatives.

## Optimization

Both models use weighted pairwise softplus loss, AdamW, learning rate 0.001,
weight decay 0.0001, batch size 4096, maximum 30 epochs, patience 5, and
minimum internal nDCG@10 improvement 0.0005.

R5 promotion requires Validation nDCG@10 at least 0.005 above V10 R4. The
five-seed mean is primary unless it trails the best single seed by at least
0.005 on Validation. No Test outcome may select a seed or ensemble.

