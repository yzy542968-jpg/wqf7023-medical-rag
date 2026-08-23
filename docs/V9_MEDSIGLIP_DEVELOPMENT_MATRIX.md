# V9 MedSigLIP Retrieval Development Matrix

## 1. Status

This document freezes the V9 MedSigLIP development matrix before any V9
MedSigLIP validation outcome is generated or inspected. The BM25 development
baseline has already been run and is not changed. V9 test retrieval remains
unexecuted.

## 2. Fixed source and model

```text
shared historical bank: 2,608 report-bearing train cases
validation qrel frame:    374 report-bearing validation cases
test queries encoded:       0 during development

encoder:  google/medsiglip-448
revision: 9cea28a1a1195f665105faa6e8544c112fd960a4
precision: float16 CUDA inference; float32 normalized saved embeddings
foundation parameters updated: no
```

Every system uses the same 2,608-case bank. Image-only systems may not add the
23 empty-report train cases excluded by the RadGraph preprocessing protocol.

## 3. Image representation

Each readable frontal/lateral source view is encoded independently. View
embeddings are L2-normalized, averaged within study, and L2-normalized again.
This deterministic normalized-mean policy applies to query and historical
studies. No view is selected based on relevance outcomes.

## 4. Report representation

Findings and impression are split with the frozen V6 sentence-aware
preprocessing into section-prefixed chunks of at most 64 MedSigLIP tokens.
Every chunk is encoded and L2-normalized.

Two report-scoring policies are evaluated:

```text
mean: normalized mean of all chunk embeddings, then image-report cosine
max:  maximum image-to-chunk cosine within the candidate report
```

Selection uses validation case-grouped equal-question nDCG@10. The maximum
policy is selected only if `max - mean >= 0.005`; otherwise normalized mean is
selected for aggregation simplicity. Both results are retained.

## 5. Retrieval components

The independently evaluated components are:

```text
R0: BM25(indication + fixed question, historical report)
R1: MedSigLIP target-image to historical-image cosine
R2: MedSigLIP target-image to historical-report score
```

R1 and R2 rankings are case-dependent but question-independent; their metrics
are repeated over the three frozen question forms only to preserve the common
case/question analysis structure.

Each component score is independently min-max normalized across the complete
2,608-case bank for a query. Constant channels map to zero. Fusion ranks by
descending fused score and then canonical case ID.

## 6. Fixed fusion grid

All nonnegative weight triples in increments of 0.25 that sum to 1.00 are
evaluated over `(BM25, image-image, image-report)`. Pure systems are retained
as diagnostics. The fixed multimodal candidate set requires:

```text
BM25 weight > 0
image-image weight + image-report weight > 0
```

The selected fixed multimodal condition maximizes validation nDCG@10. Among
candidates within 0.005 of the maximum, the deterministic simplicity rule is:

1. prefer the largest BM25 weight;
2. prefer the smallest absolute difference between image-image and
   image-report weights;
3. prefer the lexicographically smallest weight tuple.

This rule intentionally requires a material multimodal gain before reducing
the contribution of the interpretable text baseline.

## 7. Metrics and development boundary

Primary selection metric:

```text
case-grouped equal-question nDCG@10
```

Secondary development diagnostics are nDCG@1/5, binary Recall@1/5/10, MRR at
the frozen gain threshold 0.50, zero-relevant-case count, runtime, and peak GPU
memory. Continuous nDCG remains primary because some validation cases have no
candidate above the binary threshold.

No model, chunk policy, or fusion weight is selected using QA outcomes. No
test query is encoded, ranked, or evaluated in this stage. All per-query rows
and embeddings remain local; only aggregate outcomes and hashes may be
committed.

