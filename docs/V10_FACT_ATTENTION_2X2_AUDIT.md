# V10 Facts x Attention 2x2 Audit

## Scope

This document records a validation-only attribution audit over the already
frozen V10 checkpoints. It does not retrain R4/R5, inspect the V10 Test
partition, or change the frozen V10 confirmation result.

The four cells are:

| Reranker | Image view |
|---|---|
| R4, nine-feature non-fact-aware reranker | mean image embedding |
| R4, nine-feature non-fact-aware reranker | learned multiview-attention embedding |
| R5, fact-aware reranker | mean image embedding |
| R5, fact-aware reranker | learned multiview-attention embedding |

The complete machine-readable output is
`data/splits/v10/v10_fact_attention_2x2_summary.json`, with per-row audit
records in `experiments/v10_publication/v10_fact_attention_2x2_rows.jsonl`.

## Frozen validation result

The run used 2,506 Train candidates, 376 Validation cases and 1,128
findings/impression/acute rows. Embeddings were L2-normalized using the same
policy as the V10 R5 integration runner.

| Condition | nDCG@10 |
|---|---:|
| R4 + mean image | 0.340255 |
| R4 + attention image | 0.346206 |
| R5 + mean image | 0.353909 |
| R5 + attention image | 0.358540 |

The arithmetic contrasts are:

- Average fact-aware minus non-fact-aware effect: `+0.012994`.
- Average attention-view minus mean-view effect: `+0.005291`.
- Fact-by-attention interaction contrast: `-0.001319`.

These are descriptive contrasts on the Validation partition. They show that
both frozen components contribute positive numerical changes in this audit,
while the small negative interaction does not support a claim that the two
components amplify one another. The design does not establish causal effects
outside this fixed checkpoint comparison.

## Interpretation boundary

The relevance target remains the frozen report-derived V10 construct, not
physician-adjudicated clinical similarity. Patient-level independence cannot
be verified from the processed OpenI source. No claim is made about diagnostic
accuracy, safety, clinical utility or external validity. The audit is useful
for explaining the V10 architecture, but it does not replace the frozen V10
Test result or independent clinical review.
