# V10 Multi-view Development Decision Record

Status: multi-view policy selected; V10 Test not run.

## Technical rerun record

The first execution exposed a leave-one-out implementation fault: the Train
self-match had a non-finite gain but its retrieval score was not masked during
internal early-stop evaluation. This produced undefined internal nDCG values,
left the attention heads at their initial states, and could not support a model
decision. No Calibration or Test data were read. Commit `6e0a47a` restored the
same self-exclusion rule already used by R4/R5, after which the unchanged
frozen configuration was rerun. Only the corrected run is authoritative.

## Corrected run

The candidate bank contained 2,506 eligible Train cases. Attention training
used 1,811 pairwise-fit queries, internal early stopping used 366 Train-role
queries, and policy comparison used 376 eligible Validation queries.

| Policy | Validation nDCG@10 | Difference from mean |
|---|---:|---:|
| normalized mean | 0.320920 | reference |
| per-view maximum | 0.298518 | -0.022401 |
| five-seed learned attention | **0.349622** | **+0.028702** |

The learned-attention improvement exceeded the frozen 0.005 promotion margin.
It is therefore the selected query-view policy. Per-view maximum is retained as
a negative diagnostic and is not used downstream.

## Checkpoint audit

| Seed | Best epoch | Internal nDCG@10 | Checkpoint SHA-256 |
|---:|---:|---:|---|
| 7051 | 4 | 0.341244 | `88f2274fa6fcc8e8d9d4c8ef1b1183fa9a2f51d72dc09a53850080c434043720` |
| 7052 | 4 | 0.341276 | `6733979ccbb645f0894f7c4f42a7835f5a45d65fae3816976038709273752a42` |
| 7053 | 4 | 0.341201 | `ccc898d8fcaa155a1dbef4438dc9da4deeb1a030d599589dea71fda58ea99ff6` |
| 7054 | 4 | 0.341614 | `326b3ba5be96fa51ec6832fbee4c8675574269b50c54372460080042a721ac4a` |
| 7055 | 4 | 0.341353 | `105e85ba54385db31a0fbc96783cc51b4743d2a5e2d5683e1781effc6e6b6a93` |

The local corrected Validation rows have SHA-256
`cb7ebe4b116a1cbdecac43367bcf30c483b47674162dc5e5b5e3241665e11c1d`.
They remain local because they contain detailed case-level rankings.

This result supports learned multi-view aggregation on the same-source
cluster-disjoint Validation set. It does not establish Test performance,
clinical superiority, or external generalization.

