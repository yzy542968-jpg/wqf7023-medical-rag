# V10 Reranker Development Decision Record

Status: retrieval architecture decision complete; V10 Test not run.

## Inputs and eligibility

The experiment followed the frozen V10 development protocol and the fact-aware
reranker amendment. The candidate bank contained 2,506 technically eligible
Train cases. Pairwise fitting used 1,811 eligible queries, internal early
stopping used 366, and Validation contained 376 eligible queries. Eligibility
required the previously frozen image, report, and RadGraph artifacts and was
not determined from model outcomes.

Training produced 347,664 R4 comparison pairs and 521,496 R5 comparison pairs.
All target-report text, target facts, labels, and answers remained unavailable
to inference features.

## Frozen checkpoint audit

| Model | Seed | Best epoch | Internal nDCG@10 | Checkpoint SHA-256 |
|---|---:|---:|---:|---|
| R4 | 7041 | 9 | 0.331357 | `1ad78a0fa69d59d4c61b22954d36cb83ed9fbe962e9644cad5c7307a9c01856e` |
| R5 | 7041 | 3 | 0.341724 | `b153a5d4c83cd5e7a75f4c6471f3fa2477b51bffb672c453844e273fab73c135` |
| R5 | 7042 | 4 | 0.351902 | `27ba1c374d57390eb1a8d177f68940939f28942be67aae7b3fdc46ee22d6612f` |
| R5 | 7043 | 4 | 0.340806 | `977ae53f33c4b2f6ffc122c947c85a952fc495cdc7d73ec4ace290b7b4a21ba1` |
| R5 | 7044 | 1 | 0.347103 | `e0e3e3f772f5279f359cce910894ab5af68b09887a785027e3062814987b0864` |
| R5 | 7045 | 1 | 0.345734 | `68a71565d79b9aacdf8656b9837028657a80c2fed6628ffdbad84cfc5c3817a6` |

## Validation result

| Retrieval condition | Validation nDCG@10 |
|---|---:|
| MedSigLIP image-image diagnostic | 0.329753 |
| R4 nine-feature reranker | 0.340255 |
| R5 seed 7041 | 0.347932 |
| R5 seed 7042 | 0.357957 |
| R5 seed 7043 | 0.346918 |
| R5 seed 7044 | 0.357551 |
| R5 seed 7045 | 0.356228 |
| R5 five-seed score ensemble | **0.353909** |

The best single seed exceeded the ensemble by 0.004048, below the frozen
0.005 ensemble-degradation tolerance. The ensemble was therefore selected.
The selected R5 ensemble exceeded R4 by 0.013654, above the frozen 0.005
promotion margin. R5 is promoted as the V10 retrieval architecture.

## Locked decision

The five R5 checkpoints and score-mean ensemble are frozen for subsequent V10
calibration, confirmation retrieval, shuffled-image controls, and QA. No seed,
feature, hard-negative rule, architecture, or training hyperparameter may be
changed in response to later outcomes. The local per-query Validation file has
SHA-256
`602bb0ed29b7fa3baa14a3b8d9674c9eaf4cc15a97e5428c523748c8744d96dc`.
It is retained locally because it contains detailed case-level rankings.

This development result does not establish Test performance, clinical safety,
or external generalization.
