# V7 Adaptive Fusion Development Decision Record

## 1. Status

The V7 development stage is complete under the protocol committed as
`2ec6dce`. This record freezes the development choices before any V7
confirmation case IDs are instantiated. It is not a confirmation result and it
does not modify V6.

The prior-use audit and development manifest were committed as `c513a8c`.
The development retrieval matrix was generated from 720 case IDs and 1,080
questions. The fourth confirmation block has not been generated, inspected, or
used for training or validation.

## 2. Development input integrity

| Artifact | Value |
|---|---|
| Source | `data/processed/openi_cases.jsonl` |
| Source SHA-256 | `56e367190396011d4d67f43e7e733389a8346890bf8729e82fb4326d063bbd68` |
| Development manifest | `data/splits/v7/v7_development_manifest.json` |
| Development manifest SHA-256 | `e4b2025b5ee6770be30878faedb2d040fbdad130f6ea4a9c8250e2eba739bd3d` |
| Retrieval rows | local-only `experiments/post_submission_v7/development_retrieval_rows.jsonl` |
| Retrieval rows SHA-256 | `c37729122f4a562727a6663da8d81d83452ea748c36f95836b72698313612f19` |
| Candidate cases | 720 across Train A, Train B, Validation |
| Questions | 1,080, three per target case |
| Report chunks | 1,555 |
| Image views | 1,409 |
| Confirmation IDs | not instantiated |

The initial V7 frame contained 1,239 eligible cases: 873 report-indexed normal,
349 report-indexed abnormal, and 17 report-index indeterminate. After the three
development blocks were instantiated, the remaining pre-confirmation frame
contained 519 eligible cases, of which 502 were stratifiable. This is a frame
summary, not a confirmation cohort.

The development blocks had zero case-ID overlap with formal prior-use cases,
zero overlap with the post-development confirmation selection frame, and zero
pairwise overlap with one another. Patient-level independence remains
unverifiable because no reliable patient identifier is present in the
processed source.

## 3. Frozen feature family

The learner used the 13 protocol-defined query/retrieval-state features:

```text
indication_present
question_token_count
indication_token_count
bm25_top1_normalized_score
bm25_top1_top2_margin
bm25_score_std
bm25_score_iqr
image_top1_normalized_score
image_top1_top2_margin
image_score_std
image_score_iqr
text_image_top1_agreement
text_image_spearman_correlation
```

The target ID, reference answer, qrels, target-in-shortlist flag, answer
source, generated answer, verifier output, and confirmation outcome were not
features. The scaler used by the final learner was fitted on Train A plus
Train B only.

## 4. Two-fold development selection

The target-outside-shortlist counts were 7 in Train A, 8 in Train B, and 6 in
Validation. These queries were excluded from pairwise gradient construction
only; they remained in every retrieval evaluation. The final merged training
pair set excluded 15 such queries and retained all corresponding rows for
later evaluation.

The development grid used AdamW, pairwise logistic loss, batch size 64, seed
7026, maximum 100 epochs, patience 10, and minimum holdout-MRR improvement
`0.0001`. Hard negatives were the fixed union of the top 20 non-target BM25
shortlist candidates and top 20 non-target image-score candidates, capped at
40 per query.

| Learner | Learning rate | Weight decay | Mean cross-fold MRR | Fold A epoch | Fold B epoch |
|---|---:|---:|---:|---:|---:|
| Best linear | 0.001 | 0.0 | 0.644749 | 1 | 10 |
| Best MLP | 0.0003 | 0.0 | 0.641857 | 1 | 1 |

The MLP minus linear difference was `-0.002892`. Under the inclusive protocol
rule that selects linear when the MLP gain is at most `0.005`, the selected
learner is:

```text
learner:       linear sigmoid alpha model
parameters:    14
learning rate: 0.001
weight decay:  0.0
final epochs:  6
seed:          7026
```

The final epoch is deterministic round-half-up from the two selected fold
epochs. No foundation-model parameters were updated.

## 5. Validation selection

The prespecified global-alpha grid selected:

```text
alpha* = 0.52
Validation primary MRR = 0.640774
```

The final adaptive learner, refit on Train A plus Train B for six epochs, had
Validation primary MRR `0.631742`. The development-only difference was
approximately `-0.009032` for adaptive minus global fusion. This is not the H1
confirmation result and must not be described as a final failure or success of
V7. It is recorded because development outcomes cannot be hidden or used to
justify another tuning cycle.

The secondary simple gate selected a threshold of `0.85` under the frozen
validation rule. Its Validation MRR was `0.640774`; because this gate effectively
used the selected global alpha across the validation rows, it is retained only
as a descriptive comparator and is not a primary V7 contribution.

## 6. Frozen development artifacts

The model checkpoint and scaler remain local-only under repository policy:

- `experiments/post_submission_v7/v7_adaptive_fusion_final_checkpoint.pt`
  SHA-256 `ab75c54fefa2531fb98af500d733d517804434e0ee87bc687bb706d36a6143b7`;
- `experiments/post_submission_v7/v7_adaptive_fusion_feature_scaler.json`
  SHA-256 `45272a9d25029db0c01c81ea753db25b310c4512ecd13ff6576ccfe21cba0860`;
- local machine-readable result:
  `experiments/post_submission_v7/v7_development_decision.json`.

The code and lightweight decision record are reproducible from the committed
protocol, config, manifest, and scripts. Raw report text, image pixels,
per-question rows, caches, and model weights remain local.

## 7. Next authorized step

The following decisions are now frozen for the V7 confirmation protocol:

- linear adaptive learner;
- six final training epochs;
- final Train A plus Train B scaler;
- global `alpha*=0.52`;
- optional gate threshold `0.85`;
- fixed BM25/MedSigLIP shortlist and score policy;
- H1 and H2 metrics and pass/fail rules;
- MedGemma-only secondary QA transfer after retrieval freeze.

The next artifact is `docs/V7_CONFIRMATION_PROTOCOL.md` plus its matching
machine-readable config. Only after that protocol and config are committed may
the deterministic 240-case confirmation block be instantiated. No outcome,
prompt, threshold, feature, or model change is authorized before then.
