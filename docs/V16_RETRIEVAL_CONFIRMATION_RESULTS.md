# V16 Retrieval Confirmation Results

## Status and boundary

The frozen V12 retrieval winner was applied once to the 568 technically
eligible V10 Test cases after the V16 confirmation protocol commit. No Test
outcome was used to refit the ranker, alter the Reciprocal Rank Fusion (RRF)
sources, change Top-200 candidate generation, revise feature order, or select a
different model. The historical bank remained the 2,506 technically eligible
Train cases.

This is a held-out evaluation of the V12 method, not a globally untouched Test:
the same Test partition had previously been used by frozen V10 systems. The
three qrel variants are derived automatically from hidden OpenI reports and are
not physician judgments of clinical similarity.

## Primary comparison

| System | Combined qrel nDCG@10 | Label-only nDCG@10 | Fact-only nDCG@10 |
|---|---:|---:|---:|
| V10 R5, full Train bank | 0.55313 | 0.33326 | 0.33180 |
| RRF Top-200 candidate order | 0.54292 | 0.28605 | 0.28887 |
| RRF Top-200 then R5 reranking | 0.55363 | 0.33489 | 0.33239 |
| **RRF Top-200 then V12 LambdaMART** | **0.61590** | **0.37254** | **0.34507** |
| LambdaMART over the full Train bank | 0.54150 | 0.31632 | 0.29259 |

The V12 Top-200 plus LambdaMART system improved combined-qrel nDCG@10 over R5
by `+0.06277`, 95% case-bootstrap CI `[+0.05460, +0.07082]`. The direction was
also positive under the two prespecified construct sensitivities:

- label-only: `+0.03928`, CI `[+0.02450, +0.05443]`;
- fact-only: `+0.01326`, CI `[+0.00405, +0.02243]`.

All three intervals exclude zero. The appropriate conclusion is that the
Validation-selected candidate-generation plus learned-reranking method
transferred to Test under multiple report-derived relevance definitions.
It does not establish physician-rated retrieval quality.

## Spectrum sensitivity

For the combined qrel, V12 LambdaMART achieved nDCG@10 `0.64897` among 195
report-indexed normal cases, `0.59784` among 359 report-indexed abnormal cases,
and `0.61850` among 14 report-index-indeterminate cases. Corresponding V10 R5
values were `0.57158`, `0.54268`, and `0.56391`. The improvement therefore was
not confined to the larger abnormal stratum, although the 14-case
indeterminate estimate is descriptive and imprecise.

The large normal/abnormal contrast under the label-only qrel is a property of
the automated construct and source labels, not evidence that one clinical
population is inherently easier. The combined and fact-only analyses remain
necessary to expose that construct sensitivity.

## Negative and mechanism results

Candidate generation alone was worse than R5 (`-0.01021`, combined-qrel CI
`[-0.01579, -0.00480]`). R5 reranking within the RRF Top-200 frame recovered
the R5 result but did not materially exceed it (`+0.00050`, CI
`[-0.00016, +0.00115]`). The gain therefore depends on the V12 learned ranker,
not on RRF ordering alone.

Applying the same LambdaMART model to the complete Train bank also performed
worse than R5: combined-qrel difference `-0.01163`, CI
`[-0.02026, -0.00288]`. This negative result shows that the Top-200
multi-source candidate frame is part of the successful method. The learned
ranker does not generalize as a universal full-bank scoring function.

## Reproducibility

- Test cases: `568`
- Train historical cases: `2,506`
- case artifact SHA-256: `56e367190396011d4d67f43e7e733389a8346890bf8729e82fb4326d063bbd68`
- split SHA-256: `b4c1b091c3dbff0399d07c8350f4d8d68ce8ce52e0157dcc96f46af8c8baa7b3`
- LambdaMART SHA-256: `8c83d6188daa66939ae6a7865c14eada827c4cf625cc0314beaa4988ec2f086c`
- aggregate result: `experiments/v16_confirmation/v16_test_rankings.json`
- per-query rows remain local because they include report-derived details.

## Claim boundary

nDCG@10 quantifies ranking under automated report-derived qrels. It is not
diagnostic accuracy, physician agreement, patient benefit, clinical safety, or
external validation. Independent clinical review and authorized MIMIC-CXR
replication remain Future Work.
