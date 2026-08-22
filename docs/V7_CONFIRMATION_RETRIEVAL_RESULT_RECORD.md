# V7 Confirmation Retrieval Result Record

## 1. Status

The V7 primary retrieval confirmation run is complete under the frozen
protocol `4821f38` and the instantiated cohort freeze committed as `25a39d8`.
The run used one 240-case candidate pool, 120 target cases, and 360
report-derived questions. No case replacement or outcome-driven rerun was
performed.

This record reports retrieval only. Secondary MedGemma QA transfer is a
separate frozen follow-up and is not used to change the retrieval result.

## 2. Frozen input lineage

| Artifact | Value |
|---|---|
| Source cases | `data/processed/openi_cases.jsonl` |
| Source SHA-256 | `56e367190396011d4d67f43e7e733389a8346890bf8729e82fb4326d063bbd68` |
| Confirmation config | `config/v7_confirmation.json` |
| Config SHA-256 | `9c17552451db1a936bfa2b8510fb33ed032b00b18d312464df64caf5f8ca7d3f` |
| Confirmation cohort | `data/splits/v7/v7_confirmation_cohort.json` |
| Cohort SHA-256 | `7ed42bfc4851350c767f631d744d0306ee9ac5a406a3b74ceb75a568ceb89c65` |
| MedSigLIP | `google/medsiglip-448`, revision `9cea28a1a1195f665105faa6e8544c112fd960a4` |
| Adaptive checkpoint SHA-256 | `ab75c54fefa2531fb98af500d733d517804434e0ee87bc687bb706d36a6143b7` |
| Feature scaler SHA-256 | `45272a9d25029db0c01c81ea753db25b310c4512ecd13ff6576ccfe21cba0860` |
| Retrieval rows | local-only `experiments/post_submission_v7/v7_confirmation_retrieval_rows.jsonl` |
| Retrieval rows SHA-256 | `ca799a70e594983a8237e9bb18a67e226ce247aac79a6bac40e3aed2c42f0753` |

## 3. Retrieval outcomes

| System | MRR | Hit@1 | Hit@5 | Hit@10 |
|---|---:|---:|---:|---:|
| BM25 text-only | 0.590420 | 0.5167 | 0.6694 | 0.7333 |
| Image-only within BM25 Top-100 | 0.129858 | 0.0639 | 0.1694 | 0.2306 |
| Fixed alpha 0.50 | 0.609390 | 0.5528 | 0.6556 | 0.7056 |
| Global alpha 0.52 | 0.613370 | 0.5583 | 0.6583 | 0.7111 |
| Adaptive alpha_q | 0.601897 | 0.5361 | 0.6639 | 0.7222 |

The target-outside-shortlist rate was `1.9444%` for all conditions because the
same BM25 Top-100 availability boundary was used. Image reranking did not
attempt to rescue a target excluded by BM25.

## 4. H1 result: adaptive versus global fusion

The paired case-grouped difference was:

```text
adaptive alpha_q - global alpha*=0.52 MRR = -0.011473
95% case-grouped bootstrap CI = [-0.026769, +0.003109]
case count = 120
bootstrap resamples = 5,000
```

The lower confidence bound was not greater than zero. **H1 does not pass.**
The point estimate is lower for adaptive fusion, and the interval crosses zero;
the permitted conclusion is that the frozen adaptive learner did not establish
superiority over the validation-tuned global fusion weight on this confirmation
cohort.

This is not a reason to tune the learner again. The development record already
showed a negative adaptive-versus-linear selection signal, and the confirmation
result is reported unchanged under the frozen protocol.

## 5. H2 result: aligned image versus shuffled images

The complete frozen adaptive system was rerun under every shuffled image
assignment, including all image-derived features and the resulting `alpha_q`.

```text
Aligned adaptive MRR       = 0.601897
Shuffled mean MRR          = 0.595331
Shuffled median MRR        = 0.594932
Shuffled range             = [0.587374, 0.602391]
Shuffles >= aligned        = 1 / 100
Plus-one Monte Carlo p     = 0.019802
```

**H2 passes** because the plus-one value is at most `0.05`. The alignment
control supports the bounded claim that correct image pairing affected the
retrieval behavior of the frozen adaptive system. It does not show that
adaptive fusion was better than global fusion, and it is not evidence of
diagnosis or patient identification outside the indexed corpus.

## 6. Interpretation matrix outcome

The observed result is:

```text
H1 = fail
H2 = pass
```

Therefore V7 supports the narrower finding that aligned visual information
remained useful under the frozen adaptive pipeline, but it does not support a
positive claim that the learned query-conditional fusion policy improved over
the global validation-tuned weight. V6 remains the principal completed study
for the modernized alignment-specific retrieval and downstream QA result.

## 7. Runtime and audit boundary

The MedSigLIP embedding stage used 27.24 seconds and a peak allocated GPU
memory of approximately 1,763 MiB on the local RTX 5070 Laptop GPU. The 100
controls reused cached encoder outputs but recomputed their image score maps and
adaptive features. The result is a local engineering measurement, not a
production latency or energy claim.

The run used no human ratings, no physician adjudication, no external dataset,
and no patient-level identifier. Automated MRR and alignment statistics are not
clinical correctness measures.

## 8. Next step

The only remaining V7 technical experiment authorized by the frozen protocol is
the secondary MedGemma QA transfer using BM25, global alpha, and adaptive Top-1
reports with the unchanged V6 generator, prompt, decoding, and verifier. The QA
result cannot change the H1/H2 retrieval interpretation.
