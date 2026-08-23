# V9 Retrieval Confirmation Results

## 1. Confirmation status

The frozen 752-case V9 retrieval confirmation completed without outcome-driven
retuning or case replacement. Five systems were evaluated for three fixed
questions per case against the same 2,608-case historical report bank.

## 2. Primary results

| System | nDCG@10 | MRR |
|---|---:|---:|
| R0 BM25 text | 0.134156 | 0.083542 |
| R1 MedSigLIP image-image | 0.315561 | 0.328270 |
| R2 MedSigLIP image-report mean | 0.274069 | 0.256032 |
| R3 fixed multimodal | 0.246935 | 0.211322 |
| **R4 learned MLP reranker** | **0.327942** | **0.331968** |

The primary paired comparison was:

```text
R4 minus R1 nDCG@10: +0.012381
95% case-bootstrap CI: [+0.009226, +0.015584]
bootstrap iterations: 10,000
```

Because the lower confidence bound was greater than zero, the prespecified R4
superiority criterion was met.

## 3. Alignment control

The complete frozen R4 visual state was recomputed under 100 unique,
fixed-point-free wrong-image assignments.

```text
aligned R4 nDCG@10:       0.327942
shuffled mean nDCG@10:    0.220370
shuffled SD:              0.004900
shuffled 2.5 percentile:  0.210726
shuffled 97.5 percentile: 0.231474
plus-one p-value:         0.009901
```

The aligned result exceeded every shuffled assignment, satisfying the frozen
alignment-dependence criterion. The gain therefore cannot be explained only
by the clinical-indication/question text or a generic visual prior.

## 4. Strict sensitivity subset

The predefined 262-case project-history-untouched subset produced:

| System | nDCG@10 | MRR |
|---|---:|---:|
| R0 BM25 text | 0.129956 | 0.139288 |
| R1 image-image | 0.411812 | 0.586296 |
| R2 image-report | 0.331673 | 0.454241 |
| R3 fixed multimodal | 0.288307 | 0.355632 |
| **R4 learned MLP** | **0.419901** | **0.592357** |

This subset supports the direction of the main result but remains a sensitivity
analysis rather than a separate confirmatory hypothesis family.

## 5. Interpretation

Three findings are retained:

1. Image-image retrieval was much stronger than generic-question BM25 in this
   new-case task.
2. Naive fixed score fusion underperformed image-image retrieval, showing that
   adding a weak text channel can reduce ranking quality.
3. A small trainable query-conditional reranker recovered a reproducible gain
   over the strongest frozen component and generalized to the held-out test.

The third result is the principal V9 methodological contribution. The trained
component contains only 865 parameters; foundation encoders remained frozen.

## 6. Reproducibility and claim boundary

```text
per-question rows: 11,280
rows SHA-256:
baa56924928b144c9b877b8e2218e04d17df6b77a6f794ed3830f7ccf3e449fd

MLP checkpoint SHA-256:
8afa68a48de9d6c9128d190f1368d0d45d41a958e5eb12787d7e725e7eb09efa
```

These are retrospective, report-derived similarity results on OpenI/IU-Xray.
They do not establish physician-adjudicated clinical similarity, diagnostic
accuracy, clinical safety, or external generalization.

