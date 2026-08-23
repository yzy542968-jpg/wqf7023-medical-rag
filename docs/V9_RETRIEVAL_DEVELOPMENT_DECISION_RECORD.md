# V9 Retrieval Development Decision Record

## 1. Decision status

V9 retrieval development is complete. This record freezes all retrieval
choices before V9 test images are encoded or any test ranking is generated.
V5-V8 remain unchanged.

## 2. Development evidence

All results use the same 2,608-case historical bank and 374 report-bearing
Validation cases with three fixed questions per case.

| System | Validation nDCG@10 | MRR | Decision |
|---|---:|---:|---|
| BM25 text | 0.132491 | 0.084276 | retain R0 baseline |
| MedSigLIP image-image | 0.316454 | 0.333037 | retain R1 strongest single component |
| MedSigLIP image-report mean | 0.277267 | 0.272409 | select R2 report policy |
| MedSigLIP image-report max | 0.273686 | 0.277149 | no-go under frozen policy rule |
| Fixed multimodal 0.25/0.50/0.25 | 0.248485 | 0.213251 | retain R3 comparator; below R1 |
| Learned Linear reranker | 0.313816 | 0.329958 | no-go versus MLP |
| Learned MLP reranker | **0.327779** | **0.336488** | select and promote as R4 |

The fixed fusion result is mixed: it improves over BM25 but underperforms
image-image retrieval. This negative result is retained and no fusion grid or
normalization policy is changed after outcome inspection.

## 3. Frozen report and fusion policies

The normalized-mean report policy is selected because maximum-minus-mean
nDCG@10 was `-0.003582`, below the prespecified `+0.005` requirement.

The fixed multimodal comparator is:

```text
0.25 * normalized BM25
+ 0.50 * normalized image-image cosine
+ 0.25 * normalized image-report mean cosine
```

It is retained as the prespecified fixed comparator rather than relabeled as
the best system.

## 4. Frozen learned reranker

The selected scorer is the prespecified MLP:

```text
9 -> 32 -> 16 -> 1
trainable parameters: 865
training pairs: 307,176
best internal epoch: 3
best internal nDCG@10: 0.326157
checkpoint SHA-256:
8afa68a48de9d6c9128d190f1368d0d45d41a958e5eb12787d7e725e7eb09efa
```

The MLP exceeded Linear by `0.013963`, greater than the `0.005` simplicity
tolerance. It exceeded fixed multimodal by `0.079293` and the strongest single
component by `0.011325`, passing both prespecified development gates.

The checkpoint remains local. Its hash, architecture, training protocol, role
manifest, and aggregate results are public. MedSigLIP, RadGraph, and BM25
parameters remain frozen.

## 5. Confirmation systems

The retrieval confirmation matrix is fixed as:

```text
R0 BM25 text
R1 MedSigLIP image-image
R2 MedSigLIP image-report normalized mean
R3 fixed multimodal 0.25/0.50/0.25
R4 learned MLP paired reranker
R4-shuffled complete learned system under wrong-image assignments
```

R4 is the proposed V9 method. R1 is the strongest development baseline and is
the primary superiority comparator. R3 remains important evidence that fixed
fusion alone was insufficient.

## 6. Boundaries and negative results

- Validation was used for policy, architecture, and promotion decisions.
- No QA outcome selected any retrieval system.
- No test image has been encoded and no test ranking/outcome inspected.
- The binary threshold remains 0.50 despite 65/374 Validation cases having no
  binary-relevant candidate; continuous nDCG@10 remains primary.
- No feature, pair sample, architecture, optimizer, or epoch was changed after
  the learned result was observed.
- Confirmation failure will be reported and will not trigger retuning.

## 7. Reproducibility

Independent reruns produced identical SHA-256 values for:

```text
MedSigLIP per-query validation rows:
0f68b8643f2203d65e2adddfcb12fc41938df846c0deb785432cd212e74ee81e

Learned per-query validation rows:
a30061120c2dd74d360238f0ee5ee523c6769daf34a9665d52544dbb100bf2db

Linear checkpoint:
8a249d2791fab28f36097563ab404a81c53d2249dca33230d514aa55a5a45ab3

MLP checkpoint:
8afa68a48de9d6c9128d190f1368d0d45d41a958e5eb12787d7e725e7eb09efa
```

