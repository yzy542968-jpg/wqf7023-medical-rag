# V10 Retrieval Calibration Decision Record

Status: Calibration complete; V10 Test not run.

## Frozen input and fit

The prespecified L2-regularized logistic model was fitted once to 1,128 rows
from 376 Calibration cases. A positive row required an offline combined qrel
gain of at least 0.50 for the retrieved Top-1 case. The fitted data contained
309 positive and 819 negative rows.

| Apparent-fit measure | Value |
|---|---:|
| Brier score | 0.177892 |
| ECE (10 bins) | 0.045252 |
| AUROC | 0.723457 |

These are apparent-fit diagnostics, not independent generalization estimates.
They show useful ranking of offline retrieval relevance, but do not establish
clinical-risk calibration.

## Frozen operating point

The protocol selected the 80% Calibration-coverage operating point before
Calibration outcomes were read. Its frozen probability threshold is
`0.1761120354927459`. At the closest observed empirical point (79.96%
coverage), selective accuracy for the prespecified relevance target was
33.26%, compared with 27.39% at full coverage. Test will apply this threshold
without retuning.

When confidence is below the threshold, G3 suppresses historical evidence and
records that no sufficiently reliable historical case was identified. It may
still answer from the target image and indication. This is selective evidence
use, not a clinical abstention or safety guarantee.

## Audit artifacts

- Calibrator SHA-256: `e88d85e911c9214baa814d478754e34923c51568091686d63a7bebe6796a762e`
- Local Calibration rows SHA-256: `331d380eab20bfe73ba7624d1c07017fd7e6adb95ea0f1274cbdec32edc378a3`
- Seed: 7046
- Evidence policy: whole report
- Test outcomes inspected: no

No coefficient, feature, label threshold, coverage target, or downstream
generation setting may be changed after this record in response to Test.
