# V13 Target-Image Concept Development Decision

## Status

Train/Calibration development is complete under the protocol committed at
`72d5af1`. V10 Validation outcomes have not been evaluated at the time of this
decision. V10 Test remains prohibited.

The experiment used 2,510 Train cases and 383 Calibration cases. Four and seven
cases, respectively, had empty concatenated Findings/Impression text and were
retained as all-negative automated report-label targets rather than removed
after inspection. All MedSigLIP embeddings were finite, 1,152-dimensional, and
unit-normalized within floating-point tolerance.

## Model selection

The primary selection metric was macro AUPRC across the 14 report-derived
CheXbert observations.

| Candidate | Calibration macro AUPRC |
|---|---:|
| Train prevalence | 0.07889 |
| Linear, C=0.01 | 0.29246 |
| Linear, C=0.1 | 0.31188 |
| **Linear, C=1.0** | **0.33252** |
| Linear, C=10.0 | 0.33168 |
| MLP 1152-256-14 | 0.32105 |

The selected model is the one-versus-rest L2 logistic head with `C=1.0`.
The MLP did not exceed the best linear model by the predeclared `0.005`
margin, so it was not selected. The simpler linear head also slightly exceeded
the deeper linear setting.

## Calibration diagnostics

At label-specific thresholds selected only on Calibration, the frozen linear
head obtained:

| Metric | Value |
|---|---:|
| Macro AUROC | 0.81082 |
| Macro AUPRC | 0.33252 |
| Micro AUROC | 0.85232 |
| Micro AUPRC | 0.34382 |
| Micro F1-14 | 0.48873 |
| Macro F1-14 | 0.38266 |
| Micro F1-5 | 0.52571 |
| Macro F1-5 | 0.48643 |
| Exact-set accuracy-14 | 0.32637 |

Performance varied strongly by observation prevalence. Calibration AUPRC was
highest for Cardiomegaly (`0.7313`), No Finding (`0.6434`), Pleural Effusion
(`0.5918`), Lung Opacity (`0.5610`), and Atelectasis (`0.5133`). Rare labels
such as Pneumothorax and Consolidation remained weak. The model must therefore
be treated as a probabilistic automated concept hypothesis, not a complete or
clinically validated image labeler.

## Frozen decision

- Selected model: linear one-versus-rest logistic regression, `C=1.0`
- Checkpoint SHA-256:
  `9cccfe79a451f357cd6e69e21d3661f0317c7917d83f3f0177d71df310916c5f`
- Train case-ID SHA-256:
  `f0e7f609e3373bdc7aa1984608d92f594ae80609ec76dd9e55303af7bf57dd6a`
- Calibration case-ID SHA-256:
  `0d6f1c47c4ab20596220f3c487cc9214cd2abffbee2a36716c3e56aaa10e4e4c`
- Decision record:
  `data/splits/v13/v13_target_concept_decision.json`

The checkpoint, label order, thresholds, source hashes, package versions, and
selection record are frozen before the first Validation evaluation. No
candidate may be substituted after Validation results are observed.

## Claim boundary

The labels are generated automatically from source reports by CheXbert. The
result demonstrates learnable agreement between frozen target-image embeddings
and report-derived observation labels on Train/Calibration. It is not evidence
of radiologist-adjudicated diagnostic accuracy, patient benefit, clinical
safety, external generalization, or patient-level independence.

