# V13 Target-Image Concept Validation Results

## Status

The selected V13 target-image concept head was evaluated once on the 384-case
V10 Validation partition after the model, thresholds, checkpoint hash, and
development decision had been committed. V10 Test was not read or evaluated.

The frozen model was a set of 14 one-versus-rest L2 logistic heads over the
1,152-dimensional V10 MedSigLIP case image embedding. Automated report-derived
CheXbert labels served as the Validation target.

## Overall result

| Metric | Selected image head | Train-prevalence baseline |
|---|---:|---:|
| Macro AUROC | **0.82123** | 0.50000 |
| Macro AUPRC | **0.34340** | 0.07913 |
| Micro AUROC | **0.84131** | 0.76730 |
| Micro AUPRC | **0.35156** | 0.23369 |
| Micro F1-14 | **0.44624** | 0.25486 |
| Macro F1-14 | **0.29380** | 0.09036 |
| Micro F1-5 | **0.40964** | 0.18743 |
| Exact-set accuracy-14 | **0.33073** | 0.00000 |

The case-grouped macro-AUPRC difference over the prevalence baseline was
`+0.26427` with 95% CI `[+0.23347, +0.33581]`. The classifier therefore
passed the first V13 promotion condition on both Calibration and Validation.

## Observation-level behavior

The strongest Validation AUPRC values were:

| Observation | Positive cases | AUPRC | AUROC |
|---|---:|---:|---:|
| No Finding | 140 | 0.72904 | 0.81976 |
| Cardiomegaly | 51 | 0.61689 | 0.88971 |
| Pleural Effusion | 17 | 0.53695 | 0.94470 |
| Lung Opacity | 41 | 0.47745 | 0.83574 |
| Atelectasis | 31 | 0.46973 | 0.89116 |
| Support Devices | 24 | 0.42340 | 0.83009 |

Rare observations remained unstable. Pneumothorax had two positive Validation
cases and AUPRC `0.05226`; Fracture AUPRC was `0.08641`. Consolidation had no
positive Validation cases and was excluded from supported-label AUROC/AUPRC.
These rows cannot support condition-specific clinical claims.

## Spectrum and reliability

The Validation frame contained 136 report-indexed normal, 239 report-indexed
abnormal, and nine report-index indeterminate cases. Macro AUPRC was `0.30131`
for the normal subset and `0.33062` for the abnormal subset, but the label
support differs substantially between those strata and the numbers are not a
head-to-head clinical comparison.

The overall thresholded Hamming risk increased from `0.05678` at 10% coverage
to `0.11496` at full coverage, showing useful directional separation by the
model's confidence heuristic. However, probability calibration remained weak:
Brier score was `0.14466` and expected calibration error was `0.25800`. The
class-balanced logistic objective improves rare-label ranking but does not
produce calibrated clinical probabilities. Downstream prompts must therefore
describe passing labels as automated hypotheses, not probabilities of disease.

## Decision

The concept head is retained for the predeclared Validation-only concept-on/off
QA pilot because:

1. it exceeded the prevalence baseline on Calibration and Validation macro
   AUPRC;
2. the Validation macro-AUPRC interval over baseline was fully positive;
3. the selected checkpoint and thresholds were frozen before Validation;
4. no V10 Test outcome was used.

This decision does not yet establish that adding predicted concepts improves
QA. The next pilot must keep retrieval, target image, generator, question,
decoding, and provenance identical between conditions and vary only the
automated concept line.

## Reproducibility and claim boundary

- Validation cases: 384
- Validation case-ID SHA-256:
  `62d0aaa08fb207a2cfaae49c46d09dcb59bea0e408ce95355a6c5b0f3e3913f8`
- Selected checkpoint SHA-256:
  `9cccfe79a451f357cd6e69e21d3661f0317c7917d83f3f0177d71df310916c5f`
- Decision record SHA-256:
  `85d2379ab2f3dd573a61ae9537b3da4ff892ab6dc8dd5a160fe48277ea16a2d8`
- Machine-readable result:
  `data/splits/v13/v13_target_concept_validation_summary.json`

All targets are automated labels extracted from source reports. This is
development evidence of image-to-report-label agreement, not radiologist-
adjudicated diagnostic accuracy, clinical safety, patient benefit, external
generalization, or patient-level independence.

