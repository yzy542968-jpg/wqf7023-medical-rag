# V13 Target-Image Concept Development Protocol

## Status and purpose

V13 is a development-only extension. It does not amend the frozen V10/V11
study, does not use V10 Test for fitting or selection, and cannot replace the
thesis primary evidence without a separately frozen future confirmation study.

The V10 post-hoc F1CheXbert analysis found that historical RAG improved lexical
and graph consistency but did not establish a uniform 14-observation pathology
advantage. V13 tests a targeted response to that bottleneck:

> Can a small trainable pathology-concept head over frozen target-image
> embeddings improve target-image observation prediction, and can conservative
> predicted concepts improve downstream QA without exposing the hidden target
> report at inference time?

## Data boundaries

- Source: the existing OpenI artifact and V10 duplicate-cluster-disjoint split.
- Fit partition: V10 Train only.
- Hyperparameter and threshold selection: V10 Calibration only.
- Development evaluation: V10 Validation only.
- V10 Test is prohibited for model fitting, architecture selection,
  regularization selection, threshold selection, prompt selection, stopping,
  case inspection, and V13 promotion decisions.
- All partitions remain duplicate-cluster disjoint under the V10 split.
- Only case-ID and duplicate-cluster disjointness are claimed. Patient-level
  independence cannot be verified from the processed OpenI identifiers.

## Automated pathology targets

The label target is the 14-observation binary vector produced from each source
report's concatenated Findings and Impression text by:

- `f1chexbert==0.0.2`;
- official `StanfordAIMI/RRG_scorers/chexbert.pth` checkpoint;
- standard report-generation binary conversion, in which positive and
  uncertain outputs map to one and negative or blank outputs map to zero.

These are report-derived automated labels, not radiologist-adjudicated image
labels. They are used only as Train supervision and Calibration/Validation
development targets. Validation labels cannot enter model inputs.

## Frozen image representation

Each case uses the existing V10 MedSigLIP 1,152-dimensional case image
embedding. The MedSigLIP backbone remains frozen. Multi-view aggregation is
the already generated deterministic V10 case embedding; no image or embedding
is selected after observing V13 outcomes.

## Predeclared concept models

### Baseline

A Train-prevalence predictor provides a non-image baseline for AUROC/AUPRC and
thresholded label metrics.

### Linear head

Four one-versus-rest L2 logistic-regression configurations are fit:

```text
C in {0.01, 0.1, 1.0, 10.0}
class_weight = balanced
max_iter = 2000
solver = lbfgs
random_state = 7141
```

### Lightweight MLP

One multilabel network is allowed:

```text
1152 -> 256 -> 14
GELU
dropout = 0.20
BCEWithLogitsLoss with Train-derived positive weights
AdamW, lr = 1e-3, weight_decay = 1e-4
batch_size = 64
maximum_epochs = 100
random_seed = 7142
```

Early stopping uses Calibration macro AUPRC with patience 10. The best epoch is
retained. No architecture, layer width, loss, or optimizer search is permitted
after Calibration outcomes are seen.

## Selection and thresholds

The primary model-selection metric is Calibration macro AUPRC across labels
with both positive and negative support. If two models differ by less than
`0.005`, prefer the simpler linear model; linear C ties prefer the smaller C.

After model selection, each label threshold is selected on Calibration from:

```text
{0.05, 0.10, ..., 0.95}
```

The objective is label-specific F1. Exact ties prefer the higher threshold to
reduce unsupported positive concepts. Labels without both classes in
Calibration retain threshold 0.50 and are reported separately.

The selected model, thresholds, label order, model hash, input hashes, package
versions, and selection record are frozen before Validation evaluation.

## Concept-prediction metrics

Validation reports:

- per-label AUROC and AUPRC;
- macro and micro AUROC/AUPRC over supported labels;
- 14-label micro and macro F1 at frozen thresholds;
- five-label micro and macro F1;
- exact-set accuracy;
- Brier score;
- expected calibration error;
- report-indexed normal, abnormal, and indeterminate sensitivity;
- risk-coverage curves based on predictive confidence.

All metrics are automated report-label agreement, not clinical diagnostic
accuracy.

## Downstream QA pilot

Only if concept prediction completes without integrity failure, a deterministic
96-case Validation subset is instantiated by SHA-256 ordering with seed 7143:

- 48 report-indexed normal cases;
- 48 report-indexed abnormal cases;
- no replacement after output inspection.

The same frozen target image, indication, question, historical Top-3 cases,
MedGemma revision, decoding, output budget, and provenance assembly are used in
both conditions:

1. `concept_off`: historical RAG without a concept line;
2. `concept_on`: the same prompt plus predicted target-image observations whose
   probabilities exceed their frozen conservative thresholds.

The concept line must be described as an automated target-image hypothesis,
not verified evidence. It may contain at most five positive observations,
ordered by probability. If none pass threshold, it states that no confident
concept was predicted. The hidden report, report labels, reference answer, and
Validation qrel never enter either prompt.

QA evaluation uses Token-F1, F1RadGraph, F1CheXbert, schema/provenance validity,
token ceilings, input length, and case-grouped paired bootstrap intervals.

## Promotion rule

The concept route is retained as a promising development contribution only if:

- the image concept model exceeds the prevalence baseline on Calibration and
  Validation macro AUPRC;
- `concept_on - concept_off` is positive on five-observation F1CheXbert;
- Token-F1 or complete F1RadGraph does not materially degrade;
- answer-contract and provenance validity do not decline.

If only the classifier improves, report it as a target-image concept diagnostic
and do not claim QA improvement. If QA scores worsen, retain the negative
result and do not tune on Validation subgroups.

## Claim boundary

V13 trains a small task-specific head while keeping foundation models frozen.
It does not fine-tune MedSigLIP or MedGemma. Its labels are generated from
reports by CheXbert and therefore inherit label noise, domain shift, and
feature-metric coupling. No clinical safety, physician accuracy, patient
benefit, external validity, or patient-level independence claim is permitted.

