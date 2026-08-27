# V13 Concept-on/off QA Results

## Status

The paired concept-QA pilot was executed once on the frozen 96-case V10
Validation manifest. It compared identical target images, indications,
questions, V12 LambdaMART Top-3 historical cases, whole-report evidence,
MedGemma revision, 96-token budget, parsing, and deterministic provenance.
Only the precomputed target-image concept line differed.

The 96 cases comprised 48 report-indexed normal and 48 report-indexed abnormal
cases. Each condition produced 192 answers across Findings and Impression.
V10 Test was not read or evaluated.

## Main result

| Metric | Concept off | Concept on | On - off | 95% CI |
|---|---:|---:|---:|---:|
| Token-F1 | **0.19926** | 0.18273 | **-0.01652** | **[-0.02824, -0.00533]** |
| F1RadGraph entity | **0.15631** | 0.13893 | **-0.01738** | **[-0.03082, -0.00466]** |
| F1RadGraph entity-relation | **0.14055** | 0.12499 | **-0.01556** | **[-0.02888, -0.00280]** |
| F1RadGraph complete | **0.12180** | 0.10672 | **-0.01508** | **[-0.02728, -0.00363]** |
| F1CheXbert micro F1-14 | **0.57212** | 0.57074 | -0.00137 | [-0.01911, +0.01622] |
| F1CheXbert micro F1-5 | **0.38095** | 0.36559 | -0.01536 | [-0.06910, +0.04737] |
| F1CheXbert exact set-5 | **0.79688** | 0.78646 | -0.01042 | [-0.03646, +0.01563] |

The concept condition failed the predeclared QA promotion rule. It reduced
Token-F1 and all three RadGraph scores with intervals below zero, and it did
not produce a positive five-observation F1CheXbert interval.

For Findings alone, F1CheXbert point estimates increased slightly, including
micro F1-5 `+0.00816`, but the interval crossed zero. Findings Token-F1 still
decreased by `-0.02241` with CI `[-0.04064, -0.00582]`. Impression showed
negative point estimates across lexical, graph, and five-label pathology
metrics. These subgroup observations do not justify a question-specific
post-hoc promotion.

## Concept coverage and engineering integrity

| Item | Concept off | Concept on |
|---|---:|---:|
| Answer-contract validity | 100% | 100% |
| Citation validity | 100% | 100% |
| Token-ceiling rate | 4.69% | 4.69% |
| Mean input tokens | 780.25 | 808.11 |

The frozen classifier produced at least one threshold-passing concept for 90
of 96 cases, with a mean of 1.57 and maximum of five concepts per case. The
most common retained labels were `No Finding` (48 cases), `Enlarged
Cardiomediastinum` (29), `Atelectasis` (18), and `Lung Opacity` (11).

The output contract and provenance remained intact, so the degradation is not
explained by parser failure or truncation. The concept line added about 28
input tokens and changed the generated content itself.

## Interpretation and decision

V13 separates component validity from pipeline utility:

1. The frozen target-image concept head strongly exceeded a prevalence
   baseline on independent Validation macro AUPRC.
2. Directly verbalizing those automated hypotheses did not improve downstream
   answer-reference consistency and materially reduced Token-F1 and RadGraph.

The likely mechanisms are prompt anchoring, noisy report-derived labels,
redundancy with MedGemma's visual input, and the loss of uncertainty when a
continuous score is converted into a short list of disease-like words. These
are plausible explanations, not proven causal findings.

The direct concept-prompt route is stopped. No threshold, wording, label
subset, case, or question-specific rule will be retuned on these Validation
outputs. The classifier may still be evaluated in a separately governed
retrieval/risk role, where continuous concept agreement can rank or withhold
historical evidence without asserting a diagnosis to the generator.

## Claim boundary

This is a Validation-only, same-source, automated pilot. F1CheXbert and
F1RadGraph are report-derived metrics; they are not physician adjudication.
The negative QA result does not invalidate the classifier, and the classifier
does not establish diagnostic accuracy, safety, patient benefit, external
generalization, or patient-level independence.

## Reproducibility

- Manifest case-ID SHA-256:
  `809630156f74450aa2ee4bd2f2e3968bf29c28a0350da971372a4641d0eb3840`
- QA row SHA-256:
  `904c80dc093fb76314e3a401bf9ab499af2875898ff21cdd6904eb762e2b3ec1`
- Concept checkpoint SHA-256:
  `9cccfe79a451f357cd6e69e21d3661f0317c7917d83f3f0177d71df310916c5f`
- Machine-readable evaluation:
  `data/splits/v13/v13_concept_qa_evaluation_summary.json`

