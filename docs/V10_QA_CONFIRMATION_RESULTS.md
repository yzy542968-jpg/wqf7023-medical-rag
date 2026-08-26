# V10 Compact QA Confirmation Results

Status: Test QA confirmation complete; no Test-driven retuning.

## Conditions

The analysis contains 568 technically eligible Test cases, two questions per
case, four systems, and 4,544 output rows. G0 uses the target image and
indication without historical evidence. G1 adds R4 Top-3 whole reports. G2 adds
R5-attention Top-3 whole reports. G3 applies the frozen confidence threshold to
G2 and withholds historical evidence below threshold.

Validation selected whole reports over sentence/fact generator contexts, so
the legacy internal identifier `g2_hierarchical` denotes R5 retrieval but does
not imply that the rejected fine-grained context policy was used.

## Automated report-reference consistency

| Condition | Token-F1 | Findings F1 | Impression F1 | Evidence withheld | Schema valid | Provenance valid |
|---|---:|---:|---:|---:|---:|---:|
| G0 target image | 0.149416 | 0.211218 | 0.087615 | 1.000000 | 1.000 | 1.000 |
| G1 R4 whole-report RAG | 0.207524 | 0.274844 | 0.140204 | 0.000000 | 1.000 | 1.000 |
| **G2 R5 whole-report RAG** | **0.209193** | **0.276074** | **0.142311** | 0.000000 | 1.000 | 1.000 |
| G3 selective R5 RAG | 0.204981 | 0.271522 | 0.138440 | 0.170775 | 1.000 | 1.000 |

G2 exceeded G0 by 0.059776 Token-F1 (95% case-grouped bootstrap CI
0.051137 to 0.068602). Historical similar-case context therefore improved the
study's automated source-report consistency endpoint over target-image-only
generation.

G2 exceeded G1 by only 0.001669 (95% CI -0.003469 to 0.006829). The interval
crosses zero, so V10 does not claim confirmed downstream QA superiority of R5
over R4 even though R5's retrieval nDCG improvement was confirmed. This is an
important retrieval-generation bottleneck rather than a failed study.

G3 withheld history in 17.08% of the evaluated QA rows and scored 0.004211
below G2. The selective mechanism remains useful for transparently flagging
weak historical evidence, but the frozen 80% coverage policy did not improve
aggregate Token-F1. It must not be described as a clinical safety mechanism.

## Structured-output result and remaining truncation

All four systems achieved 100% assembled-schema and provenance integrity. This
directly resolves V9's failure mode in which long, model-generated JSON became
unparseable: MedGemma now generates answer text, while deterministic code adds
bounded fields and provenance.

The raw answer token-ceiling rates remained high: 91.55% for G0, 64.79% for G1,
and 68.31% for G2/G3. V10 therefore does not claim that model continuation was
eliminated. It claims that a deterministic two-sentence finalizer prevents raw
continuation from corrupting the final structured record. Qualitative and
clinical review remain necessary to determine whether the retained answer is
substantively adequate.

## Interpretation boundary

Token-F1 measures overlap with source-report references. It is not physician
adjudication, diagnosis accuracy, or patient-safety validation. The planned
F1RadGraph analysis tests clinical entity/relation overlap; the prespecified
blank clinical-review package separately enables expert assessment if a
qualified reviewer becomes available.

## Artifact trail

- QA rows: 4,544 local records
- QA rows SHA-256: `0e82b3cf5d3913fdac82f49b6742451cf095849cad88caa2c5bedb070f793944`
- Peak allocated GPU memory in the final execution segment: 5,311.56 MiB
- Frozen configuration changed after QA outcomes: no
