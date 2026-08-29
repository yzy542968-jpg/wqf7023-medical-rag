# Final QA Feasibility Audit

## Status and boundary

This document records a data-mapping audit for the proposed final structured
QA study. It does not modify the frozen V10-V16 methods, cohorts, outputs, or
metrics. The audit was run on the `final-qa-study` branch created from commit
`53e6888`, which is tagged `v16-final-thesis-freeze`.

The intended final task remains target-report-hidden chest-radiograph QA. A
target image, a clinical question, and any available pre-report indication are
model inputs. The target Findings, Impression, and structured answers are
evaluation references only. Retrieval is restricted to other-case paired
image-report records from the Train partition.

## Audited sources

- Processed OpenI/IU-Xray source: `data/processed/openi_cases.jsonl`, 3,851
  cases, SHA-256
  `56e367190396011d4d67f43e7e733389a8346890bf8729e82fb4326d063bbd68`.
- Existing V10 duplicate-cluster-disjoint split:
  `data/splits/v10/v10_cluster_disjoint_split.json`, SHA-256
  `b4c1b091c3dbff0399d07c8350f4d8d68ce8ce52e0157dcc96f46af8c8baa7b3`.
- Official Rad-ReStruct repository commit:
  `b293158f0c5c1c5fa27dd615c28005eb54d7b1de`.
- Deterministic audit implementation:
  `scripts/audit_final_qa_feasibility.py`.
- Machine-readable results:
  `data/splits/final_qa/final_qa_feasibility_audit.json` and
  `data/splits/final_qa/final_qa_case_mapping.csv`.

Rad-ReStruct does not replace OpenI. It supplies structured reports,
hierarchical QA pairs, answer paths, and an evaluator for IU-Xray/OpenI images.
OpenI remains the image source, report source, indication source, and
historical image-report retrieval bank.

## Mapping result

| Audit item | Result |
| --- | ---: |
| Rad-ReStruct reports | 3,597 |
| Reports mapped to local OpenI case IDs | 3,597 / 3,597 (100%) |
| Coverage of all local OpenI cases | 3,597 / 3,851 (93.40%) |
| Rad-ReStruct frontal-image references | 3,720 |
| References matched to local image records | 3,720 / 3,720 (100%) |
| Matched images not marked frontal locally | 0 |
| Structured QA rows | 176,855 |
| Unique question texts | 232 |
| Unique answer paths | 522 |

The official report identifiers map deterministically to the repository's
`CXR{number}` identifiers. Image matching used the stable `IM-*` image suffix
within each case and independently checked the local projection metadata.

## Partition audit

Applying the existing V10 duplicate-cluster-disjoint roles to the 3,597 mapped
cases gives:

| V10 role | Cases | QA rows |
| --- | ---: | ---: |
| Train | 2,351 | 114,253 |
| Calibration | 358 | 17,991 |
| Validation | 358 | 17,864 |
| Test | 530 | 26,747 |

The mapped cases occupy 2,845 V10 duplicate clusters. In contrast, 87 of these
clusters cross two or three of the official Rad-ReStruct train/validation/test
roles. The crossing is not evidence of an error in Rad-ReStruct; its official
split was not designed around this project's exact/near-duplicate report
clusters. It does mean that blindly adopting the official split would weaken
the leakage controls already established for this thesis.

**Decision:** retain the existing V10 cluster-disjoint roles for the final QA
study. The Rad-ReStruct official role remains provenance metadata only and is
not used to define development or confirmation membership.

## Label-distribution audit

The structured QA files contain 176,855 rows, with 28-169 questions per case
(median 39). Most rows are single-choice hierarchical questions. Of 162,257
rows whose options include both `yes` and `no`:

| Answer | Rows | Share |
| --- | ---: | ---: |
| No | 152,071 | 93.72% |
| Yes | 10,186 | 6.28% |

This imbalance makes ordinary micro accuracy unsuitable as the sole or primary
QA outcome. A trivial always-`no` system would obtain approximately 93.7%
accuracy without identifying positive findings. The final protocol must
therefore require a majority baseline, question/path-aware macro-F1, balanced
accuracy, positive-class recall, and per-category results. Ordinary accuracy
may be reported only alongside those safeguards.

The hierarchy also creates conditional questions. A negative parent answer can
make lower-level questions inapplicable. Independent row-level random sampling
would break this structure and can leak information between questions from the
same image. The case is therefore the indivisible split, bootstrap, and model
selection unit. Gold prior answers must not be supplied to a deployable QA
condition; sequential hierarchy evaluation must use model-predicted history or
be clearly labelled as an oracle diagnostic.

## OpenI field availability on mapped cases

| Field | Non-empty cases | Share |
| --- | ---: | ---: |
| Indication | 3,519 | 97.83% |
| Findings | 3,112 | 86.52% |
| Impression | 3,568 | 99.19% |
| Indication placeholder ratio <= 0.5 | 3,199 | 88.94% |

Indication is treated as optional pre-report context. Empty or heavily
de-identified indications are represented as unavailable; they are not inferred
from the target report. The final experiment must retain image-plus-question
and image-plus-indication-plus-question ablations because indication can create
a strong lexical shortcut.

Empty Findings do not invalidate structured QA labels, but they prevent a
Findings-based open-answer reference for those cases. Structured QA,
Impression-based open QA, and downstream report reconstruction must therefore
use separately recorded reference-availability denominators. Cases may not be
silently removed after outcomes are observed.

## Feasibility decision

The structured QA extension is technically feasible and has adequate
case-level scale. The audit supports continuing because:

1. every official structured report maps to a local OpenI case;
2. every official frontal image reference maps to a local frontal image;
3. all four existing V10 roles retain hundreds of mapped cases;
4. the Train role provides more than 114,000 QA rows before any hierarchy-aware
   filtering or weighting;
5. the existing OpenI historical bank and V12/V16 assets remain reusable.

The audit also fixes four non-negotiable design constraints:

1. do not use the Rad-ReStruct official split for the final thesis comparison;
2. do not present ordinary accuracy without imbalance-aware metrics;
3. do not split, tune, or bootstrap at QA-row level;
4. do not present report-derived structured labels as physician-adjudicated
   clinical accuracy or independent external validation.

## Next controlled step

Before training, the project must freeze a development protocol that specifies:

- the structured question/path inclusion policy;
- independent versus hierarchical QA conditions;
- answer normalization and inapplicable-answer handling;
- primary macro-F1 definition and majority baseline;
- OpenI indication availability policy;
- paired, report-only, image-based, random, broken-pair, and no-history arms;
- whole-report versus case-to-fact evidence arms;
- Train/Calibration/Validation use and model-selection rules;
- a Test prohibition and one-shot confirmation plan;
- negative-transfer, provenance, abstention, efficiency, and open-answer
  secondary outcomes.

No final QA Test output has been generated or inspected by this audit.
