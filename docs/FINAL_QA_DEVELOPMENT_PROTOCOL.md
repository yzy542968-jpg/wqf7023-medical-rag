# Final Paired Historical Image-Report QA Development Protocol

## Protocol status

This protocol governs development of the final structured and open-ended QA
extension. It follows the committed feasibility audit at `8a5e60b` and precedes
QA model fitting, evidence-policy selection, and any QA Test outcome. It is a
repository-timestamped prospective development protocol, not a formal external
preregistration.

The frozen V10-V16 studies remain unchanged. V10 is the duplicate-cluster and
alignment-controlled methodological foundation, V12 is the frozen learned
retriever, and V16 is the completed open-generation study. The final QA study
tests a new output construct and cannot retroactively change those results.

## Aim

The study investigates whether retrieval of paired historical radiology images
and reports improves the accuracy, evidence grounding, and reliability of
medical question answering for target chest radiographs whose reports are
withheld.

## Research questions

1. Does paired historical image-report retrieval improve QA over the same
   target image, available indication, and question without history?
2. What are the respective contributions of report relevance, visual
   similarity, and correct historical image-report pairing?
3. How do whole-report, section, sentence, and structured-fact evidence affect
   QA quality, context efficiency, and provenance?
4. How robust is paired-case RAG to irrelevant, contradictory, misaligned, and
   low-confidence history, and can filtering or abstention reduce negative
   transfer?

Question interpretation is an implementation mechanism. It does not replace
image-report pairing as the object of study and is not promoted as a separate
headline contribution.

## Data and task contract

The image, original report, indication, and historical retrieval bank come from
the local 3,851-case OpenI/IU-Xray collection. Rad-ReStruct supplies structured
reports and hierarchical QA annotations for 3,597 of those cases. The committed
audit found a 100% case mapping and a 100% match for 3,720 frontal-image
references.

The existing V10 exact/near-duplicate-cluster roles are retained:

| Role | Mapped QA cases | QA rows before hierarchy-aware processing |
| --- | ---: | ---: |
| Train | 2,351 | 114,253 |
| Calibration | 358 | 17,991 |
| Validation | 358 | 17,864 |
| Test | 530 | 26,747 |

The official Rad-ReStruct split is not used because 87 mapped V10 duplicate
clusters cross its train/validation/test roles. Official roles remain source
metadata only.

For every target, the model may receive the target image, the question, answer
options for structured closed questions, and an indication when available. It
may never receive the target Findings, Impression, structured answer vector,
or report text. Retrieved evidence must come from V10 Train cases. For a Train
target, the target case and every member of its V10 duplicate cluster are
removed from the historical bank.

Indication is optional pre-report context. Missing or heavily de-identified
content is represented as unavailable and is not reconstructed from the target
report. Image-plus-question and image-plus-indication-plus-question conditions
are both mandatory because indication may create a lexical shortcut.

## Structured QA modes

The primary mode predicts every provided question independently without gold
prior-answer history. Answer options are visible, but the correct answer and
the original Rad-ReStruct history field are hidden. This condition is
deployable and avoids propagating gold answers through the hierarchy.

Independent predictions are assembled into the official 2,470-dimensional
answer space and passed through a deterministic hierarchy-consistency cleaner.
The cleaner may suppress a child prediction when a predicted parent is
negative; it may not inspect the target report or repair predictions using gold
answers.

A secondary autoregressive diagnostic may use model-predicted prior answers.
An oracle-history diagnostic, if run, must be labelled non-deployable and may
not support the primary claim.

## Systems and controlled comparisons

| ID | Condition | Purpose |
| --- | --- | --- |
| B0 | Train-majority answer by question path | Answer-prior lower bound |
| B1 | Question and options only | Text-shortcut audit |
| B2 | Target image and question | Visual QA without indication/history |
| B3 | Target image, available indication, question | Strong no-history baseline |
| B4 | B3 plus deterministic random history | Unhelpful-context control |
| B5 | B3 plus report-only BM25 history | Text-retrieval RAG |
| B6 | Image-based retrieval followed by its paired report | Visual retrieval contribution |
| B7 | Frozen V12 multimodal paired cases, whole reports | Strong paired-RAG baseline |
| P1 | B7 candidates with case-to-fact evidence | Evidence-granularity ablation |
| P2 | P1 plus helpful-history filtering and confidence-aware abstention | Final proposed system |

All generator comparisons use the same selected generator checkpoint and
decoding policy. Retriever comparisons cannot be selected using downstream QA
Test results. The frozen V12 system is the default strong retrieval baseline;
any QA-specific retriever adaptation requires a separate committed amendment
before fitting.

## Alignment and interference controls

The retrieval-level alignment control repeats the complete visual scoring state
under 100 deterministic, unique, fixed-point-free target-image assignments.
The QA-level controls use one prespecified deterministic random Train history,
one deliberately broken historical image-report pair, and one contradictory
Train hard negative per eligible case. Hard-negative selection may use Train
labels and frozen retrieval scores but no Validation or Test answer outcome.

Negative transfer is defined at the case-question level as a no-history answer
that is correct under the frozen reference but becomes incorrect after history
is supplied. It is reported separately for relevant, random, broken-pair,
contradictory, and low-confidence histories.

## Evidence representation

Development compares whole-report, section-level, sentence-level, and
structured-fact evidence under Top-K values 1, 3, and 5 and fact budgets 4, 8,
and 12. Evidence units retain `case_id`, report section, source position or fact
identifier, and source text. Units from different cases remain separate.

The selected policy maximizes Validation supported-label macro-F1. Exact ties
are resolved by lower mean input tokens, lower latency, and then the simpler
policy. Test results cannot change this choice. Report-quality changes and
efficiency changes are reported separately; no clinical non-inferiority claim
is made from an arbitrary automated margin.

## Generator development

The first complete baseline uses the frozen local MedGemma snapshot already
validated by V16. A QA-specific QLoRA adapter may then be fitted on Train only.
Calibration supports early stopping, confidence calibration, and abstention
threshold fitting. Validation selects the final adapter, evidence policy,
Top-K, fact budget, prompt, and deterministic decoding configuration. Test is
prohibited throughout development.

The generated payload contains only a normalized answer, model confidence, and
an insufficient-evidence decision. Case IDs and citations are attached by a
deterministic program from the evidence units actually provided to the model;
the model cannot invent provenance fields.

## Outcomes

The primary endpoint is supported-label macro-F1 over the hierarchy-cleaned
2,470-dimensional report-answer representation. For each answer dimension with
at least one positive reference in the evaluation frame, binary F1 is computed;
the endpoint is the unweighted mean over those supported dimensions. This
definition excludes aggregate pseudo-rows from the official evaluator's second
average while retaining an official-compatible F1 as a secondary result.

The primary contrast is P2 minus B3. Success requires the lower bound of a 95%
paired case-bootstrap confidence interval to exceed zero. The case, not the QA
row, is the resampling unit.

Required secondary outcomes include official-compatible F1, micro-F1,
root-question macro-F1, balanced accuracy, ordinary accuracy beside the
Train-majority baseline, positive recall, specificity, exact report-vector
accuracy, and layer-specific results. Open answers use RadGraph, CheXbert,
Token-F1, negation, and laterality metrics. Retrieval uses nDCG@10, MRR, and
Hit@1/5. Evidence and reliability outcomes include citation precision,
provenance validity, negative-transfer rate, contradiction-following rate,
coverage, selective risk, input tokens, latency, and peak VRAM.

The observed yes/no distribution is 93.72% `no` and 6.28% `yes`. Ordinary
accuracy is therefore descriptive only and may never be presented without the
majority baseline, balanced accuracy, macro-F1, and positive recall.

Uncertainty uses 10,000 paired case-bootstrap samples with seed 7023. The
prespecified key-secondary family uses Holm adjustment. Undefined metrics are
reported as unavailable with their denominator; they are not replaced by zero,
one, or another favorable value.

## Role isolation and stopping rule

Train fits parameters. Calibration supports early stopping and confidence
calibration. Validation selects one final configuration. After a development
decision record is committed, a separate confirmation protocol and exact
config are committed before Test generation. Test is then run once. A technical
rerun is allowed only after a documented crash or corrupted output and must use
the identical frozen config.

The study remains valuable if paired history fails to improve QA. Such a result
would quantify negative transfer and the limits of report-derived analogies.
No Test case may be removed, replaced, or reweighted because its result is
unfavorable.

## Evidence boundary

The study evaluates report-derived structured-answer consistency on a
same-source OpenI/IU-Xray benchmark. It does not establish physician-rated
diagnostic correctness, clinical safety, patient-level independence, external
generalization, or deployment utility. Independent clinical review and
authorized patient-indexed external validation remain Future Work.
