# V6 Model-Modernized Confirmation Study: Confirmation Protocol

## 1. Protocol status

This protocol freezes the V6 confirmation design after completion and commitment
of the development decision record, but before deterministic instantiation of the
final confirmation case IDs. It is locally committed and version controlled; it
is not a formal or externally timestamped preregistration. The deterministic
selection rule mathematically determines the future cohort from the frozen source
frame, so the cases are not described as unknowable or blinded. At this point the
selection builder has not been executed and the final manifest has not been
generated or inspected.

The machine-readable specification is `config/v6_confirmation.json`. After this
protocol and config are committed, no model, revision, prompt, precision, query,
chunk rule, fusion weight, verifier threshold, metric, subgroup, statistical
criterion, or replacement rule may change in response to confirmation outcomes.

## 2. Objective and hypotheses

V6 tests whether the central V5 finding survives a modern image-text encoder, a
new medical generator, and a broader, newly instantiated case-ID-disjoint cohort
from the same OpenI/IU-Xray source.

**Primary retrieval hypothesis.** Correctly aligned MedSigLIP reranking improves
case-grouped MRR over indication-plus-question BM25. Support requires the lower
bound of the 95% case-grouped bootstrap interval for `MedSigLIP - BM25` MRR to be
greater than zero.

**Alignment-specificity hypothesis.** Correctly aligned MedSigLIP MRR exceeds the
distribution from 100 deterministic fixed-point-free shuffled-image controls.
The plus-one Monte Carlo value must be at most 0.05.

**Generator-robustness hypothesis.** The `MedSigLIP - BM25` downstream verified
Token-F1 point difference is positive under both Qwen2.5 and MedGemma 1.5. A
95% case-grouped bootstrap interval is reported for each generator. This
conjunctive direction criterion prevents a result from being called robust when
the retrieval benefit appears under only one generator.

The difference between the two generator-specific retrieval gains is reported as
a secondary difference-in-differences estimate with a case-grouped interval. It
has no separate pass/fail threshold. BioViL-T, Qwen3-Embedding, raw Token-F1,
support rate, abstention, and spectrum subgroups are secondary analyses.

## 3. Source and evidence boundaries

The processed source is `data/processed/openi_cases.jsonl`, SHA-256
`56e367190396011d4d67f43e7e733389a8346890bf8729e82fb4326d063bbd68`.
It contains 3,851 case records linked to official images. The study uses report
text, indication, images, and the dataset `problems` field.

`report-indexed normal` means normalized `problems == "normal"`.
`report-indexed abnormal` means a non-empty label other than `normal` and
`no indexing`. `report-index indeterminate` means `no indexing`. These are
dataset-index categories, not new clinical adjudications.

Reliable patient identifiers are unavailable, so only case-ID disjointness can
be verified. V6 is within-source confirmation and does not establish external
validity, patient-level independence, diagnostic performance, clinical utility,
or deployment safety.

## 4. Frozen selection frame and cohort generation

The audited frame is stored in
`data/splits/v6/v6_development_confirmation_overlap_audit.json`. After all prior
project and V5 cases are excluded, 1,479 cases meet V6 eligibility. Seventeen
`no indexing` cases are excluded from primary stratification, leaving 1,462:
1,045 report-indexed normal and 417 report-indexed abnormal. Development overlap
with both frames is zero.

The final candidate pool has 240 cases:

| Role | Report-indexed normal | Report-indexed abnormal | Total |
|---|---:|---:|---:|
| Targets | 86 | 34 | 120 |
| Distractors | 86 | 34 | 120 |
| Total | 172 | 68 | 240 |

Canonical case IDs are `str(value).strip()`, UTF-8 encoded. Selection sorts each
stratum by lowercase hexadecimal:

```text
SHA256("v6-selection|7026|" + canonical_case_id)
```

The first 172 normal and 68 abnormal cases are selected. Within each selected
stratum, role assignment sorts by:

```text
SHA256("v6-assignment|7026|" + canonical_case_id)
```

The first 86 normal and 34 abnormal are targets; the remainder are distractors.
Collection fingerprints use sorted unique canonical IDs joined by LF, UTF-8,
with no trailing LF. The builder must fail if source or frame fingerprints,
counts, readability, question construction, role composition, uniqueness, or
zero-overlap checks differ. No reserve or silent replacement pool exists.

After this protocol commit, the builder may run once to generate the manifest and
selected/target/distractor fingerprints. A true later data-integrity failure is
recorded as a protocol deviation; it is not repaired by silently taking the next
hash-ranked case.

## 5. Confirmation questions and candidate task

Each of 120 targets contributes the same three deterministic report-derived
question types used in V5: findings, impression, and summary, for 360 questions.
Distractors contribute no questions but remain in every 240-case ranking.

The retrieval target is the report paired with the target image in the source
dataset. This is closed-set paired-report retrieval, not diagnosis of an uploaded
new image. The reports and questions are not physician-authored external QA.

## 6. Frozen text retrieval

The primary text system is BM25 with `k1=1.5`, `b=0.75`. Its query is:

```text
Clinical indication: {indication}
Question: {question}
```

Qwen3-Embedding-0.6B revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` is retained as a secondary dense
baseline with 1,024-dimensional normalized embeddings and the frozen query
instruction in the config. It does not replace BM25 because it did not pass the
development selection rule.

All rankings use descending score then ascending canonical case ID. Primary
retrieval uncertainty is grouped by target case, preserving all three questions
together during resampling.

## 7. Frozen multimodal retrieval

MedSigLIP-448 revision
`9cea28a1a1195f665105faa6e8544c112fd960a4` is the primary modern image-text
encoder. Reports are represented only by findings and impression. Whitespace is
normalized, sentence order is preserved, and consecutive sentences are packed
into chunks of at most 64 MedSigLIP tokenizer tokens including special tokens.
An over-limit sentence is split consecutively without overlap; no non-empty text
is silently dropped or truncated.

Every image view is encoded independently, normalized, averaged by case, and
normalized again. The score for a candidate report is the maximum cosine between
the target case-image vector and any candidate report chunk.

BM25 first produces a 100-case shortlist. BM25 and image scores are independently
min-max normalized within that shortlist and fused as:

```text
0.5 * normalized_BM25 + 0.5 * normalized_image_report_score
```

The remaining 140 candidates retain BM25 order. Standardized BioViL-T uses text
revision `692f09e`, image-weight MD5
`a83080e2f23aa584a4f2b24c39b1bb64`, identical chunk texts, identical maximum
aggregation, identical views, and identical fusion as a secondary historical
encoder comparator.

## 8. Shuffled-image controls

One hundred deterministic wrong-image controls test alignment specificity. For
control index `i`, target IDs are ordered by:

```text
SHA256("v6-shuffle-order|7026|" + str(i) + "|" + canonical_case_id)
```

Each ordered source target receives the image of the next target in that order,
with the final source receiving the first image. Every control is therefore a
single-cycle derangement with no fixed point. The implementation must verify 100
unique assignments. The text query, candidate pool, shortlist, report chunks,
fusion, and metric code remain unchanged.

The plus-one Monte Carlo value is:

```text
(number of shuffled MRR values >= correct-image MRR + 1) / 101
```

## 9. Frozen 2 x 2 QA factorial

The four primary cells are:

| Retrieval | Qwen2.5-1.5B | MedGemma 1.5 4B |
|---|---|---|
| BM25 | required | required |
| BM25 plus MedSigLIP | required | required |

Qwen uses revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` in FP16.
MedGemma uses revision `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` with 4-bit
NF4 weights, double quantization, and BF16 compute. Precision is frozen from
technical preflight and may not change based on answers.

Both models receive identical semantic content: indication, question, and the
Top-1 selected report findings and impression. No image pixels enter the primary
factorial generator. Only official model-specific chat templates differ.
Decoding is greedy, `do_sample=false`, temperature effectively zero, and maximum
256 new tokens. Each output is written immediately with qid, source case,
retrieval condition, selected report, model revision, and token counts.

## 10. Frozen verifier

Both raw and verified answers are reported. The primary verifier is unchanged
from V5:

| Component | Frozen value |
|---|---|
| Model | `cnut1648/biolinkbert-mednli` |
| Lexical weight | 0.2 |
| Combined support threshold | 0.6 |
| Entailment threshold | 0.75 |
| Contradiction threshold | 0.5 |
| Evidence | Top-1 case ID, findings, and impression |

Verifier configuration SHA-256 is
`302e8ce368351af087259e53f63e134b4514fa4b9e1fd3a209e5e041a101fe9f`.
No V6 recalibration is permitted. Support, contradiction, revision, and
abstention are automated signals, not physician-adjudicated correctness or
safety.

## 11. Metrics and statistics

The primary retrieval metric is MRR. Secondary metrics are Hit@1, Hit@5, Hit@10,
target rank, and deterministic extractive proxy Token-F1. The primary QA metric
is verified Token-F1. Secondary QA metrics are raw Token-F1, exact match,
evidence-support rate, abstention rate, and revision rate.

All uncertainty resamples target case IDs with replacement and retains the three
questions of each sampled case. The plan uses 5,000 resamples and seed 7026.
Primary all-target results use 120 cases. Predefined spectrum sensitivity reports
86 report-indexed normal and 34 report-indexed abnormal targets separately; these
subgroups are secondary and receive no tuning or separate confirmatory claim
family. Wide intervals for the smaller abnormal subgroup are expected.

The QA analysis reports:

```text
Delta_Qwen = MedSigLIP_Qwen - BM25_Qwen
Delta_MedGemma = MedSigLIP_MedGemma - BM25_MedGemma
Difference-in-differences = Delta_MedGemma - Delta_Qwen
```

The same calculations are made for raw Token-F1, support, and abstention as
secondary trade-off analyses. Category or subgroup proportions are not used to
infer clinical prevalence.

## 12. Cost analysis

For uncached embedding builds and generation, record model-load time, encoding or
generation time, throughput, available image views, report chunks, retrieval
calls, input/output tokens, peak allocated GPU memory, precision, CUDA device, and
cache status. Machine-specific timing is descriptive, not a universal benchmark.

## 13. Execution and failure policy

The intended sequence is:

```text
commit confirmation protocol and config
-> run deterministic cohort builder once
-> commit manifest and fingerprints
-> run retrieval systems and 100 shuffled controls
-> run primary 2 x 2 generation
-> apply frozen verifier
-> run frozen case-grouped statistics
-> write confirmation result record
```

Technical OOM, process crash, or transient file-access failure may be rerun only
with byte-identical config, manifest, prompts, inputs, and code. A frozen-case data
integrity failure is documented and analyzed under the prespecified missing-case
policy; it does not authorize replacement. Outcome-driven reruns, subgroup tuning,
threshold changes, prompt changes, or selective model omission are prohibited.

## 14. Interpretation limits

Confirmation success means the prespecified same-source paired-report retrieval
and report-grounded reference-consistency findings were reproduced under the
frozen V6 systems. It does not mean the system diagnoses images, retrieves a true
report for an arbitrary new patient, generalizes to another institution, is
clinically correct, or is safe to deploy. Independent clinical evaluation and
authorized external datasets remain future work.
