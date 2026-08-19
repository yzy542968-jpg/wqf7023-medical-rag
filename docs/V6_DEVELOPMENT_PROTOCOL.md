# V6 Model-Modernized Confirmation Study: Development Protocol

## 1. Protocol status and purpose

This document prospectively specifies the V6 development stage before V6 model
selection, policy selection, prompt/schema validation, or confirmation-case
selection is performed. It is a locally committed development protocol, not a
formal preregistration and not an externally timestamped registration. V5 models,
parameters, outputs, and quantitative conclusions remain frozen and are not
modified by V6.

V6 is a model-modernized confirmation study. Its purpose is not to show that a
newer model must obtain a higher score. Its purpose is to test whether the central
V5 finding is robust to a newer image-text encoder, a modern dense text retriever,
a newer medical generator, a broader within-source case spectrum, and a newly
instantiated case-disjoint confirmation cohort. The central finding under test is
that correctly aligned image information can improve paired-report retrieval and
that this retrieval benefit can transfer to downstream report-grounded question
answering. V6 also preserves the V5 observation that retrieval quality and
evidence-grounding quality are distinct outcomes.

The development stage is allowed to select only the policies explicitly listed
in this document. After the development decision record and confirmation protocol
are committed, no model, prompt, threshold, chunk policy, fusion weight, metric,
or subgroup rule may be altered in response to confirmation outcomes.

## 2. Research questions

V6 addresses five related research questions.

**RQ1: Modern text retrieval.** Does Qwen3-Embedding-0.6B provide a material
development-set MRR advantage over the existing BM25 retriever for retrieving the
paired report from an indication-plus-question query?

**RQ2: Modern image-text retrieval.** On a common text shortlist and under a
standardized report-chunking policy, does MedSigLIP provide useful alignment
between a target chest X-ray and the corresponding report, and how does it compare
with BioViL-T on the same development cases?

**RQ3: Alignment specificity.** On the future confirmation cohort, does the
correctly aligned image condition exceed both the same text-only baseline and a
predefined distribution of shuffled-image controls?

**RQ4: Generator robustness.** Does the downstream QA advantage of multimodal
retrieval over text-only retrieval remain positive when the generator changes
from Qwen2.5-1.5B-Instruct to MedGemma 1.5 4B?

**RQ5: Cost and trade-offs.** What latency, peak GPU-memory, and retrieval-call
costs accompany each retrieval and generation condition, and does improved paired
report retrieval necessarily improve automated evidence-support outcomes?

## 3. Study role of V5 and V6

V5 is retained as the preliminary controlled study. It established the original
case-level retrieval problem, indication shortcut, alignment-specific image
signal, downstream QA transfer, and retrieval/grounding trade-off. Its qualitative
analysis remains a separate frozen interpretation of V5 outputs.

V6 is the final model-modernized confirmation study. It uses a new confirmation
selection frame and a broader within-source spectrum that includes report-indexed
normal cases. It is not an external validation because all cases still come from
the Indiana University Chest X-ray/OpenI source. The broader spectrum is intended
to improve within-source spectrum coverage and cohort representativeness. It does
not establish transportability to another institution, imaging system, patient
population, or clinical workflow.

## 4. Data boundaries

### 4.1 Source data

The source is the processed OpenI/IU-Xray case file
`data/processed/openi_cases.jsonl`, which contains 3,851 case records linked to
official source images. V6 uses report text, clinical indication, image filenames,
and the report `problems` field. The `problems` field is treated as a dataset index
field, not as an independently adjudicated clinical label.

The processed source contains no reliable patient or subject identifier. The
study can therefore verify case-ID separation but cannot verify patient-level
independence. No claim of a patient-disjoint or patient-level holdout is permitted.

### 4.2 Development source

The V6 development cases are the 120 cases in the development split of the frozen
V5 multimodal cohort. Development retrieval uses these 120 cases as targets and
the complete frozen V5 240-case pool as candidates, mirroring the intended future
120-target/240-candidate confirmation structure. The development set contributes
360 report-derived questions: findings, impression, and summary questions for
each target case.

The development case-ID manifest is
`data/splits/v6/v6_development_case_ids.txt`. Its canonical case-ID SHA-256 is:

```text
a7f381262f4f9ae29a4a68f5bdca884686d97b1de2f95e3e22b3491be9aebfa5
```

The development spectrum is 114 report-indexed abnormal cases, six report-index
indeterminate cases, and zero report-indexed normal cases. This differs from the
planned broader-spectrum confirmation cohort. The mismatch is an explicit study
limitation and robustness challenge. Confirmation subgroup outcomes may not be
used to retune a model or policy.

### 4.3 Confirmation selection frame

The confirmation frame excludes 1,260 cases used by prior project cohorts and all
240 cases in the frozen V5 multimodal cohort. The V6 eligibility rule is applied
after these exclusions:

1. At least one linked image is present.
2. Cleaned findings contain at least 40 characters.
3. Cleaned impression contains at least eight characters.
4. No more than half of indication tokens contain the de-identification marker
   `XXXX`.
5. The `problems` value is classified only after eligibility is established.

The audited V6 eligible frame contains 1,479 cases:

| Report-index class | Definition | Cases |
|---|---|---:|
| Report-indexed normal | normalized `problems == "normal"` | 1,045 |
| Report-indexed abnormal | non-empty problem labels other than `normal` and `no indexing` | 417 |
| Report-index indeterminate | normalized `problems == "no indexing"` | 17 |

The 17 indeterminate records are excluded from primary proportional
stratification. The resulting stratifiable frame contains 1,462 cases: 71.48%
report-indexed normal and 28.52% report-indexed abnormal.

### 4.4 Development-confirmation separation

The input-frame audit is stored in
`data/splits/v6/v6_development_confirmation_overlap_audit.json`. All development
cases were explicitly enumerated and verified to be case-ID-disjoint from both the
V6 confirmation-eligible and stratifiable frames. Both observed overlap counts are
zero. This establishes case-ID disjointness only; patient-level independence could
not be verified because reliable patient identifiers are unavailable in the
processed source data.

The audit script is
`scripts/audit_v6_development_confirmation_separation.py`. It reconstructs the
selection frame from source data and prior manifests, applies the broadened V6
eligibility rule, classifies the complete normalized `problems` field, and fails
if expected counts or zero-overlap requirements change.

## 5. Cohort-generation design freeze

This protocol freezes the cohort-generation design, not the instantiated final
cohort. No final V6 confirmation case IDs are generated, previewed, or inspected
during development.

The predefined final composition is:

| Role | Report-indexed normal | Report-indexed abnormal | Total |
|---|---:|---:|---:|
| Targets | 86 | 34 | 120 |
| Distractors | 86 | 34 | 120 |
| Candidate pool | 172 | 68 | 240 |

After the V6 development decision record and V6 confirmation protocol/config are
committed, a deterministic builder will instantiate the final IDs using:

```text
selection key  = SHA256("v6-selection|7026|" + canonical_case_id)
assignment key = SHA256("v6-assignment|7026|" + canonical_case_id)
```

`canonical_case_id` is `str(case_id).strip()`, encoded as UTF-8. Lowercase
hexadecimal SHA-256 digests are sorted in ascending order. Selection occurs within
the normal and abnormal strata before target/distractor assignment. The selected
normal cases are assigned 86 targets and 86 distractors; selected abnormal cases
are assigned 34 targets and 34 distractors. The independent domain tags prevent
selection ordering from being reused as assignment ordering.

Case-ID collection fingerprints use sorted unique canonical IDs joined by `\n`,
UTF-8 encoded, with no trailing newline. Input-frame fingerprints are recorded
now. The selected-240, target-120, and distractor-120 fingerprints will exist only
after the confirmation protocol is frozen and the cohort builder is run.

Because the selection rule and source frame mathematically determine the future
cohort, the cases are not described as unknowable or blinded. The defensible claim
is that deterministic selection had not been executed and the final manifest had
not been generated or inspected before development and confirmation configuration
freeze.

## 6. Development experiment structure

### 6.1 Common candidate and query structure

Each development target contributes three report-derived questions. The primary
text query is:

```text
Clinical indication: {indication}
Question: {question}
```

The primary document is the complete processed source `report_text`. A
question-only condition is retained as a secondary diagnostic to quantify the
indication shortcut but is not used to select the primary text retriever.

All rank ties are resolved by descending score followed by ascending canonical
case ID. Retrieval metrics are calculated per question and uncertainty is grouped
by target case ID.

### 6.2 Primary text retriever selection

Two text retrievers are compared:

1. Existing BM25 with `k1=1.5` and `b=0.75`.
2. `Qwen/Qwen3-Embedding-0.6B`, using its resolved model revision, full 1,024
   dimensional embedding, L2-normalized vectors, cosine similarity, and the
   official instruction-aware embedding interface.

The dense query instruction is frozen as:

> Given a radiology question and clinical indication, retrieve the chest X-ray
> report containing the evidence needed to answer the question.

The report/document embedding receives no query instruction. Exact pooling,
padding side, tokenizer version, library version, and resolved model commit must
follow the official model-card implementation and be recorded in the development
decision record.

The primary text retriever, `T*`, is selected deterministically:

1. Compute development MRR for both retrievers on identical questions and the
   identical 240-case candidate pool.
2. If Qwen3-Embedding MRR exceeds BM25 MRR by at least 0.005, select
   Qwen3-Embedding.
3. Otherwise select BM25. A difference with absolute magnitude below 0.005 is
   operationally treated as a tie and resolved in favor of BM25 for simplicity.
4. Hit@1, Hit@5, and Hit@10 are mandatory diagnostics but do not act as post-hoc
   vetoes.

Qwen3-Embedding remains a reported modern dense baseline even if BM25 is selected
as `T*`.

### 6.3 MedSigLIP report chunking

The modern image-text encoder is `google/medsiglip-448`. Its official model card
specifies 448 x 448 image input and at most 64 text tokens. Long radiology reports
therefore require a study-defined aggregation policy. This policy is an original
V6 preprocessing decision, not an official MedSigLIP long-report method.

Only findings and impression text are encoded for image-report similarity. The
indication is excluded from the image-report embedding to prevent the clinical
query from being counted again as visual-report alignment evidence.

Deterministic chunk construction follows these rules:

1. Normalize whitespace without changing words or punctuation.
2. Preserve findings before impression and preserve sentence order.
3. Pack complete sentences into the largest consecutive chunks that fit the
   MedSigLIP tokenizer limit of 64 tokens including special tokens.
4. Split an over-limit sentence into consecutive tokenizer-bounded segments with
   no overlap.
5. Do not drop a non-empty segment and do not silently truncate report text.
6. Record chunk counts, token counts, and any over-limit-sentence split.

Two aggregation policies are compared on development and only these two are
allowed:

1. `normalized_mean_chunk_embedding`: L2-normalize each report-chunk embedding,
   average chunks, then L2-normalize the resulting report vector.
2. `maximum_image_chunk_cosine`: compute cosine similarity between the aggregated
   case-image vector and every report chunk, then use the maximum similarity as
   the report score.

The policy with higher development MRR is selected. If the MRR difference is less
than 0.005, normalized mean aggregation is selected for simplicity and lower
variance. No additional chunk size, overlap, weighted mean, learned pooling, or
query-dependent chunk policy may be introduced after results are observed.

### 6.4 Image and report representation

Each available radiograph view is processed independently with the official model
processor. Every view embedding is L2-normalized, views belonging to the same case
are averaged, and the case-image vector is L2-normalized again. This is the frozen
multi-view policy for BioViL-T and MedSigLIP.

The historical image-text encoder is `microsoft/BiomedVLP-BioViL-T`. The primary
standardized encoder comparison applies the same sentence boundaries and selected
aggregation rule to BioViL-T and MedSigLIP. The historical V5 whole-report
BioViL-T representation is retained only as a secondary replication diagnostic;
it is not mixed into the standardized encoder comparison.

Exact model revisions, processor revisions, image resize/crop behavior, dtype,
batch size, and embedding dimension must be recorded. Cached embeddings include a
source fingerprint and configuration fingerprint and are rejected when either
changes.

### 6.5 Two-stage multimodal reranking

The selected text retriever `T*` ranks all 240 development candidates. Only its
top 100 candidates enter image-report reranking. Text and image scores are
independently min-max normalized within that shortlist, then fused with fixed
weights:

```text
fused_score = 0.5 * normalized_text_score
            + 0.5 * normalized_image_report_score
```

The fused shortlist is ranked by fused score and canonical case-ID tie-break. The
remaining candidates are appended in their original `T*` order. Shortlist size and
fusion weights are inherited from the frozen V5 design and are not swept in V6.
This isolates encoder modernization and long-report aggregation rather than adding
a new fusion search.

Development reports the following retrieval systems:

1. BM25.
2. Qwen3-Embedding-0.6B.
3. `T*` text-only.
4. `T*` plus standardized BioViL-T reranking.
5. `T*` plus MedSigLIP mean-chunk reranking.
6. `T*` plus MedSigLIP max-chunk reranking.
7. `T*` plus the selected MedSigLIP policy.

The confirmation protocol will retain BM25, Qwen3-Embedding, `T*`, standardized
BioViL-T, selected MedSigLIP, and shuffled-image controls. Development performance
does not authorize removal of a prespecified comparison except for a documented
irrecoverable technical incompatibility committed before confirmation.

## 7. Generator development and factorial design

### 7.1 Generator models

The historical generator is `Qwen/Qwen2.5-1.5B-Instruct`. The modern medical
generator is `google/medgemma-1.5-4b-it`. Exact resolved revisions and license
acceptance status are recorded before model execution.

The primary confirmation QA matrix is:

| Retrieval | Qwen2.5 generator | MedGemma 1.5 generator |
|---|---|---|
| `T*` text-only | required | required |
| `T*` + MedSigLIP reranking | required | required |

The primary factorial comparison gives both generators the same semantic inputs:
question, clinical indication, and top-1 retrieved report text. It does not provide
image pixels to either generator. Image pixels influence the multimodal cells only
through report retrieval. This design isolates whether retrieval gain transfers
across generators rather than confounding a generator change with an additional
input modality.

A MedGemma condition that receives the correctly aligned image plus the retrieved
report is allowed only as an explicitly exploratory full-stack analysis. It does
not enter the primary 2 x 2 retrieval-transfer conclusion.

### 7.2 Prompt and decoding policy

Both generators receive the same semantic instruction and output schema. Only the
model-specific chat-template wrapper may differ. The prompt instructs the model to
answer from the retrieved report, distinguish unavailable evidence from negative
evidence, avoid unsupported inference, and produce a concise answer without
diagnostic advice. No model-specific factual hints are allowed.

Decoding is greedy with `do_sample=false`, temperature effectively zero, and at
most 256 new tokens. Generation is run one question at a time or with a batch size
that is proven output-equivalent. Prompt text, rendered messages, tokenizer
revision, generation configuration, and output parser are frozen in the
development decision record.

### 7.3 Precision and memory policy

The local reference machine has an NVIDIA GeForce RTX 5070 Laptop GPU with 8,151
MiB reported memory, PyTorch 2.11.0+cu128, and CUDA 12.8. MedGemma 1.5 4B may not
fit with safe runtime headroom at BF16 on this GPU.

Precision is selected by a technical preflight that does not inspect QA outcomes:

1. Attempt BF16 only if model load plus one maximum-length protocol-valid example
   completes without CPU offload, OOM, or peak allocated memory above the recorded
   safe limit.
2. Otherwise use 4-bit NF4 weight quantization with BF16 compute under a pinned,
   documented implementation.
3. Quantization is held constant across all MedGemma retrieval cells.
4. Qwen2.5 keeps its frozen V5 precision unless a compatibility-only change is
   required and documented before confirmation.
5. No precision mode is selected based on answer quality.

The development decision record must include the successful load configuration,
package versions, peak GPU memory, a deterministic generation repeat check, and a
small schema-validation sample. Failure to access or execute a required model is a
protocol deviation, not permission to substitute an unlisted model after seeing
results.

## 8. Verification policy

The primary V6 verifier remains the frozen V5 verifier and sentence-filtering
configuration. It is not retrained, recalibrated, or threshold-tuned during V6.
This keeps support-rate changes interpretable while the image encoder and generator
change.

Primary QA reporting includes both raw and verified outputs. Any new claim-level
verifier or MedRAGChecker-style component may be evaluated only as exploratory
analysis. It cannot modify primary answers, primary support-rate estimates, or the
decision to confirm a hypothesis.

Automated verification is not a clinical gold standard. Apparent over-rejection,
under-detection, or unnecessary abstention remains an exploratory interpretation
unless independently adjudicated.

## 9. Development metrics and deterministic decisions

### 9.1 Retrieval metrics

Development reports MRR, Hit@1, Hit@5, Hit@10, target rank, and deterministic
top-1 report Token-F1. MRR is the sole selection metric for `T*` and the MedSigLIP
aggregation policy. All questions from a case remain grouped for resampling and
summary uncertainty.

### 9.2 QA technical validation

Development generator validation is limited to:

1. Successful local model execution.
2. Deterministic repeatability under greedy decoding.
3. Output-schema parse rate.
4. Absence of prompt leakage or oracle target-report use.
5. Correct propagation of the actual top-1 retrieved report.
6. Runtime and peak-memory measurement.

Development QA scores may be reported descriptively but may not be used for an
open-ended prompt search. One common semantic prompt is written before full
development generation. A correction is allowed only for a documented technical
failure such as an unparseable template, and the before/after prompt plus reason
must be preserved.

### 9.3 Stopping rule

Development ends when all of the following are true:

1. The zero-overlap audit passes.
2. BM25 and Qwen3-Embedding development retrieval runs complete.
3. `T*` is selected by the frozen 0.005 MRR rule.
4. Both MedSigLIP aggregation candidates complete and one is selected by the
   frozen rule.
5. Standardized BioViL-T and selected MedSigLIP runs complete.
6. Qwen2.5 and MedGemma load, deterministic generation, and schema checks pass.
7. Precision, model revisions, processors, prompts, decoding, and runtime settings
   are recorded.
8. `V6_DEVELOPMENT_DECISION_RECORD.md` is written.

No new model, alpha sweep, chunk policy, prompt family, threshold, or metric is
added after these conditions are met. The project then moves to confirmation
protocol freeze rather than further development optimization.

## 10. Planned confirmation hypotheses and analyses

This section defines the analyses that development must support. Exact resolved
configuration values will be copied into the later confirmation protocol.

### 10.1 Primary retrieval hypothesis

On all 120 future confirmation targets, correctly aligned selected-MedSigLIP
reranking will improve MRR relative to the identical `T*` text-only ranking.
The paired difference is summarized with a 95% case-grouped percentile bootstrap
interval using 5,000 resamples and seed 7026. Hit@1, Hit@5, and Hit@10 are secondary
metrics.

### 10.2 Alignment-specificity hypothesis

The correctly aligned MedSigLIP MRR gain will exceed the distribution of gains
from 100 deterministic shuffled-image derangements. For permutation `j`, target
IDs are ordered by:

```text
SHA256("v6-shuffle|7026|" + str(j) + "|" + canonical_case_id)
```

Each ordered target receives the next target's image in a cyclic shift, producing
a deterministic single-cycle derangement with no fixed point. The empirical
one-sided probability is `(1 + number of shuffled gains >= correct-image gain) /
101`. The full shuffled distribution, not only the probability, is reported.

### 10.3 Generator-robustness hypothesis

For each generator, compute:

```text
Delta_Qwen     = F1(MedSigLIP retrieval -> Qwen2.5)
               - F1(T* retrieval -> Qwen2.5)

Delta_MedGemma = F1(MedSigLIP retrieval -> MedGemma)
               - F1(T* retrieval -> MedGemma)
```

The planned robustness claim requires the estimated retrieval advantage to be
positive under both generators. Case-grouped bootstrap intervals are reported for
both deltas. The difference-in-differences,
`Delta_MedGemma - Delta_Qwen`, is reported as a secondary interaction estimate and
is not interpreted as proof that one generator is clinically superior.

Verified Token-F1 is the primary QA metric. Raw Token-F1, exact match, automated
evidence-support rate, abstention rate, and answer length are secondary. A support
rate decrease is reported rather than hidden and is interpreted jointly with raw
answer content and the frozen verifier's known limitations.

### 10.4 Spectrum sensitivity analysis

The all-target analysis is primary. Predefined secondary analyses report the same
retrieval and QA effect directions separately for 86 report-indexed normal targets
and 34 report-indexed abnormal targets. These subgroups do not form separate
confirmatory hypothesis families. Their purpose is to assess whether the aggregate
effect is driven by the majority normal stratum. Wide intervals for the 34-case
abnormal subgroup are expected.

No outcome for either subgroup may trigger retuning. The report index is not a
clinical adjudication, so subgroup labels are never shortened to clinically normal
or clinically abnormal.

## 11. Computational-cost analysis

Each required system records:

1. Model load time and one-time embedding/index build time.
2. Median, mean, standard deviation, and 95th-percentile per-query latency.
3. Peak allocated and reserved GPU memory.
4. CPU memory when measurable.
5. Number of text retrieval calls, image-text similarity evaluations, verifier
   calls, and generator calls.
6. Throughput in questions per minute.
7. Cache-hit status and whether timing is cold or warm.

Timing comparisons use the same machine and exclude one-time downloads. Warm-up
queries are identified and excluded from steady-state latency. Hardware, driver,
CUDA, PyTorch, Transformers, and quantization-library versions are recorded.

The purpose is to characterize the accuracy/latency/memory trade-off, not to claim
production readiness.

## 12. Failure and rerun policy

Technical execution failures include OOM, process crash, transient file-access
failure, or interrupted execution. They may be rerun under the identical frozen
configuration. Logs must identify the failed attempt and rerun.

A frozen-case data-integrity failure includes a missing image, unreadable image,
unresolvable report-image link, or missing required field discovered after the
final cohort is instantiated. It may not be handled by silently selecting the next
hash-ranked case. The failure is recorded as a protocol deviation and handled by
the missing-case policy in the confirmation protocol.

All readability and required-field checks must run before the final manifest is
committed. After cohort freeze, there is no reserve replacement pool. Cases are
never replaced because they are difficult, produce poor scores, or appear unusual.

## 13. Reproducibility artifacts

Before development begins, the repository contains:

1. This protocol.
2. `config/v6_development.json`.
3. The explicit 120-case development manifest.
4. The machine-readable development-confirmation overlap audit.
5. The deterministic audit script and tests.

The development decision record will add:

1. Exact resolved model and processor commits.
2. Package and hardware environment.
3. `T*` selection table.
4. MedSigLIP aggregation selection table.
5. Precision and quantization decision.
6. Frozen prompt and rendered schema.
7. Runtime preflight results.
8. Hashes of development rows and summary artifacts.

The later confirmation freeze will add the confirmation protocol, frozen config,
cohort-builder script, and builder test before any final case IDs are generated.
Only after that commit will the builder create the selected-240, target-120, and
distractor-120 manifests and their fingerprints.

## 14. Interpretation and claim limits

V6 evaluates closed-set paired-report retrieval and report-grounded QA. It does not
evaluate autonomous diagnosis from a previously unseen patient image. The uploaded
or target image is used to rerank candidate reports from an indexed corpus. A
dashboard must therefore describe its action as retrieving the top-ranked
candidate report, not finding or confirming the patient's true report.

The source questions are constructed from reports and are not radiologist-authored
clinical questions. Reference consistency, Token-F1, and evidence-support signals
are not equivalent to physician-adjudicated correctness. No result supports a
claim of clinical safety, deployment readiness, radiologist equivalence, or direct
patient-care use.

The confirmation cohort remains within OpenI/IU-Xray. Inclusion of report-indexed
normal cases improves within-source spectrum coverage but not external validity.
Public pretraining of the evaluated models also creates an unquantified source-data
contamination risk that cannot be excluded for a public dataset.

The development set has no report-indexed normal targets while the future
confirmation composition is majority report-indexed normal. This mismatch is
reported openly. It prevents outcome-driven normal-subgroup tuning and makes V6 a
stronger frozen-policy spectrum-transfer test, but it may also reduce the selected
policies' optimality for the confirmation distribution.

## 15. Required execution order

The required order is:

```text
V6_DEVELOPMENT_PROTOCOL.md + config + separation audit
        -> commit

V6 implementation and development experiments
        -> T* selection
        -> MedSigLIP aggregation selection
        -> generator/precision/schema preflight

V6_DEVELOPMENT_DECISION_RECORD.md
        -> commit

V6_CONFIRMATION_PROTOCOL.md + frozen confirmation config
        -> commit

Run deterministic cohort builder for the first time
        -> generate actual 240/120/120 manifests and fingerprints
        -> commit

Run planned confirmation evaluation
        -> documented technical reruns only under unchanged configuration
```

Running a preview cohort selection before the confirmation protocol commit is not
permitted. Confirmation outcomes may not be used to reopen development decisions.

## 16. Official model documentation used for protocol constraints

- MedSigLIP model card: <https://developers.google.com/health-ai-developer-foundations/medsiglip/model-card>
- MedSigLIP weights: <https://huggingface.co/google/medsiglip-448>
- MedGemma 1.5 model card: <https://developers.google.com/health-ai-developer-foundations/medgemma/model-card>
- MedGemma 1.5 weights: <https://huggingface.co/google/medgemma-1.5-4b-it>
- Qwen3 Embedding technical announcement: <https://qwenlm.github.io/blog/qwen3-embedding/>
- Qwen3-Embedding-0.6B model card: <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B>

The official cards establish model identity, intended use, access terms, input
limits, and implementation constraints. The report chunking, multi-view
aggregation, shortlist fusion, and selection rules in this protocol are study
methods designed for V6 and are not represented as official vendor methods.
