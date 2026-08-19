# P2 V6 Integrated Chapters 3-5

## Draft status

This document is the V6 main-study replacement for Chapters 3-5. It should be read together with the existing literature review and front matter, but it does not overwrite the submitted P1/P2 documents. V5 is treated here as a preliminary controlled study that exposed the alignment and grounding problem. V6 is the model-modernized confirmation study and supplies the principal final technical evidence.

All V6 values in this document are taken from the frozen artifacts and statistical analysis associated with the following commits:

```text
confirmation protocol:       eee7405
confirmation cohort:         43fe1a0
retrieval outcomes:          c6442c9
raw QA outcomes:             5aa8a6b
verified QA outcomes:        3ae127f
statistical outcomes:        9258e2f
```

The interpretation is deliberately bounded. The study evaluates same-source, closed-set, paired-report retrieval and report-grounded question answering. It does not evaluate autonomous image diagnosis, clinical correctness, patient-level independence, external validation, or deployment safety.

# Chapter 3: Methodology

## 3.1 Research design

This research used a staged empirical design to study retrieval-augmented medical question answering over paired chest X-ray images and radiology reports. The central methodological problem was that a generated answer can be locally supported by a retrieved report while the report itself belongs to the wrong case. The system therefore had to be evaluated at several linked levels: target-case retrieval, alignment-specific image contribution, report-grounded generation, automated evidence filtering, and computational cost.

The project was developed in two connected studies. The preliminary V5 study preserved the earlier controlled comparison and qualitative review. It exposed several risks: referral indication could act as a strong lexical shortcut; improved rank did not always mean correct Top-1 retrieval; report-level faithfulness did not guarantee target-case alignment; and the frozen verifier could remove content that appeared supported by the selected report. Those findings motivated a new confirmation design rather than a post-hoc modification of V5.

The final V6 study modernized the learning components and used a newly instantiated confirmation cohort. Its primary image encoder was MedSigLIP, its secondary historical image encoder was BioViL-T, its secondary modern dense text retriever was Qwen3-Embedding, and its two QA generators were Qwen2.5-1.5B-Instruct and MedGemma 1.5 4B. BM25 remained the primary text baseline because it was transparent, reproducible, and competitive during development. The V5 semantic verifier was deliberately retained unchanged so that image-encoder and generator changes would not be confounded by a simultaneous verifier change.

The V6 study addressed four questions:

1. Does correctly aligned MedSigLIP reranking improve target-report retrieval over the same BM25 indication-plus-question baseline on an untouched confirmation cohort?
2. Is the image contribution alignment-specific, as tested by deterministic shuffled-image controls?
3. Does the retrieval improvement transfer to report-grounded QA under both Qwen2.5 and MedGemma 1.5?
4. What trade-offs appear across retrieval quality, reference consistency, automated support, abstention, latency, and GPU memory?

The design separated development from confirmation. Development was used to select the text retriever, MedSigLIP report aggregation policy, generator revisions, precision policies, and other implementation details. The confirmation cohort-generation rule was frozen before the final case IDs were instantiated. The confirmation outcomes were then generated once under the frozen configuration. No prompt, threshold, model, cohort member, or statistical rule was changed after confirmation outcomes were observed.

The unit of statistical analysis was the case. Each target case contributed three questions, so the three rows were not independent observations. Case-grouped paired bootstrap resampling preserved this dependence. Results are therefore reported both at the question-row level for completeness and at the case-grouped level for inference.

## 3.2 Data source and ethical scope

The study used de-identified Indiana University Chest X-ray/OpenI cases with linked chest radiograph images and radiology reports. A processed case record contained a stable case identifier, clinical indication, problem label, findings, impression, and linked image metadata. The images were used as retrieval representations; the final generator received report text rather than image pixels in the primary QA factorial.

The source is suitable for controlled research because the image-report links and report sections are available in a reproducible local representation. It remains a single source dataset. The V6 confirmation cohort therefore provides within-source spectrum coverage, not external validation. The use of report-indexed normal cases improves representation of the source cohort, but it does not establish that the labels are independent clinical adjudications.

Patient-level independence could not be verified. The processed data contained reliable case identifiers but no stable patient or subject identifiers that could be used for a patient-disjoint audit. The study consequently makes a case-ID disjointness claim only. It does not use the phrases patient-disjoint, patient-independent, or patient-level holdout.

All model execution was local. Large per-question rows, generated text, image pixels, and embedding caches were retained locally according to the repository release policy. Publicly tracked summaries contain aggregate metrics, configuration hashes, lineage commits, and claim boundaries rather than the restricted or unnecessarily identifying content.

## 3.3 V6 development and model selection

The V6 development stage used a separate development cohort and a fixed comparison procedure. The main development decisions were frozen in `V6_DEVELOPMENT_DECISION_RECORD.md` before confirmation outcomes were generated.

### 3.3.1 Text retrieval

BM25 was selected as the primary text retriever. The query combined the clinical indication and question. Its shortlist size was 100. Qwen3-Embedding-0.6B was retained as a secondary modern dense text baseline, not as an automatic replacement for BM25. The development result showed that model recency alone was insufficient to justify displacing a transparent sparse baseline.

### 3.3.2 Image and report representation

MedSigLIP was selected as the primary modern image-text encoder. Its tokenizer accepts a maximum of 64 text tokens. Reports were therefore divided into deterministic sentence-aware chunks of at most 64 tokens, without overlap or silent truncation. Per-chunk image-text similarity was calculated and the maximum chunk similarity was used as the report-level image score. This maximum policy was selected during development and frozen before confirmation.

For the standardized secondary comparison, BioViL-T used the same chunk boundaries and the same maximum image-chunk aggregation policy. The historical whole-report policy was not mixed into the primary modern comparison. This distinction isolates encoder choice from an unreported change in text preprocessing.

Each case could have multiple image views. Each view was encoded, L2-normalized, averaged at case level, and L2-normalized again. The case-level image representation was then compared with report chunks. This was an explicit engineering aggregation policy defined for the study, not a claim about an official MedSigLIP long-report standard.

### 3.3.3 Fusion

The text and image components were fused only within the BM25 shortlist. Component scores were independently min-max normalized within the shortlist, and the frozen fusion weights were 0.5 for text and 0.5 for image. Ties were resolved by descending fused score and then ascending case ID.

### 3.3.4 Generation

The QA factorial used Qwen2.5-1.5B-Instruct in FP16 and MedGemma 1.5 4B with 4-bit NF4 double quantization and bfloat16 compute. Both used greedy decoding, temperature 0, and a maximum of 256 new tokens. They received the same semantic prompt content and differed only in their model-specific chat templates.

The prompt instructed the model to answer using only the selected report evidence, avoid unsupported findings and diagnoses, abstain with `Insufficient evidence.` when the selected report did not contain enough evidence, and return only a concise answer. The prompt contained clinical indication, question, selected report findings, and selected report impression. Image pixels were not passed to the generator.

### 3.3.5 Verifier

The verifier remained the frozen V5 semantic evidence checker based on `cnut1648/biolinkbert-mednli`. Its lexical weight was 0.2, combined support threshold 0.6, entailment threshold 0.75, and contradiction threshold 0.5. The evidence scope was the selected Top-1 case's findings and impression.

Keeping the verifier unchanged was methodologically important. V6 already changed the image encoder and the generators. Replacing the verifier at the same time would have made changes in support rate and verified Token-F1 difficult to attribute. The verifier is therefore treated as a fixed measurement component with known limitations, not as a clinical gold standard.

## 3.4 Confirmation cohort construction

The V6 confirmation selection frame contained 1,479 eligible cases after prior-project exclusion and development/confirmation separation checks. The report-indexed spectrum was defined from the normalized `problems` field:

```text
report-indexed normal:       problems == "normal"
report-indexed abnormal:     non-empty labels excluding "normal" and "no indexing"
report-index indeterminate:  problems == "no indexing"
```

The 17 indeterminate cases were excluded from the primary stratified sampling frame. The resulting stratifiable pool contained 1,462 cases:

```text
normal:    1,045 (71.48%)
abnormal:    417 (28.52%)
```

The confirmation composition was frozen before final case IDs were generated. Proportional stratified sampling selected 172 normal and 68 abnormal cases. A deterministic hash-based assignment then allocated 86 normal and 34 abnormal cases to targets and the same composition to distractors. The selection seed was 7026, with separate domain tags for selection and assignment. Case IDs were canonicalized as stripped strings, hashed as UTF-8, and fingerprinted as sorted unique IDs joined by LF without a trailing newline.

The final candidate pool contained 240 case IDs: 120 targets and 120 distractors. The target and distractor manifests were generated only after the confirmation protocol and frozen configuration had been committed. No case was replaced because of a poor outcome. A true data-integrity failure would have been recorded as a protocol deviation under the frozen failure policy.

The broader inclusion of normal cases was intended to improve within-source spectrum coverage and cohort representativeness. It was not described as external validity improvement. Because the normal/abnormal classification came from a report-indexed field rather than new physician adjudication, the labels are used as sensitivity-analysis strata rather than clinical truth labels.

## 3.5 Question construction and evidence scope

Each target case produced three deterministic report-derived questions:

- a findings question targeting the findings section;
- an impression question targeting the impression section;
- a summary question targeting the principal conclusion in the impression section.

The benchmark therefore contained 360 questions. The questions were constructed by code rather than written by physicians. They provide reproducible targets but limited linguistic diversity. The corresponding report field supplied the frozen reference answer. Token-F1 should therefore be interpreted as reference consistency, not as complete clinical correctness.

For every question, the retriever ranked all 240 candidate reports. The target case was known only for evaluation. The generator received the Top-1 selected candidate, including when the candidate was wrong. No oracle replacement was used. This preserved the downstream consequence of retrieval failure.

The system distinguished two grounding relations:

```text
answer supported by selected report
does not imply
selected report belongs to target case
does not imply
clinical correctness or safety
```

This distinction is central to the interpretation of the V6 results.

## 3.6 Retrieval evaluation

The primary retrieval metrics were Hit@1, Hit@5, Hit@10, and mean reciprocal rank (MRR). Hit@k measured whether the target case appeared in the top k ranked candidates. MRR used the full deterministic ranking, not only the displayed Top-10 rows. An extractive proxy Token-F1 measured overlap between the selected report-derived answer and the frozen reference; it was treated as a secondary diagnostic rather than a clinical accuracy measure.

The primary retrieval contrast was MedSigLIP reranking minus the same BM25 baseline on the same 360 questions and 240-case candidate pool. The primary hypothesis was that the case-grouped 95% bootstrap confidence interval for the MRR difference would have a lower bound above zero.

The alignment control used 100 deterministic shuffled-image assignments. Each assignment was unique and fixed-point-free: no target retained its own image. The correctly aligned MedSigLIP MRR was compared with all shuffled MRR values using a plus-one Monte Carlo calculation. The predeclared success criterion was `p <= 0.05`.

## 3.7 QA and verification evaluation

Each of the two generators was evaluated under BM25 and MedSigLIP retrieval, producing four cells with 360 rows each and 1,440 rows in total. Raw generation metrics were Token-F1, exact match, input tokens, and output tokens. The frozen verifier then produced a draft answer, final filtered answer, support rate, abstention indicator, revision indicator, and contradiction count.

The primary QA metric was verified Token-F1, computed after the unchanged V5 semantic verifier. The primary generator-robustness criterion was a positive MedSigLIP-minus-BM25 point difference for both Qwen2.5 and MedGemma. Confidence intervals were reported for interpretation. A difference-in-differences estimate was secondary:

```text
(MedSigLIP - BM25 under MedGemma)
-
(MedSigLIP - BM25 under Qwen2.5)
```

No independent confirmatory threshold was assigned to the difference-in-differences estimate.

## 3.8 Statistical analysis

All primary uncertainty estimates used 5,000 case-grouped paired bootstrap resamples with seed 7026 and a 95% confidence level. The three questions from each case were averaged before resampling. The percentile interval used NumPy's default linear quantile interpolation.

The primary retrieval analysis reconstructed complete rankings from the frozen BM25 procedure and frozen MedSigLIP embedding cache. The reconstructed MRR values were required to match the tracked retrieval summary before bootstrap analysis proceeded. This avoided the error of treating Top-10 truncation as a complete ranking.

The primary QA analysis averaged each metric within case and compared paired retrieval conditions within the same case and generator. Predefined secondary sensitivity analyses repeated the comparisons for 86 report-indexed normal targets and 34 report-indexed abnormal targets. These subgroups were descriptive and were not given an independent family of confirmatory hypotheses.

## 3.9 Qualitative interpretation and reproducibility

The V5 qualitative review was post-hoc and exploratory. Its protocol was committed before systematic case extraction, but it was not a result-blind preregistration. The researcher accepted all 24 assistant proposals under refined taxonomy v1.1, while preserving the distinction between protocol labels, assistant proposals, and researcher-reviewed outcomes.

The V6 confirmation analysis did not use qualitative inspection to modify the frozen model, prompt, threshold, cohort, or quantitative outcomes. Qualitative observations are used to explain the layered behavior of the pipeline, not to establish clinical correctness.

Tracked result summaries include configuration hashes, implementation hashes, input hashes, output hashes, model revisions, runtimes, and claim boundaries. Large per-question rows and image pixels remain local. Automated tests cover cohort generation, retrieval logic, QA matrix construction, verifier integrity, statistical calculations, and frozen result structure.

## 3.10 Computational cost

The formal run used an NVIDIA GeForce RTX 5070 Laptop GPU. The MedSigLIP embedding build took 27.24 seconds and used a peak of 1,817 MiB allocated GPU memory. Qwen3 text embedding took 3.69 seconds and used 1,345 MiB. BioViL-T took 8.13 seconds and used 822 MiB.

The Qwen2.5 generator completed its 720 rows in 192.72 seconds with a peak of 2,980 MiB. MedGemma completed its 720 rows in 1,346.05 seconds with a peak of 3,191 MiB. The verifier processed 1,440 rows in 44.86 seconds. These values describe this local execution environment and should not be presented as universal hardware benchmarks.

## 3.11 Methodological summary

The final method can be summarized as:

```text
OpenI/IU-Xray source cases
        |
        v
case-ID-disjoint V6 confirmation cohort
        |
        +--> clinical indication + question -> BM25 shortlist
        |                                      |
        |                                      +--> BM25 Top-1
        |                                      |
        |                                      +--> MedSigLIP image reranking -> Top-1
        |
        +--> 100 fixed-point-free shuffled-image controls
        |
        v
selected Top-1 report evidence
        |
        +--> Qwen2.5
        +--> MedGemma 1.5
        |
        v
frozen V5 semantic verifier
        |
        v
raw and verified reference-consistency metrics
        |
        v
case-grouped inference and bounded interpretation
```

# Chapter 4: Results and Analysis

## 4.1 Confirmation integrity and model conditions

The formal confirmation matrix contained 240 candidate cases, 120 target cases, 360 questions, 464 image views, and 514 MedSigLIP report chunks. The retrieval rows contained 1,440 rows: 360 for each of BM25, Qwen3-Embedding, standardized BioViL-T reranking, and MedSigLIP reranking. The QA matrix contained 1,440 rows: four cells with 360 rows each. The verified output retained the same four-cell and qid structure, with zero duplicate system-qid keys.

The target composition was 86 report-indexed normal and 34 report-indexed abnormal cases. The distractor composition was identical. This composition was instantiated from the frozen hash-based selection rule and was not changed after outcome inspection.

## 4.2 Retrieval performance

Table 4.1 reports the main retrieval results.

### Table 4.1. Retrieval on the V6 confirmation cohort

| System | Hit@1 | Hit@5 | Hit@10 | MRR | Extractive proxy Token-F1 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.5417 | 0.7056 | 0.7583 | 0.6168 | 0.6550 |
| Qwen3-Embedding-0.6B | 0.4778 | 0.6722 | 0.7500 | 0.5723 | 0.6600 |
| BM25 + BioViL-T | 0.5500 | 0.7139 | 0.7722 | 0.6277 | 0.6985 |
| BM25 + MedSigLIP | **0.5750** | **0.7222** | **0.7833** | **0.6474** | **0.7036** |

MedSigLIP improved MRR over BM25 by 0.03069. The 95% case-grouped paired bootstrap interval was `[0.00902, 0.05368]`, whose lower bound was above zero. The primary retrieval-over-text criterion therefore passed.

The result was not simply a consequence of replacing the text baseline with a newer embedding model. Qwen3-Embedding produced lower Hit@1 and lower MRR than BM25, although its extractive proxy Token-F1 was close to BM25. This supports the methodological decision to retain BM25 as the primary text baseline and treat dense text retrieval as a secondary comparison.

MedSigLIP also exceeded the standardized BioViL-T comparison on all displayed retrieval metrics. The difference should be interpreted as a modernized encoder comparison under the shared chunking and aggregation policy, not as a universal ranking of medical vision-language models.

## 4.3 Correctly aligned versus shuffled images

The correctly aligned MedSigLIP condition achieved MRR 0.64744. The 100 shuffled-image controls had a mean MRR of 0.59133, with a range from 0.56679 to 0.62023. None reached the correctly aligned MRR. The plus-one Monte Carlo value was 0.00990.

The control was deliberately strict. Each permutation was unique and fixed-point-free, so the test did not compare one correct run with repeated random noise or leave any target paired with its own image. The result supports the interpretation that the image signal was useful because it was aligned with the target case.

This does not mean the system solved image diagnosis. The image encoder contributed to ranking reports from a closed candidate pool. The selected report remained the evidence source for QA, and the benchmark supplied a known target case for evaluation. The result therefore supports alignment-specific paired-report retrieval, not patient identification in an open clinical environment.

## 4.4 Raw downstream question answering

Table 4.2 reports raw generation results.

### Table 4.2. Raw QA under the 2 x 2 retrieval-generator factorial

| Generator | Retrieval | Raw Token-F1 | Exact match | Top-1 retrieval accuracy | Mean output tokens |
|---|---|---:|---:|---:|---:|
| Qwen2.5 | BM25 | 0.1593 | 0.0722 | 0.5417 | 7.93 |
| Qwen2.5 | MedSigLIP | **0.1711** | **0.0750** | **0.5750** | 8.03 |
| MedGemma 1.5 | BM25 | 0.4923 | 0.0000 | 0.5417 | 22.10 |
| MedGemma 1.5 | MedSigLIP | **0.5303** | 0.0000 | **0.5750** | 21.71 |

The retrieval gain transferred to both generators. Qwen2.5 raw Token-F1 increased by 0.01183, with a case-grouped 95% interval `[0.00064, 0.02410]`. MedGemma raw Token-F1 increased by 0.03802, with interval `[0.02126, 0.05616]`.

The raw scores are modest for Qwen2.5. This is expected given the small model, concise answer policy, generic question templates, and strict reference overlap metric. MedGemma produced substantially more reference-overlapping content, but exact match remained zero because its answers were longer and not identical to the frozen report-derived references. Token-F1 was therefore the more informative primary overlap measure for this experiment.

## 4.5 Verified QA and generator interaction

Table 4.3 reports results after the unchanged V5 semantic verifier.

### Table 4.3. Verified QA and automated evidence signals

| Generator | Retrieval | Verified Token-F1 | Support rate | Abstention rate | Revision rate |
|---|---|---:|---:|---:|---:|
| Qwen2.5 | BM25 | 0.1669 | 0.3056 | 0.6944 | 0.6944 |
| Qwen2.5 | MedSigLIP | **0.1789** | 0.2889 | 0.7111 | 0.7111 |
| MedGemma 1.5 | BM25 | 0.4840 | 0.9417 | 0.0528 | 0.2889 |
| MedGemma 1.5 | MedSigLIP | **0.5226** | **0.9694** | **0.0250** | **0.2389** |

The verified Token-F1 difference was +0.01206 for Qwen2.5, with interval `[0.00165, 0.02370]`, and +0.03857 for MedGemma, with interval `[0.02198, 0.05642]`. Both point differences were positive and both intervals excluded zero. The frozen generator-robustness criterion therefore passed for both generators.

The support and abstention results reveal an important interaction. Qwen2.5 answers were frequently treated as insufficiently supported by the unchanged verifier. Its abstention rate increased from 0.6944 under BM25 to 0.7111 under MedSigLIP, while support rate decreased from 0.3056 to 0.2889. This does not show that MedSigLIP retrieval harmed the underlying answer evidence. It shows that retrieval improvement and compatibility with the verifier's sentence-level policy are not identical properties.

MedGemma showed the opposite pattern. Its support rate increased from 0.9417 to 0.9694, abstention decreased from 0.0528 to 0.0250, and revision decreased from 0.2889 to 0.2389 under MedSigLIP. The same frozen verifier therefore measured the two generators very differently. This is why support rate and abstention are reported beside Token-F1 rather than used as a substitute for answer correctness.

The secondary difference-in-differences estimate was +0.02651, with 95% interval `[0.01202, 0.04229]`. The retrieval gain transferred more strongly under MedGemma than under Qwen2.5. This is a secondary interaction result, not a separate confirmatory claim.

## 4.6 Spectrum sensitivity analysis

The predefined subgroup analysis produced a heterogeneous retrieval effect.

### Table 4.4. Retrieval sensitivity by report-indexed spectrum

| Subgroup | Cases | MedSigLIP minus BM25 MRR | 95% CI |
|---|---:|---:|---:|
| Report-indexed normal | 86 | -0.00272 | [-0.02043, 0.01651] |
| Report-indexed abnormal | 34 | +0.11520 | [0.06118, 0.17339] |

The aggregate retrieval improvement was driven mainly by the abnormal subgroup. The normal subgroup showed a difference close to zero, whereas the abnormal subgroup showed a larger positive difference. A plausible interpretation is that normal cases have more visually and lexically similar reports, so the image signal is less discriminative. This is an interpretation of the observed subgroup pattern, not a clinical generalization.

For verified Token-F1, the normal subgroup showed positive point differences for both generators: +0.01550 for Qwen2.5 and +0.02777 for MedGemma. The abnormal subgroup showed +0.00336 for Qwen2.5 and +0.06590 for MedGemma. The abnormal Qwen interval was wide and crossed zero, reflecting the small subgroup size of 34 cases. These analyses were sensitivity checks and were not used to redefine the primary hypotheses.

## 4.7 Computational cost and reproducibility

Table 4.5 summarizes the local runtime profile.

### Table 4.5. V6 execution cost

| Component | Runtime | Peak allocated GPU memory |
|---|---:|---:|
| Qwen3 text embeddings | 3.69 s | 1,345 MiB |
| MedSigLIP embeddings | 27.24 s | 1,817 MiB |
| BioViL-T embeddings | 8.13 s | 822 MiB |
| Qwen2.5, 720 generations | 192.72 s | 2,980 MiB |
| MedGemma 1.5, 720 generations | 1,346.05 s | 3,191 MiB |
| Frozen verifier, 1,440 rows | 44.86 s | not separately recorded |

MedGemma took approximately seven times longer than Qwen2.5 for the same 720-row generator condition. It also produced higher raw and verified Token-F1. The practical conclusion is a trade-off: MedGemma offered stronger answer overlap under this benchmark at a larger computational cost.

The implementation recorded model revisions, configuration hashes, cohort fingerprints, retrieval-row hashes, QA-row hashes, verifier hashes, and statistical-analysis hashes. All 160 repository tests passed after the V6 artifacts were created. A first verifier launch failed before model loading because the local Hugging Face cache path was not selected. The identical-config rerun used the existing repository cache, wrote no duplicate rows, and was recorded as a technical rerun rather than an outcome-driven rerun.

## 4.8 Results summary

The V6 evidence chain supports four main findings.

First, MedSigLIP reranking improved retrieval over BM25 on the same confirmation cohort, with a positive case-grouped interval. Second, the correctly aligned condition exceeded every shuffled-image control, establishing alignment specificity. Third, the retrieval benefit transferred to reference-consistent QA under both Qwen2.5 and MedGemma. Fourth, the magnitude of transfer depended on the generator and the verifier: MedGemma benefited more and produced much higher support signals, whereas Qwen2.5 experienced high abstention under the unchanged verifier.

The results also provide negative and mixed evidence. Qwen3-Embedding did not outperform BM25. Image contribution was weak in the report-indexed normal subgroup. Retrieval improvement did not imply that every final answer became better, and automated support did not provide an independent clinical gold standard. These limitations strengthen the interpretation by showing where the proposed mechanism does and does not operate.

# Chapter 5: Discussion and Conclusion

## 5.1 Answers to the research questions

### RQ1: Does correctly aligned image information improve target-report retrieval beyond text?

Yes, within the same-source closed-set V6 confirmation benchmark. MedSigLIP reranking increased MRR from 0.61675 for BM25 to 0.64744. The paired case-grouped difference was +0.03069, with a 95% confidence interval of `[0.00902, 0.05368]`. The lower bound exceeded zero.

This answer must be scoped carefully. The result does not say that images always improve retrieval, that MedSigLIP is universally superior, or that image input solves open-corpus patient identification. It says that under a fixed 240-case candidate pool, a frozen BM25 shortlist, and the V6 pairing procedure, correctly aligned image representations added measurable retrieval signal.

### RQ2: Is the image contribution specific to correct image-report alignment?

Yes for the frozen control. The correctly aligned MRR was 0.64744, while none of the 100 deterministic fixed-point-free shuffled-image controls reached it. The plus-one Monte Carlo value was 0.00990. This supports an alignment-specific interpretation rather than a generic “any image feature helps” interpretation.

The control does not prove clinical identity outside the benchmark. It only shows that preserving the benchmark's image-report pairing mattered to the retrieval score. The candidate pool was closed and the target was known for evaluation.

### RQ3: Does retrieval improvement transfer to downstream QA across generators?

Yes for the frozen reference-consistency metric. Verified Token-F1 increased by +0.01206 under Qwen2.5 and +0.03857 under MedGemma. Both generator-specific point criteria passed, and both case-grouped confidence intervals were positive.

The transfer was not uniform across all automated signals. Qwen2.5 had very high abstention and a small decline in support rate under MedSigLIP. MedGemma showed improved support, lower abstention, and a larger Token-F1 gain. Therefore, retrieval transfer depends on the interaction between selected evidence, generator behavior, answer form, and verifier policy.

### RQ4: What trade-offs remain?

The main trade-off is between answer quality, measurement compatibility, and computation. MedGemma was approximately seven times slower than Qwen2.5 for the same number of rows but achieved higher raw and verified Token-F1. MedSigLIP added an embedding cost that was small relative to generation cost, but it did not improve the report-indexed normal subgroup's MRR. The frozen verifier was fast, but its abstention behavior differed sharply across generators.

The study therefore rejects a single-score interpretation of a medical RAG pipeline. A useful system should report at least retrieval rank, report alignment, raw answer overlap, verified answer overlap, support, abstention, and resource cost.

## 5.2 Research contributions

### 5.2.1 Alignment-aware multimodal retrieval evaluation

The primary contribution is an evaluation design that treats image-report alignment as an explicit experimental variable. It compares the same text baseline with correctly aligned image reranking and fixed-point-free shuffled-image controls. This makes it possible to distinguish image contribution from accidental image association.

### 5.2.2 Layered grounding analysis

The study separates target-case alignment, report-level evidence support, answer-reference consistency, and automated verification. This prevents a high local support score from being mistaken for correct target-case grounding. The distinction is especially relevant when reports from different cases contain similar normal findings or common radiology phrasing.

### 5.2.3 Modernized cross-generator confirmation

V6 tests the retrieval mechanism under both Qwen2.5 and MedGemma 1.5. The difference-in-differences result provides a secondary view of whether retrieval benefit is robust to generator choice. The result is not “the newest model wins”; it is evidence that upstream retrieval changes can transfer across different generation regimes, with different magnitudes and verifier interactions.

### 5.2.4 Reproducible cohort and artifact controls

The confirmation cohort was constructed after protocol and configuration freeze using deterministic hash-based selection and assignment. Source, eligibility, cohort, model, output, and analysis hashes were recorded. The case-grouped statistics, exact model revisions, and technical rerun policy make the study auditable without publishing large restricted artifacts.

### 5.2.5 Transparent negative evidence

The study retains results that do not support a simple success narrative. Qwen3-Embedding underperformed BM25 on MRR. The normal subgroup showed little retrieval benefit. Qwen2.5 support and abstention behavior did not improve under the frozen verifier. These findings identify the boundary conditions of the proposed workflow.

## 5.3 Theoretical and practical implications

The results support a layered account of grounding. At the first layer, the system must retrieve the correct case. At the second, the selected report must contain evidence relevant to the question. At the third, the generator must express that evidence without unsupported additions or omissions. At the fourth, the verifier must measure and filter the answer without introducing excessive abstention. Performance at one layer cannot substitute for performance at another.

The alignment control also suggests that multimodal retrieval should be evaluated as a relational problem. The important object is not an isolated image embedding or an isolated report embedding, but the preserved relation between an image view, a case, and the report that describes it. A system that improves image-text similarity while breaking case identity can produce a misleading result.

For system design, the findings favor a modular architecture. BM25 remains a strong interpretable baseline. A modern image reranker can be added when the candidate pool is sufficiently narrow and image-report links are reliable. The generator should receive an explicit selected report rather than an opaque mixture of many contexts. The verifier should expose support and abstention traces instead of returning only one final score.

For deployment-oriented thinking, the cost table matters. MedGemma produced higher overlap but required substantially more generation time. A practical implementation could use a smaller model for interactive triage and reserve a larger medical model for high-value cases, but such a policy would require new validation and cannot be inferred directly from this benchmark.

## 5.4 Limitations

### Data source limitation

All confirmation cases came from OpenI/IU-Xray. The study therefore lacks external dataset validation. The broader normal/abnormal spectrum improves within-source coverage but does not remove source-specific reporting style, acquisition, and population limitations.

### Patient identity limitation

Case-ID disjointness was verified, but patient-level independence was unavailable. Multiple case records may therefore belong to the same person. The statistical unit was the processed case, and conclusions must be read with this limitation.

### Question provenance limitation

Questions were generated deterministically from report sections and were not physician-authored. They support reproducibility but may be easier and more templated than natural clinical questions. The study cannot claim robustness to the full diversity of clinician wording.

### Reference metric limitation

Token-F1 rewards overlap with one frozen textual reference. It can undervalue valid paraphrases, overvalue shared generic phrases, and fail to represent clinical importance. Verified Token-F1 is even more dependent on the behavior of the automated verifier. Neither metric is a substitute for physician adjudication.

### Verifier limitation

The unchanged V5 verifier was retained for attribution control, not because it is a clinical gold standard. Its high Qwen2.5 abstention rate demonstrates generator-verifier interaction. Independent human evaluation would be required to assess clinical adequacy, unnecessary abstention, and unsupported content.

### Model and engineering limitation

The study evaluated a selected set of models, one primary image encoder, one sparse primary text retriever, one secondary dense text retriever, and one frozen verifier. MedSigLIP report chunking and maximum similarity aggregation were study-specific engineering choices. Different encoders, chunk policies, fusion weights, or prompts may produce different results.

### Closed-set retrieval limitation

Every target report was present in the 240-case candidate pool. This creates a useful controlled test but does not represent the scale, noise, or missingness of a hospital archive. The result should be described as paired-report retrieval within a closed candidate set.

### Human evaluation limitation

The V5 qualitative review was researcher-reviewed and exploratory. It was not blinded preregistration, and no independent human rater panel produced clinical correctness scores. The current evidence therefore supports interpretation of failure patterns, not claims of clinically validated accuracy.

## 5.5 Future work

The first priority is independent human evaluation. Future raters should assess target-case alignment, factual support, clinical adequacy, unnecessary abstention, and harmful unsupported claims using a blinded protocol. Raters should receive clear definitions and should not be asked to infer a patient identity beyond the benchmark's case metadata. Inter-rater agreement and adjudication procedures should be reported.

The second priority is external validation. The same frozen retrieval and QA protocol should be evaluated on a permission-compliant, genuinely separate report-image source. Public auxiliary benchmarks may be used for complementary robustness analysis, but their provenance and overlap with OpenI/IU-Xray must be disclosed.

The third priority is broader question provenance. Physician-authored natural questions, unanswerable questions, paraphrases, laterality-sensitive cases, and uncertainty-focused questions would test whether the retrieval and abstention findings generalize beyond report-derived templates.

The fourth priority is verifier calibration. A new verifier should be calibrated against human labels rather than selected only through automated reference overlap. Its abstention threshold should be evaluated for sensitivity to generator style, answer length, and clinically meaningful uncertainty.

The fifth priority is open-corpus and patient-safe retrieval. Future studies should evaluate realistic candidate pools, missing reports, duplicate studies, repeated examinations, and reliable patient-level grouping. These extensions are necessary before any claim about clinical workflow can be considered.

## 5.6 Conclusion

This thesis developed and evaluated an auditable multimodal RAG workflow for paired radiology images and reports. The V6 confirmation study showed that correctly aligned MedSigLIP reranking improved closed-set target-report retrieval over the same BM25 baseline, exceeded 100 shuffled-image controls, and transferred positive verified Token-F1 gains to both Qwen2.5 and MedGemma 1.5.

The most important conclusion is bounded rather than absolute. Multimodal retrieval can add useful case-discriminative evidence, but the benefit depends on preserving image-report alignment and does not automatically guarantee a correct final answer. Retrieval, generation, and verification remain separate sources of error. The stronger MedGemma results came with higher computational cost, while the unchanged verifier showed high abstention for Qwen2.5. Normal cases also showed little retrieval gain, demonstrating that the effect is not uniform across the source spectrum.

The research value therefore lies in the evidence chain and its limits: it identifies when paired image information helps, demonstrates that alignment controls matter, tracks whether retrieval gains reach the answer layer, and makes uncertainty about clinical validity explicit. The resulting system is a research prototype for traceable evidence retrieval, not an autonomous diagnostic tool.
