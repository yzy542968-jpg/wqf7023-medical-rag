# Retrieval-Augmented Medical Question Answering over Paired Radiology Images and Reports

**Name:** ZHANG YUE  
**Matric No.:** 22097191  
**Programme:** Master of Artificial Intelligence  
**Course:** WQF7023 Artificial Intelligence Research Project  
**Supervisor:** Dr. Uzair Ishtiaq  
**Document status:** V5-integrated manuscript draft; technical and qualitative results frozen  
**Version date:** 19 August 2026

## Abstract

Retrieval-augmented generation can provide external evidence for medical question answering, but a fluent answer may still be grounded in the wrong patient's report. In radiology, linked images, indications, findings, and impressions form a case-level evidence unit. This research develops and critically evaluates a retrieval-augmented question-answering workflow over paired chest X-ray images and reports using de-identified OpenI/IU-Xray cases. The study asks whether correctly aligned image representations improve target-report retrieval beyond question and indication text, whether the effect depends on correct image-report alignment, and whether retrieval gains transfer to downstream report-grounded answers.

The final V5 experiment used 240 fresh cases, divided into 120 development and 120 confirmation cases, with 360 confirmation questions. BM25 provided text retrieval, while BioViL-T supplied 128-dimensional paired image-report representations for reranking. Four principal conditions compared question-only text, indication plus question, question plus correctly aligned image, and indication plus question plus correctly aligned image. One hundred fixed-point-free shuffled-image permutations provided an alignment control. The top-ranked report was passed through a fixed local Qwen2.5-1.5B-Instruct generator and an automated Medical-NLI evidence checker.

Indication text was the strongest retrieval signal: MRR increased from 0.0277 for question-only BM25 to 0.6590 for indication-plus-question BM25. Adding the correctly aligned image increased MRR to 0.6971 and extractive proxy Token-F1 from 0.6602 to 0.7245. The MRR difference was 0.0381 with case-bootstrap 95% CI [0.0159, 0.0614] and paired-randomization p=0.0012. No shuffled-image run reached the correctly aligned MRR or proxy Token-F1; the plus-one Monte Carlo value was 0.0099 for both metrics. Under the fixed downstream pipeline, multimodal retrieval improved final Token-F1 by 0.0302, CI [0.0101, 0.0511], p=0.0032, but automated support rate decreased by 0.0340.

A frozen 24-question qualitative review explained this trade-off. The researcher accepted all refined taxonomy v1.1 proposals. Target-rank improvement did not always produce Top-1 success; answers could be faithful to a retrieved report that was misaligned with the frozen target case; correct retrieval did not guarantee a reference-consistent final answer; and automated verification sometimes appeared to remove report-supported content. Other support declines reflected only template-prefix filtering rather than substantive answer loss.

The principal contribution is an auditable multimodal RAG evaluation that separates target-case alignment, report-level faithfulness, generation, verification, and abstention. The results support an alignment-specific image contribution to paired-report retrieval, but they do not establish autonomous image diagnosis, clinical correctness, external validation, or deployment safety.

**Keywords:** retrieval-augmented generation, medical question answering, radiology, chest X-ray, multimodal retrieval, BioViL-T, evidence grounding, image-report alignment

# Chapter 1: Introduction

## 1.1 Background

Large language models (LLMs) can generate fluent answers, summaries, and explanations, including responses involving medical knowledge. Fluency, however, does not establish evidential or clinical reliability. Medical LLM studies distinguish benchmark performance from clinical readiness, while hallucination benchmarks show that plausible medical statements may be unsupported or incorrect (Singhal et al., 2023; Pal et al., 2023).

Retrieval-augmented generation (RAG) addresses part of this problem by retrieving external evidence before answer generation (Lewis et al., 2020). The retrieved context can improve factual coverage and expose a provenance path. Nevertheless, RAG is a pipeline rather than one model. The query may be ambiguous, the retriever may select the wrong document, the generator may omit or alter evidence, and an automated verifier may approve an answer that is locally faithful to an incorrectly retrieved record.

These distinctions are especially important in radiology. A radiology examination links an indication, one or more images, findings, and an impression. The report and images describe the same case, but they contribute different signals. Indication text may contain strong lexical clues. Image representations may help distinguish visually different examinations. Report text supplies the evidence from which an answer can be generated. If these elements are separated or mismatched, an answer may remain coherent while referring to the wrong case.

The OpenI/IU-Xray collection contains de-identified chest radiograph examinations with linked reports and images (Demner-Fushman et al., 2016). It enables controlled study of paired image-report retrieval without requiring proprietary hospital data. Biomedical vision-language models such as BioViL and BioViL-T provide joint representations of radiology images and reports (Boecking et al., 2022; Bannur et al., 2023), creating an opportunity to test whether image information improves report retrieval rather than merely decorating a text-only interface.

This thesis therefore studies retrieval-augmented medical question answering over paired radiology images and reports. Images are used as a retrieval and reranking signal. Answers remain grounded in the selected radiology report, and the system does not claim to diagnose an image directly.

## 1.2 Problem Statement

Many RAG evaluations assume that the question identifies one relevant document and that support found in retrieved context is sufficient. These assumptions are weak for case-based medical data. Generic questions such as “What are the findings?” do not identify a unique examination. Referral indications may make retrieval much easier through lexical shortcuts. An image encoder may improve target ranking without placing the correct report first. A generator may then produce an answer that is faithful to a wrong but clinically similar report.

This creates a layered grounding problem:

```text
Sentence supported by selected evidence
                does not imply
Selected evidence belongs to the frozen target case
                does not imply
Answer is clinically correct or safe
```

Automated evidence checking adds another risk. A semantic verifier may remove a report-supported sentence, retain an unsupported sentence, or reduce a support score because it filtered only a generic answer prefix. Consequently, retrieval metrics, answer overlap, and support scores must be examined together rather than interpreted as interchangeable measures of correctness.

The core problem is therefore:

> How can paired radiology images and reports be used in an auditable RAG workflow that improves target-report retrieval, preserves case alignment, and explains downstream generation and verification failures without overstating clinical validity?

## 1.3 Research Aim

The aim is to develop and critically evaluate a reproducible multimodal retrieval-augmented workflow for medical question answering over paired chest X-ray images and radiology reports.

## 1.4 Research Objectives

The objectives are:

1. Process real OpenI/IU-Xray examinations into reproducible case-level records that preserve image-report pairing.
2. Measure patient-scope ambiguity and the retrieval shortcut introduced by indication text.
3. Compare text-only retrieval with BioViL-T image-conditioned reranking under fixed candidate and fusion policies.
4. Test whether retrieval gains depend on correct image-report alignment using fixed-point-free shuffled-image controls.
5. Evaluate whether multimodal retrieval gains transfer to downstream report-grounded generation under a fixed Qwen and semantic-checker pipeline.
6. Separate target-case alignment, report-level faithfulness, reference overlap, post-verification content loss, and abstention.
7. Conduct a frozen, researcher-reviewed qualitative analysis of representative retrieval, generation, verification, and data-ambiguity cases.
8. Report reproducibility, computational cost, limitations, and unsupported claims explicitly.

## 1.5 Research Questions

**RQ1.** How strongly do indication text and correctly aligned image representations affect target-report retrieval in a paired OpenI/IU-Xray benchmark?

**RQ2.** Does the image contribution depend on correct image-report alignment rather than the presence of arbitrary image features?

**RQ3.** Does multimodal retrieval improve downstream report-grounded QA under a fixed generation and semantic-verification pipeline, and what happens to automated evidence support?

**RQ4.** Which retrieval, generation, verification, abstention, and data-quality patterns explain the remaining errors and metric trade-offs?

## 1.6 Research Contributions

This thesis makes five scoped contributions.

First, it provides a reproducible paired image-report retrieval and QA workflow over real de-identified chest X-ray cases. The pipeline combines BM25, BioViL-T reranking, local Qwen generation, semantic evidence checking, abstention, and trace preservation.

Second, it introduces an alignment-specific evaluation design. Indication ablation measures a major text shortcut, while 100 fixed-point-free shuffled-image permutations distinguish correct alignment from generic image-conditioned reranking.

Third, it demonstrates that report-level faithfulness and target-case alignment are separate requirements. An answer can follow the selected report yet remain misaligned with the frozen target case.

Fourth, it evaluates whether multimodal retrieval improvement transfers to end-to-end QA and reports the resulting performance-grounding trade-off rather than selecting only favorable metrics.

Fifth, it provides a frozen qualitative protocol, a preserved v1.0-to-v1.1 taxonomy mapping, researcher-reviewed case coding, runtime evidence, artifact hashes, and public reproduction entry points.

The contribution is not a claim of state-of-the-art clinical QA. It is a controlled study of image-report alignment and evidence ownership in a multimodal medical RAG pipeline.

## 1.7 Scope and Boundaries

The final experiment is limited to OpenI/IU-Xray and therefore remains within one data source. It uses a closed set of 240 candidate cases, 120 confirmation targets, and three report-derived question templates. The image encoder is frozen; the project does not train a new vision-language foundation model.

Images influence retrieval and reranking, but the Qwen answer generator receives selected report evidence rather than image pixels. The system therefore does not evaluate autonomous visual diagnosis. Token-F1, retrieval qrels, and automated semantic signals do not establish clinical correctness. The researcher-reviewed qualitative analysis is explanatory and is not independent radiologist adjudication.

The dashboard is a research demonstration rather than an authenticated hospital application. No treatment recommendation, clinical decision support, deployment safety, or external validation claim is made.

## 1.8 Thesis Organization

Chapter 2 reviews RAG, biomedical retrieval, multimodal radiology representation, medical QA, evidence checking, and benchmark validity. Chapter 3 describes the paired image-report cohort, retrieval conditions, shuffled-image control, generation, verification, statistics, qualitative protocol, and reproducibility. Chapter 4 reports quantitative, qualitative, and runtime results. Chapter 5 interprets the findings, contributions, limitations, future work, and conclusion.

# Chapter 2: Literature Review

## 2.1 Retrieval-Augmented Generation

RAG combines a parametric language model with retrieved non-parametric evidence for knowledge-intensive generation (Lewis et al., 2020). This design can expose sources and update knowledge without retraining the complete generator. It also creates a multi-stage failure surface. Retrieval determines which evidence is available, generation determines how that evidence is expressed, and verification determines which claims are retained or flagged.

RAG evaluation should therefore separate retrieval relevance, answer relevance, and faithfulness. RAGAS formalizes several of these dimensions using automated metrics (Es et al., 2024). The present research adds target-case alignment as a separate dimension. In medical records, support from a related document is not equivalent to support from the intended examination.

## 2.2 Medical RAG and Question Answering

Medical RAG performance depends on the corpus, task, retriever, and generator. MedRAG/MIRAGE showed that retrieval can improve medical QA but that gains vary across datasets and configurations (Xiong et al., 2024). Practical evaluations also emphasize noisy, misleading, or insufficient evidence rather than assuming ideal retrieval (Ngo et al., 2024).

These findings support controlled component comparison. A high final answer score cannot reveal whether improvement originated from a text shortcut, correct retrieval, copied context, generation, or verification. Negative findings such as weak Top-1 improvement or reduced automated support are therefore evidence about system boundaries rather than results to hide.

## 2.3 Sparse, Dense, and Multimodal Retrieval

BM25 remains a strong transparent sparse-retrieval baseline based on probabilistic term matching (Robertson and Zaragoza, 2009). It is effective when a query shares terminology with a report, but it is sensitive to lexical overlap and may exploit benchmark shortcuts. Indication text can be especially discriminative because it describes symptoms, history, and reason for examination.

Dense retrieval encodes queries and documents in a shared vector space. MedCPT uses large-scale PubMed search logs for biomedical retrieval (Jin et al., 2023). General vision-language systems such as CLIP align images and text using contrastive learning (Radford et al., 2021). In radiology, domain-specific joint encoders can represent chest X-rays and reports within a medically relevant embedding space.

Hybrid and reranking systems combine complementary signals but do not guarantee Top-1 correctness. A target can move substantially upward while remaining below rank one. This distinction motivates reporting both rank-sensitive metrics such as MRR and decision metrics such as Hit@1.

## 2.4 Paired Radiology Images and Reports

The OpenI/IU-Xray collection was prepared for radiology distribution and retrieval research and contains chest X-ray images linked to reports (Demner-Fushman et al., 2016). The report commonly includes an indication, findings, and impression. These fields should remain associated with the same examination throughout preprocessing and evaluation.

Paired radiology data enable at least three different tasks. Image classification predicts labels from pixels. Image-report retrieval ranks matching images or reports. Visual question answering generates answers from image content. The present study evaluates paired-report retrieval followed by report-grounded QA. It does not equate retrieval of a matching report with diagnosis from an image.

## 2.5 Biomedical Vision-Language Representation

BioViL introduced radiology-specific image-text representation learning with localized and global alignment between chest X-rays and reports (Boecking et al., 2022). BioViL-T extended biomedical vision-language processing by exploiting temporal and multi-image structure (Bannur et al., 2023). Such encoders offer a stronger domain prior than generic visual embeddings for chest radiograph retrieval.

This thesis uses frozen BioViL-T representations as a reranking signal. Freezing the encoder limits computational cost and supports reproducibility, but it also limits the claim: the research evaluates the usefulness of an existing representation within RAG rather than proposing a new image encoder.

Multi-view examinations introduce an aggregation question. Frontal and lateral views may carry complementary information. The final system normalizes each view, averages views at case level, and normalizes the aggregate. This deterministic policy avoids learned fusion on the confirmation outcomes.

## 2.6 Medical Visual and Report Question Answering

VQA-RAD contains clinically generated questions and answers about radiology images and demonstrates the value of natural clinician phrasing for visual QA (Lau et al., 2018). EHRXQA combines electronic health records and chest X-rays for multimodal QA (Bae et al., 2023). These tasks are relevant but differ from retrieving one paired report from a candidate corpus.

RadQA contains physician-authored questions, report contexts, answer spans, and unanswerable cases (Soni et al., 2022). It is a valuable future report-QA benchmark, but authorized PhysioNet access was not part of the frozen V5 experiment. The present questions are generated from report roles and are therefore controlled but linguistically narrow.

The distinction matters because a model can perform well on templated questions through metadata or section shortcuts. Results from report-derived templates should not be generalized to unrestricted clinical questions.

## 2.7 Evidence Grounding and Medical Hallucination

Medical hallucinations may be fluent and difficult to identify from wording alone (Pal et al., 2023). Evidence checking can use lexical matching, semantic similarity, natural-language inference, or combinations of these signals. MedNLI established a clinical-domain NLI task using sentence pairs derived from clinical notes (Romanov and Shivade, 2018).

An NLI-based verifier still has a restricted scope. It may determine that a sentence follows from the selected report without determining whether the report belongs to the intended case. It may also reject paraphrases, mishandle composite claims, or react to de-identification placeholders. This thesis therefore separates:

- target-case alignment;
- report-level faithfulness;
- reference consistency;
- automated support;
- clinical validity.

Only the first four receive partial automatic or researcher-reviewed evidence. Clinical validity is not established.

## 2.8 Alignment Controls and Benchmark Validity

Multimodal improvement can be misattributed if text already identifies the answer or if any image embedding changes score distributions. Input ablation is needed to reveal text shortcuts. Alignment controls are needed to show that performance depends on the correctly paired image rather than arbitrary visual features.

The V5 shuffled-image condition uses fixed-point-free derangements so that no case retains its own image. Comparing correct alignment with many deterministic derangements produces an empirical null distribution while holding text, candidates, and fusion policy constant. The plus-one correction avoids reporting an exact zero probability from a finite permutation sample.

Benchmark construction can create additional shortcuts. Repeated generic questions make open-corpus retrieval underidentified, while indications copied from target reports make lexical retrieval easier. These properties must be reported as characteristics of the task rather than credited to model reasoning.

## 2.9 Agentic and Auditable RAG Workflows

Agentic RAG commonly refers to workflows that plan, retrieve, rerank, generate, verify, or abstain. The term should be used carefully. A deterministic policy is not learned reasoning, and an automated verifier is not an independent clinical judge.

The implemented workflow is agentic in a bounded engineering sense: it records retrieval intent, executes retrieval and optional image reranking, generates from selected evidence, audits sentences, and either retains, filters, or abstains. Its main value is traceability. Each action can be inspected separately, which allows errors to be attributed to retrieval, generation, verification, abstention, or data ambiguity.

## 2.10 Research Gap

Prior work establishes RAG, medical QA, biomedical retrieval, radiology vision-language encoders, and evidence checking. A narrower gap remains at their intersection: how should paired radiology images and reports be evaluated when the system must retrieve the correct case before answering, and what does report-level faithfulness mean when case alignment can fail?

Many evaluations report only final answer accuracy or image-text retrieval. Fewer connect indication shortcuts, correct-versus-shuffled image alignment, Top-1 target-case retrieval, downstream report-grounded generation, semantic verification, and stage-specific qualitative analysis within one frozen experiment.

This thesis addresses that gap through a fresh paired-case cohort, explicit input ablations, fixed-point-free shuffled-image controls, a non-oracle downstream QA path, case-grouped statistics, preserved artifact hashes, and researcher-reviewed error attribution. The central proposition is not that multimodal RAG eliminates medical error, but that it can make image-report alignment and evidence ownership measurable.

# Chapter 3: Methodology

## 3.1 Research Design

This study used a staged empirical system-comparison design to investigate retrieval-augmented medical question answering over paired radiology images and reports. Earlier text-only experiments identified two structural risks: open-corpus retrieval could select evidence from the wrong case, while a sentence-level verifier could still rate an answer as supported by that wrongly selected report. These findings motivated the final V5 experiment, which tested whether correctly paired chest X-ray information could improve target-report retrieval and whether any retrieval gain transferred to downstream report-grounded question answering.

V5 was the final technical experiment. Its configuration was specified and frozen locally before execution, but it was not formally preregistered or externally timestamped before outcomes were observed. The confirmation cohort was disjoint from all previous project cohorts, although it remained drawn from the same OpenI/IU-Xray source. V5 therefore provides fresh within-source confirmation rather than external validation.

The experiment addressed four linked questions:

1. How strongly do indication text and correctly aligned images affect paired-report retrieval?
2. Is the image contribution specific to the correct image-report alignment rather than generic image features?
3. Does multimodal retrieval improve downstream report-grounded QA under a fixed generation and verification pipeline?
4. What retrieval, generation, verification, and resource trade-offs remain after multimodal reranking?

No V5 model, prompt, threshold, cohort, or result was changed in response to the final quantitative or qualitative analysis.

## 3.2 Data Source and Cohort Construction

The study used de-identified OpenI/IU-Xray chest radiograph cases with linked reports and one or more image views. Each processed case contained a stable case identifier, indication, findings, impression, problem labels, and linked image metadata. De-identification placeholders were retained because replacing or inferring their hidden content could introduce unsupported information.

The V5 cohort contained 240 cases that were excluded from all earlier project cohort manifests. A fixed seed of 7023 divided these into 120 development cases and 120 confirmation cases. The confirmation set contributed 360 report-derived questions: one findings question, one impression question, and one summary question per case. Statistical resampling and comparison used the case identifier as the grouping unit so that the three questions from one case were not treated as independent patients.

Retrieval used all 240 fresh-cohort cases as the candidate pool. The 120 confirmation cases were the target cases for final evaluation. This design measured closed-set paired-report retrieval; it did not evaluate diagnosis of previously unseen patients.

## 3.3 Question and Input Conditions

Each target report generated three fixed question forms:

- findings: what radiographic findings were documented;
- impression: what final radiology impression was reported;
- summary: what principal abnormality or conclusion was reported.

These questions support controlled comparison but are not radiologist-authored natural questions. They also contain limited linguistic diversity, so V5 does not establish general free-form planning ability.

Four principal input conditions were compared in the main retrieval table:

1. question-only BM25;
2. indication plus question BM25;
3. question plus correctly aligned image;
4. indication plus question plus correctly aligned image.

A fifth condition, indication plus question with shuffled-image alignment, was evaluated separately as a negative control. Keeping shuffled images outside the main four-condition table separated ordinary input ablation from the alignment-specific test.

## 3.4 Case-Aware Evidence Representation and Downstream QA Workflow

V5 did not assume that the correct report was already known. For every confirmation question, the retriever ranked reports from the 240-case candidate pool. The top-ranked report was then passed to the downstream QA pipeline. Correct-case retrieval was determined only during evaluation using the frozen target case identifier.

The workflow was:

1. construct the text query from the question, with or without indication;
2. obtain a BM25 shortlist from the candidate reports;
3. optionally rerank that shortlist using the paired-image representation;
4. select the top-ranked candidate report;
5. generate an answer from that selected report using a fixed local Qwen model;
6. audit generated sentences against the selected report using the frozen semantic checker;
7. filter unsupported sentences or abstain according to the locked action policy;
8. preserve the retrieval, generation, evidence, and action trace for evaluation.

This architecture distinguished two forms of grounding. Report-level grounding asked whether the answer was supported by the selected report. Target-case alignment asked whether that selected report belonged to the frozen target case. An answer could satisfy the first condition while failing the second.

## 3.5 Text Retrieval

BM25 provided the transparent sparse retrieval baseline. The question-only condition intentionally exposed patient-scope ambiguity because the three generic question templates contained little case-specific information. The indication-plus-question condition tested how much clinical referral text reduced this ambiguity.

The multimodal conditions first produced the same text shortlist of 100 candidate cases. Scores were normalized independently within that shortlist. Ties were resolved by descending fused score and then ascending case identifier, giving a deterministic ranking.

## 3.6 Image Encoding and Multimodal Reranking

Image and report representations used `microsoft/BiomedVLP-BioViL-T` with frozen revision `692f09e` and 128-dimensional embeddings. Each available X-ray view was normalized, case views were averaged, and the resulting case vector was normalized again. No image pixels were sent to an online service.

For multimodal reranking, normalized text and image similarities received equal weights of 0.5. The correctly aligned condition used the image embedding linked to the target case. This image was used only as a retrieval query signal; the system did not generate a diagnosis directly from pixels. The downstream generator received the selected report evidence rather than raw images.

## 3.7 Shuffled-Image Control

The alignment control used 100 deterministic fixed-point-free permutations with seed 7023. In each permutation, every source case received another case's image embedding and no case retained its own image. Text queries, candidate reports, shortlist size, fusion weights, and evaluation procedure remained unchanged.

The control tested whether the correctly aligned image outperformed image-conditioned reranking with incorrect case alignment. It did not prove causal clinical image understanding. A plus-one Monte Carlo value was calculated as `(b+1)/(m+1)`, where `b` was the number of shuffled runs meeting or exceeding the correctly aligned result and `m=100` was the number of permutations.

## 3.8 Answer Generation and Semantic Verification

Both report-only and multimodal retrieval conditions used the same local `Qwen/Qwen2.5-1.5B-Instruct` generator. Generation used CUDA, float16, batch size 16, maximum 256 new tokens, temperature 0, and a direct non-oracle prompt. The generator did not receive the frozen target identifier or reference answer.

The semantic checker used `pritamdeka/PubMedBERT-MNLI-MedNLI`. It combined lexical evidence matching, entailment and contradiction probabilities, and polarity consistency. Its locked configuration used lexical weight 0.2, support threshold 0.6, entailment threshold 0.75, and contradiction threshold 0.5. Evidence scope was restricted to the top-ranked selected report. The action path could retain supported sentences, filter flagged sentences, or abstain if no usable answer remained.

The checker was an automated evidence signal rather than a clinical gold standard. Its support rate measured agreement with selected-report evidence, not target-patient correctness or clinical safety.

## 3.9 Evaluation Metrics and Statistical Analysis

Retrieval metrics were Hit@1, Hit@5, Hit@10, MRR, and an extractive proxy Token-F1 calculated from the selected report evidence. Hit@1 measured Top-1 target-case alignment; MRR retained information about target-rank movement even when the target did not reach first place.

QA metrics were draft Token-F1, final Token-F1 after semantic checking, automated evidence-support rate, revision rate, and abstention rate. Token-F1 measured reference overlap and was not interpreted as clinical correctness.

V5 used 5,000 grouped bootstrap resamples at case level and paired randomization tests with seed 7023. The primary retrieval comparison was indication-plus-question with correctly aligned image minus indication-plus-question BM25. The primary QA comparison was multimodal final Token-F1 minus report-only final Token-F1. Confidence intervals and p-values therefore preserved the dependence among questions from the same case.

## 3.10 Researcher-Reviewed Qualitative Analysis

A post-hoc qualitative protocol was committed after the technical freeze but before systematic case extraction and coding. Some individual outputs had previously been inspected during pipeline verification, so this was not a result-blind preregistration.

The fixed protocol selected 24 representative questions: six target-rank improvements, six target-rank degradations, six QA-gain/support-loss cases, and six correct-retrieval generation-error cases. Each stratum contained two findings, two impression, and two summary questions. The full 360-question numeric index was retained.

Protocol taxonomy v1.0 was preserved in the audit trail. During interpretation, a refined three-level taxonomy v1.1 separated pipeline stage, specific pattern, and outcome modifier. It distinguished target-rank movement from Top-1 success, generation omission from post-verification content loss, and abstention occurrence from its suspected cause. Assistant-proposed v1.1 labels were recorded separately from the original labels. The researcher reviewed and accepted all 24 proposals on 19 August 2026, producing 24 accepted, 0 modified, and 0 excluded cases.

Qualitative counts describe only this predefined purposive review set. They were not used for population-level inference, verifier accuracy estimation, or clinical error-rate estimation.

## 3.11 Computational Cost and Reproducibility

The frozen manifest stored the cohort fingerprint and LF-normalized SHA-256 values for configurations, code, aggregate results, and tests. Large generations, prompt packs, image pixels, model weights, and private full-text review rows remained local.

Generation timing was measured on an NVIDIA GeForce RTX 5070 Laptop GPU with 8,150.6 MiB total memory. These values are machine-, cache-, and generated-length-dependent and do not constitute a complete production latency or energy analysis.

## 3.12 Ethics and Claim Boundaries

The system was a research prototype. It did not provide treatment recommendations, authenticate clinical users, or claim deployment safety. V5 did not establish image-based diagnosis, clinical causality, external validation, natural-question generalization, or human-validated verifier correctness. Images and reports were processed locally, and no attempt was made to reverse de-identification.

# Chapter 4: Results and Analysis

## 4.1 Patient-Scope Ambiguity and the Indication Shortcut

Table 4.1 shows the four principal confirmation retrieval conditions.

| Input condition | Hit@1 | Hit@5 | Hit@10 | MRR | Extractive proxy Token-F1 |
|---|---:|---:|---:|---:|---:|
| Question only, BM25 | 0.0056 | 0.0222 | 0.0472 | 0.0277 | 0.1981 |
| Indication + question, BM25 | 0.5889 | 0.7222 | 0.7750 | 0.6590 | 0.6602 |
| Question + correctly aligned image | 0.0139 | 0.0722 | 0.1139 | 0.0515 | 0.2334 |
| Indication + question + correctly aligned image | 0.6222 | 0.7778 | 0.8389 | 0.6971 | 0.7245 |

Question-only retrieval was nearly non-identifying: Hit@1 was 0.0056 and MRR was 0.0277. This was expected because the same three templates were reused across cases. Adding indication increased Hit@1 to 0.5889 and MRR to 0.6590. The indication therefore acted as a powerful retrieval shortcut in this controlled benchmark.

The effect is methodologically important. A high retrieval score cannot be attributed only to sophisticated multimodal reasoning when referral text already contains strong case-discriminating language. For this reason, V5 reports indication ablation explicitly and treats the indication-plus-question BM25 condition as the primary text baseline.

## 4.2 Indication and Correct-Image Ablation

The correctly aligned image produced a small improvement when used with question text alone: MRR rose from 0.0277 to 0.0515 and proxy Token-F1 from 0.1981 to 0.2334. These values remained low because the generic question supplied little textual case identity.

Against the stronger indication-plus-question BM25 baseline, correctly aligned image reranking increased MRR by 0.0381, with case-bootstrap 95% CI [0.0159, 0.0614] and paired-randomization p=0.0012. Proxy Token-F1 increased by 0.0643, CI [0.0282, 0.1029], p=0.0006. Hit@5 increased by 0.0556 and Hit@10 by 0.0639, with paired-randomization p=0.0024 and p=0.0052 respectively.

The Hit@1 increase was smaller: +0.0333, from 0.5889 to 0.6222. Its confidence interval reached approximately zero and the paired-randomization p-value was 0.0886. Thus, the strongest evidence concerns improved target ordering and retrieval within the upper ranks, not a definitive Hit@1 improvement.

## 4.3 Correctly Aligned Versus Shuffled Images

Correct alignment achieved MRR 0.6971 and proxy Token-F1 0.7245. Across 100 shuffled-image derangements, mean MRR was 0.5659 with range [0.5158, 0.6084], while mean proxy Token-F1 was 0.5950 with range [0.5310, 0.6455]. No shuffled run equalled or exceeded the correctly aligned result for either metric.

The plus-one Monte Carlo value was 0.0099 for both MRR and proxy Token-F1. The result supports an alignment-specific contribution: the benefit was not reproduced by attaching arbitrary image embeddings to the same text workflow. It does not prove clinical image interpretation, because the task remained closed-set paired-report retrieval and did not test diagnosis from pixels.

## 4.4 End-to-End Question Answering

Table 4.2 compares the same generator and checker after report-only and multimodal retrieval.

| Pipeline | Draft Token-F1 | Final Token-F1 | Automated support | Final abstention | Revision rate |
|---|---:|---:|---:|---:|---:|
| Report-only retrieval | 0.3632 | 0.3563 | 0.8409 | 0.0556 | 0.7389 |
| Multimodal retrieval | 0.3897 | 0.3865 | 0.8069 | 0.0611 | 0.7250 |

Multimodal retrieval improved draft Token-F1 by 0.0265, CI [0.0094, 0.0441], paired-randomization p=0.0026. Final Token-F1 improved by 0.0302, CI [0.0101, 0.0511], p=0.0032. This demonstrates that the retrieval gain transferred to the final QA output under a fixed non-oracle generation path.

However, automated evidence support decreased by 0.0340, CI [-0.0566, -0.0122], p=0.0034. Final abstention increased by only 0.0056, with an interval crossing zero and p=0.7299. The central result is therefore a performance-grounding trade-off: reference overlap improved while the automated support signal declined.

This trade-off must not be simplified into a claim that multimodal answers were less clinically faithful. The support metric was produced by the same automated checker later shown to filter both substantive sentences and generic answer prefixes.

## 4.5 Researcher-Reviewed Qualitative Findings

The 24-case review package contained 19 unique cases and eight questions of each type. Relative to protocol v1.0, assistant interpretation was unchanged for nine rows and refined for 15. The researcher accepted all 24 v1.1 proposals without further modification.

The overlapping accepted labels included 14 Top-1 retrieval failures, 10 Top-1 retrieval successes, 11 target-rank improvements, 7 target-rank degradations, 10 post-verification content-loss cases, 9 possible verifier-over-rejection cases, and 6 QA-gain/support-loss cases. These values characterize the selected review package only.

Five exploratory findings were supported:

1. **Target-rank improvement did not always translate into Top-1 retrieval success.** The six extreme improvement examples moved targets from ranks 59-98 to ranks 10-27, but none reached first place.
2. **Report-level faithfulness did not guarantee alignment with the frozen target case.** Several answers accurately summarized an incorrectly selected report.
3. **Correct Top-1 retrieval did not guarantee a reference-consistent final answer.** One clear generation-focus case emphasized pectus deformity while omitting the report conclusion of no acute disease.
4. **In reviewed cases, automated verification sometimes appeared to remove report-supported content.** Five of six selected correct-retrieval generation-error rows contained filtered sentences, and three ended in abstention.
5. **Some declines in automated evidence-support scores did not correspond to substantive answer degradation.** In two reviewed impression cases, the checker removed only a generic answer prefix while preserving the complete reference-consistent conclusion.

Additional cases exposed data limitations. One de-identification token prevented confident adjudication of whether “Lungs are clear” was supported. Another report contained an apparent left-versus-right upper-lobe inconsistency between findings and impression. These examples show that data quality can affect both generation evaluation and verifier interpretation.

## 4.6 Computational Cost

| Pipeline condition | Records | Total process | Generation only | Generation throughput | Peak allocated GPU memory |
|---|---:|---:|---:|---:|---:|
| Report-only | 360 | 87.86 s | 78.56 s | 4.58 records/s | 3,437 MiB |
| Multimodal | 360 | 98.70 s | 89.31 s | 4.03 records/s | 3,437 MiB |

Both QA runs used the same Qwen model and generation settings. The multimodal prompt path took approximately 10.84 seconds longer in total and generated 0.55 fewer records per second, while peak allocated memory remained effectively unchanged. Earlier V4.2 measurements recorded approximately 14.91 ms mean single-image encoding, 1.73 ms BM25 retrieval, 0.28 ms cached reranking, and a 16.93 ms warm paired-request estimate.

These measurements show that the final system was feasible on a laptop GPU. They do not provide complete end-to-end production latency, energy consumption, or deployment cost.

## 4.7 Results Summary

V5 established four quantitative conclusions. Indication was the strongest single retrieval signal. Correctly aligned image reranking provided additional target-ordering and proxy-answer gains beyond indication text. Shuffled images did not reproduce the correct-alignment result. The retrieval improvement transferred to final QA Token-F1 but coincided with lower automated support.

The researcher-reviewed analysis explained why aggregate metrics moved differently. Some rank improvements stopped short of Top-1, some wrong-report answers remained locally grounded, some correct-report drafts lost content during verification, and some support-rate decreases reflected template filtering rather than substantive answer loss.

# Chapter 5: Discussion and Conclusion

## 5.1 Answers to the Research Questions

### RQ1: How do indication and aligned images affect paired-report retrieval?

Indication transformed an almost non-identifying question-only task into a substantially easier retrieval task, increasing MRR from 0.0277 to 0.6590. Correctly aligned image reranking then increased MRR to 0.6971 and improved upper-rank retrieval and proxy Token-F1. The image contribution was incremental rather than dominant and should be interpreted relative to the strong indication shortcut.

### RQ2: Was the image contribution alignment-specific?

Yes within this closed-set benchmark. None of 100 fixed-point-free shuffled-image controls reached the correctly aligned MRR or proxy Token-F1. This supports the claim that correct image-report pairing contributed useful retrieval information. It does not establish diagnostic reasoning or generalization to new clinical images.

### RQ3: Did retrieval improvement transfer to downstream QA?

Yes for automatic reference overlap. Multimodal retrieval improved final Token-F1 by 0.0302 with a case-bootstrap interval excluding zero. The same pipeline reduced automated support rate by 0.0340. Better retrieval therefore improved one outcome while exposing limitations in automated grounding measurement and verification behavior.

### RQ4: What failure modes remained?

The remaining failures occurred at several stages. Target rank could improve without reaching Top-1. Wrong-report answers could be internally supported but misaligned with the frozen target case. Correct retrieval could still be followed by generation-focus error. Finally, checker filtering could remove report-supported content or only remove harmless template prefixes. These distinct mechanisms cannot be represented by one aggregate support score.

## 5.2 Research Contributions

The first contribution is a reproducible paired image-report retrieval and QA pipeline over real OpenI/IU-Xray cases. The system links text retrieval, BioViL-T image reranking, local generation, sentence-level evidence checking, abstention, and trace preservation.

The second contribution is an alignment-specific evaluation design. Indication ablation prevents the image effect from being confused with referral-text shortcuts, while fixed-point-free shuffled images test whether gains depend on the correct image-report pairing.

The third contribution is evidence that report-level faithfulness and target-case alignment are separate requirements. This extends the earlier cross-case contamination finding: an answer can be well supported by retrieved evidence even when that evidence belongs to the wrong case.

The fourth contribution is a stage-specific qualitative taxonomy with an auditable v1.0-to-v1.1 mapping. It separates retrieval movement, Top-1 outcome, generation behavior, post-verification loss, abstention, and data ambiguity without overwriting the frozen protocol labels.

The fifth contribution is transparent negative evidence. The study reports that Hit@1 evidence was weaker than MRR evidence, support rate declined despite higher Token-F1, some verifier actions appeared excessive, and automatic metrics did not constitute clinical validation.

## 5.3 Theoretical and Practical Implications

The study supports a layered definition of grounding. Sentence-level support asks whether an answer claim appears in selected evidence. Report-level support asks whether the answer is faithful to the selected report. Target-case alignment asks whether the report is associated with the intended case. These layers are related but not interchangeable.

For system design, the result implies that evidence ownership should be checked before local faithfulness. A verifier applied only after retrieval cannot repair a wrong-case selection if it is restricted to asking whether the answer follows from the selected report. Retrieval traces should therefore expose both the selected case and the evidence used for each answer sentence.

The shuffled-image result also supports using paired images as a reranking signal when patient identity is genuinely unknown within the research task. In a real clinical workflow where an authorized patient record identifier already exists, identity should not be inferred from visual similarity. Authentication and record scope should be enforced first.

Finally, the support-rate trade-off shows that automated verifier metrics require their own evaluation. Lower support may represent removal of unsupported content, over-rejection of supported content, or filtering of harmless formatting. Treating support rate as a gold-standard faithfulness score would conceal these mechanisms.

## 5.4 Limitations

The study used one data source. The confirmation cohort was disjoint from prior project cohorts but remained within OpenI/IU-Xray, so the results are not external validation. The task used 240 candidate cases and 120 confirmation targets, which is much smaller and more controlled than a clinical archive.

The three questions were report-derived templates rather than radiologist-authored natural questions. Indication text was highly discriminative and may not reflect all real QA scenarios. References were inherited from report sections, and Token-F1 measured wording overlap rather than clinical correctness.

The image encoder was frozen and evaluated as a retrieval signal. The project did not train a vision-language model, diagnose images, localize pathology, or test image-report consistency with independent image-level annotations. The aligned-image result therefore supports paired-report retrieval, not autonomous visual diagnosis.

Only Qwen2.5-1.5B-Instruct and one frozen semantic checker were evaluated in the final path. Larger or clinically specialized generators might behave differently. The checker was not validated against independent expert entailment labels.

The qualitative set was purposively selected by frozen rules and reviewed by the researcher rather than an independent radiologist. Its counts cannot estimate population prevalence, clinical error rates, verifier sensitivity, or safety. The assistant contributed initial coding, although the original and refined labels were kept separately for auditability.

Runtime measurements came from a single laptop GPU and were not complete component-wise production benchmarks. No energy analysis, concurrent-load test, security assessment, or hospital-system integration was performed.

## 5.5 Future Work

The highest-priority extension is external evaluation on physician-authored report QA with natural unanswerable questions. RadQA remains appropriate once authorized access is available. Public auxiliary evaluation can use report-grounded datasets while clearly distinguishing datasets derived from the same IU-Xray source from truly external validation.

A stronger benchmark should use free-form clinical questions, independently annotated evidence spans, hard negative reports, and patient-level splits across institutions. Planner evaluation should separately score query reformulation, evidence-type selection, retrieval, reranking, generation, verification, and abstention.

Future verifier studies should obtain independent labels for entailment, contradiction, unsupported additions, composite claims, and appropriate abstention. They should report risk-coverage behavior rather than treating one threshold as universally valid.

Further multimodal work could compare BioViL-T with alternative medical image-text encoders, test multi-view fusion policies, evaluate calibration, and measure performance as the candidate pool grows. These experiments should preserve correct-versus-shuffled alignment controls.

Independent radiologist review remains desirable. It should assess answer correctness, evidence grounding, target-case alignment, harmfulness, and whether verifier filtering removed clinically relevant content. This remains future work rather than a fabricated result.

## 5.6 Conclusion

This thesis investigated retrieval-augmented medical question answering over paired chest X-ray images and radiology reports. The final V5 experiment showed that indication text was a strong retrieval shortcut, correctly aligned image reranking added measurable target-ordering value, and shuffled images did not reproduce the aligned result. The resulting retrieval gain transferred to final QA reference overlap.

The study also showed why these gains require careful interpretation. Target-rank improvement did not always produce Top-1 success. Answers could remain faithful to a wrongly selected report. Correct retrieval did not guarantee a reference-consistent final answer. Automated verification sometimes appeared to remove report-supported content, while other support declines reflected only template-prefix filtering.

The final contribution is therefore not a clinically autonomous diagnostic agent. It is a reproducible and auditable framework for separating target-case retrieval, report-level faithfulness, answer generation, verification, abstention, and image-report alignment. These distinctions provide a stronger foundation for future multimodal medical RAG research, but clinical validity requires independent expert evaluation and external data.

# References

Bae, S., Kyung, D., Ryu, J., et al. (2023). EHRXQA: A multi-modal question answering dataset for electronic health records with chest X-ray images. *Advances in Neural Information Processing Systems*.

Bannur, S., Hyland, S., Liu, Q., et al. (2023). Learning to exploit temporal structure for biomedical vision-language processing. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*.

Boecking, B., Usuyama, N., Bannur, S., et al. (2022). Making the most of text semantics to improve biomedical vision-language processing. *European Conference on Computer Vision*.

Demner-Fushman, D., Kohli, M. D., Rosenman, M. B., et al. (2016). Preparing a collection of radiology examinations for distribution and retrieval. *Journal of the American Medical Informatics Association, 23*(2), 304-310.

Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2024). RAGAS: Automated evaluation of retrieval augmented generation. *Proceedings of the European Chapter of the Association for Computational Linguistics: System Demonstrations*.

Jin, Q., Kim, W., Chen, Q., et al. (2023). MedCPT: Contrastive pre-trained transformers with large-scale PubMed search logs for zero-shot biomedical information retrieval. *Bioinformatics, 39*(11), btad651.

Lau, J. J., Gayen, S., Ben Abacha, A., and Demner-Fushman, D. (2018). A dataset of clinically generated visual questions and answers about radiology images. *Scientific Data, 5*, 180251.

Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*.

Ngo, N. T., Nguyen, C. V., Dernoncourt, F., and Nguyen, T. H. (2024). Comprehensive and practical evaluation of retrieval-augmented generation systems for medical question answering. *arXiv:2411.09213*.

Pal, A., Umapathi, L. K., and Sankarasubbu, M. (2023). Med-HALT: Medical domain hallucination test for large language models. *Proceedings of CoNLL*, 314-334.

Qwen Team. (2025). Qwen2.5 technical report. *arXiv:2412.15115*.

Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning transferable visual models from natural language supervision. *Proceedings of the 38th International Conference on Machine Learning*.

Robertson, S., and Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval, 3*(4), 333-389.

Romanov, A., and Shivade, C. (2018). Lessons from natural language inference in the clinical domain. *Proceedings of EMNLP*, 1586-1596.

Singhal, K., Azizi, S., Tu, T., et al. (2023). Large language models encode clinical knowledge. *Nature, 620*, 172-180.

Soni, S., Gudala, M., Pajouhi, A., and Roberts, K. (2022). RadQA: A question answering dataset to improve comprehension of radiology reports. *Proceedings of LREC 2022*, 6250-6259.

Xiong, G., Jin, Q., Lu, Z., and Zhang, A. (2024). Benchmarking retrieval-augmented generation for medicine. *Findings of ACL 2024*, 6233-6251.

# Appendices

## Appendix A: Frozen V5 Result Sources

- Configuration: `config/multimodal_v5.json`
- Cohort manifest: `data/processed/openi_multimodal_v5_cohort.json`
- Retrieval summary: `experiments/post_submission_v5/confirmation_retrieval_summary.json`
- Report-only QA summary: `experiments/post_submission_v5/qa_report_only/final_optimized_test_summary.json`
- Multimodal QA summary: `experiments/post_submission_v5/qa_multimodal/final_optimized_test_summary.json`
- Statistical analysis: `experiments/post_submission_v5/v5_statistics.json`
- Artifact manifest: `experiments/post_submission_v5/artifact_manifest.json`
- Runtime summary: `docs/V5_RUNTIME_SUMMARY.md`

The technical freeze is identified by commit `10f57ba` and tag `v5-technical-freeze`.

## Appendix B: Qualitative Audit Trail

- Frozen protocol v1.0: `docs/V5_QUALITATIVE_ANALYSIS_PROTOCOL.md`
- Refined taxonomy v1.1: `docs/V5_QUALITATIVE_TAXONOMY_V1_1.md`
- Review guide: `docs/V5_QUALITATIVE_REVIEW_GUIDE.md`
- Public 360-question numeric index: `experiments/post_submission_v5/qualitative_case_pack.csv`
- Public 24-question review index: `experiments/post_submission_v5/qualitative_representative_cases.csv`
- Researcher review record: `docs/V5_QUALITATIVE_RESEARCHER_REVIEW_RECORD.md`
- Final qualitative analysis: `docs/V5_QUALITATIVE_ERROR_ANALYSIS.md`

The qualitative freeze is identified by commit `f3fefbf` and tag `v5-qualitative-freeze`. Full report text, generated answers, prompt packs, and image pixels remain local under repository policy.

## Appendix C: Reproduction Entry Points

**Repository:** https://github.com/yzy542968-jpg/wqf7023-medical-rag  
**Branch:** `post-submission-improvements`

```powershell
& ".\.venv\Scripts\python.exe" scripts\build_multimodal_v5_cohort.py
& ".\.venv\Scripts\python.exe" scripts\run_multimodal_v5_retrieval.py --split confirmation --device cuda
& ".\.venv\Scripts\python.exe" scripts\build_multimodal_v5_prompt_packs.py --split confirmation
& ".\.venv\Scripts\python.exe" scripts\analyze_multimodal_v5_statistics.py
& ".\.venv\Scripts\python.exe" scripts\build_v5_artifact_manifest.py
& ".\.venv\Scripts\python.exe" scripts\build_v5_qualitative_review_materials.py
& ".\.venv\Scripts\python.exe" -m pytest -q
```

The generation and semantic-evaluation commands, model identifiers, batch sizes, and thresholds are recorded in `docs/V5_TECHNICAL_FREEZE.md`.

## Appendix D: Researcher Review and Human-Evaluation Boundary

The researcher reviewed and accepted all 24 taxonomy v1.1 proposals on 19 August 2026. This was an author/researcher qualitative review supported by deterministic case extraction and assistant-proposed coding. It was not an independent radiologist evaluation.

The earlier blinded independent-rating packages remain available as future-work artifacts but were not completed. No independent human correctness, grounding, preference, harmfulness, or inter-rater agreement result is claimed.

## Appendix E: Dashboard Demonstration Boundary

The dashboard demonstrates image-conditioned retrieval, top-ranked candidate report selection, report-grounded generation, semantic evidence checking, and trace display. For an arbitrary uploaded image, the interface should describe the action as retrieving the top-ranked candidate report from the indexed corpus. It must not claim to identify the patient's true report, diagnose the image, or access an authenticated clinical record.
