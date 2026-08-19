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
