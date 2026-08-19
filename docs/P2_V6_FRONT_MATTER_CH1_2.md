# Retrieval-Augmented Medical Question Answering over Paired Radiology Images and Reports

**Name:** ZHANG YUE  
**Matric No.:** 22097191  
**Programme:** Master of Artificial Intelligence  
**Course:** WQF7023 Artificial Intelligence Research Project  
**Supervisor:** Dr. Uzair Ishtiaq  
**Document status:** V6-integrated manuscript draft  
**Technical result status:** V6 confirmation outcomes frozen  

## Abstract

Retrieval-augmented generation can provide external evidence for medical question answering, but a fluent answer may still be grounded in the wrong patient's report. In radiology, the image, clinical indication, findings, and impression form a linked case-level evidence unit. This research develops and critically evaluates an auditable retrieval-augmented question-answering workflow over paired chest X-ray images and reports. The central question is whether correctly aligned image information improves target-report retrieval and whether that improvement transfers to report-grounded answers without being confused with generic image similarity, generator behavior, or automated verifier behavior.

The research was conducted as a staged study. The earlier V5 study served as a preliminary controlled investigation that exposed indication shortcuts, cross-case retrieval ambiguity, and differences between report-level support and target-case alignment. The final V6 study used a newly instantiated 240-case confirmation cohort from the same OpenI/IU-Xray source. It contained 120 target cases, 120 distractors, 360 report-derived questions, and a broader within-source spectrum including report-indexed normal and abnormal cases. BM25 with indication-plus-question queries was retained as the primary text baseline. MedSigLIP provided modern image-text representations using frozen sentence-aware chunks of at most 64 tokens and maximum image-chunk similarity for reranking. Qwen3-Embedding and BioViL-T were secondary retrieval comparisons.

The primary retrieval result supported an alignment-specific image contribution. MedSigLIP reranking increased MRR from 0.6168 under BM25 to 0.6474, a difference of +0.03069 with a 95% case-grouped bootstrap interval of [0.00902, 0.05368]. The correctly aligned MedSigLIP condition exceeded all 100 deterministic fixed-point-free shuffled-image controls, with a plus-one Monte Carlo value of 0.00990. The downstream 2 x 2 QA factorial evaluated BM25 and MedSigLIP retrieval with Qwen2.5-1.5B-Instruct and MedGemma 1.5 4B. Verified Token-F1 increased by +0.01206 for Qwen2.5 and +0.03857 for MedGemma. Both point-estimate criteria passed, although the unchanged V5 verifier abstained on approximately 69-71% of Qwen2.5 answers and only 2.5-5.3% of MedGemma answers.

The principal contribution is not a claim that a newer model is universally better. It is an auditable evidence chain that separates target-case alignment, report-level faithfulness, answer-reference consistency, automated verification, and computational cost. The results show that correctly aligned image information can improve closed-set paired-report retrieval and transfer to a report-grounded QA metric, while also showing that the effect is concentrated in report-indexed abnormal cases, varies across generators, and does not establish clinical correctness, external validity, patient-level independence, or deployment safety.

**Keywords:** retrieval-augmented generation, medical question answering, radiology, chest X-ray, multimodal retrieval, image-report alignment, MedSigLIP, evidence grounding

# Chapter 1: Introduction

## 1.1 Background

Large language models can generate fluent medical answers, but fluency is not evidence of correctness. Retrieval-augmented generation provides a way to expose external evidence before generation, yet it remains a multi-stage pipeline. A question may be too generic to identify a case, a retriever may select a report from the wrong case, a generator may omit or add content, and a verifier may accept or reject content according to a policy that does not perfectly reflect clinical adequacy.

These risks are particularly important in radiology because an examination contains linked but non-identical information sources. The image provides visual evidence, the indication provides referral context, and the findings and impression provide report-level evidence. A system can therefore produce a sentence that is supported by a report while still selecting the wrong report for the target image. This is a case-alignment problem as well as a language-generation problem.

The OpenI/IU-Xray collection supports a controlled study of this problem because it contains chest radiographs paired with report text. The dataset is not a substitute for prospective clinical data, but it permits reproducible comparisons of text retrieval, image-assisted reranking, report-grounded generation, and automated evidence filtering.

The final study uses V5 and V6 as connected stages. V5 is retained as the preliminary controlled study because it revealed the failure modes that shaped the final confirmation design. V6 is the principal model-modernized study. It uses MedSigLIP, Qwen3-Embedding as a secondary text baseline, Qwen2.5 and MedGemma as generators, a frozen V5 verifier, deterministic cohort fingerprints, and case-grouped statistical analysis. This structure preserves the development history while giving the final thesis a clear main experiment.

## 1.2 Problem statement

Many RAG evaluations implicitly assume that the query identifies one relevant document and that evidence found in the retrieved context is sufficient. Those assumptions are weak for case-based medical data. Generic questions such as asking for the findings or impression contain little case-specific information. Clinical indications may provide strong lexical shortcuts. An image reranker may improve a target's rank without placing the target report first. A generator may then give an answer that is faithful to a wrong but similar report.

The layered problem is:

```text
answer supported by selected report
                does not imply
selected report belongs to target case
                does not imply
clinical correctness or safety
```

Automated verification introduces another layer. A verifier can remove a report-supported sentence, preserve an unsupported sentence, or abstain because the answer form is difficult to match. Retrieval rank, target alignment, report support, reference overlap, and abstention must therefore be measured together.

The research problem is:

> How can paired radiology images and reports be used in an auditable RAG workflow that improves target-report retrieval, preserves case-level alignment, and explains downstream generation and verification behavior without overstating clinical validity?

## 1.3 Aim

The aim is to develop and critically evaluate a reproducible multimodal retrieval-augmented workflow for report-grounded medical question answering over paired chest X-ray images and radiology reports.

## 1.4 Objectives

The objectives are:

1. To construct a reproducible case-level retrieval and QA workflow that preserves image-report pairing.
2. To quantify the contribution of indication text and correctly aligned image representations to target-report retrieval.
3. To test whether the image contribution is specific to correct image-report alignment using shuffled-image controls.
4. To test whether retrieval gains transfer across Qwen2.5 and MedGemma 1.5 under a common report-grounded prompt.
5. To separate raw answer overlap, verified answer overlap, report-level support, abstention, and revision behavior.
6. To quantify computational cost and document the limits of automated evaluation.

## 1.5 Research questions

**RQ1.** Does correctly aligned MedSigLIP reranking improve target-report retrieval over the same BM25 indication-plus-question baseline on an untouched confirmation cohort?

**RQ2.** Is the observed image contribution specific to correct image-report alignment rather than generic or shuffled image information?

**RQ3.** Does the retrieval improvement transfer to report-grounded QA under both Qwen2.5 and MedGemma 1.5 when the prompt and verifier policy are held constant?

**RQ4.** What trade-offs appear across retrieval performance, reference consistency, automated evidence support, abstention, generator choice, latency, and GPU memory?

## 1.6 Research contributions

This thesis makes five scoped contributions.

First, it provides an alignment-aware multimodal retrieval evaluation in which correctly paired images are compared with deterministic fixed-point-free shuffled-image controls. This tests the relation between an image and its report rather than treating image input as an unexamined feature addition.

Second, it provides a layered grounding analysis that separates target-case retrieval, report-level support, answer-reference consistency, and automated verification. This prevents a locally faithful answer from being interpreted as correctly grounded when its report belongs to another case.

Third, it evaluates retrieval transfer across a general small instruction model and a modern medical generator. The comparison shows that retrieval improvements can transfer across generators while differing in magnitude and verifier interaction.

Fourth, it provides a reproducible confirmation protocol with case-ID manifests, deterministic selection, model revisions, configuration hashes, output hashes, case-grouped statistics, and explicit claim boundaries.

Fifth, it reports negative and mixed evidence. BM25 remained stronger than Qwen3-Embedding on MRR, normal cases showed little retrieval gain, and the frozen verifier showed high abstention for Qwen2.5. These results define the boundary conditions of the method rather than presenting a uniformly positive benchmark story.

The contribution is not state-of-the-art clinical QA and not a diagnostic system. It is a controlled study of whether paired image information can improve evidence retrieval and how that change behaves in a traceable RAG pipeline.

## 1.7 Scope and boundaries

The system operates on a closed candidate pool of 240 OpenI/IU-Xray cases. The image is used as a retrieval and reranking signal. The generator receives the selected report's findings and impression rather than image pixels. The questions are deterministic report-derived questions rather than physician-authored natural clinical questions.

The study claims case-ID disjointness between development and confirmation artifacts, not patient-level independence. It claims within-source confirmation, not external validation. Automated verifier scores and Token-F1 measure research-reference consistency and evidence signals, not clinical correctness. The system is a research prototype and must not be used for clinical decision making.

## 1.8 Conceptual framework

The conceptual framework treats the pipeline as four linked relations:

```text
query and image
      -> target-case retrieval
target report
      -> report-level evidence
report evidence
      -> generated answer
generated answer
      -> automated verification and abstention
```

Each arrow can fail independently. The final answer is therefore interpreted through an evidence ledger rather than one aggregate score. A retrieval gain is useful only if it improves the selected evidence, and selected evidence is useful only if the generator and verifier preserve enough of it for the intended answer.

## 1.9 Thesis organization

Chapter 2 reviews RAG, medical question answering, sparse and dense retrieval, paired radiology representation, alignment controls, evidence grounding, and auditable agentic workflows. Chapter 3 presents the V6 methodology, including development decisions, confirmation cohort construction, retrieval, generation, verification, statistics, and reproducibility controls. Chapter 4 reports retrieval, alignment-control, QA, subgroup, and cost results. Chapter 5 answers the research questions, discusses contributions and limitations, and identifies future work including independent human evaluation and external validation.

# Chapter 2: Literature review integration note

The existing literature review remains usable, but four framing points should be kept consistent with the V6 main study.

First, the work is about evidence retrieval and traceability, not direct visual diagnosis. Second, image-report alignment is treated as an experimental relation that requires a negative control. Third, physician-authored datasets and external report-QA benchmarks should be discussed as future or complementary validation unless their provenance matches the current source. Fourth, a newer model should not be treated as a contribution by itself; the research contribution is the controlled comparison and the evidence chain.

The literature review should therefore introduce MedSigLIP and MedGemma as modern model choices used in the confirmation study, while retaining BM25 as the interpretability baseline. It should also explain why the frozen V5 verifier was retained: changing every component at once would obscure attribution. The literature gap is not simply the absence of a large model. It is the absence of an evaluation that distinguishes correct target-case alignment from local report faithfulness and downstream answer overlap.
