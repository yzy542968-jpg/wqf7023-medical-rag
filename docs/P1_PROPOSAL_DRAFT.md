# Evidence-Checking Agentic RAG for Radiology Report-Grounded Question Answering with Linked X-ray Cases

Name: ZHANG YUE  
Matric No: 22097191  
Programme: Master of Artificial Intelligence  
Course: WQF7023 Artificial Intelligence Research Project  
Supervisor: Dr. Uzair Ishtiaq  

## Chapter 1: Introduction

Large Language Models (LLMs) have shown strong performance in natural language question answering, summarization, and reasoning-oriented tasks. However, their use in healthcare-related applications remains challenging because generated answers may appear fluent while not being sufficiently supported by reliable medical evidence. This problem is especially important in medical question answering, where unsupported or hallucinated claims can reduce trust and may lead to unsafe interpretation if the system is used without proper caution.

Retrieval-Augmented Generation (RAG) is a promising approach for improving the grounding of LLM outputs. Instead of relying only on the internal knowledge of a language model, a RAG system first retrieves relevant external evidence and then conditions the generated answer on that evidence. This design is useful in medical settings because answers can be tied to specific retrieved reports, documents, or clinical cases.

Radiology is a suitable domain for studying evidence-grounded medical question answering because radiology records often contain both medical images and expert-written reports. A typical chest X-ray case includes image views, such as frontal and lateral projections, together with findings and impressions written by radiologists. Many existing text-based retrieval systems treat reports as independent text documents or retrieve short text fragments. However, in real radiology practice, the image and report belong to the same clinical case. Therefore, this project studies a case-level RAG approach in which each radiology image-report pair is treated as a complete retrieval unit.

The proposed research focuses on medical question answering over paired radiology images and reports. The report text will be used as the main evidence source for answer generation, while the associated X-ray images remain linked to the retrieved case for contextual presentation and case-based analysis. The proposed system further adds an evidence-checking agent loop that verifies whether answer claims are supported by retrieved report evidence and revises or abstains when support is insufficient. This scope allows the project to remain feasible within the WQF7023 timeline while still addressing two important limitations of basic RAG: loss of case-level context and unsupported generated claims.

## Chapter 2: Research Problem

LLMs can generate answers that sound confident even when the content is unsupported or incorrect. In healthcare-oriented question answering, this is a serious limitation because factual support and traceability are essential. A medical QA system should not only produce fluent answers; it should also show that its claims are grounded in relevant medical evidence.

RAG can reduce this risk by retrieving evidence before answer generation. However, a basic report-based RAG system may still retrieve report text as isolated information, without preserving the full clinical case structure. In radiology, the image and report are linked by design. If retrieval ignores this structure, the generated answer may be grounded in text but lose the broader case context. This is especially relevant when presenting results to a user, because the retrieved report and associated X-ray image together provide a more complete case-level explanation.

The research problem is therefore:

How can retrieval-augmented medical question answering be designed to preserve case-level context from paired radiology images and reports, and can an evidence-checking agent reduce unsupported medical claims compared with LLM-only answering and basic RAG?

This problem leads to three main concerns. First, the system must retrieve relevant radiology cases from a dataset of paired reports and images. Second, the generated answers should be supported by the retrieved report evidence. Third, the system should be evaluated not only for answer relevance but also for hallucination tendency and evidence support.

## Chapter 3: Research Aim, Questions, and Objectives

### Research Aim

The aim of this research is to develop and evaluate an evidence-checking agentic RAG system for radiology report-grounded question answering, and to study whether case-level retrieval plus evidence checking can produce more supported and reliable answers than simpler baseline systems.

### Research Questions

1. How much can retrieval augmentation improve answer quality in medical question answering over paired radiology cases?
2. Does hybrid BM25 + MedCPT retrieval improve case retrieval compared with lexical or dense retrieval alone?
3. Can an evidence-checking agent reduce unsupported claims compared with an LLM-only baseline and a basic RAG baseline?
4. Does presenting retrieved paired cases together with their medical images improve contextual completeness and case-based communication?

### Research Objectives

1. To implement an LLM-only medical question answering baseline without retrieval augmentation.
2. To implement report-based and case-based RAG baselines using radiology reports and linked case metadata.
3. To implement hybrid BM25 + MedCPT case retrieval over paired radiology cases.
4. To develop an evidence-checking agentic RAG framework that can revise or abstain from unsupported answers.
5. To evaluate retrieval quality, answer relevance, evidence support, and hallucination tendency.
6. To present selected retrieved cases with both report evidence and associated medical images.

## Chapter 4: Literature Review

### Medical Question Answering and Reliability

Medical question answering requires a higher level of reliability than general-domain QA because errors may affect interpretation of health-related information. LLMs are able to generate coherent explanations, but their answers can contain unsupported claims. In medical contexts, this creates the need for evidence-grounded generation, transparent retrieval, and careful evaluation.

### Retrieval-Augmented Generation

RAG combines retrieval with language generation. A retriever identifies relevant external evidence, and a generator uses that evidence to produce an answer. This approach is useful when the answer should be based on specific documents rather than only on model parameters. For this project, RAG provides a way to ground answers in radiology reports and reduce unsupported generation.

### Biomedical Retrieval

Biomedical retrieval differs from general retrieval because medical terms can be specialized, abbreviated, and semantically related even when exact word overlap is low. Lexical methods such as TF-IDF and BM25 are transparent and reproducible but may miss synonymy and deeper clinical relationships. Dense biomedical retrievers such as MedCPT are therefore relevant for this project because they are designed for biomedical text retrieval.

### Radiology Image-Report Datasets

The IU X-Ray / OpenI chest X-ray collection provides radiology reports linked to chest X-ray images. This makes it suitable for studying case-level retrieval. The current project uses each report and its associated image filenames as one retrieval unit. This differs from fragment-only retrieval because it preserves the case structure.

### Prompt Engineering for Grounded QA

Prompt design affects whether the generated answer follows retrieved evidence or introduces unsupported information. This project will compare direct prompting, evidence-guided prompting, and structured evidence prompting. The goal is not only to improve answer fluency but also to improve faithfulness to retrieved evidence.

### Agentic RAG and Evidence Checking

Agentic RAG extends the basic retrieve-then-generate pipeline by adding explicit intermediate steps such as planning, evidence verification, answer revision, and abstention. For medical QA, this is important because an answer should not only sound plausible; its claims should be traceable to retrieved evidence. In this project, the agentic component is scoped to evidence checking over expert-written radiology reports rather than autonomous clinical diagnosis.

### Evaluation of RAG Systems

RAG systems should be evaluated at both retrieval and generation levels. Retrieval can be evaluated using metrics such as Hit@k, Recall@k, and Mean Reciprocal Rank. Generated answers can be evaluated for relevance, evidence support, and hallucination tendency. Automatic RAG metrics may be useful, but manual checking remains important in medical QA because clinical claims require careful interpretation.

## Chapter 5: Research Design and Methodology

### Dataset

The project uses the IU X-Ray / OpenI chest X-ray dataset as the initial dataset. The current local project copy contains 3,851 radiology reports and 7,466 image-projection records. Each case includes report sections such as indication, findings, impression, and problem labels, together with associated X-ray image filenames and projections.

The dataset has been normalized into case-level JSONL format:

```text
data/processed/openi_cases.jsonl
```

Each row represents one radiology case and includes both report text and image metadata. This format supports the proposed case-based retrieval design.

### System Variants

The project will compare four main systems:

1. LLM-only baseline: The model answers medical questions without retrieval.
2. Report-based RAG baseline: The system retrieves report text and generates answers using retrieved report evidence.
3. Case-based RAG baseline: The system retrieves complete image-report cases and uses report evidence for answer generation while preserving linked images for case presentation.
4. Agentic case-based RAG proposed system: The system retrieves complete cases, drafts an answer, checks the answer against retrieved report evidence, and revises or abstains when support is insufficient.

### Retrieval Methods

The preliminary implementation includes TF-IDF and BM25 lexical retrieval baselines, MedCPT dense retrieval, and hybrid BM25 + MedCPT retrieval. These methods are useful because they allow the project to compare transparent word-overlap retrieval with biomedical semantic retrieval. The current results show that MedCPT alone is weaker than BM25 on the clean QA seed, but hybrid score fusion improves over both individual methods.

### Prompting Strategies

Three prompting strategies will be compared:

1. Direct prompting: The model answers directly with minimal structure.
2. Evidence-guided prompting: The model is instructed to answer only using retrieved report evidence.
3. Structured evidence prompting: The model produces a structured response that separates retrieved evidence, reasoning, and final answer.

The prompts will be designed to reduce unsupported claims and make the relationship between retrieved evidence and generated answer more explicit.

### Evaluation Plan

Retrieval quality will be evaluated using Hit@k, Recall@k, and Mean Reciprocal Rank. The initial keyword qrels are generated from the OpenI `Problems` field for feasibility evaluation. Later evaluation will include a manually checked question set generated from report findings and impressions.

Generated answers will be evaluated using:

1. Answer relevance.
2. Evidence support.
3. Hallucination tendency.
4. Completeness of case-based response.

Selected outputs will be manually checked against the original expert-written radiology reports. The linked images will be used for case presentation and contextual analysis, but the report text will remain the primary evidence for generated answers. For the agentic system, additional metrics will include evidence-support rate, revision rate, abstention rate, and unsupported-claim rate from manual checking.

## Preliminary Feasibility Results

The real OpenI metadata has already been processed into 3,851 case-level retrieval units. A small real image subset has also been downloaded for demonstration, containing five pneumonia-related cases and ten X-ray images.

Two lexical retrieval baselines were evaluated using eight keyword qrels derived from the OpenI `Problems` field.

| Retriever | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.500 | 0.875 | 0.875 | 0.875 | 0.875 | 0.667 |
| BM25 | 0.500 | 0.875 | 0.875 | 1.000 | 1.000 | 0.688 |

For the query `right lower lobe pneumonia with cough and fever`, the TF-IDF baseline retrieved case `CXR3078` as the top result. The report findings explicitly mention right lower lobe pneumonia, and the associated frontal and lateral X-ray images were preserved in the retrieved case. This demonstrates that the case-level retrieval pipeline can retrieve clinically relevant reports while maintaining their image links.

These preliminary results show that the project is technically feasible. However, lexical retrieval is still limited by word overlap. The next stage therefore compares lexical retrieval, biomedical dense retrieval, and hybrid retrieval.

A second evaluation seed was also created for later question answering experiments. It contains 360 case-grounded questions generated from 120 OpenI reports. In this more difficult setup, BM25 achieved Hit@20 of 0.344 and MRR of 0.264, while TF-IDF achieved Hit@20 of 0.342 and MRR of 0.232. A cleaner QA seed was then created by prioritizing non-normal problem-labeled cases and reducing weak placeholder-heavy questions. On this clean seed, the best current retrieval result is hybrid BM25 + MedCPT with alpha = 0.50, achieving Hit@20 of 0.553 and MRR of 0.324.

An initial evidence-checking agentic RAG baseline was also implemented on the clean QA seed. It achieved answer token-F1 of 0.412, top-1 case accuracy of 0.242, retrieved case hit rate of 0.422, and average evidence support rate of 0.992. Because this first agent uses extractive draft answers, it should be treated as a transparent safety baseline. The next experiment will test whether the same evidence-checking loop can reduce unsupported claims when the draft answer is generated by an LLM.

A small LLM pilot was also run locally using Qwen2.5 instruction models. On 30 clean QA-seed questions, Qwen2.5-1.5B without retrieval achieved token-F1 of 0.097. Report-RAG with BM25 improved retrieval coverage but remained vulnerable to verbose and mixed evidence use. The strongest local pilot setting was hybrid BM25 + MedCPT with top-1 case-grounded prompting, which achieved token-F1 of 0.218, top-1 case accuracy of 0.333, and retrieved case hit rate of 0.600. The pilot revealed a cross-case contamination failure mode, where the generator copied findings or impressions from multiple retrieved cases into one answer. This supports the proposed design choice of top-1 case-grounded prompting plus evidence checking.

## Scope

This project focuses on text-grounded medical question answering using paired radiology images and reports. It does not aim to train a full multimodal diagnostic model or perform raw-image reasoning as the main task. The image is used as part of the retrieved case representation and presentation, while the report remains the main source of evidence for answer generation. The agentic component is limited to planning, evidence checking, answer revision, and abstention.

The project also avoids private clinical data and uses public data only. Broader generalization to other datasets can be discussed as future work.

## Expected Outcomes

The expected outcomes are:

1. A normalized case-level dataset built from IU X-Ray / OpenI.
2. TF-IDF and BM25 lexical retrieval baselines.
3. Report-based and case-based RAG baselines.
4. A hybrid BM25 + MedCPT retrieval pipeline.
5. An evidence-checking agentic RAG framework over paired radiology image-report cases.
6. A comparison of prompting and evidence-checking strategies.
7. Retrieval, answer-quality, evidence-support, and hallucination-control evaluation results.
8. Selected case-based examples showing retrieved reports and associated X-ray images.
9. A final research report and presentation.

## Timeline

### P1: Proposal Phase

May 2026:

- Complete data preparation.
- Implement initial retrieval baselines.
- Write P1 proposal report.
- Prepare proposal presentation.

June to July 2026:

- Present the proposal.
- Revise based on supervisor and examiner feedback.
- Finalize implementation plan for P2.

### P2: Implementation and Final Defense

Late July to August 2026:

- Implement LLM-only baseline.
- Implement report-based RAG.
- Implement case-based RAG.
- Extend the evidence-checking agent to LLM-generated drafts.
- Generate outputs for the evaluation set.

September 2026:

- Complete evaluation.
- Analyze results.
- Write final report.
- Prepare final presentation and selected case examples.
