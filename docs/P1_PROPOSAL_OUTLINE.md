# P1 Proposal Outline

## Chapter 1: Introduction

- Introduce medical question answering and the need for evidence-supported answers.
- Explain why hallucination is risky in healthcare-oriented LLM applications.
- Introduce retrieval-augmented generation as a way to ground model responses in external evidence.
- Explain why radiology is a suitable case study because many records contain paired images and expert-written reports.
- State the proposed idea: treat each image-report pair as a case-level retrieval unit.

## Chapter 2: Research Problem

The problem is that LLMs may generate fluent answers without sufficient support from medical evidence. Standard report-based RAG can improve grounding, but retrieving isolated text fragments may lose case-level context. In radiology, the image and report belong together clinically, so retrieval should preserve the case as a unit even if answer generation is mainly grounded in the report text.

Possible problem statement:

This research investigates whether case-level retrieval over paired radiology images and reports can improve the grounding, relevance, and contextual completeness of medical question answering compared with LLM-only answering and report-only RAG.

## Chapter 3: Research Aim, Questions, and Objectives

### Aim

To develop and evaluate a retrieval-augmented medical question answering system over paired radiology images and reports, and to study whether case-level retrieval and prompt engineering improve answer support and reliability.

### Research Questions

1. How much does retrieval augmentation improve medical question answering over paired radiology cases?
2. How do prompting strategies affect answer accuracy, relevance, and evidence support?
3. Can case-based RAG reduce hallucination tendency compared with an LLM-only baseline and a report-based RAG baseline?
4. Does presenting retrieved paired cases with their images improve contextual completeness and case-based communication?

### Objectives

1. Implement an LLM-only medical question answering baseline.
2. Implement a report-based RAG baseline.
3. Develop a case-based RAG framework over paired radiology images and reports.
4. Compare direct, evidence-guided, and structured reasoning prompts.
5. Evaluate retrieval quality, answer relevance, evidence support, and hallucination tendency.

## Chapter 4: Literature Review

Suggested subsections:

- Medical question answering and clinical reliability.
- Retrieval-augmented generation.
- Biomedical and medical information retrieval.
- Radiology report datasets and image-report pairing.
- Prompt engineering for grounded question answering.
- Evaluation of RAG systems.

## Chapter 5: Research Design and Methodology

### Dataset

Initial dataset: IU X-Ray / OpenI, using paired chest X-ray images and expert-written radiology reports.

### System Variants

1. LLM-only baseline.
2. Report-based RAG baseline.
3. Case-based RAG with paired image-report retrieval.

### Retrieval

- Baseline prototype: TF-IDF retrieval for feasibility testing.
- Planned retrieval: MedCPT embeddings with ChromaDB.

### Generation

- Planned base model: Qwen2.5-7B-Instruct.
- Main evidence source: retrieved report text.
- Associated images: kept with retrieved case for presentation and contextual analysis.

### Evaluation

- Retrieval quality: Hit@k, Recall@k, MRR.
- Answer quality: relevance and completeness.
- Evidence support: whether answer claims are supported by retrieved reports.
- Hallucination tendency: unsupported medical claims or contradictions.

