# Literature Matrix

This matrix records the core literature position for the project. The aim is not to cite every related paper, but to justify the thesis design choices: medical RAG, radiology case units, hybrid retrieval, evidence checking, and grounded evaluation.

| Theme | Source | Key Point | Use in This Project |
|---|---|---|---|
| Retrieval-augmented generation | Lewis et al. (2020) | RAG combines parametric generation with retrieved non-parametric memory and can improve factuality and provenance for knowledge-intensive tasks. | Establishes the general retrieve-then-generate paradigm used by all RAG baselines. |
| Lexical retrieval baseline | Robertson and Zaragoza (2009) | BM25 is a strong probabilistic lexical retrieval model and remains a transparent baseline. | Justifies BM25 as the main reproducible sparse retriever for radiology reports. |
| Biomedical dense retrieval | Jin et al. (2023), MedCPT | MedCPT uses large-scale PubMed search logs and contrastive learning for zero-shot biomedical retrieval. | Justifies using MedCPT as the biomedical semantic retriever and testing sparse+dense fusion. |
| Medical RAG benchmark | Xiong et al. (2024), MedRAG/MIRAGE | Medical RAG performance depends on retriever, corpus, and LLM choices; combining corpora and retrievers can improve medical QA. | Supports the decision to compare lexical, dense, and hybrid retrieval instead of assuming one retriever is sufficient. |
| Practical medical RAG evaluation | Ngo et al. (2024), MedRGB | Reliable medical RAG should be evaluated under practical conditions such as sufficiency, integration, robustness, and noisy/misleading retrieval. | Supports the project's focus on retrieved-case correctness, evidence sufficiency, and cross-case contamination. |
| RAG evaluation metrics | Es et al. (2024), Ragas | RAG evaluation needs multiple dimensions: context retrieval, faithfulness, answer quality, and generation use of evidence. | Supports reporting retrieval metrics separately from answer and evidence-support metrics. |
| Medical LLM reliability | Singhal et al. (2023) | Medical LLM evaluation requires stricter standards than general QA; benchmark scores alone are insufficient for clinical reliability. | Justifies the conservative framing: report-grounded QA, no autonomous image diagnosis, and manual evidence review. |
| Medical hallucination benchmark | Pal et al. (2023), Med-HALT | Medical-domain hallucination can be plausible but unverified or incorrect, requiring explicit testing and mitigation. | Supports the evidence-checking agent and unsupported-claim analysis. |
| Radiology image-report dataset | Demner-Fushman et al. (2016), OpenI / IU X-Ray | The OpenI/IU collection provides de-identified chest X-ray studies with associated radiology reports and images. | Justifies the selected public dataset and the case-level image-report retrieval unit. |
| Multimodal medical QA context | Bae et al. (2023), EHRXQA | Medical QA can require linked reasoning over imaging and clinical information sources. | Positions this project as a narrower, feasible report-grounded case retrieval task with linked X-ray presentation. |
| LLM backbone | Qwen Team (2025), Qwen2.5 | Qwen2.5 provides open-weight instruction models in multiple sizes, supporting local reproducible experimentation. | Justifies Qwen2.5-0.5B/1.5B pilots and the planned stronger 7B experiment if compute allows. |

## Thesis Gap

The reviewed work supports RAG and medical QA, but it leaves a narrower gap that this project can address within a master's thesis:

1. Many RAG studies evaluate document-level or passage-level retrieval, while this project treats a radiology examination as a case unit with report sections and linked image metadata.
2. Generic top-k RAG can mix evidence from multiple retrieved cases. The pilot results show this as a concrete cross-case contamination failure mode.
3. Medical RAG evaluation should separate retrieval success, generated-answer overlap, evidence support, revision/abstention behavior, and manual hallucination judgments.
4. The project therefore contributes a scoped, reproducible framework for report-grounded medical QA where evidence support must be case-specific, not merely somewhere in the top-k context.

## References to Prioritize in the Formal Report

High-priority citations:

- Lewis et al. (2020) for RAG.
- Robertson and Zaragoza (2009) for BM25.
- Jin et al. (2023) for MedCPT.
- Demner-Fushman et al. (2016) for OpenI/IU X-Ray.
- Xiong et al. (2024) and Ngo et al. (2024) for medical RAG evaluation.
- Pal et al. (2023) and Singhal et al. (2023) for medical hallucination and reliability.

Optional supporting citations:

- Bae et al. (2023) for multimodal medical QA context.
- Es et al. (2024) for RAG evaluation dimensions.
- Qwen Team (2025) for the local instruction-model backbone.
