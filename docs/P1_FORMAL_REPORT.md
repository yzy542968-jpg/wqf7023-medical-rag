# Evidence-Checking Agentic RAG for Radiology Report-Grounded Question Answering with Linked X-ray Cases

Name: ZHANG YUE  
Matric No: 22097191  
Programme: Master of Artificial Intelligence  
Course: WQF7023 Artificial Intelligence Research Project  
Supervisor: Dr. Uzair Ishtiaq  
Current version date: 2026-06-05

## Abstract

Large language models can answer medical questions fluently, but fluent answers may still contain unsupported or incorrect claims. This is a major concern in healthcare-oriented question answering, where users need evidence traceability rather than plausible text alone. Retrieval-augmented generation (RAG) can reduce this risk by retrieving external evidence before generation, but a basic top-k RAG pipeline can still mix information from multiple retrieved records. In radiology, this risk is especially important because each examination is naturally a case: one or more X-ray images are linked to an expert-written report. This project proposes an evidence-checking agentic RAG system for radiology report-grounded question answering with linked X-ray cases. The report is used as the evidence source, while the associated X-ray images remain linked for case presentation and contextual analysis. The project compares LLM-only answering, report-based RAG, case-based RAG, hybrid BM25 + MedCPT retrieval, and an evidence-checking agent that revises or abstains from unsupported answers. Preliminary experiments on the IU X-Ray / OpenI dataset show that hybrid BM25 + MedCPT retrieval improves case retrieval over BM25 or MedCPT alone, and a local Qwen2.5 pilot reveals cross-case contamination as a concrete failure mode. The expected contribution is a scoped, reproducible medical RAG framework that makes case boundaries explicit and evaluates whether generated claims remain supported by the selected radiology case.

## 1. Introduction

Large language models (LLMs) have demonstrated strong performance in natural language generation, question answering, summarization, and reasoning-oriented tasks. However, their use in healthcare-related applications remains limited by reliability and evidence-grounding concerns. Medical answers must not only sound coherent; they must also be traceable to trustworthy evidence. Prior work on medical LLMs shows that clinical knowledge benchmarks require careful interpretation because high benchmark performance does not automatically imply safe clinical use (Singhal et al., 2023). Medical hallucination benchmarks such as Med-HALT further highlight that LLMs can generate plausible but unverified or incorrect medical information (Pal et al., 2023).

Retrieval-augmented generation (RAG) is a promising method for reducing unsupported generation. RAG combines a parametric generator with retrieved external evidence, giving the model access to non-parametric memory and improving provenance for knowledge-intensive tasks (Lewis et al., 2020). In medicine, recent benchmarks such as MedRAG/MIRAGE show that retrieval can improve medical QA performance, but system behavior depends on the choice of corpus, retriever, and language model (Xiong et al., 2024). This means that a medical RAG system should be evaluated as a pipeline, not only as an LLM prompt.

Radiology provides a useful domain for studying evidence-grounded medical QA because chest X-ray examinations contain both medical images and expert-written reports. The IU X-Ray / OpenI collection contains de-identified chest X-ray studies with associated radiology reports and images (Demner-Fushman et al., 2016). This project treats each radiology examination as a complete case-level retrieval unit. The linked report provides textual evidence for answering, while the X-ray image filenames and projections preserve the case context for presentation and analysis.

The proposed project is deliberately scoped. It does not attempt autonomous raw-image diagnosis. Instead, it studies report-grounded QA over paired radiology image-report cases. This makes the project feasible within the master's timeline while still addressing a meaningful AI research problem: how to reduce unsupported medical claims when an LLM answers questions using retrieved radiology evidence.

## 2. Research Problem

Basic RAG systems retrieve evidence and feed it to a generator, but this does not guarantee that the final answer is faithful to the correct case. A generator may copy evidence from several retrieved cases into one answer. In a general document QA setting, this can be a relevance error. In radiology QA, it can become a case contamination error, because the answer may describe findings from different patients or examinations as if they belonged to one selected case.

This project therefore studies the following research problem:

How can retrieval-augmented medical question answering preserve radiology case boundaries, and can an evidence-checking agent reduce unsupported or cross-case claims compared with LLM-only answering and basic RAG?

The problem has four parts:

1. The system must retrieve relevant radiology cases from a public image-report dataset.
2. It must preserve the link between report text and X-ray case metadata.
3. It must generate or revise answers so that claims are supported by the selected report evidence.
4. It must evaluate answer quality and evidence support separately, because a fluent answer can still be unsupported.

## 3. Research Aim, Questions, and Objectives

### 3.1 Research Aim

The aim of this research is to develop and evaluate an evidence-checking agentic RAG system for radiology report-grounded question answering, and to test whether case-level retrieval plus evidence checking can produce more supported answers than simpler baseline systems.

### 3.2 Research Questions

RQ1. Does retrieval augmentation improve medical QA answer quality compared with LLM-only answering on report-grounded radiology questions?

RQ2. Does hybrid BM25 + MedCPT retrieval improve case retrieval compared with BM25 or MedCPT alone?

RQ3. Can an evidence-checking agent reduce unsupported and cross-case claims in generated answers?

RQ4. Does preserving linked X-ray case metadata improve contextual completeness and communication without claiming autonomous image diagnosis?

### 3.3 Research Objectives

1. Build a normalized case-level dataset from IU X-Ray / OpenI reports and image mappings.
2. Implement TF-IDF and BM25 lexical retrieval baselines.
3. Implement MedCPT dense retrieval and hybrid BM25 + MedCPT score fusion.
4. Implement LLM-only, report-RAG, case-RAG, and top-1 case-grounded RAG prompt variants.
5. Implement an evidence-checking agent that verifies answer claims against selected report evidence and revises or abstains when support is insufficient.
6. Evaluate retrieval quality, answer overlap, evidence support, revision rate, abstention rate, and unsupported-claim rate.
7. Present selected retrieved cases with report evidence and linked X-ray images for qualitative analysis.

## 4. Literature Review

### 4.1 Retrieval-Augmented Generation

RAG was introduced as a way to combine language generation with retrieved external memory for knowledge-intensive tasks (Lewis et al., 2020). This design is relevant to medical QA because it can provide explicit evidence and reduce reliance on model parameters alone. However, RAG is not a single method; it is a pipeline whose behavior depends on retrieval quality, prompt construction, and generation behavior.

### 4.2 Medical RAG and Evaluation

Medical RAG has become an active research area because medical knowledge is specialized, high-stakes, and frequently updated. Xiong et al. (2024) benchmarked medical RAG across multiple corpora, retrievers, and LLMs, showing that retrieval can improve medical QA but that different components interact. Ngo et al. (2024) further argued that medical RAG evaluation should test practical conditions such as information sufficiency, integration, and robustness to noisy or misleading retrieval. These findings motivate this project's pipeline-level evaluation, where retrieval and generation are not collapsed into one score.

### 4.3 Biomedical Retrieval

Lexical retrieval methods such as BM25 remain useful because they are transparent, reproducible, and strong when query terms overlap with report language (Robertson and Zaragoza, 2009). However, biomedical text often contains abbreviations, synonyms, and specialized terminology. MedCPT addresses biomedical semantic retrieval using contrastive pre-training from large-scale PubMed search logs (Jin et al., 2023). This project compares BM25, MedCPT, and hybrid score fusion because neither sparse nor dense retrieval should be assumed to dominate in a small radiology QA setting.

### 4.4 Radiology Case Data

The IU X-Ray / OpenI collection contains radiology reports paired with chest X-ray images and was created for distribution and retrieval research (Demner-Fushman et al., 2016). This makes it suitable for the proposed case-level retrieval design. Unlike a passage-only text corpus, each record can be treated as a complete radiology case containing indication, findings, impression, problem labels, image filenames, and projection metadata.

### 4.5 Medical LLM Reliability and Hallucination

Medical LLMs require stricter evaluation than general-purpose chatbots. Singhal et al. (2023) showed that medical QA benchmarks can measure clinical knowledge, but also emphasized the high bar for clinical applications. Pal et al. (2023) introduced Med-HALT to evaluate medical-domain hallucination, where models produce plausible but unverified or incorrect medical statements. These studies support this project's focus on evidence support, abstention, and unsupported-claim analysis rather than answer fluency alone.

### 4.6 Multimodal Medical QA Context

Some medical QA datasets require reasoning over multiple modalities. EHRXQA, for example, combines structured EHR information with chest X-ray images for multi-modal QA (Bae et al., 2023). This project is narrower: it does not train a multimodal diagnostic model. Instead, it uses linked images as case context while grounding answers in radiology reports. This narrow scope is a strength for a master's project because it avoids overclaiming raw-image reasoning while still preserving clinically meaningful case structure.

## 5. Methodology

### 5.1 Dataset

The project uses the IU X-Ray / OpenI chest X-ray collection. The current local processed dataset contains:

| Item | Count |
|---|---:|
| Normalized radiology report cases | 3,851 |
| Image-projection mappings | 7,466 |
| Clean QA-seed cases | 120 |
| Clean QA-seed questions | 360 |

Each processed case is stored as one JSONL record in:

```text
data/processed/openi_cases.jsonl
```

Each record includes case ID, indication, findings, impression, problem labels, report text, image filenames, and projection information. For QA experiments, a deterministic clean QA seed was generated from 120 cases, producing 360 case-grounded questions.

### 5.2 System Variants

The project compares the following systems:

| System | Description |
|---|---|
| LLM-only | The model answers without retrieved evidence. |
| Report-RAG BM25 | BM25 retrieves report text and the LLM answers using top-k report evidence. |
| Case-RAG BM25 | BM25 retrieves complete image-report cases, preserving linked image metadata. |
| Case-RAG Hybrid | Hybrid BM25 + MedCPT retrieval retrieves complete cases. |
| Case-RAG Top-1 Prompt | The generator sees only the top-ranked case evidence, while top-k metadata is retained for retrieval evaluation. |
| Agentic Case-RAG | The system plans the query, retrieves cases, drafts an answer, checks evidence support, and revises or abstains. |

### 5.3 Retrieval Methods

The retrieval stage includes:

1. TF-IDF baseline.
2. BM25 baseline.
3. MedCPT dense retrieval.
4. Hybrid BM25 + MedCPT score fusion.

Hybrid retrieval is important because preliminary results show that MedCPT alone underperforms BM25 on this dataset, but the fused signal improves recall. This is an interpretable thesis result: a biomedical dense model may provide complementary semantic signal even when it is not the strongest standalone retriever.

### 5.4 Evidence-Checking Agent

The agentic loop contains four stages:

1. Query planning: normalize the question into retrieval intent.
2. Case retrieval: retrieve complete image-report cases.
3. Draft answering: generate or extract an answer from retrieved evidence.
4. Evidence checking: split the answer into claims, test each claim against selected report evidence, revise unsupported claims, or abstain.

The evidence checker has two possible scopes:

| Scope | Meaning | Use |
|---|---|---|
| All top-k evidence | A claim is supported if it appears anywhere in the retrieved top-k pool. | Useful for broad document QA, but can hide cross-case contamination. |
| Top-1 selected case evidence | A claim is supported only if it appears in the selected top-ranked case. | Preferred for report-grounded case QA. |

The pilot results support the top-1 scope because all-top-k checking can falsely accept mixed claims copied from several different cases.

### 5.5 Evaluation Metrics

Retrieval evaluation:

1. Hit@1, Hit@5, Hit@20.
2. Mean Reciprocal Rank (MRR).
3. Retrieved case hit rate for generated-answer runs.

Answer evaluation:

1. Token-F1 against report-derived reference answers.
2. Average answer length.
3. Insufficient-answer rate.

Evidence-checking evaluation:

1. Evidence support rate.
2. Revision rate.
3. Abstention rate.
4. Unsupported sentence rate.
5. Manual annotation of selected outputs for relevance, support, hallucination, and completeness.

## 6. Preliminary Results

### 6.1 Retrieval Results on Clean QA Seed

The clean QA seed contains 360 questions linked to 120 OpenI cases. The best current retrieval setting is hybrid BM25 + MedCPT with alpha = 0.50.

| Retriever | Hit@1 | Hit@5 | Hit@20 | MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.231 | 0.383 | 0.486 | 0.297 |
| MedCPT | 0.108 | 0.253 | 0.406 | 0.172 |
| Hybrid BM25 + MedCPT, alpha = 0.50 | 0.242 | 0.422 | 0.553 | 0.324 |

Interpretation: MedCPT alone is weaker than BM25 on this small report-grounded QA seed, but hybrid fusion improves retrieval over both individual methods. This supports RQ2 and justifies the hybrid retrieval design.

### 6.2 Local LLM Pilot

A local pilot was run using Qwen2.5 instruction models on the RTX 5070 Laptop GPU. The strongest 30-question local setting so far is Qwen2.5-1.5B with hybrid top-1 case prompting.

| System | Token-F1 | Top-1 Case Accuracy | Retrieved Hit Rate | Average Answer Words |
|---|---:|---:|---:|---:|
| LLM-only Qwen2.5-1.5B | 0.097 | 0.000 | 0.000 | 63.567 |
| Report-RAG BM25 Qwen2.5-1.5B | 0.148 | 0.300 | 0.533 | 88.433 |
| Case-RAG BM25 top-1 Qwen2.5-1.5B | 0.209 | 0.300 | 0.533 | 52.900 |
| Case-RAG Hybrid top-1 Qwen2.5-1.5B | 0.218 | 0.333 | 0.600 | 57.633 |

Interpretation: Retrieval improves over LLM-only answering, and top-1 case-grounded prompting is the strongest local small-model setting. The result is preliminary because the sample size is 30 questions and the models are small.

The main local configurations have now also been run over the full 360-question clean QA seed:

| System | Questions | Token-F1 | Top-1 Case Accuracy | Retrieved Hit Rate | Average Answer Words |
|---|---:|---:|---:|---:|---:|
| LLM-only Qwen2.5-1.5B | 360 | 0.091 | 0.000 | 0.000 | 65.544 |
| Report-RAG BM25 Qwen2.5-1.5B | 360 | 0.146 | 0.231 | 0.383 | 84.464 |
| Case-RAG BM25 top-1 Qwen2.5-1.5B | 360 | 0.188 | 0.231 | 0.383 | 53.297 |
| Case-RAG Hybrid top-1 Qwen2.5-1.5B | 360 | 0.209 | 0.242 | 0.422 | 52.139 |

Interpretation: the full-run results confirm that retrieval improves over LLM-only answering. Case-level top-1 prompting improves over report-RAG BM25, and hybrid retrieval gives the best automatic answer score and retrieved-hit rate among the local full-run systems.

### 6.3 Agentic Evidence-Checking Pilot

The extractive agentic hybrid baseline on the clean QA seed achieved:

| Metric | Result |
|---|---:|
| Answer token-F1 | 0.412 |
| Top-1 case accuracy | 0.242 |
| Retrieved case hit rate | 0.422 |
| Evidence support | 0.992 |
| Revision rate | 0.008 |
| Abstention rate | 0.008 |

This result is useful as a transparent safety baseline, but it should not be overstated because the answer draft is extractive. The more important pilot is evidence checking over LLM-generated drafts. On Qwen2.5-1.5B hybrid top-1 outputs, top-1 evidence checking showed:

| Metric | Result |
|---|---:|
| Draft token-F1 | 0.218 |
| Final token-F1 after checking | 0.161 |
| Evidence support | 0.300 |
| Revision rate | 0.900 |
| Abstention rate | 0.333 |
| Unsupported sentence rate | 0.705 |

Interpretation: The checker removes or abstains from many unsupported generated claims. Token-F1 may drop, but the safety behavior is valuable because unsupported medical claims are penalized rather than rewarded.

On the full 360-question generated runs, top-1 evidence checking produced:

| System | Draft Token-F1 | Final Token-F1 | Evidence Support | Revision Rate | Abstention Rate | Unsupported Sentence Rate |
|---|---:|---:|---:|---:|---:|---:|
| Report-RAG BM25 Qwen2.5-1.5B | 0.146 | 0.093 | 0.118 | 0.992 | 0.681 | 0.819 |
| Case-RAG BM25 top-1 Qwen2.5-1.5B | 0.188 | 0.139 | 0.374 | 0.833 | 0.325 | 0.674 |
| Case-RAG Hybrid top-1 Qwen2.5-1.5B | 0.209 | 0.145 | 0.305 | 0.886 | 0.419 | 0.711 |

This confirms that the cross-case and unsupported-claim issue is not only a small pilot artifact. The checker frequently revises or abstains, suggesting that a final thesis evaluation should include manual annotation rather than relying only on Token-F1. It also shows a useful tradeoff: hybrid top-1 has the best draft Token-F1, while BM25 top-1 has higher evidence-support rate under the current checker.

### 6.4 Key Error Finding: Cross-Case Contamination

The most important pilot finding is cross-case contamination. When a model is given several retrieved cases, it can combine findings from multiple cases into one answer. If the evidence checker accepts support from the full top-k pool, these copied claims may appear supported even though they do not belong to the selected case. This supports the design choice of top-1 case-grounded prompting and top-1 evidence checking.

This finding gives the thesis a concrete contribution beyond a standard course project. The project is not only asking whether RAG improves scores; it studies when RAG can still fail in a case-based medical setting and how an agentic checker can make that failure visible.

### 6.5 Full-Run Error Analysis

Automated error analysis over the full 360-question Qwen2.5-1.5B outputs shows that `abnormality_summary` questions are the hardest question type. These questions often refer to problem labels rather than directly asking for findings or impressions, making both retrieval and answer matching more difficult.

The analysis also shows that report-RAG BM25 is especially vulnerable to unsupported or mixed evidence. Its full-run top-1 evidence-support rate is 0.118, with an abstention rate of 0.681 after checking. Case-RAG BM25 top-1 improves evidence support to 0.374, while case-RAG hybrid top-1 achieves the best draft Token-F1 but a lower evidence-support rate of 0.305. This suggests a useful tradeoff: hybrid retrieval improves coverage and answer overlap, but may retrieve semantically similar cases that still require stricter evidence checking.

Representative cases for manual review include:

1. `CXR2721_impression`: high-support success case.
2. `CXR533_impression`: high Token-F1 but low evidence support.
3. `CXR1027_summary`: retrieval miss with poor answer.
4. `CXR1027_impression`: heavy revision case caused by mixed report evidence.
5. `CXR1054_impression`: hybrid top-1 representative failure.

These cases will be used in the final report to explain why the project evaluates retrieval, answer overlap, evidence support, revision behavior, and manual hallucination labels separately.

## 7. Scope and Limitations

This project is not a clinical diagnostic system. It does not train a raw-image model and does not claim to identify disease directly from X-rays. The report remains the primary evidence source. Linked X-ray images are used for case presentation, context, and selected qualitative examples.

Current limitations:

1. The QA seed is generated from existing report sections and still needs manual review.
2. The local LLM pilot uses small models because of hardware limits.
3. Token-F1 is an imperfect metric for radiology answers, so manual evidence annotation is necessary.
4. OpenI reports are public and de-identified, but the dataset is older and may not represent all clinical settings.
5. Final claims should focus on evidence-grounded QA behavior, not clinical deployment.

## 8. Expected Contributions

The expected contributions are:

1. A reproducible OpenI case-level dataset pipeline preserving report and image links.
2. A comparison of lexical, dense biomedical, and hybrid case retrieval.
3. A report-RAG versus case-RAG comparison for radiology report-grounded QA.
4. An evidence-checking agent that revises or abstains from unsupported generated claims.
5. A cross-case contamination analysis showing why medical RAG support must be case-specific.
6. A manual annotation protocol for evidence support, hallucination tendency, and completeness.
7. A final report and presentation based on real public data and local reproducible experiments.

## 9. Project Timeline

### P1: Proposal Phase, May to July 2026

- Prepare OpenI data and retrieval baselines. Completed.
- Build clean QA seed. Completed.
- Build MedCPT and hybrid retrieval. Completed.
- Run local Qwen2.5 pilot generation and evidence checking. Completed.
- Prepare P1 report and presentation. In progress.
- Present proposal and revise based on feedback. June to July 2026.

### P2: Implementation and Final Evaluation, July to September 2026

- Complete full generation experiments over all 360 clean QA questions. Completed for LLM-only, report-RAG BM25, case-RAG BM25 top-1, and case-RAG hybrid top-1 using Qwen2.5-1.5B.
- Extend or finalize the evidence-checking agent for generated outputs.
- Complete 30 to 50 manually annotated examples.
- Analyze failures by question type and retrieval correctness.
- Write final thesis chapters and prepare final defense slides.

## 10. Conclusion

This project is suitable for an AI master's research project because it is not merely applying a generic RAG template. It identifies a medically meaningful problem: generated answers must remain grounded in the selected radiology case, not only in some retrieved text. The current implementation has already processed real OpenI data, built sparse and dense retrieval baselines, run hybrid retrieval, executed local LLM pilots, and surfaced cross-case contamination as a concrete failure mode. The remaining work is to scale evaluation, complete manual evidence annotation, and write the final analysis.

## References

Bae, S., Kyung, D., Ryu, J., Cho, E., Lee, G., Kweon, S., Oh, J., Ji, L., Chang, E. I.-C., Kim, T., and Choi, E. (2023). EHRXQA: A multi-modal question answering dataset for electronic health records with chest X-ray images. Advances in Neural Information Processing Systems.

Demner-Fushman, D., Kohli, M. D., Rosenman, M. B., Shooshan, S. E., Rodriguez, L., Antani, S., Thoma, G. R., and McDonald, C. J. (2016). Preparing a collection of radiology examinations for distribution and retrieval. Journal of the American Medical Informatics Association, 23(2), 304-310.

Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2024). Ragas: Automated evaluation of retrieval augmented generation. arXiv:2309.15217.

Jin, Q., Kim, W., Chen, Q., Comeau, D. C., Yeganova, L., Wilbur, W. J., and Lu, Z. (2023). MedCPT: Contrastive pre-trained transformers with large-scale PubMed search logs for zero-shot biomedical information retrieval. Bioinformatics, 39(11), btad651.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W., Rocktaschel, T., Riedel, S., and Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. Advances in Neural Information Processing Systems.

Ngo, N. T., Nguyen, C. V., Dernoncourt, F., and Nguyen, T. H. (2024). Comprehensive and practical evaluation of retrieval-augmented generation systems for medical question answering. arXiv:2411.09213.

Pal, A., Umapathi, L. K., and Sankarasubbu, M. (2023). Med-HALT: Medical domain hallucination test for large language models. Proceedings of the 27th Conference on Computational Natural Language Learning, 314-334.

Robertson, S., and Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4), 333-389.

Singhal, K., Azizi, S., Tu, T., et al. (2023). Large language models encode clinical knowledge. Nature, 620, 172-180.

Xiong, G., Jin, Q., Lu, Z., and Zhang, A. (2024). Benchmarking retrieval-augmented generation for medicine. Findings of the Association for Computational Linguistics: ACL 2024, 6233-6251.

Qwen Team. (2025). Qwen2.5 technical report. arXiv:2412.15115.
