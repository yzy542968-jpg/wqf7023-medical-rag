# Evidence-Checking Agentic RAG for Radiology Report-Grounded Question Answering with Linked X-ray Cases

**Name:** ZHANG YUE  
**Matric No.:** 22097191  
**Programme:** Master of Artificial Intelligence  
**Course:** WQF7023 Artificial Intelligence Research Project  
**Supervisor:** Dr. Uzair Ishtiaq  
**Document status:** Final manuscript, automated results frozen  
**Version date:** 16 August 2026

## Abstract

Large language models can produce fluent medical answers while still using evidence from the wrong patient, combining findings from multiple examinations, or adding unsupported claims. Retrieval-augmented generation (RAG) provides external evidence, but retrieval alone does not guarantee that the selected evidence belongs to the intended radiology case or that the generated answer is faithful to it. This research develops and evaluates an evidence-checking RAG workflow for radiology report-grounded question answering using the real IU X-Ray/OpenI collection. The modeled input is report text; linked X-ray files are retained for case presentation but are not supplied to the language model.

The study uses two complementary benchmarks. V1 is an open-corpus stress test containing 120 cases and 360 report-derived questions, with 84 development cases and 36 case-disjoint held-out cases. It compares LLM-only answering, report BM25, case BM25, biomedical dense retrieval, hybrid retrieval, adaptive retrieval, local Qwen2.5 generation, and a rule plus Medical-NLI evidence checker. The locked V1 system achieved held-out Token-F1 of 0.206 with a case-bootstrap 95% confidence interval of [0.167, 0.246]. Its improvement over the Case-BM25 semantic-agent baseline was +0.035, with an unadjusted paired randomization p-value of 0.0145 but a Holm-adjusted p-value of 0.0870. Oracle retrieval increased verified Token-F1 to 0.425, showing that retrieval remained the dominant bottleneck. The evidence checker rejected 100% of synthetic polarity reversals on a development-only stress test, but it still reported evidence support for 83.4% of answers produced after wrong-case retrieval. This demonstrates that evidence faithfulness and patient correctness are different properties.

V2 evaluates an explicit case-ID workflow on 600 previously unused OpenI cases and a separate 120-case confirmation cohort. It uses sentence chunks, deterministic section routing, calibration-locked top-k=6, evidence-only generation, and advisory NLI auditing. Confirmation evidence recall was 99.4% and Qwen Token-F1 was 0.570, with a 95% confidence interval of [0.556, 0.584]. However, an extractive retrieved-context baseline achieved 0.997 Token-F1, and the routed candidate set equaled the relevance set for every query. V2 therefore validates case isolation and workflow control, not semantic retrieval superiority or a generation gain.

The main contribution is a reproducible failure analysis and system design that separates patient scope, retrieval correctness, answer quality, and evidence support. The research supports explicit case scoping, calibrated abstention, and transparent advisory verification, while rejecting claims of autonomous clinical reasoning, image diagnosis, or deployment readiness. A blinded human-evaluation protocol was prepared but not conducted because no suitable independent reviewer was available before submission. Consequently, no human or clinical validation is claimed.

**Keywords:** retrieval-augmented generation, medical question answering, radiology reports, evidence checking, patient scope, hallucination, abstention

# Chapter 1: Introduction

## 1.1 Background

Large language models (LLMs) have improved natural-language generation, summarization, and question answering, including tasks that involve medical knowledge. Their fluency can nevertheless conceal factual or evidential errors. In a healthcare-oriented setting, a plausible answer is not sufficient: the answer must be traceable to the correct source and must not combine facts from different patients. Medical LLM research has therefore emphasized that benchmark performance should not be interpreted as clinical safety or deployment readiness (Singhal et al., 2023). Medical hallucination benchmarks similarly show that models can produce statements that sound medically credible but are unsupported or incorrect (Pal et al., 2023).

Retrieval-augmented generation addresses part of this problem by providing external evidence before answer generation. The original RAG formulation combines a parametric generator with retrieved non-parametric memory (Lewis et al., 2020). In principle, the retrieved evidence improves provenance and reduces the need to rely only on model parameters. In practice, a RAG pipeline can fail at several distinct stages. The query may not identify a unique record, the retriever may rank the wrong record, the prompt may expose evidence from multiple records, the generator may ignore the evidence, and an automatic verifier may confirm that an answer matches the retrieved evidence even when the retrieved evidence belongs to the wrong patient.

Radiology report question answering provides a useful setting for studying these failures. A radiology examination is naturally case-based: an indication, findings, impression, and one or more images belong to the same examination. The IU X-Ray/OpenI collection provides de-identified chest X-ray studies with linked reports and images (Demner-Fushman et al., 2016). The collection allows case-level retrieval and explicit analysis of whether generated claims remain within the selected case boundary.

This project does not analyze image pixels or perform autonomous chest X-ray diagnosis. The report is the modeled evidence source. Image filenames and projection metadata remain linked for presentation, traceability, and future multimodal extension. This distinction is important because displaying an image in a dashboard is not equivalent to providing that image to a vision-language model.

## 1.2 Problem Statement

Many RAG evaluations assume that a query uniquely identifies the relevant document and that support found anywhere in the retrieved top-k context is acceptable. These assumptions are unsafe for case-based medical data. If a generic clinical question is compatible with several radiology cases, retrieving a semantically similar report does not establish that it is the intended patient. If the generator sees top-k reports, it may combine findings from several examinations. A sentence-level checker may then mark those statements as supported because each statement appears somewhere in the pooled context.

The core research problem is therefore:

> How can a report-grounded medical RAG workflow preserve radiology case boundaries and distinguish retrieval correctness from answer faithfulness, while using evidence checking and abstention without overstating clinical reliability?

The problem contains four coupled risks:

1. **Identifiability risk:** the question text may not contain enough information to identify a unique patient or examination.
2. **Retrieval risk:** a high similarity score may select a clinically similar but incorrect case.
3. **Generation risk:** the LLM may omit evidence, add unsupported content, or combine evidence across cases.
4. **Verification risk:** an answer may be faithful to retrieved evidence while still being wrong for the intended patient.

## 1.3 Aim and Objectives

The aim is to develop and critically evaluate a case-scoped, evidence-checking RAG workflow for radiology report-grounded question answering.

The objectives are:

1. Process the real IU X-Ray/OpenI collection into reproducible case-level records with report sections and linked image metadata.
2. Compare lexical, biomedical dense, hybrid, reranked, and adaptive retrieval methods using case-disjoint evaluation.
3. Compare LLM-only, report-RAG, case-RAG, extractive, and evidence-checked answer systems.
4. Quantify query ambiguity, cross-case contamination, retrieval headroom, answer support, and abstention.
5. Test a rule plus Medical-NLI checker under both ordinary outputs and polarity-reversal stress cases.
6. Redesign the workflow so that patient identity is supplied as explicit metadata rather than inferred from semantic similarity.
7. Provide a reproducible dashboard and an auditable agent trace without claiming autonomous clinical decision making.

## 1.4 Research Questions

**RQ1.** How does retrieval augmentation affect report-grounded answer quality compared with LLM-only answering in an open-corpus radiology QA stress test?

**RQ2.** How do lexical, biomedical dense, hybrid, reranked, and adaptive retrieval choices affect correct-case retrieval and downstream answer quality?

**RQ3.** What can an evidence-checking agent reliably detect, and where does evidence verification fail to protect against wrong-patient retrieval?

**RQ4.** What changes when patient identity is supplied as an explicit case scope and evidence retrieval is restricted to the selected report?

## 1.5 Contributions

This thesis makes five scoped contributions.

First, it provides a real-data, case-disjoint radiology RAG evaluation with separately reported retrieval, generation, verification, and statistical results. Second, it quantifies a cross-case contamination failure mode in which pooled evidence can hide that an answer mixes findings from different reports. Third, it demonstrates through oracle retrieval and retrieval-conditioned analysis that downstream generation quality is strongly limited by correct-case retrieval. Fourth, it evaluates evidence checking conservatively, including a polarity stress test and a failure analysis showing that local evidence support does not imply patient correctness. Fifth, it provides a controlled patient-known workflow that enforces case isolation and uses advisory verification when automatic rewriting is not calibration-safe.

The contribution is not a claim of state-of-the-art medical QA. It is a reproducible analysis of system boundaries and a design for making those boundaries visible.

## 1.6 Scope

The study is limited to de-identified radiology report text from OpenI. Linked chest X-ray images are displayed but are not model inputs. The local generator is Qwen2.5-1.5B-Instruct, selected for reproducible execution on available hardware. The semantic checker is a Medical-NLI model used as an automatic risk indicator. None of the systems is clinically validated, authenticated against a hospital record system, or intended for diagnostic use.

## 1.7 Thesis Organization

Chapter 2 reviews related work. Chapter 3 presents the data, systems, and evaluation design. Chapter 4 reports results and validity audits. Chapter 5 discusses the findings, limitations, ethics, conclusions, and future work.

# Chapter 2: Literature Review

## 2.1 Retrieval-Augmented Generation

RAG was proposed to combine language generation with retrieved knowledge for knowledge-intensive tasks (Lewis et al., 2020). The approach is attractive because it offers a visible evidence path and allows information to be changed without retraining all model parameters. However, RAG should be evaluated as a pipeline. A strong generator cannot recover a report that the retriever never exposes, and a strong retriever cannot guarantee that a generator will use evidence correctly.

Evaluation must therefore separate context retrieval from answer generation. Frameworks such as RAGAS emphasize multiple dimensions including context relevance, answer relevance, and faithfulness (Es et al., 2024). This project follows that separation but adds a patient/case dimension: support in a semantically related report is not interchangeable with support in the intended examination.

## 2.2 Sparse, Dense, and Hybrid Retrieval

BM25 remains a strong and transparent sparse retrieval baseline based on probabilistic term matching (Robertson and Zaragoza, 2009). It performs well when questions share terms with report indications, findings, or problem labels. Its limitations include sensitivity to vocabulary mismatch and inability to model deeper semantic similarity.

Dense retrieval represents queries and documents in a learned embedding space. MedCPT was contrastively pretrained from large-scale PubMed search logs for biomedical retrieval (Jin et al., 2023), making it more domain appropriate than a generic sentence encoder. Dense retrieval can capture semantic similarity but can also rank clinically similar reports that belong to a different case. Hybrid retrieval combines normalized sparse and dense scores. This study treats hybrid fusion as an empirical question rather than assuming that biomedical dense retrieval must outperform lexical retrieval.

Reranking provides a second-stage relevance estimate over a smaller candidate set. It can improve precision when the first-stage retriever produces plausible candidates, but it cannot solve a query that lacks a unique patient identity. Adaptive retrieval and abstention are therefore also studied: the system may choose among agreeing retrievers or decline to select a case when confidence is insufficient.

## 2.3 Medical RAG

Medical RAG benchmarks show that performance depends on the corpus, retriever, generator, and task. MedRAG/MIRAGE compared retrieval and generation configurations across medical QA datasets and found that retrieval can improve performance but does not behave uniformly across settings (Xiong et al., 2024). Practical medical RAG evaluation also needs to consider noisy, misleading, or insufficient evidence rather than only ideal retrieval (Ngo et al., 2024).

These findings motivate two choices in this study. First, multiple retrieval and answer baselines are evaluated on the same split. Second, negative findings are reported rather than removed. Dense retrieval underperformance, non-significant adjusted comparisons, and verifier false confidence are part of the research result because they describe when a medical RAG system should not be trusted.

## 2.4 Radiology Reports as Case-Based Evidence

The OpenI/IU X-Ray collection was prepared for distribution and retrieval research and contains radiology examinations with associated reports and chest X-ray images (Demner-Fushman et al., 2016). A report commonly includes an indication, comparison, findings, and impression. These sections have different roles. Indication describes the clinical reason for the examination, findings describe observations, and impression summarizes the radiologist's conclusion.

Treating an examination as a case differs from treating every report or sentence as an interchangeable document. Case-level retrieval preserves ownership of findings and impressions. This study retains stable case identifiers and linked image metadata throughout preprocessing, splitting, retrieval, generation, and evaluation.

## 2.5 Medical Hallucination and Evidence Checking

Medical language generation has a higher reliability requirement than general conversation because unsupported statements may affect interpretation or action. Med-HALT shows that medical hallucinations can be plausible and difficult to detect from fluency alone (Pal et al., 2023). Medical LLM studies likewise distinguish benchmark knowledge from real clinical readiness (Singhal et al., 2023).

Evidence checking can be lexical, semantic, or hybrid. Lexical checking is transparent but may reject legitimate paraphrases. Natural-language inference estimates whether evidence entails, contradicts, or is neutral toward a claim, but domain NLI models are imperfect and can be sensitive to sentence segmentation, negation, and composite claims. A verifier can also answer the wrong question: it may accurately determine that a claim follows from a retrieved report, while failing to determine whether that report belongs to the intended patient.

For this reason, the thesis distinguishes:

- **retrieval correctness:** whether the intended case was selected;
- **evidence faithfulness:** whether the answer follows from the selected evidence;
- **answer correctness:** whether the answer matches the reference for the intended case;
- **clinical validity:** whether the answer is acceptable for real clinical use.

Only the first three are partially measured here. Clinical validity is not established.

## 2.6 Agentic RAG and Terminology

The term agentic RAG is often used for systems that choose actions such as query reformulation, retrieval, reranking, verification, and abstention. This project implements an auditable workflow with planner, retrieval, generation, evidence audit, and action-policy stages. Some decisions are adaptive, including retrieval abstention and evidence-based answer actions. Other decisions are deterministic, especially the V2 mapping from known question type to report section.

The system is therefore an evidence-checking agent workflow, but it is not presented as an autonomous clinical agent. A deterministic type-to-section rule is not learned reasoning. An NLI score is not a clinical judgment. This terminology boundary is included in the method and dashboard.

## 2.7 Benchmark Validity and Natural Questions

Benchmark construction can create shortcuts. If a query is generated directly from the target report or includes labels copied into the retrieval document, lexical overlap may make retrieval easier. Conversely, if many cases share the same generic question, globally searching for a unique target may be impossible. A benchmark should disclose these conditions rather than treating all errors as model failures.

RadQA is an important future benchmark because physicians created questions from referral information without first seeing the answer context, and the dataset contains answerable and unanswerable items (Soni et al., 2022). It is not used in the completed experiments because the files require credentialed PhysioNet access. The present thesis instead audits the limitations of its report-derived OpenI questions and uses V2 only as a controlled workflow benchmark.

## 2.8 Research Gap

Previous work establishes RAG, biomedical retrieval, medical QA, and hallucination evaluation, but a narrower gap remains: how should a report-grounded RAG system represent patient scope, and what does an evidence verifier actually prove when retrieval may select the wrong case?

This thesis addresses that gap by connecting four analyses that are often reported separately: case retrieval, generated answer overlap, evidence support, and patient/case ownership. The central hypothesis is not that one model eliminates hallucination, but that explicit scope and calibrated actions make failure modes measurable and reviewable.

# Chapter 3: Methodology

## 3.1 Research Design

The research follows an empirical system-comparison design with development-only selection and case-disjoint evaluation. It has two benchmark stages.

V1 is an open-corpus stress test. The question is used to search across candidate OpenI cases. This measures retrieval difficulty, ambiguity, contamination, generation, and evidence checking under imperfect patient identification. V2 is a patient-known workflow. The case identifier is supplied explicitly, evidence is filtered to that case, and a deterministic rule selects the report section. V2 measures case isolation and controlled workflow behavior, not the same task as V1.

The two stages must not be treated as a before-and-after model comparison because the task definition changes. V1 asks the system to infer the case from question text. V2 assumes the case is already known.

## 3.2 Data Source and Processing

The local OpenI processing pipeline produced 3,851 report cases and 7,466 image-projection mapping rows. Each case record contains a stable case ID, indication, comparison when available, findings, impression, problem labels, and linked image metadata. Whitespace was normalized while the original clinical wording and de-identification markers were preserved.

The modeled pipeline uses report text only. Images are retained in the raw data directory for dashboard previews and are excluded from the Git repository. The public, de-identified character of the data reduces privacy risk, but it does not remove the need for responsible interpretation. No attempt is made to re-identify patients.

## 3.3 V1 Open-Corpus Benchmark

The V1 benchmark contains 120 cases and 360 questions, with three questions per case. The question types are findings from indication, impression from indication, and abnormality summary. Cases, not individual questions, are the grouping unit.

The frozen split uses 84 development cases with 252 questions and 36 held-out cases with 108 questions. There is no case overlap. Retrieval weights, reranker policy, adaptive thresholds, prompt mode, and verifier thresholds were selected only using development data. The held-out split was used for the final locked evaluation.

A validity audit found that 23.1% of held-out question rows share their question text with another target case. It also found query-document shortcuts because indication or problem text may appear in indexed case fields. V1 is therefore framed as an open-corpus stress test and failure analysis, not a clean natural-question benchmark.

## 3.4 V2 Case-Scoped Benchmark

V2 uses 600 cases that were not used in V1. They are split into 360 development, 120 calibration, and 120 diagnostic-test cases. A further 120-case confirmation cohort is disjoint from V1 and the 600 main V2 cases. The confirmation cohort is the primary final automated V2 evidence because the earlier test cohort was inspected before verifier-action calibration.

Reports are segmented into sentence-level chunks with stable identifiers of the form `case_id::section::position`. Every case contributes three generic questions: documented findings, final impression, and report conclusion. The case ID is supplied separately from the question. The prototype uses it as a metadata filter; it does not implement authentication or hospital access control.

The route is deterministic:

- findings questions search the findings section;
- impression questions search the impression section;
- summary questions search the impression section.

Calibration selected top-k=6 as the smallest value achieving at least 0.95 mean evidence recall. The validity audit later established that the routed candidate pool equals the relevance set for every query. Routed Hit@1 is therefore a routing sanity check.

## 3.5 Retrieval Systems

The retrieval comparisons include:

1. TF-IDF as an initial lexical baseline.
2. BM25 as the principal transparent sparse retriever.
3. MedCPT query/document encoders as biomedical dense retrieval.
4. Weighted hybrid fusion of normalized BM25 and MedCPT scores.
5. MedCPT cross-encoder reranking over first-stage candidates.
6. Adaptive retrieval that uses agreement, margins, score thresholds, and abstention.

The hybrid weight was swept on the development split. The locked alpha was 0.30. The adaptive policy was also selected on development data subject to a minimum coverage criterion. On each held-out question the policy selected a case or abstained.

V2 adds three conditions: global BM25, case-scoped BM25 over all sections, and case-scoped routed BM25 over the target section. These conditions diagnose the effect of explicit scope and routing; they do not constitute the same retrieval task as V1.

## 3.6 Generation Systems and Prompting

The generator is `Qwen/Qwen2.5-1.5B-Instruct`, executed locally with CUDA. The experiment compares LLM-only answering and several RAG contexts. Report-RAG can expose retrieved report passages, while case-RAG keeps evidence associated with a selected case. Prompt ablations included direct, evidence-guided, and structured case-aware forms. Prompt selection was performed on development data.

The final V1 workflow uses a direct evidence-grounded prompt with adaptive case retrieval. The final V2 workflow uses an evidence-only prompt containing the explicit case scope and six routed evidence sentences. Temperature-sensitive sampling is not used in the locked evaluation, supporting reproducibility.

An extractive baseline is included where the task allows it. This is essential for V2 because the reference is inherited from an entire report section. Returning the retrieved context provides a test of whether generation adds value beyond extraction.

## 3.7 Evidence-Checking Agent

The implemented workflow has explicit stages:

1. **Scope:** identify or receive the allowed case context.
2. **Plan:** derive retrieval intent or map a known task type to a section.
3. **Retrieve:** rank evidence and optionally abstain from case selection.
4. **Generate:** produce an answer using the selected evidence.
5. **Audit:** split the answer into claims or sentences and compare them with evidence.
6. **Act:** preserve, filter, abstain, or flag the answer for review according to a calibrated policy.

The semantic checker combines lexical support, Medical-NLI entailment and contradiction probabilities, and a hard polarity guard. It records matched evidence and a decision reason for every checked sentence. The V1 policy can filter unsupported statements or abstain. V2 calibration compared sentence filtering, contradiction-only filtering, and audit-only behavior. Because all rewriting policies reduced calibration Token-F1, V2 locks `audit_only`: it reports risk without silently changing the answer.

## 3.8 Evaluation Metrics

Retrieval metrics are Hit@k, Recall@k, Mean Reciprocal Rank (MRR), coverage, selective accuracy, and abstention. Correct-case retrieval is evaluated independently of answer support.

Answer quality is measured with case-averaged Token-F1 and RadGraph entity/relation metrics. Token-F1 measures lexical overlap and can penalize concise paraphrases or reward copied text. RadGraph adds a clinical entity and relation perspective but remains an automatic metric.

Evidence behavior is measured with support rate, revision rate, contradiction count, and answer abstention. The development polarity stress test measures whether an entailed sentence is accepted and a mechanically polarity-reversed sentence is rejected. It is not a general clinical validation set.

V2 additionally reports mean evidence recall and the extractive retrieved-context baseline. The structural validity audit reports unique question strings, zero-score retrieval queries, candidate-qrel equivalence, and the generation-minus-extraction gap.

## 3.9 Statistical Analysis

Confidence intervals use 10,000 grouped bootstrap samples at the case level, preserving correlation among the three questions from each case. Pairwise system differences use paired randomization tests. Because ten exploratory pairwise comparisons were reported, Holm adjustment controls family-wise error. Both unadjusted and adjusted values are shown to prevent selective interpretation.

## 3.10 Human-Evaluation Protocol and Disposition

Two 36-case system-blinded evaluation packages were frozen: one for the V1 held-out benchmark and one for the V2 confirmation cohort. Each item presents four anonymized responses and asks an independent reviewer to rate correctness, evidence grounding, and potential harmfulness, then choose the best response. The response-to-system key is stored separately. This protocol remains a reproducible artifact for future external validation.

No suitable independent reviewer was available before the P2 submission deadline, so the protocol was not executed. Human results are excluded rather than estimated, and the empty rating fields are not interpreted as data. This decision was made without examining system identities or changing the frozen V1/V2 systems.

## 3.11 Reproducibility and Implementation

The project is implemented in Python with NumPy, pandas, PyTorch, Transformers, and Streamlit. Source code is separated into data, retrieval, agent, and evaluation modules. Random seeds, split manifests, selected configurations, generated summaries, and content fingerprints are stored in the repository. Large model caches, raw images, dense indexes, secrets, and the virtual environment are excluded.

The main dashboard presents the live V2 patient-scoped workflow, the V1 stress-test workflow, frozen experiment tables, caveats, and evidence traces. A separate rating application preserves the unexecuted blinded protocol without loading the system key.

# Chapter 4: Results and Analysis

## 4.1 V1 Retrieval Results

Development selection locked hybrid alpha at 0.30. On the 108-question held-out split, hybrid retrieval achieved Hit@1 of 0.287, Hit@20 of 0.509, and MRR of 0.331. The adaptive policy produced 92.6% coverage, 31.0% selective accuracy among non-abstained queries, and 7.4% retrieval abstention.

These values show that retrieval is difficult. Even after sparse-dense fusion and adaptive selection, fewer than one third of held-out questions selected the intended case at rank one. This limits every downstream system that expects one case-specific answer.

## 4.2 V1 Answer Results

Table 4.1 reports the principal held-out answer systems.

| System | Token-F1 | Grouped 95% CI |
|---|---:|---:|
| LLM only | 0.079 | [0.069, 0.089] |
| Report-RAG + semantic checker | 0.153 | [0.131, 0.175] |
| Case-BM25 + semantic checker | 0.172 | [0.136, 0.211] |
| Previous hybrid + semantic checker | 0.204 | [0.174, 0.238] |
| Final adaptive system | 0.206 | [0.167, 0.246] |

Retrieval augmentation improved substantially over LLM-only answering. Case-level systems also outperformed report-level RAG numerically. The final adaptive system achieved the highest point estimate, but uncertainty intervals overlap.

The final system improved over Case-BM25 + semantic checker by +0.035 Token-F1, with a paired 95% interval of [0.007, 0.062]. The unadjusted paired randomization p-value was 0.0145. After Holm adjustment across ten exploratory comparisons, p=0.0870. The result is promising but not statistically significant at the conventional 0.05 family-wise threshold. The thesis therefore reports a numerical improvement with unadjusted evidence, not a definitive superiority claim.

The final system's RadGraph complete F1 was 0.163. This was close to the unverified final draft and previous hybrid values, again suggesting that the verifier did not create a large automatic clinical-structure gain.

## 4.3 Retrieval Headroom

Oracle evaluation replaced the retrieved case with the known target case while keeping the answer pipeline comparable. Verified Token-F1 increased from 0.206 to 0.425, an absolute gap of 0.219. Retrieval-conditioned analysis provides the same conclusion:

| Retrieval condition | Questions | Verified Token-F1 | Evidence support |
|---|---:|---:|---:|
| Correct case retrieved | 31 | 0.451 | 87.8% |
| Wrong case retrieved | 69 | 0.117 | 83.4% |
| Retrieval abstained | 8 | 0.031 | 0.0% |

Correct retrieval more than tripled verified answer quality compared with wrong retrieval. However, evidence support remained high after wrong retrieval. The checker was often answering, "Is this answer supported by the selected report?" rather than, "Was the correct patient's report selected?" This is the central negative finding of the study.

## 4.4 Cross-Case Contamination

An automated detector analyzed report-level top-5 generated answers. Depending on whether a conservative lexical anchor was required, it estimated that 19.4% to 28.9% of answer sentences and 57.4% to 65.7% of complete answers contained support traceable to a retrieved non-target case.

These are detector estimates rather than human-confirmed contamination labels. They nevertheless demonstrate why all-top-k evidence checking is structurally weak for patient-specific QA. A generator can synthesize a coherent answer from several reports, and a pooled verifier can find support for each sentence while ignoring case ownership.

## 4.5 Evidence-Checker Results

The final V1 system increased Token-F1 from 0.199 for the draft to 0.206 after verification. The difference between the final verified answer and its own draft was +0.007; its confidence interval crossed zero and the paired randomization p-value was 0.324. This does not support a strong claim that verification generally improved answer overlap.

The development-only polarity stress test contained 120 entailed/contradictory pairs. The semantic checker accepted 100% of entailed claims and rejected 100% of mechanically polarity-reversed claims after the hard polarity guard. The result supports sensitivity to this narrow synthetic error class. It does not establish general clinical contradiction detection.

V2 verifier calibration further showed that action policy matters. Advisory audit preserved calibration Token-F1 at 0.538. The best sentence-filter policy reduced it to 0.493, and the best contradiction-only filter reduced it to 0.491. These failures motivated the final V2 `audit_only` policy.

## 4.6 V2 Retrieval and Evidence Coverage

The disjoint confirmation cohort contained 120 cases and 360 questions. Explicit case filtering guaranteed that retrieved chunks belonged to the supplied case. The routed system achieved Hit@1 of 1.000 and Recall@5 of 0.978. With the calibration-locked top-k=6, mean confirmation evidence recall reached 0.994.

These numbers should be interpreted narrowly. The planner used the supplied question type to select the same report section used to define relevant chunks. The routed candidate pool therefore equaled the relevance set for 100% of queries. In addition, 57.2% of confirmation routed queries returned all-zero BM25 scores, and 77.6% of returned routed scores were zero. Perfect Hit@1 primarily reflects that every candidate was labelled relevant, not that BM25 discriminated semantically.

## 4.7 V2 Generation and Extractive Baseline

Qwen2.5-1.5B achieved confirmation Token-F1 of 0.570 with a case-bootstrap 95% confidence interval of [0.556, 0.584]. Mean evidence support audit was 0.830 and evidence recall was 0.994. Findings questions achieved F1 of 0.680, impression questions 0.612, and summary questions 0.418.

The extractive retrieved-context baseline achieved Token-F1 of 0.997. Qwen was lower by 0.427. This reverses a superficially positive interpretation of the 0.570 score: the generator compressed or paraphrased a section-derived reference and lost lexical coverage. V2 does not show that generation improves the task. It shows that explicit scope can isolate evidence and that an extractive answer is more appropriate when the question requests an entire report section.

## 4.8 Replication and Evaluation Protocol

The initial V2 diagnostic test produced Qwen Token-F1 of 0.566 with interval [0.551, 0.581], close to the confirmation result of 0.570. This supports numerical stability across two disjoint cohorts. However, the first test output was inspected before the V2 verifier action was calibrated. It is consequently treated as diagnostic evidence. The later disjoint confirmation cohort is the primary final automated result.

## 4.9 Human-Evaluation Status

The V1 and V2 blinded evaluation packages each contain 36 cases and four anonymized candidate responses per case. No independent human evaluation was conducted because no suitable reviewer was available before submission; both packages therefore remain at 0 of 36 completed rows. No human correctness, grounding, preference, harmfulness, or agreement result is claimed in this chapter.

The absence of human ratings limits external validity and prevents claims of human-perceived correctness or clinical safety. It does not justify substituting automatic NLI or RadGraph scores for human judgement. The automated comparisons remain reportable because their systems, splits, metrics, and statistical procedures were frozen independently of the unexecuted protocol.

## 4.10 Summary of Findings

The results support six conclusions:

1. Retrieval augmentation outperformed LLM-only answering in V1, but absolute answer quality remained low.
2. Correct-case retrieval was the largest measured bottleneck; oracle evidence nearly doubled final Token-F1.
3. A verifier can report high evidence support after selecting the wrong case, so support is not equivalent to patient correctness.
4. The polarity guard improved rejection of synthetic negation reversals, but broader clinical verification remains unvalidated.
5. Explicit case scoping removed cross-case retrieval in V2, but the routed retrieval task was structurally trivial.
6. The V2 extractive baseline strongly outperformed Qwen, so generation should not be claimed as beneficial for that task.

# Chapter 5: Discussion and Conclusion

## 5.1 Answers to the Research Questions

### RQ1: Retrieval augmentation versus LLM-only answering

Retrieval augmentation improved held-out Token-F1 from 0.079 for LLM-only answering to 0.153 for report-RAG and 0.172 to 0.206 for case-based systems. The answer is therefore yes in this benchmark: external report evidence improved lexical and clinical-structure overlap. The result does not mean the system is accurate enough for clinical use. The best verified F1 remained 0.206 because most errors originated before generation.

### RQ2: Retrieval choices

Hybrid and adaptive retrieval produced the strongest V1 point estimates, but held-out Hit@1 remained 0.287. The result supports combining complementary sparse and biomedical semantic signals, while also showing that retriever sophistication cannot solve missing patient identity. The adjusted pairwise evidence does not justify a categorical claim that the final system is superior to every baseline.

### RQ3: Evidence-checking reliability

Evidence checking reliably detected the synthetic polarity reversals used in the development stress test and provided transparent sentence-level traces. It did not reliably protect against wrong-patient evidence. Wrong-retrieval answers still received 83.4% evidence support. Automatic rewriting also harmed V2 calibration performance. The appropriate conclusion is that the checker is an advisory local-faithfulness component, not a patient-correctness or clinical-safety oracle.

### RQ4: Explicit case scope

Supplying the case identifier as metadata and filtering evidence to that case structurally prevented cross-case retrieval. This is an important architecture correction. However, the current V2 routing benchmark is too easy to measure semantic retrieval because routing reveals the relevance section. Explicit scope solves case ownership; it does not by itself solve natural question understanding, evidence selection, or clinical answer quality.

## 5.2 Theoretical and Practical Implications

The main theoretical implication is that RAG faithfulness has levels. A claim can be faithful to a sentence, a report, or a pooled corpus while still being invalid for the intended patient. Medical RAG evaluation should include ownership or scope as an explicit relation between query, record, and evidence.

The practical implication is that patient identity should not be guessed from clinical similarity when an application already has a record identifier. A safer workflow first enforces authorized record scope, then performs semantic retrieval within that scope. The present prototype implements the metadata-filtering behavior but not real authentication or authorization.

The study also supports selective system behavior. When retrieval confidence is low, abstention can expose uncertainty rather than silently selecting a similar case. When verifier calibration shows that rewriting removes supported content, an advisory review flag is safer than automatic deletion.

## 5.3 Limitations

The V1 questions are report-derived and contain ambiguity and lexical shortcuts. The benchmark is small, with only 36 held-out cases. Qwen2.5-1.5B is substantially smaller than current frontier models, although its local execution supports reproducibility. Token-F1 is sensitive to wording and does not establish clinical correctness. RadGraph and Medical-NLI are automatic model-based metrics with their own errors.

The V2 benchmark contains only three unique question templates. Relevance is inherited from report-section membership rather than independently annotated sentence relevance. Routing receives the gold question type, and its candidate pool equals qrels. V2 therefore cannot support a claim of learned planning, difficult retrieval, or generation superiority.

The image files are not model inputs. The project does not evaluate visual question answering or image-report consistency. The dashboard's case selector is not connected to an authenticated clinical system. No independent human evaluation was conducted, so automatic answer metrics and evidence-checker outputs cannot establish human-perceived or clinical correctness.

## 5.4 Ethics and Safety

The data are de-identified and used for research. Nevertheless, medical AI outputs can be misinterpreted. The dashboard labels the system as a research prototype and separates retrieved evidence from generated answers. It avoids recommendations for treatment and does not present outputs as diagnoses.

The research reports negative and non-significant results to reduce misleading performance claims. It also keeps the V1/V2 task difference visible. The unused system-key files remain separated from the blinded packages, and no ratings were fabricated after an independent reviewer could not be obtained. Raw images, model caches, secrets, and large machine-local artifacts are excluded from the release repository.

## 5.5 Future Work

The highest-priority future study is a natural-question benchmark with independently annotated answer spans, unanswerable questions, and hard-negative evidence. RadQA is suitable when credentialed access is available because physicians generated questions from referral information and annotated concise answer spans (Soni et al., 2022).

A stronger planner should infer evidence needs from free-form questions rather than receiving a gold type. Its actions should be evaluated separately: query reformulation, section selection, retrieval, reranking, answer generation, verification, and abstention. Hard negatives should include same-report distractors and clinically similar sentences from other records permitted by the task design.

Verifier evaluation should use human-labelled entailment, contradiction, unsupported, and composite-claim cases. Risk-coverage curves and selective accuracy would be more informative than one support threshold. A separate multimodal extension could evaluate image pixels using a validated vision-language model and an appropriate dataset, but it should not be mixed with the current text-only results.

## 5.6 Conclusion

This thesis investigated evidence-checking RAG for radiology report-grounded question answering using real OpenI cases. The experiments show that retrieval helps compared with LLM-only answering, but correct-case retrieval remains the dominant bottleneck. They also show that evidence support can be misleading when the selected case is wrong, and that automatic verifier actions can reduce answer quality when not carefully calibrated.

The architecture was therefore revised around explicit case scope, case-isolated evidence retrieval, transparent action traces, and advisory review. This correction prevents cross-case evidence from entering the V2 workflow. At the same time, the structural validity audit prevents the controlled V2 score from being overclaimed as difficult semantic retrieval.

The final contribution is not a clinically autonomous agent. It is a reproducible method for exposing where a medical RAG answer came from, whether the intended case was retrieved, whether the answer follows from that case, and when the system should abstain or request review. These distinctions are necessary foundations for safer future medical RAG research, but their clinical usefulness requires independent human validation that was outside the completed study.

# References

Bae, S., Kyung, D., Ryu, J., et al. (2023). EHRXQA: A multi-modal question answering dataset for electronic health records with chest X-ray images. *Advances in Neural Information Processing Systems*.

Demner-Fushman, D., Kohli, M. D., Rosenman, M. B., et al. (2016). Preparing a collection of radiology examinations for distribution and retrieval. *Journal of the American Medical Informatics Association, 23*(2), 304-310.

Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2024). RAGAS: Automated evaluation of retrieval augmented generation. *arXiv:2309.15217*.

Jin, Q., Kim, W., Chen, Q., et al. (2023). MedCPT: Contrastive pre-trained transformers with large-scale PubMed search logs for zero-shot biomedical information retrieval. *Bioinformatics, 39*(11), btad651.

Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*.

Ngo, N. T., Nguyen, C. V., Dernoncourt, F., and Nguyen, T. H. (2024). Comprehensive and practical evaluation of retrieval-augmented generation systems for medical question answering. *arXiv:2411.09213*.

Pal, A., Umapathi, L. K., and Sankarasubbu, M. (2023). Med-HALT: Medical domain hallucination test for large language models. *Proceedings of CoNLL*, 314-334.

Robertson, S., and Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval, 3*(4), 333-389.

Singhal, K., Azizi, S., Tu, T., et al. (2023). Large language models encode clinical knowledge. *Nature, 620*, 172-180.

Soni, S., Gudala, M., Pajouhi, A., and Roberts, K. (2022). RadQA: A question answering dataset to improve comprehension of radiology reports. *Proceedings of LREC 2022*, 6250-6259.

Xiong, G., Jin, Q., Lu, Z., and Zhang, A. (2024). Benchmarking retrieval-augmented generation for medicine. *Findings of ACL 2024*, 6233-6251.

Qwen Team. (2025). Qwen2.5 technical report. *arXiv:2412.15115*.

# Appendices

## Appendix A: Locked Result Sources

- `experiments/final_submission/final_results_registry.json`
- `experiments/final_optimized/final_test/final_optimized_test_summary.json`
- `experiments/final_optimized/statistics/held_out_test_grouped_bootstrap.json`
- `experiments/final_optimized/validity_audit/research_validity_audit.json`
- `experiments/benchmark_v2/confirmation_evaluation/test_generation_summary.json`
- `experiments/benchmark_v2/validity_audit/benchmark_v2_validity_audit.json`

## Appendix B: Reproduction Entry Points

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -q
& ".\.venv\Scripts\python.exe" -m compileall -q app.py human_evaluation_app.py scripts src
& ".\.venv\Scripts\python.exe" scripts/build_final_results_registry.py
& ".\.venv\Scripts\python.exe" -m streamlit run app.py
```

## Appendix C: Human-Evaluation Protocol Status

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run human_evaluation_app.py --server.port 8502
```

The two blinded packages and rating interface are retained as reproducibility artifacts. The protocol was not conducted because no suitable independent reviewer was available before submission. No human score is reported or inferred from automatic metrics.
