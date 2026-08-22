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

The problem has both technical and methodological dimensions. Technically, the system must convert heterogeneous case elements into compatible retrieval signals, maintain the link between each image and its report, and pass only an explicitly selected report to the generator. A mistake at any one of these stages changes the meaning of all later measurements. If the wrong report is selected, for example, a semantically fluent answer may receive a high local support score because it accurately summarizes that wrong report. The verifier is then performing its stated task correctly while the end-to-end system is still misaligned with the target case. This illustrates why a medical RAG system cannot be evaluated only at the final response layer.

Methodologically, the benchmark must distinguish genuine image contribution from information already available in text. Referral indications often contain distinctive symptoms, prior procedures, or anatomical terms. When these terms also appear in the target report, conventional lexical retrieval can identify the case without using the image. Conversely, generic questions such as asking for the findings or impression contain almost no case-specific information. A multimodal experiment that compares only a weak generic-question baseline with a rich image-plus-indication input would therefore confound several effects. The present study separates these inputs so that the role of indication and image alignment can be estimated independently.

A further methodological difficulty concerns the definition of correctness. Retrieval correctness is defined against the frozen target case in the evaluation cohort. Reference consistency is measured against an answer derived from the target report. Report-level support is assessed against the report actually selected by the system. These criteria may disagree. The disagreement is not merely noise; it identifies where the pipeline failed. This thesis treats such disagreement as an analytical object and uses it to connect quantitative results with stage-specific qualitative interpretation.

Finally, the medical context requires conservative claims. The OpenI reports are expert-authored clinical documents, but the experiment is retrospective, de-identified, and restricted to a controlled corpus. No metric used here proves diagnostic accuracy, patient benefit, or deployment safety. The research problem is therefore framed as reliable evidence retrieval and traceability within a benchmark, not as replacement of radiologists or autonomous clinical decision making.

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

The significance of these contributions is threefold. From a scientific perspective, the work tests a specific causal interpretation of multimodal improvement. A correctly paired image should outperform deliberately mismatched images when all text, candidates, and scoring rules are held constant. This is a stronger design than reporting a single multimodal score because it supplies a negative control for alignment. The design does not prove that the encoder recognizes each pathology correctly, but it does test whether the paired image representation carries case-specific information that changes report ranking in the expected direction.

From an evaluation perspective, the work makes evidence ownership explicit. Many faithfulness metrics ask whether the answer is supported by the context shown to the generator. That is necessary but insufficient for case-based data. A clinically plausible answer supported by another patient's report cannot be treated as correctly grounded for the target case. By reporting target rank, Top-1 selection, reference overlap, support, verification actions, and abstention separately, the study offers a practical template for auditing other record-based RAG systems.

From an engineering perspective, the project demonstrates a complete reproducible workflow on consumer hardware. The system uses local model inference, deterministic cohort construction, frozen configuration files, artifact manifests, and explicit runtime reporting. This matters because medical AI studies are difficult to assess when only aggregate results are published. The repository exposes the steps required to rebuild the cohort, rerun retrieval, construct prompt packs, calculate statistics, and regenerate qualitative review materials, while keeping large or sensitive artifacts local.

## 1.7 Scope and Boundaries

The final experiment is limited to OpenI/IU-Xray and therefore remains within one data source. It uses a closed set of 240 candidate cases, 120 confirmation targets, and three report-derived question templates. The image encoder is frozen; the project does not train a new vision-language foundation model.

Images influence retrieval and reranking, but the Qwen answer generator receives selected report evidence rather than image pixels. The system therefore does not evaluate autonomous visual diagnosis. Token-F1, retrieval qrels, and automated semantic signals do not establish clinical correctness. The researcher-reviewed qualitative analysis is explanatory and is not independent radiologist adjudication.

The dashboard is a research demonstration rather than an authenticated hospital application. No treatment recommendation, clinical decision support, deployment safety, or external validation claim is made.

## 1.8 Conceptual Framework

The conceptual framework organizes the system around five linked stages: case construction, target retrieval, answer generation, evidence verification, and audit. Each stage has a distinct input, output, success criterion, and failure mode.

Case construction defines the unit over which evidence ownership is preserved. Each unit contains a stable case identifier, indication, findings, impression, report text, and one or more linked chest radiographs. The frozen cohort applies a reproducible case-level partition before evaluation so that development decisions cannot use confirmation cases. The processed source records do not contain a stable patient identifier, so patient-level independence cannot be verified. The resulting case record is not merely a convenient data structure. It is the boundary that determines which image and report are allowed to support one another.

Target retrieval receives a question and an optional indication or image signal. It returns an ordered list of candidate case reports. Success at this stage means that the frozen target report is ranked highly, especially at rank one because the downstream generator uses only the Top-1 report. Hit@k measures whether the target appears within a rank cutoff, while MRR preserves information about movement throughout the ranking. The extractive proxy measures how much answer-bearing text would be available if a candidate report were selected. Together these measures distinguish useful rank improvement from actual Top-1 success.

Answer generation receives the selected report and a fixed prompt. Its responsibility is to produce a concise answer grounded in that report rather than to recover the hidden target independently. This design creates a non-oracle test of transfer: better retrieval can improve generation only when it places more useful evidence into the fixed generator context. Generation failure may still occur after correct retrieval through omission, excessive focus on one finding, unsupported elaboration, or failure to express uncertainty.

Evidence verification operates sentence by sentence on the draft answer and selected report. It may retain supported content, filter content that fails the semantic criteria, or abstain when no acceptable content remains. Verification is intentionally treated as another fallible model component. Its automated support decision is not a clinical gold label and cannot determine whether the selected report belongs to the target case. The verifier is evaluated through aggregate behavior and researcher-reviewed examples rather than assumed to be correct by construction.

The audit stage joins information from all earlier stages. A single trace can show the target identifier, selected identifier, target rank, retrieved report, draft answer, verifier actions, final answer, reference, and metric changes. This supports stage-specific attribution. A low final Token-F1 after correct retrieval suggests generation or verification error; a high support score after wrong retrieval suggests target-case misalignment; and a support decrease after harmless prefix removal suggests metric sensitivity rather than substantive degradation.

The framework therefore distinguishes four nested questions. Was the correct case retrieved? Was the answer supported by the selected report? Was the answer consistent with the frozen target reference? Was the answer clinically correct and safe? The experiment provides evidence for the first three at different strengths but does not answer the fourth. Keeping these questions separate is the central conceptual discipline of the thesis.

## 1.9 Thesis Organization

Chapter 2 reviews RAG, biomedical retrieval, multimodal radiology representation, medical QA, evidence checking, and benchmark validity. Chapter 3 describes the paired image-report cohort, retrieval conditions, shuffled-image control, generation, verification, statistics, qualitative protocol, and reproducibility. Chapter 4 reports quantitative, qualitative, and runtime results. Chapter 5 interprets the findings, contributions, limitations, future work, and conclusion.

# Chapter 2: Literature Review

## 2.1 Retrieval-Augmented Generation

RAG combines a parametric language model with retrieved non-parametric evidence for knowledge-intensive generation (Lewis et al., 2020). This design can expose sources and update knowledge without retraining the complete generator. It also creates a multi-stage failure surface. Retrieval determines which evidence is available, generation determines how that evidence is expressed, and verification determines which claims are retained or flagged.

RAG evaluation should therefore separate retrieval relevance, answer relevance, and faithfulness. RAGAS formalizes several of these dimensions using automated metrics (Es et al., 2024). The present research adds target-case alignment as a separate dimension. In medical records, support from a related document is not equivalent to support from the intended examination.

The original RAG formulation retrieves passages that help a sequence-to-sequence model answer knowledge-intensive questions (Lewis et al., 2020). In an open-domain setting, several passages may legitimately contain the same fact. Medical case retrieval is structurally different. The relevant evidence is not any document that states a similar finding; it is the document associated with the intended examination. This means relevance has an ownership component. Two radiology reports may both describe cardiomegaly, but only one is evidence about the target case.

RAG changes the location of model risk rather than eliminating it. Parametric hallucination may be reduced when explicit evidence is supplied, yet retrieval can introduce irrelevant, contradictory, or cross-case context. A generator can copy unsupported detail from a retrieved document, and a verifier can certify that copied detail as entailed. The final response may consequently be locally faithful and globally wrong with respect to the target. Pipeline evaluation must therefore preserve provenance from target definition through retrieval and generation.

Another important distinction is between retrieval for knowledge access and retrieval for identity resolution. In knowledge access, the question describes a topic and several sources may be useful. In the present benchmark, repeated question templates provide almost no identity information. Indication text and image representations partly resolve which case is intended. The study is therefore evaluating a constrained form of case resolution followed by report-grounded QA, not only semantic document search.

## 2.2 Medical RAG and Question Answering

Medical RAG performance depends on the corpus, task, retriever, and generator. MedRAG/MIRAGE showed that retrieval can improve medical QA but that gains vary across datasets and configurations (Xiong et al., 2024). Practical evaluations also emphasize noisy, misleading, or insufficient evidence rather than assuming ideal retrieval (Ngo et al., 2024).

These findings support controlled component comparison. A high final answer score cannot reveal whether improvement originated from a text shortcut, correct retrieval, copied context, generation, or verification. Negative findings such as weak Top-1 improvement or reduced automated support are therefore evidence about system boundaries rather than results to hide.

Medical QA benchmarks vary considerably in what they require from a system. Some ask multiple-choice questions that can be answered from general biomedical knowledge. Others provide a passage and evaluate extractive or abstractive comprehension. Visual QA datasets ask questions about pixels, while record-based tasks combine structured and unstructured patient information. Results across these settings are not directly interchangeable because the available evidence and target definition differ.

Retrieval can help medical QA in at least three ways. It can supply knowledge absent from a small generator, constrain the answer to an authoritative source, and expose evidence for inspection. It can also fail in three corresponding ways: the corpus may be incomplete, the retriever may select misleading evidence, or the generator may ignore the evidence. A rigorous experiment should therefore define the corpus boundary, report retrieval performance independently, and test whether retrieval gains propagate to the final answer.

The present work focuses on report-grounded answers because the radiology report is the available expert interpretation of the paired images. This choice avoids pretending that the language model independently reads the radiograph. At the same time, using the image to retrieve the report creates a meaningful multimodal problem: the system must exploit visual-textual alignment without transferring image pixels directly into the answer generator.

## 2.3 Sparse, Dense, and Multimodal Retrieval

BM25 remains a strong transparent sparse-retrieval baseline based on probabilistic term matching (Robertson and Zaragoza, 2009). It is effective when a query shares terminology with a report, but it is sensitive to lexical overlap and may exploit benchmark shortcuts. Indication text can be especially discriminative because it describes symptoms, history, and reason for examination.

Dense retrieval encodes queries and documents in a shared vector space. MedCPT uses large-scale PubMed search logs for biomedical retrieval (Jin et al., 2023). General vision-language systems such as CLIP align images and text using contrastive learning (Radford et al., 2021). In radiology, domain-specific joint encoders can represent chest X-rays and reports within a medically relevant embedding space.

Hybrid and reranking systems combine complementary signals but do not guarantee Top-1 correctness. A target can move substantially upward while remaining below rank one. This distinction motivates reporting both rank-sensitive metrics such as MRR and decision metrics such as Hit@1.

Sparse retrieval offers interpretability because term overlap and document statistics determine the score. BM25 downweights frequent terms, rewards informative matches, and normalizes for document length. In this study it also acts as a diagnostic instrument. The contrast between question-only and indication-plus-question BM25 reveals how much case identity is already encoded in the textual input. A strong sparse baseline prevents a multimodal gain from being exaggerated against an artificially weak comparator.

Dense retrieval addresses vocabulary mismatch by placing semantically related expressions near one another in an embedding space. Biomedical encoders may recognize that different clinical phrases describe similar concepts even when exact tokens differ. However, dense similarity can blur patient boundaries: two reports with similar findings may be close although they belong to different cases. Domain adaptation improves semantic relevance but does not solve evidence ownership by itself.

Multimodal reranking adds a second relation. Instead of asking only whether the query text resembles a report, it asks whether the target image representation aligns with candidate report representations. The two scores have different ranges and distributions, so fusion requires normalization and a fixed weighting policy. Tuning that policy on confirmation outcomes would inflate performance. The final V5 design therefore freezes shortlist size, min-max normalization, equal weights, and deterministic tie breaking before confirmation analysis.

Reranking rather than full-corpus multimodal retrieval is also an engineering compromise. BM25 cheaply narrows the candidate set, after which cached report embeddings and one image embedding are compared. This structure limits GPU work and preserves a transparent text baseline. Its limitation is that a target excluded from the text shortlist cannot be rescued by the image. Reporting Hit@k and rank movement helps reveal this dependency.

## 2.4 Paired Radiology Images and Reports

The OpenI/IU-Xray collection was prepared for radiology distribution and retrieval research and contains chest X-ray images linked to reports (Demner-Fushman et al., 2016). The report commonly includes an indication, findings, and impression. These fields should remain associated with the same examination throughout preprocessing and evaluation.

Paired radiology data enable at least three different tasks. Image classification predicts labels from pixels. Image-report retrieval ranks matching images or reports. Visual question answering generates answers from image content. The present study evaluates paired-report retrieval followed by report-grounded QA. It does not equate retrieval of a matching report with diagnosis from an image.

Radiology reports also have an internal discourse structure. The indication states why the examination was requested, findings describe observations, and the impression summarizes the radiologist's conclusion. These sections are related but not redundant. Indication may contain clinical history not visible on the current image. Findings may include detailed normal and abnormal observations. Impression may prioritize only the most clinically relevant conclusion. A summary question may require information distributed across both findings and impression.

Preserving this structure matters for retrieval and evaluation. Flattening all text into anonymous chunks can detach a statement from its section and case. Retrieving several high-scoring chunks can also mix normal findings from one examination with abnormalities from another. Whole-report retrieval preserves local coherence, while a case record additionally retains the image association and stable identifier. The final system uses the report as the answer evidence but carries the broader case package through retrieval and trace logging.

De-identified public radiology data introduce their own limitations. Placeholder tokens may replace dates, names, or other details; reports may contain typographical inconsistencies; and images may vary in view, quality, or number. These properties are not merely preprocessing inconveniences. They can influence token overlap, NLI decisions, and qualitative interpretation. The thesis records such ambiguity as a data-level category instead of forcing every discrepancy into a model-error label.

## 2.5 Biomedical Vision-Language Representation

BioViL introduced radiology-specific image-text representation learning with localized and global alignment between chest X-rays and reports (Boecking et al., 2022). BioViL-T extended biomedical vision-language processing by exploiting temporal and multi-image structure (Bannur et al., 2023). Such encoders offer a stronger domain prior than generic visual embeddings for chest radiograph retrieval.

This thesis uses frozen BioViL-T representations as a reranking signal. Freezing the encoder limits computational cost and supports reproducibility, but it also limits the claim: the research evaluates the usefulness of an existing representation within RAG rather than proposing a new image encoder.

Multi-view examinations introduce an aggregation question. Frontal and lateral views may carry complementary information. The final system normalizes each view, averages views at case level, and normalizes the aggregate. This deterministic policy avoids learned fusion on the confirmation outcomes.

Contrastive vision-language representation learning is attractive for this task because it directly optimizes proximity between related images and text. A radiology-specific encoder is expected to capture domain features that a generic natural-image model may not represent well. Nevertheless, an embedding is not an explanation. A high image-report similarity score does not identify which anatomical feature produced the match, and it cannot be interpreted as a diagnostic probability.

Using a frozen encoder separates representation evaluation from model training. It removes training instability, reduces compute requirements, and avoids fitting to the small confirmation cohort. The trade-off is that the system cannot adapt the representation to the exact retrieval objective or local data distribution. Consequently, the study evaluates whether an established biomedical representation adds value under a fixed policy, not whether the best possible multimodal retriever has been achieved.

Case-level view averaging is similarly conservative. It treats every available view equally after normalization and produces one deterministic vector for the examination. Learned attention or view-specific fusion might improve performance, but it would introduce additional parameters and validation choices. The chosen policy is sufficient for testing the primary alignment hypothesis while keeping the experimental degrees of freedom limited.

## 2.6 Medical Visual and Report Question Answering

VQA-RAD contains clinically generated questions and answers about radiology images and demonstrates the value of natural clinician phrasing for visual QA (Lau et al., 2018). EHRXQA combines electronic health records and chest X-rays for multimodal QA (Bae et al., 2023). These tasks are relevant but differ from retrieving one paired report from a candidate corpus.

RadQA contains physician-authored questions, report contexts, answer spans, and unanswerable cases (Soni et al., 2022). It is a valuable future report-QA benchmark, but authorized PhysioNet access was not part of the frozen V5 experiment. The present questions are generated from report roles and are therefore controlled but linguistically narrow.

The distinction matters because a model can perform well on templated questions through metadata or section shortcuts. Results from report-derived templates should not be generalized to unrestricted clinical questions.

VQA-RAD and report QA differ in where evidence is located. In visual QA, the answer may depend on image content that is never stated in text. In report QA, the context contains the radiologist's interpretation and permits textual evidence spans. The present workflow occupies an intermediate position: the image helps identify a report, but the answer is generated from that report. This decomposition makes evidence inspection easier but cannot evaluate findings visible in the image and absent from the report.

Question provenance affects validity. Physician-authored questions reflect natural information needs, ambiguity, and varied phrasing. Template-derived questions offer controlled coverage and deterministic references but may repeat lexical or structural patterns. The V5 questions intentionally cover findings, impression, and summary roles, yet they remain narrow. Their primary purpose is to test transfer through the pipeline under controlled conditions, not to estimate unrestricted clinical QA performance.

Unanswerable questions are another important distinction. A safe system should abstain when the evidence does not contain the requested information. RadQA includes naturally unanswerable examples, whereas the frozen V5 references are derived from available report sections. V5 can still abstain because retrieval or verification fails, but it does not provide a complete evaluation of natural unanswerability. This is one reason external physician-authored validation remains future work.

## 2.7 Evidence Grounding and Medical Hallucination

Medical hallucinations may be fluent and difficult to identify from wording alone (Pal et al., 2023). Evidence checking can use lexical matching, semantic similarity, natural-language inference, or combinations of these signals. MedNLI established a clinical-domain NLI task using sentence pairs derived from clinical notes (Romanov and Shivade, 2018).

An NLI-based verifier still has a restricted scope. It may determine that a sentence follows from the selected report without determining whether the report belongs to the intended case. It may also reject paraphrases, mishandle composite claims, or react to de-identification placeholders. This thesis therefore separates:

- target-case alignment;
- report-level faithfulness;
- reference consistency;
- automated support;
- clinical validity.

Only the first four receive partial automatic or researcher-reviewed evidence. Clinical validity is not established.

Faithfulness itself can be measured at different granularities. Lexical overlap rewards shared wording but may penalize correct paraphrases. Embedding similarity captures semantic closeness but can overlook negation or attribution. NLI attempts to classify entailment and contradiction, yet performance depends on the training domain, premise length, hypothesis segmentation, and decision threshold. Composite medical sentences are particularly difficult because one clause may be supported while another is not.

Sentence-level verification addresses part of this problem by decomposing the draft answer. It can retain supported sentences and remove questionable ones, producing a visible audit trail. However, sentence boundaries do not always correspond to atomic clinical claims. A single sentence may combine normal and abnormal findings, laterality, temporal comparison, and uncertainty. Filtering the whole sentence can remove useful content; retaining it can preserve an unsupported clause.

Abstention introduces a further trade-off. Conservative thresholds may reduce unsupported content but lower coverage and remove correct answers. Permissive thresholds improve coverage while increasing the risk of unsupported statements. Without expert labels, one cannot declare a threshold clinically optimal. This thesis therefore reports abstention and revision behavior descriptively and interprets selected cases using cautious labels such as possible verifier over-rejection.

## 2.8 Alignment Controls and Benchmark Validity

Multimodal improvement can be misattributed if text already identifies the answer or if any image embedding changes score distributions. Input ablation is needed to reveal text shortcuts. Alignment controls are needed to show that performance depends on the correctly paired image rather than arbitrary visual features.

The V5 shuffled-image condition uses fixed-point-free derangements so that no case retains its own image. Comparing correct alignment with many deterministic derangements produces an empirical null distribution while holding text, candidates, and fusion policy constant. The plus-one correction avoids reporting an exact zero probability from a finite permutation sample.

Benchmark construction can create additional shortcuts. Repeated generic questions make open-corpus retrieval underidentified, while indications copied from target reports make lexical retrieval easier. These properties must be reported as characteristics of the task rather than credited to model reasoning.

An ablation is informative only when the remaining inputs and evaluation set are held constant. Comparing models on different cases, candidate pools, or prompts makes the source of a difference unclear. The V5 conditions use identical confirmation questions and candidate reports, varying only whether indication and correctly aligned image information are available. This paired design supports per-case statistical comparisons and direct inspection of rank changes.

Shuffled-image controls serve a different purpose from image ablation. Removing the image asks whether the multimodal path improves on text alone. Shuffling asks whether the improvement depends on correct pairing. An arbitrary image can still alter normalized fusion scores and rankings; therefore, a single shuffled run could be unusually favorable or unfavorable. Repeating the control across 100 deterministic derangements provides a distribution of outcomes and a more stable alignment test.

Case-level separation protects benchmark validity at the available identifier granularity. The split prevents the same processed case from appearing in both development and confirmation, but the absence of a stable patient identifier means that repeated examinations from one patient cannot be ruled out. This does not create institutional external validity or establish patient-level independence.

## 2.9 Agentic and Auditable RAG Workflows

Agentic RAG commonly refers to workflows that plan, retrieve, rerank, generate, verify, or abstain. The term should be used carefully. A deterministic policy is not learned reasoning, and an automated verifier is not an independent clinical judge.

The implemented workflow is agentic in a bounded engineering sense: it records retrieval intent, executes retrieval and optional image reranking, generates from selected evidence, audits sentences, and either retains, filters, or abstains. Its main value is traceability. Each action can be inspected separately, which allows errors to be attributed to retrieval, generation, verification, abstention, or data ambiguity.

The planner component is deliberately rule-based. It maps the known experimental condition to the permitted inputs and retrieval path. This is preferable to an unconstrained language-model planner for the confirmatory experiment because it prevents hidden prompt variation and makes every decision reproducible. A semantic fallback planner may be useful for a future interactive system, but its behavior would require a separately frozen evaluation set.

The workflow also illustrates a distinction between autonomy and auditability. More autonomous agents may choose tools, reformulate queries, retry retrieval, or negotiate between multiple models. Those capabilities can improve flexibility while making causal attribution harder. The V5 pipeline uses a limited action set and records each transition. Its research value lies less in open-ended autonomy than in making the consequences of retrieval and verification decisions inspectable.

Traceability supports both debugging and responsible communication. The dashboard can show which report was ranked first, what evidence was supplied to the generator, which sentences were filtered, and why the final answer abstained. It cannot prove that the top-ranked report belongs to an arbitrary uploaded patient's record. The interface therefore describes retrieval from an indexed corpus rather than claiming patient identification.

## 2.10 Comparative Synthesis of Design Alternatives

The reviewed literature offers several plausible system designs, but they answer different research questions. Comparing them clarifies why V5 uses sparse retrieval, frozen multimodal reranking, report-grounded generation, and explicit verification rather than one end-to-end vision-language model.

An LLM-only design is the simplest baseline. It accepts a question and generates from parametric memory. This can be useful for testing general medical knowledge, but it cannot expose a case-specific evidence path. In the present task, the model would not know which of 240 examinations the repeated question refers to. Any apparently correct response could arise from generic radiology language rather than the frozen target. LLM-only answering is therefore unsuitable as the main V5 retrieval comparison, although earlier project stages used it to demonstrate the need for grounding.

A chunk-level text RAG design retrieves small report passages. Its advantage is fine-grained matching and a short generator context. Its disadvantage is weak evidence ownership when chunks lose stable case metadata or when several chunks from different reports are combined. Chunk boundaries can split negation, temporal qualifiers, and findings from impressions. Such a design is valuable for large documents, but radiology reports in this dataset are short enough that whole-report retrieval provides a clearer evidence unit.

A report-level text RAG design preserves the full report and is a stronger baseline. It avoids cross-chunk assembly and supplies coherent evidence to the generator. However, report text alone cannot exploit the paired image and may confuse clinically similar examinations. V5 retains whole-report evidence while adding the image only during candidate reranking. This preserves interpretability: the answer can still be traced to one text report even though image information influenced which report was selected.

A case-scoped oracle design would retrieve only within a known patient or directly supply the correct report. This is appropriate in an authenticated clinical record where patient identity is established externally. It is not appropriate for testing whether paired image information improves target selection because the central retrieval problem has already been solved. V5 instead ranks all 240 candidate reports and uses the frozen identifier only for evaluation.

A fully generative vision-language model could receive the image and question and answer directly. This would test visual QA or diagnostic generation rather than report retrieval. It might mention findings absent from the report, making textual evidence verification difficult. It would also confound visual representation, medical reasoning, and language generation in one output. The V5 decomposition asks a narrower question: can a frozen image-text representation improve report selection, and does that evidence change a fixed report-grounded answer?

A learned multimodal retriever is another alternative. It could fine-tune image and text encoders, learn fusion weights, and optimize Top-1 selection. Such a model may achieve higher performance, but the 240-case cohort is too small to support a strong new training claim without additional validation. Training would also add choices about negatives, epochs, learning rate, checkpoints, and early stopping. Freezing BioViL-T and using predetermined equal fusion reduces these degrees of freedom and isolates representation utility.

Dense text retrieval could replace or complement BM25. Biomedical dense encoders can reduce vocabulary mismatch and may retrieve semantically similar reports. Yet semantic similarity alone is not case identity, and a dense retriever can rank a wrong but clinically similar report highly. BM25 was retained because it is transparent, fast, and exposes the indication shortcut clearly. Future work can add dense retrieval while preserving the same alignment and shuffled controls.

Verification can also be implemented in several ways. A purely lexical checker is transparent but brittle to paraphrase. Embedding similarity is flexible but weak on negation. NLI models explicitly represent entailment and contradiction but may be miscalibrated on radiology prose. Large-language-model judges can produce explanations but introduce prompt sensitivity, cost, and possible circularity. V5 uses a fixed hybrid semantic checker and treats it as a measured component rather than a gold standard.

The chosen architecture therefore reflects the research objective rather than a claim that it is universally best. BM25 supplies an auditable text baseline, BioViL-T supplies a frozen image-report relation, deterministic fusion enables controlled alignment tests, Qwen supplies a feasible local generator, and the checker supplies sentence-level actions for analysis. The modular design allows each component's effect and limitation to remain visible.

Across these alternatives, three principles emerge. First, the unit of retrieval should preserve evidence ownership. Second, multimodal gain should be tested against strong text and incorrect-alignment controls. Third, downstream faithfulness should not be evaluated without target-case alignment. These principles directly motivate the final research gap and methodology.

## 2.11 Research Gap

Prior work establishes RAG, medical QA, biomedical retrieval, radiology vision-language encoders, and evidence checking. A narrower gap remains at their intersection: how should paired radiology images and reports be evaluated when the system must retrieve the correct case before answering, and what does report-level faithfulness mean when case alignment can fail?

Many evaluations report only final answer accuracy or image-text retrieval. Fewer connect indication shortcuts, correct-versus-shuffled image alignment, Top-1 target-case retrieval, downstream report-grounded generation, semantic verification, and stage-specific qualitative analysis within one frozen experiment.

This thesis addresses that gap through a fresh paired-case cohort, explicit input ablations, fixed-point-free shuffled-image controls, a non-oracle downstream QA path, case-grouped statistics, preserved artifact hashes, and researcher-reviewed error attribution. The central proposition is not that multimodal RAG eliminates medical error, but that it can make image-report alignment and evidence ownership measurable.

The literature synthesis leads to four requirements for the present study. First, the evaluation must include a strong text baseline and expose the indication shortcut. Second, image contribution must be tested against both image removal and incorrect alignment. Third, retrieval and answer generation must be connected through a fixed non-oracle path so that transfer can be measured. Fourth, automated support must be interpreted alongside target alignment and qualitative traces rather than treated as a clinical truth label.

These requirements also explain the deliberately narrow design. Expanding to many models, datasets, and prompts could increase the number of reported comparisons while weakening control over causal interpretation. V5 instead freezes one cohort, one retrieval policy, one generator, one checker, and one statistical protocol. The resulting claims are smaller but more defensible: aligned images improved report ordering within this benchmark; shuffled images did not reproduce the result; the gain transferred to automatic answer overlap; and verification remained an independent source of error.

The remaining gap is not the absence of another general medical chatbot. It is the absence of a carefully controlled account of how evidence identity changes across a multimodal RAG pipeline. By treating case ownership, report support, answer consistency, and clinical validity as different layers, the thesis provides a framework that can later be applied to larger archives, natural clinical questions, and independently adjudicated outputs.
