# Retrieval-Augmented Medical Question Answering over Paired Radiology Images and Reports

## Abstract

This thesis develops and evaluates an auditable multimodal retrieval-augmented generation workflow for a new chest-radiograph case whose final report is hidden at inference. The system receives the target chest image, available clinical indication and a question. It retrieves image-report cases from a Train-only historical bank, preserves case and report-section ownership, and produces a concise answer whose provenance is assembled deterministically. The target report is never retrieved or shown to the generator; it is used only as an automated evaluation reference.

The study uses 3,851 OpenI/IU-Xray cases. Exact and near-duplicate report clustering produced 3,013 clusters before allocation to Train, Calibration, Validation and Test. The final split contained 2,510, 383, 384 and 574 cases; 568 Test cases were executable. V10 established the task and alignment controls. Its fact-aware multiview R5 retriever achieved nDCG@10 0.36007 versus 0.34905 for the strong R4 comparator, a difference of +0.01103 with 95% case-bootstrap confidence interval [0.00770, 0.01441]. Correctly aligned images also exceeded 100 deterministic fixed-point-free shuffled-image assignments (0.36007 versus shuffled mean 0.24963; plus-one Monte Carlo p=0.00990), demonstrating an alignment-specific visual contribution under the automated relevance construct.

The final retrieval extension combined BM25, MedCPT and MedSigLIP candidate sources by reciprocal rank fusion, retained the Top-200 union, and applied a frozen 17-feature LambdaMART reranker. On the 568-case Test frame, this V12 method achieved combined-qrel nDCG@10 0.61590 versus 0.55313 for the recomputed V10 R5 comparator. The paired difference was +0.06277, 95% CI [0.05460, 0.07082]. Positive differences also remained under label-only and fact-only relevance sensitivities. RRF ordering alone and full-bank LambdaMART were negative mechanism controls, indicating that both the multi-source candidate frame and learned reranking were necessary.

For generation, frozen MedGemma 1.5 was compared with a Train-fitted, Validation-selected QLoRA adapter. Development showed that blanket QLoRA replacement degraded no-history answers, so the final V16 policy used the base model for Findings and the adapted model only for retrieved-history Impression questions. Retrieved-history Token-F1 increased from 0.20570 to 0.25591, a paired difference of +0.05020, 95% CI [0.03973, 0.06108]. The routed system also exceeded no history (0.16922) and deterministic random history (0.19608), retained 100% answer-contract and provenance validity, and reduced token-ceiling events from 87.85% to 56.60%. BLEU-1, BLEU-4, ROUGE-L, METEOR, CIDEr, BERTScore and RadGraph all supported the routed comparison. CheXbert was mixed: micro-F1 was statistically inconclusive and reference-positive recall decreased slightly.

A post-run audit found that 81 cases had empty Findings references, producing 243 zero-reference rows per arm. The frozen 568-case denominator was retained rather than changed after outcome inspection. Because Findings outputs were identical between the compared routes and only Impression generation changed, these rows could not create the observed paired gain but did depress absolute scores. A post-hoc non-empty-reference sensitivity remained positive at +0.04571, 95% CI [0.03371, 0.05763].

The thesis concludes that correctly paired images, multi-source candidate generation, learned reranking and section-aware generator adaptation can improve automated same-source historical-case retrieval and report-reference consistency. It does not establish physician-rated similarity, diagnostic accuracy, clinical safety, verified patient-level independence or external generalization. Independent radiologist review and authorized MIMIC-CXR replication remain Future Work.

**Keywords:** multimodal retrieval-augmented generation; chest radiography; similar-case retrieval; medical question answering; MedSigLIP; MedCPT; LambdaMART; MedGemma; QLoRA; RadGraph; provenance

## Declaration of evidence boundary

All completed evaluations are retrospective and automated. Case-ID and duplicate-cluster disjointness were verified. Reliable subject identifiers were unavailable in the processed OpenI artifact, so patient-level independence could not be verified and is not claimed. Report-derived relevance, Token-F1, CheXbert and RadGraph are evaluation proxies rather than physician adjudication. No blinded radiologist score, external-dataset result, clinical-safety result or deployment claim is reported.

# Chapter 1: Introduction

## 1.1 Background

Large language models can generate fluent answers to medical questions, yet fluency does not establish whether an answer is supported, correctly attributed, or clinically safe. Retrieval-augmented generation (RAG) responds to part of this problem by placing retrieved evidence in the model context before generation. In principle, retrieval can improve factual coverage, expose provenance, and allow unsupported claims to be traced to a specific evidence source. In practice, RAG remains a multi-stage system. Query formulation, candidate retrieval, multimodal fusion, answer generation, semantic checking, and abstention can each fail independently. A final answer can therefore appear coherent even when the evidence is irrelevant, belongs to another case, or has been interpreted incorrectly.

Radiology makes this distinction especially important. A chest-radiograph examination links a clinical indication, one or more images, findings, and an impression. When a new patient is imaged, the formal findings and impression are not yet available to an automated support system. The available query is instead composed of the target image, pre-report clinical history or indication, and a question. A useful retrieval system should search a historical archive for clinically similar other-patient cases rather than recover a report that already belongs to the target patient. Retrieved reports may provide analogies, terminology, and patterns that help a multimodal generator answer the question, but they are not proof that the same finding is present in the target patient.

The OpenI/IU-Xray collection provides de-identified chest-radiograph examinations with linked reports and images. It is sufficiently large for a controlled, local study and permits the construction of a fixed historical bank. Modern biomedical vision-language models such as MedSigLIP can map chest images and report text into related representation spaces, while MedGemma can condition generation on both a target image and textual evidence. These components make it possible to test a more realistic research question than simple paired-report recovery: whether an unseen target image can retrieve clinically similar historical image-report pairs and whether those retrieved reports improve question answering relative to the same generator without retrieval.

This thesis follows an iterative research programme. Early V5-V7 experiments used controlled paired-case retrieval to expose patient-scope ambiguity, indication shortcuts, image-alignment effects, downstream grounding failures, and the limits of naive or adaptive score fusion. Those studies remain reproducible preliminary evidence, but their closed-set task is not treated as the final clinical scenario. V9 changed the construct by removing the target report from the candidate bank and hiding it at inference. V10 then added duplicate-cluster-disjoint confirmation over a fixed Train-only historical bank and established the alignment-controlled methodological foundation. V11 and V13-V15 provide development or mechanism evidence. The final integrated primary claims combine V12 learned retrieval with V16 section-aware generation under the frozen V16 confirmation protocol.

## 1.2 Problem Statement

Text-only medical RAG assumes that the query contains enough language to retrieve useful evidence. That assumption is weak when the question is generic, for example, "What are the main radiographic findings?" Before the report exists, the question itself carries little patient-specific information. A referral indication may provide symptoms or suspected disease, but it is incomplete and may also create a lexical shortcut. The chest image contains the primary patient-specific signal. The technical problem is therefore to combine image-image similarity, image-report compatibility, and indication-question text retrieval without allowing a weak channel to degrade a stronger one.

The problem is not solved merely by adding scores. The V6 and V7 preliminary studies showed that fixed or query-conditional fusion may not outperform the strongest individual component. Score distributions differ across retrieval channels, and one modality may be uninformative for a particular query. A learned reranker must therefore be evaluated against the strongest frozen component, not only against BM25 or an arbitrary equal-weight baseline. Its training labels must remain offline and report-derived; target labels, target report text, and answer references cannot become inference features.

A second problem concerns the definition of relevance. OpenI does not include physician judgments of pairwise clinical similarity for every query and historical case. Exact target-report retrieval is also inappropriate because the target report is intentionally absent from the bank. This research operationalizes graded relevance from hidden target-report annotations: active abnormal label overlap and RadGraph entity-relation overlap. This enables nDCG evaluation over many candidates while keeping the limitation explicit. The resulting relevance measure estimates report-derived similarity; it is not a replacement for physician adjudication.

A third problem is whether retrieval improvement transfers to the final answer. A stronger ranking does not guarantee that a generator uses the evidence correctly. Historical cases may distract the generator, dominate the target image, or encourage unsupported analogy. Conversely, an automated checker may remove useful content because natural-language inference is imperfect. Retrieval, answer-reference consistency, output validity, historical-evidence support, revision, and abstention must therefore be reported separately.

The central problem is summarized as follows:

> How can a new-patient chest image, clinical indication, and medical question be used to retrieve other-case image-report evidence and improve auditable multimodal question answering without exposing the hidden target report or overstating clinical validity?

This formulation separates four levels of evidence. First, a historical case may be similar according to report-derived labels and facts. Second, the generator's historical-support statement may be entailed by the cited report. Third, the final answer may overlap the hidden target reference. Fourth, the answer may be clinically correct for the patient. The study measures the first three with different limitations and does not claim to establish the fourth.

## 1.3 Research Aim

The research aim is to develop and rigorously evaluate an auditable multimodal RAG system that uses a new target chest radiograph, indication and question to retrieve analogous historical image-report cases and generate a concise answer without access to the target report.

## 1.4 Research Objectives

1. Construct a case-ID- and duplicate-cluster-disjoint OpenI study in which the target report is hidden and the historical bank contains Train cases only.
2. Determine whether correctly paired target images contribute alignment-specific retrieval information beyond indication and question text.
3. Improve historical-case retrieval using complementary sparse, dense biomedical and medical image-text candidate sources followed by learned reranking.
4. Test whether relevant retrieved history improves downstream answer-reference consistency over no-history and random-history controls.
5. Adapt the generator with parameter-efficient training while preserving stronger base behavior through a section-aware route.
6. Evaluate ranking, generation, structured clinical overlap, provenance, truncation, subgroup behavior, runtime and uncertainty without substituting automated proxies for clinical correctness.
7. Produce reproducible code, frozen aggregate artifacts, a local demonstration dashboard and a transparent Future Work plan for blinded review and external validation.

## 1.5 Research Questions

**RQ1.** Does the correctly aligned target chest image provide alignment-specific information for report-derived historical-case retrieval?

**RQ2.** Does multi-source Top-200 candidate generation followed by a Train-fitted LambdaMART reranker improve Test retrieval over the frozen V10 R5 comparator under combined, label-only and fact-only automated relevance constructs?

**RQ3.** Does V12 retrieved historical evidence improve downstream MedGemma answer-reference consistency over target-image generation without history and over deterministic random history?

**RQ4.** Does a Validation-selected, section-aware QLoRA route improve retrieved-history generation over the frozen base generator while preserving the answer/provenance contract and avoiding a statistically supported regression on the primary structured clinical metrics?

## 1.6 Research Contributions

The first contribution is a target-report-hidden task contract for multimodal historical-case QA. A paired image-report dataset is not treated as permission to retrieve the target patient's own report. The target image, indication and question are inference inputs; the target report is an evaluation reference; and retrieved reports are explicitly labelled other-case historical analogies.

The second contribution is a leakage-aware and alignment-controlled evaluation. Exact and near-duplicate reports are clustered before allocation. The historical bank is Train only. One hundred deterministic fixed-point-free shuffled-image assignments recompute the complete visual state and test whether image gains depend on the correct case-image pairing.

The third contribution is an empirically confirmed retrieval architecture. BM25, MedCPT and MedSigLIP create complementary candidate rankings; reciprocal rank fusion defines a bounded Top-200 set; and a compact 17-feature LambdaMART model learns question-conditioned reranking. Negative controls show that RRF order alone and full-bank use of the ranker are insufficient.

The fourth contribution is parameter-efficient, section-aware generation adaptation. The MedGemma foundation model and vision tower remain frozen while QLoRA parameters are trained on case-disjoint Train data. Rather than replacing the base model everywhere, the final route applies adaptation only where Validation showed a benefit: retrieved-history Impression questions.

The fifth contribution is a provenance and evaluation discipline that keeps mixed evidence visible. Case ownership, section identity and source hashes survive evidence selection. Token-F1, standard NLG metrics, RadGraph, CheXbert, random-history controls, token ceilings, subgroup sensitivity and bootstrap intervals are reported together. The reference-completeness protocol deviation and the small CheXbert recall regression are preserved rather than hidden.

## 1.7 Scope and Boundaries

The completed study is restricted to OpenI/IU-Xray chest radiographs and English reports. It is a retrospective technical evaluation, not a prospective trial. `Report-indexed normal`, `abnormal` and `indeterminate` are derived from source metadata and are not new diagnoses. Reliable patient identifiers are absent from the processed release; case and duplicate-cluster separation are verified, but patient-level independence is not.

Retrieval relevance is constructed from hidden report labels and RadGraph facts. Generation references are hidden Findings and Impression sections. These enable controlled paired comparisons but do not establish what a radiologist would regard as clinically similar, correct, useful or safe. The Dashboard is a research demonstration and not a PACS-integrated clinical device.

The primary integrated evidence is V16: V12 retrieval and V16 generation evaluated under the frozen V16 confirmation protocol. V10 remains the methodological foundation and alignment experiment. V11 and V13-V15 are development or mechanism studies. Human blind review and MIMIC-CXR external replication remain Future Work.

## 1.8 Conceptual Framework

```text
target chest image + indication + question
        -> BM25 / MedCPT / MedSigLIP candidate rankings
        -> reciprocal-rank-fusion Top-200
        -> learned multimodal LambdaMART reranking
        -> Top-3 other-case historical reports
        -> question-relevant fact and provenance representation
        -> section-aware base/QLoRA MedGemma route
        -> concise target answer
        -> deterministic historical case/section provenance
```

The evaluation follows the same decomposition. Retrieval metrics assess ranking under explicit report-derived qrels. Shuffled images test alignment specificity. No-history and random-history conditions test whether relevant historical evidence matters. Generation metrics assess reference consistency and structure. Contract and provenance checks assess auditability. No single automated metric is interpreted as clinical accuracy.

## 1.9 Thesis Organization

Chapter 2 reviews RAG, medical QA, medical vision-language retrieval, retrieval-based report generation, factual retrieval and confidence. Chapter 3 defines the task, data controls, V10 foundation, V12 retriever, V16 QLoRA route, controls, metrics and freeze chronology. Chapter 4 reports aligned-image, retrieval, generation, clinical-structure, truncation and sensitivity results. Chapter 5 interprets the evidence, negative findings, value and limitations. Chapter 6 answers the research questions and states the final contribution. Historical studies and reproducibility records are retained in the appendices.

# Chapter 2: Literature Review

This chapter synthesizes the literature by research theme rather than as a paper-by-paper catalogue. Earlier controlled-study references are retained where they motivate the final V10/V12/V16 design.

## 2.1 Retrieval-Augmented Generation

RAG combines a parametric language model with retrieved non-parametric evidence for knowledge-intensive generation (Lewis et al., 2020). This design can expose sources and update knowledge without retraining the complete generator. It also creates a multi-stage failure surface. Retrieval determines which evidence is available, generation determines how that evidence is expressed, and verification determines which claims are retained or flagged.

RAG evaluation should therefore separate retrieval relevance, answer relevance, and faithfulness. RAGAS formalizes several of these dimensions using automated metrics (Es et al., 2024). The present research adds target-case alignment as a separate dimension. In medical records, support from a related document is not equivalent to support from the intended examination.

The original RAG formulation retrieves passages that help a sequence-to-sequence model answer knowledge-intensive questions (Lewis et al., 2020). In an open-domain setting, several passages may legitimately contain the same fact. Medical case retrieval is structurally different. The relevant evidence is not any document that states a similar finding; it is the document associated with the intended examination. This means relevance has an ownership component. Two radiology reports may both describe cardiomegaly, but only one is evidence about the target case.

RAG changes the location of model risk rather than eliminating it. Parametric hallucination may be reduced when explicit evidence is supplied, yet retrieval can introduce irrelevant, contradictory, or cross-case context. A generator can copy unsupported detail from a retrieved document, and a verifier can certify that copied detail as entailed. The final response may consequently be locally faithful and globally wrong with respect to the target. Pipeline evaluation must therefore preserve provenance from target definition through retrieval and generation.

Another important distinction is between retrieval for knowledge access and retrieval for identity resolution. In knowledge access, the question describes a topic and several sources may be useful. In the present benchmark, repeated question templates provide almost no identity information. Indication text and image representations partly resolve which case is intended. The study is therefore evaluating a constrained form of case resolution followed by report-grounded QA, not only semantic document search.

## 2.2 Medical RAG and Question Answering

Medical RAG performance depends on the corpus, task, retriever, and generator. MedRAG/MIRAGE showed that retrieval can improve medical QA but that gains vary across datasets and configurations (Xiong et al., 2024). RAGAS likewise separates retrieval context from answer faithfulness instead of assuming ideal evidence (Es et al., 2024).

These findings support controlled component comparison. A high final answer score cannot reveal whether improvement originated from a text shortcut, correct retrieval, copied context, generation, or verification. Negative findings such as weak Top-1 improvement or reduced automated support are therefore evidence about system boundaries rather than results to hide.

Medical QA benchmarks vary considerably in what they require from a system. Some ask multiple-choice questions that can be answered from general biomedical knowledge. Others provide a passage and evaluate extractive or abstractive comprehension. Visual QA datasets ask questions about pixels, while record-based tasks combine structured and unstructured patient information. Results across these settings are not directly interchangeable because the available evidence and target definition differ.

Retrieval can help medical QA in at least three ways. It can supply knowledge absent from a small generator, constrain the answer to an authoritative source, and expose evidence for inspection. It can also fail in three corresponding ways: the corpus may be incomplete, the retriever may select misleading evidence, or the generator may ignore the evidence. A rigorous experiment should therefore define the corpus boundary, report retrieval performance independently, and test whether retrieval gains propagate to the final answer.

The present work focuses on report-grounded answers because the radiology report is the available expert interpretation of the paired images. This choice avoids pretending that the language model independently reads the radiograph. At the same time, using the image to retrieve the report creates a meaningful multimodal problem: the system must exploit visual-textual alignment without transferring image pixels directly into the answer generator.

## 2.3 Sparse, Dense, and Multimodal Retrieval

BM25 remains a strong transparent sparse-retrieval baseline based on probabilistic term matching (Robertson and Zaragoza, 2009). It is effective when a query shares terminology with a report, but it is sensitive to lexical overlap and may exploit indication shortcuts. The contrast between BM25 and image-assisted systems helps show whether the target image contributes information beyond the available clinical text.

Dense retrieval encodes queries and documents in a shared vector space. MedCPT uses large-scale PubMed search logs for biomedical retrieval (Jin et al., 2023), while CLIP established general contrastive image-text alignment (Radford et al., 2021). Domain-specific medical encoders can reduce vocabulary and representation mismatch, but semantic similarity does not guarantee evidence ownership or clinical usefulness.

Multimodal retrieval adds image-image and image-report relations. These channels have different score distributions and failure modes, so a convincing evaluation should report each component rather than compare only a fused model with BM25. V10 therefore evaluates BM25, MedSigLIP image-image, MedSigLIP image-report, a nine-feature multimodal reranker and a fact-aware multiview extension on the same Train-only bank.

The final runtime scores the complete technically eligible historical bank rather than relying on a BM25-only shortlist. Compact learned rerankers are fitted on Train roles and frozen before Test. Ranking ties are deterministic, all Test questions share the same candidate bank, and a target report is never inserted into that bank. V11 separately audits whether BM25, MedCPT and MedSigLIP reciprocal-rank fusion can improve first-stage candidate recall at bounded K.

## 2.4 Paired Radiology Images and Reports

The OpenI/IU-Xray collection was prepared for radiology distribution and retrieval research and contains chest X-ray images linked to reports (Demner-Fushman et al., 2016). The report commonly includes an indication, findings, and impression. These fields should remain associated with the same examination throughout preprocessing and evaluation.

Paired radiology data enable at least three different tasks. Image classification predicts labels from pixels. Image-report retrieval ranks matching images or reports. Visual question answering generates answers from image content. The present study evaluates paired-report retrieval followed by report-grounded QA. It does not equate retrieval of a matching report with diagnosis from an image.

Radiology reports also have an internal discourse structure. The indication states why the examination was requested, findings describe observations, and the impression summarizes the radiologist's conclusion. These sections are related but not redundant. Indication may contain clinical history not visible on the current image. Findings may include detailed normal and abnormal observations. Impression may prioritize only the most clinically relevant conclusion. A summary question may require information distributed across both findings and impression.

Preserving this structure matters for retrieval and evaluation. Flattening all text into anonymous chunks can detach a statement from its section and case. Retrieving several high-scoring chunks can also mix normal findings from one examination with abnormalities from another. Whole-report retrieval preserves local coherence, while a case record additionally retains the image association and stable identifier. The final system uses the report as the answer evidence but carries the broader case package through retrieval and trace logging.

De-identified public radiology data introduce their own limitations. Placeholder tokens may replace dates, names, or other details; reports may contain typographical inconsistencies; and images may vary in view, quality, or number. These properties are not merely preprocessing inconveniences. They can influence token overlap, NLI decisions, and qualitative interpretation. The thesis records such ambiguity as a data-level category instead of forcing every discrepancy into a model-error label.

## 2.5 Biomedical Vision-Language Representation

BioViL introduced radiology-specific image-text representation learning with localized and global alignment between chest X-rays and reports (Boecking et al., 2022). BioViL-T extended this line by exploiting temporal and multi-image structure (Bannur et al., 2023). MedSigLIP provides a newer medical image-text embedding model intended for semantic image retrieval and related representation tasks. These encoders supply domain priors but do not by themselves establish diagnostic correctness.

The V10 foundation and final V12 retriever use the pinned MedSigLIP-448 revision for image-image and image-report features. The foundation encoder remains frozen; only compact retrieval components are trained. This separates representation reuse from task-specific ranking and keeps the computation feasible on the available GPU. Earlier BioViL-T studies are retained as formative history, not described as the final encoder.

Multi-view examinations create an aggregation problem because frontal and lateral views may contain complementary information. V10 compares mean-view and learned-attention representations in a frozen-checkpoint mechanism audit. The final R5 ensemble uses the prespecified multiview component, but the attention-only contrast is interpreted cautiously because its interval crosses zero.

An embedding is not an explanation or a calibrated clinical probability. A high image-report score does not identify the responsible anatomy, polarity or uncertainty. R5 therefore combines image features with question-conditioned report facts and preserves the retrieved case and fact provenance. The representation is evaluated as a ranking signal within a controlled RAG workflow, not as an autonomous image diagnosis model.

## 2.6 Medical Visual and Report Question Answering

VQA-RAD contains clinically generated questions and answers about radiology images (Lau et al., 2018), while EHRXQA combines electronic health records and chest X-rays for multimodal QA (Bae et al., 2023). RadQA contains physician-authored report questions, answer spans and naturally unanswerable cases (Soni et al., 2022). These resources show the value of clinician phrasing and explicit answerability, but they differ from the same-source new-case retrieval task used here.

The completed benchmark uses three fixed question roles for retrieval and two report-derived roles for V10 generation. This controlled design supports paired system comparisons but is linguistically narrow. The V11 reserved wording set tests planner robustness only; its labels are author-defined and cannot substitute for clinician-authored natural questions.

The final workflow also differs from pure report QA. MedGemma receives the target chest image and may receive explicitly labelled other-case historical reports. The hidden target report is used only as an automated evaluation reference. Retrieved reports are analogies, not answer spans about the target patient. This separation makes it possible to evaluate visual alignment, historical evidence ownership and answer-reference consistency independently.

Natural unanswerability remains incompletely tested. The system can withhold unreliable historical support, but the fixed questions are derived from available report roles. RadQA or a new clinician-authored set would provide a stronger answerability evaluation once authorized access and an external protocol are available.

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

Multimodal improvement can be misattributed when clinical text already identifies the answer or when any image changes score distributions. Image ablation asks whether visual information adds value; wrong-image controls ask whether the value depends on the correct image-case pairing. Both are stronger when systems share the same cases, candidate bank, model state and metrics.

V10 uses 100 deterministic unique fixed-point-free image assignments. Each Test case receives the complete view set of another Test case while its indication, question and evaluation reference remain unchanged. Visual similarities, normalized features, multiview state and R5 scores are recomputed. The plus-one Monte Carlo p-value avoids reporting zero from a finite control set.

Benchmark construction can introduce additional shortcuts. Generic repeated questions make text retrieval weakly identified, indications can encode disease hints, and near-duplicate reports can leak across partitions. V10 clusters exact and near-duplicate reports before allocation, uses a common Train-only bank and reports image-only, image-report and BM25 components separately. The source design supports one study per patient, but this property cannot be independently re-verified from released subject identifiers.

The relevance construct is another validity boundary. Report labels and RadGraph facts enable deterministic graded qrels, but they are not physician judgments. Post-hoc sensitivity across combined, label-only and fact-only definitions is therefore reported as a construct audit rather than a replacement endpoint.

## 2.9 Agentic and Auditable RAG Workflows

Agentic RAG can plan, retrieve, rerank, generate, verify or abstain. The term should be used carefully: a deterministic workflow is not learned clinical reasoning, and an automated checker is not an independent physician. Research value depends on bounded actions, inspectable state and honest failure handling rather than on the number of named agents.

The final system is agent-like only in this bounded engineering sense. A deterministic planner identifies question intent, the retriever ranks historical cases, the evidence selector preserves case ownership, MedGemma generates a concise target-image answer, deterministic code attaches provenance, and a calibrated gate can withhold historical support. Each stage has an explicit contract and can be evaluated separately.

Direct Python modules are used instead of LangChain because the experiment requires stable prompts, frozen transitions and artifact-level reproducibility. This choice does not imply that orchestration libraries are inferior; it avoids introducing hidden retries, memory or tool-selection behavior into a controlled study. The Dashboard exposes the retrieved case IDs, evidence units, confidence boundary and provenance without claiming autonomous clinical agency.

Open-ended agents may be useful in future interactive systems, but they require separate evaluation of tool choice, prompt reformulation, retry policy and safety. The current contribution is an auditable multimodal RAG workflow, not a general autonomous medical assistant.

## 2.10 Closest Retrieval-Based Radiology Systems

CXR-RePaiR used CLIP-style image-to-report retrieval to generate chest X-ray reports from historical text (Endo et al., 2021). It establishes retrieval itself as a viable alternative to unconstrained generation. X-REM added multimodal image-text matching and expert error assessment, showing that coarse cosine similarity can miss fine-grained compatibility (Jeong et al., 2023). Both systems motivate retrieval-based reuse, but their primary output is a report rather than a question-conditioned answer with explicit other-case provenance.

FactMM-RAG used RadGraph-derived factual report pairs to train a multimodal retriever and augment radiology report generation (Sun et al., 2025). It is the closest precedent for factual retrieval supervision. The present study reuses the principle of fact-aware ranking but preserves whole case ownership before fact selection and evaluates a target-image, indication and question contract rather than copying restricted data or reproducing a full-report LLaVA system.

MedProbCLIP emphasizes probabilistic radiograph-report embeddings, calibration, risk-coverage behavior, multiview representation and selective retrieval (Elallaf et al., 2026). It motivates confidence and abstention analysis, but the present system does not reproduce its probabilistic training objective. Its confidence values remain report-derived technical signals.

## 2.11 Final Research Gap

Prior work establishes medical RAG, chest image-report retrieval, factual retrieval supervision and report generation, but several elements are rarely evaluated together. A paired dataset may allow accidental target-report lookup. Near-duplicate reports may cross splits. A nominally multimodal model may rely on text shortcuts. Retrieved reports may be copied as if they describe the current patient. Better retrieval may fail to improve final answers. Generator adaptation may help one report section while damaging another. Automated metrics may be overstated as clinical accuracy.

The final gap is therefore not simply a newer model. It is an integrated evidence chain for target-report-hidden, question-conditioned historical-case RAG: duplicate-aware allocation, Train-only evidence, aligned-versus-shuffled image control, multi-source candidate generation, learned reranking, retrieved-versus-random history control, case-preserving fact provenance, section-aware parameter-efficient adaptation, and transparent reporting of mixed metric behavior.

# Chapter 3: Methodology

## 3.1 Research Design and Freeze Chronology

The project used a staged empirical design. V10 established the final task contract, duplicate-cluster split, aligned-image control, strong R4/R5 retrieval comparison and downstream historical-RAG effect. V11 and V13-V15 explored candidate recall, fact selection, concept prediction and retrieval-to-generation transfer on non-Test data. V12 selected a multi-source candidate and learned-ranker method on Validation. V16 trained and selected generator adaptation without Test access, then committed a confirmation protocol before V12/V16 Test outputs were generated.

The same V10 Test partition had already been used for frozen V10 systems. It is therefore not described as a globally untouched project holdout. However, no V12 LambdaMART ranking, V16 adapter output or V16 route output was evaluated on Test before the V16 protocol. The final experiment is a held-out method confirmation rather than formal preregistration.

## 3.2 Operational Task Contract

At inference, the system receives one target chest radiograph, an optional clinical indication and a question. The target Findings and Impression are hidden. Retrieval searches historical Train cases only. Retrieved reports are other-case analogies, never the target patient's report. The answer concerns the target image; historical support is attached separately with case and section provenance.

This contract distinguishes direct, reference and analogical information. The target image is direct model input. The hidden report is a reference used only by evaluation code. Retrieved reports are analogies whose ownership must remain visible. Local faithfulness to a retrieved report cannot prove that the report applies to the target case.

## 3.3 Data Source and Case Representation

The processed OpenI/IU-Xray artifact contains 3,851 stable case records and linked radiograph views. A case may include indication, Findings, Impression, source problem labels, image paths and derived RadGraph records. The local image archive contains the paired PNG views; large source-derived text, image pixels, embeddings and generations remain local under repository policy.

Operational spectrum labels are derived from the normalized `problems` field. `normal` is report-indexed normal; non-empty clinical labels excluding `normal` and `no indexing` are report-indexed abnormal; `no indexing` is indeterminate. These strata describe source indexing, not independent clinical adjudication.

## 3.4 Duplicate-Clustering and Partitioning

Exact and near-duplicate report representations were clustered before allocation. The 3,851 cases formed 3,013 clusters. Entire clusters were assigned deterministically to Train, Calibration, Validation and Test, producing 2,510, 383, 384 and 574 cases. No cluster crossed partitions.

The executable historical bank contained 2,506 Train cases with required image and feature artifacts. The executable Test frame contained 568 cases. Case-ID and duplicate-cluster disjointness were verified. Reliable subject identifiers were unavailable, so patient-level independence could not be verified.

## 3.5 Automated Relevance Constructs

The target identity is not a relevant historical item because the target case is intentionally excluded. Graded relevance is derived from shared active report labels and RadGraph fact similarity. The primary combined construct uses both channels; label-only and fact-only variants are prespecified sensitivities. nDCG@10 is the primary ranking metric because it retains graded relevance. MRR and Hit@k are descriptive.

These qrels are automated proxies. They measure similarity to hidden source-report descriptions, not physician-rated case usefulness. RadGraph features also appear in the learned ranker, so feature-metric coupling is explicitly acknowledged and the label-only sensitivity is necessary.

## 3.6 V10 Foundation and Alignment Control

V10 compared BM25, MedSigLIP image-image, MedSigLIP image-report, R4 nine-feature reranking and R5 fact-aware multiview reranking over a common Train-only bank. R5 used five frozen seeds. Its primary comparison was R5 minus R4.

The alignment control created 100 deterministic unique fixed-point-free assignments of wrong Test images. For every assignment, image embeddings, multiview state, similarities, normalization and final scores were recomputed while indication, question, historical bank and reference remained unchanged. A plus-one Monte Carlo p-value compared the aligned score with the shuffled distribution.

## 3.7 V12 Multi-Source Candidate Generation

V12 creates three independent ranked lists: BM25 over clinical text, MedCPT biomedical dense text retrieval and MedSigLIP visual retrieval. Reciprocal rank fusion with fixed constant 60 combines ranks without requiring incomparable raw score calibration. The deterministic union is truncated at Top-200.

Candidate generation and reranking are evaluated separately. A relevant case outside Top-200 is a true candidate failure and is not deleted. The Top-200 budget was selected on Validation before Test. RRF ordering alone, R5 reranking inside Top-200 and full-bank LambdaMART are retained as mechanism controls.

The candidate design addresses a limitation of a single-source first stage. BM25 is precise when the question or indication shares explicit terminology with a historical report, but it can miss paraphrases and may overuse indication wording. MedCPT contributes biomedical semantic retrieval even when surface terms differ. MedSigLIP can surface visually similar cases despite weak text overlap. Their union increases the chance that at least one report-derived relevant case is available to the learned ranker. RRF uses only ranks, making it robust to the different scales and calibration properties of sparse scores, dense cosine similarity and visual similarity.

Top-200 is a computational and statistical boundary rather than a claim that all useful cases are present. Validation diagnostics showed that the union improved relevant-case presence but still missed many report-derived positives. The final design therefore reports candidate failure separately from reranking quality. This prevents a strong reranker from receiving credit for queries whose relevant evidence never entered its search space.

## 3.8 V12 LambdaMART Reranking

LightGBM 4.7.0 LambdaMART was trained on V10 Train role groups using the existing 17-dimensional multimodal feature pipeline. Features include normalized sparse and dense retrieval signals, image-image and image-report compatibility, source ranks, question-conditioned fact compatibility and availability indicators. Foundation encoders and V10 models remain frozen.

The primary V12 model optimizes the combined report-derived relevance construct. Proxy-specific label-only and fact-only models were development sensitivities and were not substituted after Test. The final method reranks only the frozen Top-200 frame; applying it to the full bank is an explicit negative control.

LambdaMART was selected because the data are naturally grouped by query and the outcome is an ordered list rather than an independent binary classification. Trees can model nonlinear interactions such as a visually strong candidate becoming more credible when its report facts agree with the question, or a text match being discounted when it is driven only by indication wording. Query grouping prevents candidates from different questions from being treated as interchangeable training examples.

The 17-feature vector was constructed entirely from information available at retrieval time. It included source scores and ranks, V10 multimodal features, report-fact compatibility and explicit missingness or availability indicators. Hidden target-report relevance supplied training labels on Train but was not a runtime feature. Feature order, LightGBM version, random seed and checkpoint hash were frozen. Deterministic tie handling ensured that repeated runs produce the same ranking when predicted scores are equal.

## 3.9 Historical Evidence and Provenance

The final retriever returns Top-3 historical cases. Case retrieval precedes fact selection. Every evidence unit retains `case_id`, report section, unit type and source hash. The generator prompt labels material as historical analogy and does not present retrieved statements as direct target-patient facts.

Deterministic support assembly attaches only identifiers and evidence units that occurred in the retrieved set. The language model does not invent citation IDs. Contract validity checks syntax and required fields; provenance validity checks that cited sources exist. Neither proves semantic or clinical correctness.

## 3.10 V16 QLoRA Adaptation

V16 adapts `google/medgemma-1.5-4b-it` using 4-bit NF4 QLoRA while keeping the foundation weights and vision tower frozen. The selected training uses rank 8, alpha 16, dropout 0.05, greedy decoding and the pinned model revision. Targets are source Findings or Impression sections. Training examples include no history, V12 retrieved history and deterministic random history so the adapter is exposed to relevant and irrelevant context.

Development used Train for fitting, Calibration for technical and checkpoint selection, and Validation for the frozen development comparison. Test was prohibited. Full QLoRA improved retrieved-history answers but degraded no-history output. A section-aware route was therefore fixed before confirmation: base MedGemma for Findings and all controls; QLoRA only for retrieved-history Impression.

Training supervised only answer tokens; prompt and image tokens were masked from the loss. Gradient checks verified that LoRA parameters received finite gradients while frozen foundation parameters did not. Checkpoints were saved and reloaded before evaluation. Four-bit quantization reduced memory use, while bfloat16 computation, gradient checkpointing and accumulated micro-batches allowed adaptation on the available 8 GB GPU.

The training objective is still bounded by source reports. QLoRA learns the style, concepts and evidence-use pattern of OpenI sections; it is not trained from physician preferences or prospective outcomes. Random-history examples are important because otherwise the adapter could learn to copy any historical report. They teach an invariance target: irrelevant history should not override the target image and question.

## 3.11 Generation Conditions and Negative Control

Every Test case contributes Findings and Impression questions under no-history, random-history and retrieved-history conditions. The base and QLoRA arms each contain 3,408 rows. All use one target image, the same indication and question, at most 96 new tokens, no more than two complete sentences and greedy decoding.

Random history is selected from eligible Train cases by domain-separated SHA-256 ordering, excluding the target duplicate cluster. It tests whether relevant retrieval is better than merely adding report text. The primary comparison is impression-gated route minus base under retrieved history. Within-route retrieved-minus-no-history and retrieved-minus-random comparisons test historical-evidence utility.

## 3.12 Metrics and Statistical Analysis

Retrieval uses nDCG@10 as primary, with MRR and Hit@k descriptive. Generation uses case-averaged Token-F1 as the protocol primary. Secondary metrics are BLEU-1, BLEU-4, ROUGE-L, METEOR, CIDEr, baseline-rescaled BERTScore, CheXbert label consistency, RadGraph entity/entity-relation/complete F1, contract validity, provenance validity, token-ceiling rate, token counts, latency and peak GPU memory.

All principal system differences use 10,000 case-grouped bootstrap resamples. Findings and Impression rows from one case remain together. The shuffled-image control uses 100 assignments and a plus-one p-value. Confidence intervals that cross zero are reported as inconclusive rather than positive.

nDCG@10 discounts relevant cases at lower ranks and normalizes against the best ranking available under each query's qrels. It is appropriate when several historical cases can have different relevance grades. MRR emphasizes the first case above the operational relevance threshold, while Hit@k asks only whether at least one such case occurs within a cutoff. These measures answer different retrieval questions and are not interchangeable with diagnostic accuracy.

Token-F1 measures lexical overlap after normalization. BLEU emphasizes n-gram precision, ROUGE-L sequence overlap, METEOR token alignment, CIDEr consensus-weighted n-grams and BERTScore contextual embedding similarity. RadGraph scores overlap of radiology entities and relations. CheXbert compares automated observation labels. Their agreement strengthens a conclusion, while disagreement reveals which answer properties changed. No post-Test metric was promoted to primary because it produced a more favorable result.

The unit of resampling is the case, not the generation row. This preserves within-case dependence among question types and evidence conditions. Reported intervals quantify sampling uncertainty under the observed automated benchmark; they do not capture annotation uncertainty, dataset shift or clinical disagreement.

## 3.13 Reference-Completeness Deviation

The protocol stated that both report-section references would be non-empty. The manifest implementation verified question-type presence but did not assert non-empty strings. The post-run audit found 81 cases with empty Findings references, producing 243 empty-reference rows across three history conditions. Impression references were complete.

The frozen primary denominator remains 568 cases. Empty Findings rows score zero identically in both primary arms, and the route changes only Impression generation, so they cannot create the paired gain. A post-hoc sensitivity excludes only empty-reference rows while retaining every case with at least one evaluable section. This deviation is documented rather than repaired by retrospective case removal.

## 3.14 Reproducibility, Hardware and Failure Policy

Experiments ran locally on an NVIDIA GeForce RTX 5070 Laptop GPU with approximately 8 GB memory. Model revisions, adapters, split files, result summaries and scripts are fingerprinted with SHA-256. Large generations, model caches, report text and pixels remain local. Public aggregate files contain no report text or image pixels.

Technical interruptions could be resumed under unchanged frozen settings. Outcome-driven reruns, case replacement, Test-driven model selection, prompt revision, qrel substitution and selective result deletion were prohibited. The final repository test suite contains 337 passing tests after the Final-QA Validation extension and completeness audit.

# Chapter 4: Results

## 4.1 Cohort and Executable Matrix

The final Test frame contains 568 executable cases and the historical bank contains 2,506 Train cases. Test includes 195 report-indexed normal, 359 abnormal and 14 indeterminate cases. The generation matrix contains 3,408 rows per arm: 568 cases, two question types and three evidence conditions.

## 4.2 V10 Alignment Foundation

V10 R5 achieved nDCG@10 0.36007 versus 0.34905 for R4, a difference of +0.01103, 95% CI [0.00770, 0.01441]. Correctly aligned R5 exceeded the mean of 100 wrong-image assignments, 0.36007 versus 0.24963; no shuffled run reached aligned performance and the plus-one p-value was 0.00990. These results establish that the visual contribution depends on correct image-case alignment under the report-derived construct.

## 4.3 Final Retrieval Confirmation

| Retrieval condition | Combined nDCG@10 | Label-only | Fact-only |
|---|---:|---:|---:|
| V10 R5, full Train bank | 0.55313 | 0.33326 | 0.33180 |
| RRF Top-200 order | 0.54292 | 0.28605 | 0.28887 |
| RRF Top-200 then R5 | 0.55363 | 0.33489 | 0.33239 |
| **RRF Top-200 then V12 LambdaMART** | **0.61590** | **0.37254** | **0.34507** |
| Full-bank LambdaMART | 0.54150 | 0.31632 | 0.29259 |

V12 improved combined nDCG@10 over R5 by +0.06277, 95% CI [0.05460, 0.07082]. Label-only improved by +0.03928 [0.02450, 0.05443], and fact-only improved by +0.01326 [0.00405, 0.02243]. All intervals exclude zero.

RRF alone was worse than R5, and full-bank LambdaMART was also worse. R5 reranking within Top-200 only recovered the R5 level. The result therefore supports the complete two-stage method rather than candidate fusion or the learned model in isolation.

## 4.4 Retrieval Spectrum Sensitivity

V12 combined-qrel nDCG@10 was 0.64897 for report-indexed normal, 0.59784 for abnormal and 0.61850 for indeterminate cases. Corresponding R5 values were 0.57158, 0.54268 and 0.56391. The direction is positive across strata, but the 14-case indeterminate result is descriptive. Differences between strata partly reflect source labels and qrel structure, not inherent clinical difficulty.

## 4.5 Primary Generation Confirmation

| Retrieved-history generator | Token-F1 | Difference | 95% CI |
|---|---:|---:|---:|
| Base MedGemma | 0.20570 | - | - |
| **V16 impression-gated route** | **0.25591** | **+0.05020** | **[0.03973, 0.06108]** |

Findings Token-F1 remains 0.27661 because both routes use the same base output. Impression Token-F1 increases from 0.13480 to 0.23520. Full QLoRA also improves retrieved-history Token-F1 but reduces no-history Token-F1 by 0.01616 with a negative interval, supporting the section-aware route rather than blanket replacement.

## 4.6 No-History and Random-History Controls

The final route achieved 0.16922 Token-F1 with no history, 0.19608 with random history and 0.25591 with V12 retrieved history. Retrieved minus no history was +0.08668 with a positive confidence interval. Retrieved minus random history was +0.05982, 95% CI [0.04685, 0.07289]. Relevant retrieval therefore contributes more than extra context alone.

## 4.7 Standard Language Metrics

| Retrieved-history metric | Base | V16 route | Difference | 95% CI |
|---|---:|---:|---:|---:|
| BLEU-1 | 0.09123 | 0.14406 | +0.05284 | [0.04262, 0.06330] |
| BLEU-4 | 0.00653 | 0.02016 | +0.01363 | [0.00773, 0.02041] |
| ROUGE-L | 0.10773 | 0.16788 | +0.06016 | [0.04975, 0.07064] |
| METEOR | 0.13560 | 0.17049 | +0.03489 | [0.02522, 0.04492] |
| CIDEr | 0.07902 | 0.31849 | +0.23947 | [0.17606, 0.30897] |
| BERTScore F1, rescaled | -0.13898 | -0.08802 | +0.05096 | [0.04179, 0.06020] |

All paired intervals favor the route. Negative absolute BERTScore values are possible after baseline rescaling and are also affected by retained empty references; the paired difference is the relevant comparison.

## 4.8 Structured Clinical Metrics

RadGraph entity F1 improves by +0.02900 [0.01983, 0.03840], entity-relation F1 by +0.02843 [0.01975, 0.03767], and complete F1 by +0.02687 [0.01833, 0.03562]. CheXbert does not improve uniformly. Micro-F1-14 changes by -0.00545 with CI [-0.01389, 0.00276], while reference-positive recall decreases by -0.01081 with CI [-0.02021, -0.00170]. Exact-five accuracy changes by +0.00176 with an interval crossing zero.

The result is therefore positive for lexical, semantic and RadGraph overlap but mixed for disease-label consistency. The supported CheXbert recall decline is retained as a secondary negative result.

## 4.9 Output Contract and Truncation

Answer-contract and provenance validity are 100% for base and routed outputs. Under retrieved history, the token-ceiling rate falls from 0.87852 to 0.56602, a difference of -0.31250 with CI [-0.33363, -0.29137]. Mean output tokens fall from 91.61 to 75.75. This demonstrates a substantial formatting and completion benefit, although 56.6% remains high and requires cautious interpretation.

## 4.10 Reference-Completeness Sensitivity

The audit found 243 empty Findings-reference rows from 81 cases in each arm. All Impression references were non-empty. The all-row primary result is unchanged. After excluding only empty-reference rows, retrieved-history row-mean Token-F1 is 0.22150 for base and 0.27555 for the route. The case-grouped paired difference remains positive at +0.04571, 95% CI [0.03371, 0.05763].

## 4.11 Generation Spectrum Sensitivity

The routed-minus-base Token-F1 difference is +0.03186 [0.02087, 0.04324] for 359 abnormal cases and +0.08211 [0.06052, 0.10461] for 195 normal cases. The 14 indeterminate cases have a positive point estimate of +0.07603 but an interval crossing zero. The direction is not confined to one major stratum, but subgroup magnitudes are not clinical prevalence effects.

## 4.12 Results Summary

The evidence chain is coherent and appropriately mixed. Correct image alignment matters. The two-stage V12 retriever improves a strong R5 comparator under three automated relevance definitions. Retrieved history outperforms no and random history. Section-aware QLoRA improves the primary generation metric, all standard language metrics and RadGraph while reducing truncation. CheXbert does not show uniform improvement, and empty Findings references limit absolute-score interpretation. These limitations narrow rather than erase the positive automated conclusion.

# Chapter 5: Discussion

## 5.1 What the Final System Actually Does

The system is not searching for the target patient's existing report. It operates before that report is available to the model. The target image, indication and question define a new query; other-case historical image-report pairs supply analogical evidence. The hidden target report is used only after generation to evaluate consistency.

This distinction gives the thesis practical and methodological value. In a paired dataset, simply looking up the paired report would be trivial. Retrieving other cases asks a harder question: can historical analogies help interpret a new examination without confusing evidence ownership?

## 5.2 Alignment-Specific Visual Value

The shuffled-image experiment is the strongest evidence that images are functional rather than decorative. Text, model, bank and references were held constant while only image-case pairing was broken. The large aligned-versus-shuffled gap demonstrates case-specific visual signal under the report-derived task.

This remains narrower than pixel-level diagnosis. MedSigLIP helps retrieve cases whose reports resemble the hidden target report. The experiment does not independently adjudicate whether a finding is clinically present.

## 5.3 Why the V12 Retriever Improves

BM25, MedCPT and MedSigLIP contribute complementary failure modes. RRF broadens the candidate set without calibrating heterogeneous raw scores. LambdaMART then learns how text, image, rank and fact signals should interact within a bounded candidate frame. The positive label-only and fact-only sensitivities suggest that the gain is not confined to the exact combined qrel.

The negative controls are equally informative. RRF order alone is worse, and full-bank LambdaMART is worse. The learned ranker is therefore not a universal scoring function; it depends on the candidate distribution on which it was trained. This is a more useful engineering conclusion than saying only that a newer model achieved a higher number.

The magnitude of the Test improvement is much larger than the earlier R5-minus-R4 gain because the two comparisons answer different questions. R5 added fact and multiview features to an already strong full-bank reranker. V12 changes both candidate generation and the learned ranking objective. The relevant comparison remains paired and same-frame; it should not be interpreted as a cross-paper leaderboard score because external systems use different splits, outputs and relevance definitions.

The construct sensitivities also matter. A gain under combined qrels could be driven by the same fact representation used by the model. Positive label-only results reduce, but do not remove, this concern because labels and facts still originate from the same reports. Independent physician similarity labels would be needed to establish that the ranking improvement corresponds to clinical retrieval quality.

## 5.4 Historical Evidence Improves QA

Retrieved history exceeds both no history and random history. This supports the central RAG claim: relevant historical context adds value beyond the target image and beyond additional text volume. It also connects retrieval to the final answer rather than treating ranking as an isolated benchmark.

Absolute overlap scores remain modest. Radiology reports admit multiple valid phrasings, and 81 cases lack Findings references, but neither fact makes an automated score equivalent to clinical accuracy. The positive claim is paired improvement under the same generator, cases and references.

## 5.5 Why Section-Aware Adaptation Matters

Full QLoRA improved retrieved-history output but harmed no-history generation. This is a common model-adaptation risk: fitting to one context distribution can weaken broad pretrained behavior. The section-aware route preserves the base model where it is stronger and applies adaptation where Validation showed a clear benefit.

The route is simple, deterministic and auditable. It does not require an opaque agent to choose among tools. The result is a genuine trained contribution because LoRA parameters were optimized with backpropagation while foundation parameters remained frozen. It is not foundation-model fine-tuning in the sense of updating all MedGemma weights.

The section interaction is plausible. Findings often require a broad descriptive inventory of visible observations, for which the pretrained multimodal model retained useful coverage. Impression is shorter and prioritizes clinically salient synthesis, making it more amenable to task-specific adaptation. This explanation is consistent with the observed metric pattern but remains inferential; the experiment establishes route performance, not a causal theory of model internals.

Routing also limits catastrophic interference. A global replacement would trade one improvement for a supported no-history loss. The final policy treats the base and adapter as complementary experts under a fixed observable question type. This is narrower than a free-form agent, but its behavior is easier to reproduce and defend.

## 5.6 Mixed Clinical-Structure Evidence

RadGraph gains show improved overlap of radiology entities and relations. CheXbert micro-F1 is statistically inconclusive, and positive-label recall declines slightly. A plausible interpretation is that shorter, more focused Impression answers improve textual and relational matching while omitting some secondary labels. This remains a hypothesis because no radiologist adjudicated the outputs.

The mixed result prevents an overbroad claim. V16 improves report-reference consistency on most metrics, but it does not uniformly improve every disease-label dimension. A clinically useful system may need explicit recall constraints or selective generation trained against physician priorities.

## 5.7 Truncation and Provenance

The routed system substantially reduces token-ceiling events and maintains complete contract/provenance validity. Deterministic provenance is especially important because model-generated citation IDs can be fluent but fictitious. Here, source identifiers are attached by code from the actual evidence list.

However, 56.6% of retrieved-history outputs still reach the ceiling. Contract validity therefore cannot be treated as completeness. Further work should use a shorter answer representation, constrained decoding or section-specific budgets fixed before external confirmation.

## 5.8 Protocol Deviation and Research Integrity

The empty Findings-reference issue was discovered during final freeze auditing. Deleting 81 cases after seeing results would have produced a cleaner-looking absolute score but a weaker study. The primary denominator was retained, the implementation mismatch was documented, and a separate sensitivity was computed.

The sensitivity remains positive, and the changed route affects only non-empty Impression rows. The deviation therefore limits absolute interpretation without explaining away the paired effect. This episode demonstrates why executable assertions must match protocol language.

## 5.9 Research Value and Practical Significance

The thesis contributes more than a dashboard. It defines a defensible medical RAG task, prevents target-report leakage, controls duplicate reports, proves image alignment matters, improves retrieval with a trained ranker, trains a parameter-efficient generator adapter, uses two negative controls, preserves evidence ownership and reports both positive and negative metrics.

The local implementation also shows feasibility on an 8 GB GPU. Direct Python modules rather than LangChain keep retrieval, ranking, routing, generation and evaluation states inspectable. The resulting dashboard can demonstrate the upload-to-retrieval-to-answer workflow while clearly labelling historical evidence.

## 5.10 Limitations

First, all results use one OpenI/IU-Xray source. Duplicate-cluster separation improves internal validity but not external generalization. Patient-level independence cannot be verified from processed identifiers.

Second, relevance and answer references are report-derived. The learned ranker uses some RadGraph-related features that overlap conceptually with qrels. Physician similarity and clinical usefulness were not assessed.

Third, 81 cases have empty Findings references despite stricter protocol language. The primary result retains these rows and the sensitivity remains positive, but absolute scores are depressed and the deviation weakens procedural perfection.

Fourth, CheXbert evidence is mixed and reference-positive recall decreases slightly. The result is not uniform clinical-label improvement.

Fifth, token ceilings remain frequent. Provenance validity guarantees source identity, not answer completeness or truth.

Sixth, questions and planner roles are controlled and largely researcher-defined. Natural clinician questions may differ.

Seventh, the Test partition was reused from V10 for new methods. V12/V16 were not tuned on it before protocol freeze, but it is not a globally untouched project holdout.

Eighth, no independent blinded radiologist review, prospective workflow test, fairness audit or clinical-safety evaluation was conducted.

## 5.11 Future Work

The first priority is independent blinded radiologist review of a prespecified 80-120 case sample. Reviewers should assess historical-case similarity, target-image answer consistency, evidence usefulness and potentially harmful content. This remains Future Work because no suitable reviewer completed the protocol.

The second priority is authorized external validation on MIMIC-CXR-JPG or another dataset with reliable subject/study identifiers. The full MIMIC source is multi-terabyte and access controlled; a smaller prespecified local subset can be used after credentialing and a frozen external protocol.

The third priority is to repair reference eligibility before cohort instantiation. Future builders must fail fast when required sections are empty and must publish completeness counts before model execution.

The fourth priority is clinician-authored natural questions and calibrated selective prediction. Risk-coverage analysis should use physician correctness or usefulness outcomes rather than report-derived proxy labels.

The fifth priority is reducing residual truncation and testing whether explicit clinical-recall constraints can preserve the QLoRA gains without the observed CheXbert recall loss.

# Chapter 6: Conclusion

## 6.1 Answers to the Research Questions

**RQ1: Does the correctly aligned image provide alignment-specific retrieval information?** Yes under the automated same-source relevance construct. V10 R5 achieved 0.36007 nDCG@10 with aligned images versus a shuffled mean of 0.24963, and no one of 100 wrong-image assignments reached the aligned score.

**RQ2: Does V12 improve historical-case retrieval over R5?** Yes. V12 RRF Top-200 plus LambdaMART achieved 0.61590 combined-qrel nDCG@10 versus 0.55313 for R5, a difference of +0.06277 [0.05460, 0.07082]. Label-only and fact-only intervals were also positive. Candidate RRF alone and full-bank LambdaMART were negative, so the complete two-stage design is required.

**RQ3: Does retrieved history improve downstream QA?** Yes for automated report-reference consistency. The final route achieved Token-F1 0.25591 with retrieved history, compared with 0.16922 without history and 0.19608 with random history. Retrieved evidence therefore adds value beyond extra context.

**RQ4: Does section-aware QLoRA improve generation while preserving safeguards?** Yes for the protocol primary and most secondary metrics. The route improves Token-F1 over base by +0.05020 [0.03973, 0.06108], retains 100% contract and provenance validity, reduces truncation and improves RadGraph and standard NLG metrics. CheXbert is mixed, including a small supported recall decrease, so uniform clinical-label superiority is not claimed.

## 6.2 Final Contribution

The completed work is an auditable multimodal historical-case RAG study and a functional question-answering prototype. Its central contribution is the evidence chain: target-report hiding, duplicate-aware splitting, Train-only history, aligned-image control, complementary candidate generation, learned reranking, retrieved-versus-random history, case-preserving provenance, section-aware parameter-efficient adaptation and transparent mixed-metric evaluation.

The study also clarifies the value of a trained component. MedSigLIP, MedCPT and MedGemma remain frozen foundation models. Trainable LambdaMART and QLoRA components adapt ranking and generation to the task. Their gains are evaluated separately so that model modernization does not obscure mechanism.

## 6.3 Final Boundary

The thesis does not establish diagnosis, patient benefit, clinical safety, verified patient-level separation, physician agreement or external generalization. It establishes a reproducible automated result on one public paired chest-radiograph source. Within that boundary, the project meets its graduate-research objective: a meaningful task, trained technical contributions, controlled experiments, negative controls, robust paired statistics, an auditable system and conclusions that do not exceed the evidence.

# References

Bae, S., Kyung, D., Ryu, J., et al. (2023). EHRXQA: A multi-modal question answering dataset for electronic health records with chest X-ray images. *Advances in Neural Information Processing Systems*.

Bannur, S., Hyland, S., Liu, Q., et al. (2023). Learning to exploit temporal structure for biomedical vision-language processing. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*.

Boecking, B., Usuyama, N., Bannur, S., et al. (2022). Making the most of text semantics to improve biomedical vision-language processing. *European Conference on Computer Vision*.

Demner-Fushman, D., Kohli, M. D., Rosenman, M. B., et al. (2016). Preparing a collection of radiology examinations for distribution and retrieval. *Journal of the American Medical Informatics Association, 23*(2), 304-310.

Elallaf, A., Zhang, Y., Masupalli, Y., et al. (2026). MedProbCLIP: Probabilistic adaptation of vision-language foundation model for reliable radiograph-report retrieval. *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision Workshops*, 1-10.

Endo, M., Krishnan, R., Krishna, V., Ng, A. Y., and Rajpurkar, P. (2021). Retrieval-based chest X-ray report generation using a pre-trained contrastive language-image model. *Proceedings of Machine Learning for Health*, 209-219.

Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2024). RAGAS: Automated evaluation of retrieval augmented generation. *Proceedings of EACL System Demonstrations*.

Google Health AI Developer Foundations. (2025). MedGemma model card. https://developers.google.com/health-ai-developer-foundations/medgemma/model-card

Google Health AI Developer Foundations. (2025). MedSigLIP model card. https://developers.google.com/health-ai-developer-foundations/medsiglip/model-card

Jain, S., Agrawal, A., Saporta, A., et al. (2021). RadGraph: Extracting clinical entities and relations from radiology reports. *NeurIPS Datasets and Benchmarks*.

Jin, Q., Kim, W., Chen, Q., et al. (2023). MedCPT: Contrastive pre-trained transformers with large-scale PubMed search logs for zero-shot biomedical information retrieval. *Bioinformatics, 39*(11), btad651. https://doi.org/10.1093/bioinformatics/btad651

Lau, J. J., Gayen, S., Ben Abacha, A., and Demner-Fushman, D. (2018). A dataset of clinically generated visual questions and answers about radiology images. *Scientific Data, 5*, 180251. https://doi.org/10.1038/sdata.2018.251

Jeong, J., Tian, K., Li, A., et al. (2023). Multimodal image-text matching improves retrieval-based chest X-ray report generation. *Medical Imaging with Deep Learning*.

Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*.

Pal, A., Umapathi, L. K., and Sankarasubbu, M. (2023). Med-HALT: Medical domain hallucination test for large language models. *Proceedings of CoNLL*, 314-334.

Park, J., Yoon, B., Kim, S., and Choi, K. (2026). RA-RRG: Multimodal retrieval-augmented radiology report generation with key phrase extraction. *Findings of the Association for Computational Linguistics: ACL 2026*, 5029-5048.

Qwen Team. (2025). Qwen3 Embedding: Advancing text embedding and reranking through foundation models. https://qwenlm.github.io/blog/qwen3-embedding/

Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning transferable visual models from natural language supervision. *Proceedings of the 38th International Conference on Machine Learning*, 8748-8763.

Robertson, S., and Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval, 3*(4), 333-389.

Romanov, A., and Shivade, C. (2018). Lessons from natural language inference in the clinical domain. *Proceedings of EMNLP*, 1586-1596.

Singhal, K., Azizi, S., Tu, T., et al. (2023). Large language models encode clinical knowledge. *Nature, 620*, 172-180.

Soni, S., Gudala, M., Pajouhi, A., and Roberts, K. (2022). RadQA: A question answering dataset to improve comprehension of radiology reports. *Proceedings of LREC 2022*, 6250-6259.

Sun, L., Zhao, J. J., Han, W., and Xiong, C. (2025). Fact-aware multimodal retrieval augmentation for accurate medical radiology report generation. *Proceedings of NAACL 2025*, 643-655.

Xiong, G., Jin, Q., Lu, Z., and Zhang, A. (2024). Benchmarking retrieval-augmented generation for medicine. *Findings of ACL 2024*, 6233-6251.

# Appendices

Appendices A-H preserve historical V9 and preliminary controlled-study artifacts for traceability. They are not the final V16 result. Appendices I-N register the final V10/V12/V16 evidence, reproducibility and release boundary.

## Appendix A: Historical V9 Public Evidence

- Development protocol: `docs/V9_DEVELOPMENT_PROTOCOL.md`
- Full-source split amendment: `docs/V9_FULL_SOURCE_SPLIT_PROTOCOL_AMENDMENT.md`
- RadGraph preprocessing amendment: `docs/V9_RADGRAPH_PREPROCESSING_PROTOCOL_AMENDMENT.md`
- MedSigLIP matrix: `docs/V9_MEDSIGLIP_DEVELOPMENT_MATRIX.md`
- Learned-reranker protocol: `docs/V9_LEARNED_RERANKER_DEVELOPMENT_PROTOCOL.md`
- Retrieval decision record: `docs/V9_RETRIEVAL_DEVELOPMENT_DECISION_RECORD.md`
- Retrieval confirmation protocol and results: `docs/V9_RETRIEVAL_CONFIRMATION_PROTOCOL.md`, `docs/V9_RETRIEVAL_CONFIRMATION_RESULTS.md`
- QA and agent protocol and results: `docs/V9_QA_AGENT_CONFIRMATION_PROTOCOL.md`, `docs/V9_QA_AGENT_RESULTS.md`
- Qualitative protocol and final review: `docs/V9_QUALITATIVE_ANALYSIS_PROTOCOL.md`, `docs/V9_QUALITATIVE_ERROR_ANALYSIS.md`
- Supplemental validity protocol and results: `docs/V9_SUPPLEMENTAL_VALIDITY_PROTOCOL.md`, `docs/V9_SUPPLEMENTAL_VALIDITY_RESULTS.md`
- Final freeze: `docs/V9_TECHNICAL_FREEZE.md`

## Appendix B: Aggregate Result Artifacts

- Retrieval summary: `data/splits/v9/v9_retrieval_confirmation_summary.json`
- QA summary: `data/splits/v9/v9_qa_confirmation_summary.json`
- QA statistics: `data/splits/v9/v9_qa_statistical_analysis.json`
- Agent summary: `data/splits/v9/v9_agent_evaluation_summary.json`
- Split freeze: `data/splits/v9/v9_full_source_split_freeze.json`
- Public qualitative index: `data/splits/v9/v9_qualitative_case_index.csv`
- Qualitative review summary: `data/splits/v9/v9_qualitative_review_summary.json`
- Cross-split duplicate audit: `data/splits/v9/v9_cross_split_duplicate_summary.json`
- Qrel sensitivity: `data/splits/v9/v9_qrel_sensitivity_summary.json`
- Dense baseline and wording robustness: `data/splits/v9/v9_dense_text_robustness_summary.json`
- F1-RadGraph clinical metrics: `data/splits/v9/v9_clinical_metrics_summary.json`
- Structured-output reparse audit: `data/splits/v9/v9_structured_reparse_summary.json`

Large source-derived reports, image pixels, vectors, checkpoints, prompts, and per-row generations remain local. Public hashes verify their frozen identity without redistributing source content.

## Appendix C: Reproduction Entry Points

**Repository:** https://github.com/yzy542968-jpg/wqf7023-medical-rag

**Branch:** `v12-optimization-pilot`

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -q
& ".\.venv\Scripts\python.exe" scripts\run_v9_retrieval_confirmation.py
& ".\.venv\Scripts\python.exe" scripts\analyze_v9_qa_statistics.py
& ".\.venv\Scripts\python.exe" scripts\evaluate_v9_qa_agent.py
& ".\.venv\Scripts\python.exe" scripts\audit_v9_cross_split_duplicates.py
& ".\.venv\Scripts\python.exe" scripts\audit_v9_qrel_sensitivity.py
& ".\.venv\Scripts\python.exe" scripts\run_v9_supplemental_dense_robustness.py
& ".\.venv\Scripts\python.exe" scripts\evaluate_v9_clinical_metrics.py
& ".\.venv\Scripts\python.exe" scripts\audit_v9_structured_output_reparse.py
streamlit run app.py --server.port 8504
```

Exact executable options, model revisions, hashes, failure rules, and local artifact paths are recorded in the frozen protocol and result documents. Confirmation should not be rerun under altered settings and presented as the same study.

## Appendix D: Qualitative and Human-Evaluation Boundary

The researcher reviewed all 24 deterministically selected cases and accepted the assistant-proposed exploratory labels without modification. The process was tool-assisted, author-conducted, and non-blinded; it is not independent radiologist adjudication. No clinical correctness, similarity, harmfulness, usefulness, preference, or inter-rater agreement result is claimed, and selected-pack counts are not extrapolated to the full cohort.

## Appendix E: Dashboard Demonstration Boundary

The dashboard accepts a target chest radiograph, indication, and question; retrieves Top-3 similar historical reports from the frozen 2,506-case V10 Train bank; optionally generates a MedGemma answer; and shows bounded-agent evidence actions. It describes the reports as historical analogies. It must not claim to locate the target patient's own report, diagnose the image, or replace radiologist review.

## Appendix F: Version Boundary

V5-V9 are frozen formative and historical studies. V10 is the frozen methodological foundation and alignment study. V11 and V13-V15 are development or mechanism evidence. V12 is the final learned retrieval method, and V16 is the final integrated held-out method confirmation. Supplemental audits do not change frozen models, prompts, qrels, cases or primary results.

The following appendices preserve the detailed V5 controlled study for traceability. They are formative evidence and do not replace the final V16 study.

## Appendix G: Frozen Preliminary Controlled-Study Methods

### G.1 Research Design

This study used a staged empirical system-comparison design to investigate retrieval-augmented medical question answering over paired radiology images and reports. Earlier text-only experiments identified two structural risks: open-corpus retrieval could select evidence from the wrong case, while a sentence-level verifier could still rate an answer as supported by that wrongly selected report. These findings motivated the preliminary V5 confirmation experiment, which tested whether correctly paired chest X-ray information could improve target-report retrieval and whether any retrieval gain transferred to downstream report-grounded question answering.

Within the preliminary controlled phase, V5 was the final frozen experiment. Its configuration was specified and frozen locally before execution, but it was not formally preregistered or externally timestamped before outcomes were observed. The confirmation cohort was disjoint from all previous project cohorts, although it remained drawn from the same OpenI/IU-Xray source. V5 therefore provides fresh within-source confirmation rather than external validation.

The experiment addressed four linked questions:

1. How strongly do indication text and correctly aligned images affect paired-report retrieval?
2. Is the image contribution specific to the correct image-report alignment rather than generic image features?
3. Does multimodal retrieval improve downstream report-grounded QA under a fixed generation and verification pipeline?
4. What retrieval, generation, verification, and resource trade-offs remain after multimodal reranking?

No V5 model, prompt, threshold, cohort, or result was changed in response to the final quantitative or qualitative analysis.

The staged design was chosen because an end-to-end score alone cannot identify the cause of success or failure. The same final answer error can originate from an ambiguous query, a retrieval miss, insufficient evidence in the selected report, generation omission, semantic-checker filtering, or ambiguity in the source data. Each stage was therefore evaluated with its own observable outputs before the outputs were joined for end-to-end interpretation.

The study combined confirmatory and exploratory elements. The V5 configuration, cohort membership, input conditions, primary comparisons, seeds, and quantitative procedures were frozen before the final confirmation run. The later qualitative analysis was explicitly post-hoc and exploratory, although its case-selection protocol was committed before systematic extraction and coding. This distinction prevents the qualitative findings from being presented as blinded confirmatory evidence while still allowing them to explain frozen quantitative outcomes.

The unit of analysis was the radiology case, not the individual question. Each case produced three questions, which created within-case dependence. Treating all 360 questions as independent observations would underestimate uncertainty because findings, impression, and summary questions share the same target report and retrieval ranking. Case-grouped resampling retained this structure and aligned the statistics with the implemented case-level split.

The design also fixed the downstream generator and checker across report-only and multimodal retrieval. This controlled comparison ensured that any difference in answer outcomes originated from the evidence selected upstream rather than from a larger language model, a different prompt, or a changed verification threshold. It therefore tested transfer through the pipeline rather than comparing unrelated systems.

### G.2 Data Source and Cohort Construction

The study used de-identified OpenI/IU-Xray chest radiograph cases with linked reports and one or more image views. Each processed case contained a stable case identifier, indication, findings, impression, problem labels, and linked image metadata. De-identification placeholders were retained because replacing or inferring their hidden content could introduce unsupported information.

The V5 cohort contained 240 cases that were excluded from all earlier project cohort manifests. A fixed seed of 7023 divided these into 120 development cases and 120 confirmation cases. The confirmation set contributed 360 report-derived questions: one findings question, one impression question, and one summary question per case. Statistical resampling and comparison used the case identifier as the grouping unit so that the three questions from one case were not treated as independent patients.

Retrieval used all 240 fresh-cohort cases as the candidate pool. The 120 confirmation cases were the target cases for final evaluation. This design measured closed-set paired-report retrieval; it did not evaluate diagnosis of previously unseen patients.

Cohort construction began from locally processed case records produced from the downloaded OpenI report and image archives. Records without the required report content or usable image association were not eligible for the paired experiment. Stable identifiers were carried through every artifact so that a retrieval row could be traced back to one case record without exposing an inferred patient identity. Image paths, report fields, and derived question references were validated before splitting.

The processed source contained 3,851 case records. After excluding 1,260 identifiers from earlier project manifests, 674 fresh cases satisfied the V5 eligibility rule; 240 were selected by the frozen seed. Eligibility required an image, at least 40 characters of findings, at least 8 characters of impression, an indication placeholder ratio no greater than 0.5, and a non-empty, non-`normal` problem field. These rules improve deterministic data quality but create an abnormality-enriched cohort.

Freshness was enforced against manifests from earlier development stages. This prevented previously used cases from silently reappearing in V5 and reduced the risk that manual inspection, prompt adjustment, or threshold development had indirectly adapted to the confirmation examples. The term fresh therefore means excluded from earlier project cohorts, not collected from a different hospital. The source distribution, reporting style, and acquisition practices remained those of IU X-Ray/OpenI.

The split was performed at case level. The fixed seed made the assignment reproducible and ensured that the same processed case identifier did not occur in both development and confirmation. The processed source records did not contain a released patient or subject identifier. The source collection included no more than one study per patient, so separation was supported by source design but could not be independently re-verified from identifiers. Development cases could be used to verify implementation and confirm that the frozen pipeline executed, whereas confirmation outcomes were reserved for the reported V5 comparisons.

The candidate corpus deliberately included both development and confirmation cases. Every confirmation query therefore had to identify its target among 240 plausible paired reports rather than among only the 120 evaluation targets. This creates a more demanding ranking task while retaining a known target for qrels. It is still a small closed corpus relative to a hospital archive, and the candidate set always contains the target by construction.

Data preprocessing preserved report section boundaries. Indication, findings, and impression were stored separately and also assembled into a normalized report representation for retrieval and prompting. Empty or placeholder-only fields were handled deterministically. De-identification tokens were not expanded, and laterality or wording inconsistencies were not silently corrected because such intervention would change the source evidence.

For images, the case record retained every available linked view. Pixel loading and encoder preprocessing followed the BioViL-T requirements in the local runtime. Per-view vectors and case aggregates were cached so that repeated shuffled controls did not repeatedly encode the same image. This separated expensive image encoding from lightweight reranking and made the 100-control analysis computationally feasible.

### G.3 Question and Input Conditions

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

The question templates were selected to represent three report functions rather than three unrelated clinical tasks. The findings question targets descriptive observations, the impression question targets the radiologist's final conclusion, and the summary question tests whether the system can prioritize the principal abnormality or conclusion. Using the same roles for every case creates balanced question-type coverage and allows each shuffled or ablation condition to be evaluated on identical prompts.

References were derived from the corresponding frozen report fields. They therefore provide deterministic answer targets and avoid asking a language model to invent gold labels. The findings question uses the findings section, while both the impression and summary questions use the impression section. Thus the aggregate QA metric counts impression content twice per case; this composition is disclosed and examined in a frozen-output sensitivity analysis rather than corrected after observing outcomes. The reference wording is close to the report, section boundaries may make retrieval easier, and a clinically acceptable paraphrase can receive lower Token-F1. The reference is best understood as a reproducible textual target rather than an independently adjudicated clinical answer.

Input construction was deterministic. The question-only query contained only the role-specific question. The indication condition concatenated the stored referral indication with the same question. Image conditions supplied a case-level image vector to the reranker while leaving the text query unchanged. The shuffled condition replaced only that vector through the frozen permutation mapping. No condition received the target identifier or reference answer.

This factorial interpretation is important. Question-only versus indication-plus-question measures the contribution of referral text. Question-only versus question-plus-image shows whether image information can help when text is deliberately underidentified. Indication-plus-question versus indication-plus-question-plus-image estimates incremental image value beyond the strongest text baseline. Correctly aligned versus shuffled image tests whether that incremental value depends on pairing.

### G.4 Case-Aware Evidence Representation and Downstream QA Workflow

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

The workflow was implemented as a sequence of explicit artifacts rather than one opaque application call. Retrieval outputs stored ranked candidate identifiers and component scores. Prompt packs stored the selected evidence and fixed instruction format. Generation files stored draft responses. Evaluation rows stored sentence-level support decisions, final answers, references, and metrics. Aggregate summaries were produced from these lower-level records. This structure allowed a reported value to be traced back without rerunning the complete GPU pipeline.

The Top-1 decision connected retrieval and QA. The generator did not receive several candidate reports and was not allowed to choose among them. This made the transfer test strict: a rank improvement that stopped at rank two could improve MRR but could not improve the evidence shown to the generator. The qualitative analysis later used this distinction to separate target-rank improvement from Top-1 retrieval success.

The workflow did not use oracle recovery. When the target report was not ranked first, the generator still received the selected wrong report. Substituting the known target would measure generation with perfect retrieval and would conceal the practical impact of ranking errors. The non-oracle path therefore provides a more realistic estimate of the benefit transferred from the frozen retriever.

Evidence traces also supported abstention analysis. A final empty or abstaining response could arise because the selected report lacked target information, because the draft was unsupported, or because the verifier removed report-supported content. Retaining the draft, filtered sentences, and checker scores allowed these possibilities to be distinguished during review.

### G.5 Text Retrieval

BM25 provided the transparent sparse retrieval baseline. The question-only condition intentionally exposed patient-scope ambiguity because the three generic question templates contained little case-specific information. The indication-plus-question condition tested how much clinical referral text reduced this ambiguity.

The multimodal conditions first produced the same text shortlist of 100 candidate cases. Scores were normalized independently within that shortlist. Ties were resolved by descending fused score and then ascending case identifier, giving a deterministic ranking.

BM25 scores a candidate report by accumulating query-term contributions weighted by inverse document frequency, term frequency saturation, and document-length normalization. In simplified form, the score is:

```text
BM25(q,d) = sum IDF(t) * [f(t,d) * (k1 + 1)] /
            [f(t,d) + k1 * (1 - b + b * |d| / avgdl)]
```

The implementation used the frozen tokenizer and BM25 parameters recorded by the project environment. No confirmation labels were used to edit queries or report text. The same indexed report representation and candidate order were reused across comparable conditions.

Question-only retrieval served as a deliberately difficult lower bound, not as a realistic patient-scoped clinical query. Because all cases reuse three semantic roles, the query mainly matches common radiology language. Indication-plus-question retrieval is the more informative text baseline because the indication supplies case-specific lexical detail. Reporting both prevents the indication gain from being mistaken for an image gain.

The shortlist size of 100 balances two considerations. A larger shortlist gives the image more opportunity to rescue a target that BM25 ranks poorly, while a smaller shortlist reduces reranking cost. The value was frozen before confirmation evaluation. Since only shortlisted reports receive multimodal scores, the reported image effect is conditional on the target surviving the text stage.

### G.6 Image Encoding and Multimodal Reranking

Image and report representations used `microsoft/BiomedVLP-BioViL-T` with frozen revision `692f09e` and 128-dimensional embeddings. Each available X-ray view was normalized, case views were averaged, and the resulting case vector was normalized again. No image pixels were sent to an online service.

For multimodal reranking, normalized text and image similarities received equal weights of 0.5. The correctly aligned condition used the image embedding linked to the target case. This image was used only as a retrieval query signal; the system did not generate a diagnosis directly from pixels. The downstream generator received the selected report evidence rather than raw images.

For a case with multiple views, each view vector was L2-normalized before aggregation. The arithmetic mean represented the case, followed by another L2 normalization. This prevents a case with more available views from receiving a larger vector norm merely because it contains more images. It also creates one stable query vector per examination for caching and permutation.

Candidate report embeddings were computed in the same BioViL-T representation space. Cosine similarity measured image-report alignment within the BM25 shortlist. Text and image scores were independently min-max normalized because their raw scales are not directly comparable. For candidate d, the fused score can be summarized as:

```text
fused(d) = 0.5 * normalized_BM25(d)
         + 0.5 * normalized_image_report_similarity(d)
```

Equal weighting was a fixed engineering policy rather than an optimized clinical parameter. It gives both modalities influence and keeps interpretation simple. A learned or extensively tuned weight might improve performance but would require a separate validation procedure and would add another source of confirmation-set overfitting.

The reranker returned a complete deterministic ordering for the shortlist. When fused scores were equal, ascending case identifier resolved the tie. Deterministic tie handling matters because Hit@1 and MRR can otherwise vary across runs even when component scores are unchanged.

### G.7 Shuffled-Image Control

The alignment control used 100 deterministic fixed-point-free permutations with seed 7023. In each permutation, every source case received another case's image embedding and no case retained its own image. Text queries, candidate reports, shortlist size, fusion weights, and evaluation procedure remained unchanged.

The control tested whether the correctly aligned image outperformed image-conditioned reranking with incorrect case alignment. It did not prove causal clinical image understanding. A plus-one Monte Carlo value was calculated as `(b+1)/(m+1)`, where `b` was the number of shuffled runs meeting or exceeding the correctly aligned result and `m=100` was the number of permutations.

A fixed-point-free permutation is stronger than an unconstrained shuffle because it guarantees that every target receives an incorrect image. In an ordinary random shuffle, some cases could retain their own image, weakening the negative control. Each of the 100 mappings was stored or reproducibly generated from the frozen seed so the exact null distribution could be reconstructed.

The same incorrect mapping was applied consistently to all three questions from a case within a run. This preserved the case as the unit of dependence and prevented findings, impression, and summary questions from receiving different pseudo-patients. Across runs, text input and candidate reports remained fixed, so variation arose from image assignment.

The shuffled distribution is an empirical alignment control rather than a universal null model. Incorrect images may sometimes be clinically similar to the target and can still help or harm ranking. The relevant observation is whether correct alignment lies beyond the outcomes produced by many controlled misalignments under the same fusion mechanism.

### G.8 Answer Generation and Semantic Verification

Both report-only and multimodal retrieval conditions used the same local `Qwen/Qwen2.5-1.5B-Instruct` generator. Generation used CUDA, float16, batch size 16, maximum 256 new tokens, temperature 0, and a direct non-oracle prompt. The generator did not receive the frozen target identifier or reference answer.

The semantic checker used `pritamdeka/PubMedBERT-MNLI-MedNLI`. It combined lexical evidence matching, entailment and contradiction probabilities, and polarity consistency. Its locked configuration used lexical weight 0.2, support threshold 0.6, entailment threshold 0.75, and contradiction threshold 0.5. Evidence scope was restricted to the top-ranked selected report. The action path could retain supported sentences, filter flagged sentences, or abstain if no usable answer remained.

The checker was an automated evidence signal rather than a clinical gold standard. Its support rate measured agreement with selected-report evidence, not target-patient correctness or clinical safety.

Prompt construction used the same instruction structure for both retrieval conditions. The prompt identified the selected report as the only evidence source, supplied the question, requested a concise answer, and allowed abstention when the report did not support an answer. It did not reveal whether retrieval was correct and did not include the frozen reference. Temperature zero reduced sampling variation, while the 256-token limit was sufficient for the short report-grounded task.

Qwen2.5-1.5B-Instruct was selected because it could run locally within the available laptop GPU memory and support batched deterministic generation. The study does not claim that this model is optimal for radiology. Holding it fixed was more important for causal comparison than maximizing absolute generation quality with multiple generators.

The verifier segmented each draft into sentences and evaluated each sentence against selected-report evidence. Lexical matching rewarded direct support, the NLI model estimated entailment and contradiction, and polarity checks guarded against simple negation reversals. The frozen action policy converted these signals into retained, filtered, or abstaining output. Sentence-level records were preserved so that a final revision rate could be decomposed during qualitative review.

The checker could not correct retrieval identity. Its premise was always the selected report. If that report belonged to the wrong case, a faithful draft could pass. Conversely, a correct-report paraphrase could be rejected by the checker. For this reason, automated support was reported as one pipeline measure and was not used to redefine retrieval qrels or reference answers.

### G.9 Evaluation Metrics and Statistical Analysis

Retrieval metrics were Hit@1, Hit@5, Hit@10, MRR, and an extractive proxy Token-F1 calculated from the selected report evidence. Hit@1 measured Top-1 target-case alignment; MRR retained information about target-rank movement even when the target did not reach first place.

QA metrics were draft Token-F1, final Token-F1 after semantic checking, automated evidence-support rate, revision rate, and abstention rate. Token-F1 measured reference overlap and was not interpreted as clinical correctness.

V5 used 5,000 grouped bootstrap resamples at case level and paired randomization tests with seed 7023. The primary retrieval comparison was indication-plus-question with correctly aligned image minus indication-plus-question BM25. The primary QA comparison was multimodal final Token-F1 minus report-only final Token-F1. Confidence intervals and p-values therefore preserved the dependence among questions from the same case.

Hit@k was defined as the proportion of questions for which the frozen target report appeared within the first k positions. Hit@1 corresponds to the report actually passed downstream. MRR averaged the reciprocal target rank, assigning greater credit to movement near the top. These metrics answer different questions: Hit@1 evaluates the operational selection decision, whereas MRR detects useful ordering changes that may support future multi-document or retry policies.

The extractive proxy Token-F1 compared answer-bearing target text with evidence available from a ranked candidate. It is a retrieval-oriented approximation of answer availability rather than a generated-answer metric. An increase indicates that the selected or highly ranked evidence contains more target wording, but it does not demonstrate that the generator uses that evidence correctly.

Draft Token-F1 measured overlap between the raw generated response and frozen reference. Final Token-F1 measured the response after checker actions. Precision and recall were calculated over normalized tokens and combined as their harmonic mean. This metric is transparent and reproducible but insensitive to some forms of semantic equivalence and clinical importance. It may reward copied phrasing and penalize acceptable paraphrase.

Automated support rate summarized checker judgments, revision rate measured how often the final response differed from the draft, and abstention rate measured outputs in which no substantive answer was retained. These outcomes were interpreted jointly. A higher revision rate can represent useful filtering or excessive intervention; a higher abstention rate can represent caution or loss of valid coverage.

For paired bootstrap analysis, cases were sampled with replacement and all three questions for each sampled case were included. The 2.5th and 97.5th percentiles of 5,000 paired differences formed the reported 95% interval. Paired randomization swapped condition labels within cases under the null and recalculated the difference. The seed and iteration count were fixed for reproducibility.

Statistical significance was not treated as clinical significance. The candidate pool was controlled, questions were templated, and metrics were automatic. Confidence intervals quantify uncertainty within this benchmark; they do not account for institutional shift, different clinical workflows, or expert disagreement.

### G.10 Researcher-Reviewed Qualitative Analysis

A post-hoc qualitative protocol was committed after the technical freeze but before systematic case extraction and coding. Some individual outputs had previously been inspected during pipeline verification, so this was not a result-blind preregistration.

The fixed protocol selected 24 representative questions: six target-rank improvements, six target-rank degradations, six QA-gain/support-loss cases, and six correct-retrieval generation-error cases. Each stratum contained two findings, two impression, and two summary questions. The full 360-question numeric index was retained.

Protocol taxonomy v1.0 was preserved in the audit trail. During interpretation, a refined three-level taxonomy v1.1 separated pipeline stage, specific pattern, and outcome modifier. It distinguished target-rank movement from Top-1 success, generation omission from post-verification content loss, and abstention occurrence from its suspected cause. Assistant-proposed v1.1 labels were recorded separately from the original labels. The researcher reviewed and accepted all 24 proposals on 19 August 2026, producing 24 accepted, 0 modified, and 0 excluded cases.

Qualitative counts describe only this predefined purposive review set. They were not used for population-level inference, verifier accuracy estimation, or clinical error-rate estimation.

Case extraction was deterministic. The script read frozen retrieval and QA rows, calculated rank and metric deltas, applied the protocol strata, balanced question roles, and produced a 24-row review package. The public package retained identifiers, metric changes, original protocol labels, proposed refined labels, review status, and concise notes. Full report text and model generations remained local under repository policy.

Taxonomy v1.1 used three levels. Pipeline stage located the issue in retrieval, generation, verification, abstention, or data ambiguity. A specific pattern described the mechanism, such as Top-1 retrieval failure, generation omission, possible verifier over-rejection, or de-identification ambiguity. An outcome modifier recorded cross-stage effects such as QA gain with support loss or no substantive answer loss. This structure avoided treating every observation as one mutually exclusive error.

The researcher decision field distinguished accepted, modified, excluded, and pending cases. In the completed review, all 24 assistant proposals were accepted as the researcher-reviewed labels. This does not make the labels independent or clinically adjudicated. It establishes that the named researcher reviewed the proposed interpretation and accepted it for exploratory analysis.

The qualitative analysis used cautious language. Without physician gold labels, it did not call checker decisions false positives or false negatives. Terms such as possible over-rejection, suspected unnecessary abstention, and abstention consistent with available evidence describe the observed relation among the frozen report, reference, and outputs without asserting definitive clinical correctness.

### G.11 Computational Cost and Reproducibility

The frozen manifest stored the cohort fingerprint and LF-normalized SHA-256 values for configurations, code, aggregate results, and tests. Large generations, prompt packs, image pixels, model weights, and private full-text review rows remained local.

Generation timing was measured on an NVIDIA GeForce RTX 5070 Laptop GPU with 8,150.6 MiB total memory. These values are machine-, cache-, and generated-length-dependent and do not constitute a complete production latency or energy analysis.

Reproducibility operated at several levels. Configuration reproducibility stored model identifiers, revisions, seeds, shortlist size, fusion weights, thresholds, batch sizes, and generation parameters. Data reproducibility stored case counts, case-level partitions, and a cohort fingerprint; a released patient identifier was not available for independent re-verification. Result reproducibility stored aggregate JSON summaries and statistical outputs. Implementation reproducibility stored scripts and tests. The artifact manifest joined these with LF-normalized SHA-256 values so that unintended changes could be detected across platforms.

Large files were separated from the public repository for practical and licensing reasons. Image archives, local processed image pixels, model weights, full prompt packs, per-question generations, and some detailed review material remained local. Their absence from GitHub is documented rather than hidden. Public aggregate files and indices are sufficient to audit reported counts and metrics, while authorized users can rerun the complete local pipeline after obtaining the source data and models.

Runtime measurement distinguished generation-only time from total processing time and reported throughput and peak allocated GPU memory. Earlier component measurements were retained for image encoding, BM25 retrieval, and cached reranking. The measurements were not combined into a deployment service-level claim because startup, disk cache, concurrent users, web overhead, and hardware power were outside the protocol.

Automated tests covered cohort behavior, multimodal retrieval logic, dashboard integration, and artifact assumptions. Passing tests do not validate scientific claims, but they reduce the risk that reported behavior results from accidental schema drift or broken code paths. Hash checks and tests were rerun after manuscript generation to confirm that documentation changes did not modify frozen V5 artifacts.

### G.12 Ethics and Claim Boundaries

The system was a research prototype. It did not provide treatment recommendations, authenticate clinical users, or claim deployment safety. V5 did not establish image-based diagnosis, clinical causality, external validation, natural-question generalization, or human-validated verifier correctness. Images and reports were processed locally, and no attempt was made to reverse de-identification.

The source collection is de-identified and publicly distributed for research, but de-identification does not remove every ethical responsibility. The project minimized redistribution of raw content, retained source licensing and citation requirements, and avoided sending images or reports to ordinary online language-model services. Local processing reduced exposure to third-party retention and training policies.

The system was designed for retrospective experimentation rather than clinical access control. A real deployment would require authenticated patient scope, role-based permissions, audit logging, encryption, data-retention governance, and procedures for correcting records. Visual similarity should never be used as a substitute for patient identity when an authorized record identifier is available.

The dashboard language follows this boundary. It can state that the system retrieves the top-ranked candidate report from the indexed corpus and shows the evidence used for generation. It must not state that an arbitrary uploaded image has been matched to the true patient record, that the model diagnosed the image, or that an answer is medically safe.

Finally, the analysis avoids fabricating human evaluation. The completed researcher review supports exploratory interpretation of 24 cases. Independent radiologist correctness, harmfulness, preference, and inter-rater agreement remain future work. This distinction is preserved in the manuscript, repository, and demonstration narrative.

### G.13 Methodological Summary

The methodology links each research question to a controlled comparison. RQ1 uses question and indication ablations to measure patient-scope ambiguity and then compares the strongest text baseline with correctly aligned image reranking. RQ2 holds the text workflow fixed and replaces the correct image with 100 deterministic fixed-point-free misalignments. RQ3 passes the Top-1 report from report-only and multimodal retrieval through the same non-oracle generator and checker. RQ4 joins frozen metrics with a protocol-driven 24-question qualitative review.

Several safeguards reduce avoidable bias. Cases were fresh relative to earlier project cohorts and split at case level. Confirmation targets were evaluated in a fixed 240-case candidate pool. Model identifiers, revisions, seeds, prompts, thresholds, fusion weights, shortlist size, and statistical iterations were frozen. Case-grouped resampling respected the three-question dependency. Artifact hashes made later changes detectable.

The design also preserves negative evidence. The question-only condition is retained despite poor performance because it measures scope ambiguity. The shuffled-image distribution is retained despite lower scores because it tests alignment. Hit@1 is reported even though its evidence is weaker than MRR. Automated support decline is reported alongside Token-F1 improvement. Qualitative cases include target-rank degradation and suspected verifier disagreement rather than only successful examples.

The methodology does not eliminate all bias. V5 remains within one dataset, uses templated questions, assumes the target is present, and evaluates a single frozen model path. The technical freeze was not a formal preregistration, and the qualitative review was not independent clinical adjudication. These limitations define the appropriate inference: a controlled within-source result about alignment-specific retrieval and downstream automatic QA behavior.

The complete method can be reproduced in stages. A researcher can rebuild the cohort and verify its fingerprint, regenerate embeddings and retrieval rows, rebuild prompt packs, rerun local generation and semantic evaluation, reproduce grouped statistics, verify artifact hashes, and rebuild qualitative materials. The public repository supplies code and aggregate evidence, while source data, weights, images, and large row-level artifacts must be obtained or retained locally according to their licenses and storage policy.

This balance between control, auditability, and bounded claims is central to the project. The method is complex enough to test a genuine multimodal RAG pipeline but modular enough that one favorable end-to-end score cannot hide the behavior of its components.

## Appendix H: Frozen Preliminary Controlled-Study Results

### H.1 Patient-Scope Ambiguity and the Indication Shortcut

The four principal confirmation retrieval conditions are shown in Table H.1.

Table H.1. Retrieval results under four principal input conditions

| Input condition | Hit@1 | Hit@5 | Hit@10 | MRR | Extractive proxy Token-F1 |
|---|---:|---:|---:|---:|---:|
| Question only, BM25 | 0.0056 | 0.0222 | 0.0472 | 0.0277 | 0.1981 |
| Indication + question, BM25 | 0.5889 | 0.7222 | 0.7750 | 0.6590 | 0.6602 |
| Question + correctly aligned image | 0.0139 | 0.0722 | 0.1139 | 0.0515 | 0.2334 |
| Indication + question + correctly aligned image | 0.6222 | 0.7778 | 0.8389 | 0.6971 | 0.7245 |

Question-only retrieval was nearly non-identifying: Hit@1 was 0.0056 and MRR was 0.0277. This was expected because the same three templates were reused across cases. Adding indication increased Hit@1 to 0.5889 and MRR to 0.6590. The indication therefore acted as a powerful retrieval shortcut in this controlled benchmark.

The effect is methodologically important. A high retrieval score cannot be attributed only to sophisticated multimodal reasoning when referral text already contains strong case-discriminating language. For this reason, V5 reports indication ablation explicitly and treats the indication-plus-question BM25 condition as the primary text baseline.

The question-only result quantifies the scope ambiguity built into the benchmark. With 240 candidates and only three repeated question forms, the query does not specify which examination is intended. Hit@1 of 0.0056 is close to what would be expected from a nearly non-identifying query over a large candidate set. The low value should not be interpreted as evidence that BM25 is generally unsuitable for radiology. It demonstrates that a generic question cannot perform the work of a patient or case identifier.

The indication changes the information content of the query. Terms describing pain, cough, prior surgery, device placement, trauma, or follow-up can overlap with candidate reports and sharply reduce the plausible set. The increase to Hit@1 0.5889 shows that the benchmark contains substantial lexical linkage between referral text and report content. This is a useful system feature when indication is legitimately available, but it is also a shortcut that must be disclosed before attributing performance to image understanding.

The gap between Hit@1 and Hit@10 remains informative. Even with indication, the target was not first for approximately two-fifths of questions, while Hit@10 reached 0.7750. Thus, many targets were retrievable but not selected by a strict Top-1 policy. A workflow that allowed evidence comparison across several candidates might recover more targets, but it would introduce additional risks of cross-case mixing and require a new selection protocol.

The fixed Top-100 shortlist creates an upstream availability ceiling. The target was present in the question-only BM25 shortlist for 42.50% of confirmation questions, compared with 98.61% when indication was included. Image reranking cannot rescue a target excluded before the image scores are applied. The primary aligned-image comparison is therefore an incremental reranking test conditional on the indication-defined candidate region, not a full-corpus image retrieval test.

The extractive proxy Token-F1 of 0.6602 under indication-plus-question retrieval indicates that the ranked evidence often contained target-answer wording. It does not mean the generator achieved that score. The difference between proxy evidence availability and generated final Token-F1 later shows how much performance is lost through Top-1 selection, generation, and verification.

### H.2 Indication and Correct-Image Ablation

The correctly aligned image produced a small improvement when used with question text alone: MRR rose from 0.0277 to 0.0515 and proxy Token-F1 from 0.1981 to 0.2334. These values remained low because the generic question supplied little textual case identity.

Against the stronger indication-plus-question BM25 baseline, correctly aligned image reranking increased MRR by 0.0381, with case-bootstrap 95% CI [0.0159, 0.0614] and paired-randomization p=0.0012. Proxy Token-F1 increased by 0.0643, CI [0.0282, 0.1029], p=0.0006. Hit@5 increased by 0.0556 and Hit@10 by 0.0639, with paired-randomization p=0.0024 and p=0.0052 respectively.

The Hit@1 increase was smaller: +0.0333, from 0.5889 to 0.6222. Its confidence interval reached approximately zero and the paired-randomization p-value was 0.0886. Thus, the strongest evidence concerns improved target ordering and retrieval within the upper ranks, not a definitive Hit@1 improvement.

The metric pattern is internally coherent. Image reranking produced larger gains for MRR, Hit@5, Hit@10, and proxy Token-F1 than for Hit@1. This suggests that image information often moved the correct report upward without always moving it past every competing report. The result matches the qualitative rank-improvement cases, where large changes from deep ranks to the upper list still stopped short of first place.

This distinction matters because different applications value ranks differently. A human-facing search tool may benefit when the correct report moves from rank 60 to rank 10. The V5 generator does not, because it consumes only rank one. MRR is therefore evidence of useful representation value, while Hit@1 is the operational measure for the implemented downstream path.

The question-plus-image condition provides another perspective. Its MRR of 0.0515 exceeded question-only BM25 but remained very low. The image alone could not reliably identify a report when reranking was constrained by an underidentified text shortlist. This shows complementarity rather than image dominance: the indication retrieves a plausible region of the corpus, and the aligned image improves ordering within that region.

Equal fusion weighting may also limit Top-1 effects. Text and image signals can disagree, and min-max normalization gives each relative rather than calibrated influence. A different weight might improve Hit@1, but selecting it after observing confirmation outcomes would compromise the frozen comparison. The reported result therefore characterizes the predetermined policy, not the theoretical maximum attainable with BioViL-T.

### H.3 Correctly Aligned Versus Shuffled Images

Correct alignment achieved MRR 0.6971 and proxy Token-F1 0.7245. Across 100 shuffled-image derangements, mean MRR was 0.5659 with range [0.5158, 0.6084], while mean proxy Token-F1 was 0.5950 with range [0.5310, 0.6455]. No shuffled run equalled or exceeded the correctly aligned result for either metric.

The plus-one Monte Carlo value was 0.0099 for both MRR and proxy Token-F1. The result supports an alignment-specific contribution: the benefit was not reproduced by attaching arbitrary image embeddings to the same text workflow. It does not prove clinical image interpretation, because the task remained closed-set paired-report retrieval and did not test diagnosis from pixels.

The shuffled means were lower than both the correct-alignment result and, for MRR, the indication-plus-question text baseline. Incorrect image fusion can therefore actively disturb a useful text ranking. This is an important safety-oriented observation: adding a visual modality is not automatically beneficial. Its value depends on whether the image is correctly associated with the query and whether the fusion policy handles conflicting signals.

The range across shuffled runs shows that mismatch effects vary with the accidental clinical similarity of assigned images. Some incorrect assignments may resemble the target report and preserve useful ordering, while others introduce misleading similarity. No single shuffled run can summarize this variability. The 100-run distribution demonstrates that the correctly aligned outcome was consistently stronger than the tested misalignments.

The plus-one value of 0.0099 is the smallest value available with 100 permutations when none meets or exceeds the observed statistic. It should not be described as a conventional exact p-value proving image causality. It is an empirical control result under the frozen permutation scheme. The conclusion is appropriately scoped: correct pairing carried information useful for this retrieval task beyond score perturbation caused by arbitrary images.

The control also tests data plumbing. If image paths or case mappings were ignored, correct and shuffled conditions would be similar. Their separation provides evidence that the implemented pipeline actually used the paired image association. It does not reveal which anatomical features the encoder used or whether those features are clinically appropriate.

### H.4 End-to-End Question Answering

The same generator and checker are compared after report-only and multimodal retrieval in Table H.2.

Table H.2. End-to-end QA comparison

| Pipeline | Draft Token-F1 | Final Token-F1 | Automated support | Final abstention | Revision rate |
|---|---:|---:|---:|---:|---:|
| Report-only retrieval | 0.3632 | 0.3563 | 0.8409 | 0.0556 | 0.7389 |
| Multimodal retrieval | 0.3897 | 0.3865 | 0.8069 | 0.0611 | 0.7250 |

Multimodal retrieval improved draft Token-F1 by 0.0265, CI [0.0094, 0.0441], paired-randomization p=0.0026. Final Token-F1 improved by 0.0302, CI [0.0101, 0.0511], p=0.0032. This demonstrates that the retrieval gain transferred to the final QA output under a fixed non-oracle generation path.

However, automated evidence support decreased by 0.0340, CI [-0.0566, -0.0122], p=0.0034. Final abstention increased by only 0.0056, with an interval crossing zero and p=0.7299. The central result is therefore a performance-grounding trade-off: reference overlap improved while the automated support signal declined.

This trade-off must not be simplified into a claim that multimodal answers were less clinically faithful. The support metric was produced by the same automated checker later shown to filter both substantive sentences and generic answer prefixes.

The draft improvement of 0.0265 indicates that better selected evidence affected generation before verification. The final improvement of 0.0302 is slightly larger, meaning that checker actions did not erase the overall multimodal advantage in reference overlap. This is the principal transfer result: under the same model and prompt, changing retrieval evidence changed downstream answer quality in the expected direction.

Absolute final Token-F1 remained modest. The multimodal value of 0.3865 is far below the extractive proxy of 0.7245 because the two metrics measure different objects and because several losses occur between ranking and final answer. The target may not be Top-1, the selected report may express the answer differently, the small generator may omit details, and the checker may remove content. The gap prevents the retrieval improvement from being mistaken for solved QA.

Revision rates were high in both systems, exceeding 0.72. The checker therefore intervened in most drafts rather than acting as a rare safeguard. A component that changes most outputs has substantial influence over the final metric and requires independent scrutiny. Similar revision rates between conditions do not imply similar revision quality because the content and evidence differ.

The support decline can have several explanations. Multimodal retrieval may expose reports whose wording leads the generator to make more composite claims. The checker may be sensitive to paraphrase or sentence length. It may remove substantive supported content or only formulaic prefixes. The automated aggregate cannot distinguish these mechanisms; this is why the qualitative analysis is part of the result chain rather than a decorative appendix.

Final abstention increased by only 0.0056 and the paired interval included zero. The support-rate difference therefore did not translate into a clear general increase in no-answer behavior. Much of the checker effect occurred through revision of retained answers. This reinforces the need to inspect post-verification content rather than relying only on abstention frequency.

### H.5 Researcher-Reviewed Qualitative Findings

The 24-case review package contained 19 unique cases and eight questions of each type. Relative to protocol v1.0, assistant interpretation was unchanged for nine rows and refined for 15. The researcher accepted all 24 v1.1 proposals without further modification.

The overlapping accepted labels included 14 Top-1 retrieval failures, 10 Top-1 retrieval successes, 11 target-rank improvements, 7 target-rank degradations, 10 post-verification content-loss cases, 9 possible verifier-over-rejection cases, and 6 QA-gain/support-loss cases. These values characterize the selected review package only.

Five exploratory findings were supported:

1. **Target-rank improvement did not always translate into Top-1 retrieval success.** The six extreme improvement examples moved targets from ranks 59-98 to ranks 10-27, but none reached first place.
2. **Report-level faithfulness did not guarantee alignment with the frozen target case.** Several answers accurately summarized an incorrectly selected report.
3. **Correct Top-1 retrieval did not guarantee a reference-consistent final answer.** One clear generation-focus case emphasized pectus deformity while omitting the report conclusion of no acute disease.
4. **In reviewed cases, automated verification sometimes appeared to remove report-supported content.** Five of six selected correct-retrieval generation-error rows contained filtered sentences, and three ended in abstention.
5. **Some declines in automated evidence-support scores did not correspond to substantive answer degradation.** In two reviewed impression cases, the checker removed only a generic answer prefix while preserving the complete reference-consistent conclusion.

Additional cases exposed data limitations. One de-identification token prevented confident adjudication of whether “Lungs are clear” was supported. Another report contained an apparent left-versus-right upper-lobe inconsistency between findings and impression. These examples show that data quality can affect both generation evaluation and verifier interpretation.

The first finding refines the interpretation of MRR gains. A large positive rank delta is useful evidence that the image altered ordering, but it is not equivalent to successful evidence delivery. In the six extreme improvement cases, targets rose from ranks 59-98 to ranks 10-27 and therefore remained unavailable to the Top-1 generator. These cases explain how statistically reliable MRR improvement can coexist with weaker Hit@1 evidence and a smaller downstream QA gain.

The second finding identifies a fundamental limitation of local faithfulness. When a wrong report is selected, the generator may accurately summarize its findings and the checker may correctly detect entailment. Every local component can appear successful while the response remains associated with the wrong frozen case. This is not ordinary hallucination because the statement may be present in evidence. It is an evidence-ownership failure produced upstream.

The third finding shows that correct retrieval is necessary but insufficient. Even when the target report is selected, a small generator may focus on a salient secondary detail, omit the principal conclusion, or compress findings in a way that diverges from the reference. Retrieval qrels cannot detect this behavior. It must be examined through generated text, reference comparison, and evidence traces.

The fourth and fifth findings qualify the support-rate result in opposite directions. Some filtered sentences appeared to contain report-supported clinical content, suggesting possible over-rejection and post-verification information loss. Other filtered material was only a generic answer prefix, so the lower support signal did not correspond to loss of the substantive conclusion. Aggregate support decline therefore combines meaningfully different outcomes.

Abstention cases further illustrate this heterogeneity. An abstention may be consistent with the available selected report when retrieval failed or evidence was insufficient. It may also be suspected unnecessary when the draft contained supported content that the checker removed. The review taxonomy records abstention as an outcome flag and uses separate interpretive labels for its possible cause.

Data ambiguity was retained as its own stage because not every discrepancy should be attributed to the model. De-identification placeholders can obscure syntax, and contradictions within a source report can make both a reference and verifier decision uncertain. A responsible analysis should expose such cases and avoid manufacturing certainty that the source document does not provide.

The reviewed counts cannot be extrapolated to all 360 questions. The set was purposively selected to explain extreme rank movement, QA-support trade-offs, and correct-retrieval generation problems. It overrepresents these phenomena by design. Its value is mechanism discovery and illustration, not prevalence estimation.

### H.6 Computational Cost

Table H.3. Runtime and computational cost

| Pipeline condition | Records | Total process | Generation only | Generation throughput | Peak allocated GPU memory |
|---|---:|---:|---:|---:|---:|
| Report-only | 360 | 87.86 s | 78.56 s | 4.58 records/s | 3,437 MiB |
| Multimodal | 360 | 98.70 s | 89.31 s | 4.03 records/s | 3,437 MiB |

Both QA runs used the same Qwen model and generation settings. The multimodal prompt path took approximately 10.84 seconds longer in total and generated 0.55 fewer records per second, while peak allocated memory remained effectively unchanged. Earlier V4.2 measurements recorded approximately 14.91 ms mean single-image encoding, 1.73 ms BM25 retrieval, 0.28 ms cached reranking, and a 16.93 ms warm paired-request estimate.

These measurements show that the final system was feasible on a laptop GPU. They do not provide complete end-to-end production latency, energy consumption, or deployment cost.

The similar peak memory values are expected because both downstream runs load the same generator and checker. Multimodal retrieval mainly changes which report enters the prompt and adds cached reranking operations; it does not require the language model to ingest image pixels. The additional total time is therefore modest relative to generation.

Generation dominates the measured runtime. Even the earlier uncached image encoding estimate is small compared with approximately 79-89 seconds for batched generation over 360 records. Optimizing retrieval alone would have limited effect on total batch time. Quantization, batching, prompt length, output length, and model size are more likely to determine deployment throughput.

The cost analysis also highlights the benefit of caching. Candidate report embeddings and case image vectors can be computed once and reused across questions and shuffled controls. Without caching, the 100 alignment runs would repeatedly invoke the vision-language encoder and greatly increase cost. The frozen artifacts distinguish cached reranking from image encoding so these operations are not conflated.

No cost value should be generalized directly to another machine. GPU model, driver, CUDA version, thermal conditions, model cache state, and storage all affect timing. The reported values establish feasibility and provide a reproducible reference point, not a hardware-independent performance guarantee.

### H.7 Results Summary

V5 established four quantitative conclusions. Indication was the strongest single retrieval signal. Correctly aligned image reranking provided additional target-ordering and proxy-answer gains beyond indication text. Shuffled images did not reproduce the correct-alignment result. The retrieval improvement transferred to final QA Token-F1 but coincided with lower automated support.

The researcher-reviewed analysis explained why aggregate metrics moved differently. Some rank improvements stopped short of Top-1, some wrong-report answers remained locally grounded, some correct-report drafts lost content during verification, and some support-rate decreases reflected template filtering rather than substantive answer loss.

Taken together, the results form a sequential evidence chain. The question-only condition establishes the patient-scope ambiguity. The indication ablation establishes the strength of a text shortcut. Correct-image reranking adds a smaller but statistically supported improvement in target ordering and proxy answer availability. Shuffled images fail to reproduce that improvement, supporting alignment specificity. The non-oracle QA path then shows transfer to generated reference overlap. Finally, the support trade-off and qualitative review demonstrate that retrieval improvement does not remove downstream grounding problems.

This sequence is stronger than presenting only the best multimodal score. Each claim has a corresponding comparison and boundary. The study can claim incremental aligned-image value for retrieval, but not image-only case identification. It can claim transfer to automatic answer overlap, but not clinical correctness. It can describe possible verifier over-rejection in reviewed cases, but not estimate verifier error prevalence.

The negative and mixed results contribute to the research value. Weak Hit@1 evidence, high revision rates, and declining automated support reveal where future work should focus. They also reduce the risk that the dashboard is interpreted as a finished clinical product. The final system is demonstrable and auditable precisely because its uncertainties remain visible.

## Appendix I: Final V10/V12/V16 Artifact Index

- Final result registry: `docs/FINAL_RESULTS_REGISTRY.md`
- V10 technical freeze: `docs/V10_TECHNICAL_FREEZE.md`
- V12 development protocol and results: `docs/V12_PILOT_PROTOCOL.md`, `docs/V12_PILOT_RESULTS.md`
- V16 development protocol: `docs/V16_DEVELOPMENT_PROTOCOL.md`
- V16 confirmation protocol: `docs/V16_CONFIRMATION_PROTOCOL.md`
- V16 retrieval result: `docs/V16_RETRIEVAL_CONFIRMATION_RESULTS.md`
- V16 generation result: `docs/V16_GENERATION_CONFIRMATION_RESULTS.md`
- V16 reference-completeness deviation: `docs/V16_PROTOCOL_DEVIATION_REFERENCE_COMPLETENESS.md`
- V16 final technical freeze: `docs/V16_FINAL_TECHNICAL_FREEZE.md`
- V16 aggregate evaluations: `data/splits/v16/v16_impression_gate_vs_base_confirmation.json`, `v16_impression_gate_clinical_metrics_confirmation.json`, `v16_impression_gate_standard_nlg_confirmation.json`

## Appendix J: Final Reproduction Entry Points

**Repository:** https://github.com/yzy542968-jpg/wqf7023-medical-rag

**Final integration branch at manuscript build:** `final-qa-study`

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -q
& ".\.venv\Scripts\python.exe" scripts\evaluate_v16_paired_rows.py --help
& ".\.venv\Scripts\python.exe" scripts\evaluate_v16_standard_nlg.py --help
streamlit run app.py --server.port 8504
```

Large model weights, embeddings, report-derived rows, generations and image pixels remain local. Aggregate version-controlled files provide counts, metrics and hashes without redistributing source text.

## Appendix K: Prompt and Provenance Contract

The model receives a target image, indication, question and clearly labelled other-case historical evidence. It produces only a concise target answer. Python attaches provenance from retrieved `case_id`, section and source hashes. Historical evidence is never described as the target patient's report. Contract validity does not imply clinical correctness.

## Appendix L: Version and Release Boundary

V10 remains the frozen foundation and alignment study. V11 and V13-V15 are development/mechanism evidence. V12 is the final learned retrieval method. V16 is the final integrated held-out method confirmation and does not overwrite earlier artifacts. Post-freeze work is limited to deterministic audits, manuscript integration, dashboard presentation and release packaging.

## Appendix M: Human and External Evaluation Status

Independent blinded radiologist evaluation was not conducted and remains Future Work. The blank protocol and reviewer materials are retained without invented scores. Authorized MIMIC-CXR external validation was not executed; the multi-terabyte dataset and access requirements remain outside this thesis. No external result is claimed.

## Appendix N: Reference-Completeness Deviation

The V16 protocol expected non-empty Findings and Impression references. The instantiated frame contained 81 cases with empty Findings, producing 243 empty rows per arm. The all-row primary analysis is retained. The non-empty-reference sensitivity remains positive. Full counts, impact analysis and corrective actions are recorded in `docs/V16_PROTOCOL_DEVIATION_REFERENCE_COMPLETENESS.md`.

## Appendix O: Post-Primary Structured Final-QA Validation Extension

After the V12/V16 primary study was frozen, a separate protocol-governed development extension mapped Rad-ReStruct's structured questions onto the duplicate-cluster-disjoint OpenI roles. A 384-step q/v QLoRA adapter was selected using Train and Calibration only. Full Validation then evaluated 358 cases, 17,864 questions and four frozen conditions: image-only question answering without history, deterministic random history, a Top-1 image-neighbour report, and question-conditioned evidence from Top-3 image neighbours. Test remained inaccessible.

The fine-tuned generator achieved strong question-level performance. Exact answer-set accuracy was 0.84970 without history, 0.87836 with random history, 0.87897 with Top-1 image-neighbour history and 0.86324 with question-conditioned Top-3 evidence. Single-choice accuracy reached 0.90140 for the Top-1 condition. However, the prespecified report-level primary metric did not improve. Supported-label macro-F1 was 0.30984 without history, 0.30565 with random history, 0.29334 with Top-1 image-neighbour history and 0.29226 with question-conditioned evidence. Top-1 history minus no history was -0.01650 with 95% paired case-bootstrap interval [-0.02049, -0.00235]; question-conditioned evidence minus no history was -0.01758 [-0.02172, -0.00305].

The advancement rule therefore failed and no Test confirmation was run. This extension shows that structured QA accuracy can be high while complete report-vector macro-F1 degrades, and that relevant historical context must outperform both no-history and random-history controls before it can be credited as a RAG contribution. It is retained as a negative/mixed Validation result and does not replace the frozen V12/V16 primary claims. The complete decision record is `docs/FINAL_QA_VALIDATION_DECISION_RECORD.md`; large per-question rows remain local under repository policy.
