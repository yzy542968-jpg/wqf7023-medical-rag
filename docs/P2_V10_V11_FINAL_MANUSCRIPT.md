# Retrieval-Augmented Medical Question Answering over Paired Radiology Images and Reports

## Abstract

This research investigates a bounded multimodal retrieval-augmented generation workflow for a new chest-radiograph case whose final report is not available at inference. The system receives one or two target chest images, an available clinical indication and a natural-language question. It retrieves other-case historical image-report pairs, ranks them with textual, visual and report-derived fact signals, selects question-relevant evidence within each retrieved case, generates a concise answer with MedGemma 1.5, and attaches case, section and fact provenance deterministically. Historical reports are treated as analogies rather than as facts about the target patient.

The primary V10 study used 3,851 OpenI/IU-Xray cases. Exact and near-duplicate report clustering produced 3,013 clusters before allocation to Train, Calibration, Validation and Test. The frozen split contained 2,510, 383, 384 and 574 cases respectively; 568 Test cases were technically eligible. Five retrieval systems were compared on the same Train-only historical bank. The fact-aware multiview R5 ensemble achieved nDCG@10 0.36007, compared with 0.34905 for the frozen R4 nine-feature reranker. The paired case-bootstrap difference was +0.01103, 95% CI [+0.00770, +0.01441]. Correctly aligned images also exceeded all 100 deterministic fixed-point-free shuffled-image assignments: aligned nDCG@10 was 0.36007 versus shuffled mean 0.24963, with plus-one Monte Carlo p=0.00990. These results support an alignment-specific retrieval contribution within the same source dataset.

Downstream question answering compared target-image generation without history, R4 whole-report RAG, R5 RAG, and a calibrated selective-history condition. R5 RAG improved Token-F1 over no-history generation by +0.05978, 95% CI [+0.05114, +0.06860], but its incremental Token-F1 advantage over R4 whole-report RAG was only +0.00167, CI [-0.00347, +0.00683]. Better retrieval therefore transferred to automated report-reference consistency relative to no history, but did not confirm generator-level superiority over the previous multimodal retriever.

V11 was retained as a development-only mechanism extension. A full-bank relevance audit showed that candidate generation remained a bottleneck. Reciprocal-rank-fusion candidate generation improved nDCG@10 and relevant-case presence, particularly at K=200, but remained exploratory. A clean 48-case, 432-generation MedGemma experiment compared whole-report, sentence-only and case-to-fact evidence. Case-to-fact reduced mean evidence characters by 63.4% and mean input tokens by 32.4%, while maintaining 100% answer-contract and provenance validity. Its Token-F1 advantage over whole-report evidence was +0.02195, 95% CI [-0.00026, +0.04302], and complete F1RadGraph differed by +0.01395, CI [-0.00691, +0.03442]. The intervals crossed zero, so the result supports efficiency and auditability rather than confirmed answer-quality improvement. A second frozen 96-item planner wording set achieved 0.9167 accuracy, 0.9196 macro-F1 and 1.0000 indication invariance without post-evaluation rule changes.

The study concludes that correctly paired images can improve report-derived similar-case retrieval, fact-aware ranking provides a small but confirmed retrieval gain, and historical context can improve automated answer-reference consistency. It also shows that stronger retrieval does not guarantee stronger grounding or final QA, fact selection cannot repair a missing or wrong case, and proxy confidence is not clinical calibration. No physician-adjudicated correctness, clinical safety or external patient-level generalization is claimed.

**Keywords:** multimodal retrieval-augmented generation; chest radiography; similar-case retrieval; medical question answering; MedSigLIP; MedGemma; RadGraph; provenance; shuffled-image control; evidence selection

## Declaration of evidence boundary

All reported primary and development metrics are automated and retrospective. The processed OpenI source supports case-ID and duplicate-cluster separation but does not expose a reliable patient identifier for independent patient-level verification. Independent radiologist review and external validation remain Future Work. No reviewer ratings or external results are fabricated.

# Chapter 1: Introduction

## 1.1 Background

Large language models can generate fluent answers to medical questions, yet fluency does not establish whether an answer is supported, correctly attributed, or clinically safe. Retrieval-augmented generation (RAG) responds to part of this problem by placing retrieved evidence in the model context before generation. In principle, retrieval can improve factual coverage, expose provenance, and allow unsupported claims to be traced to a specific evidence source. In practice, RAG remains a multi-stage system. Query formulation, candidate retrieval, multimodal fusion, answer generation, semantic checking, and abstention can each fail independently. A final answer can therefore appear coherent even when the evidence is irrelevant, belongs to another case, or has been interpreted incorrectly.

Radiology makes this distinction especially important. A chest-radiograph examination links a clinical indication, one or more images, findings, and an impression. When a new patient is imaged, the formal findings and impression are not yet available to an automated support system. The available query is instead composed of the target image, pre-report clinical history or indication, and a question. A useful retrieval system should search a historical archive for clinically similar other-patient cases rather than recover a report that already belongs to the target patient. Retrieved reports may provide analogies, terminology, and patterns that help a multimodal generator answer the question, but they are not proof that the same finding is present in the target patient.

The OpenI/IU-Xray collection provides de-identified chest-radiograph examinations with linked reports and images. It is sufficiently large for a controlled, local study and permits the construction of a fixed historical bank. Modern biomedical vision-language models such as MedSigLIP can map chest images and report text into related representation spaces, while MedGemma can condition generation on both a target image and textual evidence. These components make it possible to test a more realistic research question than simple paired-report recovery: whether an unseen target image can retrieve clinically similar historical image-report pairs and whether those retrieved reports improve question answering relative to the same generator without retrieval.

This thesis follows an iterative research programme. Early V5-V7 experiments used controlled paired-case retrieval to expose patient-scope ambiguity, indication shortcuts, image-alignment effects, downstream grounding failures, and the limits of naive or adaptive score fusion. Those studies remain reproducible preliminary evidence, but their closed-set task is not treated as the final clinical scenario. V9 changed the construct by removing the target report from the candidate bank and hiding it at inference. V10 then added duplicate-cluster-disjoint confirmation over a fixed Train-only historical bank. The final primary claims come from V10 Test; V11 contributes development-only mechanism evidence.

## 1.2 Problem Statement

Text-only medical RAG assumes that the query contains enough language to retrieve useful evidence. That assumption is weak when the question is generic, for example, "What are the main radiographic findings?" Before the report exists, the question itself carries little patient-specific information. A referral indication may provide symptoms or suspected disease, but it is incomplete and may also create a lexical shortcut. The chest image contains the primary patient-specific signal. The technical problem is therefore to combine image-image similarity, image-report compatibility, and indication-question text retrieval without allowing a weak channel to degrade a stronger one.

The problem is not solved merely by adding scores. The V6 and V7 preliminary studies showed that fixed or query-conditional fusion may not outperform the strongest individual component. Score distributions differ across retrieval channels, and one modality may be uninformative for a particular query. A learned reranker must therefore be evaluated against the strongest frozen component, not only against BM25 or an arbitrary equal-weight baseline. Its training labels must remain offline and report-derived; target labels, target report text, and answer references cannot become inference features.

A second problem concerns the definition of relevance. OpenI does not include physician judgments of pairwise clinical similarity for every query and historical case. Exact target-report retrieval is also inappropriate because the target report is intentionally absent from the bank. This research operationalizes graded relevance from hidden target-report annotations: active abnormal label overlap and RadGraph entity-relation overlap. This enables nDCG evaluation over many candidates while keeping the limitation explicit. The resulting relevance measure estimates report-derived similarity; it is not a replacement for physician adjudication.

A third problem is whether retrieval improvement transfers to the final answer. A stronger ranking does not guarantee that a generator uses the evidence correctly. Historical cases may distract the generator, dominate the target image, or encourage unsupported analogy. Conversely, an automated checker may remove useful content because natural-language inference is imperfect. Retrieval, answer-reference consistency, output validity, historical-evidence support, revision, and abstention must therefore be reported separately.

The central problem is summarized as follows:

> How can a new-patient chest image, clinical indication, and medical question be used to retrieve other-case image-report evidence and improve auditable multimodal question answering without exposing the hidden target report or overstating clinical validity?

This formulation separates four levels of evidence. First, a historical case may be similar according to report-derived labels and facts. Second, the generator's historical-support statement may be entailed by the cited report. Third, the final answer may overlap the hidden target reference. Fourth, the answer may be clinically correct for the patient. The study measures the first three with different limitations and does not claim to establish the fourth.

## 1.3 Research Aim

The aim is to develop and critically evaluate a reproducible multimodal RAG system that retrieves similar other-patient chest-radiograph cases and uses their reports as explicitly labeled historical analogies for new-case medical question answering.

## 1.4 Research Objectives

The objectives are:

1. Construct a deterministic Train/Validation/Test study from all 3,851 paired OpenI cases while preserving image-report linkage and transparent report-indexed spectrum labels.
2. Build one shared Train-only historical bank that excludes the target study and keeps target reports, labels, facts, and references outside the inference pipeline.
3. Define report-derived graded relevance from active labels and RadGraph facts without rewarding agreement on absent abnormalities.
4. Compare BM25, MedSigLIP image-image retrieval, MedSigLIP image-report retrieval, validation-selected fixed fusion, and a small learned multimodal reranker over the same candidate bank.
5. Test whether any learned retrieval gain exceeds the strongest frozen component under case-grouped bootstrap inference.
6. Test alignment dependence with 100 complete, fixed-point-free shuffled-image recomputations.
7. Evaluate whether retrieved historical reports improve MedGemma question-answer reference consistency over the same target-image generator without retrieval.
8. Implement a bounded agent that can check historical-support claims, perform at most one backup retrieval, revise unsupported evidence, or abstain from historical support.
9. Preserve protocols, manifests, hashes, model revisions, runtime evidence, tests, and a deterministic qualitative review pack for reproducibility.
10. State the limits of retrospective, same-source, automated evaluation and reserve independent clinician adjudication and external validation for future work.

## 1.5 Research Questions

**RQ1.** Does the correctly aligned target chest image improve report-derived similar historical-case retrieval over text-only and other frozen retrieval baselines?

**RQ2.** Does the aligned-image system exceed deterministic shuffled-image controls, indicating that the gain depends on the correct image-case pairing rather than a generic visual prior?

**RQ3.** Does fact-aware multiview retrieval improve report-derived graded relevance over the frozen nine-feature multimodal reranker?

**RQ4.** Does retrieval improvement transfer to downstream MedGemma answer-reference consistency, and can within-case fact selection reduce context and truncation without sacrificing provenance?

## 1.6 Research Contributions

The first contribution is a clinically interpretable new-case task contract. The target report is unavailable at inference and cannot leak into retrieval, prompting or answer generation. The system retrieves reports belonging to other cases and labels them as historical analogies. This distinction prevents a paired dataset from collapsing into target-report lookup and makes case ownership part of the evidence model.

The second contribution is a duplicate-cluster-disjoint multimodal retrieval study with an alignment-specific negative control. Exact and near-duplicate report clustering occurs before allocation. BM25, image-image, image-report, nine-feature and fact-aware multiview systems rank a common historical bank. One hundred deterministic fixed-point-free shuffled-image assignments recompute the visual state and test whether the improvement depends on the correctly paired image.

The third contribution is a fact-aware, provenance-preserving evidence workflow. R5 incorporates question-conditioned RadGraph and multiview signals. V11 then separates case retrieval from within-case sentence or fact selection, retaining case ID, report section, unit type and source hash. The design reduces irrelevant context without combining anonymous facts across patients.

The fourth contribution is an evaluation and reproducibility framework that preserves mixed results. Retrieval, generation, automated semantic overlap, structured-output validity, provenance, latency, GPU memory and uncertainty intervals are reported separately. V10 frozen hashes remain unchanged; V11 is explicitly development-only. Human review and external validation are not replaced with proxy metrics.

## 1.7 Scope and Boundaries

The completed research is restricted to chest radiographs and reports from OpenI/IU-Xray. It is a retrospective technical study, not a prospective clinical trial. The source design describes one study per patient, but reliable released subject identifiers are unavailable in the processed artifact. The thesis therefore claims case-ID and duplicate-cluster disjointness, not independently verified patient-level separation.

Report-indexed normal and abnormal strata are derived from the dataset `problems` field and are not new clinical labels. Graded relevance is constructed from report labels, hidden report sections and RadGraph facts. These signals support controlled ranking comparisons but are not physician judgments of clinical similarity. Token-F1 and F1RadGraph measure automated consistency with hidden report text and do not establish diagnosis, safety or patient benefit.

The V10 Dashboard is a local research prototype. It retrieves the top-ranked similar historical cases from an indexed corpus; it does not find the target patient's own report. Confidence values describe retrieval reliability under an operational proxy label. Independent clinical scoring, external patient-disjoint validation, fairness, prospective workflow effects and deployment safety remain outside the completed evidence.

## 1.8 Conceptual Framework

The final workflow contains eight auditable stages:

```text
target chest image(s) + indication + question
        -> question intent and query representation
        -> candidate historical-case generation
        -> multimodal fact-aware reranking
        -> Top-3 other-case historical reports
        -> within-case sentence/fact selection
        -> bounded MedGemma answer
        -> deterministic case/section/fact provenance
        -> retrieval-confidence research signal or no-reliable-history state
```

Evaluation follows the same decomposition. Retrieval nDCG and MRR test ranking. The shuffled-image control tests alignment dependence. Token-F1 and F1RadGraph test automated answer-reference consistency. Context length, token ceilings, schema validity and provenance test engineering behavior. The confidence audit tests selective separation under proxy labels. No single metric is treated as a substitute for clinical correctness.

## 1.9 Thesis Organization

Chapter 2 synthesizes medical VQA, multimodal RAG, radiology image-report retrieval, fact-aware evidence, grounding and confidence research. Chapter 3 defines the V10 primary protocol and the bounded V11 development extension. Chapter 4 reports retrieval, alignment, QA, evidence-selection, planner, candidate-generation and runtime results. Chapter 5 interprets positive, negative and mixed findings. Chapter 6 answers the four research questions and states final limitations and Future Work. Historical frozen studies and detailed reproduction artifacts are retained in the appendices rather than presented as the main narrative.

# Chapter 2: Literature Review

This chapter synthesizes the literature by research theme rather than as a paper-by-paper catalogue. Earlier controlled-study references are retained where they motivate the final V10/V11 design.

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

The final V10 system uses the pinned MedSigLIP-448 revision for image-image and image-report features. The foundation encoder remains frozen; only compact retrieval components are trained. This separates representation reuse from task-specific ranking and keeps the computation feasible on the available GPU. Earlier BioViL-T studies are retained as formative history, not described as the final encoder.

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

Benchmark construction can introduce additional shortcuts. Generic repeated questions make text retrieval weakly identified, indications can encode disease hints, and near-duplicate reports can leak across partitions. V10 clusters exact and near-duplicate reports before allocation, uses a common Train-only bank and reports image-only, image-report and BM25 components separately. Patient-level separation remains unverifiable because reliable subject identifiers are absent from the processed source.

The relevance construct is another validity boundary. Report labels and RadGraph facts enable deterministic graded qrels, but they are not physician judgments. Post-hoc sensitivity across combined, label-only and fact-only definitions is therefore reported as a construct audit rather than a replacement endpoint.

## 2.9 Agentic and Auditable RAG Workflows

Agentic RAG can plan, retrieve, rerank, generate, verify or abstain. The term should be used carefully: a deterministic workflow is not learned clinical reasoning, and an automated checker is not an independent physician. Research value depends on bounded actions, inspectable state and honest failure handling rather than on the number of named agents.

The final system is agent-like only in this bounded engineering sense. A deterministic planner identifies question intent, the retriever ranks historical cases, the evidence selector preserves case ownership, MedGemma generates a concise target-image answer, deterministic code attaches provenance, and a calibrated gate can withhold historical support. Each stage has an explicit contract and can be evaluated separately.

Direct Python modules are used instead of LangChain because the experiment requires stable prompts, frozen transitions and artifact-level reproducibility. This choice does not imply that orchestration libraries are inferior; it avoids introducing hidden retries, memory or tool-selection behavior into a controlled study. The Dashboard exposes the retrieved case IDs, evidence units, confidence boundary and provenance without claiming autonomous clinical agency.

Open-ended agents may be useful in future interactive systems, but they require separate evaluation of tool choice, prompt reformulation, retry policy and safety. The current contribution is an auditable multimodal RAG workflow, not a general autonomous medical assistant.

## 2.10 Comparative Synthesis of Design Alternatives

Several technically plausible architectures answer different research questions. An LLM-only system tests parametric medical knowledge but cannot expose a case-specific evidence path. A direct vision-language model can answer from the target image, but it does not test historical-case retrieval and can confound visual recognition, medical reasoning and language generation. Both remain useful generation baselines, but neither isolates the retrieval contribution studied here.

Text-only RAG preserves an auditable document path and provides a necessary baseline. BM25 is transparent and fast, while modern dense text retrieval can reduce vocabulary mismatch. However, the indication and question may be generic or incomplete before the report exists. A text retriever can therefore overuse indication shortcuts or return lexically similar reports whose images differ from the target case. V10 retains BM25 as R0 rather than treating it as an intentionally weak comparator.

Chunk-level retrieval offers fine-grained matching and short prompts, but radiology chunks can lose case ownership, split negation or detach findings from impressions. Whole-report retrieval preserves a coherent historical unit and was the development-selected V10 QA policy. V11 adds sentence and fact selection only after case retrieval, so every unit retains its owning `case_id`, section, unit type and source hash. This ordering avoids anonymous cross-patient fact assembly.

Image-only and image-report retrieval test complementary visual relations. Image-image similarity asks whether the target radiograph resembles a historical radiograph. Image-report compatibility asks whether the target image aligns with the language of a historical report. Neither relation alone guarantees clinically useful evidence. The R4 comparator therefore combines normalized text, image-image, image-report and rank features, while R5 adds question-conditioned fact signals and multiview representation. All systems rank the same Train-only historical bank.

A fully fine-tuned multimodal foundation model might achieve stronger benchmark performance, but it would introduce additional choices about negative sampling, optimization, checkpoints and model adaptation. The final study instead freezes the MedSigLIP and MedGemma foundation models and trains only compact retrieval components. This makes the incremental R5-minus-R4 comparison more attributable and feasible on the available 8 GB GPU.

An agent framework or LangChain could orchestrate retrieval, planning, generation and checking. Neither is required for scientific validity. The implemented planner, evidence selector, generator contract and confidence gate are direct modules with explicit inputs and outputs. This keeps retries, abstention, provenance and failure states inspectable without implying autonomous clinical agency.

The selected design follows four principles. First, retrieval units preserve evidence ownership. Second, multimodal gain is compared with strong individual components and complete wrong-image controls. Third, target-image answering and historical provenance are separated. Fourth, retrieval, generation, structure, confidence and clinical validity are evaluated as different layers. These principles motivate the final research gap.

## 2.11 Similar-Case Multimodal RAG and Final Research Gap

The closest line of work combines chest-image retrieval with report generation. CXR-RePaiR uses a contrastive image-to-report retriever to construct reports from retrieved exemplars. X-REM adds coarse retrieval, learned image-text matching and an NLI filter. FactMM-RAG mines factual report pairs with CheXbert and RadGraph to train a fact-aware multimodal retriever. RA-RRG retrieves clinically important key phrases to condition report generation, while MedProbCLIP introduces probabilistic image-report embeddings, calibration and risk-coverage evaluation. These systems establish that historical image-report pairs can support image-conditioned language generation and that retrieval reliability deserves explicit measurement.

They do not directly resolve the question-conditioned new-case setting examined here. At inference, the target report is hidden; the target image, available indication and medical question must retrieve other-case historical evidence. The output must distinguish observations about the target image from analogies drawn from retrieved reports. This task requires both clinically useful ranking and explicit evidence ownership.

Five connected gaps remain. First, many radiology retrieval studies focus on report generation rather than a user's question. Second, multimodal improvements are not always compared with strong individual visual and text components. Third, aligned-image gains are rarely challenged by complete wrong-image recomputation. Fourth, local support for a retrieved report is often conflated with correctness for the target case. Fifth, duplicate leakage, relevance sensitivity, candidate recall, wording robustness and selective reliability can materially change conclusions but are often treated as implementation details.

The final research gap is therefore not whether RAG can be used in radiology or whether a newer model can replace an older one. It is whether a correctly paired target image improves retrieval of clinically related other-case reports under duplicate-aware splitting and shuffled-image controls; whether fact-aware reranking adds value over a strong frozen multimodal comparator; whether retrieval gains transfer to automated answer-reference consistency; and whether the complete workflow preserves case, section and fact provenance while retaining negative and mixed results.

V10 addresses that gap through a cluster-disjoint same-source confirmation, a common Train-only historical bank, report-derived graded relevance, deterministic fixed-point-free image shuffling, bounded local generation and explicit claim boundaries. V11 investigates residual mechanisms - candidate recall, within-case evidence compression, planner wording and retrieval confidence - on development data only. Neither automated relevance nor report-reference overlap is presented as physician-adjudicated clinical correctness.

# Chapter 3: Methodology

## 3.1 Research Design and Version Boundary

The research used a staged retrospective experimental design. V10 is the final confirmatory automated study and supplies the primary evidence. Its split, candidate bank, models, prompts, metrics, shuffled-image assignments, bootstrap plan and failure policy were frozen before Test execution. V11 is a development-only extension conducted on V10 Train and Validation data. It investigates candidate generation, within-case evidence compression, question planning and bounded output contracts without reopening V10 Test selection or changing any V10 result.

Earlier project versions are retained as formative work. They established the original RAG implementation, revealed cross-case contamination, introduced paired image-report retrieval, compared updated encoders and tested adaptive fusion. They are not presented as eleven independent experiments. The final thesis is organized around one question: whether a new target chest image can help retrieve useful historical cases and support an auditable answer when its final report is hidden.

The protocol chronology is part of the methodology. V10 confirmation was committed before Test execution, and the release tag records immutable aggregate artifacts. The clean V11 48-case generation sample was selected deterministically from Validation without result-driven replacement. The second planner wording set was committed after the planner was frozen and before its first evaluation. V11 observations were not used to retune V10.

## 3.2 Operational Task Contract

At inference, the available target-case information is one or two chest radiographs, an optional clinical indication and a natural-language question. The final target report is unavailable. The retrieval bank consists only of Train cases, each containing image references, indication, findings, impression and derived report features. The desired output is a concise target-image answer accompanied, when reliable, by explicitly labelled historical analogies and deterministic provenance.

The system must not claim that a retrieved report belongs to the target patient. It must not merge free-floating sentences from different patients into an anonymous context. Case retrieval therefore precedes fact selection. Each selected sentence or RadGraph fact retains `case_id`, `section`, unit type and `source_sha256`. If retrieval confidence falls below a frozen threshold, the system may show candidates for audit but must not present them as reliable support.

This contract separates three meanings of evidence. The target image is direct input about the current examination. The hidden target report is an evaluation reference and is not available to the system. Retrieved reports are analogical historical context. A generated answer can be faithful to a historical report while still being inappropriate for the target case; provenance is necessary to expose that distinction.

## 3.3 Data Source and Case Construction

The source was the public OpenI/IU-Xray chest-radiograph collection. The processed artifact contained 3,851 case records and linked image views. Each case preserved a stable project case ID, indication, findings, impression, report-indexed problems and local image references. Twenty-five cases lacked usable findings and impression text; technical eligibility checks were performed before model execution rather than silently replacing failed cases after outcomes were observed.

Normality strata were operational. `problems == normal` was classified as report-indexed normal. Non-empty clinical problem values excluding `normal` and `no indexing` were classified as report-indexed abnormal. `no indexing` was treated as report-index indeterminate. These labels were used for spectrum description and deterministic sampling, not as physician-adjudicated disease status.

The processed artifact did not expose a reliable subject identifier. Case-ID disjointness could be verified, and duplicate report clusters could be kept within one partition. Patient-level independence could not be independently verified. This limitation is stated consistently in the protocol, Results, Discussion and release documentation.

## 3.4 Duplicate-Clustering and Frozen Split

Near-duplicate reports threatened internal validity because a case in Test could be almost textually identical to a Train report. V10 therefore clustered exact and near-duplicate report representations before partitioning. The 3,851 cases formed 3,013 clusters. Entire clusters, rather than individual cases, were assigned deterministically to Train, Calibration, Validation and Test.

The frozen partition contained 2,510 Train, 383 Calibration, 384 Validation and 574 Test cases. The technically eligible historical bank contained 2,506 Train cases. Six Test cases had unusable empty-report RadGraph records and were handled under the frozen data-integrity rule without replacement, leaving 568 Test cases. No duplicate cluster crossed partitions. The split file and confirmation configuration were hashed, and their hashes are reproduced in the technical freeze.

Train supported retriever fitting and historical-bank construction. Calibration fitted the retrieval-confidence model. Validation supported development selection, component audits and V11 mechanism experiments. Test was reserved for V10 confirmation. V11 did not instantiate a new confirmation cohort.

## 3.5 Report-Derived Relevance

Exact target identity is not the desired relevance label because the bank intentionally contains other cases. V10 used a graded report-derived construct that combines clinically active labels and RadGraph fact similarity while avoiding credit for shared negative labels. Relevance at or above 0.5 defined operationally relevant cases for MRR and Hit@k. nDCG@10 retained graded information and served as the primary retrieval metric.

This relevance is an automated proxy. It approximates whether two cases share report-described findings, anatomy, polarity and related characteristics. It cannot determine whether a radiologist would regard a retrieved case as a useful analogy. V11 exposed the proxy structure more explicitly through lesion type, anatomy, severity, polarity, uncertainty, indication and report-label components. Missing components were represented through availability rather than silently counted as agreement.

The V11 full-bank audit corrected a potential ideal-list bias. Relevance was computed against the complete 2,510-case Train bank before evaluating a Top-K candidate list. This prevents a system from receiving an inflated nDCG merely because its ideal list was constructed only from items it had already retrieved.

## 3.6 Retrieval Baselines and Candidate Generation

R0 was BM25 text retrieval over the indication and question. It provided a transparent lexical baseline and exposed text shortcuts. R1 used MedSigLIP image-image similarity between the target image representation and cached historical image representations. R2 used MedSigLIP image-report compatibility. R4 was the frozen nine-feature reranker combining normalized text, image-image and image-report features and rank signals.

V11 separately audited candidate generation. BM25, MedCPT text retrieval and MedSigLIP image retrieval each produced a ranked list. Reciprocal rank fusion with constant 60 combined their ranks, and the union was truncated to K=100 or K=200. This experiment was conducted only on Validation and was not promoted into V10. The purpose was to identify whether relevant cases were absent before evidence compression began.

For any training-oriented audit, a query whose positive candidate was outside the shortlist remained a retrieval failure for evaluation but could not provide a valid positive-negative training pair. Such queries were not deleted from evaluation. This protects against overstating a reranker's capability when candidate generation has already failed.

## 3.7 Fact-Aware Multiview Reranking

R5 extended R4 with question-conditioned RadGraph fact features and a learned multiview image representation. One or two target views were encoded with the pinned MedSigLIP-448 revision. The multiview attention component produced a query image vector, while the fact index summarized query-candidate compatibility. Five frozen seed models formed the final ensemble.

R5 ranked the same Train bank as the baselines. The primary V10 comparison was R5 minus R4 on case-grouped nDCG@10. The distinction is important: the question was not whether any new model beats BM25, but whether the additional fact and multiview mechanisms add value over an already strong frozen multimodal reranker.

A frozen-checkpoint Validation 2x2 audit crossed R4/R5 with mean-image/attention-image representations. It estimated descriptive main contrasts for the fact-aware reranker, the attention view and their interaction. Because this was not a randomized causal factorial experiment, the contrasts are interpreted as mechanism diagnostics rather than causal component effects.

## 3.8 Shuffled-Image Alignment Control

An image-assisted system can appear multimodal even if the image contributes only a generic prior or if textual indication dominates. V10 therefore compared the correctly aligned image condition with 100 deterministic unique fixed-point-free wrong-image assignments. Every Test case received the complete image-view set of another Test case, while its indication, question and evaluation reference remained unchanged.

For each assignment, visual similarities, normalized features, multiview representation and R5 scores were recomputed. The aligned score was compared with the full shuffled distribution, and a plus-one Monte Carlo p-value was reported. This control supports an alignment-specific interpretation but does not prove that the system independently diagnoses pixels.

## 3.9 Case-Level Retrieval and Within-Case Evidence Selection

V10 selected Top-3 cases and retained whole reports because whole-report evidence was the development-selected confirmation policy. V11 tested a second stage after case retrieval. `whole_report` preserved the available report context. `sentence_only` ranked report sentences within each selected case. `case_to_fact` allowed question-relevant sentences and RadGraph facts while retaining provenance.

The selector imposed deterministic budgets: no more than two primary units per case, six units overall and 1,200 characters in the V11 development configuration. It never moved a fact to another case or deleted its section identity. Selection used the indication, question and planner intent, but it did not inspect the hidden target answer.

Evidence compression can reduce irrelevant normal descriptions, prompt length and serialization pressure. It cannot recover a clinically useful case that was missing from the candidate list, and it cannot transform a wrong case into correct target evidence. The study therefore evaluates retrieval and evidence selection separately.

## 3.10 Deterministic Question Planner

The V11 planner is a small inspectable rule system with eight operational intents: presence, location, severity, comparison, device, uncertainty, insufficient information and summary. The non-empty question is the primary control signal. Indication is used only when the question is empty, preventing distractor indication terms from silently changing intent.

The original 64-item author-defined set assessed rule coverage during development. A second 96-item wording set was committed after the planner source was frozen. It contained 12 examples per intent and paired every question with a distracting indication. The planner was evaluated once on accuracy, macro-F1, confusion and indication invariance; no post-evaluation rule repair was permitted.

The planner does not diagnose the image. It selects evidence preferences and answer style. Both benchmarks are researcher-authored and do not constitute independent or clinician-authored natural-language validation.

## 3.11 MedGemma Generation and Provenance Contract

All V10 QA conditions used the same pinned `google/medgemma-1.5-4b-it` model and target image. G0 supplied no historical report. G1 supplied R4 Top-3 whole reports. G2 supplied R5 Top-3 evidence under the frozen V10 policy. G3 suppressed historical evidence below the frozen confidence threshold. The answer was limited to at most two complete sentences and 64 new tokens in V10.

Early one-pass JSON generation showed that asking a small quantized model to produce the answer, uncertainty, evidence IDs and metadata in one object increased truncation and malformed output. The final contract separates language generation from metadata. MedGemma produces only a concise answer. Deterministic code then attaches evidence state, supporting case IDs and case/section/fact provenance. This guarantees machine-verifiable provenance structure but does not guarantee that the generated answer is clinically correct or semantically complete.

The clean V11 generation audit used 48 deterministic Validation cases, balanced between 24 report-indexed normal and 24 report-indexed abnormal cases. Each case had three fixed question roles and three evidence policies, producing 432 generations. The target image, Top-3 cases, prompt family, decoding, model revision, maximum tokens and case selection were fixed across policies. No case was replaced after viewing output.

## 3.12 Retrieval-Confidence Research Signal

V10 fitted a retrieval calibrator only on Calibration. Features included top score, top1-top2 margin, component agreement, ensemble variance, evidence score, evidence redundancy, view count and question type. The frozen 80% target-coverage threshold was then applied to Test. The output estimates operational retrieval reliability under report-derived labels; it is not the probability that an answer is correct.

V11 also tested a simplified development gate after within-list score normalization. It accepted 99.83% of rows and therefore did not create meaningful selective separation. That negative result was retained. The Dashboard labels the value a `retrieval-confidence research signal`, and clinical probability calibration remains Future Work.

## 3.13 Metrics and Statistical Analysis

Retrieval metrics were nDCG@10, MRR at relevance 0.5 and Hit@1/5/10. The primary V10 effect was the case-level R5-minus-R4 nDCG@10 difference. Case-grouped bootstrap intervals used 10,000 resamples, preserving all question rows belonging to a case. The shuffled control used 100 deterministic assignments and a plus-one Monte Carlo p-value.

Generation metrics included Token-F1, entity, entity-relation and complete F1RadGraph, answer-contract validity, citation/provenance validity, evidence abstention, token-ceiling rate, input tokens, output tokens, latency and peak GPU allocation. F1RadGraph is automated graph overlap, not clinical adjudication. Empty V11 report references were retained and assigned zero overlap rather than deleted after inspection.

The primary V11 generation comparison was `case_to_fact - whole_report`, evaluated with paired case-grouped bootstrap intervals. V11 results remained development-only even if an interval excluded zero. Candidate-generation K=200, planner wording robustness and the 2x2 mechanism audit were prespecified or protocol-recorded secondary/exploratory analyses and were not used to modify V10.

## 3.14 Reproducibility, Hardware and Failure Policy

Experiments ran locally on an NVIDIA GeForce RTX 5070 Laptop GPU with 8,151 MiB reported memory, CUDA driver 591.91 and pinned model revisions. V10 QA peak allocated memory was approximately 5,311.6 MiB at batch size 8. The clean V11 generation run used batch size 4, peaked at approximately 4,261.6 MiB and required 773.8 seconds for 432 generations.

Configurations, aggregate summaries, split fingerprints, model revisions, scripts and tests are versioned. Large per-row generations, image pixels, model weights and protected source-derived text remain local. A technical execution failure could be rerun under the identical frozen configuration. A frozen-case data-integrity failure had to be recorded; silent case replacement and outcome-driven retuning were prohibited.

V10 artifact hashes were rechecked before final release. The repository was scanned for secrets, and human-review fields remained blank. These controls support computational reproducibility but do not convert a same-source automated study into clinical validation.

# Chapter 4: Results

## 4.1 Dataset Partition and Duplicate Control

The 3,851 source cases formed 3,013 exact or near-duplicate report clusters. Cluster-level allocation produced 2,510 Train, 383 Calibration, 384 Validation and 574 Test cases. No duplicate cluster crossed partitions. Six frozen Test identities failed the predefined usable-report/RadGraph integrity requirement and were excluded without replacement, leaving 568 technically eligible Test cases.

This split directly addresses the strongest weakness observed in the earlier full-source study, where 187 of 752 Test reports had cosine similarity at least 0.95 to a Train report. The V10 result is therefore not merely a sensitivity analysis after evaluation; duplicate control is built into the partition itself. Patient-level independence remains unverifiable because the processed source lacks reliable subject IDs.

## 4.2 Retrieval Baselines

| System | Partition | Cases | Question rows | nDCG@10 | MRR | Hit@1 | Hit@10 | Role |
|---|---|---:|---:|---:|---:|---:|---:|---|
| R0 BM25 | Test | 568 | 1,704 | 0.14076 | 0.07676 | 0.03228 | 0.16608 | baseline |
| R1 image-image | Test | 568 | 1,704 | 0.33485 | 0.30864 | 0.22535 | 0.44014 | baseline |
| R2 image-report | Test | 568 | 1,704 | 0.31760 | 0.25746 | 0.17077 | 0.41549 | baseline |
| R4 nine-feature | Test | 568 | 1,704 | 0.34905 | 0.31115 | 0.23005 | 0.44131 | primary comparator |
| R5 fact + attention | Test | 568 | 1,704 | 0.36007 | 0.31360 | 0.23826 | 0.44425 | primary system |

Image-only and image-report retrieval substantially exceeded BM25 under the report-derived metric, showing that the task was not solved by text alone. R4 combined the channels and exceeded each single baseline on nDCG@10. R5 achieved the highest nDCG@10, MRR, Hit@1 and Hit@10, although the absolute Hit@1 of 0.23826 shows that reliable top-ranked historical-case selection remains difficult.

The primary R5-minus-R4 nDCG@10 difference was +0.01103, 95% case-bootstrap CI [+0.00770, +0.01441]. The interval excluded zero under the frozen automated relevance construct. This answers RQ3 positively at the retrieval level without implying physician-perceived similarity.

## 4.3 Correctly Aligned Versus Shuffled Images

The correctly aligned R5 condition achieved nDCG@10 0.36007. Across 100 deterministic fixed-point-free shuffled-image assignments, the mean was 0.24963 and the range was [0.23621, 0.26404]. Every shuffled assignment scored below the aligned condition; the plus-one Monte Carlo p-value was 0.00990.

The control demonstrates that the result depends on the relationship between each target case and its correct image rather than only on the indication, question or a generic image prior. The magnitude of the shuffled drop also shows that the visual channel is not decorative. This remains an alignment result under report-derived relevance and should not be restated as standalone diagnostic accuracy from pixels.

## 4.4 Fact and Multiview Attribution Audit

| Frozen-checkpoint condition | Partition | Cases | Rows | nDCG@10 | Status |
|---|---|---:|---:|---:|---|
| R4 + mean image | Validation | 376 | 1,128 | 0.340255 | exploratory |
| R4 + attention image | Validation | 376 | 1,128 | 0.346206 | exploratory |
| R5 + mean image | Validation | 376 | 1,128 | 0.353909 | exploratory |
| R5 + attention image | Validation | 376 | 1,128 | 0.358540 | exploratory |

The fact-aware main contrast was +0.01299, 95% CI [+0.00883, +0.01729]. The attention-view contrast and fact-by-attention interaction intervals crossed zero. The result suggests that fact-aware features explain the more stable share of the R5 improvement, whereas the incremental attention contribution is smaller and uncertain. Because the audit uses frozen checkpoints rather than randomized component assignment, it is descriptive mechanism evidence.

## 4.5 Downstream QA

| Condition | Partition | Cases | Questions | Token-F1 | Complete F1RadGraph | Schema valid | Citation valid | Evidence abstention |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| G0 target image, no history | Test | 568 | 1,136 | 0.14942 | 0.08265 | 100% | 100% | 100.00% |
| G1 R4 whole-report RAG | Test | 568 | 1,136 | 0.20752 | 0.10507 | 100% | 100% | 0.00% |
| G2 R5 historical RAG | Test | 568 | 1,136 | 0.20919 | 0.11053 | 100% | 100% | 0.00% |
| G3 calibrated selective RAG | Test | 568 | 1,136 | 0.20498 | 0.11041 | 100% | 100% | 17.08% |

G2-minus-G0 Token-F1 was +0.05978, 95% CI [+0.05114, +0.06860]. Retrieved history therefore improved automated target-report consistency over the same target-image generator without history. Complete F1RadGraph improved by +0.02788, CI [+0.01977, +0.03639].

The stronger retrieval model did not yield a confirmed Token-F1 advantage over the previous R4 RAG system. G2-minus-G1 was +0.00167, CI [-0.00347, +0.00683]. Complete F1RadGraph was +0.00546, CI [+0.00005, +0.01089], while entity and entity-relation variants crossed zero. The correct interpretation is mixed: historical RAG helped relative to no history, but the incremental R5 retrieval gain did not consistently propagate across answer metrics.

G3 withheld historical evidence for 17.08% of answers but did not improve aggregate Token-F1. The threshold was not retuned after Test. This negative result shows that a retrieval-confidence model can behave as designed without improving average generation quality.

## 4.6 Structured Output and Token Ceilings

Deterministic assembly produced 100% schema and citation validity in all V10 conditions. This repaired the machine-readable provenance layer but did not eliminate generation truncation. Raw answer token-ceiling rates were 91.55% for G0, 64.79% for G1 and 68.31% for G2/G3 under the 64-token answer budget.

The distinction matters. A valid output object proves that fields and citations are structurally well formed. It does not prove that MedGemma completed every intended sentence or that the answer is clinically sufficient. The final system therefore reports token-ceiling behavior alongside contract validity.

## 4.7 V11 Evidence Compression and Generation

The clean V11 sample contained 48 Validation cases, evenly split between report-indexed normal and abnormal strata. Three question roles and three evidence policies yielded 432 generations.

| Evidence policy | Partition | Cases | Rows | Token-F1 (95% CI) | Complete F1RadGraph (95% CI) | Mean input tokens | Evidence chars | Token ceiling |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Whole report | Validation | 48 | 144 | 0.1312 [0.1033, 0.1612] | 0.0669 [0.0470, 0.0883] | 798.2 | 672.3 | 11.81% |
| Sentence only | Validation | 48 | 144 | 0.1451 [0.1165, 0.1757] | 0.0845 [0.0597, 0.1121] | 604.1 | 351.9 | 9.72% |
| Case-to-fact | Validation | 48 | 144 | 0.1531 [0.1260, 0.1814] | 0.0809 [0.0601, 0.1029] | 539.3 | 245.9 | 11.81% |

Case-to-fact reduced evidence characters by 63.4% and input tokens by 32.4% relative to whole reports. Answer-contract validity and deterministic provenance validity were 100% for all policies. Peak GPU allocation was approximately 4,261.6 MiB, and mean latency was 1.74-1.78 seconds per generation row.

The primary case-to-fact-minus-whole-report Token-F1 difference was +0.02195, CI [-0.00026, +0.04302]. Complete F1RadGraph differed by +0.01395, CI [-0.00691, +0.03442]. Both intervals crossed zero. The evidence supports compression and auditability, not confirmed answer-quality superiority. The case-to-fact token-ceiling rate also did not improve over whole reports, reinforcing the bounded interpretation.

## 4.8 Candidate-Generation Audit

At K=100, BM25 achieved full-bank qrel nDCG@10 0.5537, relevant-case presence 55.12% and relevant-item recall 12.00%. RRF achieved nDCG@10 0.5867 and presence 57.99%, but recall was slightly lower at 11.82%. Hybrid generation therefore improved ranking quality and the probability of retrieving at least one relevant case without uniformly improving all recall measures under the fixed budget.

At K=200, BM25 recall increased to 18.58% and presence to 61.46%. RRF reached recall 19.11% and presence 65.71%. The larger budget was more favorable to RRF but increased retrieval and reranking cost. K=200 remains an exploratory candidate-generation direction and does not replace the frozen V10 system.

The audit also exposed the dominant bottleneck: many report-derived relevant cases remained outside Top-100. Within-case fact selection cannot recover an item that candidate generation never surfaced.

## 4.9 Planner Robustness

The original 64-item author-defined development set achieved accuracy and macro-F1 of 1.00. Because those examples participated in rule development, the result is best interpreted as rule coverage.

The frozen 96-item reserved wording set produced accuracy 0.9167 and macro-F1 0.9196. All 96 predictions were unchanged by distracting indications, giving indication invariance 1.0000. Device, location and severity had perfect recall. Six errors fell back to summary; two uncertainty questions were routed to presence or severity. The errors were retained without changing planner rules.

The lower reserved-set score is informative rather than embarrassing. It demonstrates that the planner handles varied wording reasonably but is not general natural-language understanding. Clinician-authored wording remains Future Work.

## 4.10 Confidence and Runtime

The V10 retrieval calibrator achieved Brier score 0.16739, 10-bin ECE 0.04579 and AUROC 0.70546 on Test. The frozen threshold 0.176112 targeted 80% coverage and produced observed coverage 81.51%. These metrics concern the operational report-derived retrieval label, not answer correctness.

The separate V11 normalized gate accepted 99.83% of development rows and did not provide meaningful selective stratification. It was not promoted. The final interface therefore displays retrieval confidence as a research signal and preserves a no-reliable-history state without presenting the value as clinical probability.

V10 QA required 232.4 seconds for the recorded confirmation execution and peaked at approximately 5,311.6 MiB allocated GPU memory. The clean V11 generation matrix required 773.8 seconds with batch size 4 and peaked at 4,261.6 MiB. These measurements demonstrate feasibility on the local 8 GB GPU but are hardware-specific rather than deployment benchmarks.

## 4.11 Post-hoc Relevance-Construct Sensitivity

The frozen rankings were re-evaluated under three report-derived qrels without changing any model, threshold or ranking. The original combined construct weighted active report labels at 0.60 and RadGraph facts at 0.40. Label-only and fact-only variants were exploratory sensitivity analyses.

| Qrel variant | R4 nDCG@10 | R5 nDCG@10 | R5 minus R4 | 95% case-bootstrap CI |
|---|---:|---:|---:|---:|
| Combined 0.60 label + 0.40 fact | 0.34905 | 0.36007 | +0.01103 | [+0.00770, +0.01446] |
| Label only | 0.33725 | 0.34242 | +0.00517 | [+0.00096, +0.00951] |
| Fact only | 0.31076 | 0.33159 | +0.02084 | [+0.01750, +0.02422] |

The aggregate R5 advantage remained positive under all three definitions, but subgroup behavior was not uniform. Among 359 report-indexed abnormal Test cases, the combined-qrel difference was +0.00215 with CI [-0.00129, +0.00560], while the label-only difference was -0.00733 with CI [-0.01092, -0.00381]. Fact-only results were positive, but R5 itself uses RadGraph-derived features, so this variant shares representation with the evaluation construct and is not an independent clinical validation.

The audit also found that all 195 evaluated report-indexed normal cases and all 14 indeterminate cases had empty active-label sets. Under the frozen similarity function, empty-versus-empty active-label agreement received a score of 1.0. These queries consequently had an average of 968 Train candidates above the combined relevance threshold, compared with 9.36 for abnormal queries. The primary overall result remains unchanged, but its meaning is spectrum-dependent and limited to the frozen report-derived construct.

## 4.12 Results Summary

The four-RQ evidence chain is coherent but deliberately mixed. Correct images improved retrieval and strongly exceeded shuffled controls. Fact-aware R5 improved the primary retrieval metric over R4. Historical context improved QA over no history, but R5 did not confirm a Token-F1 advantage over R4 RAG. Case-to-fact evidence substantially reduced context and preserved provenance, but its answer-quality intervals crossed zero. Candidate generation and calibrated abstention remain open technical problems.

# Chapter 5: Discussion

## 5.1 Correct Images Provide Alignment-Specific Information

The strongest finding is not simply that an image-assisted model scored higher than BM25. The aligned system exceeded every deterministic wrong-image assignment after recomputation of the visual state. This makes the claim narrower and stronger: the useful signal depends on the relationship between the target case and its correct image.

The result addresses a common weakness in multimodal studies. A nominally multimodal pipeline can rely on text, dataset prevalence or a generic image prior. The shuffled control isolates alignment by preserving the question, indication, bank and model while breaking only the image-case pairing. The substantial nDCG drop indicates that MedSigLIP and the R5 representation capture case-specific visual information relevant to report-derived similarity.

The evidence should not be stretched into diagnostic performance. The target report defines automated relevance, and the retrieved reports are historical analogies. The experiment does not ask MedSigLIP to independently produce a diagnosis. Its contribution is to improve which historical cases are surfaced.

## 5.2 Fact-Aware Retrieval Adds a Small Confirmed Gain

R5 improved nDCG@10 over R4 by approximately 0.011 with an interval that excluded zero. The absolute difference is modest, but the comparator is already a strong multimodal reranker. The result is therefore more informative than comparison only against an intentionally weak text baseline.

The 2x2 audit suggests that fact-aware features account for the more stable mechanism contrast. Attention-view gains were smaller and the interaction crossed zero. This does not prove that attention is useless; it indicates that the final ensemble's advantage should not be attributed to component synergy without stronger evidence.

The post-hoc qrel audit narrows that interpretation. R5 remained above R4 overall under combined, label-only and fact-only definitions, but the abnormal combined interval crossed zero and the abnormal label-only result favored R4. The large aggregate fact-only gain may partly reflect shared RadGraph representation between R5 features and the evaluation construct. The positive primary result is internally credible under the frozen qrel, but it is not uniform across relevance definitions or report-indexed spectrum groups.

## 5.3 Better Retrieval Does Not Guarantee Better QA

Historical RAG clearly improved Token-F1 and F1RadGraph over target-image generation without history. This supports the practical premise that related reports can help a small local generator produce text more consistent with a hidden radiology report.

However, R5's confirmed retrieval improvement did not yield a confirmed Token-F1 advantage over R4 whole-report RAG. Several mechanisms can explain the gap. The top-ranked cases may differ while their Top-3 contexts contain similar information. The generator may ignore fine-grained retrieval improvements. Historical details can distract from the target image. Automated references may reward lexical overlap that does not align with the retrieval qrel.

This negative result is central to the thesis. Optimizing retrieval nDCG alone is not sufficient for end-to-end medical QA. Retrieval, evidence ownership, generation and answer evaluation must be measured separately. A system can be faithful to a retrieved report and still be misaligned with the target case; conversely, a short target-consistent answer may receive limited overlap against a long report reference.

## 5.4 Fact Selection Improves Efficiency, Not Proven Correctness

Case-to-fact selection reduced context by more than half and preserved complete provenance. This directly improves auditability and reduces prompt cost. It also makes the interface more useful because a reviewer can see the specific historical statement and its owning case rather than a long anonymous report.

The answer-quality intervals crossed zero. The correct conclusion is therefore that fact selection is an efficient evidence representation, not a confirmed accuracy improvement. The result also clarifies why a fact selector cannot be the whole solution: selecting the best facts from an irrelevant case only compresses the wrong evidence.

Token-ceiling behavior did not improve under case-to-fact despite shorter input. Output length depends on the generator's answer behavior and fixed decoding budget, not only on context length. Separating answer generation from deterministic provenance fixed the structural JSON problem, but it did not guarantee semantic completeness.

## 5.5 Candidate Recall Remains the Main Bottleneck

The V11 full-bank audit showed that a large proportion of report-derived relevant items never reached Top-100. RRF improved nDCG and relevant-case presence, and K=200 improved recall, but the K=100 result was mixed. This is a realistic engineering trade-off: larger and more diverse candidate pools improve rescue opportunity while increasing downstream cost.

The finding argues against repeatedly replacing the final reranker without improving candidate generation. A sophisticated reranker cannot rank an absent candidate. Future work should jointly evaluate candidate recall, reranking quality, latency and memory, with the candidate budget frozen before confirmation.

## 5.6 Confidence and Abstention

The V10 calibrator provided moderate discrimination under its report-derived label and enabled a reproducible no-reliable-history state. Yet selective G3 did not improve aggregate Token-F1. The V11 gate accepted nearly every row, showing little practical stratification.

Confidence should therefore remain a research signal. Calling it clinical confidence would confuse retrieval-label calibration with diagnostic uncertainty. A clinically meaningful abstention policy requires human labels of case similarity, answer correctness or potential harm, plus external calibration and prospective evaluation.

## 5.7 Planner Interpretation

The deterministic planner is useful because it is transparent, inexpensive and easy to test. The reserved set demonstrated good but imperfect wording robustness and complete indication invariance. The errors reveal the cost of a default summary fallback when wording lacks a recognized lexical cue.

The planner should not be presented as an LLM-level semantic reasoner. Its role is routing. For the thesis, this is sufficient because the primary scientific contribution is multimodal retrieval and evidence provenance, not general dialogue understanding. A clinician-authored benchmark would be the appropriate next validation step.

## 5.8 Research Value and Practical Significance

The research contributes a disciplined way to study historical case assistance. It moves beyond a dashboard that merely uploads an image and returns a fluent answer. The target report is hidden, retrieved reports belong to other cases, visual alignment is tested, duplicate leakage is controlled, provenance is deterministic, and negative results remain visible.

For clinical AI research, the ownership distinction is especially important. A statement can be supported by a document without being evidence about the current patient. The final system makes that difference explicit by labelling historical analogies and preserving case boundaries.

For engineering, the project demonstrates that a complete multimodal RAG pipeline can run locally on an 8 GB GPU with frozen foundation models, a compact learned reranker, deterministic evidence selection and reproducible evaluation. LangChain is not necessary for this contribution; direct modules keep each state and artifact inspectable.

## 5.9 Limitations

The first limitation is source scope. All primary and development results come from OpenI/IU-Xray. Duplicate clustering strengthens internal validity but does not establish external generalization. The processed artifact lacks reliable subject identifiers, so patient-level independence cannot be proven.

The second limitation is evaluation reference. Relevance, Token-F1 and F1RadGraph are derived from reports. The retrieval model and qrel also share RadGraph-derived information, creating feature-metric coupling. Empty active-label sets make the frozen qrel unusually broad for report-indexed normal and indeterminate cases, and label-only abnormal sensitivity did not support an R5 gain. These proxies are not physician judgments of similarity, correctness, harmfulness or clinical usefulness. Independent blind review remains unexecuted and is not replaced by author interpretation.

The third limitation is absolute performance. Retrieval Hit@1 remains low, candidate recall is incomplete, and generation overlap scores are modest. Token ceilings and occasional incomplete answers remain. Deterministic assembly guarantees provenance format, not semantic truth.

The fourth limitation is modality and workflow. The work covers chest radiographs and does not establish transfer to CT, MRI, ultrasound or other radiology domains. It does not integrate with a PACS, electronic record or clinical reporting workflow.

The fifth limitation is confidence. The calibrator targets an operational report-derived retrieval label. It is not probability of diagnosis, answer correctness or safety. The development gate's 99.83% acceptance demonstrates weak selective value.

## 5.10 Future Work

The highest-priority next step is independent radiologist review of a prespecified sample. Reviewers should score historical-case similarity, target-image consistency, usefulness of the cited historical evidence and potentially harmful content. The protocol and blank packages can be preserved now, but results must not be reported until real reviewers complete them.

The second priority is external patient-level validation on an authorized dataset such as MIMIC-CXR-JPG. A smaller prespecified subset is sufficient if subject/study identifiers, licensing and local model execution are respected. The external protocol must be frozen before outcome inspection.

The third priority is stronger candidate generation. K=200 RRF, learned sparse-dense fusion or modern medical retrieval encoders can be compared on a new development/confirmation design. Success should include relevant-item recall, nDCG, latency, memory and downstream QA rather than one ranking metric.

The fourth priority is calibrated selective prediction using human outcomes. Risk-coverage analysis should be based on physician similarity or answer-correctness labels and should include a meaningful no-reliable-history operating point.

The fifth priority is a clinician-authored planner wording set and broader modality testing. These extensions should remain separate from the current frozen evidence rather than being retrofitted into V10.

# Chapter 6: Conclusion

## 6.1 Answers to the Research Questions

**RQ1: Does the correctly aligned image improve similar-case retrieval?** Yes, within the same-source automated relevance construct. Image-image and image-report baselines exceeded BM25, and the final aligned R5 system achieved the highest primary retrieval score. The result shows that the target chest image contributes useful information beyond the indication and question.

**RQ2: Is the gain alignment specific?** Yes. Correctly aligned R5 nDCG@10 was 0.36007, while the mean across 100 fixed-point-free shuffled assignments was 0.24963 and no shuffled assignment reached the aligned score. The control supports dependence on the correct image-case pairing.

**RQ3: Does fact-aware multiview retrieval improve graded relevance?** Yes at the aggregate level under the frozen report-derived metric. R5 exceeded R4 by +0.01103 nDCG@10, 95% CI [+0.00770, +0.01441]. Post-hoc qrel sensitivity retained an overall positive difference under label-only and fact-only definitions, but the abnormal combined interval crossed zero and abnormal label-only retrieval favored R4. The result is therefore confirmed for the prespecified aggregate construct, not for every spectrum group or an independent clinical relevance standard.

**RQ4: Does retrieval improvement transfer to QA, and does fact selection help?** Historical RAG improved automated answer-reference consistency over no history, but the incremental R5 Token-F1 advantage over R4 RAG was not confirmed. Case-to-fact evidence substantially reduced context and preserved provenance, but its Token-F1 and complete F1RadGraph improvement intervals crossed zero. The answer is therefore mixed: retrieval and evidence efficiency improved, while final answer superiority was not established.

## 6.2 Final Contribution

The completed project is a bounded multimodal similar-case RAG study rather than a generic chatbot or an autonomous clinical agent. Its main contribution is the evidence chain: duplicate-aware splitting, common-bank retrieval comparison, alignment-specific shuffled controls, fact-aware ranking, downstream QA transfer, deterministic provenance, confidence boundaries and honest reporting of negative results.

The research demonstrates why medical RAG must preserve evidence ownership. Historical reports can be useful analogies, but they are not facts about the target patient. A fluent answer supported by the wrong case is not made correct by local faithfulness. The final workflow makes source identity visible and separates target-image generation from historical provenance.

## 6.3 Final Boundary

The results do not establish clinical diagnosis, safety, treatment utility or external generalization. Independent radiologist evaluation and external patient-level validation remain Future Work. Within those boundaries, the project achieves its graduate-research objective: a clear problem, a complete automated experimental loop, reproducible code and artifacts, an operational demonstration system, and conclusions that match rather than exceed the available evidence.

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

Appendices A-H preserve historical V9 and preliminary controlled-study artifacts for traceability. They are not the primary V10 result. Appendices I-L register the final V10/V11 evidence and release boundary.

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

**Branch:** `post-submission-improvements`

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

V5-V9 are frozen formative and historical studies. V10 is the final automated primary extension. V11 is development-only and does not instantiate a new confirmation cohort. Supplemental audits do not change frozen models, prompts, thresholds, metrics, splits, cases or primary results.

The following appendices preserve the detailed V5 controlled study for traceability. They are formative evidence and do not replace the V10 primary study.

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

The split was performed at case level. The fixed seed made the assignment reproducible and ensured that the same processed case identifier did not occur in both development and confirmation. The processed source records did not contain a stable patient or subject identifier, so patient-level independence could not be verified. Development cases could be used to verify implementation and confirm that the frozen pipeline executed, whereas confirmation outcomes were reserved for the reported V5 comparisons.

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

Reproducibility operated at several levels. Configuration reproducibility stored model identifiers, revisions, seeds, shortlist size, fusion weights, thresholds, batch sizes, and generation parameters. Data reproducibility stored case counts, case-level partitions, and a cohort fingerprint; a stable patient identifier was not available for verification. Result reproducibility stored aggregate JSON summaries and statistical outputs. Implementation reproducibility stored scripts and tests. The artifact manifest joined these with LF-normalized SHA-256 values so that unintended changes could be detected across platforms.

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

## Appendix I: Final V10/V11 Artifact Index

- V10 technical freeze: `docs/V10_TECHNICAL_FREEZE.md`
- V10 retrieval summary: `data/splits/v10/v10_confirmation_retrieval_summary.json`
- V10 QA summary: `data/splits/v10/v10_confirmation_qa_summary.json`
- V10 F1RadGraph summary: `data/splits/v10/v10_radgraph_metrics_summary.json`
- V10 post-hoc qrel sensitivity: `data/splits/v10/v10_qrel_sensitivity_summary.json`
- V11 clean generation summary: `data/splits/v11/v11_medgemma_generation_48_clean_summary.json`
- V11 statistical summary: `data/splits/v11/v11_medgemma_generation_48_statistical_summary.json`
- V11 candidate-generation summaries: `data/splits/v11/v11_candidate_generation_audit_summary.json` and `v11_candidate_generation_audit_k200_summary.json`
- V11 reserved planner summary: `data/splits/v11/v11_question_planner_reserved_summary.json`

## Appendix J: Prompt and Provenance Contract

The target image, indication and question are always separated from historical context. Historical units are labelled with case and section identity. The generator returns only a concise answer. Deterministic code attaches uncertainty state, abstention state, supporting case IDs and evidence provenance. Unknown evidence IDs are rejected, and malformed model output is marked rather than silently accepted.

## Appendix K: Release Acceptance Boundary

The final public release contains aggregate summaries, configurations, protocols, scripts, tests, manuscript artifacts and checksums. It excludes source image pixels, protected or source-derived report rows, model weights, private prompts containing report text, per-row generations, reviewer keys, credentials and access tokens. The release audit verifies V10 frozen hashes, test status, document rendering and numeric consistency.

## Appendix L: Human and External Evaluation Status

Independent radiologist review was not completed. The blinded V10 package remains blank and is labelled Future Work. The MIMIC-CXR adapter is implemented, but authorized source data were not available for the completed study. No human or external metric is claimed.
