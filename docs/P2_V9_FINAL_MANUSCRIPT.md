# Retrieval-Augmented Medical Question Answering over Paired Radiology Images and Reports

## Abstract

Retrieval-augmented generation can provide language models with traceable evidence, but conventional text-only retrieval is poorly matched to a new radiology case whose formal report has not yet been written. This research develops and evaluates a multimodal similar-case medical question-answering workflow in which a target chest radiograph, pre-report clinical indication, and question retrieve other-case image-report pairs from a frozen historical bank. The target report is hidden from every inference component and used only for offline evaluation. The study uses 3,851 paired OpenI/IU-Xray examinations. A deterministic report-indexed split assigns 2,631 cases to Train, 376 to Validation, and 752 to Test; 2,608 report-bearing Train cases form the historical bank. Report-derived graded relevance combines active-label similarity and RadGraph fact overlap. Baselines include BM25, MedSigLIP image-image and image-report retrieval, and fixed fusion. The proposed improvement is a project-trained 865-parameter multilayer perceptron reranker learned from 307,176 weighted pairs while all foundation models remain frozen.

On 752 Test cases, the learned reranker achieved nDCG@10 of 0.327942 and exceeded the strongest frozen image-only component by 0.012381, with 95% case-bootstrap confidence interval [0.009226, 0.015584]. Aligned retrieval also exceeded all 100 fixed-point-free shuffled-image controls (shuffled mean 0.220370; plus-one p = 0.009901). Downstream evaluation used 685 cases, two questions per case, four retrieval conditions, and 5,480 local MedGemma 1.5 generations. Learned multimodal RAG achieved Token-F1 0.184803 versus 0.145559 without retrieval, a difference of 0.039244 [0.032572, 0.045745], although its advantage over fixed multimodal RAG was unresolved. A bounded evidence-control agent reduced automatically unsupported historical-support fields from 16.42% to 0% through one backup route or evidence abstention while preserving the target-image answer by design.

Post-hoc protocol-governed audits qualified these findings. After excluding 187 Test cases with Train-report cosine similarity at least 0.95, the learned reranker remained first at nDCG@10 0.279730 versus 0.264642 for image-image retrieval. It also ranked first under label-only, RadGraph-only, and combined qrels. Qwen3-Embedding-0.6B improved the modern text baseline to 0.195633 but remained below the learned multimodal system and was more wording-sensitive. F1-RadGraph showed a clear learned-RAG advantage over BM25-RAG but no resolved advantage over no retrieval or fixed multimodal RAG. Robust reparsing recovered no additional structured outputs, confirming truncation as a genuine engineering limitation. A researcher accepted all labels in a purposively selected 24-case tool-assisted review; no independent radiologist adjudication was performed.

The evidence supports a scoped conclusion: correctly aligned chest-image information improves report-derived similar-case retrieval, and multimodal retrieved context improves report-reference consistency over weak text retrieval and the same generator without retrieval. The study does not establish diagnostic accuracy, patient safety, external generalization, or deployment readiness. Its main contribution is an auditable separation of retrieval, alignment, generation, evidence control, and validity threats in a realistic new-case multimodal RAG task.

# Chapter 1: Introduction

## 1.1 Background

Large language models can generate fluent answers to medical questions, yet fluency does not establish whether an answer is supported, correctly attributed, or clinically safe. Retrieval-augmented generation (RAG) responds to part of this problem by placing retrieved evidence in the model context before generation. In principle, retrieval can improve factual coverage, expose provenance, and allow unsupported claims to be traced to a specific evidence source. In practice, RAG remains a multi-stage system. Query formulation, candidate retrieval, multimodal fusion, answer generation, semantic checking, and abstention can each fail independently. A final answer can therefore appear coherent even when the evidence is irrelevant, belongs to another case, or has been interpreted incorrectly.

Radiology makes this distinction especially important. A chest-radiograph examination links a clinical indication, one or more images, findings, and an impression. When a new patient is imaged, the formal findings and impression are not yet available to an automated support system. The available query is instead composed of the target image, pre-report clinical history or indication, and a question. A useful retrieval system should search a historical archive for clinically similar other-patient cases rather than recover a report that already belongs to the target patient. Retrieved reports may provide analogies, terminology, and patterns that help a multimodal generator answer the question, but they are not proof that the same finding is present in the target patient.

The OpenI/IU-Xray collection provides de-identified chest-radiograph examinations with linked reports and images. It is sufficiently large for a controlled, local study and permits the construction of a fixed historical bank. Modern biomedical vision-language models such as MedSigLIP can map chest images and report text into related representation spaces, while MedGemma can condition generation on both a target image and textual evidence. These components make it possible to test a more realistic research question than simple paired-report recovery: whether an unseen target image can retrieve clinically similar historical image-report pairs and whether those retrieved reports improve question answering relative to the same generator without retrieval.

This thesis follows an iterative research programme. Early V5-V7 experiments used controlled paired-case retrieval to expose patient-scope ambiguity, indication shortcuts, image-alignment effects, downstream grounding failures, and the limits of naive or adaptive score fusion. Those studies remain reproducible preliminary evidence, but their closed-set task is not treated as the final clinical scenario. V9 changes the construct: the target report is removed from the candidate bank and hidden from inference. A fixed Train-only bank supplies other-case evidence to Validation and Test queries. The final claims are based on the V9 held-out study rather than on the earlier development versions.

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

**RQ1.** Does learned multimodal similar-case retrieval improve report-derived graded retrieval quality over text-only, image-only, image-report, and fixed-fusion alternatives?

**RQ2.** Is the retrieval gain dependent on the correctly aligned target chest image rather than on clinical text or a generic visual prior?

**RQ3.** Does learned multimodal retrieval improve downstream MedGemma answer-reference consistency over the same generator without retrieval and over text-only RAG?

**RQ4.** Can a bounded evidence-control agent reduce unsupported historical-support claims while preserving the target-image answer and maintaining a transparent retry and abstention trace?

**RQ5.** Which limitations remain at the retrieval, generation, structured-output, automated-verification, data, and evaluation levels?

## 1.6 Research Contributions

The first contribution is a clear new-patient task contract. The target report is hidden from retrieval, prompting, generation, verification, and agent routing. The system receives only the target radiograph, pre-report indication, and question. Historical reports belong to other cases and are presented as analogies rather than as patient evidence. This prevents the final experiment from collapsing into paired-report lookup.

The second contribution is a common-bank, graded retrieval evaluation. Five systems rank the same 2,608 report-bearing Train cases for every Validation and Test query. The report-derived gain combines active-label and RadGraph fact similarity while assigning no reward to shared negative labels. This design supports nDCG, MRR, and sensitivity analyses without pretending that exact case identity is the relevant answer.

The third contribution is a compact learned reranker. A 9-to-32-to-16-to-1 multilayer perceptron with 865 parameters combines normalized scores, reciprocal ranks, and question-type indicators from frozen BM25 and MedSigLIP channels. It is trained using weighted pairwise relevance differences, while MedSigLIP, MedGemma, RadGraph, and the NLI verifier remain frozen. The model therefore constitutes genuine project training without confounding the study through foundation-model fine-tuning.

The fourth contribution is an alignment-specific negative control. One hundred unique wrong-image assignments recompute all visual scores, ranks, normalized features, and learned outputs. The aligned system is compared with the complete shuffled distribution using a plus-one randomization p-value. This design tests whether the result depends on the correct image rather than on text leakage or reuse of an aligned visual state.

The fifth contribution is an end-to-end transfer analysis. The same MedGemma revision receives the target image in every condition. No-retrieval, BM25 RAG, fixed multimodal RAG, and learned multimodal RAG differ only in the historical reports supplied. This isolates whether better retrieval changes reference consistency. The study reports both the positive learned-RAG versus no-RAG effect and the unresolved learned-versus-fixed difference.

The sixth contribution is a bounded agentic evidence-control layer. The agent does not diagnose the target image. It checks only statements presented as historical support, permits one deterministic backup route, and otherwise removes unsupported historical evidence and records an abstention. This makes the agent useful and auditable without implying autonomous clinical reasoning.

The seventh contribution is methodological transparency. Protocols were committed before their corresponding outcome stages; large source-derived artifacts remain local; public summaries contain hashes and aggregate metrics; and automated verification increased from 206 passing tests before the supplemental additions to 223 in the final V9 suite. The work preserves negative findings, including the weakness of BM25, the underperformance of naive fixed fusion, incomplete JSON output, low absolute Token-F1, and the absence of clinical human evaluation.

## 1.7 Scope and Boundaries

The final study is restricted to chest radiographs from one source, OpenI/IU-Xray. It is a retrospective technical benchmark, not a prospective clinical trial. The source design describes one study per patient, but reliable released patient identifiers are unavailable; the thesis therefore reports case-level disjointness and source-design patient uniqueness rather than independently verified patient-level separation.

The report-indexed normal and abnormal labels come from the dataset `problems` field. They are used for stratification and sensitivity analysis, not as new clinical adjudication. Report-derived relevance comes from hidden annotations and RadGraph facts. It is suitable for controlled ranking evaluation but cannot establish physician-perceived similarity.

MedGemma receives one deterministically selected target chest image. Historical images influence retrieval but are not passed as additional generator pixels under the primary 8 GB GPU configuration. Token-F1 measures lexical overlap with a hidden target report section and is not diagnostic accuracy. The BioLinkBERT-MedNLI checker is an automated signal and may over-reject or under-detect support.

The dashboard demonstrates research inference over a frozen local bank. It does not connect to hospital systems, identify a patient's actual prior report, provide treatment advice, or replace radiologist review. Independent clinician scoring, external datasets, prospective workflow effects, calibration, fairness, and deployment safety remain outside the completed study.

## 1.8 Conceptual Framework

The framework contains six linked stages. First, source construction defines each case as paired image views, indication, findings, impression, and report-indexed metadata. Second, deterministic partitioning assigns cases to Train, Validation, and Test before outcome inspection. Third, frozen retrieval components score the Train-only historical bank using query text, image-image similarity, and image-report compatibility. Fourth, fixed or learned fusion produces a Top-3 evidence set. Fifth, MedGemma generates a structured answer from the target image, indication, question, and optional historical reports. Sixth, the bounded agent checks historical-support claims and records retry, revision, or abstention.

Evaluation mirrors this structure. Retrieval quality is measured with report-derived nDCG and MRR. Alignment is tested by replacing each target image with a wrong Test image while retaining its indication and question. Generation is measured against hidden findings and impression references. Historical support is checked against cited reports. Each metric answers a different question and is not allowed to stand in for clinical correctness.

The central causal chain is:

```text
correctly aligned target image
        + clinical indication and question
                    ↓
better ranking of report-derived similar cases
                    ↓
more useful historical report context
                    ↓
greater target-report reference consistency
                    ↓
bounded checking of claims about historical evidence
```

Every arrow is tested separately. The shuffled-image control interrogates the first arrow. Retrieval confirmation tests the second. The four-condition MedGemma experiment tests the third. Agent traces test the fourth. The thesis does not add an unsupported arrow from automated reference consistency to patient benefit.

## 1.9 Thesis Organization

Chapter 2 reviews RAG, biomedical retrieval, multimodal radiology representations, question answering, evidence checking, and related similar-case systems. Chapter 3 distinguishes the preliminary controlled studies from the final V9 design and describes the source split, relevance construction, retrieval models, learned reranker, generation conditions, agent, statistics, and reproducibility controls. Chapter 4 reports preliminary findings briefly and gives the full V9 retrieval, alignment, QA, agent, sensitivity, and cost results. Chapter 5 answers the research questions, explains the contributions and limitations, and identifies appropriately scoped future work. The appendices provide frozen artifacts, hashes, commands, and review boundaries.

# Chapter 2: Literature Review

Sections 2.1-2.11 preserve the literature synthesis that motivated the preliminary V5 controlled study. Task-specific references to V5 in those sections describe that preliminary phase. Section 2.12 updates the gap for the final V9 new-patient similar-case study.

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

## 2.12 Similar-Case Multimodal RAG and the Final Research Gap

The closest line of work combines chest-image retrieval with report generation. CXR-RePaiR uses a contrastive image-to-report retriever to construct reports from retrieved exemplars. X-REM adds coarse retrieval, learned image-text matching, and an NLI filter. FactMM-RAG mines factual report pairs with CheXbert and RadGraph to train a fact-aware multimodal retriever. The 2026 RA-RRG system retrieves clinically important key phrases and uses them to condition report generation, while MedProbCLIP introduces probabilistic image-report embeddings, calibration, and risk-coverage evaluation. These systems establish that historical image-report pairs can support image-conditioned language generation and that retrieval reliability deserves explicit measurement. They do not, however, directly test a question-conditioned new-patient workflow in which the target image, clinical indication, and medical question jointly retrieve other cases and the final output explicitly separates target observations from historical analogies.

Recent multimodal and agentic systems also motivate a stricter evaluation boundary. Concept-enhanced RAG methods combine visual embeddings with medical concepts; agentic radiology systems separate planning, retrieval, generation, and validation roles; and generated-report approaches use an intermediate radiology description to improve VQA. These designs show that orchestration can improve modularity, but additional agents do not automatically create stronger evidence. An agent may simply repeat the same unsupported claim through more steps. The relevant contribution is therefore not the number of roles but whether actions are bounded, inputs are permitted, failures are traceable, and abstention is available.

The literature leaves five connected gaps. First, many retrieval-augmented radiology studies focus on report generation rather than answering a user question. Second, evaluations often compare multimodal fusion with text baselines but do not require superiority over the strongest individual visual component. Third, aligned-image gains are rarely challenged by complete shuffled-image recomputation. Fourth, evidence verification is commonly reported without separating support for a historical analogy from correctness about the target image. Fifth, near-duplicate sensitivity, relevance-definition sensitivity, and wording robustness are often treated as implementation details even though they can materially change retrieval conclusions.

V9 addresses these gaps with a scoped design. It compares text-only, image-only, image-report, fixed fusion, and learned fusion over one bank; trains only a small reranker; evaluates alignment with 100 fixed-point-free controls; uses the same multimodal generator across retrieval conditions; and limits the agent to historical-evidence checking. The final research gap is consequently not "whether RAG can be used in radiology." It is whether correctly aligned visual evidence can produce reproducible gains in other-case retrieval and whether those gains transfer to reference-consistent QA under an auditable evidence contract.

# Chapter 3: Methodology

## 3.1 Final Research Design and Version Boundary

The final study is V9. Preliminary V5-V8 experiments established the evaluation vocabulary of target alignment, indication shortcuts, shuffled-image controls, retrieval-to-generation transfer, and automated-verifier disagreement. Their role is formative, and their detailed methods and results are retained in Appendices G and H rather than interleaved with the primary study. V9 changes the task construct: the target report is removed from the bank and every retrieved report is an other-case historical analogy.

The transition was protocol governed. V5-V8 artifacts remained frozen. A technology-reuse audit defined which repository components could be retained and which external methods could be independently reimplemented. A full-source split amendment was committed before the final split was instantiated. RadGraph preprocessing, MedSigLIP development, learned-reranker training, retrieval confirmation, QA confirmation, qualitative extraction, and supplemental validity analysis each had separate boundaries. No V9 Test outcome was used to alter a model, prompt, threshold, metric, case, or hypothesis.

## 3.2 Task Contract, Systems, and Hypotheses

The intended use case is a new examination for which a chest radiograph and pre-report clinical indication are available but no formal target report has yet been written. The system receives the target image, indication, and a question such as a request for findings or impression. It searches a historical Train-only bank for similar other-patient image-report pairs. The retrieved reports provide terminology and analogies to a multimodal generator; they are never presented as the current patient's own report. During offline evaluation, the hidden target findings and impression supply references and report-derived relevance annotations.

This contract resolves a common ambiguity. The research does not retrieve an already known report that belongs to the target examination. Nor does it assume that a new patient arrives with a complete report and then ask the system to rediscover similar text. The scientific question is whether the image contributes patient-specific retrieval information before reporting, and whether other-case evidence improves a bounded answer when the generator must still inspect the target image.

The retrieval baselines and improvement are explicit. R0 is BM25 text retrieval. The supplemental modern text baseline is Qwen3-Embedding-0.6B. R1 is MedSigLIP image-image retrieval and was the strongest frozen component on Validation. R2 is target-image-to-historical-report retrieval. R3 is validation-selected fixed score fusion. R4, the proposed improvement, is a learned nine-feature MLP reranker. Generation uses G0 without retrieval, G1 with BM25 reports, G2 with fixed-fusion reports, and G3 with learned-reranker reports. G4 is a bounded post-generation evidence-control agent. This separation makes clear that R4 is the improved retrieval model, G3 is the improved end-to-end RAG condition, and G4 is an auditable control layer rather than a replacement generator.

The primary retrieval hypothesis was that R4 would exceed R1 on case-grouped nDCG@10. The alignment hypothesis required correctly paired images to exceed the complete shuffled-image distribution. The primary generation hypothesis was that G3 would exceed G0 on case-grouped Token-F1. Learned-versus-fixed generation, subgroup, clinical semantic, wording, duplicate, and qrel analyses were secondary or post-hoc exploratory. No result was allowed to redefine these roles after outcome inspection.

The implementation uses fixed prompt engineering but does not depend on LangChain. Prompt instructions separate target-image observations from historical support, constrain output to one JSON object, restrict citations to retrieved IDs, and permit uncertainty or abstention. The agent is implemented as a deterministic state machine with one optional retry because an orchestration framework would not add scientific evidence to this bounded workflow.

## 3.3 Data Source, Eligibility, and Deterministic Partition

The source artifact contains 3,851 paired OpenI/IU-Xray studies. Cases were classified from the normalized `problems` field before splitting. There were 1,379 report-indexed normal cases, 2,380 report-indexed abnormal cases, and 92 report-index indeterminate cases. "Report-indexed" denotes the dataset annotation and does not represent new physician review. The 92 empty or `no indexing` cases were excluded from the primary stratifiable frame rather than reclassified. The resulting primary universe contained 3,759 cases.

The predefined split approximated 70/10/20 while preserving the exact normal/abnormal totals. Train contained 965 normal and 1,666 abnormal cases (2,631 total). Validation contained 138 normal and 238 abnormal cases (376 total). Test contained 276 normal and 476 abnormal cases (752 total). All 262 previously unused eligible cases identified by the project-history audit were placed in Test, and deterministic SHA-256 ranking selected the remaining 490 Test cases and all Validation cases. The split seed was 7029 with domain-separated payloads for Test supplementation and Validation selection. Canonical case identifiers were stripped strings, UTF-8 encoded, sorted by digest and identifier, and sampled without replacement.

Twenty-five source cases had neither findings nor impression. Twenty-three occurred in Train and two in Validation; none occurred in Test. To keep every comparator on the same evidence universe, the 23 empty-report Train cases were excluded from all retrieval conditions, including image-only retrieval. The shared historical bank therefore contained 2,608 report-bearing Train cases. Primary Validation relevance evaluation used 374 report-bearing queries. All 752 Test queries had usable report-derived relevance references. The downstream QA frame required both findings and impression and contained 685 Test cases.

The source publication describes one study per patient, but released patient identifiers were unavailable in the processed data. Case-ID disjointness was verified. The thesis does not claim independently verified patient-level separation. Validation and Test queries nevertheless retrieve only from the Train bank, so their own study reports are absent from inference.

## 3.4 Report-Derived Graded Relevance

V9 requires a graded notion of clinical similarity because exact report identity is neither available nor desirable in the historical bank. Offline relevance was constructed from hidden target-report annotations and candidate-report annotations. It was never exposed to retrieval, generation, verification, or agent routing.

Active label similarity used weighted Jaccard overlap. Positive labels received weight 1.0, uncertain labels 0.5, and negative or missing labels 0.0. Shared absent abnormalities therefore contributed no reward. If both reports had no active abnormal label, label similarity was defined as 1.0; if only one had active abnormalities, it was 0.0. This special case allowed report-indexed normal cases to be evaluated without rewarding a large vector of explicitly absent labels.

RadGraph XL was run locally on normalized findings plus impression. Entities and relations were flattened with the deterministic complete-reward representation. Report-derived fact similarity was the F1 overlap of the normalized fact sets. The continuous primary gain was:

```text
gain(query, candidate)
  = 0.60 × active-label similarity
  + 0.40 × RadGraph fact similarity
```

The score was constrained to [0,1]. nDCG@10 used the continuous gain. Binary MRR and Recall diagnostics used a frozen gain threshold of 0.50. Label-only and fact-only results were sensitivity analyses. The operational relevance supports reproducible ranking comparisons but is not physician-adjudicated similarity.

## 3.5 Retrieval Components and Fixed Fusion

R0 was BM25 over historical findings and impression, queried with the clinical indication and one of three fixed question templates. R1 was cosine similarity between the normalized mean MedSigLIP embedding of the target image views and the normalized mean embedding of each historical study's image views. R2 was MedSigLIP cross-modal similarity between the target image representation and the historical report representation.

Historical reports were split into sentence-aware, section-prefixed chunks of at most 64 MedSigLIP tokens. Each chunk was encoded and normalized. Validation compared normalized mean chunk aggregation with maximum image-to-chunk similarity. Mean aggregation was retained because the maximum policy did not satisfy the frozen material-improvement rule. For multi-view studies, each readable view was encoded independently, normalized, averaged, and normalized again. No view was selected based on outcome.

Each retrieval channel was independently min-max normalized across all 2,608 candidates for a query. Constant channels mapped to zero. The fixed fusion grid evaluated nonnegative weight triples in increments of 0.25 that summed to one. A valid multimodal candidate required positive BM25 weight and positive total image weight. Validation nDCG@10 selected the final R3 weights under a simplicity rule that favored more BM25 weight among candidates within 0.005 of the maximum. Canonical case ID resolved every ranking tie.

## 3.6 Learned Multimodal Reranker

The project-trained component was deliberately small. A nine-feature vector represented each query-candidate state: independently normalized BM25, image-image, and image-report scores; normalized reciprocal rank under each component; and three one-hot indicators for findings, impression, or acute question type. Raw text, image pixels, identifiers, filenames, labels, RadGraph facts, references, and QA outcomes were prohibited as features.

The 2,608 Train-bank cases were deterministically assigned to 1,600 pairwise-fit queries, 500 internal early-stop queries, and 508 bank-only cases while remaining available as historical candidates. For a fit query, candidate mining formed a deterministic union of Top-32 candidates under each component and offline relevance, plus Bottom-32 relevance candidates. The eight highest-gain and eight lowest-gain members formed ordered pairs when their gain difference was at least 0.05. Pair weights equaled the gain difference. This generated 307,176 weighted pairwise examples.

Two scorers were prespecified: a linear 9-to-1 model and a multilayer perceptron `9 → 32 → 16 → 1` with ReLU activations. Both used weighted pairwise softplus loss, AdamW, learning rate 0.001, weight decay 0.0001, batch size 4,096 pairs, at most 30 epochs, seed 7030, and internal early stopping. Validation selected the architecture. The MLP was promoted because it met the frozen margin over fixed fusion and also exceeded the strongest individual component. Its 865 parameters were the only learned model parameters in the final system. Foundation encoders and language models remained frozen.

## 3.7 Retrieval Confirmation and Alignment Control

Five frozen systems were evaluated once on 752 Test cases and three fixed questions: BM25, image-image, image-report, fixed fusion, and the learned MLP. The primary metric was case-grouped equal-question nDCG@10. The primary paired comparison was learned MLP minus image-image because image-image was the strongest frozen component on Validation. A 10,000-iteration case bootstrap with seed 7031 produced a percentile 95% confidence interval. Confirmed superiority required the lower bound to be greater than zero.

Alignment dependence was evaluated through 100 deterministic, unique, fixed-point-free wrong-image assignments. Test cases were ordered by a domain-separated SHA-256 rule, and cyclic shifts 1-100 supplied complete image-view sets from other Test cases. For each assignment, image-image scores, image-report scores, normalization, ranks, features, MLP scores, and rankings were recomputed. BM25 remained attached to the original indication and question. The plus-one randomization p-value counted how many shuffled nDCG@10 values equaled or exceeded aligned performance. A predefined 262-case project-history-untouched subset was reported only as sensitivity analysis.

## 3.8 Downstream Multimodal Question Answering

The downstream frame contained 685 Test cases with nonempty findings and impression. Each contributed a findings question and an impression question, producing 1,370 questions. The acute question used for retrieval was excluded from generation scoring because the dataset lacked a physician-adjudicated binary acute-abnormality reference.

All four generation conditions used the same `google/medgemma-1.5-4b-it` revision, local 4-bit NF4 inference, deterministic decoding, maximum 192 new tokens, target indication, question, and one target chest image. The image policy preferred a frontal view and then lexicographic filename order. G0 received no historical report. G1 received Top-3 BM25 reports. G2 received Top-3 fixed-fusion reports. G3 received Top-3 learned-reranker reports. Historical images influenced retrieval but were not additional generator pixel inputs.

The prompt required JSON fields for answer, target-image findings, supporting case IDs, historical support, uncertainty, and abstention. It stated explicitly that historical reports were analogies and not proof about the target patient. Parser failures and token-ceiling outputs were retained; no selective regeneration was allowed. The primary metric was case-grouped equal-question Token-F1 against hidden target findings or impression. G3 minus G0 was the primary comparison, with 10,000 case-bootstrap iterations. Exact match, JSON completeness, per-question effects, subgroup effects, latency, tokens, and memory were secondary.

## 3.9 Bounded Evidence-Control Agent

G4 was a bounded control layer applied after G3. Its verifier checked only whether statements in the `historical_support` field were supported by the cited historical reports. It did not verify the `answer` or `target_image_findings` against the target image. If G3 historical support failed the frozen checker, the agent could perform one deterministic retry using the frozen image-image R1 Top-3 route. If the backup evidence still failed, it removed the historical-support statement and citations, retained the target-image answer, and recorded historical-evidence abstention.

The agent had no internet access, no model-selection authority, no threshold changes, no target-report access, and no unbounded loop. Every initial route, cited ID, support score, retry, revision, abstention, and reason was recorded. The primary agent outcome was the paired change in automated unsupported historical-support rate. A Token-F1 noninferiority margin of -0.01 guarded against unintended answer modification, although the target answer was preserved by design.

## 3.10 Researcher-Reviewed Qualitative Analysis

A post-hoc qualitative protocol was committed before systematic case extraction and coding. The deterministic pack contained six cases with the largest mean G3-minus-G0 Token-F1 gains, six with the largest losses, six agent retry cases, and six historical-evidence abstention cases. The extraction tool assembled frozen retrieval, generation, reference, and agent evidence and proposed taxonomy labels. The named researcher then reviewed all 24 rows and accepted the proposed label sets without modification on 19 August 2026. The audit trail retains original proposals, researcher-reviewed labels, status, initials, date, and bounded notes.

This process is described as researcher-reviewed, tool-assisted exploratory analysis. It was not blinded because some outputs had been inspected during pipeline verification; it was not independent because the researcher was the project author; and it was not clinical adjudication because no radiologist scored the cases. Category counts characterize the purposively selected pack only and are not population-rate estimates.

## 3.11 Supplemental Validity and Robustness Audits

After the V9 technical freeze, a separate post-hoc protocol was committed before supplemental outcomes. Five analyses were prespecified. First, normalized report text and image dHash were used to audit cross-split similarity; a sensitivity analysis excluded Test reports with maximum Train cosine similarity at least 0.95. Second, frozen rankings were evaluated under active-label-only, RadGraph-fact-only, and original combined qrels. Third, pinned Qwen3-Embedding-0.6B provided a modern dense text comparator and nine fixed question wordings tested retrieval robustness. Fourth, frozen answers were evaluated with F1-RadGraph and case-grouped bootstrap intervals; F1CheXbert would be reported only if an official compatible local dependency was available. Fifth, balanced JSON extraction, fence removal, and trailing-comma removal tested whether structured-output failures were repairable without fabricating truncated content.

All supplemental analyses were interpretive. They could strengthen, qualify, or weaken the final claims but could not trigger retraining, prompt changes, threshold changes, case replacement, or a new primary hypothesis. The duplicate hash is not treated as proof of patient identity; fixed paraphrases are not treated as physician-authored questions; and automated graph overlap is not treated as clinical correctness.

## 3.12 Reproducibility, Ethics, and Evidence Boundaries

The implementation used local CUDA inference and preserved model revisions, configuration files, checkpoint hashes, result hashes, split fingerprints, and protocol commits. Large source-derived texts, image pixels, vectors, checkpoints, prompts, and per-row generations remained local under repository policy. Aggregate summaries, source-neutral code, hashes, tests, and a lightweight case index were public. The verified suite contained 206 passing automated tests before the supplemental additions and 223 passing tests in the final V9 integration run.

No radiologist evaluated pairwise similarity, retrieved reports, target-image answers, or agent decisions. The completed researcher review supports exploratory pipeline interpretation only. The study therefore reports retrospective technical performance and explicitly excludes claims of diagnostic safety, clinical utility, or deployment readiness.

# Chapter 4: Results and Analysis

## 4.1 Final V9 Retrieval Confirmation

The V9 Test evaluation contained 752 cases, three questions per case, five systems, and 11,280 ranking rows. Every system ranked the same 2,608-case historical bank. BM25 produced nDCG@10 of 0.134156 and MRR of 0.083542. Image-image retrieval was substantially stronger at 0.315561 nDCG@10 and 0.328270 MRR. Image-report retrieval achieved 0.274069 and 0.256032 respectively. Fixed multimodal fusion reached 0.246935 nDCG@10 and 0.211322 MRR, below the image-only component. The learned MLP produced the strongest nDCG@10, 0.327942, and MRR, 0.331968.

The primary learned-minus-image difference was +0.012381 nDCG@10. Its 10,000-iteration case-bootstrap 95% confidence interval was [0.009226, 0.015584]. Because the lower bound was greater than zero, the frozen superiority criterion passed. The magnitude is modest relative to the large image-versus-text gap, but it is consistent and attributable to the learned combination rather than to foundation-model fine-tuning.

The fixed-fusion result is an important negative finding. More modalities did not automatically improve ranking. Equal or validation-selected weighted addition can dilute a strong visual signal with a weak text channel when generic questions and short indications provide limited discrimination. The learned model recovered a gain because it used scores, reciprocal ranks, and question type to condition candidate ordering. This supports learned fusion as the methodological contribution, while cautioning against describing any multimodal combination as inherently superior.

The 262-case strict project-history-untouched subset showed the same direction. In that subset, BM25, image-image, image-report, fixed fusion, and learned MLP achieved nDCG@10 values of 0.129956, 0.411812, 0.331673, 0.288307, and 0.419901 respectively. These values were not used for selection and are not a separate confirmatory family. They reduce concern that the full result was driven only by cases encountered in earlier project stages.

## 4.2 Alignment-Specific Image Contribution

Aligned R4 nDCG@10 was 0.327942. Across 100 complete wrong-image recomputations, mean nDCG@10 was 0.220370, standard deviation 0.004900, and the 2.5th and 97.5th percentiles were 0.210726 and 0.231474. The aligned score exceeded every shuffled assignment. The plus-one p-value was 0.009901.

This control is stronger than substituting one arbitrary image or shuffling only the final score. Every visual component and all derived learned features were recomputed under the wrong image. The original clinical indication and question were retained. The result therefore shows that correct target-image alignment materially influenced similar-case ranking. It does not prove that every retrieved case is clinically appropriate, but it rules out an explanation based only on text or a generic image prior.

## 4.3 Downstream QA Transfer

The QA study used 685 complete-reference Test cases, 1,370 questions, four systems, and 5,480 local MedGemma generations. G0 target-image generation without retrieval achieved Token-F1 0.145559 and complete JSON rate 42.04%. G1 BM25 RAG achieved 0.147947 and 39.42%. G2 fixed multimodal RAG achieved 0.179090 and 46.50%. G3 learned multimodal RAG achieved the highest Token-F1, 0.184803, and complete JSON rate, 57.23%.

The primary G3-minus-G0 Token-F1 difference was +0.039244, with 95% confidence interval [0.032572, 0.045745]. The criterion therefore passed. The strict 262-case sensitivity subset also favored G3 by +0.047187, interval [0.034911, 0.059704]. BM25 RAG differed from no retrieval by only +0.002388, interval [-0.003189, 0.008120]. Fixed multimodal RAG improved over no retrieval by +0.033532, interval [0.027790, 0.039502]. Learned multimodal RAG improved over BM25 by +0.036856, interval [0.030410, 0.043524].

G3 exceeded G2 by +0.005713, but the interval [-0.000958, 0.012280] crossed zero. The thesis therefore claims that learned multimodal RAG improved over no retrieval and text-only RAG; it does not claim confirmed downstream superiority over fixed multimodal RAG. Retrieval superiority and generation superiority are distinct. A different Top-3 ordering may not change the available evidence enough to produce a resolved answer-level difference.

The effect was larger for impression questions (+0.059903) than findings questions (+0.018585). It remained positive for report-indexed normal cases (+0.060784) and abnormal cases (+0.025570). These are prespecified sensitivity results rather than independent confirmatory hypotheses. One interpretation is that retrieved analogies help synthesize an impression more than enumerate visible findings, but this explanation remains inferential because no clinician reviewed individual responses.

Absolute performance remained low. MedGemma sometimes reached the 192-token limit or returned non-strict JSON. Only 57.23% of G3 rows contained the complete requested object. All outputs were retained under a tolerant parser rather than selectively regenerated. These failures matter because a system can improve mean Token-F1 while remaining unsuitable for deployment.

## 4.4 Bounded Agent Results

The G4 agent evaluated all 1,370 G3 rows. Before control, 16.423% of rows contained historical-support statements that the frozen automated checker did not substantiate from the cited reports. After one optional R1 backup route or evidence-field removal, the final automated unsupported historical-support rate was 0%. The paired reduction was 16.423 percentage points, with 95% case-bootstrap interval [14.4526, 18.4672] percentage points in magnitude.

The agent retried 16.423% of rows, revised historical support in 17.299%, and abstained from historical evidence in 15.985%. Mean retrieval calls increased from one to 1.164. Token-F1 remained 0.184803 because the target answer was preserved by design. The zero final unsupported rate is therefore partly structural: when no support could be established, the field was removed. It demonstrates auditable claim suppression and abstention, not that all retained target-image answers were correct.

## 4.5 Computational Cost and Artifact Integrity

The 5,480 local MedGemma generations required 13,496 seconds, or approximately 3.75 hours, at 0.406 records per second. Peak allocated GPU memory was 5,184.5 MiB on the RTX 5070 Laptop GPU with batch size eight. The small reranker trained on CPU-sized feature tensors and introduced negligible inference cost compared with MedSigLIP and MedGemma.

Frozen hashes identify the principal artifacts. Retrieval rows hash to `baa56924928b144c9b877b8e2218e04d17df6b77a6f794ed3830f7ccf3e449fd`. The MLP checkpoint hash is `8afa68a48de9d6c9128d190f1368d0d45d41a958e5eb12787d7e725e7eb09efa`. The Top-3 ranking pack hash is `28639821abc5fba8189c7c0149822ed0e3935325d0136578803155cc5a4ebd9b`. QA raw rows hash to `89c69c9a27e393c93c85e572587b330f908598e835cb8162a8678cd15ba512b4`, and agent rows hash to `9cc8b4513f2ef12f7e849d7b5853a79ef07495b022699c5a84785d1d94624fc1`.

## 4.6 Researcher-Reviewed Qualitative Findings

The researcher reviewed all 24 selected cases and accepted the tool-assisted taxonomy labels without modification; no case was excluded. Labels overlapped. Fifteen cases showed retrieval relevance gain and nine showed retrieval relevance failure. Two were labeled reference-consistent and thirteen reference-inconsistent. Ten exposed structured-output failure, five showed historical-support retry recovery, thirteen historical-support abstention, and five citation repair.

The review confirms heterogeneous pipeline behavior rather than a prevalence estimate. Retrieval gain did not guarantee a reference-consistent answer. Correctly retrieved evidence could still be omitted or misinterpreted during generation, while the bounded agent sometimes recovered support and more often removed an unsupported analogy. These are researcher-reviewed exploratory interpretations of frozen references and traces, not radiologist judgments of clinical correctness.

## 4.7 Cross-Split Similarity Sensitivity

The audit found 162 exact normalized Train-report duplicates among 752 Test cases. Maximum Train-report cosine similarity was at least 0.90 for 214 Test cases, 0.95 for 187, and 0.99 for 170. Image dHash distance was zero for 11 Test cases and at most four for 447; because chest radiographs share layout and anatomy, these perceptual collisions do not prove repeated patients.

The prespecified report sensitivity excluded the 187 Test cases with cosine similarity at least 0.95. On the retained 565 cases, R4 nDCG@10 was 0.279730, followed by R1 at 0.264642, R2 at 0.247601, R3 at 0.226451, and R0 at 0.139179. R4 therefore remained first and its margin over R1 was +0.015088. The direction of the primary finding survived, but the high same-source near-duplicate prevalence remains a material threat to external and population validity.

## 4.8 Qrel, Dense-Baseline, and Wording Robustness

R4 ranked first under each prespecified relevance construct. Under active labels alone, R4 and R1 achieved nDCG@10 of 0.333863 and 0.318698, a difference of +0.015165. Under RadGraph facts alone, they achieved 0.292220 and 0.289271, a much smaller +0.002950. Under the frozen combined qrel, the difference was +0.012381. The ordering is therefore robust, but effect magnitude depends on how clinical similarity is operationalized.

The pinned Qwen3 dense baseline achieved canonical nDCG@10 of 0.195633, exceeding BM25 by +0.061476 but remaining below R4 at 0.327942. Across two fixed paraphrases for each of three question roles, Top-1 agreement with the canonical wording was 11.41% for BM25, 35.70% for Qwen3 dense, and 99.69% for R4. Mean Top-10 Jaccard was 0.1096, 0.3226, and 0.8887 respectively. R4's stability is consistent with the strong contribution of its visual channels; it is wording robustness under researcher-written variants, not physician-authored language validation.

## 4.9 Clinical Semantic and Structured-Output Audits

F1-RadGraph provided a different view of generation quality. G0, G1, G2, and G3 complete F1 values were 0.124852, 0.103866, 0.124971, and 0.124803. G3 exceeded G1 by +0.020937 with 95% interval [0.012863, 0.028992], but differed from G0 by -0.000049 [-0.006803, 0.006897] and from G2 by -0.000168 [-0.008553, 0.008127]. Entity and entity-relation results followed the same broad pattern. Thus learned multimodal RAG clearly repaired the weak BM25-RAG route, but automated clinical graph overlap did not establish general superiority over target-image-only or fixed multimodal generation. F1CheXbert was not run because an official compatible local dependency was unavailable; no substitute was used.

Balanced JSON extraction, markdown-fence removal, and trailing-comma repair recovered no additional rows. Across all systems, 2,537 of 5,480 outputs were valid before and after robust reparsing; 2,943 remained unrecoverable and answer-change rate was zero. The same held for G3: 784 of 1,370 were valid. The incomplete-output problem is therefore primarily token-ceiling truncation rather than a removable parser artifact.

## 4.10 Integrated Results Summary

The final quantitative evidence chain is complete. Correct images improved retrieval relative to shuffled images. The learned reranker exceeded the strongest frozen component. Learned multimodal RAG improved reference consistency over no retrieval and text-only RAG. The bounded agent suppressed unsupported historical evidence. At the same time, fixed fusion underperformed image-only retrieval, learned QA did not clearly beat fixed QA, structured output was incomplete, and no clinical human score was obtained.

The supplemental evidence makes this conclusion more precise. Retrieval ordering persisted after a strict near-duplicate exclusion and across three qrel definitions. A modern dense model improved the text baseline but did not close the multimodal gap. Wording robustness strongly favored R4. In contrast, F1-RadGraph did not reproduce a learned-RAG advantage over G0 or G2. The study therefore supports its retrieval claim more strongly than a claim of universal answer-quality superiority.

# Chapter 5: Discussion and Conclusion

## 5.1 Answers to the Research Questions

### RQ1: Does learned multimodal similar-case retrieval improve graded retrieval quality?

Yes, within the frozen OpenI V9 benchmark and the report-derived relevance definition. The learned reranker achieved nDCG@10 of 0.327942 compared with 0.315561 for the strongest frozen image-image component. The paired difference of +0.012381 had a 95% confidence interval entirely above zero. The result is not merely "multimodal beats text." Image-image retrieval already exceeded BM25 by a large margin, and naive fixed fusion underperformed image-only retrieval. The supported conclusion is more specific: a small learned candidate-level combination of frozen component state can add a reproducible gain over the strongest component.

The result also clarifies what was trained. MedSigLIP, MedGemma, RadGraph, and BioLinkBERT were frozen. The project trained an 865-parameter MLP on 307,176 weighted pairwise examples. This is a genuine learning contribution, but it is not foundation-model fine-tuning. Keeping the encoders frozen isolates the fusion problem and makes the experiment feasible on consumer hardware.

### RQ2: Is the gain alignment-specific?

Yes under the prespecified negative control. Aligned nDCG@10 exceeded all 100 fixed-point-free shuffled-image runs, with plus-one p = 0.009901. Because the complete visual state was recomputed, the difference cannot be attributed to reusing aligned features or changing only a final scalar. The correct image contains case-specific information that changes the rank ordering of historical reports.

The control does not establish that MedSigLIP localizes each pathology or that the retrieved cases match a radiologist's concept of similarity. It establishes a narrower and important fact: the observed retrieval quality depends on the correct query image rather than only on the indication-question text or generic properties of chest radiographs.

### RQ3: Does retrieval improvement transfer to downstream QA?

Yes for learned multimodal RAG versus no retrieval and text-only RAG. G3 improved Token-F1 over G0 by +0.039244 with a confidence interval above zero, and it improved over BM25 RAG by +0.036856. The same target image and generator were used in every condition, so the difference is associated with the supplied historical reports rather than a generator change.

The answer requires an important qualification. G3 exceeded fixed multimodal G2 by only +0.005713 and the interval crossed zero. The learned reranker's confirmed retrieval advantage did not yield confirmed answer-level superiority over fixed multimodal retrieval. Top-3 sets may overlap, the generator may not exploit small ordering improvements, and Token-F1 may be insensitive to some clinically meaningful differences. The study therefore supports retrieval-to-QA transfer at the multimodal-versus-no-retrieval level, not a universal monotonic relationship between nDCG and final answer quality.

### RQ4: Can a bounded agent reduce unsupported historical-support claims?

Yes according to the frozen automated checker. The agent reduced unsupported historical-support rows from 16.42% to 0% through one backup route or historical-evidence abstention while leaving the target answer unchanged. The trace shows exactly why each row was accepted, retried, revised, or stripped of historical support.

This is useful but narrower than diagnosis verification. A zero unsupported-history rate was possible because the agent could remove the evidence claim. It does not establish that the remaining target-image answer was correct. The system is appropriately described as an evidence-control agent, not an autonomous diagnostic agent.

### RQ5: What limitations remain?

The principal limitations are same-source near duplicates, report-derived rather than physician-derived similarity, qrel-construct dependence, researcher-written questions, low absolute Token-F1, incomplete structured outputs, automated verification, unavailable identifier-level patient auditing, one image passed to the generator, and no independent clinician evaluation. Robustness analyses constrain rather than erase these limitations. They define the boundary between a reproducible master's research contribution and a clinically validated system.

## 5.2 Interpretation of the Evidence Chain

The most important finding is not a single score. It is the agreement of several controls. Image-image retrieval strongly exceeded generic-question BM25. The learned MLP produced a smaller but statistically resolved improvement over image-image. Shuffled images substantially reduced the learned score. The ordering remained after near-duplicate exclusion and under label-only, fact-only, and combined qrels. A modern dense model improved over BM25 but did not approach R4, and R4 was almost invariant to fixed wording changes. Multimodal historical reports improved MedGemma Token-F1, and the agent then reduced unsupported claims about those reports. Together, these results support a coherent retrieval chain from aligned visual input to more controlled evidence use.

The chain also contains productive negative results. BM25 alone barely changed QA relative to no retrieval. Fixed score fusion underperformed the strongest image component. Learned retrieval did not clearly outperform fixed retrieval at the final QA stage. F1-RadGraph did not show a learned-RAG advantage over image-only generation or fixed multimodal RAG. More than two fifths of G3 generations lacked a complete JSON object, and robust reparsing confirmed that truncation could not be repaired by a better extractor. These results prevent the thesis from becoming a simple demonstration in which every added component appears beneficial. They identify where engineering complexity is justified and where it is not.

The distinction between target evidence and historical analogy is central. A retrieved report may be highly similar and its summary may be faithfully cited, yet it still belongs to another patient. The generator must inspect the target image and use historical cases only as contextual analogies. This is why the output separates `answer`, `target_image_findings`, and `historical_support`. The distinction is also why the agent checks the historical field only. Treating a historical report as proof would create precisely the cross-case contamination risk that the earlier studies exposed.

## 5.3 Research Contributions

### 5.3.1 Task Contribution

The final task is a new-case, other-patient similar-case QA problem rather than target-report recovery. It preserves the original thesis title because it remains retrieval-augmented question answering over paired radiology images and reports, but it gives the pairing a clinically defensible role. Historical images are paired with their reports in the bank; the target image is paired with a hidden report only for evaluation.

### 5.3.2 Method Contribution

The method combines frozen biomedical foundation models with a small learned reranker. Training focuses on the research question: how to combine retrieval state. The pairwise objective uses graded differences, and the model is tested against the strongest component under a frozen protocol. This is more informative than replacing every model or adding an opaque agent framework.

### 5.3.3 Evaluation Contribution

The evaluation integrates graded nDCG, component baselines, case-grouped bootstrap intervals, shuffled-image negative controls, retrieval-to-QA transfer, near-duplicate exclusion, qrel variants, a modern dense text baseline, wording perturbations, F1-RadGraph, structured-output reparsing, automated historical support, runtime, and artifact hashes. Each measure has a stated scope. The result is an auditable evidence chain rather than a single favorable metric.

### 5.3.4 Agent Contribution

The bounded agent demonstrates that agentic behavior can be defined as a controlled state machine rather than an unbounded collection of language-model roles. One retry, explicit reasons, citation tracking, and evidence abstention are sufficient to create a meaningful agentic contribution. The design is reproducible and easier to audit than a general-purpose orchestration framework.

### 5.3.5 Reproducibility Contribution

The repository records protocol chronology, deterministic splits, source and cohort fingerprints, model revisions, checkpoint hashes, aggregate results, tests, and dashboard behavior. Local retention of source-derived artifacts respects practical data and repository boundaries while public hashes enable integrity checks. The technical freeze prohibits outcome-driven V9 retuning.

## 5.4 Practical and Theoretical Significance

Practically, similar-case retrieval can expose historical reports that may help a model describe or contextualize a new image. The value is not automatic diagnosis. It is access to analogous language and patterns with explicit provenance. In a future clinician-facing system, retrieved cases could support education, audit, differential consideration, or report drafting, provided that access control, external validation, and human oversight are established.

Theoretically, the study shows that multimodal fusion should be understood as a conditional ranking problem. A modality can be strong on average yet weak for a particular candidate or question. Fixed addition assumes stable reliability across queries. The learned reranker uses observable retrieval state to combine channels, which explains its improvement over fixed fusion. At the same time, the small downstream difference between learned and fixed RAG shows that retrieval metrics and generation metrics are connected but not equivalent.

The study also extends the concept of grounding. Local faithfulness asks whether a statement follows from the context. Case alignment asks whether that context is appropriate for the target. Historical-evidence control asks whether claims about an analogy are supported. Target-image correctness asks whether the answer reflects the current patient's radiograph. Clinical safety asks whether the system can be trusted in practice. These layers require different evidence and should not be collapsed into one support score.

## 5.5 Limitations

### Same-source and modality scope

All final results come from OpenI/IU-Xray chest radiographs. The study improves within-source spectrum coverage but does not demonstrate external generalization. Scanner distributions, reporting styles, disease prevalence, and image quality may differ in MIMIC-CXR, CheXpert Plus, PadChest, or local hospital data. Other radiology modalities such as CT, MRI, ultrasound, or mammography would require different encoders, study-level aggregation, and relevance definitions.

The supplemental audit found substantial normalized-report similarity between Train and Test. R4 remained first after excluding Test reports with cosine similarity at least 0.95, but the prevalence of exact and near duplicates means that the benchmark should not be presented as a clean estimate for a different hospital or reporting environment. Image dHash collisions were also frequent, although perceptual similarity in standardized chest radiographs cannot establish patient duplication.

### Patient identity boundary

The source design reports one study per patient, but stable patient identifiers were unavailable for independent verification. The split is case-ID disjoint. It should not be described as identifier-verified patient-disjoint evaluation.

### Operational relevance

The primary gain combines active dataset labels and RadGraph facts. It is reproducible and avoids rewarding negative-label agreement, but it reflects report similarity rather than physician judgment of which historical cases are useful. Label extraction and RadGraph annotation can both contain errors. Normal-normal gain is necessarily coarse. R4 remained first under label-only and fact-only qrels, but its fact-only advantage over image retrieval was only 0.002950 nDCG@10, demonstrating construct-dependent effect size.

### Question and reference provenance

The findings and impression questions are fixed templates, not physician-authored questions arising during care. References are report sections and Token-F1 rewards lexical overlap. Fixed paraphrase tests showed that R4 was stable, but researcher-written variants cannot substitute for clinician language. Clinically equivalent terminology and valid image observations absent from the report can be under-rewarded. ReportQA, RadQA, or clinician-authored evaluation would complement the current design but cannot be substituted without respecting licensing and access rules.

### Generation reliability

Absolute Token-F1 remained low, and strict JSON completeness peaked at 57.23%. Robust formatting-only reparsing recovered no additional rows, indicating that token-ceiling truncation rather than a simple regular-expression defect caused most failures. The maximum generation length and 4-bit local configuration may constrain output, but changing them after confirmation would invalidate the freeze. A stronger model or constrained decoder could improve usability, yet must be evaluated in a new protocol.

### Automated verification

BioLinkBERT-MedNLI is not a clinical gold standard. The agent's zero final unsupported rate partly results from removing the historical-support field. The checker does not inspect image pixels and cannot validate the target diagnosis. Over-rejection and under-detection remain possible.

### Human evaluation

The author completed the deterministic 24-case review and accepted the tool-assisted labels. This is not independent radiologist adjudication, blinded review, or inter-rater evaluation. The selected-case counts cannot estimate cohort-wide error prevalence. Independent clinical evaluation remains the strongest missing validation layer.

### Clinical and deployment boundary

The prototype lacks prospective workflow integration, calibration, privacy/security assessment, monitoring, access control, fairness analysis, and regulatory evaluation. It is not a medical device or decision-support product.

## 5.6 Future Work

The highest-priority extension is independent clinical review. Radiologists could judge whether retrieved cases are clinically similar, whether target-image answers are correct and complete, whether historical analogies are useful, and whether abstention is appropriate. Multiple reviewers would permit inter-rater agreement and adjudication.

The second priority is external replication. A new protocol could use a licensed subset of MIMIC-CXR, CheXpert Plus, PadChest, or another paired image-report source. The historical bank need not contain millions of images; a prespecified, sufficiently powered patient-disjoint subset could test whether the direction of the V9 effects transfers. Source-specific labels and patient identifiers would need a fresh audit.

Third, generation can be improved without changing the completed V9 result. Constrained JSON decoding, larger token budgets, multi-view image input, report-section-aware prompts, and alternative local multimodal generators should be compared on a new development and confirmation split. Retrieval and generation changes should remain factorial so that causal attribution is preserved.

Fourth, relevance can be strengthened with physician-labeled pairwise similarity or a task-specific utility label. Such judgments could train the reranker directly and reveal whether report-derived relevance correlates with actual clinical usefulness. Calibration and selective prediction should accompany any such study.

Fifth, the agent can be expanded cautiously. A future version could route by question type, retrieve at both case and evidence-span levels, compare independent visual and historical claims, and request human review under uncertainty. The loop should remain bounded and every tool action should be logged. LangChain or another orchestration library is unnecessary unless it removes concrete engineering complexity; the scientific value lies in the state and evidence contract, not the framework name.

## 5.7 Conclusion

This thesis developed a complete multimodal similar-case RAG workflow for new-patient chest-radiograph question answering. From 3,851 OpenI cases, it constructed a deterministic Train/Validation/Test study and a 2,608-case historical bank. Frozen BM25 and MedSigLIP channels were compared with a current dense text baseline, fixed fusion, and a trained 865-parameter reranker. The learned reranker achieved a statistically resolved nDCG@10 gain over image-only retrieval, and aligned images substantially exceeded 100 shuffled controls. Its ordering remained first after strict near-duplicate exclusion, across three qrel variants, and under fixed wording changes. Retrieved multimodal reports improved MedGemma Token-F1 over the same generator without retrieval and over text-only RAG. A bounded agent then suppressed unsupported historical-support claims through one retry or evidence abstention.

The contribution is deliberately scoped. The system does not prove diagnostic accuracy, clinical safety, or external validity. It demonstrates that correct image alignment matters, that learned fusion can outperform the strongest frozen retrieval component, that retrieval gains can transfer to downstream QA, and that agentic evidence control can be made bounded and auditable. Equally important, it records where the evidence remains weak: fixed fusion can degrade retrieval, learned retrieval need not produce confirmed learned-QA superiority, F1-RadGraph did not establish broad G3 superiority, structured output can fail through truncation, automated verification is not clinical judgment, and independent clinical evaluation is still required.

The final value of the research is therefore not a claim that an agent can replace a radiologist. It is a reproducible account of how images, clinical text, historical reports, learned fusion, generation, and evidence control interact in a medically sensitive RAG pipeline, together with the controls needed to distinguish technical improvement from unsupported clinical claims.

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

Jeong, J., Tian, K., Li, A., et al. (2023). Multimodal image-text matching improves retrieval-based chest X-ray report generation. *Medical Imaging with Deep Learning*.

Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*.

Pal, A., Umapathi, L. K., and Sankarasubbu, M. (2023). Med-HALT: Medical domain hallucination test for large language models. *Proceedings of CoNLL*, 314-334.

Park, J., Yoon, B., Kim, S., and Choi, K. (2026). RA-RRG: Multimodal retrieval-augmented radiology report generation with key phrase extraction. *Findings of the Association for Computational Linguistics: ACL 2026*, 5029-5048.

Qwen Team. (2025). Qwen3 Embedding: Advancing text embedding and reranking through foundation models. https://qwenlm.github.io/blog/qwen3-embedding/

Robertson, S., and Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval, 3*(4), 333-389.

Romanov, A., and Shivade, C. (2018). Lessons from natural language inference in the clinical domain. *Proceedings of EMNLP*, 1586-1596.

Singhal, K., Azizi, S., Tu, T., et al. (2023). Large language models encode clinical knowledge. *Nature, 620*, 172-180.

Soni, S., Gudala, M., Pajouhi, A., and Roberts, K. (2022). RadQA: A question answering dataset to improve comprehension of radiology reports. *Proceedings of LREC 2022*, 6250-6259.

Sun, L., Zhao, J. J., Han, W., and Xiong, C. (2025). Fact-aware multimodal retrieval augmentation for accurate medical radiology report generation. *Proceedings of NAACL 2025*, 643-655.

Xiong, G., Jin, Q., Lu, Z., and Zhang, A. (2024). Benchmarking retrieval-augmented generation for medicine. *Findings of ACL 2024*, 6233-6251.

# Appendices

## Appendix A: Final V9 Public Evidence

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

The dashboard accepts a target chest radiograph, indication, and question; retrieves Top-3 similar reports from the frozen 2,608-case bank; optionally generates a MedGemma answer; and shows bounded-agent evidence actions. It describes the reports as historical analogies. It must not claim to locate the target patient's own report, diagnose the image, or replace radiologist review.

## Appendix F: Version Boundary

V5-V7 are preliminary controlled studies and remain frozen. V8 ended in a documented development no-go. V9 is the final primary study. Supplemental V9 audits were committed after the technical freeze and before their own outcomes; they did not change a frozen model, prompt, threshold, metric, split, case, or primary conclusion. Reporting edits do not alter technical artifacts.

The following appendices preserve the detailed V5 controlled study for traceability. They are formative evidence and do not replace the V9 primary study.

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

The four principal confirmation retrieval conditions are shown in Table 4.1.

Table 4.1. Retrieval results under four principal input conditions

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

The same generator and checker are compared after report-only and multimodal retrieval in Table 4.2.

Table 4.2. End-to-end QA comparison

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

Table 4.3. Runtime and computational cost

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
