from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "P2_V5_INTEGRATED_MANUSCRIPT.md"
OUTPUT = ROOT / "docs" / "P2_V9_FINAL_MANUSCRIPT.md"


ABSTRACT = r"""## Abstract

Retrieval-augmented generation can provide language models with traceable evidence, but conventional text-only retrieval is poorly matched to a new radiology case whose formal report has not yet been written. This research develops and evaluates a multimodal, similar-case medical question-answering workflow in which a target chest radiograph, pre-report clinical indication, and question retrieve other-case image-report pairs from a frozen historical bank. The target report is hidden from every inference component and is used only as an offline reference. The study uses 3,851 paired OpenI/IU-Xray examinations. A deterministic, report-indexed stratified split assigns 2,631 cases to Train, 376 to Validation, and 752 to an untouched V9 Test partition; 2,608 report-bearing Train cases form the shared historical bank. Report-derived graded relevance combines active-label similarity and RadGraph fact overlap. The retrieval comparison includes BM25 text retrieval, MedSigLIP image-image retrieval, image-report retrieval, fixed multimodal fusion, and a project-trained 865-parameter multilayer perceptron reranker. The reranker is trained with 307,176 weighted pairwise examples while all foundation models remain frozen.

On 752 Test cases, the learned reranker achieved nDCG@10 of 0.327942 and MRR of 0.331968. It exceeded the strongest frozen image-only component by 0.012381 nDCG@10, with a 95% case-bootstrap confidence interval of [0.009226, 0.015584]. Under 100 fixed-point-free shuffled-image controls, aligned retrieval achieved 0.327942 compared with a shuffled mean of 0.220370 (plus-one p = 0.009901), demonstrating that the gain depended on correct image alignment. Downstream evaluation used 685 complete-reference Test cases, two fixed questions per case, four conditions, and 5,480 local MedGemma 1.5 generations. Learned multimodal RAG achieved Token-F1 of 0.184803 versus 0.145559 without retrieval, a difference of 0.039244 with 95% confidence interval [0.032572, 0.045745]. Its advantage over fixed multimodal RAG was numerical but not statistically resolved. A bounded evidence-control agent reduced automatically unsupported historical-support rows from 16.42% to 0% through one backup route or evidence-field abstention while preserving the target-image answer by design.

The results support a scoped conclusion: correctly aligned chest-image information improves report-derived similar-case retrieval, and retrieved multimodal historical evidence improves report-reference consistency over the same generator without retrieval. The work does not establish physician-adjudicated similarity, diagnostic accuracy, patient safety, external generalization, or deployment readiness. Its main contribution is an auditable evaluation of retrieval, generation, alignment, and evidence-control stages in a realistic new-case multimodal RAG task.
"""


CHAPTER_1 = r"""# Chapter 1: Introduction

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

The seventh contribution is methodological transparency. Protocols were committed before their corresponding outcome stages; large source-derived artifacts remain local; public summaries contain hashes and aggregate metrics; and 206 automated tests verify core behavior. The work preserves negative findings, including the weakness of BM25, the underperformance of naive fixed fusion, incomplete JSON output, low absolute Token-F1, and the absence of clinical human evaluation.

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
"""


CHAPTER_2_ADDENDUM = r"""
## 2.12 Similar-Case Multimodal RAG and the Final Research Gap

The closest line of work combines chest-image retrieval with report generation. CXR-RePaiR uses a CLIP-style image-to-report retriever to construct reports from retrieved exemplars. X-REM adds coarse retrieval, learned image-text matching, and an NLI filter. FactMM-RAG uses factual pairs derived from CheXbert and RadGraph to improve factual report generation. These systems establish that historical image-report pairs can support image-conditioned language generation. They do not, however, directly test a question-conditioned new-patient workflow in which the target image, clinical indication, and medical question jointly retrieve other cases and the final output explicitly separates target observations from historical analogies.

Recent multimodal and agentic systems also motivate a stricter evaluation boundary. Concept-enhanced RAG methods combine visual embeddings with medical concepts; agentic radiology systems separate planning, retrieval, generation, and validation roles; and generated-report approaches use an intermediate radiology description to improve VQA. These designs show that orchestration can improve modularity, but additional agents do not automatically create stronger evidence. An agent may simply repeat the same unsupported claim through more steps. The relevant contribution is therefore not the number of roles but whether actions are bounded, inputs are permitted, failures are traceable, and abstention is available.

The literature leaves four connected gaps. First, many retrieval-augmented radiology studies focus on report generation rather than answering a user question. Second, evaluations often compare multimodal fusion with text baselines but do not require superiority over the strongest individual visual component. Third, aligned-image gains are rarely challenged by complete shuffled-image recomputation. Fourth, evidence verification is commonly reported without separating support for a historical analogy from correctness about the target image.

V9 addresses these gaps with a scoped design. It compares text-only, image-only, image-report, fixed fusion, and learned fusion over one bank; trains only a small reranker; evaluates alignment with 100 fixed-point-free controls; uses the same multimodal generator across retrieval conditions; and limits the agent to historical-evidence checking. The final research gap is consequently not "whether RAG can be used in radiology." It is whether correctly aligned visual evidence can produce reproducible gains in other-case retrieval and whether those gains transfer to reference-consistent QA under an auditable evidence contract.
"""


V9_METHODS = r"""
## 3.14 Transition from Preliminary Studies to the Final V9 Study

Sections 3.1-3.13 describe the preliminary controlled study that established the project's evaluation vocabulary: target alignment, indication shortcut, shuffled-image control, retrieval-to-generation transfer, and automated-verifier disagreement. Those experiments retrieved the paired target report from a controlled corpus and are retained because they explain the methodological development. They are not the final task. V9 removes the target report from the bank and treats every retrieved report as an other-case historical analogy.

The transition was protocol governed. V5-V8 artifacts remained frozen. A technology-reuse audit defined which repository components could be retained and which external methods could be independently reimplemented. A full-source split amendment was committed before the final split was instantiated. RadGraph preprocessing, MedSigLIP development, learned-reranker training, retrieval confirmation, QA confirmation, and qualitative extraction each had separate frozen boundaries. No Test outcome was used to alter a model, prompt, threshold, metric, case, or hypothesis.

## 3.15 Data Source, Eligibility, and Deterministic Partition

The source artifact contains 3,851 paired OpenI/IU-Xray studies. Cases were classified from the normalized `problems` field before splitting. There were 1,379 report-indexed normal cases, 2,380 report-indexed abnormal cases, and 92 report-index indeterminate cases. "Report-indexed" denotes the dataset annotation and does not represent new physician review. The 92 empty or `no indexing` cases were excluded from the primary stratifiable frame rather than reclassified. The resulting primary universe contained 3,759 cases.

The predefined split approximated 70/10/20 while preserving the exact normal/abnormal totals. Train contained 965 normal and 1,666 abnormal cases (2,631 total). Validation contained 138 normal and 238 abnormal cases (376 total). Test contained 276 normal and 476 abnormal cases (752 total). All 262 previously unused eligible cases identified by the project-history audit were placed in Test, and deterministic SHA-256 ranking selected the remaining 490 Test cases and all Validation cases. The split seed was 7029 with domain-separated payloads for Test supplementation and Validation selection. Canonical case identifiers were stripped strings, UTF-8 encoded, sorted by digest and identifier, and sampled without replacement.

Twenty-five source cases had neither findings nor impression. Twenty-three occurred in Train and two in Validation; none occurred in Test. To keep every comparator on the same evidence universe, the 23 empty-report Train cases were excluded from all retrieval conditions, including image-only retrieval. The shared historical bank therefore contained 2,608 report-bearing Train cases. Primary Validation relevance evaluation used 374 report-bearing queries. All 752 Test queries had usable report-derived relevance references. The downstream QA frame required both findings and impression and contained 685 Test cases.

The source publication describes one study per patient, but released patient identifiers were unavailable in the processed data. Case-ID disjointness was verified. The thesis does not claim independently verified patient-level separation. Validation and Test queries nevertheless retrieve only from the Train bank, so their own study reports are absent from inference.

## 3.16 Report-Derived Graded Relevance

V9 requires a graded notion of clinical similarity because exact report identity is neither available nor desirable in the historical bank. Offline relevance was constructed from hidden target-report annotations and candidate-report annotations. It was never exposed to retrieval, generation, verification, or agent routing.

Active label similarity used weighted Jaccard overlap. Positive labels received weight 1.0, uncertain labels 0.5, and negative or missing labels 0.0. Shared absent abnormalities therefore contributed no reward. If both reports had no active abnormal label, label similarity was defined as 1.0; if only one had active abnormalities, it was 0.0. This special case allowed report-indexed normal cases to be evaluated without rewarding a large vector of explicitly absent labels.

RadGraph XL was run locally on normalized findings plus impression. Entities and relations were flattened with the deterministic complete-reward representation. Report-derived fact similarity was the F1 overlap of the normalized fact sets. The continuous primary gain was:

```text
gain(query, candidate)
  = 0.60 × active-label similarity
  + 0.40 × RadGraph fact similarity
```

The score was constrained to [0,1]. nDCG@10 used the continuous gain. Binary MRR and Recall diagnostics used a frozen gain threshold of 0.50. Label-only and fact-only results were sensitivity analyses. The operational relevance supports reproducible ranking comparisons but is not physician-adjudicated similarity.

## 3.17 Retrieval Components and Fixed Fusion

R0 was BM25 over historical findings and impression, queried with the clinical indication and one of three fixed question templates. R1 was cosine similarity between the normalized mean MedSigLIP embedding of the target image views and the normalized mean embedding of each historical study's image views. R2 was MedSigLIP cross-modal similarity between the target image representation and the historical report representation.

Historical reports were split into sentence-aware, section-prefixed chunks of at most 64 MedSigLIP tokens. Each chunk was encoded and normalized. Validation compared normalized mean chunk aggregation with maximum image-to-chunk similarity. Mean aggregation was retained because the maximum policy did not satisfy the frozen material-improvement rule. For multi-view studies, each readable view was encoded independently, normalized, averaged, and normalized again. No view was selected based on outcome.

Each retrieval channel was independently min-max normalized across all 2,608 candidates for a query. Constant channels mapped to zero. The fixed fusion grid evaluated nonnegative weight triples in increments of 0.25 that summed to one. A valid multimodal candidate required positive BM25 weight and positive total image weight. Validation nDCG@10 selected the final R3 weights under a simplicity rule that favored more BM25 weight among candidates within 0.005 of the maximum. Canonical case ID resolved every ranking tie.

## 3.18 Learned Multimodal Reranker

The project-trained component was deliberately small. A nine-feature vector represented each query-candidate state: independently normalized BM25, image-image, and image-report scores; normalized reciprocal rank under each component; and three one-hot indicators for findings, impression, or acute question type. Raw text, image pixels, identifiers, filenames, labels, RadGraph facts, references, and QA outcomes were prohibited as features.

The 2,608 Train-bank cases were deterministically assigned to 1,600 pairwise-fit queries, 500 internal early-stop queries, and 508 bank-only cases while remaining available as historical candidates. For a fit query, candidate mining formed a deterministic union of Top-32 candidates under each component and offline relevance, plus Bottom-32 relevance candidates. The eight highest-gain and eight lowest-gain members formed ordered pairs when their gain difference was at least 0.05. Pair weights equaled the gain difference. This generated 307,176 weighted pairwise examples.

Two scorers were prespecified: a linear 9-to-1 model and a multilayer perceptron `9 → 32 → 16 → 1` with ReLU activations. Both used weighted pairwise softplus loss, AdamW, learning rate 0.001, weight decay 0.0001, batch size 4,096 pairs, at most 30 epochs, seed 7030, and internal early stopping. Validation selected the architecture. The MLP was promoted because it met the frozen margin over fixed fusion and also exceeded the strongest individual component. Its 865 parameters were the only learned model parameters in the final system. Foundation encoders and language models remained frozen.

## 3.19 Retrieval Confirmation and Alignment Control

Five frozen systems were evaluated once on 752 Test cases and three fixed questions: BM25, image-image, image-report, fixed fusion, and the learned MLP. The primary metric was case-grouped equal-question nDCG@10. The primary paired comparison was learned MLP minus image-image because image-image was the strongest frozen component on Validation. A 10,000-iteration case bootstrap with seed 7031 produced a percentile 95% confidence interval. Confirmed superiority required the lower bound to be greater than zero.

Alignment dependence was evaluated through 100 deterministic, unique, fixed-point-free wrong-image assignments. Test cases were ordered by a domain-separated SHA-256 rule, and cyclic shifts 1-100 supplied complete image-view sets from other Test cases. For each assignment, image-image scores, image-report scores, normalization, ranks, features, MLP scores, and rankings were recomputed. BM25 remained attached to the original indication and question. The plus-one randomization p-value counted how many shuffled nDCG@10 values equaled or exceeded aligned performance. A predefined 262-case project-history-untouched subset was reported only as sensitivity analysis.

## 3.20 Downstream Multimodal Question Answering

The downstream frame contained 685 Test cases with nonempty findings and impression. Each contributed a findings question and an impression question, producing 1,370 questions. The acute question used for retrieval was excluded from generation scoring because the dataset lacked a physician-adjudicated binary acute-abnormality reference.

All four generation conditions used the same `google/medgemma-1.5-4b-it` revision, local 4-bit NF4 inference, deterministic decoding, maximum 192 new tokens, target indication, question, and one target chest image. The image policy preferred a frontal view and then lexicographic filename order. G0 received no historical report. G1 received Top-3 BM25 reports. G2 received Top-3 fixed-fusion reports. G3 received Top-3 learned-reranker reports. Historical images influenced retrieval but were not additional generator pixel inputs.

The prompt required JSON fields for answer, target-image findings, supporting case IDs, historical support, uncertainty, and abstention. It stated explicitly that historical reports were analogies and not proof about the target patient. Parser failures and token-ceiling outputs were retained; no selective regeneration was allowed. The primary metric was case-grouped equal-question Token-F1 against hidden target findings or impression. G3 minus G0 was the primary comparison, with 10,000 case-bootstrap iterations. Exact match, JSON completeness, per-question effects, subgroup effects, latency, tokens, and memory were secondary.

## 3.21 Bounded Evidence-Control Agent

G4 was a bounded control layer applied after G3. Its verifier checked only whether statements in the `historical_support` field were supported by the cited historical reports. It did not verify the `answer` or `target_image_findings` against the target image. If G3 historical support failed the frozen checker, the agent could perform one deterministic retry using the frozen image-image R1 Top-3 route. If the backup evidence still failed, it removed the historical-support statement and citations, retained the target-image answer, and recorded historical-evidence abstention.

The agent had no internet access, no model-selection authority, no threshold changes, no target-report access, and no unbounded loop. Every initial route, cited ID, support score, retry, revision, abstention, and reason was recorded. The primary agent outcome was the paired change in automated unsupported historical-support rate. A Token-F1 noninferiority margin of -0.01 guarded against unintended answer modification, although the target answer was preserved by design.

## 3.22 Reproducibility, Ethics, and Qualitative Review

The implementation used local CUDA inference and preserved model revisions, configuration files, checkpoint hashes, result hashes, split fingerprints, and protocol commits. Large source-derived texts, image pixels, vectors, checkpoints, prompts, and per-row generations remained local under repository policy. Aggregate summaries, source-neutral code, hashes, tests, and a lightweight case index were public. The final suite contained 206 passing automated tests.

A qualitative protocol was committed before deterministic extraction. Twenty-four cases include six largest G3 gains, six largest losses, six agent retry cases, and six historical-evidence abstentions. Tool-assisted proposals remain pending researcher review. They are not reported as completed human findings. No radiologist evaluated similarity, retrieved reports, target-image answers, or agent decisions. The study therefore reports retrospective technical performance and explicitly excludes claims of diagnostic safety, clinical utility, or deployment readiness.
"""


V9_RESULTS = r"""
## 4.8 Final V9 Retrieval Confirmation

The V9 Test evaluation contained 752 cases, three questions per case, five systems, and 11,280 ranking rows. Every system ranked the same 2,608-case historical bank. BM25 produced nDCG@10 of 0.134156 and MRR of 0.083542. Image-image retrieval was substantially stronger at 0.315561 nDCG@10 and 0.328270 MRR. Image-report retrieval achieved 0.274069 and 0.256032 respectively. Fixed multimodal fusion reached 0.246935 nDCG@10 and 0.211322 MRR, below the image-only component. The learned MLP produced the strongest nDCG@10, 0.327942, and MRR, 0.331968.

The primary learned-minus-image difference was +0.012381 nDCG@10. Its 10,000-iteration case-bootstrap 95% confidence interval was [0.009226, 0.015584]. Because the lower bound was greater than zero, the frozen superiority criterion passed. The magnitude is modest relative to the large image-versus-text gap, but it is consistent and attributable to the learned combination rather than to foundation-model fine-tuning.

The fixed-fusion result is an important negative finding. More modalities did not automatically improve ranking. Equal or validation-selected weighted addition can dilute a strong visual signal with a weak text channel when generic questions and short indications provide limited discrimination. The learned model recovered a gain because it used scores, reciprocal ranks, and question type to condition candidate ordering. This supports learned fusion as the methodological contribution, while cautioning against describing any multimodal combination as inherently superior.

The 262-case strict project-history-untouched subset showed the same direction. In that subset, BM25, image-image, image-report, fixed fusion, and learned MLP achieved nDCG@10 values of 0.129956, 0.411812, 0.331673, 0.288307, and 0.419901 respectively. These values were not used for selection and are not a separate confirmatory family. They reduce concern that the full result was driven only by cases encountered in earlier project stages.

## 4.9 Alignment-Specific Image Contribution

Aligned R4 nDCG@10 was 0.327942. Across 100 complete wrong-image recomputations, mean nDCG@10 was 0.220370, standard deviation 0.004900, and the 2.5th and 97.5th percentiles were 0.210726 and 0.231474. The aligned score exceeded every shuffled assignment. The plus-one p-value was 0.009901.

This control is stronger than substituting one arbitrary image or shuffling only the final score. Every visual component and all derived learned features were recomputed under the wrong image. The original clinical indication and question were retained. The result therefore shows that correct target-image alignment materially influenced similar-case ranking. It does not prove that every retrieved case is clinically appropriate, but it rules out an explanation based only on text or a generic image prior.

## 4.10 Downstream QA Transfer

The QA study used 685 complete-reference Test cases, 1,370 questions, four systems, and 5,480 local MedGemma generations. G0 target-image generation without retrieval achieved Token-F1 0.145559 and complete JSON rate 42.04%. G1 BM25 RAG achieved 0.147947 and 39.42%. G2 fixed multimodal RAG achieved 0.179090 and 46.50%. G3 learned multimodal RAG achieved the highest Token-F1, 0.184803, and complete JSON rate, 57.23%.

The primary G3-minus-G0 Token-F1 difference was +0.039244, with 95% confidence interval [0.032572, 0.045745]. The criterion therefore passed. The strict 262-case sensitivity subset also favored G3 by +0.047187, interval [0.034911, 0.059704]. BM25 RAG differed from no retrieval by only +0.002388, interval [-0.003189, 0.008120]. Fixed multimodal RAG improved over no retrieval by +0.033532, interval [0.027790, 0.039502]. Learned multimodal RAG improved over BM25 by +0.036856, interval [0.030410, 0.043524].

G3 exceeded G2 by +0.005713, but the interval [-0.000958, 0.012280] crossed zero. The thesis therefore claims that learned multimodal RAG improved over no retrieval and text-only RAG; it does not claim confirmed downstream superiority over fixed multimodal RAG. Retrieval superiority and generation superiority are distinct. A different Top-3 ordering may not change the available evidence enough to produce a resolved answer-level difference.

The effect was larger for impression questions (+0.059903) than findings questions (+0.018585). It remained positive for report-indexed normal cases (+0.060784) and abnormal cases (+0.025570). These are prespecified sensitivity results rather than independent confirmatory hypotheses. One interpretation is that retrieved analogies help synthesize an impression more than enumerate visible findings, but this explanation remains inferential because no clinician reviewed individual responses.

Absolute performance remained low. MedGemma sometimes reached the 192-token limit or returned non-strict JSON. Only 57.23% of G3 rows contained the complete requested object. All outputs were retained under a tolerant parser rather than selectively regenerated. These failures matter because a system can improve mean Token-F1 while remaining unsuitable for deployment.

## 4.11 Bounded Agent Results

The G4 agent evaluated all 1,370 G3 rows. Before control, 16.423% of rows contained historical-support statements that the frozen automated checker did not substantiate from the cited reports. After one optional R1 backup route or evidence-field removal, the final automated unsupported historical-support rate was 0%. The paired reduction was 16.423 percentage points, with 95% case-bootstrap interval [14.4526, 18.4672] percentage points in magnitude.

The agent retried 16.423% of rows, revised historical support in 17.299%, and abstained from historical evidence in 15.985%. Mean retrieval calls increased from one to 1.164. Token-F1 remained 0.184803 because the target answer was preserved by design. The zero final unsupported rate is therefore partly structural: when no support could be established, the field was removed. It demonstrates auditable claim suppression and abstention, not that all retained target-image answers were correct.

## 4.12 Computational Cost and Artifact Integrity

The 5,480 local MedGemma generations required 13,496 seconds, or approximately 3.75 hours, at 0.406 records per second. Peak allocated GPU memory was 5,184.5 MiB on the RTX 5070 Laptop GPU with batch size eight. The small reranker trained on CPU-sized feature tensors and introduced negligible inference cost compared with MedSigLIP and MedGemma.

Frozen hashes identify the principal artifacts. Retrieval rows hash to `baa56924928b144c9b877b8e2218e04d17df6b77a6f794ed3830f7ccf3e449fd`. The MLP checkpoint hash is `8afa68a48de9d6c9128d190f1368d0d45d41a958e5eb12787d7e725e7eb09efa`. The Top-3 ranking pack hash is `28639821abc5fba8189c7c0149822ed0e3935325d0136578803155cc5a4ebd9b`. QA raw rows hash to `89c69c9a27e393c93c85e572587b330f908598e835cb8162a8678cd15ba512b4`, and agent rows hash to `9cc8b4513f2ef12f7e849d7b5853a79ef07495b022699c5a84785d1d94624fc1`.

## 4.13 Qualitative Status and Results Summary

The deterministic 24-case review pack is complete, but researcher review remains pending. Provisional extraction shows heterogeneous gains and losses, cases in which retrieval improvement did not guarantee a reference-consistent answer, retries that occasionally recovered historical support, and abstentions that removed unsupported analogies. These observations guide review but are not presented as human-confirmed findings.

The final quantitative evidence chain is complete. Correct images improved retrieval relative to shuffled images. The learned reranker exceeded the strongest frozen component. Learned multimodal RAG improved reference consistency over no retrieval and text-only RAG. The bounded agent suppressed unsupported historical evidence. At the same time, fixed fusion underperformed image-only retrieval, learned QA did not clearly beat fixed QA, structured output was incomplete, and no clinical human score was obtained.
"""


CHAPTER_5 = r"""# Chapter 5: Discussion and Conclusion

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

The principal limitations are same-source evaluation, report-derived rather than physician-derived similarity, low absolute Token-F1, incomplete structured outputs, automated verification, unavailable identifier-level patient auditing, one image passed to the generator, and no independent clinician evaluation. These limitations constrain the claim but do not nullify the controlled comparisons. They define the boundary between a reproducible master's research contribution and a clinically validated system.

## 5.2 Interpretation of the Evidence Chain

The most important finding is not a single score. It is the agreement of several controls. Image-image retrieval strongly exceeded generic-question BM25. The learned MLP produced a smaller but statistically resolved improvement over image-image. Shuffled images substantially reduced the learned score. Multimodal historical reports improved MedGemma reference consistency. The agent then reduced unsupported claims about those reports. Together, these results support a coherent chain from aligned visual retrieval to more useful historical context and more controlled evidence use.

The chain also contains productive negative results. BM25 alone barely changed QA relative to no retrieval. Fixed score fusion underperformed the strongest image component. Learned retrieval did not clearly outperform fixed retrieval at the final QA stage. More than two fifths of G3 generations lacked a complete JSON object. These results prevent the thesis from becoming a simple demonstration in which every added component appears beneficial. They identify where engineering complexity is justified and where it is not.

The distinction between target evidence and historical analogy is central. A retrieved report may be highly similar and its summary may be faithfully cited, yet it still belongs to another patient. The generator must inspect the target image and use historical cases only as contextual analogies. This is why the output separates `answer`, `target_image_findings`, and `historical_support`. The distinction is also why the agent checks the historical field only. Treating a historical report as proof would create precisely the cross-case contamination risk that the earlier studies exposed.

## 5.3 Research Contributions

### 5.3.1 Task Contribution

The final task is a new-case, other-patient similar-case QA problem rather than target-report recovery. It preserves the original thesis title because it remains retrieval-augmented question answering over paired radiology images and reports, but it gives the pairing a clinically defensible role. Historical images are paired with their reports in the bank; the target image is paired with a hidden report only for evaluation.

### 5.3.2 Method Contribution

The method combines frozen biomedical foundation models with a small learned reranker. Training focuses on the research question: how to combine retrieval state. The pairwise objective uses graded differences, and the model is tested against the strongest component under a frozen protocol. This is more informative than replacing every model or adding an opaque agent framework.

### 5.3.3 Evaluation Contribution

The evaluation integrates graded nDCG, component baselines, case-grouped bootstrap intervals, shuffled-image negative controls, retrieval-to-QA transfer, strict-subset sensitivity, structured-output validity, automated historical support, runtime, and artifact hashes. Each measure has a stated scope. The result is an auditable evidence chain rather than a single favorable metric.

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

### Patient identity boundary

The source design reports one study per patient, but stable patient identifiers were unavailable for independent verification. The split is case-ID disjoint. It should not be described as identifier-verified patient-disjoint evaluation.

### Operational relevance

The primary gain combines active dataset labels and RadGraph facts. It is reproducible and avoids rewarding negative-label agreement, but it reflects report similarity rather than physician judgment of which historical cases are useful. Label extraction and RadGraph annotation can both contain errors. Normal-normal gain is necessarily coarse.

### Question and reference provenance

The findings and impression questions are fixed templates, not physician-authored questions arising during care. References are report sections and Token-F1 rewards lexical overlap. Paraphrases, clinically equivalent terminology, and valid image observations absent from the report can be under-rewarded. ReportQA, RadQA, or clinician-authored evaluation would complement the current design but cannot be substituted without respecting licensing and access rules.

### Generation reliability

Absolute Token-F1 remained low, and strict JSON completeness peaked at 57.23%. The maximum generation length and 4-bit local configuration may constrain output, but changing them after confirmation would invalidate the freeze. A stronger model or constrained decoder could improve usability, yet must be evaluated in a new protocol.

### Automated verification

BioLinkBERT-MedNLI is not a clinical gold standard. The agent's zero final unsupported rate partly results from removing the historical-support field. The checker does not inspect image pixels and cannot validate the target diagnosis. Over-rejection and under-detection remain possible.

### Human evaluation

The deterministic 24-case pack awaits researcher review, and no independent radiologist adjudication was completed. The thesis must not report assistant proposals as human findings or claim inter-rater reliability. Human evaluation remains the strongest missing validation layer.

### Clinical and deployment boundary

The prototype lacks prospective workflow integration, calibration, privacy/security assessment, monitoring, access control, fairness analysis, and regulatory evaluation. It is not a medical device or decision-support product.

## 5.6 Future Work

The highest-priority extension is independent clinical review. Radiologists could judge whether retrieved cases are clinically similar, whether target-image answers are correct and complete, whether historical analogies are useful, and whether abstention is appropriate. Multiple reviewers would permit inter-rater agreement and adjudication.

The second priority is external replication. A new protocol could use a licensed subset of MIMIC-CXR, CheXpert Plus, PadChest, or another paired image-report source. The historical bank need not contain millions of images; a prespecified, sufficiently powered patient-disjoint subset could test whether the direction of the V9 effects transfers. Source-specific labels and patient identifiers would need a fresh audit.

Third, generation can be improved without changing the completed V9 result. Constrained JSON decoding, larger token budgets, multi-view image input, report-section-aware prompts, and alternative local multimodal generators should be compared on a new development and confirmation split. Retrieval and generation changes should remain factorial so that causal attribution is preserved.

Fourth, relevance can be strengthened with physician-labeled pairwise similarity or a task-specific utility label. Such judgments could train the reranker directly and reveal whether report-derived relevance correlates with actual clinical usefulness. Calibration and selective prediction should accompany any such study.

Fifth, the agent can be expanded cautiously. A future version could route by question type, retrieve at both case and evidence-span levels, compare independent visual and historical claims, and request human review under uncertainty. The loop should remain bounded and every tool action should be logged. LangChain or another orchestration library is unnecessary unless it removes concrete engineering complexity; the scientific value lies in the state and evidence contract, not the framework name.

## 5.7 Conclusion

This thesis developed a complete multimodal similar-case RAG workflow for new-patient chest-radiograph question answering. From 3,851 OpenI cases, it constructed a deterministic Train/Validation/Test study and a 2,608-case historical bank. Frozen BM25 and MedSigLIP channels were compared with fixed fusion and a trained 865-parameter reranker. The learned reranker achieved a statistically resolved nDCG@10 gain over image-only retrieval, and aligned images substantially exceeded 100 shuffled controls. Retrieved multimodal reports improved MedGemma answer-reference consistency over the same generator without retrieval and over text-only RAG. A bounded agent then suppressed unsupported historical-support claims through one retry or evidence abstention.

The contribution is deliberately scoped. The system does not prove diagnostic accuracy, clinical safety, or external validity. It demonstrates that correct image alignment matters, that learned fusion can outperform the strongest frozen retrieval component, that retrieval gains can transfer to downstream QA, and that agentic evidence control can be made bounded and auditable. Equally important, it records where the evidence remains weak: fixed fusion can degrade retrieval, learned retrieval need not produce confirmed learned-QA superiority, structured output can fail, automated verification is not clinical judgment, and human evaluation is still required.

The final value of the research is therefore not a claim that an agent can replace a radiologist. It is a reproducible account of how images, clinical text, historical reports, learned fusion, generation, and evidence control interact in a medically sensitive RAG pipeline, together with the controls needed to distinguish technical improvement from unsupported clinical claims.
"""


REFERENCES_AND_APPENDICES = r"""# References

Bae, S., Kyung, D., Ryu, J., et al. (2023). EHRXQA: A multi-modal question answering dataset for electronic health records with chest X-ray images. *Advances in Neural Information Processing Systems*.

Bannur, S., Hyland, S., Liu, Q., et al. (2023). Learning to exploit temporal structure for biomedical vision-language processing. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*.

Boecking, B., Usuyama, N., Bannur, S., et al. (2022). Making the most of text semantics to improve biomedical vision-language processing. *European Conference on Computer Vision*.

Demner-Fushman, D., Kohli, M. D., Rosenman, M. B., et al. (2016). Preparing a collection of radiology examinations for distribution and retrieval. *Journal of the American Medical Informatics Association, 23*(2), 304-310.

Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2024). RAGAS: Automated evaluation of retrieval augmented generation. *Proceedings of EACL System Demonstrations*.

Google Health AI Developer Foundations. (2025). MedGemma model card. https://developers.google.com/health-ai-developer-foundations/medgemma/model-card

Google Health AI Developer Foundations. (2025). MedSigLIP model card. https://developers.google.com/health-ai-developer-foundations/medsiglip/model-card

Jain, S., Agrawal, A., Saporta, A., et al. (2021). RadGraph: Extracting clinical entities and relations from radiology reports. *NeurIPS Datasets and Benchmarks*.

Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*.

Pal, A., Umapathi, L. K., and Sankarasubbu, M. (2023). Med-HALT: Medical domain hallucination test for large language models. *Proceedings of CoNLL*, 314-334.

Robertson, S., and Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval, 3*(4), 333-389.

Romanov, A., and Shivade, C. (2018). Lessons from natural language inference in the clinical domain. *Proceedings of EMNLP*, 1586-1596.

Singhal, K., Azizi, S., Tu, T., et al. (2023). Large language models encode clinical knowledge. *Nature, 620*, 172-180.

Soni, S., Gudala, M., Pajouhi, A., and Roberts, K. (2022). RadQA: A question answering dataset to improve comprehension of radiology reports. *Proceedings of LREC 2022*, 6250-6259.

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
- Final freeze: `docs/V9_TECHNICAL_FREEZE.md`

## Appendix B: Aggregate Result Artifacts

- Retrieval summary: `data/splits/v9/v9_retrieval_confirmation_summary.json`
- QA summary: `data/splits/v9/v9_qa_confirmation_summary.json`
- QA statistics: `data/splits/v9/v9_qa_statistical_analysis.json`
- Agent summary: `data/splits/v9/v9_agent_evaluation_summary.json`
- Split freeze: `data/splits/v9/v9_full_source_split_freeze.json`
- Public qualitative index: `data/splits/v9/v9_qualitative_case_index.csv`

Large source-derived reports, image pixels, vectors, checkpoints, prompts, and per-row generations remain local. Public hashes verify their frozen identity without redistributing source content.

## Appendix C: Reproduction Entry Points

**Repository:** https://github.com/yzy542968-jpg/wqf7023-medical-rag

**Branch:** `post-submission-improvements`

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -q
& ".\.venv\Scripts\python.exe" scripts\run_v9_retrieval_confirmation.py
& ".\.venv\Scripts\python.exe" scripts\analyze_v9_qa_statistics.py
& ".\.venv\Scripts\python.exe" scripts\evaluate_v9_qa_agent.py
streamlit run app.py --server.port 8504
```

Exact executable options, model revisions, hashes, failure rules, and local artifact paths are recorded in the frozen protocol and result documents. Confirmation should not be rerun under altered settings and presented as the same study.

## Appendix D: Qualitative and Human-Evaluation Boundary

The deterministic 24-case pack and assistant-proposed labels are complete. The student's final row-by-row review is pending. Until it is signed, all case-level interpretations remain provisional and no researcher-reviewed count is reported. No independent radiologist correctness, similarity, grounding, preference, harmfulness, or inter-rater agreement result is claimed.

## Appendix E: Dashboard Demonstration Boundary

The dashboard accepts a target chest radiograph, indication, and question; retrieves Top-3 similar reports from the frozen 2,608-case bank; optionally generates a MedGemma answer; and shows bounded-agent evidence actions. It describes the reports as historical analogies. It must not claim to locate the target patient's own report, diagnose the image, or replace radiologist review.

## Appendix F: Version Boundary

V5-V7 are preliminary controlled studies and remain frozen. V8 ended in a documented development no-go. V9 is the final primary study. No V9 outcome was used to rewrite a frozen model, prompt, threshold, metric, split, or confirmation case. Reporting edits do not alter technical artifacts.
"""


def between(text: str, start: str, end: str) -> str:
    start_idx = text.index(start)
    end_idx = text.index(end, start_idx)
    return text[start_idx:end_idx]


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8-sig")
    chapter_2 = between(source, "# Chapter 2:", "# Chapter 3:").rstrip()
    preliminary_methods_results = between(source, "# Chapter 3:", "# Chapter 5:").rstrip()

    preliminary_methods_results = preliminary_methods_results.replace(
        "# Chapter 3: Methodology",
        "# Chapter 3: Methodology\n\n"
        "The chapter reports two linked stages. Sections 3.1-3.13 retain the "
        "preliminary V5 controlled study because it established the alignment "
        "and grounding controls. Sections 3.14-3.22 define the final V9 "
        "new-patient similar-case study on which the primary thesis claims rest.",
        1,
    )
    preliminary_methods_results = preliminary_methods_results.replace(
        "# Chapter 4: Results and Analysis",
        "# Chapter 4: Results and Analysis\n\n"
        "Sections 4.1-4.7 summarize the frozen preliminary evidence. Sections "
        "4.8-4.13 report the final V9 held-out study and control the conclusions "
        "of this thesis.",
        1,
    )

    chapter_2 = chapter_2.replace(
        "# Chapter 2: Literature Review",
        "# Chapter 2: Literature Review\n\n"
        "Sections 2.1-2.11 preserve the literature synthesis that motivated the "
        "preliminary V5 controlled study. Task-specific references to V5 in those "
        "sections describe that preliminary phase. Section 2.12 updates the gap "
        "for the final V9 new-patient similar-case study.",
        1,
    )
    chapter_2 = chapter_2 + "\n\n" + CHAPTER_2_ADDENDUM.strip()
    preliminary_methods_results = preliminary_methods_results.replace(
        "V5 was the final technical experiment.",
        "Within the preliminary controlled phase, V5 was the final frozen experiment.",
    )
    preliminary_methods_results = preliminary_methods_results.replace(
        "the final V5 experiment",
        "the preliminary V5 confirmation experiment",
    )
    preliminary_methods_results = preliminary_methods_results.replace(
        "# Chapter 4: Results and Analysis",
        V9_METHODS.strip() + "\n\n# Chapter 4: Results and Analysis",
        1,
    )
    body = "\n\n".join(
        [
            "# Retrieval-Augmented Medical Question Answering over Paired Radiology Images and Reports",
            ABSTRACT.strip(),
            CHAPTER_1.strip(),
            chapter_2,
            preliminary_methods_results,
            V9_RESULTS.strip(),
            CHAPTER_5.strip(),
            REFERENCES_AND_APPENDICES.strip(),
        ]
    )
    OUTPUT.write_text(body.strip() + "\n", encoding="utf-8")
    words = len(body.split())
    if not 10000 < words < 30000:
        raise RuntimeError(f"Unexpected manuscript word count: {words}")
    print(f"Wrote {OUTPUT} ({words:,} whitespace-delimited words)")


if __name__ == "__main__":
    main()
