from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "P2_V9_FINAL_MANUSCRIPT.md"
OUTPUT = ROOT / "docs" / "P2_V10_V11_FINAL_MANUSCRIPT.md"


FRONT_MATTER = r"""# Retrieval-Augmented Medical Question Answering over Paired Radiology Images and Reports

## Abstract

This research investigates a bounded multimodal retrieval-augmented generation workflow for a new chest-radiograph case whose final report is not available at inference. The system receives one or two target chest images, an available clinical indication and a natural-language question. It retrieves other-case historical image-report pairs, ranks them with textual, visual and report-derived fact signals, selects question-relevant evidence within each retrieved case, generates a concise answer with MedGemma 1.5, and attaches case, section and fact provenance deterministically. Historical reports are treated as analogies rather than as facts about the target patient.

The primary V10 study used 3,851 OpenI/IU-Xray cases. Exact and near-duplicate report clustering produced 3,013 clusters before allocation to Train, Calibration, Validation and Test. The frozen split contained 2,510, 383, 384 and 574 cases respectively; 568 Test cases were technically eligible. Five retrieval systems were compared on the same Train-only historical bank. The fact-aware multiview R5 ensemble achieved nDCG@10 0.36007, compared with 0.34905 for the frozen R4 nine-feature reranker. The paired case-bootstrap difference was +0.01103, 95% CI [+0.00770, +0.01441]. Correctly aligned images also exceeded all 100 deterministic fixed-point-free shuffled-image assignments: aligned nDCG@10 was 0.36007 versus shuffled mean 0.24963, with plus-one Monte Carlo p=0.00990. These results support an alignment-specific retrieval contribution within the same source dataset.

Downstream question answering compared target-image generation without history, R4 whole-report RAG, R5 RAG, and a calibrated selective-history condition. R5 RAG improved Token-F1 over no-history generation by +0.05978, 95% CI [+0.05114, +0.06860], but its incremental Token-F1 advantage over R4 whole-report RAG was only +0.00167, CI [-0.00347, +0.00683]. Better retrieval therefore transferred to automated report-reference consistency relative to no history, but did not confirm generator-level superiority over the previous multimodal retriever.

V11 was retained as a development-only mechanism extension. A full-bank relevance audit showed that candidate generation remained a bottleneck. Reciprocal-rank-fusion candidate generation improved nDCG@10 and relevant-case presence, particularly at K=200, but remained exploratory. A clean 48-case, 432-generation MedGemma experiment compared whole-report, sentence-only and case-to-fact evidence. Case-to-fact reduced mean evidence characters by 63.4% and mean input tokens by 32.4%, while maintaining 100% answer-contract and provenance validity. Its Token-F1 advantage over whole-report evidence was +0.02195, 95% CI [-0.00026, +0.04302], and complete F1RadGraph differed by +0.01395, CI [-0.00691, +0.03442]. The intervals crossed zero, so the result supports efficiency and auditability rather than confirmed answer-quality improvement. A second frozen 96-item planner wording set achieved 0.9167 accuracy, 0.9196 macro-F1 and 1.0000 indication invariance without post-evaluation rule changes.

The study concludes that correctly paired images can improve report-derived similar-case retrieval, fact-aware ranking provides a small but confirmed retrieval gain, and historical context can improve automated answer-reference consistency. It also shows that stronger retrieval does not guarantee stronger grounding or final QA, fact selection cannot repair a missing or wrong case, and proxy confidence is not clinical calibration. No physician-adjudicated correctness, clinical safety or external patient-level generalization is claimed.

**Keywords:** multimodal retrieval-augmented generation; chest radiography; similar-case retrieval; medical question answering; MedSigLIP; MedGemma; RadGraph; provenance; shuffled-image control; evidence selection

## Declaration of evidence boundary

All reported primary and development metrics are automated and retrospective. The processed OpenI source supports case-ID and duplicate-cluster separation but does not expose a reliable patient identifier for independent patient-level verification. Independent radiologist review and external validation remain Future Work. No reviewer ratings or external results are fabricated.
"""


RESEARCH_QUESTIONS = r"""## 1.5 Research Questions

**RQ1.** Does the correctly aligned target chest image improve report-derived similar historical-case retrieval over text-only and other frozen retrieval baselines?

**RQ2.** Does the aligned-image system exceed deterministic shuffled-image controls, indicating that the gain depends on the correct image-case pairing rather than a generic visual prior?

**RQ3.** Does fact-aware multiview retrieval improve report-derived graded relevance over the frozen nine-feature multimodal reranker?

**RQ4.** Does retrieval improvement transfer to downstream MedGemma answer-reference consistency, and can within-case fact selection reduce context and truncation without sacrificing provenance?
"""


CONTRIBUTIONS = r"""## 1.6 Research Contributions

The first contribution is a clinically interpretable new-case task contract. The target report is unavailable at inference and cannot leak into retrieval, prompting or answer generation. The system retrieves reports belonging to other cases and labels them as historical analogies. This distinction prevents a paired dataset from collapsing into target-report lookup and makes case ownership part of the evidence model.

The second contribution is a duplicate-cluster-disjoint multimodal retrieval study with an alignment-specific negative control. Exact and near-duplicate report clustering occurs before allocation. BM25, image-image, image-report, nine-feature and fact-aware multiview systems rank a common historical bank. One hundred deterministic fixed-point-free shuffled-image assignments recompute the visual state and test whether the improvement depends on the correctly paired image.

The third contribution is a fact-aware, provenance-preserving evidence workflow. R5 incorporates question-conditioned RadGraph and multiview signals. V11 then separates case retrieval from within-case sentence or fact selection, retaining case ID, report section, unit type and source hash. The design reduces irrelevant context without combining anonymous facts across patients.

The fourth contribution is an evaluation and reproducibility framework that preserves mixed results. Retrieval, generation, automated semantic overlap, structured-output validity, provenance, latency, GPU memory and uncertainty intervals are reported separately. V10 frozen hashes remain unchanged; V11 is explicitly development-only. Human review and external validation are not replaced with proxy metrics.
"""


SCOPE_FRAMEWORK = r"""## 1.7 Scope and Boundaries

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
"""


FINAL_RETRIEVAL_REVIEW = r"""## 2.3 Sparse, Dense, and Multimodal Retrieval

BM25 remains a strong transparent sparse-retrieval baseline based on probabilistic term matching (Robertson and Zaragoza, 2009). It is effective when a query shares terminology with a report, but it is sensitive to lexical overlap and may exploit indication shortcuts. The contrast between BM25 and image-assisted systems helps show whether the target image contributes information beyond the available clinical text.

Dense retrieval encodes queries and documents in a shared vector space. MedCPT uses large-scale PubMed search logs for biomedical retrieval (Jin et al., 2023), while CLIP established general contrastive image-text alignment (Radford et al., 2021). Domain-specific medical encoders can reduce vocabulary and representation mismatch, but semantic similarity does not guarantee evidence ownership or clinical usefulness.

Multimodal retrieval adds image-image and image-report relations. These channels have different score distributions and failure modes, so a convincing evaluation should report each component rather than compare only a fused model with BM25. V10 therefore evaluates BM25, MedSigLIP image-image, MedSigLIP image-report, a nine-feature multimodal reranker and a fact-aware multiview extension on the same Train-only bank.

The final runtime scores the complete technically eligible historical bank rather than relying on a BM25-only shortlist. Compact learned rerankers are fitted on Train roles and frozen before Test. Ranking ties are deterministic, all Test questions share the same candidate bank, and a target report is never inserted into that bank. V11 separately audits whether BM25, MedCPT and MedSigLIP reciprocal-rank fusion can improve first-stage candidate recall at bounded K.
"""


FINAL_VISION_LANGUAGE_REVIEW = r"""## 2.5 Biomedical Vision-Language Representation

BioViL introduced radiology-specific image-text representation learning with localized and global alignment between chest X-rays and reports (Boecking et al., 2022). BioViL-T extended this line by exploiting temporal and multi-image structure (Bannur et al., 2023). MedSigLIP provides a newer medical image-text embedding model intended for semantic image retrieval and related representation tasks. These encoders supply domain priors but do not by themselves establish diagnostic correctness.

The final V10 system uses the pinned MedSigLIP-448 revision for image-image and image-report features. The foundation encoder remains frozen; only compact retrieval components are trained. This separates representation reuse from task-specific ranking and keeps the computation feasible on the available GPU. Earlier BioViL-T studies are retained as formative history, not described as the final encoder.

Multi-view examinations create an aggregation problem because frontal and lateral views may contain complementary information. V10 compares mean-view and learned-attention representations in a frozen-checkpoint mechanism audit. The final R5 ensemble uses the prespecified multiview component, but the attention-only contrast is interpreted cautiously because its interval crosses zero.

An embedding is not an explanation or a calibrated clinical probability. A high image-report score does not identify the responsible anatomy, polarity or uncertainty. R5 therefore combines image features with question-conditioned report facts and preserves the retrieved case and fact provenance. The representation is evaluated as a ranking signal within a controlled RAG workflow, not as an autonomous image diagnosis model.
"""


FINAL_QA_REVIEW = r"""## 2.6 Medical Visual and Report Question Answering

VQA-RAD contains clinically generated questions and answers about radiology images (Lau et al., 2018), while EHRXQA combines electronic health records and chest X-rays for multimodal QA (Bae et al., 2023). RadQA contains physician-authored report questions, answer spans and naturally unanswerable cases (Soni et al., 2022). These resources show the value of clinician phrasing and explicit answerability, but they differ from the same-source new-case retrieval task used here.

The completed benchmark uses three fixed question roles for retrieval and two report-derived roles for V10 generation. This controlled design supports paired system comparisons but is linguistically narrow. The V11 reserved wording set tests planner robustness only; its labels are author-defined and cannot substitute for clinician-authored natural questions.

The final workflow also differs from pure report QA. MedGemma receives the target chest image and may receive explicitly labelled other-case historical reports. The hidden target report is used only as an automated evaluation reference. Retrieved reports are analogies, not answer spans about the target patient. This separation makes it possible to evaluate visual alignment, historical evidence ownership and answer-reference consistency independently.

Natural unanswerability remains incompletely tested. The system can withhold unreliable historical support, but the fixed questions are derived from available report roles. RadQA or a new clinician-authored set would provide a stronger answerability evaluation once authorized access and an external protocol are available.
"""


FINAL_ALIGNMENT_REVIEW = r"""## 2.8 Alignment Controls and Benchmark Validity

Multimodal improvement can be misattributed when clinical text already identifies the answer or when any image changes score distributions. Image ablation asks whether visual information adds value; wrong-image controls ask whether the value depends on the correct image-case pairing. Both are stronger when systems share the same cases, candidate bank, model state and metrics.

V10 uses 100 deterministic unique fixed-point-free image assignments. Each Test case receives the complete view set of another Test case while its indication, question and evaluation reference remain unchanged. Visual similarities, normalized features, multiview state and R5 scores are recomputed. The plus-one Monte Carlo p-value avoids reporting zero from a finite control set.

Benchmark construction can introduce additional shortcuts. Generic repeated questions make text retrieval weakly identified, indications can encode disease hints, and near-duplicate reports can leak across partitions. V10 clusters exact and near-duplicate reports before allocation, uses a common Train-only bank and reports image-only, image-report and BM25 components separately. Patient-level separation remains unverifiable because reliable subject identifiers are absent from the processed source.

The relevance construct is another validity boundary. Report labels and RadGraph facts enable deterministic graded qrels, but they are not physician judgments. Post-hoc sensitivity across combined, label-only and fact-only definitions is therefore reported as a construct audit rather than a replacement endpoint.
"""


FINAL_AGENT_REVIEW = r"""## 2.9 Agentic and Auditable RAG Workflows

Agentic RAG can plan, retrieve, rerank, generate, verify or abstain. The term should be used carefully: a deterministic workflow is not learned clinical reasoning, and an automated checker is not an independent physician. Research value depends on bounded actions, inspectable state and honest failure handling rather than on the number of named agents.

The final system is agent-like only in this bounded engineering sense. A deterministic planner identifies question intent, the retriever ranks historical cases, the evidence selector preserves case ownership, MedGemma generates a concise target-image answer, deterministic code attaches provenance, and a calibrated gate can withhold historical support. Each stage has an explicit contract and can be evaluated separately.

Direct Python modules are used instead of LangChain because the experiment requires stable prompts, frozen transitions and artifact-level reproducibility. This choice does not imply that orchestration libraries are inferior; it avoids introducing hidden retries, memory or tool-selection behavior into a controlled study. The Dashboard exposes the retrieved case IDs, evidence units, confidence boundary and provenance without claiming autonomous clinical agency.

Open-ended agents may be useful in future interactive systems, but they require separate evaluation of tool choice, prompt reformulation, retry policy and safety. The current contribution is an auditable multimodal RAG workflow, not a general autonomous medical assistant.
"""


FINAL_COMPARATIVE_SYNTHESIS = r"""## 2.10 Comparative Synthesis of Design Alternatives

Several technically plausible architectures answer different research questions. An LLM-only system tests parametric medical knowledge but cannot expose a case-specific evidence path. A direct vision-language model can answer from the target image, but it does not test historical-case retrieval and can confound visual recognition, medical reasoning and language generation. Both remain useful generation baselines, but neither isolates the retrieval contribution studied here.

Text-only RAG preserves an auditable document path and provides a necessary baseline. BM25 is transparent and fast, while modern dense text retrieval can reduce vocabulary mismatch. However, the indication and question may be generic or incomplete before the report exists. A text retriever can therefore overuse indication shortcuts or return lexically similar reports whose images differ from the target case. V10 retains BM25 as R0 rather than treating it as an intentionally weak comparator.

Chunk-level retrieval offers fine-grained matching and short prompts, but radiology chunks can lose case ownership, split negation or detach findings from impressions. Whole-report retrieval preserves a coherent historical unit and was the development-selected V10 QA policy. V11 adds sentence and fact selection only after case retrieval, so every unit retains its owning `case_id`, section, unit type and source hash. This ordering avoids anonymous cross-patient fact assembly.

Image-only and image-report retrieval test complementary visual relations. Image-image similarity asks whether the target radiograph resembles a historical radiograph. Image-report compatibility asks whether the target image aligns with the language of a historical report. Neither relation alone guarantees clinically useful evidence. The R4 comparator therefore combines normalized text, image-image, image-report and rank features, while R5 adds question-conditioned fact signals and multiview representation. All systems rank the same Train-only historical bank.

A fully fine-tuned multimodal foundation model might achieve stronger benchmark performance, but it would introduce additional choices about negative sampling, optimization, checkpoints and model adaptation. The final study instead freezes the MedSigLIP and MedGemma foundation models and trains only compact retrieval components. This makes the incremental R5-minus-R4 comparison more attributable and feasible on the available 8 GB GPU.

An agent framework or LangChain could orchestrate retrieval, planning, generation and checking. Neither is required for scientific validity. The implemented planner, evidence selector, generator contract and confidence gate are direct modules with explicit inputs and outputs. This keeps retries, abstention, provenance and failure states inspectable without implying autonomous clinical agency.

The selected design follows four principles. First, retrieval units preserve evidence ownership. Second, multimodal gain is compared with strong individual components and complete wrong-image controls. Third, target-image answering and historical provenance are separated. Fourth, retrieval, generation, structure, confidence and clinical validity are evaluated as different layers. These principles motivate the final research gap.
"""


FINAL_LITERATURE_GAP = r"""## 2.11 Similar-Case Multimodal RAG and Final Research Gap

The closest line of work combines chest-image retrieval with report generation. CXR-RePaiR uses a contrastive image-to-report retriever to construct reports from retrieved exemplars. X-REM adds coarse retrieval, learned image-text matching and an NLI filter. FactMM-RAG mines factual report pairs with CheXbert and RadGraph to train a fact-aware multimodal retriever. RA-RRG retrieves clinically important key phrases to condition report generation, while MedProbCLIP introduces probabilistic image-report embeddings, calibration and risk-coverage evaluation. These systems establish that historical image-report pairs can support image-conditioned language generation and that retrieval reliability deserves explicit measurement.

They do not directly resolve the question-conditioned new-case setting examined here. At inference, the target report is hidden; the target image, available indication and medical question must retrieve other-case historical evidence. The output must distinguish observations about the target image from analogies drawn from retrieved reports. This task requires both clinically useful ranking and explicit evidence ownership.

Five connected gaps remain. First, many radiology retrieval studies focus on report generation rather than a user's question. Second, multimodal improvements are not always compared with strong individual visual and text components. Third, aligned-image gains are rarely challenged by complete wrong-image recomputation. Fourth, local support for a retrieved report is often conflated with correctness for the target case. Fifth, duplicate leakage, relevance sensitivity, candidate recall, wording robustness and selective reliability can materially change conclusions but are often treated as implementation details.

The final research gap is therefore not whether RAG can be used in radiology or whether a newer model can replace an older one. It is whether a correctly paired target image improves retrieval of clinically related other-case reports under duplicate-aware splitting and shuffled-image controls; whether fact-aware reranking adds value over a strong frozen multimodal comparator; whether retrieval gains transfer to automated answer-reference consistency; and whether the complete workflow preserves case, section and fact provenance while retaining negative and mixed results.

V10 addresses that gap through a cluster-disjoint same-source confirmation, a common Train-only historical bank, report-derived graded relevance, deterministic fixed-point-free image shuffling, bounded local generation and explicit claim boundaries. V11 investigates residual mechanisms - candidate recall, within-case evidence compression, planner wording and retrieval confidence - on development data only. Neither automated relevance nor report-reference overlap is presented as physician-adjudicated clinical correctness.
"""


CHAPTER_3 = r"""# Chapter 3: Methodology

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
"""


CHAPTER_4 = r"""# Chapter 4: Results

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
"""


CHAPTER_5 = r"""# Chapter 5: Discussion

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
"""


CHAPTER_6 = r"""# Chapter 6: Conclusion

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
"""


FINAL_APPENDICES = r"""## Appendix I: Final V10/V11 Artifact Index

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
"""


def top_section(text: str, start: str, end: str | None) -> str:
    begin = text.index(start)
    finish = text.index(end, begin) if end else len(text)
    return text[begin:finish].strip()


def replace_section(text: str, start: str, end: str, replacement: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[:begin] + replacement.strip() + "\n\n" + text[finish:]


def build() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    chapter_1 = top_section(source, "# Chapter 1:", "# Chapter 2:")
    chapter_2 = top_section(source, "# Chapter 2:", "# Chapter 3:")
    references = top_section(source, "# References", "# Appendices")
    appendices = top_section(source, "# Appendices", None)

    chapter_1 = chapter_1.replace("final V9", "V10 primary").replace("Final V9", "V10 primary")
    chapter_1 = chapter_1.replace(
        "V9 changes the construct: the target report is removed from the candidate bank and hidden from inference. A fixed Train-only bank supplies other-case evidence to Validation and Test queries. The final claims are based on the V9 held-out study rather than on the earlier development versions.",
        "V9 changed the construct by removing the target report from the candidate bank and hiding it at inference. V10 then added duplicate-cluster-disjoint confirmation over a fixed Train-only historical bank. The final primary claims come from V10 Test; V11 contributes development-only mechanism evidence.",
    )
    chapter_1 = replace_section(chapter_1, "## 1.5 Research Questions", "## 1.6 Research Contributions", RESEARCH_QUESTIONS)
    chapter_1 = replace_section(chapter_1, "## 1.6 Research Contributions", "## 1.7 Scope and Boundaries", CONTRIBUTIONS)
    chapter_1 = chapter_1[: chapter_1.index("## 1.7 Scope and Boundaries")] + SCOPE_FRAMEWORK.strip()

    chapter_2 = chapter_2.replace(
        "Sections 2.1-2.11 preserve the literature synthesis that motivated the preliminary V5 controlled study. Task-specific references to V5 in those sections describe that preliminary phase. Section 2.12 updates the gap for the final V9 new-patient similar-case study.\n\n",
        "This chapter synthesizes the literature by research theme rather than as a paper-by-paper catalogue. Earlier controlled-study references are retained where they motivate the final V10/V11 design.\n\n",
    )
    chapter_2 = chapter_2.replace("final V9", "V10 primary").replace("Final V9", "V10 primary")
    chapter_2 = chapter_2.replace(
        "Practical evaluations also emphasize noisy, misleading, or insufficient evidence rather than assuming ideal retrieval (Ngo et al., 2024).",
        "RAGAS likewise separates retrieval context from answer faithfulness instead of assuming ideal evidence (Es et al., 2024).",
    )
    chapter_2 = chapter_2.replace("the frozen V5 experiment", "the completed study")
    chapter_2 = chapter_2.replace(
        "the frozen V5 references are derived from available report sections",
        "the template-derived references in the present benchmark come from available report sections",
    )
    chapter_2 = chapter_2.replace("The V5 shuffled-image condition", "The V10 shuffled-image condition")
    chapter_2 = chapter_2.replace("The V5 conditions", "The V10 conditions")
    chapter_2 = replace_section(chapter_2, "## 2.3 Sparse, Dense, and Multimodal Retrieval", "## 2.4 Paired Radiology Images and Reports", FINAL_RETRIEVAL_REVIEW)
    chapter_2 = replace_section(chapter_2, "## 2.5 Biomedical Vision-Language Representation", "## 2.6 Medical Visual and Report Question Answering", FINAL_VISION_LANGUAGE_REVIEW)
    chapter_2 = replace_section(chapter_2, "## 2.6 Medical Visual and Report Question Answering", "## 2.7 Evidence Grounding and Medical Hallucination", FINAL_QA_REVIEW)
    chapter_2 = replace_section(chapter_2, "## 2.8 Alignment Controls and Benchmark Validity", "## 2.9 Agentic and Auditable RAG Workflows", FINAL_ALIGNMENT_REVIEW)
    chapter_2 = replace_section(chapter_2, "## 2.9 Agentic and Auditable RAG Workflows", "## 2.10 Comparative Synthesis of Design Alternatives", FINAL_AGENT_REVIEW)
    chapter_2 = (
        chapter_2[: chapter_2.index("## 2.10 Comparative Synthesis of Design Alternatives")]
        + FINAL_COMPARATIVE_SYNTHESIS.strip()
        + "\n\n"
        + FINAL_LITERATURE_GAP.strip()
    )

    references = references.replace(
        "Jain, S., Agrawal, A., Saporta, A., et al. (2021). RadGraph: Extracting clinical entities and relations from radiology reports. *NeurIPS Datasets and Benchmarks*.",
        "Jain, S., Agrawal, A., Saporta, A., et al. (2021). RadGraph: Extracting clinical entities and relations from radiology reports. *NeurIPS Datasets and Benchmarks*.\n\nJin, Q., Kim, W., Chen, Q., et al. (2023). MedCPT: Contrastive pre-trained transformers with large-scale PubMed search logs for zero-shot biomedical information retrieval. *Bioinformatics, 39*(11), btad651. https://doi.org/10.1093/bioinformatics/btad651\n\nLau, J. J., Gayen, S., Ben Abacha, A., and Demner-Fushman, D. (2018). A dataset of clinically generated visual questions and answers about radiology images. *Scientific Data, 5*, 180251. https://doi.org/10.1038/sdata.2018.251",
    )
    references = references.replace(
        "Qwen Team. (2025). Qwen3 Embedding: Advancing text embedding and reranking through foundation models. https://qwenlm.github.io/blog/qwen3-embedding/",
        "Qwen Team. (2025). Qwen3 Embedding: Advancing text embedding and reranking through foundation models. https://qwenlm.github.io/blog/qwen3-embedding/\n\nRadford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning transferable visual models from natural language supervision. *Proceedings of the 38th International Conference on Machine Learning*, 8748-8763.",
    )

    appendices = appendices.replace(
        "# Appendices",
        "# Appendices\n\nAppendices A-H preserve historical V9 and preliminary controlled-study artifacts for traceability. They are not the primary V10 result. Appendices I-L register the final V10/V11 evidence and release boundary.",
        1,
    )
    appendices = appendices.replace(
        "Top-3 similar reports from the frozen 2,608-case bank",
        "Top-3 similar historical reports from the frozen 2,506-case V10 Train bank",
    )
    appendices = appendices.replace("## Appendix A: Final V9 Public Evidence", "## Appendix A: Historical V9 Public Evidence")
    appendices = appendices.replace(
        "V5-V7 are preliminary controlled studies and remain frozen. V8 ended in a documented development no-go. V9 is the final primary study. Supplemental V9 audits were committed after the technical freeze and before their own outcomes; they did not change a frozen model, prompt, threshold, metric, split, case, or primary conclusion. Reporting edits do not alter technical artifacts.",
        "V5-V9 are frozen formative and historical studies. V10 is the final automated primary extension. V11 is development-only and does not instantiate a new confirmation cohort. Supplemental audits do not change frozen models, prompts, thresholds, metrics, splits, cases or primary results.",
    )
    appendices = appendices.replace(
        "They are formative evidence and do not replace the V9 primary study.",
        "They are formative evidence and do not replace the V10 primary study.",
    )
    for old, new in (("Table 4.1", "Table H.1"), ("Table 4.2", "Table H.2"), ("Table 4.3", "Table H.3")):
        appendices = appendices.replace(old, new)
    appendices += "\n\n" + FINAL_APPENDICES.strip()

    body = "\n\n".join(
        [
            FRONT_MATTER.strip(),
            chapter_1.strip(),
            chapter_2.strip(),
            CHAPTER_3.strip(),
            CHAPTER_4.strip(),
            CHAPTER_5.strip(),
            CHAPTER_6.strip(),
            references.strip(),
            appendices.strip(),
        ]
    )
    body = re.sub(r"\n{3,}", "\n\n", body).strip() + "\n"
    words = len(body.split())
    if not 10_000 < words < 30_000:
        raise RuntimeError(f"Unexpected manuscript word count: {words}")
    OUTPUT.write_text(body, encoding="utf-8", newline="\n")
    print(f"Wrote {OUTPUT} ({words:,} whitespace-delimited words)")


if __name__ == "__main__":
    build()
