from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "P2_V10_V11_FINAL_MANUSCRIPT.md"
OUTPUT = ROOT / "docs" / "P2_FINAL_MANUSCRIPT.md"


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    left = text.index(start)
    right = text.index(end, left)
    return text[:left] + replacement.rstrip() + "\n\n" + text[right:]


FRONT = """# Retrieval-Augmented Medical Question Answering over Paired Radiology Images and Reports

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
"""


CHAPTER1_TAIL = """## 1.3 Research Aim

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
"""


LITERATURE_TAIL = """## 2.10 Closest Retrieval-Based Radiology Systems

CXR-RePaiR used CLIP-style image-to-report retrieval to generate chest X-ray reports from historical text (Endo et al., 2021). It establishes retrieval itself as a viable alternative to unconstrained generation. X-REM added multimodal image-text matching and expert error assessment, showing that coarse cosine similarity can miss fine-grained compatibility (Jeong et al., 2023). Both systems motivate retrieval-based reuse, but their primary output is a report rather than a question-conditioned answer with explicit other-case provenance.

FactMM-RAG used RadGraph-derived factual report pairs to train a multimodal retriever and augment radiology report generation (Sun et al., 2025). It is the closest precedent for factual retrieval supervision. The present study reuses the principle of fact-aware ranking but preserves whole case ownership before fact selection and evaluates a target-image, indication and question contract rather than copying restricted data or reproducing a full-report LLaVA system.

MedProbCLIP emphasizes probabilistic radiograph-report embeddings, calibration, risk-coverage behavior, multiview representation and selective retrieval (Elallaf et al., 2026). It motivates confidence and abstention analysis, but the present system does not reproduce its probabilistic training objective. Its confidence values remain report-derived technical signals.

## 2.11 Final Research Gap

Prior work establishes medical RAG, chest image-report retrieval, factual retrieval supervision and report generation, but several elements are rarely evaluated together. A paired dataset may allow accidental target-report lookup. Near-duplicate reports may cross splits. A nominally multimodal model may rely on text shortcuts. Retrieved reports may be copied as if they describe the current patient. Better retrieval may fail to improve final answers. Generator adaptation may help one report section while damaging another. Automated metrics may be overstated as clinical accuracy.

The final gap is therefore not simply a newer model. It is an integrated evidence chain for target-report-hidden, question-conditioned historical-case RAG: duplicate-aware allocation, Train-only evidence, aligned-versus-shuffled image control, multi-source candidate generation, learned reranking, retrieved-versus-random history control, case-preserving fact provenance, section-aware parameter-efficient adaptation, and transparent reporting of mixed metric behavior.
"""


CHAPTER3 = """# Chapter 3: Methodology

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

Technical interruptions could be resumed under unchanged frozen settings. Outcome-driven reruns, case replacement, Test-driven model selection, prompt revision, qrel substitution and selective result deletion were prohibited. The final repository test suite contains 315 passing tests after the completeness audit.
"""


CHAPTER4 = """# Chapter 4: Results

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
"""


CHAPTER5 = """# Chapter 5: Discussion

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
"""


CHAPTER6 = """# Chapter 6: Conclusion

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
"""


FINAL_APPENDICES = r"""## Appendix I: Final V10/V12/V16 Artifact Index

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

**Final development branch at manuscript build:** `v12-optimization-pilot`

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
"""


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    text = replace_between(text, "# Retrieval-Augmented", "# Chapter 1: Introduction", FRONT)
    text = replace_between(text, "## 1.3 Research Aim", "# Chapter 2: Literature Review", CHAPTER1_TAIL)
    text = replace_between(text, "## 2.10 Comparative Synthesis", "# Chapter 3: Methodology", LITERATURE_TAIL)
    text = replace_between(text, "# Chapter 3: Methodology", "# Chapter 4: Results", CHAPTER3)
    text = replace_between(text, "# Chapter 4: Results", "# Chapter 5: Discussion", CHAPTER4)
    text = replace_between(text, "# Chapter 5: Discussion", "# Chapter 6: Conclusion", CHAPTER5)
    text = replace_between(text, "# Chapter 6: Conclusion", "# References", CHAPTER6)
    text = text.replace(
        "V10 then added duplicate-cluster-disjoint confirmation over a fixed Train-only historical bank. The final primary claims come from V10 Test; V11 contributes development-only mechanism evidence.",
        "V10 then added duplicate-cluster-disjoint confirmation over a fixed Train-only historical bank and established the alignment-controlled methodological foundation. V11 and V13-V15 provide development or mechanism evidence. The final integrated primary claims combine V12 learned retrieval with V16 section-aware generation under the frozen V16 confirmation protocol.",
    )
    text = text.replace(
        "Earlier controlled-study references are retained where they motivate the final V10/V11 design.",
        "Earlier controlled-study references are retained where they motivate the final V10/V12/V16 design.",
    )
    text = text.replace(
        "The final V10 system uses the pinned MedSigLIP-448 revision for image-image and image-report features.",
        "The V10 foundation and final V12 retriever use the pinned MedSigLIP-448 revision for image-image and image-report features.",
    )
    text = replace_between(
        text,
        "## Appendix F: Version Boundary",
        "## Appendix G: Frozen Preliminary Controlled-Study Methods",
        """## Appendix F: Version Boundary

V5-V9 are frozen formative and historical studies. V10 is the frozen methodological foundation and alignment study. V11 and V13-V15 are development or mechanism evidence. V12 is the final learned retrieval method, and V16 is the final integrated held-out method confirmation. Supplemental audits do not change frozen models, prompts, qrels, cases or primary results.

The following appendices preserve the detailed V5 controlled study for traceability. They are formative evidence and do not replace the final V16 study.""",
    )
    text = replace_between(text, "## Appendix I: Final V10/V11 Artifact Index", "", FINAL_APPENDICES) if False else text
    appendix_start = text.index("## Appendix I: Final V10/V11 Artifact Index")
    text = text[:appendix_start] + FINAL_APPENDICES.rstrip() + "\n"
    text = text.replace("**Branch:** `post-submission-improvements`", "**Branch:** `v12-optimization-pilot`")
    text = text.replace(
        "Appendices A-H preserve historical V9 and preliminary controlled-study artifacts for traceability. They are not the primary V10 result. Appendices I-L register the final V10/V11 evidence and release boundary.",
        "Appendices A-H preserve historical V9 and preliminary controlled-study artifacts for traceability. They are not the final V16 result. Appendices I-N register the final V10/V12/V16 evidence, reproducibility and release boundary.",
    )
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Words: {len(text.split())}")


if __name__ == "__main__":
    main()
