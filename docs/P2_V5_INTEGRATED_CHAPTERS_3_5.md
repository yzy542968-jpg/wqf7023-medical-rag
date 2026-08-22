# V5-Integrated Chapters 3-5 Draft

## Draft status

This document is the first V5-integrated replacement for Chapters 3-5. It does not overwrite the submitted P2 manuscript. All quantitative values are taken from the frozen V5 artifacts at commit `10f57ba`; the qualitative analysis is taken from tag `v5-qualitative-freeze`.

# Chapter 3: Methodology

## 3.1 Research Design

This study used a staged empirical system-comparison design to investigate retrieval-augmented medical question answering over paired radiology images and reports. Earlier text-only experiments identified two structural risks: open-corpus retrieval could select evidence from the wrong case, while a sentence-level verifier could still rate an answer as supported by that wrongly selected report. These findings motivated the final V5 experiment, which tested whether correctly paired chest X-ray information could improve target-report retrieval and whether any retrieval gain transferred to downstream report-grounded question answering.

V5 was the final technical experiment. Its configuration was specified and frozen locally before execution, but it was not formally preregistered or externally timestamped before outcomes were observed. The confirmation cohort was disjoint from all previous project cohorts, although it remained drawn from the same OpenI/IU-Xray source. V5 therefore provides fresh within-source confirmation rather than external validation.

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

## 3.2 Data Source and Cohort Construction

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

## 3.3 Question and Input Conditions

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

## 3.4 Case-Aware Evidence Representation and Downstream QA Workflow

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

## 3.5 Text Retrieval

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

## 3.6 Image Encoding and Multimodal Reranking

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

## 3.7 Shuffled-Image Control

The alignment control used 100 deterministic fixed-point-free permutations with seed 7023. In each permutation, every source case received another case's image embedding and no case retained its own image. Text queries, candidate reports, shortlist size, fusion weights, and evaluation procedure remained unchanged.

The control tested whether the correctly aligned image outperformed image-conditioned reranking with incorrect case alignment. It did not prove causal clinical image understanding. A plus-one Monte Carlo value was calculated as `(b+1)/(m+1)`, where `b` was the number of shuffled runs meeting or exceeding the correctly aligned result and `m=100` was the number of permutations.

A fixed-point-free permutation is stronger than an unconstrained shuffle because it guarantees that every target receives an incorrect image. In an ordinary random shuffle, some cases could retain their own image, weakening the negative control. Each of the 100 mappings was stored or reproducibly generated from the frozen seed so the exact null distribution could be reconstructed.

The same incorrect mapping was applied consistently to all three questions from a case within a run. This preserved the case as the unit of dependence and prevented findings, impression, and summary questions from receiving different pseudo-patients. Across runs, text input and candidate reports remained fixed, so variation arose from image assignment.

The shuffled distribution is an empirical alignment control rather than a universal null model. Incorrect images may sometimes be clinically similar to the target and can still help or harm ranking. The relevant observation is whether correct alignment lies beyond the outcomes produced by many controlled misalignments under the same fusion mechanism.

## 3.8 Answer Generation and Semantic Verification

Both report-only and multimodal retrieval conditions used the same local `Qwen/Qwen2.5-1.5B-Instruct` generator. Generation used CUDA, float16, batch size 16, maximum 256 new tokens, temperature 0, and a direct non-oracle prompt. The generator did not receive the frozen target identifier or reference answer.

The semantic checker used `pritamdeka/PubMedBERT-MNLI-MedNLI`. It combined lexical evidence matching, entailment and contradiction probabilities, and polarity consistency. Its locked configuration used lexical weight 0.2, support threshold 0.6, entailment threshold 0.75, and contradiction threshold 0.5. Evidence scope was restricted to the top-ranked selected report. The action path could retain supported sentences, filter flagged sentences, or abstain if no usable answer remained.

The checker was an automated evidence signal rather than a clinical gold standard. Its support rate measured agreement with selected-report evidence, not target-patient correctness or clinical safety.

Prompt construction used the same instruction structure for both retrieval conditions. The prompt identified the selected report as the only evidence source, supplied the question, requested a concise answer, and allowed abstention when the report did not support an answer. It did not reveal whether retrieval was correct and did not include the frozen reference. Temperature zero reduced sampling variation, while the 256-token limit was sufficient for the short report-grounded task.

Qwen2.5-1.5B-Instruct was selected because it could run locally within the available laptop GPU memory and support batched deterministic generation. The study does not claim that this model is optimal for radiology. Holding it fixed was more important for causal comparison than maximizing absolute generation quality with multiple generators.

The verifier segmented each draft into sentences and evaluated each sentence against selected-report evidence. Lexical matching rewarded direct support, the NLI model estimated entailment and contradiction, and polarity checks guarded against simple negation reversals. The frozen action policy converted these signals into retained, filtered, or abstaining output. Sentence-level records were preserved so that a final revision rate could be decomposed during qualitative review.

The checker could not correct retrieval identity. Its premise was always the selected report. If that report belonged to the wrong case, a faithful draft could pass. Conversely, a correct-report paraphrase could be rejected by the checker. For this reason, automated support was reported as one pipeline measure and was not used to redefine retrieval qrels or reference answers.

## 3.9 Evaluation Metrics and Statistical Analysis

Retrieval metrics were Hit@1, Hit@5, Hit@10, MRR, and an extractive proxy Token-F1 calculated from the selected report evidence. Hit@1 measured Top-1 target-case alignment; MRR retained information about target-rank movement even when the target did not reach first place.

QA metrics were draft Token-F1, final Token-F1 after semantic checking, automated evidence-support rate, revision rate, and abstention rate. Token-F1 measured reference overlap and was not interpreted as clinical correctness.

V5 used 5,000 grouped bootstrap resamples at case level and paired randomization tests with seed 7023. The primary retrieval comparison was indication-plus-question with correctly aligned image minus indication-plus-question BM25. The primary QA comparison was multimodal final Token-F1 minus report-only final Token-F1. Confidence intervals and p-values therefore preserved the dependence among questions from the same case.

Hit@k was defined as the proportion of questions for which the frozen target report appeared within the first k positions. Hit@1 corresponds to the report actually passed downstream. MRR averaged the reciprocal target rank, assigning greater credit to movement near the top. These metrics answer different questions: Hit@1 evaluates the operational selection decision, whereas MRR detects useful ordering changes that may support future multi-document or retry policies.

The extractive proxy Token-F1 compared answer-bearing target text with evidence available from a ranked candidate. It is a retrieval-oriented approximation of answer availability rather than a generated-answer metric. An increase indicates that the selected or highly ranked evidence contains more target wording, but it does not demonstrate that the generator uses that evidence correctly.

Draft Token-F1 measured overlap between the raw generated response and frozen reference. Final Token-F1 measured the response after checker actions. Precision and recall were calculated over normalized tokens and combined as their harmonic mean. This metric is transparent and reproducible but insensitive to some forms of semantic equivalence and clinical importance. It may reward copied phrasing and penalize acceptable paraphrase.

Automated support rate summarized checker judgments, revision rate measured how often the final response differed from the draft, and abstention rate measured outputs in which no substantive answer was retained. These outcomes were interpreted jointly. A higher revision rate can represent useful filtering or excessive intervention; a higher abstention rate can represent caution or loss of valid coverage.

For paired bootstrap analysis, cases were sampled with replacement and all three questions for each sampled case were included. The 2.5th and 97.5th percentiles of 5,000 paired differences formed the reported 95% interval. Paired randomization swapped condition labels within cases under the null and recalculated the difference. The seed and iteration count were fixed for reproducibility.

Statistical significance was not treated as clinical significance. The candidate pool was controlled, questions were templated, and metrics were automatic. Confidence intervals quantify uncertainty within this benchmark; they do not account for institutional shift, different clinical workflows, or expert disagreement.

## 3.10 Researcher-Reviewed Qualitative Analysis

A post-hoc qualitative protocol was committed after the technical freeze but before systematic case extraction and coding. Some individual outputs had previously been inspected during pipeline verification, so this was not a result-blind preregistration.

The fixed protocol selected 24 representative questions: six target-rank improvements, six target-rank degradations, six QA-gain/support-loss cases, and six correct-retrieval generation-error cases. Each stratum contained two findings, two impression, and two summary questions. The full 360-question numeric index was retained.

Protocol taxonomy v1.0 was preserved in the audit trail. During interpretation, a refined three-level taxonomy v1.1 separated pipeline stage, specific pattern, and outcome modifier. It distinguished target-rank movement from Top-1 success, generation omission from post-verification content loss, and abstention occurrence from its suspected cause. Assistant-proposed v1.1 labels were recorded separately from the original labels. The researcher reviewed and accepted all 24 proposals on 19 August 2026, producing 24 accepted, 0 modified, and 0 excluded cases.

Qualitative counts describe only this predefined purposive review set. They were not used for population-level inference, verifier accuracy estimation, or clinical error-rate estimation.

Case extraction was deterministic. The script read frozen retrieval and QA rows, calculated rank and metric deltas, applied the protocol strata, balanced question roles, and produced a 24-row review package. The public package retained identifiers, metric changes, original protocol labels, proposed refined labels, review status, and concise notes. Full report text and model generations remained local under repository policy.

Taxonomy v1.1 used three levels. Pipeline stage located the issue in retrieval, generation, verification, abstention, or data ambiguity. A specific pattern described the mechanism, such as Top-1 retrieval failure, generation omission, possible verifier over-rejection, or de-identification ambiguity. An outcome modifier recorded cross-stage effects such as QA gain with support loss or no substantive answer loss. This structure avoided treating every observation as one mutually exclusive error.

The researcher decision field distinguished accepted, modified, excluded, and pending cases. In the completed review, all 24 assistant proposals were accepted as the researcher-reviewed labels. This does not make the labels independent or clinically adjudicated. It establishes that the named researcher reviewed the proposed interpretation and accepted it for exploratory analysis.

The qualitative analysis used cautious language. Without physician gold labels, it did not call checker decisions false positives or false negatives. Terms such as possible over-rejection, suspected unnecessary abstention, and abstention consistent with available evidence describe the observed relation among the frozen report, reference, and outputs without asserting definitive clinical correctness.

## 3.11 Computational Cost and Reproducibility

The frozen manifest stored the cohort fingerprint and LF-normalized SHA-256 values for configurations, code, aggregate results, and tests. Large generations, prompt packs, image pixels, model weights, and private full-text review rows remained local.

Generation timing was measured on an NVIDIA GeForce RTX 5070 Laptop GPU with 8,150.6 MiB total memory. These values are machine-, cache-, and generated-length-dependent and do not constitute a complete production latency or energy analysis.

Reproducibility operated at several levels. Configuration reproducibility stored model identifiers, revisions, seeds, shortlist size, fusion weights, thresholds, batch sizes, and generation parameters. Data reproducibility stored case counts, case-level partitions, and a cohort fingerprint; a stable patient identifier was not available for verification. Result reproducibility stored aggregate JSON summaries and statistical outputs. Implementation reproducibility stored scripts and tests. The artifact manifest joined these with LF-normalized SHA-256 values so that unintended changes could be detected across platforms.

Large files were separated from the public repository for practical and licensing reasons. Image archives, local processed image pixels, model weights, full prompt packs, per-question generations, and some detailed review material remained local. Their absence from GitHub is documented rather than hidden. Public aggregate files and indices are sufficient to audit reported counts and metrics, while authorized users can rerun the complete local pipeline after obtaining the source data and models.

Runtime measurement distinguished generation-only time from total processing time and reported throughput and peak allocated GPU memory. Earlier component measurements were retained for image encoding, BM25 retrieval, and cached reranking. The measurements were not combined into a deployment service-level claim because startup, disk cache, concurrent users, web overhead, and hardware power were outside the protocol.

Automated tests covered cohort behavior, multimodal retrieval logic, dashboard integration, and artifact assumptions. Passing tests do not validate scientific claims, but they reduce the risk that reported behavior results from accidental schema drift or broken code paths. Hash checks and tests were rerun after manuscript generation to confirm that documentation changes did not modify frozen V5 artifacts.

## 3.12 Ethics and Claim Boundaries

The system was a research prototype. It did not provide treatment recommendations, authenticate clinical users, or claim deployment safety. V5 did not establish image-based diagnosis, clinical causality, external validation, natural-question generalization, or human-validated verifier correctness. Images and reports were processed locally, and no attempt was made to reverse de-identification.

The source collection is de-identified and publicly distributed for research, but de-identification does not remove every ethical responsibility. The project minimized redistribution of raw content, retained source licensing and citation requirements, and avoided sending images or reports to ordinary online language-model services. Local processing reduced exposure to third-party retention and training policies.

The system was designed for retrospective experimentation rather than clinical access control. A real deployment would require authenticated patient scope, role-based permissions, audit logging, encryption, data-retention governance, and procedures for correcting records. Visual similarity should never be used as a substitute for patient identity when an authorized record identifier is available.

The dashboard language follows this boundary. It can state that the system retrieves the top-ranked candidate report from the indexed corpus and shows the evidence used for generation. It must not state that an arbitrary uploaded image has been matched to the true patient record, that the model diagnosed the image, or that an answer is medically safe.

Finally, the analysis avoids fabricating human evaluation. The completed researcher review supports exploratory interpretation of 24 cases. Independent radiologist correctness, harmfulness, preference, and inter-rater agreement remain future work. This distinction is preserved in the manuscript, repository, and demonstration narrative.

## 3.13 Methodological Summary

The methodology links each research question to a controlled comparison. RQ1 uses question and indication ablations to measure patient-scope ambiguity and then compares the strongest text baseline with correctly aligned image reranking. RQ2 holds the text workflow fixed and replaces the correct image with 100 deterministic fixed-point-free misalignments. RQ3 passes the Top-1 report from report-only and multimodal retrieval through the same non-oracle generator and checker. RQ4 joins frozen metrics with a protocol-driven 24-question qualitative review.

Several safeguards reduce avoidable bias. Cases were fresh relative to earlier project cohorts and split at case level. Confirmation targets were evaluated in a fixed 240-case candidate pool. Model identifiers, revisions, seeds, prompts, thresholds, fusion weights, shortlist size, and statistical iterations were frozen. Case-grouped resampling respected the three-question dependency. Artifact hashes made later changes detectable.

The design also preserves negative evidence. The question-only condition is retained despite poor performance because it measures scope ambiguity. The shuffled-image distribution is retained despite lower scores because it tests alignment. Hit@1 is reported even though its evidence is weaker than MRR. Automated support decline is reported alongside Token-F1 improvement. Qualitative cases include target-rank degradation and suspected verifier disagreement rather than only successful examples.

The methodology does not eliminate all bias. V5 remains within one dataset, uses templated questions, assumes the target is present, and evaluates a single frozen model path. The technical freeze was not a formal preregistration, and the qualitative review was not independent clinical adjudication. These limitations define the appropriate inference: a controlled within-source result about alignment-specific retrieval and downstream automatic QA behavior.

The complete method can be reproduced in stages. A researcher can rebuild the cohort and verify its fingerprint, regenerate embeddings and retrieval rows, rebuild prompt packs, rerun local generation and semantic evaluation, reproduce grouped statistics, verify artifact hashes, and rebuild qualitative materials. The public repository supplies code and aggregate evidence, while source data, weights, images, and large row-level artifacts must be obtained or retained locally according to their licenses and storage policy.

This balance between control, auditability, and bounded claims is central to the project. The method is complex enough to test a genuine multimodal RAG pipeline but modular enough that one favorable end-to-end score cannot hide the behavior of its components.

# Chapter 4: Results and Analysis

## 4.1 Patient-Scope Ambiguity and the Indication Shortcut

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

## 4.2 Indication and Correct-Image Ablation

The correctly aligned image produced a small improvement when used with question text alone: MRR rose from 0.0277 to 0.0515 and proxy Token-F1 from 0.1981 to 0.2334. These values remained low because the generic question supplied little textual case identity.

Against the stronger indication-plus-question BM25 baseline, correctly aligned image reranking increased MRR by 0.0381, with case-bootstrap 95% CI [0.0159, 0.0614] and paired-randomization p=0.0012. Proxy Token-F1 increased by 0.0643, CI [0.0282, 0.1029], p=0.0006. Hit@5 increased by 0.0556 and Hit@10 by 0.0639, with paired-randomization p=0.0024 and p=0.0052 respectively.

The Hit@1 increase was smaller: +0.0333, from 0.5889 to 0.6222. Its confidence interval reached approximately zero and the paired-randomization p-value was 0.0886. Thus, the strongest evidence concerns improved target ordering and retrieval within the upper ranks, not a definitive Hit@1 improvement.

The metric pattern is internally coherent. Image reranking produced larger gains for MRR, Hit@5, Hit@10, and proxy Token-F1 than for Hit@1. This suggests that image information often moved the correct report upward without always moving it past every competing report. The result matches the qualitative rank-improvement cases, where large changes from deep ranks to the upper list still stopped short of first place.

This distinction matters because different applications value ranks differently. A human-facing search tool may benefit when the correct report moves from rank 60 to rank 10. The V5 generator does not, because it consumes only rank one. MRR is therefore evidence of useful representation value, while Hit@1 is the operational measure for the implemented downstream path.

The question-plus-image condition provides another perspective. Its MRR of 0.0515 exceeded question-only BM25 but remained very low. The image alone could not reliably identify a report when reranking was constrained by an underidentified text shortlist. This shows complementarity rather than image dominance: the indication retrieves a plausible region of the corpus, and the aligned image improves ordering within that region.

Equal fusion weighting may also limit Top-1 effects. Text and image signals can disagree, and min-max normalization gives each relative rather than calibrated influence. A different weight might improve Hit@1, but selecting it after observing confirmation outcomes would compromise the frozen comparison. The reported result therefore characterizes the predetermined policy, not the theoretical maximum attainable with BioViL-T.

## 4.3 Correctly Aligned Versus Shuffled Images

Correct alignment achieved MRR 0.6971 and proxy Token-F1 0.7245. Across 100 shuffled-image derangements, mean MRR was 0.5659 with range [0.5158, 0.6084], while mean proxy Token-F1 was 0.5950 with range [0.5310, 0.6455]. No shuffled run equalled or exceeded the correctly aligned result for either metric.

The plus-one Monte Carlo value was 0.0099 for both MRR and proxy Token-F1. The result supports an alignment-specific contribution: the benefit was not reproduced by attaching arbitrary image embeddings to the same text workflow. It does not prove clinical image interpretation, because the task remained closed-set paired-report retrieval and did not test diagnosis from pixels.

The shuffled means were lower than both the correct-alignment result and, for MRR, the indication-plus-question text baseline. Incorrect image fusion can therefore actively disturb a useful text ranking. This is an important safety-oriented observation: adding a visual modality is not automatically beneficial. Its value depends on whether the image is correctly associated with the query and whether the fusion policy handles conflicting signals.

The range across shuffled runs shows that mismatch effects vary with the accidental clinical similarity of assigned images. Some incorrect assignments may resemble the target report and preserve useful ordering, while others introduce misleading similarity. No single shuffled run can summarize this variability. The 100-run distribution demonstrates that the correctly aligned outcome was consistently stronger than the tested misalignments.

The plus-one value of 0.0099 is the smallest value available with 100 permutations when none meets or exceeds the observed statistic. It should not be described as a conventional exact p-value proving image causality. It is an empirical control result under the frozen permutation scheme. The conclusion is appropriately scoped: correct pairing carried information useful for this retrieval task beyond score perturbation caused by arbitrary images.

The control also tests data plumbing. If image paths or case mappings were ignored, correct and shuffled conditions would be similar. Their separation provides evidence that the implemented pipeline actually used the paired image association. It does not reveal which anatomical features the encoder used or whether those features are clinically appropriate.

## 4.4 End-to-End Question Answering

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

## 4.5 Researcher-Reviewed Qualitative Findings

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

## 4.6 Computational Cost

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

## 4.7 Results Summary

V5 established four quantitative conclusions. Indication was the strongest single retrieval signal. Correctly aligned image reranking provided additional target-ordering and proxy-answer gains beyond indication text. Shuffled images did not reproduce the correct-alignment result. The retrieval improvement transferred to final QA Token-F1 but coincided with lower automated support.

The researcher-reviewed analysis explained why aggregate metrics moved differently. Some rank improvements stopped short of Top-1, some wrong-report answers remained locally grounded, some correct-report drafts lost content during verification, and some support-rate decreases reflected template filtering rather than substantive answer loss.

Taken together, the results form a sequential evidence chain. The question-only condition establishes the patient-scope ambiguity. The indication ablation establishes the strength of a text shortcut. Correct-image reranking adds a smaller but statistically supported improvement in target ordering and proxy answer availability. Shuffled images fail to reproduce that improvement, supporting alignment specificity. The non-oracle QA path then shows transfer to generated reference overlap. Finally, the support trade-off and qualitative review demonstrate that retrieval improvement does not remove downstream grounding problems.

This sequence is stronger than presenting only the best multimodal score. Each claim has a corresponding comparison and boundary. The study can claim incremental aligned-image value for retrieval, but not image-only case identification. It can claim transfer to automatic answer overlap, but not clinical correctness. It can describe possible verifier over-rejection in reviewed cases, but not estimate verifier error prevalence.

The negative and mixed results contribute to the research value. Weak Hit@1 evidence, high revision rates, and declining automated support reveal where future work should focus. They also reduce the risk that the dashboard is interpreted as a finished clinical product. The final system is demonstrable and auditable precisely because its uncertainties remain visible.

# Chapter 5: Discussion and Conclusion

## 5.1 Answers to the Research Questions

### RQ1: How do indication and aligned images affect paired-report retrieval?

Indication transformed an almost non-identifying question-only task into a substantially easier retrieval task, increasing MRR from 0.0277 to 0.6590. Correctly aligned image reranking then increased MRR to 0.6971 and improved upper-rank retrieval and proxy Token-F1. The image contribution was incremental rather than dominant and should be interpreted relative to the strong indication shortcut.

The answer is therefore conditional on input availability. When neither patient scope nor indication is available, the repeated question does not define a unique target and image reranking within a BM25 shortlist provides only limited recovery. When indication is available, sparse retrieval identifies a much stronger candidate region and the aligned image improves ordering within it. The two modalities are complementary, but text supplies most of the measured retrieval signal in this benchmark.

The result also changes how the system should be described. It is not an image-to-report identification system operating from pixels alone. It is a case-aware retrieval pipeline that combines referral text with a paired image representation. This wording accurately reflects both the strong indication baseline and the incremental image effect.

### RQ2: Was the image contribution alignment-specific?

Yes within this closed-set benchmark. None of 100 fixed-point-free shuffled-image controls reached the correctly aligned MRR or proxy Token-F1. This supports the claim that correct image-report pairing contributed useful retrieval information. It does not establish diagnostic reasoning or generalization to new clinical images.

The control strengthens the result because arbitrary images were processed through the same encoder, normalization, and fusion code. A gain caused only by activating the image branch or changing score distributions should also appear in some shuffled conditions. The consistent separation instead supports the importance of the stored pairing relation.

The evidence remains representation-level. BioViL-T may exploit anatomy, acquisition view, devices, broad disease patterns, or other image properties. The experiment does not localize these features or compare them with radiologist annotations. Alignment-specific retrieval value and clinically valid visual reasoning are therefore distinct claims.

### RQ3: Did retrieval improvement transfer to downstream QA?

Yes for automatic reference overlap. Multimodal retrieval improved final Token-F1 by 0.0302 with a case-bootstrap interval excluding zero. The same pipeline reduced automated support rate by 0.0340. Better retrieval therefore improved one outcome while exposing limitations in automated grounding measurement and verification behavior.

The transfer result is important because retrieval improvements do not always survive downstream generation. Here, the draft and final Token-F1 differences both favored multimodal retrieval under identical model settings. The evidence selected upstream therefore changed the answer, not only the retrieval table.

At the same time, the modest absolute score and support decline show that transfer was incomplete. The downstream system remained sensitive to Top-1 errors, generator focus, and verifier behavior. The answer to RQ3 is consequently positive but qualified: multimodal retrieval improved automatic reference consistency in the frozen pipeline, not clinical answer validity.

### RQ4: What failure modes remained?

The remaining failures occurred at several stages. Target rank could improve without reaching Top-1. Wrong-report answers could be internally supported but misaligned with the frozen target case. Correct retrieval could still be followed by generation-focus error. Finally, checker filtering could remove report-supported content or only remove harmless template prefixes. These distinct mechanisms cannot be represented by one aggregate support score.

The failure analysis also reveals interactions. A retrieval failure can create an apparently faithful answer; a generation omission can be worsened by verification; a de-identification token can cause a checker disagreement; and an abstention can be either cautious or unnecessary depending on earlier stages. Pipeline stages should therefore be logged separately but interpreted together.

The refined taxonomy is useful precisely because it permits multiple labels. A case can be a Top-1 retrieval failure, a report-faithful generation, and a target-case alignment failure at the same time. Forcing it into one category would conceal the mechanism and lead to an oversimplified error count.

## 5.2 Research Contributions

The first contribution is a reproducible paired image-report retrieval and QA pipeline over real OpenI/IU-Xray cases. The system links text retrieval, BioViL-T image reranking, local generation, sentence-level evidence checking, abstention, and trace preservation.

The second contribution is an alignment-specific evaluation design. Indication ablation prevents the image effect from being confused with referral-text shortcuts, while fixed-point-free shuffled images test whether gains depend on the correct image-report pairing.

The third contribution is evidence that report-level faithfulness and target-case alignment are separate requirements. This extends the earlier cross-case contamination finding: an answer can be well supported by retrieved evidence even when that evidence belongs to the wrong case.

The fourth contribution is a stage-specific qualitative taxonomy with an auditable v1.0-to-v1.1 mapping. It separates retrieval movement, Top-1 outcome, generation behavior, post-verification loss, abstention, and data ambiguity without overwriting the frozen protocol labels.

The fifth contribution is transparent negative evidence. The study reports that Hit@1 evidence was weaker than MRR evidence, support rate declined despite higher Token-F1, some verifier actions appeared excessive, and automatic metrics did not constitute clinical validation.

Together, these contributions move the work beyond a conventional system implementation. The novelty does not lie in inventing BM25, BioViL-T, Qwen, or NLI. It lies in arranging these components into a controlled experiment that asks a specific question about paired evidence identity and follows the answer through retrieval, generation, verification, and qualitative audit.

The contribution is also methodological rather than purely performance-based. A small positive metric difference can be scientifically useful when the comparison isolates alignment and its limitations are explicit. Conversely, a high benchmark score can be weak evidence when indication leakage, patient overlap, or oracle evidence selection is hidden. The thesis prioritizes attribution and auditability over leaderboard breadth.

The public and local artifacts support this contribution. Aggregate results, configuration, hashes, taxonomy documents, and review indices provide a visible audit trail. Large data and generation files remain local, but their role and paths are documented. This balance supports reproducibility without misrepresenting repository completeness or redistributing source data indiscriminately.

Finally, the work connects an engineering concern to a clinical-data principle. Evidence about one patient must not be treated as evidence about another merely because the reports are semantically similar. Target-case alignment is therefore not just another retrieval metric; it is a computational expression of evidence ownership.

## 5.3 Theoretical and Practical Implications

The study supports a layered definition of grounding. Sentence-level support asks whether an answer claim appears in selected evidence. Report-level support asks whether the answer is faithful to the selected report. Target-case alignment asks whether the report is associated with the intended case. These layers are related but not interchangeable.

For system design, the result implies that evidence ownership should be checked before local faithfulness. A verifier applied only after retrieval cannot repair a wrong-case selection if it is restricted to asking whether the answer follows from the selected report. Retrieval traces should therefore expose both the selected case and the evidence used for each answer sentence.

The shuffled-image result also supports using paired images as a reranking signal when patient identity is genuinely unknown within the research task. In a real clinical workflow where an authorized patient record identifier already exists, identity should not be inferred from visual similarity. Authentication and record scope should be enforced first.

Finally, the support-rate trade-off shows that automated verifier metrics require their own evaluation. Lower support may represent removal of unsupported content, over-rejection of supported content, or filtering of harmless formatting. Treating support rate as a gold-standard faithfulness score would conceal these mechanisms.

Theoretically, the results support a layered rather than binary account of grounding. At the lowest layer, an answer sentence has a textual relation to a report. At the next layer, that report has an ownership relation to a target case. At another layer, the answer has a semantic relation to a frozen reference. Clinical validity adds external knowledge, patient context, and expert judgment. A system can succeed at one layer and fail at another, so a single grounded/not-grounded label is insufficient.

This layered account helps clarify the relationship between hallucination and misalignment. A hallucinated statement lacks support in the evidence. A misaligned statement may be fully supported, but by evidence belonging to the wrong target. The mitigation strategies differ. Better generation or NLI may reduce unsupported claims, whereas target misalignment requires stronger scoping, retrieval, identity controls, or abstention before generation.

For benchmark design, indication should be treated as an experimental variable rather than neutral metadata. Referral text can legitimately support retrieval, but it may also leak distinctive report language. Future benchmarks should report performance with and without indication, quantify overlap, and include hard negatives that share clinical terms. Otherwise, improvements may primarily reflect metadata matching.

For multimodal system design, correct pairing is a data-governance requirement as well as a modeling assumption. The shuffled experiment demonstrates that an image from another case can degrade retrieval while still producing a normal-looking score. Production pipelines need reliable association keys, provenance checks, and failure alarms before visual features are fused with records.

For verifier design, the results favor atomic claim representation over coarse sentence filtering. A sentence can contain several findings and one unsupported modifier. Future checkers could decompose statements into finding, location, laterality, severity, temporal, and uncertainty attributes, then preserve supported components. Such a system would still require target-case verification before local entailment.

For user-interface design, uncertainty should be shown through evidence and rank information rather than hidden behind fluent prose. A research dashboard can expose the candidate report identifier, component scores, selected evidence, draft, verifier actions, and final response. It should avoid turning an automatic support score into a green clinical approval badge.

For research practice, mixed outcomes should remain visible. The support-rate decline and weak Hit@1 comparison constrain the claim but also make the analysis more credible. Reporting only MRR and final Token-F1 would omit important failure evidence. A mature medical AI study should present performance, coverage, grounding, alignment, and cost together.

## 5.4 Limitations

The study used one data source. The confirmation cohort was disjoint from prior project cohorts but remained within OpenI/IU-Xray, so the results are not external validation. The task used 240 candidate cases and 120 confirmation targets, which is much smaller and more controlled than a clinical archive. The selection rule required usable images, adequate report text, and a non-empty, non-`normal` problem field, so the benchmark is enriched for abnormality-bearing cases and is not representative of all chest radiographs.

The three questions were report-derived templates rather than radiologist-authored natural questions. Indication text was highly discriminative and may not reflect all real QA scenarios. References were inherited from report sections, and Token-F1 measured wording overlap rather than clinical correctness.

The image encoder was frozen and evaluated as a retrieval signal. The project did not train a vision-language model, diagnose images, localize pathology, or test image-report consistency with independent image-level annotations. The aligned-image result therefore supports paired-report retrieval, not autonomous visual diagnosis.

Only Qwen2.5-1.5B-Instruct and one frozen semantic checker were evaluated in the final path. Larger or clinically specialized generators might behave differently. The checker was not validated against independent expert entailment labels.

The qualitative set was purposively selected by frozen rules and reviewed by the researcher rather than an independent radiologist. Its counts cannot estimate population prevalence, clinical error rates, verifier sensitivity, or safety. The assistant contributed initial coding, although the original and refined labels were kept separately for auditability.

Runtime measurements came from a single laptop GPU and were not complete component-wise production benchmarks. No energy analysis, concurrent-load test, security assessment, or hospital-system integration was performed.

These limitations can be organized as threats to internal, construct, statistical-conclusion, and external validity.

**Internal validity.** The paired design controls many factors, but the multimodal effect may still depend on implementation choices such as shortlist size, normalization, equal fusion weights, and view averaging. These were frozen before confirmation analysis, which prevents outcome-driven tuning but does not prove they are optimal. Cached embeddings and deterministic tie breaking reduce runtime variation and ordering ambiguity. The shuffled control provides evidence that image pairing matters under this implementation, yet it cannot exclude every confound in the learned representation.

The indication is both a legitimate input and a potential shortcut. Because referral language can overlap with report text, part of retrieval performance may reflect lexical leakage rather than general clinical reasoning. The ablation makes this visible, but the benchmark does not contain an independent indication rewritten to remove report-specific phrases. Consequently, the image effect should be interpreted as incremental to this particular text baseline.

The technical freeze was local and prospectively specified, not formally preregistered or externally timestamped before all outcomes were observed. Earlier project development informed the final design, and some individual outputs were inspected during pipeline verification. The fresh cohort and artifact hashes reduce direct adaptation to confirmation cases, but they do not provide the evidential strength of an external preregistration.

**Construct validity.** Hit@1 operationalizes target-report selection in the closed corpus, but it is not patient identification in a clinical system. MRR rewards rank movement even when the downstream generator still receives the wrong report. The extractive proxy measures answer-bearing lexical availability, not retrieval relevance judged by a radiologist. Each metric therefore captures only part of the intended construct.

Token-F1 is sensitive to wording and treats all tokens similarly. It may undervalue valid paraphrases, overvalue copied phrases, and fail to represent the clinical importance of negation, laterality, or severity. References were derived from report sections rather than independently written and adjudicated answers. The QA results establish reproducible reference consistency, not clinical correctness.

Automated support is also an imperfect construct. The NLI checker uses selected-report evidence and cannot assess target ownership. Its sentence segmentation and thresholds may reject supported paraphrases or retain partially unsupported composite claims. The qualitative review identified plausible examples of both substantive and non-substantive filtering, but no independent expert verifier labels were available.

The image construct is limited to BioViL-T embedding utility. The encoder is pretrained on radiology image-text data, and its similarity can improve retrieval, but the experiment does not demonstrate explicit pathology recognition, localization, explanation, or calibration. Calling the result image diagnosis would therefore exceed the measured construct.

**Statistical-conclusion validity.** Grouped bootstrap and paired randomization preserve within-case dependence and are appropriate for paired metric differences. However, the number of independent confirmation cases is 120, not 360. Confidence intervals may remain sensitive to cohort composition, especially for Hit@1 where the estimated difference is small. Because impression and summary references use the same frozen impression text, the aggregate QA metric also gives impression content two question slots per case; question-type sensitivity results are therefore reported as secondary descriptive evidence. Multiple secondary metrics were reported, so individual p-values should be interpreted as supporting evidence within the predefined analysis rather than as independent discoveries.

The 100 shuffled permutations limit the resolution of the Monte Carlo result to 1/101 after plus-one correction. The fact that no shuffled run matched the aligned statistic is strong within that empirical set, but more permutations would provide finer resolution. The shuffled runs are also correlated because they reuse the same cases, questions, and candidate reports.

The qualitative sample was purposive rather than random. Counts within its 24 rows cannot estimate population frequencies, and labels were accepted by the researcher without an independent second reviewer. The analysis supports mechanism interpretation and auditability, not prevalence, sensitivity, specificity, or inter-rater reliability.

**External validity.** All cases came from IU X-Ray/OpenI. The fresh split protects against reuse within the project but does not test another institution, scanner population, reporting style, demographic distribution, or disease prevalence. Candidate-pool size was only 240 and the target was guaranteed to be present. Real archives may contain millions of studies, prior examinations, duplicate text, incomplete linkage, and absent targets.

Questions were three report-derived templates. Natural clinicians use varied phrasing, ask follow-up questions, refer to temporal comparisons, and sometimes request information absent from the report. The system's planner and abstention behavior were not validated under these conditions. Physician-authored RadQA or a newly adjudicated external set would better test linguistic and clinical generalization.

Only one small generator, one vision-language encoder, one sparse retriever, and one checker configuration were frozen in the final path. Results may differ with larger models, alternative medical encoders, dense text retrieval, learned fusion, or claim-level verification. The narrow comparison improves attribution but limits model generalization.

**Ecological and clinical validity.** A real hospital workflow already has authenticated patient and encounter identifiers. It should retrieve within that authorized scope rather than infer identity from image similarity. The benchmark intentionally removes this identity information to study paired retrieval, so its operational setting differs from safe clinical architecture.

No clinicians used the dashboard in practice, and no decision or patient outcome was measured. The project does not establish usefulness, trust, workflow integration, fairness, or safety. These are not minor deployment details; they are separate research questions requiring governance, prospective study, and expert oversight.

## 5.5 Future Work

The highest-priority extension is external evaluation on physician-authored report QA with natural unanswerable questions. RadQA remains appropriate once authorized access is available. Public auxiliary evaluation can use report-grounded datasets while clearly distinguishing datasets derived from the same IU-Xray source from truly external validation.

A stronger benchmark should use free-form clinical questions, independently annotated evidence spans, hard negative reports, and patient-level splits across institutions. Planner evaluation should separately score query reformulation, evidence-type selection, retrieval, reranking, generation, verification, and abstention.

Future verifier studies should obtain independent labels for entailment, contradiction, unsupported additions, composite claims, and appropriate abstention. They should report risk-coverage behavior rather than treating one threshold as universally valid.

Further multimodal work could compare BioViL-T with alternative medical image-text encoders, test multi-view fusion policies, evaluate calibration, and measure performance as the candidate pool grows. These experiments should preserve correct-versus-shuffled alignment controls.

Independent radiologist review remains desirable. It should assess answer correctness, evidence grounding, target-case alignment, harmfulness, and whether verifier filtering removed clinically relevant content. This remains future work rather than a fabricated result.

Future work can be organized into four phases rather than adding components without a validation plan.

The first phase should strengthen the benchmark. Authorized RadQA access would introduce physician-authored questions, natural unanswerable cases, and answer evidence over radiology reports. Because RadQA is derived from a different restricted clinical source, it would also provide a more meaningful external test than another subset of IU-Xray. Until access is obtained, a public auxiliary benchmark can test additional report attributes, but it must be described accurately when it shares the IU-Xray source.

An improved internal benchmark should include paraphrased questions, indication-reduced queries, hard negatives with similar findings, targets absent from the candidate pool, and temporal comparison questions. Patient-level splitting should be supplemented by institution-level separation when possible. Reference answers and evidence spans should be written or verified independently rather than derived only from report sections.

The second phase should improve retrieval while preserving the V5 controls. Alternative sparse and biomedical dense retrievers can be compared on the same candidate set. BioViL-T can be compared with other radiology image-text encoders, and learned fusion can be evaluated using a development-only tuning protocol. Candidate-pool scaling experiments should measure how Hit@1, MRR, latency, and memory change from hundreds to thousands or more reports.

Multi-view fusion deserves a dedicated ablation. Frontal-only, lateral-only, average fusion, max similarity, and attention-based fusion could be compared. Such experiments should retain correctly aligned and fixed-point-free shuffled controls so a more complex fusion method is not credited for generic score perturbation.

The third phase should improve generation and verification. Multiple generators can be tested under the same retrieved evidence, including medically adapted and larger local models. Evaluation should separate concise answer quality, omission, unsupported addition, negation, laterality, uncertainty, and abstention. Claim decomposition could allow a verifier to filter unsupported attributes without deleting an otherwise correct sentence.

Verifier thresholds should be calibrated against independent human labels. A labeled set should include entailed, contradicted, partially supported, and insufficient-evidence claims, with explicit treatment of composite statements. Risk-coverage curves could then show how correctness changes as the system abstains more often. Calibration should be performed on development data and evaluated once on a held-out set.

The fourth phase should evaluate people and workflows. Independent radiologists or appropriately qualified clinicians should review randomized outputs without knowing the system condition. They should score target-case alignment, report faithfulness, answer correctness, clinically important omission, harmfulness, and whether an abstention is appropriate. More than one reviewer would permit agreement analysis and adjudication of disagreements.

A future dashboard study should assess whether traces help users detect errors or merely increase apparent confidence. Participants could compare answers with and without selected-report identifiers, rank information, evidence sentences, and verifier actions. Usability outcomes should be separated from clinical performance, and the interface should continue to avoid claims of patient matching.

Deployment-oriented work would require security and governance beyond model evaluation. Required components include authenticated record scope, access logging, encryption, versioned evidence, incident reporting, human override, monitoring for distribution shift, and procedures for withdrawing an unsafe model. These requirements should be designed with clinicians, information-security staff, and institutional governance rather than added after model development.

The current V5 freeze provides a baseline for these extensions. New experiments should use new version identifiers, predefine their comparisons, preserve the V5 artifacts, and avoid retroactively changing reported outcomes. This allows future improvements to accumulate evidence rather than rewriting the history of the project.

## 5.6 Conclusion

This thesis investigated retrieval-augmented medical question answering over paired chest X-ray images and radiology reports. The final V5 experiment showed that indication text was a strong retrieval shortcut, correctly aligned image reranking added measurable target-ordering value, and shuffled images did not reproduce the aligned result. The resulting retrieval gain transferred to final QA reference overlap.

The study also showed why these gains require careful interpretation. Target-rank improvement did not always produce Top-1 success. Answers could remain faithful to a wrongly selected report. Correct retrieval did not guarantee a reference-consistent final answer. Automated verification sometimes appeared to remove report-supported content, while other support declines reflected only template-prefix filtering.

The final contribution is therefore not a clinically autonomous diagnostic agent. It is a reproducible and auditable framework for separating target-case retrieval, report-level faithfulness, answer generation, verification, abstention, and image-report alignment. These distinctions provide a stronger foundation for future multimodal medical RAG research, but clinical validity requires independent expert evaluation and external data.

In practical terms, the project demonstrates both possibility and restraint. A paired chest X-ray can contribute information beyond referral text when ranking its associated report, and that improvement can benefit a fixed local QA pipeline. The same experiment also shows that a well-grounded sentence can belong to the wrong case, a correct report can still produce an incomplete answer, and a verifier can alter supported content. These are not peripheral exceptions; they define the conditions under which the system should be evaluated and demonstrated.

The central lesson is that evidence must be correct in two senses: it must support the claim, and it must belong to the intended case. Multimodal retrieval addresses part of the second problem, while report-grounded generation and verification address part of the first. Neither layer is sufficient alone. A responsible medical RAG system must preserve both relations and expose them for audit.

Accordingly, the thesis contributes a bounded research result rather than a deployment claim. Within the frozen OpenI/IU-Xray benchmark, correctly aligned images improved target ordering beyond a strong indication baseline, shuffled images did not reproduce the effect, and the gain transferred to automatic answer overlap. Beyond that benchmark, external data, natural questions, independent clinical review, and prospective workflow evaluation remain necessary.
