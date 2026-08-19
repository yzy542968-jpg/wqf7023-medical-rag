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

## 3.2 Data Source and Cohort Construction

The study used de-identified OpenI/IU-Xray chest radiograph cases with linked reports and one or more image views. Each processed case contained a stable case identifier, indication, findings, impression, problem labels, and linked image metadata. De-identification placeholders were retained because replacing or inferring their hidden content could introduce unsupported information.

The V5 cohort contained 240 cases that were excluded from all earlier project cohort manifests. A fixed seed of 7023 divided these into 120 development cases and 120 confirmation cases. The confirmation set contributed 360 report-derived questions: one findings question, one impression question, and one summary question per case. Statistical resampling and comparison used the case identifier as the grouping unit so that the three questions from one case were not treated as independent patients.

Retrieval used all 240 fresh-cohort cases as the candidate pool. The 120 confirmation cases were the target cases for final evaluation. This design measured closed-set paired-report retrieval; it did not evaluate diagnosis of previously unseen patients.

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

## 3.5 Text Retrieval

BM25 provided the transparent sparse retrieval baseline. The question-only condition intentionally exposed patient-scope ambiguity because the three generic question templates contained little case-specific information. The indication-plus-question condition tested how much clinical referral text reduced this ambiguity.

The multimodal conditions first produced the same text shortlist of 100 candidate cases. Scores were normalized independently within that shortlist. Ties were resolved by descending fused score and then ascending case identifier, giving a deterministic ranking.

## 3.6 Image Encoding and Multimodal Reranking

Image and report representations used `microsoft/BiomedVLP-BioViL-T` with frozen revision `692f09e` and 128-dimensional embeddings. Each available X-ray view was normalized, case views were averaged, and the resulting case vector was normalized again. No image pixels were sent to an online service.

For multimodal reranking, normalized text and image similarities received equal weights of 0.5. The correctly aligned condition used the image embedding linked to the target case. This image was used only as a retrieval query signal; the system did not generate a diagnosis directly from pixels. The downstream generator received the selected report evidence rather than raw images.

## 3.7 Shuffled-Image Control

The alignment control used 100 deterministic fixed-point-free permutations with seed 7023. In each permutation, every source case received another case's image embedding and no case retained its own image. Text queries, candidate reports, shortlist size, fusion weights, and evaluation procedure remained unchanged.

The control tested whether the correctly aligned image outperformed image-conditioned reranking with incorrect case alignment. It did not prove causal clinical image understanding. A plus-one Monte Carlo value was calculated as `(b+1)/(m+1)`, where `b` was the number of shuffled runs meeting or exceeding the correctly aligned result and `m=100` was the number of permutations.

## 3.8 Answer Generation and Semantic Verification

Both report-only and multimodal retrieval conditions used the same local `Qwen/Qwen2.5-1.5B-Instruct` generator. Generation used CUDA, float16, batch size 16, maximum 256 new tokens, temperature 0, and a direct non-oracle prompt. The generator did not receive the frozen target identifier or reference answer.

The semantic checker used `pritamdeka/PubMedBERT-MNLI-MedNLI`. It combined lexical evidence matching, entailment and contradiction probabilities, and polarity consistency. Its locked configuration used lexical weight 0.2, support threshold 0.6, entailment threshold 0.75, and contradiction threshold 0.5. Evidence scope was restricted to the top-ranked selected report. The action path could retain supported sentences, filter flagged sentences, or abstain if no usable answer remained.

The checker was an automated evidence signal rather than a clinical gold standard. Its support rate measured agreement with selected-report evidence, not target-patient correctness or clinical safety.

## 3.9 Evaluation Metrics and Statistical Analysis

Retrieval metrics were Hit@1, Hit@5, Hit@10, MRR, and an extractive proxy Token-F1 calculated from the selected report evidence. Hit@1 measured Top-1 target-case alignment; MRR retained information about target-rank movement even when the target did not reach first place.

QA metrics were draft Token-F1, final Token-F1 after semantic checking, automated evidence-support rate, revision rate, and abstention rate. Token-F1 measured reference overlap and was not interpreted as clinical correctness.

V5 used 5,000 grouped bootstrap resamples at case level and paired randomization tests with seed 7023. The primary retrieval comparison was indication-plus-question with correctly aligned image minus indication-plus-question BM25. The primary QA comparison was multimodal final Token-F1 minus report-only final Token-F1. Confidence intervals and p-values therefore preserved the dependence among questions from the same case.

## 3.10 Researcher-Reviewed Qualitative Analysis

A post-hoc qualitative protocol was committed after the technical freeze but before systematic case extraction and coding. Some individual outputs had previously been inspected during pipeline verification, so this was not a result-blind preregistration.

The fixed protocol selected 24 representative questions: six target-rank improvements, six target-rank degradations, six QA-gain/support-loss cases, and six correct-retrieval generation-error cases. Each stratum contained two findings, two impression, and two summary questions. The full 360-question numeric index was retained.

Protocol taxonomy v1.0 was preserved in the audit trail. During interpretation, a refined three-level taxonomy v1.1 separated pipeline stage, specific pattern, and outcome modifier. It distinguished target-rank movement from Top-1 success, generation omission from post-verification content loss, and abstention occurrence from its suspected cause. Assistant-proposed v1.1 labels were recorded separately from the original labels. The researcher reviewed and accepted all 24 proposals on 19 August 2026, producing 24 accepted, 0 modified, and 0 excluded cases.

Qualitative counts describe only this predefined purposive review set. They were not used for population-level inference, verifier accuracy estimation, or clinical error-rate estimation.

## 3.11 Computational Cost and Reproducibility

The frozen manifest stored the cohort fingerprint and LF-normalized SHA-256 values for configurations, code, aggregate results, and tests. Large generations, prompt packs, image pixels, model weights, and private full-text review rows remained local.

Generation timing was measured on an NVIDIA GeForce RTX 5070 Laptop GPU with 8,150.6 MiB total memory. These values are machine-, cache-, and generated-length-dependent and do not constitute a complete production latency or energy analysis.

## 3.12 Ethics and Claim Boundaries

The system was a research prototype. It did not provide treatment recommendations, authenticate clinical users, or claim deployment safety. V5 did not establish image-based diagnosis, clinical causality, external validation, natural-question generalization, or human-validated verifier correctness. Images and reports were processed locally, and no attempt was made to reverse de-identification.

# Chapter 4: Results and Analysis

## 4.1 Patient-Scope Ambiguity and the Indication Shortcut

Table 4.1 shows the four principal confirmation retrieval conditions.

| Input condition | Hit@1 | Hit@5 | Hit@10 | MRR | Extractive proxy Token-F1 |
|---|---:|---:|---:|---:|---:|
| Question only, BM25 | 0.0056 | 0.0222 | 0.0472 | 0.0277 | 0.1981 |
| Indication + question, BM25 | 0.5889 | 0.7222 | 0.7750 | 0.6590 | 0.6602 |
| Question + correctly aligned image | 0.0139 | 0.0722 | 0.1139 | 0.0515 | 0.2334 |
| Indication + question + correctly aligned image | 0.6222 | 0.7778 | 0.8389 | 0.6971 | 0.7245 |

Question-only retrieval was nearly non-identifying: Hit@1 was 0.0056 and MRR was 0.0277. This was expected because the same three templates were reused across cases. Adding indication increased Hit@1 to 0.5889 and MRR to 0.6590. The indication therefore acted as a powerful retrieval shortcut in this controlled benchmark.

The effect is methodologically important. A high retrieval score cannot be attributed only to sophisticated multimodal reasoning when referral text already contains strong case-discriminating language. For this reason, V5 reports indication ablation explicitly and treats the indication-plus-question BM25 condition as the primary text baseline.

## 4.2 Indication and Correct-Image Ablation

The correctly aligned image produced a small improvement when used with question text alone: MRR rose from 0.0277 to 0.0515 and proxy Token-F1 from 0.1981 to 0.2334. These values remained low because the generic question supplied little textual case identity.

Against the stronger indication-plus-question BM25 baseline, correctly aligned image reranking increased MRR by 0.0381, with case-bootstrap 95% CI [0.0159, 0.0614] and paired-randomization p=0.0012. Proxy Token-F1 increased by 0.0643, CI [0.0282, 0.1029], p=0.0006. Hit@5 increased by 0.0556 and Hit@10 by 0.0639, with paired-randomization p=0.0024 and p=0.0052 respectively.

The Hit@1 increase was smaller: +0.0333, from 0.5889 to 0.6222. Its confidence interval reached approximately zero and the paired-randomization p-value was 0.0886. Thus, the strongest evidence concerns improved target ordering and retrieval within the upper ranks, not a definitive Hit@1 improvement.

## 4.3 Correctly Aligned Versus Shuffled Images

Correct alignment achieved MRR 0.6971 and proxy Token-F1 0.7245. Across 100 shuffled-image derangements, mean MRR was 0.5659 with range [0.5158, 0.6084], while mean proxy Token-F1 was 0.5950 with range [0.5310, 0.6455]. No shuffled run equalled or exceeded the correctly aligned result for either metric.

The plus-one Monte Carlo value was 0.0099 for both MRR and proxy Token-F1. The result supports an alignment-specific contribution: the benefit was not reproduced by attaching arbitrary image embeddings to the same text workflow. It does not prove clinical image interpretation, because the task remained closed-set paired-report retrieval and did not test diagnosis from pixels.

## 4.4 End-to-End Question Answering

Table 4.2 compares the same generator and checker after report-only and multimodal retrieval.

| Pipeline | Draft Token-F1 | Final Token-F1 | Automated support | Final abstention | Revision rate |
|---|---:|---:|---:|---:|---:|
| Report-only retrieval | 0.3632 | 0.3563 | 0.8409 | 0.0556 | 0.7389 |
| Multimodal retrieval | 0.3897 | 0.3865 | 0.8069 | 0.0611 | 0.7250 |

Multimodal retrieval improved draft Token-F1 by 0.0265, CI [0.0094, 0.0441], paired-randomization p=0.0026. Final Token-F1 improved by 0.0302, CI [0.0101, 0.0511], p=0.0032. This demonstrates that the retrieval gain transferred to the final QA output under a fixed non-oracle generation path.

However, automated evidence support decreased by 0.0340, CI [-0.0566, -0.0122], p=0.0034. Final abstention increased by only 0.0056, with an interval crossing zero and p=0.7299. The central result is therefore a performance-grounding trade-off: reference overlap improved while the automated support signal declined.

This trade-off must not be simplified into a claim that multimodal answers were less clinically faithful. The support metric was produced by the same automated checker later shown to filter both substantive sentences and generic answer prefixes.

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

## 4.6 Computational Cost

| Pipeline condition | Records | Total process | Generation only | Generation throughput | Peak allocated GPU memory |
|---|---:|---:|---:|---:|---:|
| Report-only | 360 | 87.86 s | 78.56 s | 4.58 records/s | 3,437 MiB |
| Multimodal | 360 | 98.70 s | 89.31 s | 4.03 records/s | 3,437 MiB |

Both QA runs used the same Qwen model and generation settings. The multimodal prompt path took approximately 10.84 seconds longer in total and generated 0.55 fewer records per second, while peak allocated memory remained effectively unchanged. Earlier V4.2 measurements recorded approximately 14.91 ms mean single-image encoding, 1.73 ms BM25 retrieval, 0.28 ms cached reranking, and a 16.93 ms warm paired-request estimate.

These measurements show that the final system was feasible on a laptop GPU. They do not provide complete end-to-end production latency, energy consumption, or deployment cost.

## 4.7 Results Summary

V5 established four quantitative conclusions. Indication was the strongest single retrieval signal. Correctly aligned image reranking provided additional target-ordering and proxy-answer gains beyond indication text. Shuffled images did not reproduce the correct-alignment result. The retrieval improvement transferred to final QA Token-F1 but coincided with lower automated support.

The researcher-reviewed analysis explained why aggregate metrics moved differently. Some rank improvements stopped short of Top-1, some wrong-report answers remained locally grounded, some correct-report drafts lost content during verification, and some support-rate decreases reflected template filtering rather than substantive answer loss.

# Chapter 5: Discussion and Conclusion

## 5.1 Answers to the Research Questions

### RQ1: How do indication and aligned images affect paired-report retrieval?

Indication transformed an almost non-identifying question-only task into a substantially easier retrieval task, increasing MRR from 0.0277 to 0.6590. Correctly aligned image reranking then increased MRR to 0.6971 and improved upper-rank retrieval and proxy Token-F1. The image contribution was incremental rather than dominant and should be interpreted relative to the strong indication shortcut.

### RQ2: Was the image contribution alignment-specific?

Yes within this closed-set benchmark. None of 100 fixed-point-free shuffled-image controls reached the correctly aligned MRR or proxy Token-F1. This supports the claim that correct image-report pairing contributed useful retrieval information. It does not establish diagnostic reasoning or generalization to new clinical images.

### RQ3: Did retrieval improvement transfer to downstream QA?

Yes for automatic reference overlap. Multimodal retrieval improved final Token-F1 by 0.0302 with a case-bootstrap interval excluding zero. The same pipeline reduced automated support rate by 0.0340. Better retrieval therefore improved one outcome while exposing limitations in automated grounding measurement and verification behavior.

### RQ4: What failure modes remained?

The remaining failures occurred at several stages. Target rank could improve without reaching Top-1. Wrong-report answers could be internally supported but misaligned with the frozen target case. Correct retrieval could still be followed by generation-focus error. Finally, checker filtering could remove report-supported content or only remove harmless template prefixes. These distinct mechanisms cannot be represented by one aggregate support score.

## 5.2 Research Contributions

The first contribution is a reproducible paired image-report retrieval and QA pipeline over real OpenI/IU-Xray cases. The system links text retrieval, BioViL-T image reranking, local generation, sentence-level evidence checking, abstention, and trace preservation.

The second contribution is an alignment-specific evaluation design. Indication ablation prevents the image effect from being confused with referral-text shortcuts, while fixed-point-free shuffled images test whether gains depend on the correct image-report pairing.

The third contribution is evidence that report-level faithfulness and target-case alignment are separate requirements. This extends the earlier cross-case contamination finding: an answer can be well supported by retrieved evidence even when that evidence belongs to the wrong case.

The fourth contribution is a stage-specific qualitative taxonomy with an auditable v1.0-to-v1.1 mapping. It separates retrieval movement, Top-1 outcome, generation behavior, post-verification loss, abstention, and data ambiguity without overwriting the frozen protocol labels.

The fifth contribution is transparent negative evidence. The study reports that Hit@1 evidence was weaker than MRR evidence, support rate declined despite higher Token-F1, some verifier actions appeared excessive, and automatic metrics did not constitute clinical validation.

## 5.3 Theoretical and Practical Implications

The study supports a layered definition of grounding. Sentence-level support asks whether an answer claim appears in selected evidence. Report-level support asks whether the answer is faithful to the selected report. Target-case alignment asks whether the report is associated with the intended case. These layers are related but not interchangeable.

For system design, the result implies that evidence ownership should be checked before local faithfulness. A verifier applied only after retrieval cannot repair a wrong-case selection if it is restricted to asking whether the answer follows from the selected report. Retrieval traces should therefore expose both the selected case and the evidence used for each answer sentence.

The shuffled-image result also supports using paired images as a reranking signal when patient identity is genuinely unknown within the research task. In a real clinical workflow where an authorized patient record identifier already exists, identity should not be inferred from visual similarity. Authentication and record scope should be enforced first.

Finally, the support-rate trade-off shows that automated verifier metrics require their own evaluation. Lower support may represent removal of unsupported content, over-rejection of supported content, or filtering of harmless formatting. Treating support rate as a gold-standard faithfulness score would conceal these mechanisms.

## 5.4 Limitations

The study used one data source. The confirmation cohort was disjoint from prior project cohorts but remained within OpenI/IU-Xray, so the results are not external validation. The task used 240 candidate cases and 120 confirmation targets, which is much smaller and more controlled than a clinical archive.

The three questions were report-derived templates rather than radiologist-authored natural questions. Indication text was highly discriminative and may not reflect all real QA scenarios. References were inherited from report sections, and Token-F1 measured wording overlap rather than clinical correctness.

The image encoder was frozen and evaluated as a retrieval signal. The project did not train a vision-language model, diagnose images, localize pathology, or test image-report consistency with independent image-level annotations. The aligned-image result therefore supports paired-report retrieval, not autonomous visual diagnosis.

Only Qwen2.5-1.5B-Instruct and one frozen semantic checker were evaluated in the final path. Larger or clinically specialized generators might behave differently. The checker was not validated against independent expert entailment labels.

The qualitative set was purposively selected by frozen rules and reviewed by the researcher rather than an independent radiologist. Its counts cannot estimate population prevalence, clinical error rates, verifier sensitivity, or safety. The assistant contributed initial coding, although the original and refined labels were kept separately for auditability.

Runtime measurements came from a single laptop GPU and were not complete component-wise production benchmarks. No energy analysis, concurrent-load test, security assessment, or hospital-system integration was performed.

## 5.5 Future Work

The highest-priority extension is external evaluation on physician-authored report QA with natural unanswerable questions. RadQA remains appropriate once authorized access is available. Public auxiliary evaluation can use report-grounded datasets while clearly distinguishing datasets derived from the same IU-Xray source from truly external validation.

A stronger benchmark should use free-form clinical questions, independently annotated evidence spans, hard negative reports, and patient-level splits across institutions. Planner evaluation should separately score query reformulation, evidence-type selection, retrieval, reranking, generation, verification, and abstention.

Future verifier studies should obtain independent labels for entailment, contradiction, unsupported additions, composite claims, and appropriate abstention. They should report risk-coverage behavior rather than treating one threshold as universally valid.

Further multimodal work could compare BioViL-T with alternative medical image-text encoders, test multi-view fusion policies, evaluate calibration, and measure performance as the candidate pool grows. These experiments should preserve correct-versus-shuffled alignment controls.

Independent radiologist review remains desirable. It should assess answer correctness, evidence grounding, target-case alignment, harmfulness, and whether verifier filtering removed clinically relevant content. This remains future work rather than a fabricated result.

## 5.6 Conclusion

This thesis investigated retrieval-augmented medical question answering over paired chest X-ray images and radiology reports. The final V5 experiment showed that indication text was a strong retrieval shortcut, correctly aligned image reranking added measurable target-ordering value, and shuffled images did not reproduce the aligned result. The resulting retrieval gain transferred to final QA reference overlap.

The study also showed why these gains require careful interpretation. Target-rank improvement did not always produce Top-1 success. Answers could remain faithful to a wrongly selected report. Correct retrieval did not guarantee a reference-consistent final answer. Automated verification sometimes appeared to remove report-supported content, while other support declines reflected only template-prefix filtering.

The final contribution is therefore not a clinically autonomous diagnostic agent. It is a reproducible and auditable framework for separating target-case retrieval, report-level faithfulness, answer generation, verification, abstention, and image-report alignment. These distinctions provide a stronger foundation for future multimodal medical RAG research, but clinical validity requires independent expert evaluation and external data.
