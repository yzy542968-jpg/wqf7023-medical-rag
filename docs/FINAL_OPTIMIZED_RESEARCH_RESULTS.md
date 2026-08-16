# Final Optimized Research Results

## Study Claim

The final project is not simply a dashboard or a prompt-engineering exercise. It tests a specific safety claim: radiology QA should retrieve and expose one coherent patient case, adapt when retrieval confidence is weak, and verify generated claims against that case before returning an answer.

The proposed system is therefore:

1. A planner converts the question into a retrieval query.
2. BM25 and MedCPT scores are fused using a development-selected hybrid weight.
3. A MedCPT cross-encoder conditionally challenges uncertain hybrid rankings.
4. A locked policy selects one case or abstains under low retrieval confidence.
5. Qwen2.5-1.5B generates an answer from the selected case only.
6. A hybrid lexical, negation-rule, and Medical NLI checker removes unsupported or contradictory sentences and abstains if none remain.

## Case-Disjoint Evaluation

The benchmark contains 120 OpenI cases and 360 generated report-grounded questions. Every case contributes findings, impression, and abnormality-summary questions.

The split was frozen by case, not by question:

| Split | Cases | Questions | Questions per type |
|---|---:|---:|---:|
| Development | 84 | 252 | 84 |
| Held-out test | 36 | 108 | 36 |

No case ID occurs in both splits. Retrieval weights, reranking depth, adaptive-policy thresholds, prompt choice, and semantic-checker thresholds were selected on development data only. The held-out test was then evaluated once.

## Selected Configuration

| Component | Locked choice |
|---|---|
| Sparse retriever | BM25 |
| Dense retriever | MedCPT Query Encoder |
| Hybrid BM25 weight | 0.30 |
| Reranker | MedCPT Cross-Encoder over top 3 |
| Adaptive reranker margin | 0.25 |
| Adaptive base margin | 0.15 |
| Minimum base score | 0.90 |
| Minimum selected margin | 0.10 |
| Prompt | Direct |
| Generator | Qwen2.5-1.5B-Instruct, deterministic decoding |
| Medical NLI model | BioLinkBERT fine-tuned on MedNLI |
| Lexical weight | 0.20 |
| Combined support threshold | 0.60 |
| Entailment threshold | 0.75 |
| Contradiction threshold | 0.50 |

## Retrieval Results

The development-selected hybrid weight was `alpha=0.30`.

| Split | Hit@1 | Hit@20 | MRR |
|---|---:|---:|---:|
| Development | 0.246 | 0.516 | 0.323 |
| Held-out test | 0.287 | 0.509 | 0.331 |

The MedCPT reranker improved development MRR from 0.323 to 0.335 at candidate depth 3, but reduced held-out MRR to 0.317 and Hit@1 to 0.259. It was therefore not accepted as an unconditional replacement.

The adaptive policy uses the reranker only when it is confident and the hybrid ranker is uncertain. On held-out data it achieved:

- coverage: 0.926;
- overall top-1 accuracy: 0.287;
- selective accuracy among answered cases: 0.310;
- retrieval abstention: 0.074.

This policy did not raise overall top-1 accuracy above the fixed hybrid retriever. Its value is calibrated selectivity and an auditable decision trace, not a claimed ranking breakthrough.

## Prompt Ablation

Prompt choice was made on the development split.

| Prompt | Draft Token-F1 | Verified Token-F1 | Support | Abstention |
|---|---:|---:|---:|---:|
| Direct | 0.233 | 0.255 | 0.779 | 0.067 |
| Evidence-guided | 0.218 | 0.207 | 0.678 | 0.119 |
| Structured case-aware | 0.199 | 0.186 | 0.697 | 0.083 |

Direct prompting was selected. More instructions did not improve this local 1.5B generator, which is itself a useful negative result.

## Held-Out Answer Results

Grouped bootstrap resampling used the 36 held-out cases as the sampling unit with 10,000 iterations.

| System | Mean Token-F1 | Grouped 95% CI |
|---|---:|---:|
| LLM only | 0.079 | [0.069, 0.089] |
| Report-RAG BM25 + semantic checker | 0.153 | [0.131, 0.175] |
| Case BM25 + semantic checker | 0.172 | [0.136, 0.211] |
| Previous Hybrid + semantic checker | 0.204 | [0.174, 0.238] |
| Final adaptive system, draft | 0.199 | [0.166, 0.233] |
| Final adaptive system, verified | 0.206 | [0.167, 0.246] |

Important paired comparisons:

| Comparison | Mean difference | 95% CI | Randomization p | Holm-adjusted p |
|---|---:|---:|---:|---:|
| Final verified vs Case BM25 verified | +0.035 | [0.007, 0.062] | 0.0145 | 0.0870 |
| Final verified vs LLM only | +0.128 | [0.090, 0.167] | <0.0001 | 0.0010 |
| Final verified vs final draft | +0.007 | [-0.005, 0.022] | 0.3244 | 1.0000 |
| Final verified vs previous Hybrid verified | +0.002 | [-0.028, 0.032] | 0.8787 | 1.0000 |

The defensible conclusion is that the final system clearly outperforms LLM-only. Its positive difference over Case BM25 has a 95% bootstrap CI above zero and nominal randomization p below 0.05, but it is not significant after Holm correction across the 10 exploratory comparisons. It is statistically indistinguishable from the previous fixed Hybrid system on Token-F1, while adding grouped model selection, semantic contradiction checks, adaptive abstention, and single-case evidence isolation.

## Medical Entity Evaluation

RadGraph was applied to all 108 held-out questions.

| Final system output | Entity F1 | Entity-relation F1 | Complete F1 |
|---|---:|---:|---:|
| Draft | 0.187 | 0.162 | 0.164 |
| Semantic-agent output | 0.189 | 0.164 | 0.163 |

The semantic checker improves Token-F1 slightly but does not improve RadGraph complete F1. This suggests that sentence removal can improve lexical alignment while occasionally deleting clinically structured content. The limitation is reported rather than hidden.

## Cross-Case Contamination

A separate sentence-aligned hybrid Medical NLI analysis evaluated report-level BM25 top-5 answers on the held-out split. For each answer sentence, support from the first retrieved case was compared with support from the remaining four cases. Because automatic NLI can produce false entailments, both a broad semantic estimate and a conservative estimate requiring lexical overlap of at least 0.20 are reported.

- Broad semantic estimate: 28.9% of sentences and 65.7% of answers contained cross-case support.
- Lexically anchored conservative estimate: 19.4% of sentences and 57.4% of answers contained cross-case support.
- The corresponding unsupported-sentence estimates were 20.4% broad and 42.6% conservative.

These automatic estimates support the final architecture but are not treated as human-confirmed prevalence. A conventional top-k context can produce a fluent answer assembled from different patients. The final system structurally prevents multi-case assembly by exposing one selected case to generation and verification.

## Oracle Retrieval Diagnostic

An oracle diagnostic exposed the correct target case to the same direct prompt, Qwen2.5-1.5B generator, and semantic checker. It was run after the final system was frozen and was not used for model selection.

| Retrieval condition | Draft Token-F1 | Verified Token-F1 |
|---|---:|---:|
| Actual adaptive retrieval | 0.199 | 0.206 |
| Oracle target case | 0.365 | 0.425 |

The absolute verified gap is 0.219 and the oracle score is 2.06 times the actual score. Retrieval is therefore the dominant remaining performance bottleneck; increasing generator size is not the first optimization priority.

Conditioned analysis supports the same conclusion. For the 31 correctly retrieved questions, verified Token-F1 is 0.451. For the 69 wrong but answered retrievals, it is 0.117 even though evidence support against the wrong retrieved case remains 83.4%. The verifier measures conditional faithfulness, not target-case correctness.

## Verifier Stress Test

A development-only synthetic stress test paired 120 explicit negative evidence sentences with polarity-flipped contradictory claims. After adding a high-overlap polarity hard guard:

- original entailed sentence acceptance: 100%;
- polarity-flipped contradiction rejection: 100%;
- unit and integration tests: 33 passed at the time of this result freeze.

This test demonstrates explicit polarity sensitivity, not general clinical verifier accuracy. Independent human evidence labels remain necessary.

## Validity Audit

The second-pass audit found two benchmark limitations that must be reported:

1. Query-document shortcut: the indication is copied verbatim into all findings and impression queries and is also present in the indexed BM25 and MedCPT document. Problem labels used for summary queries occur in the MedCPT title. The benchmark therefore measures metadata matching as well as semantic retrieval.
2. Query non-identifiability: 68 of 360 rows (18.9%) share an identical query with at least one different target case. In the held-out set this affects 25 of 108 rows (23.1%). A deterministic retriever cannot uniquely recover different target reports from identical input strings.

These issues do not invalidate the case-boundary and evidence-checking experiments, but they weaken claims about general open-corpus retrieval accuracy. A v2 benchmark should use identifiable real questions with independently defined multi-case relevance judgments, or should evaluate evidence retrieval within a known patient case.

## Limitations

1. Questions are automatically derived from radiology metadata and contain query-document shortcuts; the benchmark does not represent unconstrained clinical questioning.
2. The held-out set contains only 36 cases; grouped confidence intervals remain wide.
3. Token-F1 and RadGraph are imperfect proxies for expert clinical correctness.
4. The Medical NLI model was trained on MedNLI rather than this exact OpenI task.
5. Adaptive retrieval improved selective accuracy but not overall top-1 accuracy.
6. Identical queries can map to different target reports, making part of strict target-case retrieval ill-posed.
7. Evidence support is conditional on the retrieved case and does not detect a fluent answer grounded in the wrong patient.
8. Only 10 local X-ray examples are available; images are displayed but are not inputs to retrieval, generation, or verification. The modeled pipeline is text-only, not multimodal RAG.
9. Cross-case contamination rates are automatic detector estimates and require human confirmation.
10. The dashboard is a research demonstrator and not a diagnostic system.
11. Independent human evaluation remains necessary and must not be used for further tuning.

## Evidence Files

- `data/splits/openi_qa_grouped_case_seed7023.json`
- `experiments/final_optimized/retrieval/hybrid_alpha_selection.json`
- `experiments/final_optimized/reranking/medcpt_reranker_selection.json`
- `experiments/final_optimized/adaptive_retrieval/adaptive_policy_selection.json`
- `experiments/final_optimized/prompt_ablation/prompt_selection.json`
- `experiments/final_optimized/semantic_agent/semantic_agent_selection.json`
- `experiments/final_optimized/final_test/final_optimized_test_summary.json`
- `experiments/final_optimized/statistics/held_out_test_grouped_bootstrap.json`
- `experiments/final_optimized/radgraph/held_out_radgraph_summary.csv`
- `experiments/final_optimized/contamination/report_rag_cross_case_contamination.json`
- `experiments/final_optimized/oracle_test/final_optimized_test_summary.json`
- `experiments/final_optimized/verifier_stress_test/development_polarity_stress_test.json`
- `experiments/final_optimized/validity_audit/research_validity_audit.json`

## Method Foundations

- MedCPT implementation: <https://github.com/ncbi/MedCPT>
- MedNLI paper: <https://aclanthology.org/K18-1234/>
- RadGraph implementation: <https://github.com/Stanford-AIMI/radgraph>
- RAGChecker framework: <https://github.com/amazon-science/RAGChecker>
