# Post-submission experiments

These experiments were run after the P2 submission was frozen. They strengthen the software and provide additional research evidence, but they are not claims made by the submitted manuscript or defence deck.

## V2.1 hard evidence benchmark

The v2.1 benchmark contains 240 OpenI cases and 1,440 questions. Its cases exclude the earlier V1 seed, V2 main benchmark, and V2 confirmation benchmark. Each case contributes three answerable questions and three unanswerable questions. The set includes paraphrased section requests, report-derived fact probes, out-of-scope requests, near-domain absent-finding requests, and non-target report sections as distractors.

The Agent receives only the case scope and natural-language question. The recorded `expected_intent` is retained for scoring and is not passed to `ClosedLoopEvidenceAgent.run`. Both systems may return at most six unique chunks per question. The baseline uses one report-wide BM25 call; the Agent uses up to two calls with a fixed plan, evidence assessment, optional rewrite, and retry. Answerability thresholds are selected on development only and then frozen for calibration and test.

### Independent test results

| Metric | Fixed report BM25 | Route-only ablation | Closed-loop Agent V2 |
|---|---:|---:|---:|
| Answerability macro F1 | 0.6934 | 0.9306 | 0.9583 |
| False-answer rate | 0.1991 | 0.0648 | 0.0556 |
| Evidence retrieval hit rate | 0.8426 | 0.9676 | 0.9954 |
| End-to-end action accuracy | 0.6759 | 0.9306 | 0.9583 |
| Answerable Token-F1 | 0.2845 | 0.7161 | 0.6893 |
| AURC, lower is better | 0.2829 | 0.0360 | 0.0136 |
| Raw ECE, lower is better | 0.2681 | 0.2293 | 0.2433 |
| Mean retrieved chunks | 5.9861 | 3.3032 | 2.2454 |
| Mean retrieval calls | 1.0000 | 0.6667 | 1.0324 |

The route-only ablation uses the same lexical planner but disables query rewriting and the second retrieval call. Its strong result shows that most of the gain comes from scoped routing and explicit out-of-scope rejection. The closed loop adds `0.0277` macro F1, raises evidence hit rate by `0.0278`, and uses about one fewer chunk per question. Route-only has slightly higher extractive Token-F1 because it returns more text; the loop does not dominate every metric.

Platt scaling is fitted on the calibration split only. On untouched test data it reduces closed-loop ECE from `0.2433` to `0.0171` and Brier score from `0.1358` to `0.0361`; macro F1 changes from `0.9583` to `0.9537` at the fixed calibrated probability threshold of `0.5`. For route-only, ECE falls from `0.2293` to `0.0120`. Calibration improves probability interpretation but does not create the underlying ranking.

The benchmark is deterministic and automatically labelled from report structure. It tests report-grounded routing and selective behavior; it is not a substitute for expert clinical evaluation. Because the task and policy were co-designed, the separate locked replication below remains necessary.

### Reserved wording-transfer stress test

The frozen Agent and development-selected answerability threshold were evaluated on the same 72 v2.1 test cases after every question was rewritten with reserved wording that did not occur in development or calibration. No parameter was changed after this evaluation.

| Metric | Original wording | Reserved transfer wording |
|---|---:|---:|
| Answerability macro F1 | 0.9583 | 0.6972 |
| False-answer rate | 0.0556 | 0.1806 |
| Evidence retrieval hit rate | 0.9954 | 0.6944 |
| End-to-end action accuracy | 0.9583 | 0.6968 |
| Answerable Token-F1 | 0.6893 | 0.3459 |

Of 432 transfer questions, there are 90 missed answerable questions, 39 false answers to unanswerable questions, 2 retrieval misses after choosing to answer, and 16 wrong-section routes despite retrieving some relevant evidence. The phrase “overall interpretation” is consistently routed to Findings rather than Impression. This establishes a clear boundary: V2.1 is an auditable rule-based evidence Agent, not a language-general clinical Agent.

## V2.2 constrained semantic planner

A frozen zero-shot planner prompt asks `Qwen/Qwen2.5-1.5B-Instruct` to choose exactly one of `FINDINGS`, `IMPRESSION`, `REPORT_FACT`, or `OUTSIDE_REPORT`. Its output is passed to the unchanged closed-loop evidence Agent as a constrained plan. The prompt hash was recorded before generation; development is used only for answerability-threshold selection, calibration is used only for Platt scaling, and neither original test nor transfer test is used for tuning. All 1,872 planner outputs parsed successfully.

| Evaluation | V2.1 lexical planner | V2.2 semantic planner |
|---|---:|---:|
| Original wording macro F1 | 0.9583 | 0.7961 |
| Original wording false-answer rate | 0.0556 | 0.3843 |
| Transfer wording macro F1 | 0.6972 | 0.8706 |
| Transfer wording false-answer rate | 0.1806 | 0.2546 |
| Transfer wording evidence hit rate | 0.6944 | 0.9769 |

The semantic planner improves wording transfer substantially but is not a replacement for the lexical planner. On original wording, 119 of 144 `REPORT_FACT` questions are misrouted to `IMPRESSION`, and 83 unanswerable questions receive false answers. On transfer wording, the model handles paraphrases much better and recovers 211 answerable questions with only 5 retrieval misses, but selective safety remains weaker than the lexical system. Calibration fitted only on the calibration split reduces transfer false-answer rate to `0.0463`, at the cost of lowering macro F1 to `0.8455` and abstaining on `60.65%` of questions.

This is a useful negative and complementary result: lexical rules provide precision on known forms, while semantic planning provides linguistic coverage. A rule-first semantic-fallback cascade is a plausible next hypothesis, but it must be designed before and evaluated on a second reserved wording set; combining the two after observing these tests would otherwise tune to test behavior.

## V2.3 preregistered hybrid planner

The hybrid hypothesis was frozen and pushed in Git commit `47a2c1a` before any V2.3 planner generation or outcome evaluation. The policy keeps recognized lexical intents and the development-known report-fact frame; it calls the unchanged V2.2 semantic planner only for other unknown wording. A second deterministic reserved wording set contains 432 questions over the same 72 test cases and has fingerprint `d23ea907ec7da80c73f9b862976d59c4cd4cf2a3c99250b704ba359d6f0733e2`. No threshold, calibration model, prompt, or policy component was fitted on either wording-transfer set.

| Evaluation | Lexical Macro F1 | Hybrid Macro F1 | Lexical false-answer rate | Hybrid false-answer rate | Hybrid semantic-call rate |
|---|---:|---:|---:|---:|---:|
| Original wording | 0.9583 | 0.9583 | 0.0556 | 0.0556 | 0.0324 |
| Reserved wording set 1 | 0.6972 | 0.9139 | 0.1806 | 0.1574 | 1.0000 |
| Reserved wording set 2 | 0.7357 | 0.8204 | 0.2222 | 0.3194 | 0.8727 |

On the result-blind second set, the hybrid-minus-lexical Macro F1 difference is `+0.0847`; a paired 5,000-resample case bootstrap gives 95% CI `[+0.0515, +0.1186]`. The false-answer-rate difference is also positive at `+0.0972`, 95% CI `[+0.0370, +0.1528]`. Thus V2.3 improves wording robustness and preserves original performance, but it is not safety-dominant under distribution shift. Its second-set failures include 69 false answers to unanswerable questions, 25 retrieval misses, 7 missed answerable questions, and 6 wrong-section routes despite an evidence hit.

The frozen V2.1 Platt model also fails to transfer cleanly: for V2.3 hybrid on wording set 2, raw ECE is `0.1002`, while applying the original calibration model raises ECE to `0.1688` and false-answer rate from `0.3194` to `0.3380`. Calibration is therefore distribution-specific in this experiment. The V2.3 table uses the V2.1 frozen answerability threshold for every system; its wording-set-1 values should not be compared as a pure planner effect against V2.2, which selected a separate semantic-system threshold on development.

The new planner pack was profiled locally on an NVIDIA GeForce RTX 5070 Laptop GPU with 8,150.6 MiB, batch size 32, and eight maximum new tokens. Model and tokenizer loading took `2.813s`; generation for 432 planner prompts took `8.159s`; total process time was `14.696s`. Generation-only throughput was `52.95` prompts/second and throughput including loading was `29.40` prompts/second. PyTorch measured peak CUDA memory of `3,387.6 MiB` allocated and `3,654.0 MiB` reserved. This is one cache- and machine-dependent run, not a deployment latency claim.

The hybrid policy's semantic-planner call rate is `3.24%` on original wording, `100%` on reserved wording set 1, and `87.27%` on set 2. Corresponding mean evidence-retrieval calls are `1.032`, `1.046`, and `0.868`; mean returned chunks are `2.245`, `2.178`, and `1.956`. Cost therefore depends strongly on language distribution: the lexical gate nearly eliminates LLM planning on familiar forms but saves only `12.73%` of planner calls on the second transfer set.

## Untouched locked-system replication

A second cohort contains 300 cases and 900 questions. These cases exclude all 1,080 cases used by V1, V2, V2 confirmation, and v2.1. The experiment reuses the original question construction while searching the full 3,851-case corpus.

No parameter is selected on this cohort. The script reads the previously frozen hybrid alpha (`0.30`), MedCPT Cross-Encoder candidate depth (`3`), adaptive retrieval policy, direct prompt, Qwen2.5-1.5B generator, and BioLinkBERT-MedNLI verifier. SHA-256 hashes of the source configuration files are embedded in the result summary.

### Replication results

| Metric | Original held-out test, n=108 | Untouched replication, n=900 |
|---|---:|---:|
| Adaptive Top-1 | 0.2870 | 0.2944 |
| Retrieval coverage | 0.9259 | 0.9656 |
| Draft Token-F1 | 0.1991 | 0.2181 |
| Verified Token-F1 | 0.2063 | 0.2282 |
| Evidence support rate | 0.7850 | 0.7581 |
| Final abstention rate | 0.0833 | 0.0800 |

The replication adaptive Top-1 Wilson 95% CI is `[0.2656, 0.3250]`; the absolute difference from the original held-out estimate is `+0.0074`. The replication verified Token-F1 case-grouped bootstrap 95% CI is `[0.2138, 0.2426]`, using 5,000 resamples over 300 case IDs.

## Research position and value

Medical RAG remains an active problem: the MedRAG/MIRAGE study found that retrieval can improve medical QA but that performance depends strongly on corpus, retriever, and context placement ([Xiong et al., 2024](https://aclanthology.org/2024.findings-acl.372/)). RadQA demonstrates the harder target: physician-authored radiology questions, naturally unanswerable cases, and expert answer spans ([Soni et al., 2022](https://aclanthology.org/2022.lrec-1.672/)). Unanswerability is now treated as a first-class RAG evaluation problem rather than an edge case ([Peng et al., 2025](https://aclanthology.org/2025.acl-long.415/)). Agentic retrieval has also already been studied for radiology QA, so “using an Agent” alone is not a novelty claim ([Wind et al., 2025](https://arxiv.org/abs/2508.00743)).

The defensible contribution of this project is narrower: a locally runnable, case-scoped, auditable evidence Agent that exposes its action trace; evaluates answerability, calibration, and retrieval budget; documents a negative wording-transfer result; and reproduces the locked earlier system on 300 disjoint cases. This is meaningful for an AI master's project because it combines system design, controlled experimentation, risk analysis, reproducibility, and honest boundary testing. It is not yet evidence of clinical effectiveness or a state-of-the-art general radiology QA model.

## Reproduce

With the local OpenI cases, MedCPT index, and cached model weights available:

```powershell
python scripts/evaluate_case_scoped_hard_v21.py
python scripts/evaluate_v21_template_transfer.py
python scripts/build_v22_semantic_planner_pack.py
python scripts/evaluate_v22_semantic_planner.py
python scripts/build_v23_hybrid_preregistration.py
python scripts/run_hf_generation.py `
  --prompt-pack data/processed/prompt_packs/v23_hybrid_transfer2_planner.jsonl `
  --output experiments/post_submission_v23/planner_generations_qwen15.jsonl `
  --metrics-output experiments/post_submission_v23/generation_runtime_profile.json `
  --model Qwen/Qwen2.5-1.5B-Instruct --device cuda `
  --max-new-tokens 8 --batch-size 32 --temperature 0 --local-files-only
python scripts/evaluate_v23_hybrid_planner.py
python scripts/run_locked_replication_cohort.py
python scripts/run_hf_generation.py `
  --prompt-pack experiments/locked_replication/direct_prompt_pack.jsonl `
  --output experiments/locked_replication/generations_qwen15.jsonl `
  --model Qwen/Qwen2.5-1.5B-Instruct --device cuda `
  --max-new-tokens 256 --batch-size 16 --local-files-only --resume
python scripts/evaluate_final_optimized_test.py `
  --generations experiments/locked_replication/generations_qwen15.jsonl `
  --device cuda --batch-size 128 `
  --system-name locked_replication_adaptive_direct_semantic_agent `
  --output-dir experiments/locked_replication/semantic_final
python scripts/finalize_locked_replication.py
```

The large generation and per-question JSONL files remain local by repository policy. Versioned cohort manifests and summaries contain fingerprints, configuration hashes, aggregate metrics, and confidence intervals.
