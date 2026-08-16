# Post-submission experiments

These experiments were run after the P2 submission was frozen. They strengthen the software and provide additional research evidence, but they are not claims made by the submitted manuscript or defence deck.

## V2.1 hard evidence benchmark

The v2.1 benchmark contains 240 OpenI cases and 1,440 questions. Its cases exclude the earlier V1 seed, V2 main benchmark, and V2 confirmation benchmark. Each case contributes three answerable questions and three unanswerable questions. The set includes paraphrased section requests, report-derived fact probes, out-of-scope requests, near-domain absent-finding requests, and non-target report sections as distractors.

The Agent receives only the case scope and natural-language question. The recorded `expected_intent` is retained for scoring and is not passed to `ClosedLoopEvidenceAgent.run`. Both systems may return at most six unique chunks per question. The baseline uses one report-wide BM25 call; the Agent uses up to two calls with a fixed plan, evidence assessment, optional rewrite, and retry. Answerability thresholds are selected on development only and then frozen for calibration and test.

### Independent test results

| Metric | Fixed report BM25 | Closed-loop Agent V2 |
|---|---:|---:|
| Answerability macro F1 | 0.6934 | 0.9583 |
| False-answer rate | 0.1991 | 0.0556 |
| Evidence retrieval hit rate | 0.8426 | 0.9954 |
| End-to-end action accuracy | 0.6759 | 0.9583 |
| Answerable Token-F1 | 0.2845 | 0.6893 |
| AURC, lower is better | 0.2829 | 0.0136 |
| ECE, lower is better | 0.2681 | 0.2433 |
| Mean retrieved chunks | 5.9861 | 2.2454 |
| Mean retrieval calls | 1.0000 | 1.0324 |

The benchmark is deterministic and automatically labelled from report structure. It tests report-grounded routing and selective behavior; it is not a substitute for expert clinical evaluation. Because the task and policy were co-designed, the separate locked replication below remains necessary.

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

## Reproduce

With the local OpenI cases, MedCPT index, and cached model weights available:

```powershell
python scripts/evaluate_case_scoped_hard_v21.py
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
