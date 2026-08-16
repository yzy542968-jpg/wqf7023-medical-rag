# Optimized Methods and Reproducibility

## Environment

The final experiments were run locally on Windows with CUDA. From the repository root:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
$env:PYTHONPATH = "$(Resolve-Path src);$(Get-Location)"
$env:HF_HOME = "$(Resolve-Path .hf_cache)"
```

The frozen package constraint `transformers>=4.45,<5` is required because the current RadGraph package is incompatible with Transformers 5.x.

## Frozen Data Split

```powershell
python scripts/build_case_level_splits.py
```

Primary artifact:

```text
data/splits/openi_qa_grouped_case_seed7023.json
```

The manifest stores the exact case IDs, qids, class counts, random seed, and dataset fingerprint. Never tune on `test.case_ids` or `test.qids`.

## Retrieval Selection

```powershell
python scripts/sweep_hybrid_alpha_grouped_split.py
python scripts/evaluate_medcpt_reranker_grouped_split.py --device cuda
python scripts/select_adaptive_retrieval_policy.py
```

Selection order matters:

1. Select hybrid alpha by development MRR.
2. Select reranker candidate depth by development MRR.
3. Select adaptive confidence thresholds by development selective accuracy subject to minimum coverage.
4. Evaluate each locked choice on the held-out test once.

The unconditional reranker is retained as a negative ablation because its development gain did not transfer to the held-out split.

## Semantic Evidence Checker

```powershell
python scripts/run_semantic_evidence_ablation.py --device cuda --batch-size 64
```

The checker aligns each generated sentence to candidate evidence sentences. It combines lexical overlap with Medical NLI entailment, while explicit polarity conflicts and NLI contradictions override apparent support.

The selected model is `cnut1648/biolinkbert-mednli`. Label orientation was verified before use: entailment, neutral, and contradiction are mapped from the model configuration rather than assumed by index.

## Prompt Selection

Build prompt packs from the adaptive development decisions:

```powershell
python scripts/build_optimized_prompt_pack.py `
  --split-name development `
  --decisions experiments/final_optimized/adaptive_retrieval/adaptive_policy_development_decisions.jsonl
```

Generate all development prompts:

```powershell
python scripts/run_hf_generation.py --device cuda --local-files-only `
  --prompt-pack data/processed/prompt_packs/final_optimized/adaptive_development_direct.jsonl `
  --output experiments/final_optimized/generations/adaptive_development_direct_qwen15.jsonl

python scripts/run_hf_generation.py --device cuda --local-files-only `
  --prompt-pack data/processed/prompt_packs/final_optimized/adaptive_development_evidence_guided.jsonl `
  --output experiments/final_optimized/generations/adaptive_development_evidence_guided_qwen15.jsonl

python scripts/run_hf_generation.py --device cuda --local-files-only `
  --prompt-pack data/processed/prompt_packs/final_optimized/adaptive_development_structured_case_aware.jsonl `
  --output experiments/final_optimized/generations/adaptive_development_structured_case_aware_qwen15.jsonl
```

Evaluate and lock the prompt:

```powershell
python scripts/evaluate_optimized_prompt_ablation.py --device cuda --batch-size 64
```

## Final Held-Out Run

```powershell
python scripts/build_optimized_prompt_pack.py `
  --split-name test `
  --decisions experiments/final_optimized/adaptive_retrieval/adaptive_policy_test_decisions.jsonl

python scripts/run_hf_generation.py --device cuda --local-files-only `
  --prompt-pack data/processed/prompt_packs/final_optimized/adaptive_test_direct.jsonl `
  --output experiments/final_optimized/generations/adaptive_test_direct_qwen15.jsonl

python scripts/evaluate_final_optimized_test.py --device cuda --batch-size 64
```

Do not regenerate alternative held-out prompts after inspecting test results. That would turn the test split into another development split.

## Statistical and Clinical Metrics

```powershell
python scripts/run_grouped_statistical_analysis.py
python scripts/run_radgraph_held_out_evaluation.py
python scripts/analyze_cross_case_contamination.py --device cuda --batch-size 64
python scripts/run_verifier_polarity_stress_test.py --device cuda --batch-size 64
python scripts/run_research_validity_audit.py
```

Bootstrap resampling is grouped by case because the three questions belonging to the same report are not independent observations. RadGraph adds clinical entity and relation overlap beyond generic token overlap. The contamination analysis tests whether a report-level top-k answer is supported by the first patient or only by another retrieved patient.

The statistical output includes paired case-level randomization p-values and Holm family-wise error correction across the 10 exploratory comparisons. Report the adjusted value unless a comparison was explicitly declared as the single primary hypothesis before analysis.

## Oracle Retrieval Diagnostic

The oracle run is diagnostic only and must never be described as a deployable system:

```powershell
python scripts/build_oracle_test_prompt_pack.py
python scripts/run_hf_generation.py --device cuda --local-files-only `
  --prompt-pack data/processed/prompt_packs/final_optimized/oracle_test_direct.jsonl `
  --output experiments/final_optimized/generations/oracle_test_direct_qwen15.jsonl
python scripts/evaluate_final_optimized_test.py --device cuda --batch-size 64 `
  --generations experiments/final_optimized/generations/oracle_test_direct_qwen15.jsonl `
  --output-dir experiments/final_optimized/oracle_test `
  --system-name oracle_direct_semantic_agent
```

It quantifies the maximum headroom available from retrieval improvement while holding generation and verification fixed.

## Human Evaluation Package

```powershell
python scripts/build_blinded_held_out_human_evaluation.py
```

This creates a 36-case sheet with one balanced question per held-out case and four shuffled system answers per question. Keep the key hidden from reviewers until all ratings are frozen.

After the completed ratings file is frozen:

```powershell
python scripts/analyze_blinded_human_evaluation.py
```

The analyzer validates every required cell, merges the hidden key, summarizes all systems, and runs paired bootstrap comparisons against the final system.

## Verification

```powershell
python -m pytest -q
python -m compileall -q app.py scripts src
& ".\.venv\Scripts\python.exe" -m streamlit run app.py
```

Acceptance criterion: the complete discovered test suite passes with no failures. Record the exact count in the final submission audit rather than maintaining it manually here.

## Benchmark V2 Reproduction

Build the 600-case benchmark while excluding every V1 case:

```powershell
python scripts/build_case_scoped_benchmark_v2.py
python scripts/evaluate_case_scoped_retrieval_v2.py --split development
python scripts/evaluate_case_scoped_retrieval_v2.py --split calibration
python scripts/evaluate_case_scoped_retrieval_v2.py --split test
```

Lock top-k only on calibration and build prompt packs:

```powershell
python scripts/select_case_scoped_top_k_v2.py
python scripts/build_case_scoped_prompt_pack_v2.py --split calibration `
  --output data/processed/prompt_packs/benchmark_v2/calibration_case_scoped_routed.jsonl
python scripts/build_case_scoped_prompt_pack_v2.py --split test
```

Generate calibration answers, calibrate the verifier action, then evaluate the unchanged test system:

```powershell
python scripts/run_hf_generation.py --device cuda --local-files-only `
  --model Qwen/Qwen2.5-1.5B-Instruct `
  --prompt-pack data/processed/prompt_packs/benchmark_v2/calibration_case_scoped_routed.jsonl `
  --output experiments/benchmark_v2/generations/calibration_case_scoped_routed_qwen15.jsonl
python scripts/calibrate_case_scoped_verifier_v2.py --device cuda
python scripts/evaluate_case_scoped_generation_v2.py --device cuda `
  --semantic-config experiments/benchmark_v2/calibration/semantic_verifier/semantic_agent_selection.json `
  --output-dir experiments/benchmark_v2/final_test_evaluation
```

The once-only confirmation cohort excludes all 120 V1 and 600 main-V2 cases:

```powershell
python scripts/build_case_scoped_confirmation_v2.py
python scripts/build_case_scoped_prompt_pack_v2.py `
  --benchmark data/processed/openi_case_scoped_confirmation_v2.json `
  --split confirmation `
  --output data/processed/prompt_packs/benchmark_v2/confirmation_case_scoped_routed.jsonl
python scripts/evaluate_case_scoped_retrieval_v2.py `
  --benchmark data/processed/openi_case_scoped_confirmation_v2.json `
  --split confirmation --top-k 6 `
  --output-dir experiments/benchmark_v2/confirmation_retrieval
python scripts/run_benchmark_v2_validity_audit.py
```

The V2 verifier selection includes sentence filtering, contradiction-only filtering, and advisory audit. Calibration selected advisory audit because every automatic rewriting policy reduced answer Token-F1. NLI scores remain risk indicators and are not presented as clinical correctness labels.

The validity audit must be reported with the headline V2 metrics. It records that routed candidates equal qrels by construction and that the extractive retrieved-context baseline outperforms Qwen. The diagnostic V2 test was inspected before verifier-action calibration; the disjoint confirmation cohort is therefore the primary final automated cohort.

## Reproducibility Boundaries

- The OpenI source dataset and cached model weights may not be committed due to size and distribution constraints.
- Generated outputs, split manifests, selected configurations, summary tables, and source code must be versioned.
- Random seeds are fixed at 7023 unless a script explicitly reports another seed.
- V1 final numbers must come from `experiments/final_optimized/`; V2 final numbers must come from `experiments/benchmark_v2/confirmation_evaluation/` together with `experiments/benchmark_v2/validity_audit/benchmark_v2_validity_audit.json`.
