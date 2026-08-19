# V5 Technical Freeze

## Status

V5 is the final post-submission technical extension. Its configuration was prospectively specified and frozen locally before execution, but it was not formally preregistered or externally timestamped before outcomes were observed. The 120-case confirmation cohort is fresh and disjoint from all prior project cohorts, while remaining part of the same OpenI/IU-Xray source.

## Supported Claims

- Indication text is a strong retrieval shortcut in this controlled benchmark and must be reported explicitly.
- Correctly aligned image reranking improves paired-report retrieval over the indication-plus-question BM25 baseline on the fresh confirmation cohort.
- None of 100 fixed-point-free shuffled-image controls reached the correct-alignment MRR or extractive proxy Token-F1; the plus-one Monte Carlo value is `p=0.0099`.
- Under the same non-oracle Qwen generation and automated semantic checking path, multimodal retrieval improves final Token-F1 by `0.0302`, with case-bootstrap 95% CI `[0.0101, 0.0511]`.
- The multimodal system has a lower automated evidence-support rate, exposing a performance-grounding trade-off.

## Unsupported Claims

- Formal preregistration or independent external-dataset validation.
- Image-based diagnosis, clinical causality, clinical effectiveness, or deployment safety.
- General natural-language planning beyond the three report-derived question templates.
- Human-validated faithfulness, correctness, or safety of the semantic verifier.

## Frozen Artifacts

The complete local reproduction sequence is:

```powershell
& ".\.venv\Scripts\python.exe" scripts\build_multimodal_v5_cohort.py
& ".\.venv\Scripts\python.exe" scripts\run_multimodal_v5_retrieval.py --split confirmation --device cuda
& ".\.venv\Scripts\python.exe" scripts\build_multimodal_v5_prompt_packs.py --split confirmation
& ".\.venv\Scripts\python.exe" scripts\run_hf_generation.py `
  --prompt-pack data\processed\prompt_packs\multimodal_v5\confirmation_v5_report_only.jsonl `
  --output experiments\post_submission_v5\confirmation_report_only_generations.jsonl `
  --metrics-output experiments\post_submission_v5\confirmation_report_only_runtime.json `
  --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --batch-size 16 `
  --max-new-tokens 256 --temperature 0 --local-files-only
& ".\.venv\Scripts\python.exe" scripts\run_hf_generation.py `
  --prompt-pack data\processed\prompt_packs\multimodal_v5\confirmation_v5_multimodal.jsonl `
  --output experiments\post_submission_v5\confirmation_multimodal_generations.jsonl `
  --metrics-output experiments\post_submission_v5\confirmation_multimodal_runtime.json `
  --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --batch-size 16 `
  --max-new-tokens 256 --temperature 0 --local-files-only
& ".\.venv\Scripts\python.exe" scripts\evaluate_final_optimized_test.py `
  --generations experiments\post_submission_v5\confirmation_report_only_generations.jsonl `
  --system-name v5_report_only --output-dir experiments\post_submission_v5\qa_report_only `
  --device cuda --batch-size 128
& ".\.venv\Scripts\python.exe" scripts\evaluate_final_optimized_test.py `
  --generations experiments\post_submission_v5\confirmation_multimodal_generations.jsonl `
  --system-name v5_multimodal --output-dir experiments\post_submission_v5\qa_multimodal `
  --device cuda --batch-size 128
& ".\.venv\Scripts\python.exe" scripts\analyze_multimodal_v5_statistics.py
& ".\.venv\Scripts\python.exe" scripts\build_v5_artifact_manifest.py
```

The BioViL-T cache, local OpenI case data, image archive, Qwen weights, and semantic-checker weights must already be available. Runtime values are machine- and cache-dependent.

Run the following command after any intentional artifact update:

```powershell
& ".\.venv\Scripts\python.exe" scripts\build_v5_artifact_manifest.py
```

The resulting `experiments/post_submission_v5/artifact_manifest.json` records the cohort fingerprint and LF-normalized SHA-256 values for the configuration, aggregate results, implementation, and tests. Text line endings are normalized only for hashing so the checks remain portable across Git platforms. Large per-question rows, generations, prompt packs, and image pixels remain local by repository policy.
