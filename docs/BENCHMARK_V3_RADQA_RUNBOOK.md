# Benchmark V3 RadQA Runbook

## Status

The V3 framework is implemented. Real results are not available because credentialed RadQA files are not present locally. Synthetic schema-fixture output is for software verification only and must not appear in thesis result tables.

## Required Official Files

Place legally obtained files at:

```text
data/raw/radqa/train.json
data/raw/radqa/dev.json
data/raw/radqa/test.json
```

Access is controlled by [PhysioNet](https://physionet.org/content/radqa/). Only the credentialed user who signs the data-use agreement may access the files; do not commit, redistribute, or ask another person to transfer them.

As verified on 16 August 2026, access requires a PhysioNet credentialed account, the specified CITI human-research training, and acceptance of the PhysioNet Credentialed Health Data Use Agreement. After approval, download the official files while signed in and place them at the paths above. The project cannot automate or bypass these identity-bound requirements.

## Build and Audit

```powershell
& ".\.venv\Scripts\python.exe" scripts/build_radqa_benchmark_v3.py
```

The builder:

1. parses the official SQuAD-style files;
2. derives patient, report, and section metadata;
3. maps answer character spans to sentence-level qrels;
4. preserves unanswerable questions with empty qrels;
5. enforces patient-disjoint train/dev/test splits;
6. creates a content SHA-256 fingerprint;
7. reports whether candidate pools collapse to qrels.

## Retrieval and Agent Evaluation

```powershell
& ".\.venv\Scripts\python.exe" scripts/evaluate_radqa_agent_v3.py
```

The evaluation compares report-scoped, patient-scoped, and global BM25 on answerable natural questions. It selects the evidence-sufficiency threshold on development only, evaluates the fixed threshold on test, and emits a test prompt pack containing `ANSWER_FROM_EVIDENCE` or `ABSTAIN_LOW_EVIDENCE` actions.

## Generation

```powershell
& ".\.venv\Scripts\python.exe" scripts/run_hf_generation.py `
  --device cuda --local-files-only `
  --model Qwen/Qwen2.5-1.5B-Instruct `
  --prompt-pack experiments/benchmark_v3_radqa/radqa_v3_test_agent_prompt_pack.jsonl `
  --output experiments/benchmark_v3_radqa/radqa_v3_test_qwen15.jsonl
```

Generation results may enter the thesis only after multi-reference answer evaluation, unanswerable accuracy, retrieval ablations, and frozen test reporting are complete.

Evaluate the completed generations with:

```powershell
& ".\.venv\Scripts\python.exe" scripts/evaluate_radqa_generation_v3.py
```

The evaluator uses the best score across available reference spans and reports exact match, Token-F1, unanswerable accuracy, false-answer rate, and overall abstention. It fails if any frozen prompt is missing a generation.

## Software-Only Fixture

```powershell
& ".\.venv\Scripts\python.exe" scripts/build_radqa_benchmark_v3.py `
  --input-dir data/sample/radqa_synthetic `
  --output outputs/radqa_synthetic_benchmark.json
& ".\.venv\Scripts\python.exe" scripts/evaluate_radqa_agent_v3.py `
  --benchmark outputs/radqa_synthetic_benchmark.json `
  --output-dir outputs/radqa_synthetic_evaluation
```

Every file in this fixture is synthetic and proves only that the code path executes.
