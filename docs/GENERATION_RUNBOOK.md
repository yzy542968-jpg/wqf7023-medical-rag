# Generation Runbook

This runbook describes how to run the planned LLM and RAG generation experiments once a local or cloud LLM environment is available.

## Prompt Packs

The current prompt packs are generated from the clean OpenI QA seed:

```text
data/processed/prompt_packs/llm_only_direct_clean.jsonl
data/processed/prompt_packs/report_rag_bm25_evidence_guided_top5_clean.jsonl
data/processed/prompt_packs/case_rag_bm25_structured_top5_clean.jsonl
data/processed/prompt_packs/case_rag_bm25_top1_structured_top5_clean.jsonl
data/processed/prompt_packs/case_rag_hybrid_a050_structured_top5_clean.jsonl
data/processed/prompt_packs/case_rag_hybrid_a050_top1_structured_top5_clean.jsonl
```

Each JSONL row contains:

- `qid`
- `case_id`
- `question`
- `reference_answer`
- `system`
- `prompt_mode`
- `retrieved_case_ids`
- `retrieved_context`
- `prompt`

## Planned Systems

| Prompt pack | System role |
|---|---|
| `llm_only_direct_clean.jsonl` | LLM-only baseline |
| `report_rag_bm25_evidence_guided_top5_clean.jsonl` | Report-based RAG baseline |
| `case_rag_bm25_structured_top5_clean.jsonl` | Case-based RAG baseline |
| `case_rag_bm25_top1_structured_top5_clean.jsonl` | BM25 case-grounded top-1 evidence prompt |
| `case_rag_hybrid_a050_structured_top5_clean.jsonl` | Hybrid case-based RAG baseline and agentic RAG draft source |
| `case_rag_hybrid_a050_top1_structured_top5_clean.jsonl` | Preferred case-grounded hybrid RAG prompt that gives the generator only top-1 case evidence |

## Generation Model

Planned model:

```text
Qwen2.5-7B-Instruct
```

Current local pilot models:

```text
Qwen2.5-0.5B-Instruct
Qwen2.5-1.5B-Instruct
```

If local compute is not sufficient for 7B, use a smaller instruction model for early testing and keep Qwen2.5-7B-Instruct as the target final model with quantization/offload or cloud compute.

## Run Hugging Face Generation

Once `torch` and `transformers` are installed, run a small smoke test first:

```powershell
python scripts\run_hf_generation.py --prompt-pack data\processed\prompt_packs\case_rag_bm25_structured_top5_clean.jsonl --output experiments\generations_case_rag_qwen_smoke.jsonl --model Qwen/Qwen2.5-7B-Instruct --max-items 5 --device cpu
```

Then evaluate the generated answers:

```powershell
python scripts\evaluate_generated_answers.py --generations experiments\generations_case_rag_qwen_smoke.jsonl --output experiments\generations_case_rag_qwen_smoke_eval.json
```

For full experiments, remove `--max-items 5` and run the four prompt packs:

```powershell
python scripts\run_hf_generation.py --prompt-pack data\processed\prompt_packs\llm_only_direct_clean.jsonl --output experiments\generations_llm_only_qwen.jsonl --model Qwen/Qwen2.5-7B-Instruct
python scripts\run_hf_generation.py --prompt-pack data\processed\prompt_packs\report_rag_bm25_evidence_guided_top5_clean.jsonl --output experiments\generations_report_rag_qwen.jsonl --model Qwen/Qwen2.5-7B-Instruct
python scripts\run_hf_generation.py --prompt-pack data\processed\prompt_packs\case_rag_bm25_structured_top5_clean.jsonl --output experiments\generations_case_rag_qwen.jsonl --model Qwen/Qwen2.5-7B-Instruct
python scripts\run_hf_generation.py --prompt-pack data\processed\prompt_packs\case_rag_hybrid_a050_structured_top5_clean.jsonl --output experiments\generations_case_rag_hybrid_qwen.jsonl --model Qwen/Qwen2.5-7B-Instruct
python scripts\run_hf_generation.py --prompt-pack data\processed\prompt_packs\case_rag_hybrid_a050_top1_structured_top5_clean.jsonl --output experiments\generations_case_rag_hybrid_top1_qwen.jsonl --model Qwen/Qwen2.5-7B-Instruct
```

If a long local generation run is interrupted, rerun the same command with:

```powershell
--resume
```

The script will append missing generations and skip prompt records whose `qid` already exists in the output JSONL.

## Run Agentic Evidence Checking

After generating RAG answers, run the evidence-checking agent over the generated drafts:

```powershell
python scripts\run_agentic_evidence_check_on_generations.py --generations experiments\generations_case_rag_hybrid_qwen.jsonl --cases data\processed\openi_cases.jsonl --output experiments\generations_case_rag_hybrid_qwen_agentic_top1.jsonl --metrics-output experiments\generations_case_rag_hybrid_qwen_agentic_top1_eval.json --evidence-scope top1
```

This produces a revised answer file plus metrics for draft token-F1, final token-F1, evidence support, revision rate, abstention rate, and unsupported sentence rate.

For case-grounded QA, use `--evidence-scope top1` unless the experiment explicitly studies all-top-k support. Top-1 checking is stricter and avoids cross-case contamination, where a generated answer copies findings from multiple retrieved cases.

## Output Schema

Generated outputs should be saved as JSONL with at least:

```json
{
  "qid": "CXR1027_impression",
  "system": "case_rag",
  "model": "Qwen2.5-7B-Instruct",
  "prompt_mode": "structured",
  "answer": "...",
  "reference_answer": "...",
  "retrieved_case_ids": ["CXR1027", "CXR1311"],
  "metadata": {}
}
```

## Evaluation After Generation

After generation, evaluate:

1. Token-F1 against the reference report answer.
2. Retrieval correctness: whether the original case is retrieved.
3. Manual evidence support on selected examples.
4. Hallucination tendency using the rubric in `docs/EVALUATION_PLAN.md`.
5. Evidence-checking impact: whether agentic revision reduces unsupported claims without removing too many correct answers.
