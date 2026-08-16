# Agentic RAG Design

## Thesis Motivation

The project should not stop at a simple "retrieve then answer" chatbot. A plain RAG prototype may look too close to a normal course assignment, especially if the evaluation only reports answer token overlap. The stronger thesis direction is an evidence-checking agentic RAG system for radiology report-grounded question answering.

The agent is not intended to autonomously diagnose chest X-rays. The expert-written report remains the primary evidence source. The linked X-ray images are preserved as case context and for qualitative presentation.

## Current Agent Loop

The first implemented agent is deliberately transparent and reproducible:

1. Query planner: identifies which report field is needed, such as findings or impression.
2. Hybrid retriever: retrieves complete OpenI cases using BM25 + MedCPT score fusion.
3. Draft answer builder: creates an extractive draft from the top retrieved case.
4. Evidence checker: checks whether each answer sentence is supported by retrieved report evidence.
5. Revision or abstention: removes unsupported sentences or abstains when evidence is insufficient.

Implementation files:

```text
src/medical_rag/agentic/planner.py
src/medical_rag/agentic/evidence_checker.py
src/medical_rag/agentic/agent.py
src/medical_rag/retrieval/hybrid_retriever.py
scripts/run_agentic_rag_baseline.py
scripts/run_agentic_evidence_check_on_generations.py
```

## Current Result

Experiment:

```text
experiments/agentic_hybrid_a050_qa_seed_clean_answers.json
```

Dataset:

```text
data/processed/openi_case_qa_seed_clean.json
```

Configuration:

- Retriever: hybrid BM25 + MedCPT
- Alpha: 0.50
- Top-k evidence cases: 5
- Questions: 360

Metrics:

| Metric | Score |
|---|---:|
| Answer token-F1 | 0.412 |
| Top-1 case accuracy | 0.242 |
| Retrieved case hit rate | 0.422 |
| Average evidence support rate | 0.992 |
| Revision rate | 0.008 |
| Abstention rate | 0.008 |
| Non-empty answer rate | 1.000 |

## Interpretation

The current agent has a higher answer token-F1 than the clean-seed extractive BM25 baseline and similar top-1 case accuracy to the hybrid retrieval run. Its evidence support rate is very high because the current draft answer is extractive rather than fully generated. This is expected and should be framed carefully.

The useful contribution is the architecture and evaluation hook: every answer is passed through an evidence-checking step, and the output stores the plan, retrieved cases, draft answer, final answer, evidence support score, and revision decision. This gives the thesis a clear path beyond a basic RAG demo.

## Next Research Step

The next important experiment is to replace the extractive draft answer with an LLM-generated draft while keeping the same evidence checker. That will make the agentic component more meaningful:

- LLM-only answer: no retrieval, likely fluent but weakly grounded.
- Basic RAG answer: retrieval-conditioned generation without post-checking.
- Agentic RAG answer: retrieval-conditioned generation followed by evidence checking and revision.

This comparison directly supports a thesis claim about hallucination control and evidence-grounded medical QA.

The first Qwen2.5-0.5B pilot exposed an important design issue: checking support against all top-k retrieved cases can miss cross-case contamination, because the model may combine findings from several different retrieved cases. The stricter generated-answer checker therefore defaults to top-1 evidence scope. This better matches the case-grounded QA setting, where an answer should be supported by the selected case rather than by any case in the retrieval pool.

After LLM generation outputs are available, run:

```powershell
.venv\Scripts\python.exe scripts\run_agentic_evidence_check_on_generations.py --generations experiments\generations_case_rag_hybrid_qwen.jsonl --cases data\processed\openi_cases.jsonl --output experiments\generations_case_rag_hybrid_qwen_agentic_top1.jsonl --metrics-output experiments\generations_case_rag_hybrid_qwen_agentic_top1_eval.json --evidence-scope top1
```

For manual checking, the current 30-item sample sheet is:

```text
experiments/manual_annotation_agentic_sample.csv
```
