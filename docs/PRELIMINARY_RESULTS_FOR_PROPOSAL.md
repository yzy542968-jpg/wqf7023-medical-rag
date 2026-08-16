# Preliminary Feasibility Results

## Dataset Preparation

The initial feasibility experiment uses the IU X-Ray / OpenI chest X-ray collection. The project currently uses 3,851 radiology reports and 7,466 image-projection records. Each report was converted into a case-level retrieval unit containing the case identifier, indication, findings, impression, problem labels, and the associated X-ray image filenames. This structure supports the proposed research design, where a retrieved unit represents a complete radiology case rather than an isolated report fragment.

The normalized dataset is stored as:

```text
data/processed/openi_cases.jsonl
```

Each JSONL record contains:

- `case_id`
- `indication`
- `findings`
- `impression`
- `report_text`
- `problems`
- associated image filenames and projections

## Real Image Subset

A small image subset was downloaded for feasibility demonstration. It contains the top five retrieved cases for the query:

```text
right lower lobe pneumonia with cough and fever
```

The subset contains five real OpenI cases and ten X-ray images, including frontal and lateral projections where available. The subset is used to demonstrate the case-level retrieval idea without requiring the full image dataset during the early proposal stage.

The contact sheet is stored as:

```text
experiments/openi_pneumonia_subset_contact_sheet.jpg
```

## Retrieval Baselines

Two transparent lexical retrieval baselines have been implemented:

1. TF-IDF retrieval over case-level report text.
2. BM25 retrieval over case-level report text.

These baselines are intentionally lightweight and reproducible. They provide an initial comparison point before the planned dense biomedical retriever is implemented using MedCPT embeddings and ChromaDB.

## Keyword-Based Retrieval Evaluation

For the preliminary evaluation, keyword qrels were generated from the OpenI `Problems` field. Eight query categories were used:

| Query ID | Relevant cases |
|---|---:|
| q_normal | 1379 |
| q_cardiomegaly | 345 |
| q_pleural_effusion | 149 |
| q_atelectasis | 315 |
| q_pneumonia | 40 |
| q_pneumothorax | 26 |
| q_pulmonary_edema | 42 |
| q_nodule | 106 |

The retrieval metrics are Hit@k, Recall@k, and Mean Reciprocal Rank (MRR).

| Retriever | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.500 | 0.875 | 0.875 | 0.875 | 0.875 | 0.667 |
| BM25 | 0.500 | 0.875 | 0.875 | 1.000 | 1.000 | 0.688 |

## Example Retrieval Result

For the query:

```text
right lower lobe pneumonia with cough and fever
```

The TF-IDF baseline retrieved `CXR3078` as the top case. The report states:

- Findings: `There is a right lower lobe pneumonia.`
- Impression: `Right lower lobe pneumonia. Consider followup radiograph to document resolution.`

This result shows that the case-level retrieval pipeline can identify clinically relevant radiology cases and preserve their associated image links.

## Interpretation

The preliminary results show that the proposed data pipeline is feasible. Real OpenI reports can be normalized into case-level retrieval units, and simple lexical retrieval baselines can retrieve relevant cases for disease-oriented queries. BM25 performs slightly better than TF-IDF in this early evaluation, especially at Hit@10 and Hit@20.

However, lexical retrieval depends heavily on exact word overlap. It may fail when a query and report use different medical expressions for related findings. The next implementation stage will therefore use biomedical dense retrieval with MedCPT embeddings and ChromaDB. This will test whether semantic retrieval can improve case retrieval beyond lexical matching.

## Case-Grounded QA Seed

In addition to keyword qrels, a deterministic case-grounded QA seed set was created from 120 OpenI reports, producing 360 questions. Each question is linked to its source case and reference report answer. This seed set will support later comparison between LLM-only answering, report-based RAG, and case-based RAG.

Initial retrieval over this QA seed is more difficult than keyword retrieval. BM25 achieves Hit@20 of 0.344 and MRR of 0.264, while TF-IDF achieves Hit@20 of 0.342 and MRR of 0.232. These lower results are expected because the questions are more case-specific and often contain indication-level information that may not uniquely identify one report. This supports the need for improved retrieval and prompting strategies.

A cleaner QA seed was then created by prioritizing non-normal problem-labeled cases and revising summary questions to include problem labels. On this clean seed, BM25 improves to Hit@20 of 0.486 and MRR of 0.297, while TF-IDF reaches Hit@20 of 0.400 and MRR of 0.220. The clean QA seed will be used as the preferred input for the first LLM-based generation experiments.

The dense MedCPT retriever was also tested on the clean QA seed. Standalone MedCPT achieved Hit@20 of 0.406 and MRR of 0.172, which is lower than BM25 in this dataset. However, a hybrid retrieval method combining normalized BM25 and MedCPT scores improved retrieval performance. With alpha = 0.50, the hybrid retriever achieved Hit@20 of 0.553 and MRR of 0.324. This suggests that lexical and dense biomedical retrieval signals are complementary for this task.

An extractive report-RAG baseline was also tested by returning the findings or impression from the top retrieved report. On the QA seed, extractive TF-IDF RAG achieved token-F1 of 0.488, while extractive BM25 RAG achieved token-F1 of 0.341. These values are preliminary and should not be interpreted as final medical QA quality because radiology reports often contain repeated phrases. The final evaluation will therefore combine answer metrics with retrieval accuracy and manual evidence-support checking.

An initial evidence-checking agentic RAG baseline was then implemented on the clean QA seed using hybrid BM25 + MedCPT retrieval with alpha = 0.50. The agent performs query planning, retrieves complete image-report cases, drafts an extractive answer, checks whether answer sentences are supported by retrieved report evidence, and revises or abstains when evidence support is insufficient. This agentic baseline achieved answer token-F1 of 0.412, top-1 case accuracy of 0.242, retrieved case hit rate of 0.422, and average evidence support rate of 0.992. Since the current draft answer is extractive, the evidence checker rarely revises the answer; the next stage will test the same evidence-checking loop on LLM-generated drafts.

A small LLM pilot was then run locally using Qwen2.5 instruction models. On a 30-question pilot subset, Qwen2.5-1.5B with hybrid BM25 + MedCPT top-1 case prompting achieved answer token-F1 of 0.218, top-1 case accuracy of 0.333, retrieved case hit rate of 0.600, and average answer length of 57.633 words. This outperformed the LLM-only Qwen2.5-1.5B pilot, which achieved token-F1 of 0.097 without retrieval. The pilot also exposed a cross-case contamination problem: when the generator was given all top-k retrieved cases, it sometimes combined findings or impressions from several different cases into one answer. The revised top-1 case-grounded prompt and top-1 evidence checker are therefore important design choices for the proposed system.

## Limitations of the Preliminary Evaluation

This evaluation is still a feasibility check. The qrels are generated from dataset problem labels, so they are not equivalent to expert manual relevance judgments. The current agentic result also uses extractive drafts rather than full LLM generation. The next evaluation stage should therefore include LLM-only, basic RAG, and agentic RAG outputs, followed by manual evidence-support and hallucination checking on selected cases.
