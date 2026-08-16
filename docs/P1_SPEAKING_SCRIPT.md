# P1 Speaking Script

Target length: 8 to 10 minutes.

## Slide 1: Title

Good morning. My project is titled "Evidence-Checking Agentic RAG for Radiology Report-Grounded Question Answering with Linked X-ray Cases."

The core idea is to build a medical question-answering system that retrieves radiology cases, answers using the expert-written report evidence, and then checks whether the generated answer is actually supported by the selected case.

## Slide 2: Problem

Large language models can answer medical questions fluently, but fluent answers are not always supported. In healthcare-oriented QA, this is risky because an answer should be traceable to evidence, not just plausible.

RAG can help by retrieving evidence before generation, but basic RAG still has failure modes. If several similar radiology cases are retrieved, the model may mix findings from different cases into one answer.

## Slide 3: Research Gap

My project focuses on this gap. Many systems retrieve report chunks or top-k documents, but radiology data is naturally case-based. A chest X-ray case includes image views and a radiologist report.

So the problem is not only whether the system retrieves useful text. The problem is whether the answer stays inside the selected case boundary.

## Slide 4: Research Questions

The project asks four questions:

First, does retrieval improve medical QA compared with LLM-only answering?

Second, does hybrid BM25 plus MedCPT improve case retrieval compared with lexical or dense retrieval alone?

Third, can evidence checking reduce unsupported or cross-case claims?

Fourth, can linked X-ray case context improve presentation without claiming autonomous image diagnosis?

## Slide 5: Dataset

I use the IU X-Ray / OpenI dataset. Locally, I processed 3,851 radiology report cases and retained 7,466 linked image-projection mappings.

Each case includes report sections such as indication, findings, and impression, plus problem labels and image filenames. I also created a clean QA seed of 360 questions from 120 cases.

## Slide 6: System Design

The proposed pipeline has four main stages.

First, the question is planned into a retrieval query. Second, the system retrieves complete radiology cases using hybrid BM25 and MedCPT retrieval. Third, the LLM drafts an answer using selected case evidence. Finally, the evidence checker tests whether answer claims are supported by the report evidence, and revises or abstains if they are not.

The scope is report-grounded QA, not raw-image diagnosis.

## Slide 7: Retrieval Results

On the clean QA seed, BM25 is stronger than MedCPT alone. BM25 reaches Hit@20 of 0.486, while MedCPT reaches 0.406.

However, the hybrid retrieval setting improves over both. With alpha 0.50, hybrid BM25 plus MedCPT reaches Hit@20 of 0.553 and MRR of 0.324.

This suggests that dense biomedical retrieval contributes complementary signal when fused with lexical retrieval.

## Slide 8: Generation Pilot

I also ran a local generation pilot using Qwen2.5 models on my CUDA GPU.

For Qwen2.5-1.5B, LLM-only answering had Token-F1 of 0.097. The strongest local RAG setting was hybrid top-1 case prompting, which reached Token-F1 of 0.218 and retrieved hit rate of 0.600.

This is still a pilot, but it confirms that the full pipeline can run locally and that retrieval improves over LLM-only answering.

## Slide 9: Error Analysis

The most important finding is cross-case contamination.

When the model sees several retrieved cases, it can copy impressions from multiple cases into one answer. If the checker allows support from anywhere in the top-k pool, those extra claims may appear supported, even though they do not belong to the selected case.

So my design uses top-1 case prompting and top-1 evidence checking for the case-grounded QA setting.

## Slide 10: Plan

The remaining work is evaluation depth.

I will run full-generation experiments over the 360 clean QA questions, apply evidence checking, and manually annotate 30 to 50 outputs for relevance, evidence support, hallucination, completeness, and case contamination.

The expected final contribution is a reproducible medical RAG framework showing why evidence support in radiology QA should be case-specific, not only top-k-document supported.

## Likely Q&A

### Is this image diagnosis?

No. The report is the evidence source. The images are linked case context and presentation material. The project does not claim to diagnose directly from raw X-rays.

### Why use OpenI?

It is public, de-identified, and contains paired chest X-ray images and radiology reports. This makes it suitable for a reproducible master's project.

### Why does MedCPT perform worse than BM25 alone?

The QA seed is case-specific and often depends on exact report wording. BM25 captures this lexical overlap well. The useful result is that MedCPT still adds complementary signal when fused with BM25.

### Why does evidence checking lower Token-F1?

Because the checker removes or abstains from unsupported generated claims. This can reduce lexical overlap with the reference, but it improves safety and faithfulness. In medical QA, unsupported claims should be penalized even if they sound complete.

### What is the main contribution?

The main contribution is showing and evaluating case-specific evidence checking for radiology report-grounded QA, especially the cross-case contamination failure mode in top-k medical RAG.
