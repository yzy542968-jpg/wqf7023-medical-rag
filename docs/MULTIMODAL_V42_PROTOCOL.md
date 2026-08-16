# Multimodal V4.2 protocol

## Research question

Can chest-X-ray pixels improve report-grounded retrieval when their role is limited to reranking a text-retrieved candidate set?

V4 and V4.1 showed that unrestricted global fusion does not improve the strong BM25 baseline. V4.1 development-only analysis found that BioViL-T has useful local discrimination after BM25 narrows the pool. V4.2 freezes that two-stage architecture before confirmation is inspected.

## Fixed systems

1. `report_only_bm25`: rank all 720 reports from indication plus question.
2. `image_only_biovil_t`: rank all 720 report embeddings by image-to-report cosine similarity.
3. `paired_biovil_t_shortlist_reranker`: retrieve the top 100 reports with BM25, min-max normalize BM25 and image cosine scores within that shortlist, average them with weights 0.5 and 0.5, rerank the shortlist, then append the remaining reports in BM25 order.

All parameters are fixed. There is no V4.2 grid search, prompt selection, threshold selection, or confirmation tuning.

## Data boundary

- Candidate pool: fixed union of 720 cases.
- Development: 600 cases and 1,800 questions.
- Confirmation: 120 disjoint cases and 360 questions.
- Pixel source: verified official NLM archive with SHA-256 `baf3abfe19ba5d58efe69002aed1e71aa2e6d5efb3238db9adcac210ad44bdf2`.
- Encoder: official BioViL-T image and text components in their normalized 128-dimensional joint space.

The V4.2 development run must reproduce a paired MRR above BM25 and be committed before confirmation. If it does not, confirmation is not run.

## Confirmation analysis

Confirmation is evaluated once. Primary comparison is paired reranker minus report-only BM25 for MRR. Hit@1, Hit@5, Hit@10, and deterministic top-1 report Token-F1 are also reported. Uncertainty uses 5,000 paired bootstrap resamples over case IDs with seed 7023 and 95% percentile intervals.

## Interpretation limits

An improvement demonstrates that actual image pixels can contribute to retrieving paired textual evidence under a constrained two-stage design. It does not demonstrate autonomous image diagnosis, radiologist-level QA, external generalization, or clinical safety.
