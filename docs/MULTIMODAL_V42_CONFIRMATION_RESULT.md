# Multimodal V4.2 confirmation result

## Locked evaluation

- Preregistration commit: `5846649`
- Development policy commit: `a0358c4`
- One-shot confirmation result commit: `9bb6bf7`
- Confirmation cohort: 120 disjoint cases and 360 questions
- Bootstrap: 5,000 paired case resamples, seed 7023

## Metrics

| System | Hit@1 | Hit@5 | Hit@10 | MRR | Token-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Report-only BM25 | 0.4972 | 0.6056 | 0.6639 | 0.5558 | 0.5941 |
| Image-only BioViL-T | 0.0167 | 0.0667 | 0.1000 | 0.0505 | 0.3180 |
| Paired shortlist reranker | 0.5222 | 0.6722 | 0.7361 | 0.5962 | 0.6454 |

## Paired uncertainty

Differences are paired reranker minus report-only BM25.

| Metric | Difference | Case-bootstrap 95% CI | Paired randomization p | Holm-adjusted p |
| --- | ---: | ---: | ---: | ---: |
| MRR | +0.0404 | [+0.0140, +0.0681] | 0.0054 | 0.0108 |
| Hit@1 | +0.0250 | [-0.0139, +0.0639] | 0.2877 | 0.2877 |
| Hit@5 | +0.0667 | [+0.0306, +0.1056] | 0.0004 | 0.0020 |
| Hit@10 | +0.0722 | [+0.0278, +0.1194] | 0.0020 | 0.0060 |
| Token-F1 | +0.0513 | [+0.0214, +0.0835] | 0.0014 | 0.0056 |

The preregistered primary MRR improvement transfers to the held-out confirmation cases and its interval excludes zero. Hit@5, Hit@10, and Token-F1 also show positive intervals. Hit@1 improves numerically, but its interval crosses zero and must not be described as a statistically reliable gain.

## Research conclusion

The evidence supports a constrained multimodal contribution: chest X-ray pixels improve retrieval and downstream report-grounded QA when used to rerank a text-retrieved shortlist. Image-only retrieval remains far below report-only BM25, and unrestricted global fusion did not help in V4.1. The result therefore supports staged multimodal evidence retrieval, not replacement of radiology reports or autonomous image diagnosis.

The confirmation cohort remains within IU-Xray. External-dataset validity and independent clinician evaluation remain future work.
