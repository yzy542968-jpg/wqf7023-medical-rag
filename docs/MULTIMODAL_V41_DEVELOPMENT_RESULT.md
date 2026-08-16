# Multimodal V4.1 development result

## Status

- Preregistration commit: `a8cd6d1`
- Evaluated split: development only
- Candidate pool: 720 cases with 1,399 official image views
- Development questions: 1,800 across 600 cases
- Confirmation split: not evaluated

## Result

| System | Hit@1 | Hit@5 | Hit@10 | MRR | Token-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Report-only BM25 | 0.4839 | 0.6333 | 0.6878 | 0.5583 | 0.5798 |
| Image-only BioViL-T | 0.0217 | 0.0717 | 0.1217 | 0.0561 | 0.2883 |
| Paired global RRF | 0.4839 | 0.6333 | 0.6878 | 0.5583 | 0.5798 |

BioViL-T improved image-only MRR by 3.33 times over V4 BiomedCLIP and improved Hit@10 from 0.0250 to 0.1217. The domain-specific encoder therefore passed the registered encoder-improvement condition.

Global weighted RRF still selected text weight `1.0` and did not improve over BM25. The confirmation gate failed its fusion-improvement and non-text-only conditions. Confirmation remains sealed.

## Development-only error analysis

The image signal is substantially better than in V4 but remains too weak for unrestricted global fusion. A declared development-only exploratory grid tested whether the image signal is more useful after text retrieval narrows the candidate set. The best observed policy was:

- take the top 100 BM25 candidates;
- min-max normalize BM25 and image cosine scores within that shortlist;
- combine them with equal weights;
- rerank the shortlist and append the remaining candidates in BM25 order.

This development policy reached MRR 0.608660 versus 0.558307 for BM25. It was selected from shortlist sizes 3, 5, 10, 20, 50, and 100 and text weights from 0.00 to 1.00 in increments of 0.05. It is exploratory until separately frozen and rerun under the V4.2 protocol.

## Interpretation

The result supports a two-stage architecture: text retrieval supplies precision-oriented candidate generation, while image-report alignment reranks a constrained candidate set. It does not support unrestricted rank fusion. V4.2 will freeze the selected two-stage policy before any confirmation result is inspected.
