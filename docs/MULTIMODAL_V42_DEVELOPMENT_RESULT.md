# Multimodal V4.2 development result

## Status

- Preregistration commit: `5846649`
- Fixed policy: BM25 top 100, shortlist min-max normalization, text/image weights 0.5/0.5
- Evaluated split: development only
- Questions: 1,800 across 600 cases
- Confirmation status at this record: not evaluated

## Result

| System | Hit@1 | Hit@5 | Hit@10 | MRR | Token-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Report-only BM25 | 0.4839 | 0.6333 | 0.6878 | 0.5583 | 0.5798 |
| Image-only BioViL-T | 0.0217 | 0.0717 | 0.1217 | 0.0561 | 0.2883 |
| Paired shortlist reranker | 0.5367 | 0.6861 | 0.7450 | 0.6087 | 0.6530 |

The paired reranker improved development MRR by 0.05035 absolute, or 9.02% relative to BM25. Hit@1 improved by 5.28 percentage points and deterministic top-1 report-grounded Token-F1 improved by 0.07326.

The registered development gate passed. The exact summary and implementation must be committed before the single confirmation evaluation is permitted.

## Meaning

The improvement isolates a useful role for image pixels: they refine a text-generated candidate set rather than replace report retrieval. This supports a two-stage multimodal architecture and rejects the earlier unrestricted-fusion design. Confirmation determines whether the development-selected policy transfers to disjoint held-out cases.
