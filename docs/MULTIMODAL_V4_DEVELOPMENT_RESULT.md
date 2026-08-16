# Multimodal V4 development result

## Status

- Preregistration commit: `795b1c9`
- Evaluated split: development only
- Candidate pool: 720 cases with 1,399 matched views
- Development questions: 1,800 across 600 cases
- Confirmation split: not evaluated

The source adapter matched all 7,466 image references in the 3,851 normalized IU-Xray cases to the official NLM PNG archive. The archive SHA-256 is recorded in `data/processed/openi_multimodal_source_manifest.json`.

## Result

| System | Hit@1 | Hit@5 | Hit@10 | MRR | Token-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Report-only BM25 | 0.4839 | 0.6333 | 0.6878 | 0.5583 | 0.5798 |
| Image-only BiomedCLIP | 0.0050 | 0.0150 | 0.0250 | 0.0168 | 0.2181 |
| Paired weighted RRF | 0.4839 | 0.6333 | 0.6878 | 0.5583 | 0.5798 |

The registered development sweep selected text weight `1.0`. Thus, the fused system reduced exactly to BM25. This falsifies the V4 development hypothesis that an off-the-shelf generic biomedical CLIP encoder would improve exact image-to-own-report retrieval in this setting.

## Decision

The V4 confirmation split remains sealed because running it with a development-rejected policy would spend the one-shot confirmation evaluation without scientific benefit. V4 remains a reproducible negative pilot rather than being overwritten.

V4.1 will be preregistered separately. It replaces the generic encoder with the chest-X-ray-specific BioViL-T joint image-report encoder and keeps the same frozen cohorts, candidate pool, metrics, weight grid, and no-confirmation-tuning rule. Any further design change must be based on development data and committed before confirmation evaluation.

## Interpretation

The result does not show that chest X-ray pixels are useless. It shows that generic BiomedCLIP embeddings are poorly aligned to the exact paired-report retrieval task on this cohort. The stronger report-only baseline is expected to benefit from question and indication words that overlap the report text, while the image encoder receives no question text. V4.1 tests whether domain-specific CXR alignment changes that conclusion.
