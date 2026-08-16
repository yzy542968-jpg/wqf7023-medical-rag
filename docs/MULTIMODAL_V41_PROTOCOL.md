# Multimodal V4.1 protocol

## Rationale

V4 tested a generic biomedical CLIP encoder and failed on the development split: image-only MRR was 0.016839 and weighted RRF selected a text weight of 1.0. That result is preserved at commit `cf7eb66`. V4.1 is a declared model-family correction, not an overwrite of V4.

BioViL-T is designed specifically for chest X-rays and radiology reports. Its official image and text components project into the same normalized 128-dimensional space. V4.1 tests whether this domain alignment improves exact image-to-report retrieval under the same frozen data boundary.

## Frozen protocol

- Development: 600 cases and 1,800 questions.
- Confirmation: 120 cases and 360 questions.
- Candidate pool: the fixed union of 720 cases.
- Images: all available projections from the verified official NLM archive.
- Report text: findings plus impression, truncated to 256 tokens.
- Image transform: resize to 512, center crop to 448, following the official BioViL-T inference pipeline.
- Multiple projections: encode independently without treating a projection as a temporal prior, average, then normalize.
- Retrieval systems: BM25, BioViL-T image-to-report cosine, and weighted RRF.
- Weight selection: development MRR over the fixed 0.0 to 1.0 grid.

The candidate pool, questions, answer routing, metrics, RRF constant, weight grid, and tie-break are unchanged from V4. This isolates the encoder-family change.

## Confirmation gate

Confirmation is run only when all three development conditions hold:

1. BioViL-T image-only MRR exceeds the V4 BiomedCLIP MRR of 0.016839.
2. Selected paired-fusion MRR exceeds report-only BM25 MRR.
3. The selected text weight is less than 1.0.

If any condition fails, confirmation remains sealed and V4.1 is reported as a development-only negative result. If all conditions pass, the selected development result must be committed before the single confirmation run.

## Outcomes and limits

Retrieval outcomes are Hit@1, Hit@5, Hit@10, and MRR. Downstream Token-F1 uses the deterministic answer field from the top-ranked report, keeping generation effects out of the retrieval comparison. Confirmation uncertainty uses 5,000 case-level paired bootstrap samples with seed 7023.

Success would show that actual pixels contribute to paired evidence retrieval. It would not establish diagnostic performance, clinical safety, or external-dataset generalization.
