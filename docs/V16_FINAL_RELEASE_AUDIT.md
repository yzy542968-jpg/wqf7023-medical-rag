# V16 Final Release Audit

## Release identity

- Release: `v16-final-thesis-freeze`
- Final manuscript source: `docs/P2_FINAL_MANUSCRIPT.md`
- Final DOCX: `deliverables/22097191_ZHANG_YUE_Final_Research_Project.docx`
- Final PDF: `deliverables/22097191_ZHANG_YUE_Final_Research_Project.pdf`
- Defence deliverable: `docs/P2_FINAL_DEFENCE_SLIDE_OUTLINE.md`
- Aggregate manifest: `artifacts/v16_final_release_manifest.json`

No final PPTX is part of this release. The student will create the presentation from the maintained page-by-page outline.

## Scientific identity

V10 remains the methodological foundation and image-alignment study. V12 is the final learned retrieval method. V16 is the final integrated held-out method confirmation, combining frozen V12 retrieval with the Validation-selected section-aware MedGemma/QLoRA route. V11 and V13-V15 remain development or mechanism evidence.

The release preserves the following non-negotiable evidence boundaries:

- the target report is hidden at inference and excluded from the historical bank;
- relevance and answer references are report-derived automated proxies;
- reliable patient identifiers were unavailable, so patient-level independence is not claimed;
- independent blinded radiologist evaluation and MIMIC-CXR external validation remain Future Work;
- the 81-case empty-Findings-reference deviation is disclosed and the frozen primary denominator is retained;
- no model, split, qrel, prompt, output, metric, or clinical score was changed during release packaging.

## Document verification

- Markdown length: 19,366 words, within the requested 10,000-30,000 range and close to the 20,000-word target.
- Rendered document length: 56 pages.
- All 56 rendered pages were visually inspected in contact sheets.
- The final cover, running header, table of contents, chapter starts, tables, lists, code blocks, appendices, and final page were inspected after version-label corrections.
- Table-of-contents page numbers were reconciled to the final render.
- No replacement characters, missing cited author-year references, or duplicate numbered section headings were found by the release tests.

## Software verification

- Python compilation: passed.
- Full local-data test suite: 315 passed before final release packaging.
- Dashboard aggregate panel reads only version-controlled frozen V12/V16 summaries.
- The live upload workflow remains explicitly labelled as the V10 demonstration path.
- Raw reports, image pixels, model weights, prompt packs, per-row generations, private reviewer keys, and secrets remain excluded under repository policy.

## Result identity

- V12 nDCG@10: `0.61590`; V10 R5 comparator: `0.55313`; paired difference `+0.06277`, 95% CI `[+0.05460,+0.07082]`.
- V16 retrieved-history Token-F1: `0.25591`; base comparator: `0.20570`; paired difference `+0.05020`, 95% CI `[+0.03973,+0.06108]`.
- Final answer-contract and provenance validity: `100%`.
- Token-ceiling rate: `87.85% -> 56.60%`.
- Non-empty-reference sensitivity: `+0.04571`, 95% CI `[+0.03371,+0.05763]`.
- CheXbert reference-positive recall decreased slightly; this negative secondary result is retained.

## Acceptance decision

The package is accepted for thesis submission and local demonstration within the stated automated, retrospective, same-source evidence boundary. It is not accepted or represented as a clinical diagnostic system, a prospective safety evaluation, independent physician validation, or external patient-level generalization study.
