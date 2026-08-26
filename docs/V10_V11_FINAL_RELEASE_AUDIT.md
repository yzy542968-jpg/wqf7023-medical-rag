# V10/V11 Final Release Audit

## Release identity

- Release label: `v10-v11-final-thesis-freeze`
- Primary evidence: frozen V10 confirmation
- Development extension: V11 Train/Validation audits only
- Repository: `https://github.com/yzy542968-jpg/wqf7023-medical-rag`
- Artifact manifest: `artifacts/v10_v11_final_release_manifest.json`

## Research boundary

The final thesis models a new chest-radiograph case whose report is unavailable at inference. The input is one or two target images, an available indication and a question. The system retrieves other-patient historical image-report pairs, performs fact-aware reranking and case-scoped evidence selection, produces a bounded target-image answer and attaches deterministic case/section/fact provenance.

V10 is the final primary study. V11 is a development-only extension and did not instantiate a confirmation cohort. The release does not claim physician-adjudicated diagnostic correctness, patient-level independence, clinical safety, treatment utility, external generalization or deployment performance.

## Final artifacts

| Artifact | Status |
|---|---|
| `docs/P2_V10_V11_FINAL_MANUSCRIPT.md` | six chapters, 21,010 whitespace-delimited words |
| `deliverables/22097191_ZHANG_YUE_Final_Research_Project.docx` | generated and rendered |
| `deliverables/22097191_ZHANG_YUE_Final_Research_Project.pdf` | 58 A4 pages |
| `deliverables/22097191_ZHANG_YUE_Final_Defence.pptx` | 15 slides |
| `README.md` | V10/V11 scope and results synchronized |
| `docs/FINAL_RESULTS_REGISTRY.md` | V1/V2 historical, V9 historical, V10 primary and V11 development results separated |
| `docs/PROMPT_TEMPLATES.md` | final target/history separation and provenance contract recorded |

Every public artifact listed above is fingerprinted in the JSON release manifest. Large source data, image pixels, model weights, private reviewer keys and per-generation rows remain local under repository policy.
Text artifacts use UTF-8/LF canonicalized payload hashes so their fingerprints remain stable across Git line-ending conversion; DOCX, PDF and PPTX artifacts retain raw-byte hashes.

## Numerical audit

- V10 source/split counts agree across the manuscript, README, result registry and frozen summaries.
- V10 R5-minus-R4 nDCG@10 is `+0.01103`, 95% CI `[+0.00770,+0.01441]`.
- The post-hoc qrel audit preserves an overall positive R5-minus-R4 result under combined, label-only and fact-only definitions, but the abnormal combined interval crosses zero and the abnormal label-only difference is negative. The release therefore does not claim uniform clinical-similarity improvement.
- V10 aligned nDCG@10 is `0.36007`; shuffled mean is `0.24963`; plus-one p is `0.00990`.
- V10 G2-minus-G0 Token-F1 is `+0.05978`, 95% CI `[+0.05114,+0.06860]`.
- V10 G2-minus-G1 Token-F1 is `+0.00167`, 95% CI `[-0.00347,+0.00683]`; no G2-over-G1 superiority is claimed.
- V11 clean generation contains 48 cases and 432 rows. Case-to-fact-minus-whole-report Token-F1 and complete F1RadGraph intervals both cross zero.
- V11 reserved planner evaluation contains 96 author-defined examples; accuracy is `0.9167`, macro-F1 `0.9196`, and indication invariance `1.0000`.

## Software verification

- `python -m compileall -q app.py human_evaluation_app.py scripts src`: passed.
- Full local-data run, `python -m pytest -q --basetemp=.test_tmp/pytest -p no:cacheprovider`: `276 passed`.
- Clean-clone run: `272 passed, 4 skipped`; only the intentionally untracked OpenI source-integrity checks were skipped.
- Runtime validation rejects malformed candidate banks, duplicate case identifiers, inconsistent or non-finite embedding tensors, incompatible score arrays and invalid dashboard requests before ranking.
- A tracked-file secret scan found no Hugging Face or GitHub token patterns.
- The clean-clone GitHub Actions workflow covers compilation and all repository-runnable tests on `main`, `post-submission-improvements` and `v10-publication-extension`. Four historical source-integrity checks are explicitly skipped when the intentionally untracked `data/processed/openi_cases.jsonl` artifact is unavailable.
- The Dashboard reads the frozen V10 summaries and the completed V11 48-case/planner summaries; it no longer presents a one-case smoke test as the current generation result.
- Retrieval confidence is labeled as a research signal rather than diagnostic confidence.

## Document and presentation verification

- The DOCX was rendered through LibreOffice and Poppler using the repository Windows QA wrapper.
- All 58 PDF pages were inspected in contact sheets; the title page, contents, tables, chapter starts, references and appendices were also checked at full size.
- A blank page caused by an explicit Word page-break paragraph was removed by using heading-level `page_break_before`.
- Actual chapter page numbers were extracted from the rendered PDF and synchronized with the static contents page.
- The PowerPoint was generated with `@oai/artifact-tool`, rendered slide by slide, checked with `slides_test.py`, and inspected at full size.
- The final PowerPoint overflow test passed with no detected off-canvas elements.

## Human and external evaluation disposition

Independent radiologist review remains Future Work. The prepared blinded package retains empty rating fields and is not counted as completed evaluation. Authorized external patient-level validation also remains Future Work; the MIMIC-CXR adapter/runbook does not constitute an external result.

## Acceptance decision

The V10/V11 technical, reporting and demonstration package is complete for thesis submission. Further model development is not required to support the completed claims. Future work should add genuinely new evidence through independent clinical review and authorized external patient-level validation rather than post-outcome tuning of the frozen V10 study.
