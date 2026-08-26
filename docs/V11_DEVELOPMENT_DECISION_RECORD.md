# V11 Development Decision Record

## Decision

The V11 development extension is retained as a technical development scaffold with a deterministic case-to-fact evidence selector. No V11 confirmation claim is promoted, and V10 remains unchanged.

## Run

- Script: `scripts/run_v11_development_evidence_ablation.py`
- Source split: V10 cluster-disjoint `train` as development candidate bank and `validation` as development evaluation partition.
- Rows: 1,152 (384 cases x 3 fixed questions).
- Candidate shortlist: BM25 Top-100 for evidence construction, with qrel
  relevance computed against the complete 2,510-case Train bank.
- Output: `data/splits/v11/v11_development_evidence_ablation_summary.json`.
- Per-row audit index: `data/splits/v11/v11_development_evidence_ablation_rows.jsonl`.

## Observed development results

| Measure | Result |
|---|---:|
| Top-1 qrel-v2 proxy | 0.3271 |
| Top-3 mean qrel-v2 proxy | 0.3229 |
| full-bank qrel-v2 nDCG@10 proxy | 0.5537 |
| proxy-relevant case present in Top-100 | 55.12% |
| qrel>=0.5 relevant-item recall@100 | 12.00% |
| rows with a relevant item outside Top-100 | 79.69% |
| mean qrel-component availability in Top-100 | 72.99% |
| whole-report mean characters | 790.5 |
| case-to-fact mean characters | 316.5 |
| character reduction | 59.96% |
| whole-report mean units | 17.33 |
| case-to-fact mean units | 5.98 |
| case-to-fact provenance completeness | 100% |

## Interpretation

The deterministic selector gives a substantial context-size reduction while preserving complete case/section/source provenance on this development run. This is an engineering result, not evidence that final medical answer accuracy improved. The corrected full-bank calculation is materially lower than the earlier shortlist-internal nDCG diagnostic: 0.5537 rather than 0.6715. Only 12.00% of qrel>=0.5 relevant items were recovered in Top-100, and 79.69% of rows had at least one such relevant item outside Top-100. A compressed wrong case would still be wrong evidence.

The corrected within-list confidence normalization selected a proxy threshold of 0.7143 and accepted 99.83% of rows under the predefined 80% minimum-coverage rule. This threshold is not a clinical threshold. It is an artifact of the current score scale and report-derived proxy labels, so it must not be advertised as calibrated clinical risk.

## Decisions retained

1. Keep case-level retrieval as the first stage.
2. Add within-case fact/sentence selection as a separate, auditable stage.
3. Keep the compact output contract and deterministic provenance assembly.
4. Keep the target-outside-shortlist protection: evaluate it as retrieval failure, but do not construct a training positive that is absent from the shortlist.
5. Do not tune V10 or rewrite V10 metrics with qrel-v2.

## Decisions not made

- No learned V11 reranker was promoted.
- A clean deterministic MedGemma development diagnostic was completed on 48 Validation cases (24 report-indexed normal and 24 report-indexed abnormal), producing 432 rows across whole-report, sentence-only and case-to-fact policies. Case-to-fact produced Token-F1 0.1531 on all rows, mean input length 539.3 tokens and mean evidence length 245.9 characters, compared with 0.1312, 798.2 tokens and 672.3 characters for whole-report evidence. All policies had 100% answer-only contract validity and deterministic provenance validity. The prespecified case-to-fact minus whole-report Token-F1 difference was +0.02195, 95% case-bootstrap CI [-0.00026, +0.04302]; complete F1RadGraph differed by +0.01395, CI [-0.00691, +0.03442]. Neither interval excluded zero. This remains a development efficiency/auditability diagnostic, not a clinical accuracy estimate or superiority claim. The earlier interrupted 17-case trace remains separate and is not pooled with this result.
- A full Validation candidate-generation audit compared BM25, MedCPT, MedSigLIP and RRF union. RRF was mixed at K=100 and more promising at K=200, but no candidate policy was promoted and no downstream confirmation was run.
- A case-grouped bootstrap was added to the frozen V10 2x2 attribution audit. The fact-aware main effect remained positive, while the attention main effect and interaction confidence intervals crossed zero.
- The deterministic planner was evaluated on 64 author-defined development examples and, after the planner was frozen, on a second committed 96-item reserved wording set. The reserved set achieved accuracy 0.9167, macro-F1 0.9196 and indication invariance 1.0000. The observed errors were retained without rule changes. Both sets remain researcher-authored diagnostics; neither is independent, blinded or physician-validated.
- No human clinical relevance or answer review was completed.
- No external dataset validation was completed.
- No V11 confirmation cohort was instantiated.

## Required next experiment before confirmation

The reproducible commands and outputs are documented in
`docs/V11_REMAINING_OPTIMIZATION_AUDIT.md` and
`docs/V11_MEDGEMMA_GENERATION_RESULTS.md`. Only after candidate-budget
selection, generation policy selection, and a new confirmation protocol may a
V11 confirmation cohort be generated.
