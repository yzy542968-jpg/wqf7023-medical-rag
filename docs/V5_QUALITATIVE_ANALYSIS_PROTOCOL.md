# V5 Qualitative Analysis Protocol

## Protocol status

- Protocol version: `1.0`
- Protocol date: `2026-08-19`
- Analysis status: post-hoc, exploratory, researcher-reviewed
- V5 technical freeze tag: `v5-technical-freeze`
- V5 technical freeze commit: `10f57ba`
- Repository branch for this protocol: `post-submission-improvements`

This protocol is committed before systematic case extraction, selection, and coding. A small number of individual outputs had already been inspected during pipeline verification; therefore, this is not result-blind preregistration. The protocol freezes the subsequent selection and coding procedure and does not change any V5 model, prompt, threshold, dataset, primary result, or confirmatory experiment.

## Frozen evidence source

Only the V5 confirmation outputs are eligible. The analysis must not read development outputs, earlier project cohorts, or newly generated model outputs.

The frozen artifacts are identified by the following LF-normalized SHA-256 values from `experiments/post_submission_v5/artifact_manifest.json`:

| Artifact | SHA-256 |
|---|---|
| `config/multimodal_v5.json` | `579525a0d3e70b4cba92d74c44576c3ccee2dcb0ff7c21285621bc5d1249ce9a` |
| `data/processed/openi_multimodal_v5_cohort.json` | `501af7411f3f066c76a308e97f0e96c09c3230d5631250bd247508ec9e555489` |
| `experiments/post_submission_v5/confirmation_retrieval_summary.json` | `58c600613513bac888e19364990c184b6a8c4f590045f0cb9c5fcc1d15896ecc` |
| `experiments/post_submission_v5/qa_report_only/final_optimized_test_summary.json` | `4db90ca2b460dfb7fd69b3e30bf81083c58b27cfd6841b65a9153866cef1bcc0` |
| `experiments/post_submission_v5/qa_multimodal/final_optimized_test_summary.json` | `c4acd16395ee2af4d245c3fbc31f3c57994ef920431e43c4c8d83bf40bd960b4` |
| `experiments/post_submission_v5/v5_statistics.json` | `c8299585757bda9b77fcdc7af5228724896ff3d54c14827db4f71667892c7df8` |

The detailed retrieval rows, generation rows, prompt packs, report texts, and image pixels remain local. They are used only to produce the private review worksheet and are not primary evidence separate from the frozen aggregate outputs.

## Unit of analysis

- Retrieval comparisons are recorded at `qid` level because the three report-derived questions can produce different rankings.
- Case-level aggregation is reported separately using `case_id`; no question is treated as an independent patient.
- QA comparisons are recorded at `qid` level and summarized by `case_id` where a grouped summary is required.
- Question type is derived from the frozen question identifier suffix: `findings`, `impression`, or `summary`.

## Fixed selection rules

The following rules are applied before reading the selected report text or generated answer text. All ties are broken by ascending `case_id`, then ascending `qid`.

### Retrieval cases

1. Include every confirmation `qid` in the numeric index with the indication-plus-question BM25 and correctly aligned image conditions.
2. Mark `retrieval_improvement` when the correctly aligned image condition has a better target rank than the BM25 condition, or when target rank is tied and its extractive proxy Token-F1 is higher.
3. Mark `retrieval_degradation` when the correctly aligned image condition has a worse target rank than the BM25 condition, or when target rank is tied and its extractive proxy Token-F1 is lower. Every such `qid` is retained; it is not sampled away.
4. For representative text inspection, select the two largest rank improvements and two largest rank degradations within each question type. Rank improvements are ordered by rank gain, then MRR gain; degradations are ordered by rank loss, then MRR loss. The full numeric index remains available even when a category has fewer or more cases than the representative set.

### End-to-end QA cases

5. Mark `qa_gain_support_loss` when multimodal final Token-F1 is higher than report-only final Token-F1 and multimodal automated support rate is lower. Select the two largest Token-F1 gains within each question type, breaking ties by the largest support-rate decrease.
6. Mark `correct_retrieval_generation_error` when the selected report is correctly paired according to the frozen qrels but the final Token-F1 is below `0.5`. Select the two lowest multimodal final Token-F1 cases per question type.
7. Mark `possible_generation_unsupported_addition` when the generated answer contains a sentence removed or filtered by the semantic checker. These are inspected as possible unsupported additions, not treated as confirmed hallucinations.
8. Mark `possible_verifier_over_rejection` when sentence filtering lowers final Token-F1 relative to draft Token-F1 while the selected report is correctly paired. Mark `verifier_evidence_disagreement` when the draft/final change and evidence-support signal do not align. These labels describe suspected disagreement only; they are not verifier false-positive or false-negative labels.
9. Include every multimodal or report-only abstention in the numeric index. Representative abstention cases are selected by question type using ascending `qid` after the rule is applied.

## Coding taxonomy

Each inspected case receives zero or more provisional categories:

- `retrieval_improvement`
- `retrieval_degradation`
- `metadata_shortcut`
- `generation_omission`
- `negation_or_polarity_error`
- `unsupported_addition`
- `possible_verifier_over_rejection`
- `verifier_evidence_disagreement`
- `abstention_case`
- `no_obvious_error`

The category is an explanatory label, not a clinical correctness judgment. A case may receive multiple categories when more than one failure mode is visible.

## Review fields

The private review worksheet should retain, where available:

- `case_id`, `qid`, and question type;
- indication and question;
- report-only and multimodal ranks;
- selected report identifiers;
- reference answer and both generated answers;
- draft/final Token-F1, support rate, and abstention status;
- provisional category;
- researcher review status;
- final category and short evidence-based note.

The public case index must omit full report text, full generated answers, image pixels, prompt packs, and sensitive local paths. Final category labels require researcher review before they are described as findings in the thesis.

## Interpretation limits

This analysis is illustrative and explanatory. It does not provide an independent clinical rating, estimate clinical error rates, validate verifier correctness, or create new confirmatory evidence. The terms `possible over-rejection`, `possible under-detection`, and `verifier-evidence disagreement` must be used instead of verifier false-positive or false-negative claims.

After this protocol is committed, no new model development, prompt or threshold tuning, data expansion, or confirmatory experiment is permitted. Subsequent work may only extract and interpret the frozen outputs, summarize observed runtime artifacts, and integrate the findings into the thesis.
