# V5 Researcher-Reviewed Qualitative Error Analysis

## Analysis status

This document reports deterministic screening, tool-assisted initial coding, and researcher-reviewed interpretation of frozen V5 confirmation outputs. On `2026-08-19`, the researcher accepted all 24 taxonomy v1.1 proposals without modification. The review outcome is `24 accepted, 0 modified, 0 excluded`. Cautious terms such as `possible` and `appeared` remain necessary because this review is explanatory rather than clinical adjudication.

No model, prompt, threshold, dataset, or frozen result was changed. Selection follows `docs/V5_QUALITATIVE_ANALYSIS_PROTOCOL.md`, committed as `d3b0765` after the V5 technical freeze (`10f57ba`, tag `v5-technical-freeze`).

## Evidence base

The numeric index contains 360 question-level rows from 120 paired cases: 120 findings, 120 impression, and 120 summary questions. Categories overlap and therefore must not be summed as mutually exclusive outcomes.

| Automated screening category | Questions | Unique cases | Findings | Impression | Summary |
|---|---:|---:|---:|---:|---:|
| Retrieval improvement | 101 | 41 | 31 | 36 | 34 |
| Retrieval degradation | 33 | 16 | 11 | 10 | 12 |
| QA gain with support loss | 20 | 18 | 4 | 10 | 6 |
| Correct retrieval with final Token-F1 below 0.5 | 95 | 70 | 20 | 10 | 65 |
| Possible generation unsupported addition | 261 | 119 | 115 | 105 | 41 |
| Possible verifier over-rejection | 62 | 52 | 43 | 7 | 12 |
| Verifier-evidence disagreement | 41 | 32 | 18 | 12 | 11 |
| Abstention case | 25 | 22 | 14 | 2 | 9 |
| No obvious error from automatic rules | 24 | 21 | 4 | 5 | 15 |

These are screening counts, not human-validated error rates. The `possible_generation_unsupported_addition` rule is intentionally sensitive: it is triggered by a revised answer and cannot establish that the draft contained a hallucination.

## Representative case profile

The fixed rules select 24 unique questions from 19 cases, balanced across question type (8 findings, 8 impression, and 8 summary). Fourteen selected questions contain at least one sentence filtered by the checker, four include an abstention in at least one compared condition, and ten retrieve the paired case in the multimodal condition.

| Stratum | Selected | Paired case retrieved | Filtered sentence present | Multimodal abstention |
|---|---:|---:|---:|---:|
| Retrieval improvement | 6 | 0 | 1 | 0 |
| Retrieval degradation | 6 | 0 | 2 | 0 |
| QA gain with support loss | 6 | 4 | 6 | 0 |
| Correct-retrieval generation error | 6 | 6 | 5 | 3 |

The public case list is `experiments/post_submission_v5/qualitative_representative_cases.csv`. Full text and editable researcher fields remain in `outputs/v5_qualitative_researcher_review.csv`.

## Assistant-prefilled review recommendations

All 24 private worksheet rows contain an evidence-based assistant recommendation under refined taxonomy v1.1. Relative to the substantive protocol v1.0 interpretations, nine are `unchanged` and 15 are `refined`. This comparison describes assistant action only. The researcher subsequently accepted all 24 v1.1 proposals without further modification.

The overlapping researcher-accepted v1.1 labels include target-rank improvement (11 rows), target-rank degradation (7), Top-1 retrieval failure (14), Top-1 retrieval success (10), post-verification content loss (10), possible verifier over-rejection (9), QA gain with support loss (6), and generation omission (1). Less frequent refinements identify template-prefix filtering, de-identification ambiguity, report-internal inconsistency, and generation-focus error. These counts describe only the predefined review set and must not be interpreted as prevalence estimates.

## Provisional observations

### Rank improvement does not guarantee retrieval success

The six largest representative improvements move the target from ranks 59-98 to ranks 10-27. None reaches top-1. The image signal therefore improves relative ordering in these examples without selecting the paired report used for downstream QA. This pattern supports a narrow interpretation: retrieval metrics can improve while the actual top-ranked evidence remains mismatched. It does not show that the image is clinically misleading.

### Correct images can also degrade ranking

The six representative degradations move the target from ranks 5-22 to ranks 10-40, and none retrieves the paired case as the selected report. The selected reports often share common chest-radiograph language with the target, making visual or textual near-neighbor confusion a plausible explanation. Image inspection and researcher confirmation are required before assigning a more specific mechanism.

### QA gains can coexist with lower automated support

Across the six selected trade-off examples, final Token-F1 increases by 0.123-1.000 while automated support rate decreases by 0.050-0.500. Every example contains at least one filtered sentence, and four retrieve the paired case. The filtered material ranges from generic answer framing to substantive radiographic content. This suggests that a lower support rate may reflect either useful filtering or over-rejection; aggregate support alone cannot distinguish the two.

### Some downstream errors occur after correct retrieval

All six selected generation-error examples include the paired case at rank 1. Five contain filtered sentences and three end in multimodal abstention. In several rows, the draft answer closely repeats visible report wording before the checker removes it; v1.1 attributes the observed loss to `post_verification_content_loss` and treats `possible_verifier_over_rejection` as a cautious interpretation. In contrast, `CXR1897_v2_summary` already focuses on pectus deformity in the draft and omits the frozen reference conclusion of no acute disease, supporting a provisional generation-focus and omission interpretation.

### Locked exploratory findings

The following statements define the researcher-reviewed Chapter 4 and Chapter 5 synthesis:

1. **Target-rank improvement did not always translate into Top-1 retrieval success.**
2. **Report-level faithfulness did not guarantee alignment with the frozen target case.**
3. **Correct Top-1 retrieval did not guarantee a reference-consistent final answer.**
4. **In reviewed cases, automated verification sometimes appeared to remove report-supported content.**
5. **Some declines in automated evidence-support scores did not correspond to substantive answer degradation.**

Here, substantive answer degradation is a researcher judgment based on the frozen reference, retrieved report, and model outputs. It is not a clinical correctness judgment.

### Automatic labels require narrowing during review

The broad screening label `possible_generation_unsupported_addition` appears in 261 of 360 rows. Its prevalence reflects the extraction rule, not confirmed unsupported clinical content. The final thesis should report confirmed subtypes from the 24-row worksheet, or otherwise describe this only as an automated trigger frequency. The same caution applies to checker disagreement and abstention labels.

## Researcher review outcome

The researcher accepted all 24 v1.1 proposals after reviewing the case summaries and evidence notes. The review considered:

- whether a filtered sentence is actually supported by the selected report;
- whether failure follows wrong-report retrieval, generation omission/focus, polarity, or checker behavior;
- whether an abstention is excessive or appropriate given the selected evidence;
- whether common indication or report wording plausibly acts as a metadata shortcut;
- whether the proposed explanation can be stated without making a clinical correctness claim.

The completed fields and decision rubric are defined in `docs/V5_QUALITATIVE_REVIEW_GUIDE.md`. The recorded acceptance does not constitute radiologist adjudication or a clinical safety assessment.

## Interpretation limits

This selected set is purposive and enriched for extreme or diagnostically useful metric patterns. It cannot estimate population prevalence, patient-level clinical error, verifier sensitivity or specificity, or safety. Question-level rows from the same case are correlated. No independent radiologist adjudication or external validation is included.

The defensible conclusion is that V5 exposes a retrieval-grounding trade-off: aligned image features can materially change target ranking and sometimes improve downstream overlap, but better ranking does not ensure Top-1 target-case alignment, and checker-based filtering can reduce or remove answer content even after correct retrieval. These interpretations are researcher-reviewed but remain exploratory and non-clinical.

## Reproduction and validation

Regenerate the representative materials without running any model:

```powershell
python scripts/build_v5_qualitative_review_materials.py
```

Expected output is 24 unique rows, six in each representative stratum, with `9 unchanged / 15 refined` assistant actions and `24 accepted / 0 modified / 0 excluded` researcher decisions. The generated private worksheet must remain under `outputs/` and outside version control.
