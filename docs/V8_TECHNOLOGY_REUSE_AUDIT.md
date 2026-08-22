# V8 Technology Reuse Audit

## 1. Purpose

V8 is a new development and confirmation study for a candidate-level
multimodal reranker. It is not a modification of the frozen V7 adaptive-alpha
study. V7 remains a completed mixed/negative extension and all V7 protocols,
cohorts, outputs, and claims remain immutable.

The audit establishes which existing components can be reused and which
component is genuinely new.

## 2. Reused components

The following components are reused without changing their frozen behavior:

| Component | V8 use |
|---|---|
| OpenI/IU-Xray processed cases | Same source dataset and case-ID canonicalization |
| BM25 retriever | Text-only Top-100 shortlist baseline |
| MedSigLIP `google/medsiglip-448` | Frozen image/report similarity features |
| V6 sentence-aware chunking | Frozen report chunk construction |
| V6 MedGemma 1.5 | Secondary descriptive QA transfer only |
| V5/V6 BioLinkBERT verifier | Secondary automated evidence filtering only |
| Case-grouped bootstrap | Primary uncertainty unit remains case ID |

No foundation-model weights are fine-tuned in the primary V8 experiment.

## 3. New component

V8 introduces a candidate-level learned scoring function. Instead of predicting
one alpha for an entire query, the model assigns a score to every report in the
BM25 Top-100 shortlist using text/image scores, candidate ranks, score gaps,
query state, and question-type indicators.

The primary comparison is:

```text
validation-tuned global text/image fusion
                    vs
candidate-level learned multimodal reranker
```

The reranker cannot recover a target report that is outside the BM25 Top-100
shortlist. Such cases remain in evaluation and are reported as retrieval
failures.

## 4. Case separation audit

The source contains 3,851 processed cases. After formal prior-use exclusions,
V7 development blocks, and the instantiated V7 confirmation cohort, the V8
post-V7 frame contains:

```text
Eligible cases                         279
Report-indexed normal                  185
Report-indexed abnormal                 77
Report-index indeterminate              17
Stratifiable cases                     262
Readable cases                         279
Readable image views                   536
```

The V8 development source is the already case-ID-disjoint V7 development
manifest containing 720 cases in Train A, Train B, and Validation blocks. The
V8 confirmation frame is case-ID-disjoint from those blocks and from the V7
confirmation cohort.

Patient-level independence cannot be asserted because reliable patient IDs
are unavailable in the processed data.

## 5. Confirmation composition rule

Before V8 confirmation IDs are generated, the composition is fixed as:

```text
Selected candidate pool       240 cases
Report-indexed normal         170 cases
Report-indexed abnormal        70 cases

Targets                       120 cases
  normal                       85
  abnormal                     35

Distractors                   120 cases
  normal                       85
  abnormal                     35
```

The 17 report-indexed indeterminate cases are excluded from the primary
stratification frame. The remaining 22 stratifiable cases are not a silent
replacement pool.

## 6. Audit boundary

This audit does not claim that V8 will outperform V7 or the global baseline.
The result may be positive, null, or negative. V8 is only promoted to a new
positive methodological contribution if its prespecified case-grouped
confidence interval establishes superiority over the validation-selected
global fusion comparator on the untouched confirmation cohort.
