# V7 Confirmation QA Transfer Result Record

## 1. Status

The V7 secondary QA transfer is complete under the frozen V7 confirmation
protocol `4821f38`, instantiated cohort freeze `25a39d8`, and retrieval result
record `ff629f4`. The transfer used the same 240-case candidate pool, 120
target cases, and 360 report-derived questions as the primary retrieval run.

This is a secondary descriptive transfer. It does not change the V7 H1/H2
retrieval outcomes, select a new retrieval model, or modify any V5/V6 artifact.

## 2. Frozen input lineage

| Artifact | Value |
|---|---|
| V7 config SHA-256 | `9c17552451db1a936bfa2b8510fb33ed032b00b18d312464df64caf5f8ca7d3f` |
| V6 generator config SHA-256 | `2e56bd1b08a5190ebee873e5e0346b74f15cddc3df5448de958f58fbfd0a6b26` |
| V7 cohort SHA-256 | `7ed42bfc4851350c767f631d744d0306ee9ac5a406a3b74ceb75a568ceb89c65` |
| V7 retrieval rows SHA-256 | `ca799a70e594983a8237e9bb18a67e226ce247aac79a6bac40e3aed2c42f0753` |
| V7 raw QA rows SHA-256 | `426809f860119e56f7d56c79348f05a0eeaf44aba762001619fb1d4c37b511ed` |
| V7 verified QA rows SHA-256 | `f88045f0ccccf01f09a6584fe7f976dfdd4d00cdadf4b9ebc9391e0104271d8e` |
| Verifier config SHA-256 | `302e8ce368351af087259e53f63e134b4514fa4b9e1fd3a209e5e041a101fe9f` |
| Generator | `google/medgemma-1.5-4b-it`, revision `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` |
| Verifier | `cnut1648/biolinkbert-mednli` with unchanged V5/V6 thresholds |

The generator received the clinical indication, question, and selected
report findings/impression. It did not receive image pixels. Decoding was
deterministic (`do_sample=false`, temperature `0.0`, maximum 256 new tokens).

## 3. Conditions

The same frozen MedGemma generator, prompt implementation, decoding policy,
and verifier were used for all three conditions:

```text
BM25 text-only Top-1 report       -> MedGemma 1.5
Global alpha*=0.52 Top-1 report   -> MedGemma 1.5
Adaptive alpha_q Top-1 report     -> MedGemma 1.5
```

Each condition contains exactly 360 rows with the same qid set. The verifier
was applied to the selected Top-1 report only. The reported verified Token-F1
is consistency with the frozen reference after automated filtering; it is not
physician-adjudicated clinical correctness.

## 4. Raw QA outcomes

| Condition | Raw Token-F1 | Source-balanced raw Token-F1 | Top-1 retrieval accuracy | Mean output tokens |
|---|---:|---:|---:|---:|
| BM25 | 0.53708 | 0.53476 | 0.5167 | 26.91 |
| Global alpha*=0.52 | 0.54741 | 0.55101 | 0.5583 | 32.89 |
| Adaptive alpha_q | 0.54651 | 0.54756 | 0.5361 | 28.49 |

The raw QA transfer shows higher descriptive Token-F1 for both multimodal
retrieval conditions than BM25. Global alpha*=0.52 is marginally higher than
adaptive alpha_q in this transfer. No inferential QA superiority claim is made;
the V7 protocol specifies QA as descriptive secondary evidence.

## 5. Verified QA outcomes

| Condition | Verified Token-F1 | Source-balanced verified Token-F1 | Evidence support | Final abstention | Revision rate | Exact match |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.53728 | 0.53496 | 0.9901 | 0.0000 | 0.2389 | 0.0000 |
| Global alpha*=0.52 | 0.54728 | 0.55070 | 0.9887 | 0.0000 | 0.2611 | 0.0000 |
| Adaptive alpha_q | 0.54667 | 0.54771 | 0.9908 | 0.0000 | 0.2444 | 0.0000 |

Relative to BM25, the verified Token-F1 differences are approximately:

```text
Global alpha*=0.52 - BM25   = +0.009999
Adaptive alpha_q - BM25     = +0.009392
Adaptive alpha_q - global   = -0.000607
```

The source-balanced sensitivity has the same qualitative ordering: global is
highest, adaptive is slightly lower, and both exceed BM25 descriptively.

The high automated support values should not be interpreted as evidence that
the answers are clinically correct. The unchanged verifier is an automated
report-grounding filter, and independent human evaluation remains future work.

## 6. Interpretation boundary

The complete V7 result is therefore:

```text
H1 retrieval: fail
H2 alignment control: pass
Secondary MedGemma QA transfer: descriptive mixed result
```

Correctly aligned image information remained useful under the frozen adaptive
retrieval pipeline, as supported by H2. However, the learned adaptive policy
did not outperform the validation-selected global fusion weight, and the
secondary MedGemma transfer does not reverse that conclusion. V6 remains the
principal completed study; V7 is a transparent mixed/negative adaptive-fusion
extension.

No claim is made about diagnosis, clinical utility, patient-level
independence, external validation, deployment safety, or physician-validated
correctness.

## 7. Technical execution note

The raw generation was initially interrupted by an execution time limit after
75 complete rows. The output was checked for duplicate keys and then resumed
under the identical frozen configuration, producing the remaining 1,005 rows.
No row was replaced, no case was changed, and the final matrix contains all
1,080 required tasks exactly once. The raw summary runtime describes the
continuation run; the verifier processed all 1,080 rows in one run.

## 8. Reproduction artifacts

The executable records are local-only under repository policy:

```text
experiments/post_submission_v7/v7_confirmation_qa_raw_rows.jsonl
experiments/post_submission_v7/v7_confirmation_qa_raw_summary.json
experiments/post_submission_v7/v7_confirmation_qa_verified_rows.jsonl
experiments/post_submission_v7/v7_confirmation_qa_verified_summary.json
```

The tracked runners are:

```text
scripts/run_v7_confirmation_qa.py
scripts/evaluate_v7_confirmation_qa.py
```

Large generation rows and model outputs remain local and are not uploaded to
the public repository. The lightweight result record preserves their hashes,
lineage, dimensions, and bounded interpretation.
