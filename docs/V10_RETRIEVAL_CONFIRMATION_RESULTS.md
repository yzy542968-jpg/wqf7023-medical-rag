# V10 Cluster-Disjoint Retrieval Confirmation Results

Status: Test retrieval confirmation complete; no Test-driven retuning.

## Execution and eligible population

The frozen Test partition contained 574 cases. Six prespecified cases
(`CXR894`, `CXR1293`, `CXR1297`, `CXR1615`, `CXR2601`, and `CXR2765`) had
`empty_report` status in the frozen RadGraph artifact and therefore failed the
pre-existing technical eligibility rule. They were not replaced. The complete
case analysis contains 568 cases and 1,704 question-role rows per retrieval
system.

An initial execution was terminated by the external 120-second process limit
after aligned scoring and 9 shuffled assignments. The script writes formal
artifacts only after all assignments finish, so no partial result was retained
or inspected. It was rerun under the identical frozen configuration with a
longer process allowance. The completed run took 886.43 seconds.

## Retrieval results

| System | nDCG@10 | MRR | Hit@1 | Hit@5 | Hit@10 |
|---|---:|---:|---:|---:|---:|
| R0 BM25 | 0.140764 | 0.076758 | 0.032277 | 0.104460 | 0.166080 |
| R1 image-image | 0.334853 | 0.308640 | 0.225352 | 0.406690 | 0.440141 |
| R2 image-report | 0.317601 | 0.257459 | 0.170775 | 0.350352 | 0.415493 |
| R4 nine-feature | 0.349049 | 0.311148 | 0.230047 | 0.404930 | 0.441315 |
| **R5 fact-attention** | **0.360074** | **0.313605** | **0.238263** | **0.404930** | **0.444249** |

The prespecified case-grouped R5-minus-R4 nDCG@10 difference was 0.011025
(95% bootstrap CI 0.007698 to 0.014414; 568 cases). Because the lower bound is
above zero, the primary retrieval improvement is confirmed under the frozen
decision rule.

This is stronger evidence than the previous V9 near-duplicate sensitivity
analysis: exact and near-duplicate reports were grouped before Train,
Calibration, Validation, and Test allocation rather than removed only after
outcomes were available.

## Image-alignment control

Correctly aligned R5 achieved mean nDCG@10 0.360074. Across 100 deterministic,
fixed-point-free image derangements, the mean was 0.249631 (range 0.236211 to
0.264042). The aligned result exceeded every shuffled assignment; the plus-one
Monte Carlo p-value was 0.009901. The gain therefore depends on correct
image-case alignment rather than merely adding an arbitrary image vector.

## Retrieval confidence

On Test, the frozen calibrator obtained Brier score 0.167393, 10-bin ECE
0.045792, and AUROC 0.705460 for the offline Top-1 qrel target. The fixed 80%
Calibration-coverage threshold produced 81.51% observed Test evidence coverage.
No threshold was retuned.

These results support selective use of historical evidence, not diagnosis-risk
calibration. A low-confidence decision means only that the retrieved Top-1 case
is predicted not to meet the study's offline relevance threshold.

## Artifact trail

- Retrieval rows SHA-256: `68a1e5db7db21a0de258e30cb9f9c6f9cee892d89576c707b81ffb63a7617c91`
- Shuffled assignment summary SHA-256: `addfac05ea8788888adc3d485edccfc4046a3f51f55062d1f30f4b583a434bd3`
- Frozen configuration changed after Test: no
- Clinical interpretation: not claimed
