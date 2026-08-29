# Oracle-Structured-History Diagnostic

This addendum freezes a low-cost development diagnostic before its Calibration
or Validation outcomes are generated. It does not modify the primary Final QA
protocol or any V10-V16 artifact.

The diagnostic retrieves mapped V10 Train cases using the frozen MedSigLIP
target-image embedding. The retrieved cases vote in the 2,470-dimensional
Rad-ReStruct answer space, after which the prediction is decoded and cleaned by
the official-compatible hierarchy rules. Calibration selects one prespecified
configuration; Validation is reported once; Test is prohibited.

The experiment answers a narrow question: do visually similar, correctly paired
historical cases contain structured answer signal for the target case? It is an
upper-bound diagnostic because the historical payload uses gold Rad-ReStruct
vectors. A deployed system would need to infer those vectors from historical
report text. Consequently, this condition is not B6 or P2, cannot support the
primary RAG claim, and must be labelled `oracle-structured-history` in every
table and figure.

The frozen grid is recorded in
`config/final_qa_structured_history_diagnostic.json`. Selection uses supported-
label macro-F1 on Calibration. Ties prefer lower Top-K, uniform weighting, and
thresholds closest to 0.5. Validation outcomes cannot change the selected
configuration.
