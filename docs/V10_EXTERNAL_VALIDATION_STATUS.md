# V10 External Validation Status

## Recorded status

Status: `adapter_ready_authorized_data_absent`

No authorized MIMIC-CXR record list, report archive, split table, or JPG tree was present in the local research workspace when the V10 publication extension was frozen. Therefore, this repository reports no external-validation cohort, metric, comparison, or clinical generalization claim.

This is a resource boundary, not a failed statistical result. OpenI/IU-Xray remains the only source used for V10 model development and confirmation.

## Completed readiness work

The project contains a tested adapter in `src/medical_rag/similar_case/mimic_cxr_adapter.py`. It:

- preserves `subject_id`, `study_id`, and `dicom_id` rather than replacing them with case-only identifiers;
- constructs the official nested report and JPG paths;
- parses findings and impression sections;
- rejects missing reports or images instead of silently replacing cases;
- detects studies whose images cross inconsistent official split labels;
- creates deterministic subject-level partitions by SHA-256 ordering; and
- supports a patient-disjoint external split when authorized inputs become available.

The corresponding unit tests verify report parsing, path construction, deterministic partitioning, subject uniqueness, blinded review rows, and rejection of incomplete reviewer ratings.

## Frozen future execution rule

An external run must use the frozen V10 retrieval, evidence, calibration, generation, and metric policies without OpenI-specific retuning on the external Test partition. Any development required for source-format adaptation must be documented separately and may not inspect external Test outcomes.

Protected reports, image pixels, identifiers, embeddings, prompts, and generations must remain local. Only project-authored code, aggregate metrics, non-identifying fingerprints, and permitted synthetic fixtures may enter the public repository.

A smaller external subset is acceptable only if its deterministic patient-level sampling rule, eligibility checks, sample size, metrics, and failure policy are committed before selected identities or outcomes are inspected.

## Permitted manuscript statement

> A patient-aware MIMIC-CXR adapter and deterministic external-validation protocol were implemented, but authorized MIMIC-CXR data were not available in the study environment. Consequently, external generalization was not evaluated and remains future work.

The manuscript must not describe the adapter, synthetic tests, or runbook as completed external validation.
