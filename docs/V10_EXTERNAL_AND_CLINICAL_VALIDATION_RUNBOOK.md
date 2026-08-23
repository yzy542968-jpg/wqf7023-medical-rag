# V10 External and Clinical Validation Runbook

## Status

The external adapter and clinical-review package are implementation artifacts,
not completed evidence. No external or independent clinical result may be
reported until authorized MIMIC-CXR files or genuine reviewer ratings exist.

## MIMIC-CXR external confirmation

Required local inputs are the official record list, split CSV, report archive,
and corresponding authorized JPG files. The adapter preserves `subject_id`,
`study_id`, and `dicom_id` and refuses missing image/report files. V10 uses a
patient-level deterministic split; no subject may cross partitions.

The external experiment must load the frozen internal V10 retrieval,
calibration, evidence-selection, generation, and metric configuration without
OpenI-specific retuning. A smaller deterministic subset is allowed only when
its sampling rule is committed before identities and outcomes are inspected.

Protected reports, pixels, identifiers, prompts, embeddings, and generations
remain local. The public repository may contain code, aggregate metrics,
fingerprints, and synthetic fixtures only.

## Independent clinical review

After V10 Test outputs are frozen, the builder selects 100 cases by deterministic
hash and randomizes system presentation within case. The reviewer sees target
image reference, indication, question, answer, and retrieved evidence but not
system names or automatic scores.

Required ratings are:

1. retrieval similarity, 1-5;
2. target-answer consistency, 1-5;
3. historical usefulness, 1-5;
4. potential harm, 0-2;
5. within-case preference rank;
6. optional explanatory note.

Reviewer specialty, years of experience, review date, conflicts, exclusions,
and missing ratings are recorded separately. The validation program rejects
partially completed rows. Assistant-generated or researcher-imputed ratings
are prohibited.

