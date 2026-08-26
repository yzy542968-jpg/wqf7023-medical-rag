# OpenI Patient-Separation Evidence Audit

## Status

This is a post-freeze evidence-boundary audit. It does not alter the frozen V10
configuration, cohort, outputs, metrics, or conclusions.

## Verified evidence

The OpenI/IU-Xray source publication states that the original collection drew
4,000 reports from different patients and included no more than one study per
patient. Filtering records from that collection cannot introduce a second study
for an already represented patient. This supports patient separation by source
design. The evidence source is Demner-Fushman et al., DOI
[`10.1093/jamia/ocv080`](https://doi.org/10.1093/jamia/ocv080).

The local source audit found:

- 3,851 report rows and 3,851 unique raw `uid` values;
- 7,466 projection rows linked to the same 3,851 raw `uid` values;
- 3,851 processed cases and 3,851 unique `case_id` values;
- an exact one-to-one set mapping from raw `uid` to processed `CXR{uid}`;
- no released `patient_id`, `subject_id`, or equivalent linkage field.

The machine-readable audit is
`experiments/post_freeze_audits/openi_patient_separation_audit.json`.

## Claim matrix

| Claim | Status | Evidence boundary |
|---|---:|---|
| Case-ID disjointness | Verified | Deterministic split manifests |
| Exact/near-duplicate cluster disjointness | Verified in V10 | Cluster-disjoint split |
| Patient separation supported by source design | Supported | Original collection states no more than one study per patient |
| Patient separation re-verified from released identifiers | Not verified | No released patient/subject identifier is present |
| External patient-level generalization | Not established | Completed experiments use one source |

## Required wording

Use:

> Patient separation was supported by the original OpenI collection design,
> which included no more than one study per patient. However, it could not be
> independently re-verified from released subject identifiers because those
> identifiers were unavailable in the processed source.

Do not use:

- `identifier-verified patient-disjoint split`;
- `released patient IDs were used`;
- `external patient-level generalization was established`.

The OpenI adapter's `openi-source-unique:{study_id}` value is a source-design
surrogate used to operationalize the published one-study-per-patient design. It
is not a released patient identifier.

## Future work

Identifier-verified external patient-level validation remains Future Work on an
authorized source such as MIMIC-CXR/MIMIC-CXR-JPG. That extension requires a
separately frozen protocol and official `subject_id`/`study_id` fields. The full
source is multi-terabyte in scale, so it is not downloaded or represented as a
completed experiment in this thesis. A future study may prespecify a smaller
authorized subset rather than downloading the entire image collection.
