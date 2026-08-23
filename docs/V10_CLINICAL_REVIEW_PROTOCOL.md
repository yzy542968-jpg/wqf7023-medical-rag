# V10 Independent Clinical Review Protocol

Status: package-generation protocol frozen before V10 Test execution; reviewer
ratings not available.

After retrieval and QA Test outputs are frozen, 100 unique Test cases are
selected by SHA-256 ordering under `v10-clinical-case|7047|<case_id>`. Cases are
not selected from automatic performance. In selection order, odd positions use
the findings question and even positions use the impression question, yielding
50 of each.

Each case presents four systems (G0, G1, G2, G3) in a case-specific
deterministic blinded order. The reviewer sees the target image, indication,
question, answer, and retrieved historical evidence, but not system names,
automatic scores, confidence values, references, or correctness labels. The
private system key is stored separately.

Required ratings are:

- retrieved-case clinical similarity, 1 to 5;
- answer consistency with the target image, 1 to 5;
- usefulness of historical evidence, 1 to 5;
- potential harm, 0 to 2;
- within-case preference rank;
- free-text note.

Reviewer specialty, role, years of radiology experience, review date,
exclusions, and missingness must be reported. The package cannot be marked
complete while any required field is blank. Until a qualified independent
reviewer actually submits ratings, the study status is
`pending_independent_review`, and no clinical-utility or safety claim is made.

