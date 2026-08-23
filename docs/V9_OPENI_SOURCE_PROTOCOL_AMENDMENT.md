# V9 OpenI Source Protocol Amendment

## 1. Status

This amendment corrects the source-eligibility boundary in
`V9_DEVELOPMENT_PROTOCOL.md`. It is committed before V9 retrieval outcomes,
model selection, or confirmation-case instantiation. V5-V8 artifacts and
results remain frozen.

The original protocol conservatively restricted OpenI/IU-Xray to engineering
smoke tests because the processed rows do not expose patient identifiers. A
review of the primary source-collection paper provides stronger provenance
evidence and changes that eligibility decision.

## 2. Primary-source evidence

Demner-Fushman et al. describe collection of approximately 4,000 chest-X-ray
reports, each from a different patient, and state that no patient contributed
more than one study. The current normalized OpenI source contains 3,851 valid
paired cases from that collection.

Primary reference:

- Demner-Fushman D, Kohli MD, Rosenman MB, et al. Preparing a collection of
  radiology examinations for distribution and retrieval. *J Am Med Inform
  Assoc.* 2016;23(2):304-310. DOI: <https://doi.org/10.1093/jamia/ocv080>

## 3. Superseded clause

The following original-protocol statement is superseded:

> OpenI/IU-Xray may be used only for implementation smoke tests because
> reliable patient identity is unavailable in the processed source.

The amended source rule is:

> OpenI/IU-Xray is eligible as the primary V9 source. The one-study-per-patient
> property is inherited from the documented source-collection design rather
> than reverified from released patient identifiers. Case-disjoint partitions
> therefore operationalize patient-disjoint partitions under this provenance
> assumption.

The thesis must distinguish `source-design patient uniqueness` from
`identifier-verified patient disjointness`. It must not claim that released
patient identifiers were available or machine verified.

## 4. Formal V9 data roles

The normalized source is fixed at:

```text
path: data/processed/openi_cases.jsonl
case count: 3,851
SHA-256: 56e367190396011d4d67f43e7e733389a8346890bf8729e82fb4326d063bbd68
```

V9 uses the source as follows:

```text
prior-used OpenI cases
  -> development queries, validation queries, and historical candidate bank

previously untouched OpenI source frame
  -> future V9 confirmation-query source only
```

The historical candidate bank is fixed from development cases and excludes
every confirmation query report. Because the source collection contains at
most one study per patient, excluding the target study also excludes the
target patient under the documented provenance assumption.

The existing V8 reuse audit records 279 previously untouched eligible cases.
This amendment does not instantiate a final V9 confirmation subset from that
frame.

## 5. Offline relevance annotations

The target report remains hidden from every inference condition. Its fields
may be used only after retrieval for offline reference construction and
evaluation.

Primary graded retrieval relevance remains:

```text
0.60 * active OpenI indexed-problem similarity
+ 0.40 * locally generated RadGraph entity/relation fact similarity
```

OpenI `problems` and MeSH-style indexing are report-derived annotations. They
are not physician similarity judgments. `no indexing` cases are excluded from
graded-qrel eligibility rather than treated as normal.

RadGraph annotations are generated deterministically from hidden reports and
stored locally. Annotation generation is preprocessing, not an inference
feature. No target report text, target label, target fact, or target-report
embedding may enter BM25, MedSigLIP scoring, reranking, prompting, or agent
actions.

## 6. External replication

CheXpert Plus and MIMIC-CXR are no longer prerequisites for V9. They remain
future external-replication candidates because of their storage, access, and
data-governance costs.

## 7. Freeze boundary

At this amendment:

- V9 confirmation IDs remain uninstantiated;
- no V9 retrieval or QA outcome has been inspected;
- no fusion weight, learned reranker, prompt threshold, or agent threshold has
  been selected;
- the original V9 task, research questions, comparators, metrics, shuffled
  control, and promotion rules remain unchanged.
