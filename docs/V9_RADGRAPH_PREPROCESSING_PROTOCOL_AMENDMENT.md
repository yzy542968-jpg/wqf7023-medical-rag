# V9 RadGraph Preprocessing Protocol Amendment

## 1. Status

This amendment freezes the V9 RadGraph preprocessing and report-readability
rules before batch annotation, retrieval outcome inspection, or model
selection. It supplements the full-source split protocol at commit `afd7ef7`.
The instantiated split at commit `035f092` is not changed.

## 2. Preprocessing model and input

Formal V9 RadGraph facts are generated locally with:

```text
package: radgraph 0.1.18
model_type: modern-radgraph-xl
device: local CUDA when available
input: normalized findings + newline + normalized impression
```

Empty report sections are omitted before joining. The model and its foundation
encoder are frozen; no V9 case is used to update model parameters. Generated
entities and relations are flattened with the repository's deterministic
complete-reward fact representation.

The source report text, model annotation, and per-case facts remain local.
Only aggregate counts, software/model identifiers, hashes, and case-ID
fingerprints may be committed to the public repository.

## 3. Field-completeness audit

The fixed 3,851-case source contains:

```text
nonempty findings or impression: 3,826
empty findings and impression:      25
```

All 25 empty-report cases are report-indexed normal. Their frozen split
distribution is:

```text
train:       23
validation:   2
test:         0
```

An empty report is not interpreted as a RadGraph annotation containing no
clinical facts. Doing so would incorrectly reward empty-empty fact agreement.
The 25 cases therefore receive `radgraph_annotation_available = false`.

## 4. Shared candidate bank and graded-qrel frames

Every retrieval comparator uses the same report-bearing historical bank:

```text
frozen train partition:             2,631
empty-report exclusions:               23
shared historical candidate bank:  2,608
```

The excluded 23 train cases are retained in the source split manifest and
reported in the eligibility audit. They cannot be included by image-only
conditions because that would give those conditions a different candidate
bank.

Primary `0.60 label + 0.40 RadGraph` development/evaluation frames are:

```text
train qrel queries:       2,608
validation qrel queries:    374
test qrel queries:          752
strict untouched test:      262
total primary qrel frame:  3,734
```

The two empty-report validation cases may be retained for explicitly ungraded
engineering diagnostics but cannot enter primary qrel metrics or model
selection. The 752-case test and its 262-case strict sensitivity subset are
unchanged because every test report is nonempty.

## 5. Leakage boundary

RadGraph generation is offline reference preprocessing. For validation and
test queries, generated target-report facts are used only after retrieval to
construct qrels and calculate metrics. They are prohibited from:

- query construction;
- BM25 input;
- MedSigLIP input or score fusion;
- reranker features;
- generation prompts;
- verification or agent actions.

Candidate-bank RadGraph facts are also not inference features in the primary
retrieval systems; they are used only in offline relevance judgments.

## 6. Determinism, resumption, and audit

The batch generator processes canonical case IDs in ascending order and
writes one checkpointed JSONL record per case. Each record stores the source
text SHA-256, model type, annotation status, flattened fact list, and raw
model annotation. A completed run is canonicalized by case ID before its file
hash is calculated.

Interrupted execution may resume only when every existing record matches the
same source case and source-text hash. Errors are retained and cause a
non-zero final status; they are not silently converted to empty annotations.

The public completion audit must report source/model hashes, counts by split,
empty/error counts, fact-count distribution, output SHA-256, and primary qrel
frame fingerprints without including report text or generated facts.

## 7. Temporal declaration

At this amendment:

- one source case was used for a technical CUDA/structure smoke test;
- no retrieval score, ranking metric, QA answer, or agent outcome was
  inspected;
- the full RadGraph batch has not been generated;
- the test split remains unchanged and has not been evaluated.

The one-case smoke test verified executable behavior only and was not used to
select a model, threshold, fusion weight, or prompt.

