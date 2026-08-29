# V16 Standard Metric Implementation Record

## Status

This record fixes the implementation used for the secondary V16 language-
generation metrics. It was committed while deterministic confirmation
generation was still running and before the complete V16 generation matrix or
any V16 standard-metric result was available. It does not modify the frozen
primary outcome, model, prompt, route, retrieval configuration, or Test frame.

## Implementations

The following exact local packages are used:

- `pycocoevalcap==1.2` for BLEU-1, BLEU-4, ROUGE-L, and CIDEr;
- `nltk==3.10.3` for METEOR, including the WordNet and OMW 1.4 resources;
- `bert-score==0.3.13` with `roberta-large`, baseline rescaling enabled, and
  the package-reported model hash retained in the result JSON.

The single local `roberta-large/model.safetensors` evaluation weight has
SHA-256 `047c85f0b96269cd62e6f732644f067004eebd95af5b5d35965ae2528f13bf38`.
The cache is machine-local and excluded from Git; its identity is public here
and will also be written into the final metric summary.

The local machine does not provide a Java runtime. The Java-backed COCO METEOR
implementation is therefore not used or silently approximated. NLTK METEOR is
reported by name and version so that it is not conflated with values produced
by another METEOR implementation.

## Aggregation and inference

The evaluator reports standard corpus scores for the full matrix and separately
for `no_history`, `retrieved_history`, and `random_history`. It also retains a
per-row score for paired comparisons. Paired differences are averaged within
each case across Findings and Impression before 10,000 bootstrap resamples are
drawn with seed `1626`. This preserves the case as the statistical unit.

BERTScore F1 is computed with the same reference/prediction strings used by the
lexical metrics. Empty text is retained; no row is deleted based on model
output. The standard metrics are secondary and cannot be used to choose a new
route, revise the adapter, alter the prompt, or change the primary Token-F1
decision rule after Test evaluation.

## Interpretation boundary

These scores measure automated similarity to hidden OpenI report sections.
They improve comparability with report-generation literature but do not measure
physician agreement, diagnostic correctness, clinical safety, treatment
utility, patient benefit, or external generalization. Cross-paper comparisons
remain descriptive because splits, report targets, image views, training data,
and metric implementations differ.
