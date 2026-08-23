# V10 Cluster-Disjoint Confirmation Protocol

Status: frozen before V10 Test execution and outcome inspection.

## Purpose

V10 is a publication-oriented extension of the immutable V9 study. It tests
whether V9's central retrieval and downstream QA findings survive a split that
groups exact and near-duplicate reports before partitioning. It also evaluates
fact-aware retrieval, multi-view aggregation, bounded structured generation,
and calibrated selective use of historical evidence. It is not an external or
physician-adjudicated clinical validation.

## Frozen population and split

The 3,851 OpenI cases were grouped into 3,013 connected duplicate clusters
before partitioning. The fixed cluster-disjoint allocation contains 2,510
Train, 383 Calibration, 384 Validation, and 574 Test cases. Model and policy
development used only Train/Validation; confidence calibration used only
Calibration. Test has not been used for model, prompt, threshold, evidence,
or stopping-rule selection.

All comparisons are case-ID and duplicate-cluster disjoint across partitions.
Patient-level independence cannot be claimed because reliable patient
identifiers are unavailable in the processed OpenI source.

## Retrieval confirmation

The primary comparison is the frozen fact-aware, learned-attention R5 ensemble
against the original nine-feature R4 reranker. The primary endpoint is
case-grouped mean nDCG@10 across the three prespecified question roles.
Secondary endpoints are MRR and Hit@1/5/10 at qrel gain at least 0.50.

The paired R5-minus-R4 difference uses a 10,000-iteration case-grouped
bootstrap with seed 7050 and reports a percentile 95% confidence interval.
Interpretation is:

- lower bound above zero: confirmed positive retrieval improvement;
- point estimate above zero with interval crossing zero: numerical improvement;
- point estimate at or below zero: no confirmed improvement.

One hundred deterministic, unique, fixed-point-free image derangements use
seed 7049. Each assignment recomputes the complete visual and learned-attention
state. The aligned R5 result is compared with the shuffled distribution using
a plus-one Monte Carlo p-value.

## Confidence and selective evidence

The frozen logistic calibrator estimates whether the R5 Top-1 case reaches
offline qrel gain 0.50. Its Calibration-only 80% coverage threshold is applied
once to Test. Test reports Brier score, 10-bin ECE, AUROC, observed coverage,
and the risk-coverage curve. Low confidence means that historical evidence is
withheld in G3; target-image answering remains available. This is not a
clinical abstention or safety claim.

## QA confirmation

Each technically eligible Test case contributes findings and impression
questions to four frozen conditions:

| Condition | Target image | Historical retrieval | Evidence input |
|---|---|---|---|
| G0 | yes | none | none |
| G1 | yes | R4 Top-3 | whole findings/impression |
| G2 | yes | R5-attention Top-3 | whole findings/impression |
| G3 | yes | R5-attention Top-3 | G2 evidence only above frozen confidence threshold |

Validation selected whole-report E0 over sentence/fact generator contexts.
Consequently, G2's historical context is intentionally the same granularity as
G1; their difference isolates the downstream effect of R5 versus R4 retrieval.
The internal `g2_hierarchical` identifier is retained for compatibility but
does not imply that the rejected E1/E2 generator policy is used.

MedGemma produces answer text only. At most two complete sentences are retained
and Python deterministically assembles uncertainty, historical support,
case/section provenance, and schema fields. The primary automated endpoint is
case-grouped Token-F1 against the source-report reference. Planned comparisons
are G2-minus-G0 and G2-minus-G1 with 10,000-iteration case-grouped bootstrap
intervals. F1RadGraph is a prespecified secondary metric. F1CheXbert is reported
only if its required implementation is already available; no substitute proxy
will be relabelled as F1CheXbert.

These metrics measure same-source report-reference consistency and graph
overlap. They do not establish diagnosis correctness or clinical utility.

## Technical failures and exclusions

The frozen case list cannot be silently replaced. A transient execution error
may be rerun under the identical configuration. A data-integrity failure is
retained as a documented protocol deviation and handled by complete-case
reporting; no next-ranked reserve case is substituted. Partial output files may
be resumed only by exact case/question/system keys without regenerating
completed rows.

## Frozen artifacts

| Input | SHA-256 |
|---|---|
| OpenI case JSONL | `56e367190396011d4d67f43e7e733389a8346890bf8729e82fb4326d063bbd68` |
| RadGraph JSONL | `631aa3e11cc52005656ee8a66de3de1ee5d3411a2f271a3c5f8a14de39b51599` |
| Cluster-disjoint split | `b4c1b091c3dbff0399d07c8350f4d8d68ce8ce52e0157dcc96f46af8c8baa7b3` |
| MedSigLIP embeddings | `f81f4629a8f6eb10dc6b35d868f719384adb40903e19d165f9fba2039fce8867` |
| Retrieval calibrator | `e88d85e911c9214baa814d478754e34923c51568091686d63a7bebe6796a762e` |

All R5 and learned-attention checkpoints are the tracked artifacts frozen in
the current V10 branch before this protocol. Test execution is authorized only
by `config/v10_confirmation.json`. No Test-driven retuning, model replacement,
prompt revision, threshold revision, or evidence-policy revision is permitted.
