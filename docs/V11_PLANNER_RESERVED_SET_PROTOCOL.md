# V11 Reserved Question-Planner Wording Protocol

## Purpose

This post-development protocol defines a second, non-clinical wording set for
the deterministic V11 question planner. The planner source was frozen in
commit `e106bfb` before this set was evaluated. No planner rule, label, example
or indication may be changed after the first evaluation.

## Design

- 96 English questions;
- eight operational intents with 12 questions per intent;
- no question duplicated from the earlier 64-item rule-coverage set;
- one deliberately distracting indication per question;
- accuracy, macro-F1, per-intent precision/recall/F1, confusion matrix and
  indication-invariance rate;
- deterministic one-shot evaluation with no result-driven repair.

The question is the primary planner input. A non-empty question must produce
the same intent with or without the distracting indication. The set was
constructed by the researcher with tool assistance after planner development;
it is reserved wording robustness evidence, not independent, physician-authored
or clinically adjudicated natural-language understanding.

## Artifacts

- Input: `data/splits/v11/v11_question_planner_reserved_benchmark.json`
- Output: `data/splits/v11/v11_question_planner_reserved_summary.json`
- Evaluator: `scripts/evaluate_v11_question_planner_benchmark.py`

The benchmark and this protocol must be committed before the output is
generated. The output records hashes of both the benchmark and planner source.
