# Final-QA Nested-OOF Robust Gate Protocol

## Status and purpose

This is the final bounded post-hoc development audit for the Final-QA
historical-source branch. It is frozen before nested out-of-fold outcomes are
computed. Previous Validation results are known: question-conditional policies
improved ordinary QA by approximately 2.7 percentage points but remained below
image-only B3 on supported-label macro-F1.

The audit asks one narrow question: can conservative use of B6, restricted to
question types with adequate training support and a minimum macro-F1 advantage,
retain some ordinary QA benefit without sacrificing rare supported labels?

## Nested case-level design

Five deterministic outer folds estimate the final development behavior. For
each outer fold:

1. the outer cases are hidden from all policy and hyperparameter decisions;
2. the remaining cases are divided into four deterministic inner folds;
3. every support/margin pair is evaluated by inner case-level OOF predictions;
4. the best admissible pair is selected using inner OOF only;
5. question-type utilities are refitted on all outer-training cases;
6. the frozen fold-specific policy is applied once to the outer cases.

All questions belonging to a case remain in the same outer and inner fold.

## Policy

For a structured question ID, B6 replaces B3 only when both conditions hold on
the current training cases:

```text
question record count >= minimum support

and

B6 option-label macro-F1 - B3 option-label macro-F1 >= minimum margin
```

Otherwise B3 is retained. Ties retain B3. Gold answers are used only to estimate
training-fold utilities and metrics; they are never an inference feature.

## Fixed grid and inner selection

Minimum support:

```text
5, 10, 20, 40, 80, 120
```

Minimum macro advantage:

```text
0.00, 0.01, 0.02, 0.03, 0.05,
0.075, 0.10, 0.15, 0.20, 1.00
```

The `1.00` margin is an explicit B3-like high-conservatism candidate. Inner OOF
selection maximizes supported-label macro-F1 subject to question exact and
option micro-F1 remaining within `-0.001` of B3. Ties prefer higher ordinary QA,
then larger margin and support.

## Advancement and stopping

The final nested-OOF policy must exceed both B3 and random-history macro-F1,
keep exact and option micro-F1 within their non-inferiority margins, and use B6
on at least one disagreement. Passing would justify considering a separately
frozen final fit and confirmation protocol; it would not itself unlock Test.

If the nested audit fails, this branch closes. No further threshold, MLP,
retriever, prompt or generation search will be performed on Final-QA Validation.
V10/V11 and all earlier Final-QA artifacts remain unchanged.

## Evidence boundary

Nested OOF reduces same-case hyperparameter optimism, but the policy family was
designed after earlier Validation outcomes were known. Results remain
development evidence and automated report-reference consistency, not clinical
accuracy or external validation. Final-QA Test remains inaccessible throughout.
