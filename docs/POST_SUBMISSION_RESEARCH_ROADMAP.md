# Post-submission research roadmap

This document describes work added after the frozen P2 submission and work that remains future research. It does not revise or retroactively extend the submitted manuscript, PDF, or presentation.

## Completed post-submission engineering

- Fresh-clone Demo Mode using three redistributable sample cases.
- Streamlit integration smoke testing and runtime data preflight.
- CI, dependency metadata, direct-version lock, citation metadata, software license, and data-use boundaries.
- A leakage-resistant v2.1 benchmark with disjoint cases, same-report distractors, natural paraphrases, fact probes, and unanswerable questions.
- A bounded closed-loop evidence agent with an auditable plan/retrieve/assess/rewrite/retry/answer-or-abstain trace.
- Answerability calibration, false-answer rate, ECE, Brier score, risk-coverage curves, and AURC.
- A locked-system replication cohort that cannot alter previously selected hyperparameters.
- A result-blind second wording-transfer set and preregistered lexical-first semantic-fallback V2.3 policy.
- Paired case-bootstrap intervals, transfer failure taxonomy, and semantic-planner compute-budget reporting.

## Future work: independent human evaluation

Human evaluation is intentionally future work because no suitable independent clinical raters were available for the current study. Automated metrics, the semantic verifier, or an LLM judge must not be described as a substitute for clinical expert assessment.

### Rater requirements

- Preferred: at least two radiologists or radiology trainees with supervised chest-imaging reporting experience.
- Acceptable feasibility study: one radiologist plus one clinician with documented experience interpreting radiology reports.
- Raters must not have contributed to system development, prompt selection, threshold calibration, or case selection.
- Raters must disclose training level, years of relevant experience, and conflicts of interest.

### Rater task

Each rater independently reviews a randomized, system-blinded set containing the question, source report evidence, and anonymized candidate answers. They score:

1. Clinical factual correctness on a 1-5 anchored scale.
2. Evidence support on a 1-5 anchored scale.
3. Completeness relative to the question on a 1-5 anchored scale.
4. Presence of a clinically meaningful unsupported claim as yes/no.
5. Appropriateness of abstention as yes/no for unanswerable questions.
6. Pairwise preference when comparing the locked baseline and closed-loop agent.

The interface must not reveal system name, retrieval route, confidence, or answer order. Answer order should be randomized independently per case.

### Sampling and analysis

- Pre-register the primary endpoint and analysis before ratings are revealed.
- Use at least 100 unique questions, stratified by answerability and question family; a larger sample should be chosen from an a priori power analysis.
- Keep all cases disjoint from development and threshold-selection data.
- Report raw score distributions, paired confidence intervals, and effect sizes rather than only p-values.
- Report inter-rater agreement: weighted Cohen's kappa for two ordinal raters or Krippendorff's alpha when there are more raters or missing ratings.
- Resolve disagreements only for qualitative error analysis; do not replace independent ratings with consensus scores after seeing system identity.

### Ethical boundary

The dashboard remains a research demonstrator and must not be used for diagnosis, triage, or patient-care decisions. A prospective clinical study would require institutional governance, privacy review, and task-specific safety monitoring.

## Other future validation

- Pre-register a shift-aware abstention or escalation policy on new development data, then evaluate it on a third untouched wording distribution; V2.3 improved robustness but increased false answers on the second reserved set.
- External replication on a separately licensed institution or benchmark rather than another split of OpenI.
- Prospective evaluation of abstention and escalation under distribution shift.
- Multimodal image-and-report retrieval once image licensing and image-level ground truth are available.
- Robustness tests for negation, temporal comparison, rare findings, and adversarially phrased questions.
- Cost and latency evaluation for local and hosted deployment targets.
