# Final-QA Selective History Gate

The 384-step adapter improved over no history when supplied the Top-1
image-neighbour whole report, but it also showed negative transfer. This
Calibration-only development step tests whether a retrieval-confidence gate can
retain helpful history and fall back to no history for weak matches.

For each target case, the confidence score is the frozen MedSigLIP cosine
similarity of the eligible Top-1 V10 Train image neighbour. Candidate thresholds
are the empirical Calibration score quantiles `0.0, 0.1, ..., 0.9`. A question
uses the already generated B6 answer when its case score is at least the
threshold and the B3 answer otherwise. No answer text, gold label, report text,
or QA outcome enters the gate feature.

The threshold maximizing Calibration option micro-F1 is selected. Exact ties
prefer the lower threshold and therefore the simpler, higher-coverage policy.
The selected gate advances only if it exceeds ungated B6 by at least 0.005
micro-F1 and does not increase negative transfer from B3. Otherwise ungated B6
is retained. Validation and Test remain untouched.
