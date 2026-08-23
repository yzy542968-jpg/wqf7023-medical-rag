# V10 R5 and Multi-view Integration Decision Record

Status: combined retrieval policy frozen; V10 Test not run.

The shared V10 runtime first reproduced the frozen R5 mean-view Validation
nDCG@10 exactly (`0.3539093679342802`). This equality verifies that the reusable
runtime preserves the development feature and ranking semantics before the
view-policy substitution.

Replacing the mean-view query representation with the frozen five-seed
attention embedding produced Validation nDCG@10 `0.3585402599198427`, a
numerical difference of `+0.004630891985562491`.

The integration amendment required rejection only for degradation of at least
`0.005`. The attention integration is therefore accepted. Because the combined
gain itself is below `0.005`, it is described as a numerical improvement and
not as independently confirmed superiority of the combined system.

The final V10 retrieval policy is now:

```text
all target views
-> frozen five-seed attention aggregation
-> BM25 + MedSigLIP image-image + image-report components
-> question-aware sentence/RadGraph features
-> frozen five-seed R5 score ensemble
-> ranked historical cases
```

The local 1,128-row integration audit has SHA-256
`b91bcd65b1af8c629a6c865d269018cd23837508306f3d49417f3ce4497d2df4`.
Calibration may fit confidence on its separate partition, but no later stage
may alter the attention or R5 checkpoints, feature definitions, or ensemble
rules.
