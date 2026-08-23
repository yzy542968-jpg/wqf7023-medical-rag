# V10 R5 and Multi-view Integration Amendment

Status: frozen before combined R5-attention Validation evaluation.

R5 and learned view attention were selected independently under their frozen
development rules. This amendment defines their integration before inspecting
the combined outcome.

For each query case, every frozen attention seed produces an L2-normalized
weighted view embedding. Their arithmetic mean is L2-normalized and replaces
the normalized mean-view query embedding in the R5 image-image and
image-report components. BM25, question indicators, sentence/fact features,
candidate embeddings, R5 checkpoints, and all other inputs remain unchanged.
R5 is not retrained.

The combined R5-attention ensemble is accepted if its Validation nDCG@10 is no
more than 0.005 below the frozen R5-mean result. If degradation is at least
0.005, mean-view R5 remains primary and learned attention is reported only as
an image-retrieval diagnostic. This is a non-degradation guard, not a new
promotion search. Calibration and Test cannot change the rule.

