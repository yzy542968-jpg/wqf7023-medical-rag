# V10 Retrieval Calibration Protocol

Status: frozen after retrieval and evidence-policy development, before reading
Calibration outcomes. V10 Test has not been run.

The frozen R5-attention ensemble is applied to every technically eligible
Calibration case and all three retrieval question roles. Logistic regression
with L2 regularization and seed 7046 predicts whether the Top-1 historical case
has offline combined qrel gain at least 0.50.

Features are fixed as:

- raw R5 ensemble Top-1 score;
- Top-1 minus Top-2 R5 score margin;
- fraction of BM25, image-image, and image-report component Top-1 choices that
  agree with the final Top-1;
- variance of five R5 seed scores at final Top-1;
- mean question-relevance score and redundancy of E0 selected evidence;
- target view count;
- findings, impression, and acute question indicators.

Calibration reports apparent-fit Brier score, 10-bin ECE, AUROC, and the full
risk-coverage curve. Thresholds are fixed for 100%, 90%, 80%, 70%, and 50%
Calibration coverage. G3 uses the prespecified 80% coverage threshold; Test
cannot choose another operating point. Low confidence suppresses historical
evidence but does not suppress target-image answering.

Calibration concerns offline retrieval relevance and selective coverage. It
does not establish diagnosis correctness, clinical safety, or calibrated
patient risk.

