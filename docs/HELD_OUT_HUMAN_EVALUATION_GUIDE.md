# Held-Out Human Evaluation Guide

## Recommended Rating Interface

Launch the independent rating application from the project root:

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run human_evaluation_app.py --server.port 8502
```

Open `http://localhost:8502`, select `V1 held-out stress test`, and score one sample at a time. The application does not load the system-key file and saves directly to the frozen blinded CSV. The reviewer must not open the key until all ratings are complete.

## Files

Give the reviewer only:

```text
experiments/final_optimized/human_evaluation/held_out_blinded_human_evaluation_36.csv
```

Do not give the reviewer:

```text
experiments/final_optimized/human_evaluation/held_out_blinded_human_evaluation_key.csv
```

The key identifies which blinded response came from each system. Open it only after all ratings are complete and frozen.

## Sampling Design

- 36 questions from 36 distinct held-out cases.
- 12 findings questions, 12 impression questions, and 12 summary questions.
- Four answers per question: final adaptive agent, previous Hybrid agent, BM25 agent, and LLM-only.
- Response labels A-D are independently shuffled for every question using seed 7023.

## Rating Rubric

### Correctness: 1-5

| Score | Meaning |
|---:|---|
| 1 | Incorrect or contradicts the reference evidence |
| 2 | Mostly incorrect with a small correct element |
| 3 | Partly correct but incomplete or imprecise |
| 4 | Correct with only a minor omission or wording issue |
| 5 | Correct, complete, and appropriately concise |

### Evidence grounding: 1-5

| Score | Meaning |
|---:|---|
| 1 | Major unsupported clinical claims |
| 2 | Several unsupported or mismatched claims |
| 3 | Mixed support or uncertain grounding |
| 4 | Substantially supported with a minor unsupported detail |
| 5 | Every material claim is supported by the gold-case evidence |

Use `gold_case_evidence`, not only surface similarity to `reference_answer`, for this rating. The same target-case findings and impression are shown for all four blinded responses. A response that is internally faithful to a different retrieved patient but unsupported by this gold case must receive a low grounding score.

### Potentially harmful: 0-1

- `0`: no clinically consequential contradiction or invented finding.
- `1`: contains a contradiction, invented diagnosis, wrong location/severity, or confident unsupported claim that could mislead a reader.

### Best response

Enter `A`, `B`, `C`, `D`, or `tie`. Judge the best overall balance of correctness, grounding, concision, and appropriate abstention.

## Review Procedure

1. Read the question, reference answer, and full gold-case evidence.
2. Score all four responses independently before choosing the best response.
3. Treat an appropriate insufficiency statement as safer than a fabricated answer, but do not award full correctness if the reference clearly contains an answer.
4. Record brief notes for contradictions, unsupported findings, or difficult cases.
5. Complete all 36 rows without consulting the system key.
6. Freeze the reviewed file before decoding system identities.

## Analysis Plan

After unblinding, report by system:

- mean and median correctness;
- mean and median evidence grounding;
- harmful-response proportion;
- best-response win or tie rate;
- paired comparisons using the same 36 questions;
- inter-rater agreement if a second reviewer is available.

Human results are confirmatory external validation. They must not be used to change retrieval, prompts, or checker thresholds after the held-out test has been opened.
