# V16 Generation Adaptation and Evidence-Gating Development Protocol

## Status and purpose

V16 is a development-only exploration intended to address the principal
remaining limitation identified by V10--V15: historical retrieval improved
automated report-reference consistency, but did not establish a reliable
pathology-label improvement over target-image-only generation. V16 tests
whether task adaptation and explicit separation of target observations from
historical analogies improve that transfer.

This protocol is prospective for the V16 development run. It does not amend,
replace, or reinterpret the frozen V10/V11 confirmation studies. V10 Test is
not available for V16 fitting, model selection, threshold selection, prompt
selection, early stopping, case inspection, or promotion decisions.

The target research task remains:

> Given a new patient's chest radiograph, clinical indication, and question,
> retrieve analogous image--report cases from other cases and produce a
> question-specific answer without observing the target patient's report.

All pathology targets and answer references in this protocol are derived from
source reports. They are automated report-reference consistency targets, not
radiologist-adjudicated diagnostic labels or clinical truth.

## Data partitions and leakage controls

V16 uses the existing V10 duplicate-cluster-disjoint partition:

- Train: 2,510 cases for fitting trainable parameters;
- Calibration: 383 cases for early stopping, threshold calibration, and
  selection among predeclared variants;
- Validation: 384 cases for the final V16 development comparison;
- Test: prohibited for all V16 development activity.

Only case-ID and duplicate-cluster disjointness are claimed. Patient-level
independence cannot be verified from the processed OpenI identifiers.

For every training target, the historical bank excludes the target case and
every case in the target's duplicate cluster. The same exclusion rule is used
when constructing validation inputs. Target reports and reference answers are
used as supervision only in the fitting split; they are never included in an
inference prompt.

The V16 training and evaluation manifests must record:

- sorted case-ID fingerprints for Train, Calibration, and Validation;
- duplicate-cluster fingerprints;
- source and RadGraph artifact hashes;
- historical-bank exclusion counts;
- the exact retrieval ranking artifact used for each condition;
- the exact prompt and answer-template version;
- model, processor, dependency, and adapter hashes.

## V16 conditions

The primary comparison matrix is:

| ID | Generator | History | Purpose |
|---|---|---|---|
| M0 | frozen MedGemma | none | original no-history baseline |
| M1 | frozen MedGemma | V12 retrieved Top-3 | current RAG baseline |
| M2 | frozen MedGemma | deterministic random history | negative history control |
| M3 | QLoRA MedGemma | none | adaptation-only effect |
| M4 | QLoRA MedGemma | V12 retrieved Top-3 | adapted RAG candidate |
| M5 | QLoRA MedGemma | deterministic random history | adapted negative control |
| M6 | QLoRA MedGemma | gated retrieved facts | target/history separation candidate |

M0--M2 are evaluation references. M3--M6 are developed only from Train and
Calibration and are selected before reading Validation outcomes. V12 retrieval
is held fixed during the first V16 generation experiments so that generation
adaptation is not confounded with another ranking change.

## Training examples

The main supervised target is the report section appropriate to the question:

- Findings question: source Findings section;
- Impression question: source Impression section;
- the source-derived acute proxy is excluded from the primary training target
  and may be used only as a clearly labelled secondary analysis.

Each Train case contributes the following predeclared context variants:

1. no historical context;
2. V12 retrieved historical context;
3. deterministic random historical context.

The target answer is always the target case's section answer. The random-history
examples therefore teach the model to ignore unrelated historical statements,
rather than rewarding generic report copying. Random assignments are generated
from the Train bank with a fixed seed, exclude the target and its duplicate
cluster, and are not selected after inspecting generated answers.

The first implementation uses supervised fine-tuning. Direct preference
optimization is optional and can only be opened if the SFT candidate improves
Calibration without a contract or provenance regression. DPO is not part of the
primary V16 claim unless its own protocol is committed before training.

## QLoRA configuration

The initial smoke-test configuration is fixed as follows:

```text
base model: google/medgemma-1.5-4b-it
quantization: 4-bit NF4, double quantization
compute dtype: bfloat16 when supported
LoRA rank: 8
LoRA alpha: 16
LoRA dropout: 0.05
target modules: attention projections and language MLP projections supported by the model
micro batch size: 1
gradient accumulation: 16
gradient checkpointing: enabled
maximum sequence length: 768 tokens initially
maximum epochs: 3
learning rate: 2e-4
weight decay: 1e-4
optimizer: paged AdamW 8-bit when available
seed: 1616
decoding: greedy, temperature 0
```

The model's vision tower remains frozen in the first V16 run. If the smoke
test fails for memory or processor reasons, the allowable technical changes
are lower sequence length, lower LoRA rank, or lower accumulation; such a
change must be recorded before the full run. It is not permissible to change
the configuration after seeing Validation scores.

The full run is preceded by a 200-example smoke test that verifies:

- image and text processor compatibility;
- labels mask the prompt and supervise only the answer tokens;
- gradients reach LoRA parameters;
- no frozen parameter receives a gradient;
- loss is finite;
- peak memory and one-step latency are recorded;
- an adapter checkpoint can be saved and reloaded.

## Target-state head and evidence gating

The existing V13 target-image concept head is not inserted as an unverified
diagnosis line in the generation prompt. V13 showed that direct concept prompt
injection can reduce QA quality. V16 may use a target-state head only as an
internal evidence filter.

The allowed target state is a 14-observation CheXbert-derived vector with
values represented internally as positive, negative, uncertain, or unavailable.
Thresholds are selected on Calibration only. A historical fact is retained for
M6 when it is relevant to the question and is compatible or unresolved with
the target-state signal. A fact that is strongly contradictory is withheld and
recorded in an audit trace. The gate never converts a historical fact into a
target-patient fact.

M6 must retain, for every selected fact:

- historical case ID;
- report section;
- fact or sentence provenance ID;
- gate decision and reason;
- source text hash.

## Output contract

V16 uses a two-stage output contract:

### Stage 1: target answer

The model produces only a short target-patient answer and uncertainty state.
It must not generate citation IDs or historical support prose in this stage.

### Stage 2: deterministic support assembly

Python code attaches only facts that were present in the retrieved and gated
evidence list. It adds case IDs, section provenance, evidence status, and the
no-reliable-history flag deterministically. The model cannot invent citations.

The structured representation is compact and bounded. Complete-sentence
normalization may remove malformed trailing text, but it may not invent a
missing answer or repair unsupported clinical content.

## Development selection and stopping rules

Training uses Train. The internal early-stopping split, if needed, is a fixed
case-level subdivision of Train created before training. Calibration is used
once for selecting among the predeclared SFT checkpoint, output policy, target
state threshold, and evidence-gating variant. Validation is then used for the
final V16 development comparison. No Validation result may be used to revise
the V16 model.

The primary candidate is M6. It is promoted over M1 only when all of the
following hold on the predeclared Calibration selection:

- pathology Macro-F1 does not decrease;
- RadGraph Complete F1 does not decrease;
- Token-F1 is not materially lower than M1;
- random-history performance remains below retrieved-history performance;
- answer-contract validity is at least 0.99;
- provenance validity is 1.00;
- token-ceiling rate does not materially increase.

If several candidates pass, choose in this order:

1. higher pathology Macro-F1;
2. higher RadGraph Complete F1;
3. higher Token-F1;
4. lower contradiction proxy rate;
5. lower input tokens;
6. simpler model and gate.

If no candidate passes, V10 remains the primary thesis evidence and V16 is
reported as a negative or mixed development extension.

## Evaluation metrics

The main objective is not an arbitrary absolute score. It is consistent paired
improvement over both no-history and random-history controls.

Primary development metric:

```text
CheXbert report-reference Macro-F1-14
```

Secondary metrics:

- CheXbert Micro-F1-14;
- CheXbert Macro/Micro-F1 on the five-observation subset;
- exact-set accuracy, reported with prevalence context;
- F1RadGraph entity, relation, and complete scores;
- Token-F1;
- positive-label precision and Hamming agreement;
- contradiction and omission proxies;
- answer-contract validity;
- provenance validity;
- token-ceiling rate;
- input tokens, output tokens, latency, and peak memory.

All model comparisons use paired case-grouped bootstrap intervals with 10,000
resamples. The primary comparisons are M6 minus M3 and M6 minus M5. M4 minus
M3 and M4 minus M5 are secondary. A confidence interval crossing zero is
reported as numerical improvement only, not confirmed superiority.

## Promotion boundary and interpretation

The strongest possible V16 result would show:

```text
M6 > M3  (adapted RAG beats adapted no-history)
M6 > M5  (relevant history beats random history)
pathology Macro-F1 improves with CI above zero
RadGraph and Token-F1 do not regress
```

Even that result remains automated, same-source report-reference consistency.
It does not establish clinical diagnostic accuracy, patient benefit, safety,
external validity, or patient-level independence.

If only Token-F1 improves, the result is described as lexical/report-style
improvement. If only the target-state head improves, it is described as an
automated image-to-report-label development result. If all QA metrics worsen,
the negative result is retained and no post-hoc tuning is allowed.

## Reproducibility and repository boundary

V16 code, adapters, local generation rows, and model caches remain separate
from frozen V10/V11 artifacts. Large local data and model outputs are not
committed. Public summaries contain hashes, counts, configurations, and
aggregate metrics only.

The protocol must be committed before the V16 training dataset is generated.
The development decision record must be committed before any final V16
confirmation design is considered. No new confirmation case IDs may be
instantiated from this protocol alone.

## Claim boundary

V16 tests task-adapted multimodal RAG. It does not claim that a retrieved
historical case is the target patient, that a report-derived label is a clinical
gold standard, or that the resulting dashboard can be used for autonomous
diagnosis. Independent physician review and external patient-level validation
remain Future Work.
