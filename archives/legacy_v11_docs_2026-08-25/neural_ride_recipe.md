# Neural-RIDE v11 recipe and entry-point contract

The single source of truth is `src/train/recipe.py`.
The operator-facing route map is [`routes.md`](routes.md).

## Route selection

There are two named recipes, with one explicit switch:

| `lambda_innovation` | recipe | status |
|---:|---|---|
| `0` | `neural_ride_v11_pcw_fail_closed` | canonical production route |
| `>0` | `neural_ride_v11_strict_past_research` | opt-in strict-past research route |

The GTN entry point resolves the recipe from this value. Do not describe a run as
strict-past unless its resolved record contains the research recipe and
`use_innovation_likelihood=true`. An explicit ERP reconstruction ablation is neither
a route switch nor a production classification input.

## Formal route evidence

The canonical model exposes:

- `pcw`: the only neural discriminative logit;
- `final = pcw`;
- `tau/sigma`: PCW localization outputs.

The strict-past research recipe additionally exposes `prequential_llr` and a
cross-fitted `prequential_contribution`. These remain zero/absent in canonical use.

There is no residual classifier or ERP subtraction path.
`prequential_llr` is the uncentered sum of per-time conditional log-density ratios;
training and audit use a separate NLL normalized by observed scalar count.

## Physical profile

The registered GTN profile uses:

- continuous-domain 0.1 Hz high-pass preprocessing;
- 256 Hz sampling;
- `[-200,+1200)` ms epochs;
- a real pre-stimulus baseline;
- fixed-coordinate trial baseline standardization;
- post-stimulus likelihood scoring, normally `[0,+800)` ms.

The model rejects inconsistent sample counts or physical axes. The strict-past claim
begins only after the complete baseline segment. It does not convert an acausal
upstream filter into an online-causal filter.

## Formal route defaults

| group | setting |
|---|---|
| recipe | `neural_ride_v11_pcw_fail_closed` |
| PCW | width 64, depth-3 TCN, BN, dropout 0.25 |
| tokenizer | bandpass initialization, no post normalization/activation |
| innovation | disabled; research opt-in is width 28, kernel 9, dilations 1/2/4/8/16 |
| density baseline | research only: fold-local VAR(32) plus online strict-past AR(1) |
| covariance | research only: diagonal plus optional rank-2 factor |
| ERP decoder | disabled |
| `lambda_pcw` | 0.3 |
| `lambda3` / `L_tau` | 0.0; only explicit research identification studies may enable it |
| `lambda_innovation` | 0.0; research recipe 1.0 |
| `lambda_recon` | 0.0 |
| variance warmup/ramp | 5/10 epochs |
| early-stop patience | 6 |

The strict-past research recipe keeps the PCW settings above and only changes the
explicit innovation likelihood switch and weight. Its density graph, subject roles,
candidate gates and fusion contract are not implicit production defaults; see
[`blueprint.md`](blueprint.md) sections 5--8.

The optional ERP decoder may be enabled only together with an explicit positive
reconstruction objective. It is an interpretation experiment and cannot feed the
production logit.

The density supports one fold-stable observation pattern. Runtime missing channels
are excluded before all conditional histories and scored through the corresponding
Gaussian marginal, but an observation pattern not represented by the fold profile
cannot activate audit or fusion. All-missing trials contribute exactly zero LLR and
zero density coverage.

## Dataset capabilities

`GTN_DIGIT_TASK` enables digit-set and repetition objectives because those labels
exist in GTN. `BINARY_ODDBALL_TASK` forces those objectives to zero. Dataset names
never silently switch model architecture.

The exact fold class ratio resolves `pos_weight`; a recipe value is only a nominal
default.

## Split contract

A fitted LOSO fold uses four disjoint roles:

- optimization subjects: gradients and all fold-local profiles;
- audit subjects: likelihood structure selection;
- validation subjects: epoch selection, threshold/calibration and fusion coefficient;
- outer test subject: final evaluation only.

If too few subjects exist for an untouched audit, or fewer than four validation
subjects are available for leave-one-subject-out fusion cross-fitting, likelihood
fusion remains disabled.

## Entry points

GTN:

```powershell
.\.venv\Scripts\python.exe experiments/run_n2p3net_gtn.py \
  --subjects 12 --max-folds 5 --epochs 24 --batch-size 256 \
  --run-dir tmp/acceptance --run-name issue2_v11_5fold_seed0
```

Generic binary oddball:

```powershell
.\.venv\Scripts\python.exe experiments/run_eeg_loso.py \
  --dataset-cache experiments/cache/bnci008_neural_ride_v8.npz \
  --models n2p3net,eegnet --epochs 30
```

Heterogeneous montage transfer:

```powershell
.\.venv\Scripts\python.exe experiments/run_multidataset_transfer.py \
  --dataset primary=experiments/cache/primary_epoch_dataset_v2.npz \
  --dataset auxiliary=experiments/cache/auxiliary_epoch_dataset_v2.npz \
  --main-domain primary --sampling balanced --acceptance \
  --output-dir experiments/runs/v11-transfer
```

Both paths must be prepared `n2p3net_epoch_dataset/2` caches. The GTN
`n2p3net_gtn_cache/2` scheduled-event cache is only for the GTN runner and is
not accepted by this heterogeneous EpochDataset entry point.

## Acceptance

Every fold record must preserve:

- subject identities for optimization, audit and validation roles;
- the complete prequential candidate table and checks;
- profile/runtime observed-channel counts and observation-pattern support;
- selected variant or fail-closed reason;
- validation-only coefficient and BCE before/after fusion;
- PCW, prequential and final outer metrics;
- fixed-K decision coverage and calibration source.

Five to eight outer folds are the minimum empirical review unit for this rewrite.
No record may call one passing fold or a green unit-test suite SOTA evidence.
