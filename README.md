# N2P3-Net

Accuracy-first oddball P300 decoding research framework. The product endpoint
is adult BrainSync 9-choice subject-macro `hit@R >= 0.90` after a fixed labelled
calibration budget, on later independent decisions whose target digit was
unknown during calibration.

GTN is a child/3-channel, one-selection-per-subject development benchmark. Its
same-selection labelled prefix/suffix path is an oracle proxy, not an
unknown-number calibration estimate. BI2014a is used for the legal mechanism
test: early known character decisions -> later unknown character decisions.
Linear `full_unfold` is the adopted N2P3-Net readout and K35 is the provisional
engineering default; `ms_flatten + K65` remains the explicit MS-EEGNet baseline.
On the current all-evidence, count-neutral endpoint, the unchanged v4 K35
checkpoint reaches 71.43% versus 68.30% for K65. The paired difference is
+3.13 points (95% CI +0.41 to +5.85), but seed direction reverses and the
eight-contrast Holm result is not significant. K35 therefore remains a smaller
development default, not a confirmatory or deployment champion.

Legacy causal records used a zero-state forward IIR and reused the same suffix
for recipe selection; their numeric rankings have been removed from current
guidance. Four steady-state causal GTN caches have now been rebuilt and
independently audited. The current best audited fixed-budget Z0 baseline is
0.1 Hz/1200 ms, source QC 100 uV, with `hit@5` coverage `230/245` and operational
accuracy `0.543`; it remains far below 0.90. The kernel experiment now uses all
245 subjects. For unequal candidate counts, the current development default uses
every available trial and compares candidate means (`count_power=0`); balanced
truncation and raw sum remain compatibility endpoints. A 24-checkpoint,
30-epoch end-to-end listwise fine-tune was completed and did not improve this
frozen-backbone mean baseline, so that training recipe is not adopted. Read
`doc/research_program.zh.md`
before running experiments.

## Environment

Use the repository-local Python environment. Bootstrap pip first when the
bundled environment does not include it:

```powershell
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install -e ".[signals,baselines,dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

## CUDA Training Defaults

All supervised models built from `DeepConfig` default to fused Adam and
`torch.compile(mode="reduce-overhead")` on CUDA. This covers N2P3Net and every
registered pooling head, EEGNet, EEG-Inception, and EEG Conformer through the
LOSO, sensitivity, and model-factory entrypoints. CPU and XPU retain the same
requested config in audit metadata but automatically execute eager, non-fused
training.

Use `--no-fused-adam --compile-mode none` for the matched eager ablation. The
64-subject / 8-fold RTX 5090 check reduced wall time from 72.161 s to 46.875 s;
full settings, requested/effective record fields, and cold-start boundaries are
documented in `doc/device-portability.md`.

The LOSO runner requires the signal and baseline extras. The living research
guide is `doc/research_program.zh.md`; frozen ablations and evidence are physically
isolated in `frozen/research_evidence_through_20260901-d1db8e4.tar.gz`.
The current concise research briefing is `doc/research_status_report_20260831.zh.md`.
The current project-wide critical review is `doc/project_critical_review_20260901.zh.md`.
`doc/constitution.md` and `doc/blueprint.md` contain stable engineering principles.

## Data Contract

Executable profiles are defined in `src/data/contract.py`. The single active
default is 128 Hz, 0.1-30 Hz and `[-200,1200) ms` (`179` samples). Offline
analysis uses zero-phase IIR; chronological analysis uses forward IIR with
`causal_iir_initial_state=steady_state_first_sample`; legacy zero-state forward
caches are rejected. BrainSync acquisition and raw event indices remain at the
device-native 250 Hz; only the derived model tensor is resampled. BrainSync,
BI and GTN chronological builders all derive from this same causal contract.
Repeated BrainSync sessions are accepted; block/selection markers remain distinct
target-changing decisions. The paper-aligned 0.5 Hz contract is a named anchor,
not a fallback default.
Use the adapter to preserve the frontend's raw recording boundary while applying preprocessing:

```powershell
.\.venv\Scripts\python.exe experiments\prepare_eeg_dataset.py brainsync `
  --session-dir "D:\path\to\calibration-session" `
  --session-dir "D:\path\to\test-session" `
  --output "D:\path\to\epochs.npz"
```

Cross-dataset checkpoints remain fail-closed. The implemented deterministic
domain route is an explicit common channel subset followed by CAR on that same
subset in every domain:

```powershell
.\.venv\Scripts\python.exe experiments\adapt_eeg_domain.py `
  --dataset-cache source.npz --target-channels "Cz,P3,Pz,P4,Oz" `
  --output source_common_car.npz
```

Source and target caches must then share preprocessing, channel order, CAR
provenance and a newly trained checkpoint. Existing 128-sample BI/BNCI caches
cannot be renamed or reused; they must be rebuilt from raw data. Missing channels
are not padded.

The adapter reads `recording.path`, filters onset `recording_marker` rows from
`events/events.jsonl`, derives labels from the confirmed target digit, uses
`montage.channel_positions_m` when present, and applies the explicitly selected
preprocessing profile.

All standard ingress paths execute a half-open `[-200,0) ms` per-trial,
per-channel mean baseline correction and record `baseline_mode=mean_only`.
Versioned QC caches contain only fold-independent epoch statistics; thresholds
remain source/calibration-fold parameters. Target-prefix QC is an accuracy
ablation and is off in zero-shot. Checkpoints bind ordered channels, reference,
preprocessing, cache identity and training subjects. Legacy zero-state, 2 Hz/800 ms,
GTN-v4 and BI candidate-v1/v2 caches are rejected by the active contract. Their
audit snapshot is compressed under `frozen/`. Current BI/BNCI/GTN source caches
and checkpoints must be rebuilt under `p300_single_subject_causal_v3` before new
promotion experiments.

The discrete-time equations, physical receptive-field convention, and
counterexamples are documented in `doc/input_contract_math.zh.md`.

## Dashboard

The training runner writes the dashboard protocol beside each model under
`experiments/runs/<run>/<model>/`: flushed `progress.jsonl`, per-fold epoch
JSONL, and the final `record.json`. Start the local dashboard with:

```powershell
.\.venv\Scripts\python.exe experiments\dashboard_server.py `
  --port 8812 --bind 127.0.0.1 --directory experiments
```

Open `http://127.0.0.1:8812/dashboard.html`. To start the dashboard together
with a training command, use the companion runner:

```powershell
.\.venv\Scripts\python.exe experiments\run_with_dashboard.py -- `
  .\.venv\Scripts\python.exe experiments\run_eeg_loso.py `
  --dataset-cache experiments\cache\dataset.npz --run-name smoke --max-folds 2
```

The previous cloud workflow is also available through `open_dashboard.ps1` or
`open_dashboard.cmd`; it creates the SSH tunnel and opens the same page without
affecting the remote training process.
