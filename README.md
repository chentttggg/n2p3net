# N2P3-Net

Accuracy-first oddball P300 decoding research framework. The product endpoint
is adult BrainSync 9-choice subject-macro `hit@R >= 0.90` after a fixed labelled
calibration budget, on later independent decisions whose target digit was
unknown during calibration.

## Status

- Compact MS-EEGNet-style trunk; the linear `full_unfold` readout with K35 is
  the provisional engineering default (`ms_flatten + K65` remains the explicit
  MS-EEGNet baseline control).
- GTN development benchmark (245 subjects, child/3-channel, one selection per
  subject): the frozen all-evidence count-neutral endpoint reaches **71.43%**
  9-choice digit-selection accuracy (trial AUC 0.694). GTN is a development
  cohort; this is a development result, not a confirmatory or deployment claim.
- Feature-domain transfer research (2026-09): identity-initialized per-domain
  residual feature adapters, class-conditional source alignment, a unified
  candidate-evidence decision core, and leakage-safe domain LODO/ceiling folds.
- The living evidence ledger, preregistered single-axis arms, and promotion
  gates are in `doc/research_program.zh.md`. No deployment champion is
  confirmed yet; read it before running experiments.

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
Repeated BIDS raw BrainSync sessions are accepted; each active v4 session is one
decision and `block_id` is scheduling metadata only. Only `completed` sessions
enter model input. Every block must contain declared complete random-permutation
cycles over all nine candidates; this acquisition balance does not itself prove
target-switch calibration. Target-sequence behavior remains an explicit evaluation
policy. The paper-aligned 0.5 Hz contract is a named anchor, not a fallback default.
Use the adapter to preserve the frontend's raw recording boundary while applying preprocessing:

```powershell
.\.venv\Scripts\python.exe experiments\prepare_brainsync_cache.py `
  --session-dir "D:\path\to\calibration-session" `
  --session-dir "D:\path\to\test-session" `
  --output "D:\path\to\epochs.npz" `
  --invalid-session error
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

The adapter validates the v4 session plus BIDS 1.11 raw dataset, reads stimulus
and retained-rest rows from `events.tsv`, derives labels only from the confirmed
post-experiment target digit, and reads channel geometry from `electrodes.tsv`
plus `coordsystem.json`. Filtering runs on the intact continuous recording.
Epochs intersecting a half-open rest interval are excluded when the cache is
generated; event times after rest are never shifted. `--invalid-session skip`
is available only as an explicit batch policy and records every skipped session
and error in cache provenance.

Historical v1/v2/v3 BrainSync sessions may remain as acquisition evidence, but they
are not accepted by the active loader and must not be mixed with v4 BIDS raw.
The removed v2 source contract is physically isolated in
`frozen/brainsync_rest_removed_adapter_bb70dfe.source-only.tar.gz`; its adjacent
manifest records the source commit and archive hash.

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

Raw, active-derived, legacy-derived, frozen-code, and evidence layers are defined
in `doc/data_layers.zh.md`. Active loaders contain no legacy schema flags,
checkpoint aliases, or downgrade paths.

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
