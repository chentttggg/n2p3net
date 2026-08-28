# N2P3-Net

Performance-first oddball P300 decoding research framework. BI2014a is now an
exploratory architecture screen: linear `full_unfold` has the best observed
mean AUC, but no model is a confirmatory champion. GTN chronological 9-choice
`hit@R` is the final promotion protocol. LMBC failed promotion only on the
BI2014a binary contract and remains a GTN/latency-stratified hypothesis.

> Update 2026-08-28: the later prior-free readout ablation
> (`doc/prior_free_unfold_result_20260828.zh.md`) puts `full_unfold` ahead of
> `ms_flatten` on both AUC and BACC, and ahead of EEGNet on AUC only. This
> registers `full_unfold` as a GTN candidate; it does not promote the default
> from one dataset. The BI2014a training-budget runs (`bi2014a_full_unfold_*`)
> are performance exploration only and are not promotion evidence.

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
guide is `doc/research_program.zh.md`; dated ablations remain historical evidence.
`doc/constitution.md` and `doc/blueprint.md` contain stable engineering principles.

## Data Contract

The shared P300 defaults are defined once in `src/data/contract.py`: 128 Hz,
2-30 Hz, `-200..800 ms`, and 128 samples. This restores the physical time
scales assumed by the MS-EEGNet 65/5/17-sample kernels. BrainSync acquisition
and raw event indices remain at the device-native 250 Hz; only the derived
model tensor is anti-aliased and resampled. Use the BrainSync adapter
to preserve the frontend's raw recording boundary while applying preprocessing:

```powershell
.\.venv\Scripts\python.exe experiments\prepare_eeg_dataset.py brainsync `
  --session-dir "D:\path\to\session" `
  --output "D:\path\to\epochs.npz"
```

The adapter reads `recording.path`, filters onset `recording_marker` rows from
`events/events.jsonl`, derives labels from the confirmed target digit, uses
`montage.channel_positions_m` when present, and applies the GTN window
(`-200..800 ms`, 128 samples at 128 Hz) by default.

All standard ingress paths execute a half-open `[-200,0) ms` per-trial,
per-channel mean baseline correction and record `baseline_mode=mean_only`.
Versioned QC caches contain only fold-independent epoch statistics; thresholds
remain outer-training-fold parameters. Mainline training fails closed on older
physical input contracts; historical result records remain audit evidence, but
their caches must be regenerated before reuse.

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
