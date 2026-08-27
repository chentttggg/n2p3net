# N2P3-Net

Performance-first oddball P300 decoding research framework. The production
candidate is the compact MS-EEGNet-style N2P3-Net with latency-marginal
contrast pooling; promotion remains conditional on matched held-out ablations.

## Environment

Use the repository-local Python environment. Bootstrap pip first when the
bundled environment does not include it:

```powershell
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install -e ".[signals,baselines,dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

The LOSO runner requires the signal and baseline extras. The research contract is in
`doc/constitution.md`, `doc/blueprint.md`, and `doc/roadmap.md`.

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
