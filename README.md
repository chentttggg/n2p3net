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

The shared P300 defaults are defined once in `src/data/contract.py`: 250 Hz,
`-200..1200 ms`, and 350 samples for the general contract. The GTN contract
derives its 250 samples from the same 250 Hz source. Use the BrainSync adapter
to preserve the frontend's raw recording boundary while applying preprocessing:

```powershell
.\.venv\Scripts\python.exe experiments\prepare_eeg_dataset.py brainsync `
  --session-dir "D:\path\to\session" `
  --output "D:\path\to\epochs.npz"
```

The adapter reads `recording.path`, filters onset `recording_marker` rows from
`events/events.jsonl`, derives labels from the confirmed target digit, uses
`montage.channel_positions_m` when present, and applies the GTN window
(`-200..800 ms`, 250 samples at 250 Hz) by default.

All standard ingress paths currently preserve unbaselined epochs and record
`baseline_mode=none`; a requested transform fails closed until it has an
implemented, tested signal path. Versioned QC caches contain only
fold-independent epoch statistics; thresholds remain outer-training-fold
parameters.
