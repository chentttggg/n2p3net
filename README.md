# N2P3-Net

Performance-first oddball P300 decoding research framework. The current scope
is a validated common data contract and evaluation foundation before committing
to one model family.

## Environment

Use the repository-local Python environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

Install `.[baselines]`, `.[signals]`, and device-specific PyTorch plus
`.[train]` only when beginning those phases. The research contract is in
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
