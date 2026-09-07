# N2P3-Net

Public ERP pretraining and participant-specific calibration for BrainSync
nine-choice P300 decoding. The acquisition layout is seven scalp EEG channels
(`Fz,Cz,P3,Pz,P4,PO8,Oz`) and one independent EOG channel.

Read [the active research guide](doc/research_program.zh.md) first, then the
[verified technical map](doc/technical_structure.zh.md) and
[document index](doc/README.md). Historical experiment results are development
evidence. The integrated workstation's block-completion prediction path has
been tested with declared mock data; real-device decoding accuracy is not validated.

## Current Chain

1. Validate raw source data, participant identity, physical channels and timing.
2. Build attested causal epoch caches and explicit binary ERP source views.
3. Train a supervised N2P3Net source model with participant exclusion.
4. Calibrate on the participant's declared earlier labelled recordings.
5. Score independent later EEG and aggregate evidence over the declared candidates.

The default research path uses ordinary supervised source training and retained
source-head adaptation. EA, AS-MMD, Split-BN, source selection and other existing
mechanisms remain explicit research controls, with their own fit scopes.

The requested product budget is 9 calibration targets x 10 blocks x 45 flashes
= 4,050 flashes, followed by a new 45-flash test decision. Each balanced block
contains five repetitions of all nine candidates, so the product endpoint is
`hit@R5`. Trial AUC is a separate diagnostic. See the prospective
[collection plan](configs/brainsync_collection_plan.json).

The integrated BrainSync workstation (a separate acquisition application)
now provides calibration record selection, internal neural recipe selection,
frozen EOG correction, personal model save/load and automatic prediction after
a new 45-flash block. The target is confirmed only after prediction. The backend
reuses the existing checkpoint, preprocessing, adaptation and candidate modules.
This is completed-recording inference, not per-SDK-chunk streaming prediction.

This repository contains the model backend, not the acquisition application or
its SDK. Local launch paths and operation are described in the
[workstation guide](doc/workstation.zh.md).

## Environment

Python 3.11 or later is required. From the repository root, create the environment
with [uv](https://docs.astral.sh/uv/). `pyproject.toml` is the dependency authority and
`uv.lock` records the resolved environment. See the [engineering guide](doc/engineering.zh.md)
for installation, validation and CPU/CUDA behavior.

```powershell
uv sync --locked --extra signals --extra baselines --extra dev
.\.venv\Scripts\python.exe -m ruff check src experiments tests
.\.venv\Scripts\python.exe -m pytest
git diff --check
```

On Linux, use `.venv/bin/python`. For an unlocked pip installation, create a
virtual environment with `python -m venv .venv`, then run its Python with
`-m pip install -r requirements.txt`. Accelerator wheel selection and validated
hardware are separate from dependency resolution; see the engineering guide.

Tests should exercise the changed interface and counterexamples. A full-suite
pass does not establish scientific validity or real-device performance.

## Entrypoints

| Task | Script |
|---|---|
| Public EEG / candidate cache | `prepare_eeg_dataset.py`, `prepare_bnci2014_008_candidate.py`, `prepare_bi2014a_candidate.py` |
| Source channel/reference/binary view | `adapt_eeg_domain.py`, `prepare_multidomain_source.py` |
| Supervised source pretraining | `run_pretrain_supervised.py` |
| Public binary calibration comparison | `run_binary_subject_transfer.py` |
| Whole-dataset held-out transfer | `run_cross_dataset_transfer.py` |
| External Mainsah source admission audit | `audit_mainsah_s2.py` |
| Completed BrainSync recordings to cache | `prepare_brainsync_cache.py` |
| Calibration-only EOG regression | `fit_brainsync_eog.py` |
| Offline BrainSync later-decision evaluation | `run_brainsync_cross_decision.py` |
| Personal calibration / model inspection / unlabelled prediction | `brainsync_model_task.py` |
| Candidate experiment orchestration | `run_candidate_promotion_matrix.py`, `run_candidate_result_batch.py` |
| Generic LOSO / sensitivity | `run_eeg_loso.py`, `run_n2p3_sensitivity.py` |

Scripts live under `experiments/`; use their `--help` for actual accepted
arguments. Source training and evaluation must identify the code snapshot that
executed each stage. A new evaluation does not rename an old source checkpoint.

```powershell
.\.venv\Scripts\python.exe experiments\prepare_brainsync_cache.py `
  --session-dir "D:\path\to\completed-session" `
  --output "D:\path\to\epochs.npz" --invalid-session error
```

The default causal input contract is 128 Hz, volts, continuous 0.1-30 Hz forward
IIR filtering, `[-200,1200)` ms and mean-only baseline correction. Native raw
BrainSync acquisition remains 250 Hz. Reference changes, missing channels and
EOG cleaning are explicit, provenance-bound operations; EOG is never a scalp
classifier input. See [the input contract](doc/input_contract_math.zh.md).

## Data And History

Raw datasets, cache/run directories and frozen evidence have different roles;
see [the workspace map](doc/technical_structure.zh.md#目录与数据层级).
GitHub contains source, tests, guidance and evidence manifests. Historical archive
payloads remain local; see [archive distribution](frozen/README.md). A clone does
not include datasets, pretrained checkpoints, run outputs or the BrainSync app.
Superseded planning documents, examples and the incompatible legacy `patch.sh`
were physically frozen before removal. Their hashes and original paths are in
the [audit snapshot](frozen/technical_audit_before_20260906.manifest.json).
The [technical audit](doc/evidence/technical_audit_20260906.zh.md) records fixes, tests
and remaining integration work. Existing data and measured scores retain their
original provenance.

## Dashboard

```powershell
.\.venv\Scripts\python.exe experiments\dashboard_server.py `
  --port 8812 --bind 127.0.0.1 --directory experiments
```

Open [the local dashboard](http://127.0.0.1:8812/dashboard.html). Only runners emitting the dashboard
`progress.jsonl`/`record.json` protocol appear there; completed JSON results from
other entrypoints are not live dashboard telemetry.
