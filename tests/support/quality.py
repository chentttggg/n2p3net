"""Synthetic quality builders for contract tests."""

from __future__ import annotations

from dataclasses import replace

import mne
import numpy as np

from data.artifact import FoldLocalArtifactPolicy
from data.brainsync import load_brainsync_sessions
from data.brainsync_contract import validate_brainsync_bids_session
from data.continuous_qc import ContinuousQCSpec, fit_continuous_qc
from data.epochs import materialize_dataset_identity
from data.recovery import fit_epoch_state_recovery
from research.contracts import semantic_sha256
from tests.support.brainsync import write_bids_session
from tests.support.datasets import epoch_dataset
from transfer.quality import apply_epoch_quality, fit_epoch_quality


def quality_pair():
    dataset = personal_dataset()
    amplitude = np.linspace(0.8, 1.2, dataset.n_epochs)[:, None, None] * 1e-6
    waveform = np.sin(np.arange(dataset.n_times) / 8.0)[None, None, :]
    values = np.broadcast_to(amplitude * waveform, dataset.X.shape).astype(np.float32).copy()
    dataset = replace(dataset, X=values)
    policy = FoldLocalArtifactPolicy(min_clean_epochs=2)
    fitted = fit_epoch_quality(dataset, np.ones(dataset.n_epochs, dtype=bool), policy)
    return dataset, apply_epoch_quality(dataset, fitted), policy


def masked_bids_calibration(tmp_path, *, drop_all_targets=False, continuous=True):
    from data.brainsync import load_brainsync_sessions

    roots = [
        write_bids_session(
            tmp_path / f"cal-{index}",
            session_id=f"cal-{index}",
            target=target,
            started_utc=f"2026-09-06T08:{index * 3:02d}:00Z",
            blocks=(tuple(range(1, 10)) * 5,),
        )
        for index, target in enumerate((1, 3, 5))
    ]
    dataset = load_brainsync_sessions(roots)
    if continuous:
        first = declare_continuous_quality(load_brainsync_sessions([roots[0]]))
        dataset = replace(
            dataset,
            provenance={
                **dataset.provenance,
                "continuous_artifact_processing": first.provenance[
                    "continuous_artifact_processing"
                ],
            },
        )
    observed = np.ones(dataset.X.shape[:2], dtype=bool)
    if drop_all_targets:
        observed[dataset.y == 1, 0] = False
    else:
        observed[np.arange(dataset.n_channels) * 3, np.arange(dataset.n_channels)] = False
    values = dataset.X.copy()
    values[~observed] = 0.0
    return replace(dataset, X=values, trial_channel_mask=observed, qc_features=None)


def personal_training_arrays():
    values = np.random.default_rng(91).normal(size=(60, 3, 100)).astype(np.float32) * 1e-6
    return values, np.repeat("P001", 60), np.repeat(["session-a", "session-b", "session-c"], 20)


def personal_dataset():
    import pandas as pd

    from data.epochs import materialize_dataset_identity
    from data.events import observed_only_timeline

    values, subjects, sessions = personal_training_arrays()
    dataset = replace(
        epoch_dataset(),
        X=values,
        y=np.arange(len(values)) % 2,
        subject_ids=subjects,
        metadata=pd.DataFrame({"session": sessions, "block_id": np.tile(np.repeat([1, 2], 10), 3)}),
        event_timeline=observed_only_timeline(
            dataset_id="synthetic_p300",
            subject_ids=subjects,
            stimulus_ids=np.arange(len(values)),
            onset_times_s=np.arange(len(values)) * 2.0,
            evidence_available_times_s=np.arange(len(values)) * 2.0 + 0.8,
            session_ids=sessions,
            run_ids=sessions,
            group_ids=sessions,
        ),
    )
    materialize_dataset_identity(dataset)
    return dataset


def fit_raw_quality(root, *, flatline_min_duration_s=5.0, include_eog=False):
    session = validate_brainsync_bids_session(root)
    dataset = load_brainsync_sessions([root])
    raw = mne.io.read_raw_edf(session.bids.raw_path, preload=True, verbose=False)
    raw.set_channel_types(
        dict(zip(raw.ch_names, [kind.lower() for kind in session.bids.channel_types], strict=True))
    )
    provenance = {
        "participant_key": materialize_dataset_identity(dataset).authority_keys()[0],
        "source_reference": session.bids.source_reference,
        "calibration_session_ids": [session.session_id],
        "calibration_end_timestamp_s": session.ended_timestamp_s,
    }
    names = tuple(raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, eog=include_eog, exclude="bads"))
    recovery = fit_epoch_state_recovery(raw, preprocessing=dataset.preprocessing, channels=names,
        fit_sample_mask=np.ones(raw.n_times, bool), tolerance_v=1e-3, inflation=2.,
        warmup_samples=0, probe_step_samples=max(351, int(raw.n_times)//4), fit_provenance=provenance)
    # Lifecycle tests isolate technical-fault gates with a generous declared
    # boundary tolerance; separate numerical tests exercise the 1 uV bound.
    return fit_continuous_qc(
        raw,
        spec=ContinuousQCSpec(
            12.0,
            1e-7,
            10,
            channel_local=True,
            include_eog=include_eog,
            minimum_channel_fraction=0.5,
            flatline_min_duration_s=flatline_min_duration_s,
        ),
        fit_sample_mask=np.ones(raw.n_times, dtype=bool),
        epoch_recovery=recovery,
        fit_provenance=provenance,
    )


def declare_continuous_quality(dataset, *, channel_local=True, minimum_channel_fraction=1.0):
    identity = materialize_dataset_identity(dataset)
    raw = mne.io.RawArray(
        np.random.default_rng(83).normal(0.0, 1e-6, (dataset.n_channels, 1000)),
        mne.create_info(dataset.channel_names, dataset.preprocessing.sfreq, "eeg"),
        verbose=False,
    )
    fitted = fit_continuous_qc(
        raw,
        spec=ContinuousQCSpec(
            12.0,
            1e-7,
            10,
            channel_local=channel_local,
            minimum_channel_fraction=minimum_channel_fraction,
        ),
        fit_sample_mask=np.ones(raw.n_times, dtype=bool),
        fit_provenance={
            "participant_key": identity.authority_keys()[0],
            "calibration_session_ids": sorted(dataset.metadata["session"].unique()),
        },
    )
    record = fitted.record()
    return replace(
        dataset,
        provenance={
            **dataset.provenance,
            "continuous_artifact_processing": {
                "status": "applied",
                "model": record,
                "model_digest": semantic_sha256(record),
            },
        },
    )
