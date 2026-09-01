from __future__ import annotations

import numpy as np
import pandas as pd

from data.channel import build_channel_identity
from data.epochs import EpochDataset, PreprocessingSpec, load_epoch_dataset, save_epoch_dataset
from data.events import ScheduledEventTimeline
from data.qc_features import compute_epoch_qc_features


def _dataset() -> EpochDataset:
    subjects = np.asarray(["s1", "s1", "s2", "s2"])
    n_epochs = len(subjects)
    identity = build_channel_identity(("Fz", "Cz", "Pz"), allow_missing_positions=False)
    X = np.random.default_rng(44).normal(size=(n_epochs, 3, 100)).astype(np.float32) * 1e-6
    trial_mask = np.ones((n_epochs, 3), dtype=bool)
    trial_mask[1, 2] = False
    X[~trial_mask] = 0.0
    timeline = ScheduledEventTimeline(
        event_ids=np.asarray([f"event:{index}" for index in range(n_epochs)]),
        group_ids=subjects,
        subject_ids=subjects,
        stimulus_ids=np.arange(n_epochs, dtype=np.int64),
        onset_samples=np.arange(n_epochs, dtype=np.int64),
        onset_times_s=np.arange(n_epochs, dtype=float),
        evidence_available_times_s=np.arange(n_epochs, dtype=float) + 0.8,
        evidence_indices=np.arange(n_epochs, dtype=np.int64),
        statuses=np.repeat("available", n_epochs),
        status_details=np.repeat("", n_epochs),
        dataset_ids=np.repeat("synthetic", n_epochs),
        session_ids=np.repeat("", n_epochs),
        run_ids=np.repeat("1", n_epochs),
        selection_ids=subjects,
        complete=True,
        online_causal=False,
        timing_source="synthetic",
        candidate_ids=np.asarray(["left", "right", "left", "right"]),
        target_candidate_ids=np.repeat("right", n_epochs),
        repetition_indices=np.zeros(n_epochs, dtype=np.int64),
    )
    return EpochDataset(
        name="qc_synthetic",
        X=X,
        y=np.asarray([0, 1, 0, 1], dtype=np.int64),
        subject_ids=subjects,
        channel_names=identity.names,
        channel_positions_m=identity.coords,
        channel_mask=np.ones(3, dtype=bool),
        preprocessing=PreprocessingSpec(
            name="qc_test",
            sfreq=100.0,
            l_freq=None,
            tmin_ms=-200.0,
            tmax_ms=800.0,
            n_times=100,
            baseline_mode="none",
        ),
        event_timeline=timeline,
        metadata=pd.DataFrame({"subject": subjects}),
        trial_channel_mask=trial_mask,
    )


def test_qc_features_keep_global_scale_separate_from_relative_ptp() -> None:
    X = np.random.default_rng(9).normal(size=(3, 3, 32)).astype(np.float32)
    scaled = X.copy()
    scaled[1] *= 25.0
    features = compute_epoch_qc_features(scaled, channel_mask=np.ones(3, dtype=bool))
    reference = compute_epoch_qc_features(X, channel_mask=np.ones(3, dtype=bool))

    np.testing.assert_allclose(features.relative_ptp[1], reference.relative_ptp[1])
    assert features.epoch_scale_v[1] > 20.0 * reference.epoch_scale_v[1]


def test_epoch_cache_round_trip_persists_qc_features(tmp_path) -> None:
    source = _dataset()
    path = save_epoch_dataset(tmp_path / "qc_epochs.npz", source)
    loaded = load_epoch_dataset(path, require_labels=True)

    assert source.qc_features is not None
    assert loaded.qc_features is not None
    assert loaded.record()["schema"] == "n2p3net_epoch_dataset/5"
    np.testing.assert_allclose(loaded.qc_features.relative_ptp, source.qc_features.relative_ptp)
    np.testing.assert_allclose(loaded.qc_features.channel_std_v, source.qc_features.channel_std_v)
    np.testing.assert_allclose(loaded.qc_features.epoch_scale_v, source.qc_features.epoch_scale_v)
    assert not loaded.qc_features.observed_mask[1, 2]
