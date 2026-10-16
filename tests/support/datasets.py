"""Synthetic datasets builders for contract tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from data.channel import build_channel_identity
from data.epochs import EpochDataset, PreprocessingSpec, materialize_dataset_identity
from data.events import ScheduledEventTimeline, observed_only_timeline


def reference_dataset(
    values: np.ndarray, *, reference: str, channels=("Fz", "Cz", "Pz")
) -> EpochDataset:
    identity = build_channel_identity(channels, allow_missing_positions=False)
    n_epochs = len(values)
    timeline = ScheduledEventTimeline(
        event_ids=np.asarray([f"e{index}" for index in range(n_epochs)]),
        group_ids=np.repeat("g", n_epochs),
        subject_ids=np.repeat("s", n_epochs),
        stimulus_ids=np.arange(n_epochs, dtype=np.int64),
        onset_samples=np.arange(n_epochs, dtype=np.int64),
        onset_times_s=np.arange(n_epochs, dtype=float),
        evidence_available_times_s=np.arange(n_epochs, dtype=float) + 1.0,
        evidence_indices=np.arange(n_epochs, dtype=np.int64),
        statuses=np.repeat("available", n_epochs),
        status_details=np.repeat("", n_epochs),
        dataset_ids=np.repeat("toy", n_epochs),
        session_ids=np.repeat("session", n_epochs),
        run_ids=np.repeat("run", n_epochs),
        selection_ids=np.repeat("selection", n_epochs),
        complete=True,
        online_causal=True,
        timing_source="toy",
    )
    return EpochDataset(
        name=f"toy-{reference}",
        X=np.asarray(values, dtype=np.float32),
        y=np.arange(n_epochs, dtype=np.int64) % 2,
        subject_ids=np.repeat("s", n_epochs),
        channel_names=identity.names,
        channel_positions_m=identity.coords,
        channel_mask=np.ones(len(channels), dtype=bool),
        preprocessing=PreprocessingSpec(
            name="causal",
            sfreq=4.0,
            l_freq=None,
            h_freq=None,
            tmin_ms=-250.0,
            tmax_ms=750.0,
            n_times=4,
            filter_phase="forward",
            causal_iir_initial_state="steady_state_first_sample",
        ),
        event_timeline=timeline,
        metadata=pd.DataFrame({"subject": np.repeat("s", n_epochs)}),
        provenance={"source": "toy", "source_reference": reference},
    )


def epoch_dataset() -> EpochDataset:
    channels = ("Fz", "Cz", "Pz")
    identity = build_channel_identity(channels, allow_missing_positions=False)
    subjects = np.array(["s1", "s1", "s2", "s2", "s3", "s3"])
    n_epochs = len(subjects)
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
        dataset_ids=np.repeat("synthetic_p300", n_epochs),
        session_ids=np.repeat("", n_epochs),
        run_ids=np.repeat("1", n_epochs),
        selection_ids=subjects,
        complete=True,
        online_causal=False,
        timing_source="synthetic_schedule",
        candidate_ids=np.tile(np.asarray(["left", "right"]), 3),
        target_candidate_ids=np.repeat("right", n_epochs),
        repetition_indices=np.zeros(n_epochs, dtype=np.int64),
    )
    return EpochDataset(
        name="synthetic_p300",
        X=np.arange(6 * 3 * 100, dtype=np.float32).reshape(6, 3, 100) * 1e-8,
        y=np.array([0, 1, 0, 1, 0, 1], dtype=np.int64),
        subject_ids=subjects,
        channel_names=identity.names,
        channel_positions_m=identity.coords,
        channel_mask=np.ones(3, dtype=bool),
        preprocessing=PreprocessingSpec(
            name="test_profile",
            sfreq=100.0,
            l_freq=None,
            tmin_ms=-200.0,
            tmax_ms=800.0,
            n_times=100,
            reject_threshold_v=None,
        ),
        event_timeline=timeline,
        metadata=pd.DataFrame({"subject": ["s1", "s1", "s2", "s2", "s3", "s3"], "run": [1] * 6}),
        provenance={"source": "unit_test", "source_reference": "average"},
    )


def source_dataset(library="source_library", namespace="source"):
    from data.domain import annotate_source_domain, namespace_epoch_dataset
    from data.identity import DatasetIdentityTable

    dataset = epoch_dataset()
    dataset.identity_table = DatasetIdentityTable.from_source_rows(
        dataset.subject_ids.tolist(),
        [library] * dataset.n_epochs,
    )
    return annotate_source_domain(namespace_epoch_dataset(dataset, namespace), namespace)


def binary_dataset(count=40):

    dataset = epoch_dataset()
    from data.contract import CAUSAL_IIR_INITIAL_STATE

    dataset.preprocessing = replace(dataset.preprocessing, filter_phase="forward",
                                     causal_iir_initial_state=CAUSAL_IIR_INITIAL_STATE)
    subjects = np.repeat("p1", count)
    dataset = replace(
        dataset,
        X=np.random.default_rng(9).normal(size=(count, 3, 100)).astype(np.float32) * 1e-6,
        y=np.tile([0, 1], count // 2),
        subject_ids=subjects,
        metadata=pd.DataFrame({"subject": subjects, "run": np.repeat("r1", count)}),
        event_timeline=observed_only_timeline(
            dataset_id="synthetic_p300",
            subject_ids=subjects,
            stimulus_ids=np.arange(count),
            onset_times_s=np.arange(count) * 2.0,
            evidence_available_times_s=np.arange(count) * 2.0 + 0.8,
            session_ids=np.repeat("s1", count),
            run_ids=np.repeat("r1", count),
            online_causal=True,
        ),
    )
    materialize_dataset_identity(dataset)
    return dataset


def make_p300_data(n_target=120, n_nontarget=480, seed=0, *, n_times, sfreq=128.0, tmin_s=-0.2):
    """Eight-channel noise with a 300-500 ms positive target pulse at channel 3."""
    rng = np.random.default_rng(seed)
    n = n_target + n_nontarget
    X = rng.standard_normal((n, 8, n_times)).astype(np.float32)
    y = np.zeros(n, dtype=int)
    y[:n_target] = 1
    i0, i1 = round((0.3 - tmin_s) * sfreq), round((0.5 - tmin_s) * sfreq)
    center, width = (i0 + i1) / 2, (i1 - i0) / 6
    X[:n_target, 3, :] += 5.0 * np.exp(-0.5 * ((np.arange(n_times) - center) / width) ** 2)
    indices = rng.permutation(n)
    return X[indices], y[indices]
