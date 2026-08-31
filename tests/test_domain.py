from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from data.channel import build_channel_identity
from data.domain import (
    adapt_common_channel_average_reference,
    common_channel_intersection,
    namespace_epoch_dataset,
)
from data.epochs import EpochDataset, PreprocessingSpec
from data.events import ScheduledEventTimeline


def _dataset(values: np.ndarray, *, reference: str, channels=("Fz", "Cz", "Pz")) -> EpochDataset:
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


def test_common_car_cancels_different_original_reference_offsets() -> None:
    underlying = np.asarray([[[1, 2, 3, 4], [3, 4, 5, 6], [7, 8, 9, 10]]], dtype=np.float32)
    first = _dataset(underlying - 100.0, reference="A1")
    second = _dataset(underlying + 37.0, reference="A2")

    first_car = adapt_common_channel_average_reference(first, ("Fz", "Cz", "Pz"))
    second_car = adapt_common_channel_average_reference(second, ("Fz", "Cz", "Pz"))

    np.testing.assert_allclose(first_car.X, second_car.X, atol=1e-6)
    assert first_car.provenance["source_reference"] == second_car.provenance["source_reference"]
    assert first_car.channel_names == second_car.channel_names == ("FZ", "CZ", "PZ")


def test_common_car_rejects_missing_trial_channel() -> None:
    dataset = _dataset(np.ones((1, 3, 4), dtype=np.float32), reference="A1")
    values = dataset.X.copy()
    values[:, 1, :] = 0.0
    dataset = replace(
        dataset,
        X=values,
        trial_channel_mask=np.asarray([[True, False, True]]),
    )

    with pytest.raises(ValueError, match="requires every selected channel"):
        adapt_common_channel_average_reference(dataset, ("Fz", "Cz", "Pz"))


def test_common_channel_intersection_preserves_first_dataset_order() -> None:
    first = _dataset(np.ones((1, 3, 4)), reference="A1")
    second = _dataset(
        np.ones((1, 3, 4)),
        reference="A2",
        channels=("Pz", "Cz", "Oz"),
    )

    assert common_channel_intersection(first, second) == ("CZ", "PZ")


def test_namespace_qualifies_subject_and_event_identity() -> None:
    dataset = _dataset(np.ones((1, 3, 4)), reference="A1")

    namespaced = namespace_epoch_dataset(dataset, "BI")

    assert namespaced.subject_ids.tolist() == ["BI::s"]
    assert namespaced.event_timeline.subject_ids.tolist() == ["BI::s"]
    assert namespaced.event_timeline.event_ids.tolist() == ["BI::e0"]
    assert namespaced.provenance["subject_namespace"] == "BI"
