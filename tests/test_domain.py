from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from data.channel import build_channel_identity
from data.domain import (
    adapt_common_channel_average_reference,
    common_channel_intersection,
    ensure_common_channel_average_reference,
    ensure_epoch_dataset_namespace,
    namespace_epoch_dataset,
)
from data.epochs import (
    EpochDataset,
    PreprocessingSpec,
    concatenate_epoch_datasets,
    materialize_dataset_identity,
)
from data.events import ScheduledEventTimeline
from data.identity import DatasetIdentityTable, ParticipantIdentityRecord


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
    assert first_car.preprocessing is first.preprocessing
    assert second_car.preprocessing is second.preprocessing


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


def test_common_car_ensure_reuses_exact_existing_adapter_without_rounding_again() -> None:
    source = _dataset(
        np.asarray([[[1, 2, 3, 4], [3, 4, 5, 6], [7, 8, 9, 10]]]),
        reference="A1",
    )
    adapted = adapt_common_channel_average_reference(source, ("Fz", "Cz", "Pz"))
    signal_before = adapted.X.copy()
    lineage_before = adapted.lineage.payload()

    ensured = ensure_common_channel_average_reference(adapted, ("Fz", "Cz", "Pz"))

    assert ensured is adapted
    np.testing.assert_array_equal(ensured.X, signal_before)
    assert ensured.lineage.payload() == lineage_before
    with pytest.raises(ValueError, match="already satisfies"):
        adapt_common_channel_average_reference(adapted, ("Fz", "Cz", "Pz"))


def test_common_car_ensure_rejects_provenance_contradicted_by_tensor() -> None:
    source = _dataset(
        np.asarray([[[1, 2, 3, 4], [3, 4, 5, 6], [7, 8, 9, 10]]]),
        reference="A1",
    )
    adapted = adapt_common_channel_average_reference(source, ("Fz", "Cz", "Pz"))
    forged = replace(adapted, X=source.X.copy())
    forged.validate(require_labels=True)

    with pytest.raises(ValueError, match="contradicted by non-zero channel means"):
        ensure_common_channel_average_reference(forged, ("Fz", "Cz", "Pz"))


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
    source_identity = materialize_dataset_identity(dataset).record_for("s")

    namespaced = namespace_epoch_dataset(dataset, "BI")

    assert namespaced.subject_ids.tolist() == ["BI::s"]
    assert namespaced.event_timeline.subject_ids.tolist() == ["BI::s"]
    assert namespaced.event_timeline.event_ids.tolist() == ["BI::e0"]
    assert namespaced.provenance["subject_namespace"] == "BI"
    assert namespaced.identity_table is not None
    namespaced_identity = namespaced.identity_table.record_for("BI::s")
    assert namespaced_identity.origin_subject_keys == source_identity.origin_subject_keys


def test_namespace_ensure_reuses_complete_preexisting_namespace_after_car() -> None:
    source = _dataset(np.ones((1, 3, 4)), reference="A1")
    namespaced = namespace_epoch_dataset(source, "BI")
    adapted = adapt_common_channel_average_reference(namespaced, ("Fz", "Cz", "Pz"))
    lineage_before = adapted.lineage.payload()

    ensured = ensure_epoch_dataset_namespace(adapted, "BI")

    assert ensured is adapted
    assert ensured.subject_ids.tolist() == ["BI::s"]
    assert ensured.event_timeline.subject_ids.tolist() == ["BI::s"]
    assert ensured.lineage.payload() == lineage_before


def test_namespace_transform_rejects_accidental_second_namespace() -> None:
    namespaced = namespace_epoch_dataset(
        _dataset(np.ones((1, 3, 4)), reference="A1"),
        "BI",
    )

    with pytest.raises(ValueError, match="already declares subject_namespace"):
        namespace_epoch_dataset(namespaced, "BI")


def test_namespace_ensure_rejects_conflicting_or_forged_provenance() -> None:
    source = _dataset(np.ones((1, 3, 4)), reference="A1")
    namespaced = namespace_epoch_dataset(source, "BI")
    with pytest.raises(ValueError, match="not requested"):
        ensure_epoch_dataset_namespace(namespaced, "BNCI")

    forged = replace(
        source,
        provenance={**source.provenance, "subject_namespace": "BI"},
    )
    with pytest.raises(ValueError, match="inconsistent with qualified identity axes"):
        ensure_epoch_dataset_namespace(forged, "BI")


def test_car_and_channel_representation_changes_preserve_origin_identity() -> None:
    dataset = _dataset(np.ones((1, 3, 4)), reference="A1")
    original = materialize_dataset_identity(dataset)

    car = adapt_common_channel_average_reference(dataset, ("Fz", "Cz", "Pz"))

    assert car.name != dataset.name
    assert car.identity_table is original
    assert car.identity_table.digest() == original.digest()


def test_identity_concat_rejects_unqualified_local_collision() -> None:
    first = _dataset(np.ones((1, 3, 4)), reference="A1")
    second = _dataset(np.ones((1, 3, 4)), reference="A1")
    second.event_timeline = replace(
        second.event_timeline,
        event_ids=np.asarray(["other-e0"]),
        group_ids=np.asarray(["other-group"]),
        dataset_ids=np.asarray(["unrelated-source"]),
        session_ids=np.asarray(["other-session"]),
        run_ids=np.asarray(["other-run"]),
        selection_ids=np.asarray(["other-selection"]),
    )

    with pytest.raises(ValueError, match="local-subject collision"):
        concatenate_epoch_datasets((first, second), name="ambiguous")

    merged = concatenate_epoch_datasets(
        (
            namespace_epoch_dataset(first, "A"),
            namespace_epoch_dataset(second, "B"),
        ),
        name="qualified",
    )
    assert merged.identity_table is not None
    first_origin = merged.identity_table.record_for("A::s").origin_subject_keys
    second_origin = merged.identity_table.record_for("B::s").origin_subject_keys
    assert first_origin != second_origin


def test_explicit_global_identity_survives_namespace_without_local_aliasing() -> None:
    dataset = _dataset(np.ones((1, 3, 4)), reference="A1")
    source = materialize_dataset_identity(dataset).record_for("s")
    dataset.identity_table = DatasetIdentityTable(
        records=(
            ParticipantIdentityRecord(
                local_subject_id="s",
                origin_subject_keys=source.origin_subject_keys,
                global_person_keys=("registry:participant-17",),
                identity_status="global_verified",
            ),
        )
    )

    namespaced = namespace_epoch_dataset(dataset, "DERIVED")

    assert namespaced.identity_table is not None
    record = namespaced.identity_table.record_for("DERIVED::s")
    assert record.global_person_keys == ("registry:participant-17",)
    assert record.origin_subject_keys == source.origin_subject_keys
    assert record.authority_key() == "registry:participant-17"
    assert record.authority_key("source") == source.origin_subject_keys[0]


def test_sampling_authority_key_rejects_ambiguous_source_aliases() -> None:
    record = ParticipantIdentityRecord(
        local_subject_id="s",
        origin_subject_keys=("source-a:s", "source-b:s"),
        identity_status="source_verified",
    )

    with pytest.raises(ValueError, match="exactly one"):
        record.authority_key("source_or_global")
