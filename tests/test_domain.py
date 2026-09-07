from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.channel import build_channel_identity
from data.domain import (
    NATIVE_REFERENCE_SCHEMA,
    adapt_common_channel_average_reference,
    adapt_native_reference_selection,
    adapt_observed_montage,
    common_channel_intersection,
    ensure_common_channel_average_reference,
    ensure_epoch_dataset_namespace,
    ensure_native_reference_selection,
    namespace_epoch_dataset,
    project_binary_evidence_source_view,
)
from data.epochs import (
    EpochDataset,
    PreprocessingSpec,
    concatenate_epoch_datasets,
    materialize_dataset_identity,
    save_epoch_dataset,
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


def test_common_car_is_invariant_to_time_varying_additive_reference() -> None:
    underlying = np.asarray(
        [
            [[1, 2, 3, 4], [3, 4, 5, 6], [7, 8, 9, 10]],
            [[2, 4, 6, 8], [1, 3, 5, 7], [9, 11, 13, 15]],
        ],
        dtype=np.float32,
    )
    offset_a = np.asarray([[[10, 20, 30, 40]], [[-5, -4, -3, -2]]], dtype=np.float32)
    offset_b = np.asarray([[[100, 70, 40, 10]], [[8, 6, 4, 2]]], dtype=np.float32)

    first = _dataset(underlying + offset_a, reference="time-varying-A")
    second = _dataset(underlying + offset_b, reference="time-varying-B")

    first_car = adapt_common_channel_average_reference(first, ("Fz", "Cz", "Pz"))
    second_car = adapt_common_channel_average_reference(second, ("Fz", "Cz", "Pz"))

    np.testing.assert_allclose(first_car.X, second_car.X, atol=1e-6)


@pytest.mark.parametrize("n_channels", (2, 3, 5))
def test_common_car_projection_matrix_has_expected_subspace_properties(n_channels: int) -> None:
    one = np.ones((n_channels, 1), dtype=np.float64)
    projection = np.eye(n_channels) - (one @ one.T) / n_channels

    np.testing.assert_allclose(projection.T, projection)
    np.testing.assert_allclose(projection @ projection, projection)
    np.testing.assert_allclose(projection @ one, 0.0, atol=1e-15)
    assert np.linalg.matrix_rank(projection) == n_channels - 1


def test_two_channel_car_is_rank_one_contrast_and_removes_selected_mean() -> None:
    values = np.asarray(
        [[[11, 12, 13, 14], [3, 5, 7, 9]], [[-2, 0, 2, 4], [8, 6, 4, 2]]],
        dtype=np.float32,
    )
    dataset = _dataset(values, reference="A1", channels=("Cz", "Pz"))

    adapted = adapt_common_channel_average_reference(dataset, ("Cz", "Pz"))

    np.testing.assert_allclose(adapted.X.sum(axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(adapted.X[:, 0, :], (values[:, 0, :] - values[:, 1, :]) / 2.0)
    np.testing.assert_allclose(adapted.X[:, 1, :], -adapted.X[:, 0, :])


def test_two_channel_car_cannot_distinguish_vectors_differing_only_in_common_mode() -> None:
    base = np.asarray([[[1, 2, 3, 4], [4, 3, 2, 1]]], dtype=np.float32)
    common_mode = np.asarray([[[100, -20, 7, 11], [100, -20, 7, 11]]], dtype=np.float32)
    first = _dataset(base, reference="A1", channels=("Cz", "Pz"))
    second = _dataset(base + common_mode, reference="A2", channels=("Cz", "Pz"))

    first_car = adapt_common_channel_average_reference(first, ("Cz", "Pz"))
    second_car = adapt_common_channel_average_reference(second, ("Cz", "Pz"))

    np.testing.assert_array_equal(first_car.X, second_car.X)


def test_native_reference_selection_preserves_tensor_and_records_reference() -> None:
    values = np.asarray([[[1, 2, 3, 4], [3, 4, 5, 6], [7, 8, 9, 10]]], dtype=np.float32)
    dataset = _dataset(values, reference="right_mastoid", channels=("Fz", "Cz", "Pz"))

    selected = adapt_native_reference_selection(dataset, ("Cz", "Pz"))

    np.testing.assert_array_equal(selected.X, values[:, 1:, :])
    assert selected.channel_names == ("CZ", "PZ")
    assert selected.provenance["source_reference"] == "native:right_mastoid"
    assert selected.provenance["domain_adapter"] == {
        "schema": NATIVE_REFERENCE_SCHEMA,
        "target_channels": ["CZ", "PZ"],
        "operation": "select channels without re-referencing",
        "preserved_source_reference": "right_mastoid",
        "preprocessing_identity": dataset.preprocessing.name,
    }


def test_native_reference_selection_rejects_car_input_and_missing_reference() -> None:
    values = np.asarray([[[1, 2, 3, 4], [3, 4, 5, 6], [7, 8, 9, 10]]], dtype=np.float32)
    dataset = _dataset(values, reference="right_mastoid")
    car = adapt_common_channel_average_reference(dataset, ("Fz", "Cz", "Pz"))
    with pytest.raises(ValueError, match="already domain-adapted"):
        adapt_native_reference_selection(car, ("Fz", "Cz", "Pz"))

    missing = replace(dataset, provenance={"source": "toy"})
    with pytest.raises(ValueError, match="non-empty source_reference"):
        adapt_native_reference_selection(missing, ("Fz", "Cz", "Pz"))


def test_native_reference_selection_ensure_reuses_complete_contract() -> None:
    source = _dataset(
        np.asarray([[[1, 2, 3, 4], [3, 4, 5, 6], [7, 8, 9, 10]]]),
        reference="A1",
    )
    selected = adapt_native_reference_selection(source, ("Fz", "Cz", "Pz"))
    reused = ensure_native_reference_selection(selected, ("Fz", "Cz", "Pz"))
    assert reused is selected
    forged = replace(
        selected,
        provenance={**selected.provenance, "source_reference": "native:forged"},
    )
    with pytest.raises(ValueError, match="disagrees|mismatch"):
        ensure_native_reference_selection(forged, ("Fz", "Cz", "Pz"))


def test_native_selection_requires_complete_selected_channels() -> None:
    values = np.ones((1, 3, 4), dtype=np.float32)
    values[:, 1, :] = 0.0
    dataset = _dataset(values, reference="A1")
    masked = replace(
        dataset,
        trial_channel_mask=np.asarray([[True, False, True]]),
    )
    with pytest.raises(ValueError, match="requires every selected channel"):
        adapt_native_reference_selection(masked, ("Fz", "Cz", "Pz"))


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


def test_binary_evidence_source_projection_is_explicit_and_tensor_preserving() -> None:
    source = _dataset(np.ones((2, 3, 4)), reference="A1")
    source.event_timeline = replace(
        source.event_timeline,
        candidate_ids=np.asarray(["left", "right"]),
        target_candidate_ids=np.asarray(["right", "right"]),
        repetition_indices=np.asarray([0, 0], dtype=np.int64),
    )
    source.validate(require_labels=True)
    assert source.event_timeline.supports_full_candidate_chain is True
    signal_before = source.X.copy()
    labels_before = source.y.copy()
    timeline_before = source.event_timeline

    projected = project_binary_evidence_source_view(source)

    np.testing.assert_array_equal(projected.X, signal_before)
    np.testing.assert_array_equal(projected.y, labels_before)
    assert projected.event_timeline.has_candidate_ids is False
    assert projected.event_timeline.has_candidate_sets is False
    assert projected.event_timeline.has_repetition_structure is False
    for field_name in (
        "event_ids",
        "group_ids",
        "subject_ids",
        "stimulus_ids",
        "onset_samples",
        "onset_times_s",
        "evidence_available_times_s",
        "evidence_indices",
        "statuses",
        "dataset_ids",
        "session_ids",
        "run_ids",
        "selection_ids",
    ):
        np.testing.assert_array_equal(
            getattr(projected.event_timeline, field_name),
            getattr(timeline_before, field_name),
        )
    record = projected.provenance["binary_evidence_projection"]
    assert record["source_event_fingerprint"] == timeline_before.fingerprint()
    assert record["projected_event_fingerprint"] == projected.event_timeline.fingerprint()
    assert projected.lineage.entities[-1].operation == "project_binary_evidence_source_view"
    assert project_binary_evidence_source_view(projected) is projected


def test_common_channel_intersection_preserves_first_dataset_order() -> None:
    first = _dataset(np.ones((1, 3, 4)), reference="A1")
    second = _dataset(
        np.ones((1, 3, 4)),
        reference="A2",
        channels=("Pz", "Cz", "Oz"),
    )

    assert common_channel_intersection(first, second) == ("CZ", "PZ")


def test_binary_source_projects_candidate_metadata_without_timeline_candidates() -> None:
    source = _dataset(np.ones((2, 3, 4)), reference="A1")
    source.metadata["candidate_id"] = [0, 1]
    source.metadata["target_row"] = [1, 1]
    source.metadata["repetition_index"] = [0, 0]
    source.metadata["selection_id"] = ["source-episode", "source-episode"]
    source.provenance["candidate_task_contract"] = {"membership_kind": "row_column"}
    source.provenance["event_contract"] = "candidate_source"
    source.validate(require_labels=True)
    assert not source.event_timeline.has_candidate_ids

    projected = project_binary_evidence_source_view(source)

    assert projected.provenance["event_contract"] == "binary_evidence"
    assert "candidate_task_contract" not in projected.provenance
    assert set(projected.metadata) == {"subject", "selection_id"}
    assert projected.X is source.X
    assert projected.y is source.y
    assert "candidate_id" in source.metadata
    assert "candidate_task_contract" in source.provenance
    assert projected.provenance["binary_evidence_projection"]["source_candidate_task_contract"]
    assert project_binary_evidence_source_view(projected) is projected


def test_already_binary_source_still_records_explicit_training_purpose() -> None:
    source = _dataset(np.ones((2, 3, 4)), reference="A1")
    projected = project_binary_evidence_source_view(source)
    assert projected.provenance["event_contract"] == "binary_evidence"
    assert projected.provenance["binary_evidence_projection"]["labels_unchanged"]
    assert project_binary_evidence_source_view(projected) is projected


def test_masked_montage_never_promotes_absent_electrodes_to_observed_data() -> None:
    source = _dataset(np.arange(24).reshape(2, 3, 4), reference="A1")
    projected = adapt_observed_montage(source, ("Fz", "Cz", "Pz", "Oz"))
    assert projected.X.shape == (2, 4, 4)
    np.testing.assert_array_equal(projected.X[:, :3], source.X)
    assert not projected.channel_mask[3]
    assert not projected.trial_channel_mask[:, 3].any()
    assert np.all(projected.X[:, 3] == 0)
    assert projected.provenance["missing_target_channels"] == ["OZ"]
    assert projected.identity_table.digest() == materialize_dataset_identity(source).digest()
    assert adapt_observed_montage(projected, projected.channel_names) is projected


def test_measured_reference_cancels_unknown_time_varying_original_reference() -> None:
    underlying = np.arange(24).reshape(2, 3, 4)
    first = _dataset(
        underlying - np.array([100.0, 30.0, -20.0, 80.0])[None, None, :], reference="A1"
    )
    second = _dataset(underlying + np.array([7.0, -19.0, 23.0, 1.0])[None, None, :], reference="T7")
    a = adapt_observed_montage(first, ("Fz", "Pz", "Oz"), reference_channel="Cz")
    b = adapt_observed_montage(second, ("Fz", "Pz", "Oz"), reference_channel="Cz")
    np.testing.assert_array_equal(a.X, b.X)
    assert (
        a.provenance["source_reference"]
        == b.provenance["source_reference"]
        == "measured_channel:CZ"
    )
    assert not a.trial_channel_mask[:, 2].any()
    assert np.all(a.X[:, 2] == 0.0)
    assert adapt_observed_montage(a, a.channel_names, reference_channel="Cz") is a


def test_missing_measurement_is_not_replaced_with_negative_reference_waveform() -> None:
    source = _dataset(np.arange(24).reshape(2, 3, 4), reference="A1")
    source.trial_channel_mask = np.ones(source.X.shape[:2], dtype=bool)
    source.trial_channel_mask[0, 0] = False
    source.X[0, 0] = 0.0
    output = adapt_observed_montage(source, ("Fz", "Pz"), reference_channel="Cz")
    assert not output.trial_channel_mask[0, 0]
    assert np.all(output.X[0, 0] == 0.0)
    source.trial_channel_mask[0, 1] = False
    source.X[0, 1] = 0.0
    with pytest.raises(ValueError, match="reference must be observed"):
        adapt_observed_montage(source, ("Fz", "Pz"), reference_channel="Cz")


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


def test_prepare_domain_features_identity_flatten(tmp_path: Path, monkeypatch) -> None:
    from experiments.prepare_domain_features import main

    values = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) * 1e-7
    dataset = _dataset(values, reference="A1")
    cache = save_epoch_dataset(tmp_path / "source.npz", dataset)
    output = tmp_path / "features.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_domain_features.py",
            "--dataset-cache",
            str(cache),
            "--output",
            str(output),
        ],
    )
    main()

    with np.load(output, allow_pickle=False) as archive:
        assert archive["features"].shape == (2, 12)
        assert archive["schema"].item() == "n2p3_domain_feature_matrix/1"
    record = json.loads(output.with_suffix(".record.json").read_text(encoding="utf-8"))
    assert record["spatial_lift"]["kind"] == "identity"
    assert record["temporal_basis"]["kind"] == "identity"
