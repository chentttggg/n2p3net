from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import data.epochs as epochs_module
from data.channel import build_channel_identity
from data.contract import assert_default_p300_input_contract
from data.epochs import (
    EpochDataset,
    PreprocessingSpec,
    concatenate_epoch_datasets,
    load_epoch_dataset,
    save_epoch_dataset,
    select_epoch_channels,
    write_epoch_dataset_record,
)
from data.events import ScheduledEventTimeline


def _dataset() -> EpochDataset:
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


def test_epoch_dataset_safe_round_trip(tmp_path) -> None:
    source = _dataset()
    path = save_epoch_dataset(tmp_path / "epochs.npz", source)
    loaded = load_epoch_dataset(
        path,
        expected_preprocessing=source.preprocessing,
        require_labels=True,
    )
    assert loaded.name == source.name
    assert loaded.channel_names == source.channel_names
    assert np.array_equal(loaded.X, source.X)
    assert np.array_equal(loaded.y, source.y)
    assert np.allclose(loaded.channel_positions_m, source.channel_positions_m)
    assert loaded.metadata.to_dict("list") == source.metadata.to_dict("list")
    assert loaded.provenance == source.provenance
    assert np.array_equal(loaded.event_timeline.candidate_ids, source.event_timeline.candidate_ids)
    assert loaded.event_timeline.supports_full_candidate_chain is True


def test_attested_cache_load_skips_repeated_full_contract_scan(tmp_path, monkeypatch) -> None:
    path = save_epoch_dataset(tmp_path / "epochs.npz", _dataset())
    assert path.with_suffix(".record.json").is_file()

    def fail_full_validation(*_args, **_kwargs):
        raise AssertionError("attested load must not repeat the full dataset scan")

    monkeypatch.setattr(EpochDataset, "validate", fail_full_validation)
    loaded = load_epoch_dataset(path, require_labels=True, validation="attested")

    assert loaded.X.shape == _dataset().X.shape


def test_attested_cache_rejects_missing_record(tmp_path) -> None:
    path = save_epoch_dataset(tmp_path / "epochs.npz", _dataset())
    path.with_suffix(".record.json").unlink()

    with pytest.raises(ValueError, match="lacks a cache attestation"):
        load_epoch_dataset(path, validation="attested")


def test_attested_cache_rejects_hash_disagreement(tmp_path) -> None:
    path = save_epoch_dataset(tmp_path / "epochs.npz", _dataset())
    record_path = path.with_suffix(".record.json")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["cache_attestation"]["sha256"] = "0" * 64
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        load_epoch_dataset(path, validation="attested")


def test_attested_cache_rejects_sidecar_event_contract_tampering(tmp_path) -> None:
    path = save_epoch_dataset(tmp_path / "epochs.npz", _dataset())
    record_path = path.with_suffix(".record.json")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["events"]["fingerprint"] = "0" * 64
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="events"):
        load_epoch_dataset(path, validation="attested")


def test_attestation_can_reuse_a_completed_full_validation(tmp_path, monkeypatch) -> None:
    path = save_epoch_dataset(tmp_path / "epochs.npz", _dataset())
    dataset = load_epoch_dataset(path, require_labels=True)

    def fail_full_validation(*_args, **_kwargs):
        raise AssertionError("the full validation was already completed")

    monkeypatch.setattr(EpochDataset, "validate", fail_full_validation)
    record = write_epoch_dataset_record(path, dataset, already_validated=True)

    assert record.is_file()


def test_epoch_dataset_decompresses_signal_member_once(tmp_path, monkeypatch) -> None:
    path = save_epoch_dataset(tmp_path / "epochs.npz", _dataset())
    original_load = epochs_module.np.load
    calls: dict[str, int] = {}

    class CountingArchive:
        def __init__(self, archive):
            self._archive = archive

        def __enter__(self):
            self._archive.__enter__()
            return self

        def __exit__(self, *args):
            return self._archive.__exit__(*args)

        @property
        def files(self):
            return self._archive.files

        def __getitem__(self, key):
            calls[key] = calls.get(key, 0) + 1
            return self._archive[key]

    def counted_load(*args, **kwargs):
        return CountingArchive(original_load(*args, **kwargs))

    monkeypatch.setattr(epochs_module.np, "load", counted_load)
    loaded = load_epoch_dataset(path, require_labels=True)

    assert loaded.X.shape == _dataset().X.shape
    assert calls["X"] == 1


def test_epoch_dataset_v2_loads_without_promoting_stimulus_codes(tmp_path) -> None:
    source = _dataset()
    current = save_epoch_dataset(tmp_path / "current.npz", source)
    with np.load(current, allow_pickle=False) as archive:
        payload = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key
            not in {
                "event_candidate_ids",
                "event_target_candidate_ids",
                "event_repetition_indices",
            }
        }
    payload["schema"] = np.asarray("n2p3net_epoch_dataset/2")
    payload["event_schema"] = np.asarray("n2p3net_scheduled_events/1")
    legacy = tmp_path / "legacy_v2.npz"
    np.savez(legacy, **payload)

    loaded = load_epoch_dataset(legacy)

    assert loaded.event_timeline.has_candidate_ids is False
    assert loaded.event_timeline.has_candidate_sets is False
    assert loaded.event_timeline.supports_full_candidate_chain is False


def test_epoch_dataset_rejects_candidate_truth_label_disagreement() -> None:
    dataset = _dataset()
    dataset.y[0] = 1

    with pytest.raises(ValueError, match="candidate_id == target_candidate_id"):
        dataset.validate()


def test_candidate_decision_support_is_derived_from_event_contract() -> None:
    dataset = _dataset()
    assert dataset.event_timeline.supports_full_candidate_chain is True

    dataset.event_timeline = replace(dataset.event_timeline, complete=False)
    assert dataset.event_timeline.supports_full_candidate_chain is False


def test_epoch_dataset_trial_channel_mask_round_trip(tmp_path) -> None:
    source = _dataset()
    trial_mask = np.ones(source.X.shape[:2], dtype=bool)
    trial_mask[1, 0] = False
    trial_mask[4, 2] = False
    source.trial_channel_mask = trial_mask
    source.X[~trial_mask] = 0.0

    loaded = load_epoch_dataset(save_epoch_dataset(tmp_path / "masked.npz", source))

    assert np.array_equal(loaded.trial_channel_mask, trial_mask)
    assert loaded.record()["has_trial_channel_mask"] is True


def test_epoch_dataset_rejects_nonzero_trial_masked_channel() -> None:
    dataset = _dataset()
    dataset.trial_channel_mask = np.ones(dataset.X.shape[:2], dtype=bool)
    dataset.trial_channel_mask[0, 1] = False

    with pytest.raises(ValueError, match="Trial-masked channels"):
        dataset.validate()


def test_epoch_dataset_rejects_legacy_npz(tmp_path) -> None:
    path = tmp_path / "legacy.npz"
    np.savez(path, X=np.zeros((2, 3, 100), dtype=np.float32), y=np.array([0, 1]))
    with pytest.raises(ValueError, match="lacks EpochDataset fields"):
        load_epoch_dataset(path)


def test_epoch_dataset_rejects_nonphysical_time_axis() -> None:
    dataset = _dataset()
    dataset.preprocessing = PreprocessingSpec(
        sfreq=100.0,
        tmin_ms=-200.0,
        tmax_ms=800.0,
        n_times=20,
    )
    with pytest.raises(ValueError, match="Physical time axis"):
        dataset.validate()


def test_preprocessing_trial_reference_accepts_stimulus_locked_window() -> None:
    profile = PreprocessingSpec(
        name="stimulus_locked_trial_reference",
        sfreq=256.0,
        l_freq=None,
        tmin_ms=0.0,
        tmax_ms=1000.0,
        n_times=256,
        baseline_mode="trial_reference",
        trial_reference_window_ms=(0.0, 50.0),
        trial_reference_center="median",
        trial_reference_scale="none",
        reject_threshold_v=None,
    )
    profile.validate()
    assert profile.trial_reference_window_ms == (0.0, 50.0)


def test_preprocessing_rejects_dimensionless_trial_scaling_before_physical_qc() -> None:
    with pytest.raises(ValueError, match="destroys the volts contract"):
        PreprocessingSpec(baseline_mode="trial").validate()
    with pytest.raises(ValueError, match="destroys the volts contract"):
        PreprocessingSpec(
            baseline_mode="trial_reference",
            trial_reference_window_ms=(-200.0, 0.0),
            trial_reference_scale="mad",
        ).validate()


def test_trial_scale_normalization_can_erase_erp_amplitude_information() -> None:
    """Counterexample motivating the volts-preserving ingress restriction."""

    base = np.array([-1.0, 1.0, 2.0, 4.0])
    doubled = 2.0 * base
    normalized_base = (base - base[:2].mean()) / base[:2].std()
    normalized_doubled = (doubled - doubled[:2].mean()) / doubled[:2].std()

    assert np.ptp(doubled) == 2.0 * np.ptp(base)
    np.testing.assert_allclose(normalized_base, normalized_doubled)


def test_preprocessing_trial_reference_rejects_window_outside_epoch() -> None:
    profile = PreprocessingSpec(
        sfreq=250.0,
        tmin_ms=0.0,
        tmax_ms=1000.0,
        n_times=250,
        baseline_mode="trial_reference",
        trial_reference_window_ms=(-1.0, 50.0),
    )
    with pytest.raises(ValueError, match="inside the physical epoch"):
        profile.validate()


def test_select_epoch_channels_is_exact_and_ordered() -> None:
    dataset = _dataset()
    selected = select_epoch_channels(dataset, ("Pz", "Fz"))
    assert selected.channel_names == ("PZ", "FZ")
    assert np.array_equal(selected.X, dataset.X[:, [2, 0], :])
    with pytest.raises(ValueError, match="cannot supply channels"):
        select_epoch_channels(dataset, ("Pz", "Oz"))


def test_epoch_dataset_rejects_nonfinite_signal() -> None:
    dataset = _dataset()
    dataset.X[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        dataset.validate()


def test_epoch_dataset_rejects_non_integer_labels() -> None:
    dataset = _dataset()
    dataset.y = dataset.y.astype(np.float32)
    with pytest.raises(ValueError, match="integer dtype"):
        dataset.validate()


def test_epoch_dataset_rejects_integer_signal_and_masks() -> None:
    dataset = _dataset()
    dataset.X = dataset.X.astype(np.int16)
    with pytest.raises(ValueError, match="floating-point dtype"):
        dataset.validate()

    dataset = _dataset()
    dataset.channel_mask = dataset.channel_mask.astype(np.int8)
    with pytest.raises(ValueError, match="channel_mask must have boolean dtype"):
        dataset.validate()


def test_channel_selection_cannot_hide_invalid_source_data() -> None:
    dataset = _dataset()
    dataset.channel_mask[1] = False
    with pytest.raises(ValueError, match="Absent channels"):
        select_epoch_channels(dataset, ("Fz", "Pz"))


def test_preprocessing_rejects_invalid_filter_contract() -> None:
    with pytest.raises(ValueError, match="smaller than h_freq"):
        PreprocessingSpec(l_freq=40.0, h_freq=20.0).validate()
    with pytest.raises(ValueError, match="Nyquist"):
        PreprocessingSpec(h_freq=200.0).validate()
    with pytest.raises(ValueError, match="Nyquist"):
        PreprocessingSpec(h_freq=128.0).validate()
    with pytest.raises(ValueError, match="Nyquist"):
        PreprocessingSpec(l_freq=128.0, h_freq=None).validate()
    with pytest.raises(ValueError, match="must be numeric"):
        PreprocessingSpec(sfreq="256").validate()


def test_default_preprocessing_matches_ms_eegnet_physical_scales() -> None:
    profile = PreprocessingSpec()

    assert profile.name == "p300_ms_eegnet_input_v2"
    assert (profile.sfreq, profile.l_freq, profile.h_freq) == (128.0, 2.0, 30.0)
    assert (profile.tmin_ms, profile.tmax_ms, profile.n_times) == (-200.0, 800.0, 128)
    assert profile.baseline_mode == "mean_only"
    assert profile.signal_unit == "V"
    assert profile.resample_method == "fft"
    assert profile.resample_npad == "auto"
    assert profile.resample_window == "auto"
    assert profile.resample_pad == "edge"


def test_mainline_contract_rejects_the_retired_250_hz_cache_geometry() -> None:
    assert_default_p300_input_contract(PreprocessingSpec())
    retired = PreprocessingSpec(
        name="retired_250_hz",
        sfreq=250.0,
        l_freq=0.1,
        h_freq=None,
        tmin_ms=-200.0,
        tmax_ms=800.0,
        n_times=250,
        baseline_mode="none",
    )

    with pytest.raises(ValueError, match="Regenerate the cache"):
        assert_default_p300_input_contract(retired)


def test_concatenation_rejects_unharmonized_source_references() -> None:
    left = _dataset()
    right = _dataset()
    left.provenance["source_reference"] = "right earlobe"
    right.provenance["source_reference"] = "nose"

    with pytest.raises(ValueError, match="common re-reference"):
        concatenate_epoch_datasets((left, right), name="invalid_transfer")


def test_reference_choice_creates_a_common_mode_dataset_confound() -> None:
    """Counterexample: identical scalp potentials differ under two references."""

    scalp = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [4.0, 8.0, 12.0]])
    reference_a = np.array([0.5, 1.0, 1.5])
    reference_b = np.array([-1.0, -2.0, -3.0])
    observed_a = scalp - reference_a[None, :]
    observed_b = scalp - reference_b[None, :]

    assert not np.allclose(observed_a, observed_b)
    car_a = observed_a - observed_a.mean(axis=0, keepdims=True)
    car_b = observed_b - observed_b.mean(axis=0, keepdims=True)
    np.testing.assert_allclose(car_a, car_b)


def test_epoch_dataset_rejects_unit_sphere_values_mislabeled_as_metres() -> None:
    dataset = _dataset()
    dataset.channel_positions_m *= 10.0
    with pytest.raises(ValueError, match="registered head-frame coordinates in metres"):
        dataset.validate()
