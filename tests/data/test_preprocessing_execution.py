"""Execution contracts preserve samples and expose their actual time support."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import mne
import numpy as np
import pytest

from data.contract import CAUSAL_IIR_INITIAL_STATE
from data.epochs import (
    PreprocessingSpec,
    load_preprocessing_spec,
    parse_preprocessing_spec,
    validate_preprocessing_execution,
)
from data.preprocess import (
    execute_preprocessing,
    preprocess,
    transform_epoch_samples,
)
from research.contracts import semantic_sha256


def _raw(*, sfreq=250.0, first_samp=0, values=None):
    if values is None:
        values = np.random.default_rng(23).normal(0, 2e-6, (2, int(8 * sfreq)))
    return mne.io.RawArray(
        values,
        mne.create_info(["Pz", "Fz"], sfreq, ch_types="eeg"),
        first_samp=first_samp,
        verbose=False,
    )


def _events(samples):
    samples = np.asarray(samples, dtype=np.int64)
    return np.column_stack((samples, np.zeros_like(samples), np.ones_like(samples)))


def _spec(**changes):
    return replace(
        PreprocessingSpec(
            name="finite_execution_test",
            filter_phase="forward",
            causal_iir_initial_state=CAUSAL_IIR_INITIAL_STATE,
        ),
        **changes,
    )


@pytest.mark.parametrize("phase", ["forward", "zero"])
def test_executor_preserves_legacy_samples_layout_and_rejection_ledger(phase):
    raw = _raw()
    raw.set_annotations(mne.Annotations([3.7], [0.8], ["BAD_motion"]))
    events = _events([20, 500, 1000, 1500, 1950])
    initial_state = CAUSAL_IIR_INITIAL_STATE if phase == "forward" else "not_applicable"
    spec = _spec(filter_phase=phase, causal_iir_initial_state=initial_state)
    original = raw.get_data().copy()

    # The pre-refactor public call is deliberately written out as an independent
    # consumer, rather than deriving its keyword mapping from the new executor.
    expected = preprocess(
        raw,
        events,
        sfreq=128.0,
        l_freq=0.1,
        h_freq=30.0,
        tmin=-0.2,
        tmax=1.2,
        n_times=179,
        baseline_mode="mean_only",
        filter_method="iir",
        filter_order=4,
        filter_phase=phase,
        causal_iir_initial_state=initial_state,
        channels=["Fz", "Pz"],
    )
    actual = execute_preprocessing(raw, events, spec, channels=["Fz", "Pz"])

    for field in (
        "data",
        "channel_positions_m",
        "channel_mask",
        "position_mask",
        "dropped",
        "event_indices",
        "event_samples",
        "event_times_s",
        "event_statuses",
        "event_status_details",
        "event_evidence_indices",
    ):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert actual.channel_names == expected.channel_names == ("FZ", "PZ")
    assert actual.online_causal == expected.online_causal
    assert actual.data.dtype == np.float32
    assert actual.event_statuses.tolist() == [
        "boundary_dropped", "available", "acquisition_rejected", "available", "boundary_dropped",
    ]
    np.testing.assert_array_equal(raw.get_data(), original)
    assert raw.ch_names == ["Pz", "Fz"]


@pytest.mark.parametrize("n_times", [180, True, 179.0])
def test_invalid_declared_sample_count_is_rejected_before_raw_mutation(n_times):
    raw = _raw()
    original = raw.get_data().copy()
    with pytest.raises(ValueError):
        execute_preprocessing(
            raw, _events([500]), _spec(n_times=n_times), channels=["Fz"], copy=False,
        )
    np.testing.assert_array_equal(raw.get_data(), original)
    assert raw.ch_names == ["Pz", "Fz"]


@pytest.mark.parametrize(
    "changes",
    [
        {"sfreq": 200.0, "n_times": 280, "h_freq": 60.0},
        {"notch_freqs_hz": (50.0,)},
    ],
)
def test_source_nyquist_is_rejected_before_channel_selection(changes):
    raw = _raw(sfreq=100.0)
    original = raw.get_data().copy()
    with pytest.raises(ValueError, match="Nyquist|nyquist"):
        execute_preprocessing(
            raw, _events([200]), _spec(**changes), channels=["Fz"], copy=False,
        )
    np.testing.assert_array_equal(raw.get_data(), original)
    assert raw.ch_names == ["Pz", "Fz"]


def test_low_level_none_sample_count_preserves_full_resampled_epoch():
    raw = _raw()
    events = _events([500])
    expected = mne.Epochs(
        raw, events, tmin=-0.2, tmax=1.2, baseline=None, preload=True, verbose=False,
    )
    expected.resample(128.0, npad="auto", window="auto", pad="edge", method="fft", verbose=False)
    actual = preprocess(
        raw,
        events,
        l_freq=None,
        h_freq=None,
        n_times=None,
        baseline_mode="none",
        filter_phase="forward",
        causal_iir_initial_state=CAUSAL_IIR_INITIAL_STATE,
    )
    assert actual.n_times == expected.get_data().shape[-1] == 180
    np.testing.assert_array_equal(actual.data, expected.get_data().astype(np.float32))


def test_rounded_source_endpoint_can_influence_retained_fft_samples():
    values = np.zeros((2, 2000))
    modified = values.copy()
    modified[0, 1300] = 1e-6
    spec = _spec(tmax_ms=1199.0, n_times=179, l_freq=None, h_freq=None, baseline_mode="none")
    reference = execute_preprocessing(_raw(values=values), _events([1000]), spec)
    actual = execute_preprocessing(_raw(values=modified), _events([1000]), spec)

    assert actual.data.shape == (1, 2, 179)
    assert np.count_nonzero(actual.data - reference.data) > 0
    np.testing.assert_array_equal(actual.source_epoch_start_samples, [950])
    np.testing.assert_array_equal(actual.source_epoch_end_samples, [1300])
    assert actual.evidence_available_times_s[0] >= 5.2
    assert actual.evidence_available_times_s[0] >= 4.0 + 1.199
    assert isinstance(actual.timing_source, str) and actual.timing_source


def test_absolute_source_support_includes_first_sample_offset_and_rejected_events():
    first_samp = 500
    raw = _raw(first_samp=first_samp)
    spec = _spec(l_freq=None, h_freq=None, baseline_mode="none")
    result = execute_preprocessing(raw, _events([520, 1500]), spec)

    np.testing.assert_array_equal(result.source_epoch_start_samples, [470, 1450])
    np.testing.assert_array_equal(result.source_epoch_end_samples, [820, 1800])
    np.testing.assert_array_equal(result.event_times_s, [520 / 250.0, 1500 / 250.0])
    assert result.event_statuses.tolist() == ["boundary_dropped", "available"]
    assert np.isnan(result.evidence_available_times_s[0])
    assert result.evidence_available_times_s[1] >= 1800 / 250.0


def test_median_baseline_transforms_numerically_but_is_not_a_linear_operator():
    spec = _spec(
        sfreq=250.0,
        n_times=350,
        baseline_mode="trial_reference",
        trial_reference_window_ms=(-200.0, 0.0),
        trial_reference_center="median",
    )
    values = np.arange(2 * 3 * 351, dtype=float).reshape(2, 3, 351)
    values[:, :, 0] = 1e8
    original = values.copy()
    expected = values[:, :, :350] - np.median(values[:, :, :50], axis=-1, keepdims=True)

    actual = transform_epoch_samples(values, 250.0, spec)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(values, original)
    with pytest.raises(ValueError, match="linear|median"):
        transform_epoch_samples(values, 250.0, spec, require_linear=True)


def test_historical_optional_fields_keep_original_no_notch_legacy_identity():
    record = _spec(filter_phase="zero", causal_iir_initial_state="not_applicable").record()
    assert "notch_freqs_hz" not in record
    assert "zero_phase_edge_policy" not in record
    restored = parse_preprocessing_spec(record)
    assert restored.notch_freqs_hz == ()
    assert restored.zero_phase_edge_policy == "legacy"
    assert restored.record() == record


def test_empty_new_configuration_is_not_a_request_for_current_defaults(tmp_path):
    path = tmp_path / "preprocessing.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_preprocessing_spec(path)


def test_explicit_new_configuration_round_trips_and_rejects_missing_physical_field(tmp_path):
    spec = _spec(notch_freqs_hz=(50.0,))
    record = asdict(spec)
    path = tmp_path / "preprocessing.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert load_preprocessing_spec(path) == spec
    record.pop("h_freq")
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        load_preprocessing_spec(path)


def test_explicit_notch_requires_its_quality_factor():
    record = _spec(notch_freqs_hz=(50.0,)).record()
    record.pop("notch_quality_factor")
    with pytest.raises(ValueError):
        parse_preprocessing_spec(record)


def test_execution_receipt_is_separate_from_the_processing_declaration():
    spec = _spec()
    before = spec.record()
    result = execute_preprocessing(_raw(), _events([500]), spec)

    receipt = result.execution_receipt
    assert isinstance(receipt, dict)
    assert receipt["schema"] == "n2p3_preprocessing_execution/1"
    assert receipt["declaration"] == before
    assert receipt["semantic_identity"]["schema"] == "n2p3_preprocessing_operator/1"
    assert receipt["semantic_identity"]["backend"] == "mne_raw"
    assert receipt["semantic_identity"]["operations"]
    assert receipt["source_context"]["sample_rate_hz"] == 250.0
    assert receipt["source_context"]["first_samp"] == 0
    assert receipt["source_context"]["n_times"] == 2000
    expected_grid = {
        "start_offset_samples": -50,
        "end_offset_samples": 300,
        "source_n_times": 351,
        "resampled_n_times": 180,
        "output_n_times": 179,
        "requested_tmin_s": -0.2,
        "actual_source_tmin_s": -0.2,
        "resampled": True,
    }
    assert expected_grid.items() <= receipt["epoch_grid"].items()
    assert {"python", "mne", "numpy", "scipy", "source_sha256"} <= receipt["implementation"].keys()
    assert len(receipt["receipt_sha256"]) == 64
    int(receipt["receipt_sha256"], 16)
    assert spec.record() == before
    json.dumps(receipt, allow_nan=False)


def test_legacy_mne_receipt_records_actual_global_annotation_filter_segments():
    values = np.zeros((2, 2000))
    values[:, 1000:] = 1e-6
    raw = _raw(values=values)
    raw.set_annotations(mne.Annotations(
        [4.0], [0.0], ["bad_continuous_artifact"], ch_names=[["Pz"]],
    ))
    spec = _spec(filter_phase="zero", causal_iir_initial_state="not_applicable")
    result = execute_preprocessing(raw, _events([500, 1500]), spec)

    # Legacy Raw.filter applies this reset to both channels. Its receipt must
    # describe that behavior, even though the source annotation names only Pz.
    np.testing.assert_array_equal(result.data[:, 0], result.data[:, 1])
    assert result.execution_receipt["source_context"]["filter_segments"] == {
        "PZ": [[0, 1000], [1000, 2000]],
        "FZ": [[0, 1000], [1000, 2000]],
    }
    boundaries = result.execution_receipt["source_context"]["filter_segment_boundaries"]
    assert boundaries["PZ"] == boundaries["FZ"]
    assert boundaries["FZ"][0]["right_causes"] == ["artifact"]
    assert boundaries["FZ"][1]["left_causes"] == ["artifact"]


def test_filter_plan_preserves_artifact_causes_through_overlaps_and_coincident_cuts():
    from data.preprocess import continuous_filter_plan, continuous_filter_segments

    raw = _raw(first_samp=1000)
    raw.set_annotations(mne.Annotations(
        [2, 2.5, 3, 4, 5, 5, 6], [.75, 1.5, 0, 0, 0, 0, 0],
        ["BAD_continuous_artifact", "BAD_acq_skip", "EDGE boundary", "EDGE boundary",
         "QUALITY_LOCAL_ARTIFACT", "EDGE boundary", "EDGE unexplained_transition"],
        ch_names=[(), (), (), (), ("Pz",), (), ()]))
    plans = {name: continuous_filter_plan(raw, name) for name in raw.ch_names}
    assert continuous_filter_segments(raw, "Pz") == ((0, 500), (1000, 1250), (1250, 1500), (1500, 2000))
    pz = plans["Pz"]
    assert pz[0].left_causes == frozenset({"record_start"})
    assert pz[0].right_causes == pz[1].left_causes == frozenset({"artifact", "acquisition_discontinuity"})
    assert pz[1].right_causes == pz[2].left_causes == frozenset({"artifact", "acquisition_discontinuity"})
    assert pz[2].right_causes == pz[3].left_causes == frozenset({"unclassified_reset"})
    assert pz[3].right_causes == frozenset({"record_end"})
    assert plans["Fz"][1].right_causes == frozenset({"acquisition_discontinuity"})


def test_filter_plan_is_invariant_to_annotation_split_duplicates_and_order():
    from data.preprocess import continuous_filter_plan

    first, second = _raw(), _raw()
    first.set_annotations(mne.Annotations([2, 3], [1, 1], ["BAD_continuous_artifact", "BAD_acq_skip"]))
    second.set_annotations(mne.Annotations([3, 2.5, 2, 2.5], [1, .5, .5, .5],
        ["BAD_acq_skip", "BAD_continuous_artifact", "BAD_continuous_artifact", "BAD_continuous_artifact"]))
    assert continuous_filter_plan(first, "Pz") == continuous_filter_plan(second, "Pz")


def test_filter_plan_preserves_nonreset_annotations_and_rejects_ambiguous_local_scope():
    from data.preprocess import continuous_filter_plan

    clean, annotated = _raw(), _raw()
    annotated.set_annotations(mne.Annotations([2, 3, 4], [1, 1, 1],
        ["BAD_brainsync_rest", "BAD_continuous_recovery", "unknown_observation"]))
    assert continuous_filter_plan(annotated, "Pz") == continuous_filter_plan(clean, "Pz")
    annotated.annotations.append(5, 0, "QUALITY_LOCAL_ARTIFACT")
    with pytest.raises(ValueError, match="explicit affected channels"):
        continuous_filter_plan(annotated, "Pz")


def test_receipt_boundary_causes_are_optional_for_history_but_validated_when_present():
    spec = _spec()
    receipt = deepcopy(execute_preprocessing(_raw(), _events([500]), spec).execution_receipt)
    assert "filter_segment_boundaries" in receipt["source_context"]
    receipt["source_context"].pop("filter_segment_boundaries")
    receipt["receipt_sha256"] = semantic_sha256({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    validate_preprocessing_execution(receipt, spec)
    receipt["source_context"]["filter_segment_boundaries"] = {
        "PZ": [{"start_sample": 0, "stop_sample": 2000, "left_causes": ["guessed_natural"], "right_causes": ["record_end"]}],
        "FZ": [{"start_sample": 0, "stop_sample": 2000, "left_causes": ["record_start"], "right_causes": ["record_end"]}],
    }
    receipt["receipt_sha256"] = semantic_sha256({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    with pytest.raises(ValueError, match="boundar"):
        validate_preprocessing_execution(receipt, spec)


@pytest.mark.parametrize(
    "field,value",
    [("backend", "identity"), ("sos", []), ("padding_limit_samples", 10)],
)
def test_receipt_rejects_filter_claims_that_contradict_its_declaration(field, value):
    spec = _spec()
    receipt = deepcopy(execute_preprocessing(_raw(), _events([500]), spec).execution_receipt)
    receipt["filter"][field] = value
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = semantic_sha256(receipt)
    with pytest.raises(ValueError):
        validate_preprocessing_execution(receipt, spec)


@pytest.mark.parametrize("source_rate", [128.0, 250.0, 512.0])
@pytest.mark.parametrize("baseline_mode", ["none", "mean_only"])
def test_sample_transform_matches_independent_mne_epoch_reference(source_rate, baseline_mode):
    raw = _raw(sfreq=source_rate)
    events = _events([round(2 * source_rate)])
    epochs = mne.Epochs(
        raw, events, tmin=-0.2, tmax=1.2, baseline=None, preload=True, verbose=False,
    )
    source_samples = epochs.get_data().copy()
    if source_rate != 128.0:
        epochs.resample(
            128.0, method="fft", npad="auto", window="auto", pad="edge", verbose=False,
        )
    expected = epochs.get_data()[:, :, :179]
    if baseline_mode == "mean_only":
        # Source contract fixes the half-open prestimulus baseline to 26 model
        # samples at 128 Hz; MNE's inclusive epoch baseline is not substituted.
        expected = expected - expected[:, :, :26].mean(axis=-1, keepdims=True)
    result = transform_epoch_samples(
        source_samples, source_rate, _spec(baseline_mode=baseline_mode), require_linear=True,
    )
    np.testing.assert_array_equal(result, expected)


def test_receipt_source_identity_does_not_follow_post_import_disk_edits(tmp_path):
    import data.preprocess as preprocessing_module

    for filename in ("preprocess.py", "epochs.py"):
        source = Path(preprocessing_module.__file__).with_name(filename)
        (tmp_path / filename).write_bytes(source.read_bytes())
    source = tmp_path / "preprocess.py"
    loaded_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    module_name = "data._preprocessing_source_identity_test"
    module_spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
        source.write_bytes(source.read_bytes() + b"\n# source edited after module import\n")
        assert hashlib.sha256(source.read_bytes()).hexdigest() != loaded_digest
        result = module.execute_preprocessing(_raw(), _events([500]), _spec())
        assert result.execution_receipt["implementation"]["source_sha256"]["preprocess.py"] == loaded_digest
    finally:
        sys.modules.pop(module_name, None)
