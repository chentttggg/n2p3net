"""Double-pass state coupling, empirical fit scope and drift-aware QC counterexamples."""

from dataclasses import replace

import mne
import numpy as np
import pytest
from scipy.signal import sosfilt, sosfilt_zi

from data.bidirectional import BidirectionalStateRecovery, fit_bidirectional_state_recovery
from data.continuous_qc import ContinuousQCSpec, continuous_qc_from_record, fit_continuous_qc
from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from data.preprocess import (
    design_continuous_sos,
    filter_bidirectional_segment,
    transform_epoch_samples,
)
from data.recovery import epoch_linear_projection, epoch_recovery_from_record, modal_system

SPEC = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
PROVENANCE = dict(participant_key="synthetic", source_reference="synthetic_ref",
                  calibration_session_ids=["calibration"], calibration_end_timestamp_s=1000.)


def raw_signal(values):
    return mne.io.RawArray(np.array(values, dtype=float, copy=True)[None],
                           mne.create_info(["Cz"], 250, "eeg"), verbose=False)


def sos_and_padding():
    sos = design_continuous_sos(250, l_freq=.1, h_freq=30, notch_freqs_hz=(50,))
    padding = int(mne.filter.construct_iir_filter(dict(sos=sos), verbose=False)["padlen"])
    return sos, padding


@pytest.mark.parametrize("source_rate", [128., 250., 500.5])
def test_cached_linear_projection_matches_actual_epoch_operator(source_rate):
    samples = round(SPEC.tmax_ms/1000*source_rate)-round(SPEC.tmin_ms/1000*source_rate)+1
    values = np.random.default_rng(18).normal(0, 1e-5, (7, samples))
    actual = values @ epoch_linear_projection(source_rate, SPEC)
    np.testing.assert_allclose(actual, transform_epoch_samples(values, source_rate, SPEC),
                               rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("length", [1, 2, 500, 2500])
def test_shared_segment_exactly_replays_mne_and_keeps_both_effective_states(length):
    values = np.random.default_rng(4).normal(0, 5e-6, (2, length)) + .02
    sos, padding = sos_and_padding()
    result = filter_bidirectional_segment(sos, values, padding_limit=padding)
    expected = mne.filter.filter_data(values, 250, .1, 30, method="iir", iir_params=dict(sos=sos),
                                      phase="zero", verbose=False)
    np.testing.assert_array_equal(result.data, expected)
    forward, _ = sosfilt(sos, values, axis=-1, zi=result.left_state)
    reverse, _ = sosfilt(sos, forward[:, ::-1], axis=-1, zi=result.right_state)
    np.testing.assert_array_equal(reverse[:, ::-1], result.data)


def test_two_effective_state_errors_bound_actual_epoch_and_reference_projection():
    sos, padding = sos_and_padding()
    t = np.arange(12000)/250
    values = np.sin(2*np.pi*.06*t)*8e-4
    values[7000:] += 2e-3
    left, right = 4000, 6500
    full = filter_bidirectional_segment(sos, values[None], padding_limit=padding)
    _, reference_left = sosfilt(sos, values[None, :left], axis=-1, zi=full.left_state)
    _, reference_right = sosfilt(sos, full.forward_core[:, right:][:, ::-1], axis=-1, zi=full.right_state)
    local = filter_bidirectional_segment(sos, values[None, left:right], padding_limit=padding)
    delta_left = local.left_state-reference_left
    delta_right = local.right_state-reference_right
    first, _ = sosfilt(sos, np.zeros((1, right-left)), axis=-1, zi=delta_left)
    second, _ = sosfilt(sos, first[:, ::-1], axis=-1, zi=delta_right)
    delta = local.data-full.data[:, left:right]
    np.testing.assert_allclose(delta, second[:, ::-1], atol=2e-12, rtol=0)
    omitted = sosfilt(sos, first[:, ::-1], axis=-1)[:, ::-1]
    assert np.max(np.abs(delta-omitted)) > 100e-6
    inverse = modal_system(sos)[2]
    gamma = np.concatenate([np.abs(inverse @ d.ravel()) for d in (delta_left, delta_right)])[None]
    model = BidirectionalStateRecovery.from_envelope(sos, 250, SPEC, ("Cz",), gamma,
                                                     tolerance_v=1e-6, fit_provenance=PROVENANCE)
    starts = np.array([0, 900, 2149])
    bounds = model.epoch_error_bounds(right-left, starts)
    for start, bound in zip(starts, bounds[:, 0], strict=True):
        actual = transform_epoch_samples(delta[:, start:start+351], 250, SPEC)
        assert np.abs(actual).max() <= bound+2e-12
    projected = model.project([[2]], ("Pz",))
    np.testing.assert_allclose(projected.epoch_error_bounds(right-left, starts), 2*bounds, rtol=1e-12)
    restored = epoch_recovery_from_record(projected.record())
    assert restored.record() == projected.record()
    np.testing.assert_array_equal(restored.epoch_error_bounds(right-left, starts), 2*bounds)


def test_fit_cannot_read_outside_clean_calibration_mask_and_rejects_single_pass_contract():
    t = np.arange(20000)/250
    values = 20e-6*np.sin(2*np.pi*.2*t) + np.random.default_rng(7).normal(0, 2e-6, len(t))
    mask = np.zeros(len(t), dtype=bool)
    mask[2500:17500] = True
    options = dict(preprocessing=SPEC, channels=("Cz",), fit_sample_mask=mask, tolerance_v=1e-6,
                   inflation=2., warmup_samples=1000, probe_step_samples=2000, fit_provenance=PROVENANCE)
    first = fit_bidirectional_state_recovery(raw_signal(values), **options)
    changed = values.copy()
    changed[~mask] += 5
    second = fit_bidirectional_state_recovery(raw_signal(changed), **options)
    assert first.record() == second.record()
    assert first.fit_provenance["probe_count"] >= 2
    with pytest.raises(ValueError, match="bidirectional recovery requires"):
        replace(first, preprocessing=replace(SPEC, filter_phase="forward",
            causal_iir_initial_state="steady_state_first_sample", zero_phase_edge_policy="legacy"))


def test_raw_range_can_be_diagnostic_while_spikes_and_flatline_stay_rejected():
    t = np.arange(15000)/250
    calibration = raw_signal(2e-6*np.sin(2*np.pi*8*t))
    spec = ContinuousQCSpec(difference_z=8, difference_floor_v=1e-6, settle_samples=25,
        channel_local=True, range_window_samples=250, range_floor_v=20e-6,
        flatline_min_duration_s=5, range_action="diagnostic")
    sos, _ = sos_and_padding()
    recovery = BidirectionalStateRecovery.from_envelope(sos, 250, SPEC, ("Cz",), np.zeros((1, 20)),
                                                         tolerance_v=1e-6, fit_provenance=PROVENANCE)
    model = fit_continuous_qc(calibration, spec=spec, fit_sample_mask=np.ones(len(t), bool),
                              epoch_recovery=recovery, fit_provenance=PROVENANCE)
    assert continuous_qc_from_record(model.record()).record() == model.record()
    drift = 1e-3*np.sin(2*np.pi*.04*t)
    raw = raw_signal(drift)
    assert not model.detect(raw).intervals
    assert model.signal_diagnostics(raw)["raw_range_monitor"]["exceeded_by_channel"][0] > 0
    rejected = replace(model, spec=replace(spec, range_action="reject")).detect(raw)
    assert any(row.reset_filter for row in rejected.intervals)
    changed = drift.copy()
    changed[2500] += 1e-3
    changed[5000:6501] = 0
    reasons = {row.reason for row in model.detect(raw_signal(changed)).intervals}
    assert "raw_exact_constant" in reasons and "raw_dependency_outlier" in reasons


def test_boundary_assessment_checks_both_segment_edges_and_preserves_artifact_holes():
    sos, _ = sos_and_padding()
    poles, _, inverse, _, _ = modal_system(sos)
    gamma = np.abs(inverse @ (sosfilt_zi(sos)*1e-4).ravel())
    model = BidirectionalStateRecovery.from_envelope(sos, 250, SPEC, ("Cz",),
        np.r_[gamma, gamma][None], tolerance_v=1e-6, fit_provenance=PROVENANCE,
        mode_identity=np.column_stack((poles.real, poles.imag)))
    raw = raw_signal(np.zeros(15000))
    raw.set_annotations(mne.Annotations([30], [1], ["QUALITY_LOCAL_ARTIFACT"], ch_names=[("Cz",)]))
    onsets = np.array([50, 7400, 7600, 7800, 14649])
    mask, record = model.assess_epochs(raw, onsets)
    assert not mask.any()
    assert record["unsupported_by_channel"][0] >= 1
    assert not record["monotone_release_time_assumed"]
    natural = raw_signal(np.zeros(15000))
    accepted, diagnostics = model.assess_epochs(natural, onsets, boundary_action="artifact_edges")
    assert accepted.all()
    assert np.max(np.asarray(diagnostics["epoch_error_bounds_v"])) > model.tolerance_v
    assert np.max(np.asarray(diagnostics["hard_error_bounds_v"])) == 0


@pytest.mark.parametrize("description", ["BAD_continuous_artifact", "EDGE unexplained_transition"])
def test_global_artifact_and_unclassified_reset_cannot_claim_natural_boundary(description):
    sos, _ = sos_and_padding()
    gamma = np.abs(modal_system(sos)[2] @ (sosfilt_zi(sos)*1e-4).ravel())
    model = BidirectionalStateRecovery.from_envelope(sos, 250, SPEC, ("Cz",),
        np.r_[gamma, gamma][None], tolerance_v=1e-6, fit_provenance=PROVENANCE)
    local = raw_signal(np.zeros(15000))
    local.set_annotations(mne.Annotations([30], [1], ["QUALITY_LOCAL_ARTIFACT"], ch_names=[("Cz",)]))
    global_raw = raw_signal(np.zeros(15000))
    global_raw.set_annotations(mne.Annotations([30], [1], [description]))
    onsets = np.array([7000, 8000, 7600])
    expected, local_record = model.assess_epochs(local, onsets, boundary_action="artifact_edges")
    actual, record = model.assess_epochs(global_raw, onsets, boundary_action="artifact_edges")
    assert not expected.any()
    np.testing.assert_array_equal(actual, expected)
    assert record["hard_error_bounds_v"] == local_record["hard_error_bounds_v"]
    assert record["unsupported_by_channel"] == [1]


def test_natural_boundary_cannot_hide_nonfinite_computed_error(monkeypatch):
    sos, _ = sos_and_padding()
    model = BidirectionalStateRecovery.from_envelope(sos, 250, SPEC, ("Cz",), np.zeros((1, 20)),
                                                     tolerance_v=1e-6, fit_provenance=PROVENANCE)
    monkeypatch.setattr(BidirectionalStateRecovery, "epoch_error_bounds",
                        lambda self, size, starts: np.full((len(starts), 1), np.inf))
    mask, record = model.assess_epochs(raw_signal(np.zeros(15000)), np.array([500]),
                                      boundary_action="artifact_edges")
    assert not mask.any()
    assert record["hard_error_bounds_v"] == [[None]]


def test_coincident_acquisition_and_artifact_edges_keep_channel_specific_gate():
    sos, _ = sos_and_padding()
    gamma = np.abs(modal_system(sos)[2] @ (sosfilt_zi(sos)*1e-4).ravel())
    model = BidirectionalStateRecovery.from_envelope(sos, 250, SPEC, ("Cz", "Pz"),
        np.tile(np.r_[gamma, gamma], (2, 1)), tolerance_v=1e-6, fit_provenance=PROVENANCE)
    raw = mne.io.RawArray(np.zeros((2, 15000)), mne.create_info(["Cz", "Pz"], 250, "eeg"), verbose=False)
    raw.set_annotations(mne.Annotations([30, 30], [0, 0],
        ["EDGE boundary", "QUALITY_LOCAL_ARTIFACT"], ch_names=[(), ("Cz",)]))
    mask, record = model.assess_epochs(raw, np.array([7000, 8000]), boundary_action="artifact_edges")
    np.testing.assert_array_equal(mask, [[False, True], [False, True]])
    np.testing.assert_array_equal(np.asarray(record["epoch_error_bounds_v"])[:, 0],
                                  np.asarray(record["epoch_error_bounds_v"])[:, 1])
    assert record["filter_segment_boundaries"]["Cz"][0]["right_causes"] == ["acquisition_discontinuity", "artifact"]
    assert record["filter_segment_boundaries"]["Pz"][0]["right_causes"] == ["acquisition_discontinuity"]


@pytest.mark.parametrize("boundary,onset", [(0., 50), (60., 14649)])
def test_artifact_at_record_endpoint_cannot_become_a_natural_edge(boundary, onset):
    sos, _ = sos_and_padding()
    gamma = np.abs(modal_system(sos)[2] @ (sosfilt_zi(sos)*1e-4).ravel())
    model = BidirectionalStateRecovery.from_envelope(sos, 250, SPEC, ("Cz",),
        np.r_[gamma, gamma][None], tolerance_v=1e-6, fit_provenance=PROVENANCE)
    raw = raw_signal(np.zeros(15000))
    raw.set_annotations(mne.Annotations([boundary], [0], ["BAD_continuous_artifact"]))
    mask, _ = model.assess_epochs(raw, np.array([onset]), boundary_action="artifact_edges")
    assert not mask.any()


def test_zero_phase_inner_split_rejects_stale_early_availability():
    from tests.support.datasets import binary_dataset
    from transfer.calibration_split import dataset_calibration_inner_split

    dataset = binary_dataset(400)
    dataset.preprocessing = replace(dataset.preprocessing, filter_phase="zero",
        causal_iir_initial_state="not_applicable", zero_phase_edge_policy="mne_reflect_limited")
    dataset.event_timeline = replace(dataset.event_timeline, online_causal=False)
    # Even an old/artificial ledger claiming early availability cannot enable a
    # prefix/suffix split of a recording processed bidirectionally as a whole.
    split = dataset_calibration_inner_split(dataset)
    assert not split.legal and "independent complete recordings" in split.reason
