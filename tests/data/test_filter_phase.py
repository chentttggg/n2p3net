"""Physical counterexamples and ingress gates for completed-recording filtering."""

import json
from dataclasses import replace

import mne
import numpy as np
import pytest
from scipy.signal import sosfiltfilt

from data.brainsync import load_brainsync_session
from data.contract import (
    BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT,
    BRAIN_SYNC_CAUSAL_DATA_CONTRACT,
    assert_causal_p300_input_contract,
)
from data.epochs import (
    PreprocessingSpec,
    load_epoch_dataset,
    preprocessing_spec_from_contract,
    save_epoch_dataset,
)
from data.preprocess import design_continuous_sos, filter_continuous, preprocess
from data.recovery import filter_sos
from tests.support.brainsync import write_bids_session
from tests.support.checkpoints import write_checkpoint
from transfer.calibrated import (
    fit_calibrated_predictor,
    load_calibrated_predictor,
    load_calibration_source_checkpoint,
    save_calibrated_predictor,
)

OFFLINE = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
ZERO = dict(l_freq=0.1, h_freq=30, phase="zero", notch_freqs_hz=(50,),
            zero_phase_edge_policy="mne_reflect_limited")


def raw_signal(values, *, first_samp=0):
    return mne.io.RawArray(np.array(values, dtype=float, copy=True)[None], mne.create_info(["Cz"], 250, "eeg"),
                           first_samp=first_samp, verbose=False)


@pytest.mark.parametrize("notch", [(), (50,)])
def test_reverse_filter_has_future_support_but_acquisition_break_isolates_it(notch):
    values = np.zeros(10000)
    changed = values.copy()
    changed[5000:] = 1e-3
    options = {**ZERO, "notch_freqs_hz": notch}
    zero = filter_continuous(raw_signal(changed), **options).get_data()[0]
    assert np.max(np.abs(zero[4000:4990])) > 1e-5
    boundary = raw_signal(changed, first_samp=123)
    boundary.set_annotations(mne.Annotations([20], [0], ["EDGE boundary"]))
    isolated = filter_continuous(boundary, **options).get_data()[0]
    np.testing.assert_allclose(isolated[:5000], 0, atol=1e-14)
    np.testing.assert_allclose(isolated[5000:], 0, atol=1e-14)


def test_legacy_notch_waveform_remains_scipy_default():
    values = np.random.default_rng(7).normal(0, 1e-5, 3000)
    sos = design_continuous_sos(250, l_freq=.1, h_freq=30, notch_freqs_hz=(50,))
    actual = filter_continuous(raw_signal(values), **{**ZERO, "zero_phase_edge_policy": "legacy"})
    np.testing.assert_array_equal(actual.get_data()[0], sosfiltfilt(sos, values))


@pytest.mark.parametrize("size", [1, 2, 32, 250, 5000])
def test_declared_padding_handles_short_constant_segments_without_inventing_signal(size):
    actual = filter_continuous(raw_signal(np.full(size, .02)), **ZERO).get_data()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, 0, atol=1e-13)


@pytest.mark.parametrize("phase,policy", [("zero", "legacy"), ("zero", "mne_reflect_limited"),
                                         ("forward", "legacy")])
def test_epoch_availability_accounts_for_reverse_filter_support(phase, policy):
    raw = raw_signal(np.random.default_rng(12).normal(0, 1e-5, 5000), first_samp=1000)
    result = preprocess(raw, np.array([[1500, 0, 1], [5990, 0, 1]]), filter_phase=phase,
                        notch_freqs_hz=(50,), zero_phase_edge_policy=policy,
                        causal_iir_initial_state=("steady_state_first_sample" if phase == "forward"
                                                  else "not_applicable"))
    assert result.evidence_available_times_s[0] == pytest.approx(7.2 if phase == "forward" else 24)
    assert np.isnan(result.evidence_available_times_s[1])


def test_edge_policy_round_trip_and_forward_recovery_rejection(tmp_path):
    causal = preprocessing_spec_from_contract(BRAIN_SYNC_CAUSAL_DATA_CONTRACT)
    assert "zero_phase_edge_policy" not in causal.record()
    assert PreprocessingSpec(**json.loads(json.dumps(OFFLINE.record()))) == OFFLINE
    with pytest.raises(ValueError, match="zero_phase_edge_policy"):
        replace(causal, zero_phase_edge_policy="mne_reflect_limited").validate()
    with pytest.raises(ValueError, match="zero_phase_edge_policy"):
        replace(OFFLINE, zero_phase_edge_policy="unknown").validate()
    with pytest.raises(ValueError, match="forward"):
        filter_sos(250, OFFLINE)
    with pytest.raises(ValueError, match="filter_phase"):
        assert_causal_p300_input_contract(OFFLINE)
    root = write_bids_session(tmp_path / "raw", session_id="independent", target=2,
                              started_utc="2026-09-09T00:00:00+00:00")
    dataset = load_brainsync_session(root, preprocessing=OFFLINE)
    restored = load_epoch_dataset(save_epoch_dataset(tmp_path / "offline.npz", dataset))
    assert restored.preprocessing == OFFLINE
    np.testing.assert_array_equal(restored.X, dataset.X)
    assert not restored.event_timeline.online_causal
    assert len(np.unique(restored.event_timeline.evidence_available_times_s)) == 1
    source = tmp_path / "forward_source.pt"
    forward = load_brainsync_session(root, preprocessing=causal)
    write_checkpoint(source, forward, source_snapshot_sha256="f" * 64)
    assert load_calibration_source_checkpoint(restored, source)[3]["status"] == "incompatible"
    model = fit_calibrated_predictor(restored, declared_preprocessing=OFFLINE)
    saved = load_calibrated_predictor(save_calibrated_predictor(tmp_path / "bidirectional.pt", model))
    np.testing.assert_array_equal(saved.predict_logit(restored), model.predict_logit(restored))


def test_retrospective_r5_uses_first_five_cycles_of_one_session_decision(tmp_path):
    from experiments.run_filter_phase_study import candidate_metrics

    root = write_bids_session(tmp_path / "repeated", session_id="repeated", target=2,
                              started_utc="2026-09-09T00:00:00+00:00",
                              blocks=(tuple(range(1, 10)) * 5,) * 2)
    dataset = load_brainsync_session(root)
    scores = dataset.y.astype(float)
    summary = candidate_metrics(dataset, scores)
    assert summary["requested"] == summary["eligible"] == summary["hits"] == 1
    timeline = dataset.event_timeline
    later = timeline.evidence_indices[timeline.repetition_indices >= 5]
    later = later[later >= 0]
    scores[later] = -1e9
    assert candidate_metrics(dataset, scores) == summary


def test_new_zero_policy_isolates_local_artifacts_and_preserves_eog_role():
    raw = mne.io.RawArray(np.vstack([np.arange(2500) * 1e-8] * 2),
                         mne.create_info(["Cz", "EOG"], 250, ["eeg", "eog"]), verbose=False)
    eog = raw.get_data(picks=["EOG"])
    filtered = filter_continuous(raw.copy(), **ZERO)
    np.testing.assert_array_equal(filtered.get_data(picks=["EOG"]), eog)
    raw.set_annotations(mne.Annotations([1], [1], ["quality_local_artifact"], ch_names=[("Cz",)]))
    result = filter_continuous(raw.copy(), **ZERO).get_data()[0]
    for start, stop in ((0, 250), (500, 2500)):
        expected = filter_continuous(raw_signal(raw.get_data()[0, start:stop]), **ZERO).get_data()[0]
        np.testing.assert_array_equal(result[start:stop], expected)
