"""Signal, time, recovery and artifact-contract counterexamples for mains rejection."""

from dataclasses import asdict, replace

import mne
import numpy as np
import pytest
import torch
from scipy.signal import sosfilt, sosfilt_zi, sosfreqz

from data.brainsync import (
    BRAIN_SYNC_PREPROCESSING,
    load_brainsync_inference_session,
    load_brainsync_session,
)
from data.contract import (
    BRAIN_SYNC_CAUSAL_DATA_CONTRACT,
    CAUSAL_IIR_INITIAL_STATE,
    SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
)
from data.epochs import (
    PreprocessingSpec,
    bind_named_preprocessing_contract,
    load_epoch_dataset,
    preprocessing_spec_from_contract,
    save_epoch_dataset,
)
from data.preprocess import apply_trial_baseline, design_continuous_sos, filter_continuous
from data.recovery import EpochStateRecovery, epoch_recovery_from_record, filter_sos
from research.contracts import TrainingRunContract
from tests.support.brainsync import write_bids_session
from tests.support.checkpoints import write_checkpoint
from transfer.calibrated import (
    fit_calibrated_predictor,
    load_calibrated_predictor,
    load_calibration_source_checkpoint,
    save_calibrated_predictor,
)


def raw_signal(values, sfreq=250.0):
    return mne.io.RawArray(
        np.array(values, copy=True)[None], mne.create_info(["Cz"], sfreq, "eeg"), verbose=False
    )


OPTIONS = dict(
    l_freq=0.1,
    h_freq=30.0,
    phase="forward",
    notch_freqs_hz=(50.0,),
    notch_quality_factor=30.0,
    causal_iir_initial_state=CAUSAL_IIR_INITIAL_STATE,
)


@pytest.mark.parametrize("mains", [50.0, 60.0])
def test_mains_rejection_preserves_passband_and_agrees_with_frequency_response(mains):
    rate = 250.0
    time = np.arange(30 * int(rate)) / rate
    signal = (
        10e-6 * np.sin(2 * np.pi * 10 * time) + 100e-6 * np.sin(2 * np.pi * mains * time) + 0.02
    )
    result = filter_continuous(
        raw_signal(signal), **{**OPTIONS, "notch_freqs_hz": (mains,)}
    ).get_data()[0]
    # Fit only fixed sin/cos components plus intercept in the late steady-state interval.
    t = time[20 * int(rate) :]
    basis = np.column_stack(
        [np.sin(2 * np.pi * f * t) for f in (10, mains)]
        + [np.cos(2 * np.pi * f * t) for f in (10, mains)]
        + [np.ones(len(t))]
    )
    coefficients = np.linalg.lstsq(basis, result[-len(t) :], rcond=None)[0]
    assert np.hypot(coefficients[1], coefficients[3]) < 1e-9
    assert np.hypot(coefficients[0], coefficients[2]) == pytest.approx(10e-6, rel=0.001)
    sos = design_continuous_sos(rate, l_freq=0.1, h_freq=30, notch_freqs_hz=(mains,))
    plain = design_continuous_sos(rate, l_freq=0.1, h_freq=30)
    frequencies = np.linspace(1, 30, 291)
    _, h = sosfreqz(sos, worN=frequencies, fs=rate)
    _, h0 = sosfreqz(plain, worN=frequencies, fs=rate)
    assert np.max(np.abs(np.abs(h / h0) - 1)) < 0.002


@pytest.mark.parametrize("phase", ["forward", "zero"])
def test_notch_resets_at_acquisition_breaks_and_never_filters_eog(phase):
    options = {
        **OPTIONS,
        "phase": phase,
        "causal_iir_initial_state": CAUSAL_IIR_INITIAL_STATE
        if phase == "forward"
        else "not_applicable",
    }
    t = np.arange(2000) / 250
    left = raw_signal(0.01 + 30e-6 * np.sin(2 * np.pi * 50 * t))
    right = raw_signal(-0.02 + 15e-6 * np.sin(2 * np.pi * 50 * t + 0.7))
    combined = mne.concatenate_raws([left.copy(), right.copy()], verbose=False)
    expected = np.concatenate(
        [filter_continuous(x.copy(), **options).get_data() for x in [left, right]], axis=1
    )
    actual = filter_continuous(combined, **options).get_data()
    np.testing.assert_array_equal(actual, expected)
    raw = mne.io.RawArray(
        np.vstack([t, t]), mne.create_info(["Cz", "EOG"], 250, ["eeg", "eog"]), verbose=False
    )
    result = filter_continuous(raw, **options)
    np.testing.assert_array_equal(result.get_data(picks=["EOG"])[0], t)


def test_notch_is_causal_and_state_carry_matches_whole_signal():
    rng = np.random.default_rng(4)
    x = 0.02 + rng.normal(0, 1e-5, 4000)
    changed = x.copy()
    changed[2000:] += 0.1
    first = filter_continuous(raw_signal(x), **OPTIONS).get_data()[0]
    second = filter_continuous(raw_signal(changed), **OPTIONS).get_data()[0]
    np.testing.assert_array_equal(first[:2000], second[:2000])
    sos = filter_sos(250, preprocessing_spec_from_contract(BRAIN_SYNC_CAUSAL_DATA_CONTRACT))
    state = sosfilt_zi(sos) * x[0]
    chunks = []
    for chunk in np.array_split(x, 13):
        values, state = sosfilt(sos, chunk, zi=state)
        chunks.append(values)
    np.testing.assert_array_equal(first, np.concatenate(chunks))


@pytest.mark.parametrize(
    "frequencies,q",
    [
        ([50, 50], 30),
        ([0], 30),
        ([-50], 30),
        ([np.nan], 30),
        ([True], 30),
        ([50], 0),
        ([50], np.inf),
        (50, 30),
    ],
)
def test_invalid_notch_declaration_fails_before_filtering(frequencies, q):
    with pytest.raises(ValueError, match="notch"):
        replace(
            BRAIN_SYNC_PREPROCESSING, notch_freqs_hz=frequencies, notch_quality_factor=q
        ).validate()


def test_notch_uses_source_nyquist_and_preserves_no_notch_serialization():
    spec = replace(BRAIN_SYNC_PREPROCESSING, sfreq=64, n_times=89)
    spec.validate()  # Notch precedes model resampling; source still runs at 250 Hz.
    with pytest.raises(ValueError, match="Nyquist"):
        filter_continuous(raw_signal(np.ones(1000), 100), **OPTIONS)
    legacy = preprocessing_spec_from_contract(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT)
    old = asdict(legacy)
    old.pop("notch_freqs_hz")
    old.pop("notch_quality_factor")
    old.pop("zero_phase_edge_policy")
    assert PreprocessingSpec(**old).record() == old
    assert PreprocessingSpec(**BRAIN_SYNC_PREPROCESSING.record()) == BRAIN_SYNC_PREPROCESSING
    disabled = replace(legacy, notch_quality_factor=20.0)
    assert PreprocessingSpec(**disabled.record()) == disabled


def test_notch_recovery_certifies_the_full_cascade_and_rejects_old_sos():
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_CAUSAL_DATA_CONTRACT)
    sos = filter_sos(250, spec)
    model = EpochStateRecovery.from_envelope(
        sos,
        250,
        spec,
        ("Cz",),
        np.full((1, 2 * len(sos)), 1e-5),
        tolerance_v=1e-6,
        fit_provenance={"scope": "synthetic_known_state_box"},
    )
    restored = epoch_recovery_from_record(model.record())
    assert restored.record() == model.record()
    wait = restored.recovery_samples()[0]
    assert restored.epoch_error_bound(wait)[0] <= 1e-6
    rng = np.random.default_rng(9)
    for _ in range(3):
        state = rng.normal(0, 1e-8, (len(sos), 2))
        scale = np.max(np.abs(model.operators[1] @ state.ravel()))
        state *= min(1.0, 9e-6 / scale)
        assert model.contains_state_error(state.ravel(), 0)
        waveform, _ = sosfilt(sos, np.zeros(wait + 351), zi=state)
        epoch = mne.EpochsArray(
            waveform[None, None, wait : wait + 351],
            mne.create_info(["Cz"], 250, "eeg"),
            tmin=-0.2,
            baseline=None,
            verbose=False,
        )
        epoch.resample(128, npad="auto", window="auto", pad="edge", verbose=False)
        values = apply_trial_baseline(
            epoch.get_data()[:, :, :179], sfreq=128, tmin=-0.2, baseline_mode="mean_only"
        )
        assert np.abs(values).max() <= restored.epoch_error_bound(wait)[0]
    old = replace(spec, notch_freqs_hz=())
    with pytest.raises(ValueError, match="SOS"):
        replace(model, preprocessing=old)


def test_notch_survives_cache_and_personal_model_and_rejects_old_source(tmp_path):
    session = write_bids_session(
        tmp_path / "cal", session_id="cal", target=4, started_utc="2026-09-06T00:00:00Z"
    )
    test = write_bids_session(
        tmp_path / "test", session_id="test", target=7, started_utc="2026-09-07T00:00:00Z"
    )
    data = load_brainsync_session(session)
    assert data.preprocessing.notch_freqs_hz == (50.0,)
    cache = save_epoch_dataset(tmp_path / "epochs.npz", data)
    restored = load_epoch_dataset(cache)
    assert restored.preprocessing == data.preprocessing
    np.testing.assert_array_equal(restored.X, data.X)
    legacy = preprocessing_spec_from_contract(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT)
    old = load_brainsync_session(session, preprocessing=legacy)
    with pytest.raises(ValueError, match="notch_freqs_hz"):
        bind_named_preprocessing_contract(old, BRAIN_SYNC_CAUSAL_DATA_CONTRACT)
    checkpoint = tmp_path / "old_source.pt"
    write_checkpoint(checkpoint, old, source_snapshot_sha256="f" * 64)
    assert load_calibration_source_checkpoint(data, checkpoint)[3]["status"] == "incompatible"
    assert load_calibration_source_checkpoint(old, checkpoint)[3]["status"] == "compatible"
    incomplete = torch.load(checkpoint, weights_only=False)
    incomplete["input_preprocessing"].pop("h_freq")
    incomplete["training_contract"]["preprocessing"]["epoch"].pop("h_freq")
    incomplete["training_contract_digest"] = TrainingRunContract(
        **incomplete["training_contract"]
    ).digest()
    missing = tmp_path / "incomplete_source.pt"
    torch.save(incomplete, missing)
    assert load_calibration_source_checkpoint(old, missing)[3]["status"] == "incompatible"
    model = fit_calibrated_predictor(data)
    artifact = save_calibrated_predictor(tmp_path / "model.pt", model)
    saved = load_calibrated_predictor(artifact)
    inference = load_brainsync_inference_session(test)
    np.testing.assert_array_equal(model.predict_logit(inference), saved.predict_logit(inference))
    with pytest.raises(ValueError, match="preprocessing"):
        saved.validate_input(load_brainsync_inference_session(test, preprocessing=legacy))
