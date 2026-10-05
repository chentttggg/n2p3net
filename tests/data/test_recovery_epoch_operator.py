"""Recovery certificates must use the exact finite production epoch transform."""

from dataclasses import replace

import mne
import numpy as np
import pytest

from data.bidirectional import BidirectionalStateRecovery
from data.continuous_qc import epoch_operator_gain
from data.contract import (
    BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT,
    SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
)
from data.epochs import preprocessing_spec_from_contract
from data.preprocess import apply_trial_baseline, design_continuous_sos
from data.recovery import EpochStateRecovery, epoch_linear_projection

FORWARD = preprocessing_spec_from_contract(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT)
ZERO = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)


def _mne_epoch_reference(values, source_rate, spec):
    """Independent pre-refactor production sequence, including rate skip policy."""
    epochs = mne.EpochsArray(
        values[:, None, :], mne.create_info(["Cz"], source_rate, "eeg"),
        tmin=spec.tmin_ms / 1000, baseline=None, verbose=False)
    if not np.isclose(source_rate, spec.sfreq):
        epochs.resample(spec.sfreq, method=spec.resample_method, npad=spec.resample_npad,
                        window=spec.resample_window, pad=spec.resample_pad, verbose=False)
    return apply_trial_baseline(
        epochs.get_data()[:, :, :spec.n_times], sfreq=float(epochs.info["sfreq"]),
        tmin=spec.tmin_ms / 1000, baseline_mode=spec.baseline_mode,
        trial_reference_window_ms=spec.trial_reference_window_ms,
        trial_reference_center=spec.trial_reference_center,
        trial_reference_scale=spec.trial_reference_scale)[:, 0]


@pytest.mark.parametrize("source_rate", [128., 250., 500.5, 127.999])
@pytest.mark.parametrize("baseline", [
    {"baseline_mode": "none"},
    {"baseline_mode": "mean_only"},
    {"baseline_mode": "trial_reference", "trial_reference_window_ms": (-100., 150.)},
])
def test_projection_and_gain_equal_frozen_mne_sequence(source_rate, baseline):
    spec = replace(FORWARD, **baseline)
    samples = round(spec.tmax_ms / 1000 * source_rate) - round(spec.tmin_ms / 1000 * source_rate) + 1
    expected = _mne_epoch_reference(np.eye(samples), source_rate, spec)
    actual = epoch_linear_projection(source_rate, spec)
    np.testing.assert_array_equal(actual, expected)
    expected_gain = float(np.abs(expected).sum(axis=0).max())
    assert epoch_operator_gain(source_rate, spec) == expected_gain
    # The induced infinity norm must bound actual output, with an input that
    # attains it. This detects transposed or incomplete operator norm reductions.
    column = np.argmax(np.abs(expected).sum(axis=0))
    worst = np.sign(expected[:, column])[None, :]
    observed = np.abs(_mne_epoch_reference(worst, source_rate, spec)).max()
    # Evaluating the signed waveform with FFT and summing the explicit basis
    # are different reductions; use the n-term floating-point rounding bound.
    epsilon_sum = samples * np.finfo(np.float64).eps
    np.testing.assert_allclose(observed, expected_gain,
                               rtol=epsilon_sum / (1 - epsilon_sum), atol=0)


@pytest.mark.parametrize("baseline_mode", ["none", "mean_only"])
@pytest.mark.parametrize("base, model_class, multiplicity", [
    (FORWARD, EpochStateRecovery, 2),
    (ZERO, BidirectionalStateRecovery, 4),
])
def test_unused_median_declaration_does_not_reject_a_linear_operator(baseline_mode, base, model_class, multiplicity):
    spec = replace(base, baseline_mode=baseline_mode, trial_reference_center="median")
    sos = design_continuous_sos(250., l_freq=spec.l_freq, h_freq=spec.h_freq,
                               order=spec.filter_order, notch_freqs_hz=spec.notch_freqs_hz,
                               notch_quality_factor=spec.notch_quality_factor)
    model = model_class.from_envelope(sos, 250., spec, ("Cz",),
        np.full((1, multiplicity * len(sos)), 1e-4), tolerance_v=1e-6,
        fit_provenance={"scope": "synthetic_fixed_modal_box"})
    assert model.preprocessing == spec
    np.testing.assert_array_equal(epoch_linear_projection(250., spec),
                                  epoch_linear_projection(250., replace(spec, trial_reference_center="mean")))


@pytest.mark.parametrize("base, model_class, multiplicity", [
    (FORWARD, EpochStateRecovery, 2),
    (ZERO, BidirectionalStateRecovery, 4),
])
def test_active_median_baseline_rejected_before_recovery_certificate(base, model_class, multiplicity):
    spec = replace(base, baseline_mode="trial_reference", trial_reference_window_ms=(-200., 0.),
                   trial_reference_center="median")
    sos = design_continuous_sos(250., l_freq=spec.l_freq, h_freq=spec.h_freq,
                               order=spec.filter_order, notch_freqs_hz=spec.notch_freqs_hz,
                               notch_quality_factor=spec.notch_quality_factor)
    with pytest.raises(ValueError, match="linear"):
        model_class.from_envelope(sos, 250., spec, ("Cz",),
            np.full((1, multiplicity * len(sos)), 1e-4), tolerance_v=1e-6,
            fit_provenance={"scope": "synthetic_fixed_modal_box"})
    with pytest.raises(ValueError, match="linear"):
        epoch_linear_projection(250., spec)
    with pytest.raises(ValueError, match="linear"):
        epoch_operator_gain(250., spec)


def test_projection_bounds_identity_storage_and_caches_readonly_result(monkeypatch):
    import data.recovery as recovery

    original = recovery.transform_epoch_samples
    basis_sizes = []

    def observe(values, *args, **kwargs):
        if values.size:
            basis_sizes.append(values.nbytes)
        return original(values, *args, **kwargs)

    monkeypatch.setattr(recovery, "transform_epoch_samples", observe)
    epoch_linear_projection.cache_clear()
    matrix = epoch_linear_projection(2048., FORWARD)
    assert len(basis_sizes) > 1
    assert max(basis_sizes) <= 8 * 1024 * 1024
    assert not matrix.flags.writeable
    count = len(basis_sizes)
    assert epoch_linear_projection(2048., FORWARD) is matrix
    assert len(basis_sizes) == count
    # Contracts must have a bounded cache rather than accumulate across runs.
    for number in range(5):
        epoch_linear_projection(128., replace(FORWARD, name=f"bounded-cache-{number}"))
    assert epoch_linear_projection.cache_info().currsize == 4
    epoch_linear_projection.cache_clear()


def test_projection_rejects_one_row_larger_than_basis_budget(monkeypatch):
    import data.recovery as recovery

    epoch_linear_projection.cache_clear()
    monkeypatch.setattr(recovery, "_EPOCH_BASIS_BYTES", 8)
    with pytest.raises(ValueError, match="basis budget"):
        epoch_linear_projection(250., FORWARD)


@pytest.mark.parametrize("source_rate", [0., -1., float("nan"), float("inf"), True])
def test_projection_rejects_invalid_source_rate(source_rate):
    with pytest.raises(ValueError, match="source sample rate"):
        epoch_linear_projection(source_rate, FORWARD)


def test_cached_numeric_rate_cannot_admit_boolean_rate():
    spec = replace(FORWARD, l_freq=None, h_freq=None, baseline_mode="none")
    epoch_operator_gain(1., spec)
    # bool and float compare equal; cache hits must not bypass type validation.
    with pytest.raises(ValueError, match="source sample rate"):
        epoch_operator_gain(True, spec)
