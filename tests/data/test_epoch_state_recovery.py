from dataclasses import replace

import mne
import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from data.contract import SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from data.preprocess import apply_trial_baseline

BANDPASS_PREPROCESSING = preprocessing_spec_from_contract(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT)

def test_epoch_modal_bound_covers_true_state_errors_and_spatial_mixing():
    from data.recovery import EpochStateRecovery

    sos = butter(4, [.1, 30], btype="bandpass", fs=250, output="sos")
    envelope = np.full((2, 8), 1e-4)
    model = EpochStateRecovery.from_envelope(sos, 250, BANDPASS_PREPROCESSING,
        ("Cz", "Pz"), envelope, tolerance_v=1e-6,
        fit_provenance={"scope": "known_synthetic_state_box"})
    reference = model.project(np.array([[-1., 1.]]), ("Pz",))
    wait = reference.recovery_samples()[0]
    assert wait > 0
    rng = np.random.default_rng(41)
    for _ in range(5):
        state = rng.normal(0, 1e-6, (2, 4, 2))
        for channel in range(2):
            magnitude = np.max(np.abs(model.operators[1] @ state[channel].ravel()))
            state[channel] *= min(1., 9e-5 / magnitude)
            assert model.contains_state_error(state[channel].ravel(), channel)
        outputs = [sosfilt(sos, np.zeros(wait + 1000), zi=values)[0] for values in state]
        waveform = outputs[1] - outputs[0]
        epoch = mne.EpochsArray(waveform[None, None, wait:wait+351],
            mne.create_info(["Pz"], 250, "eeg"), tmin=-.2, verbose=False)
        epoch.resample(128, npad="auto", window="auto", pad="edge", verbose=False)
        values = apply_trial_baseline(epoch.get_data()[:, :, :179], sfreq=128,
                                      tmin=-.2, baseline_mode="mean_only")
        assert np.abs(values).max() <= reference.epoch_error_bound(wait)[0] <= 1e-6


def test_serialized_modal_coordinates_survive_equivalent_eigenpair_permutation(monkeypatch):
    from data.recovery import EpochStateRecovery, epoch_recovery_from_record

    sos = butter(4, [.1,30], btype="bandpass", fs=250, output="sos")
    model = EpochStateRecovery.from_envelope(sos, 250, BANDPASS_PREPROCESSING, ("Cz",),
        [np.geomspace(1e-8, 1e-3, 8)], tolerance_v=1e-6, fit_provenance={"scope": "known_box"})
    record, expected = model.record(), model.recovery_samples()
    eig = np.linalg.eig
    def reversed_eig(values):
        poles, vectors = eig(values)
        return poles[::-1], vectors[:, ::-1]
    monkeypatch.setattr(np.linalg, "eig", reversed_eig)
    restored = epoch_recovery_from_record(record)
    assert restored.record() == record
    assert restored.recovery_samples() == expected


def test_serialized_modal_identity_preserves_digest_across_lapack_rounding():
    import copy

    from data.recovery import EpochStateRecovery, epoch_recovery_from_record

    model = EpochStateRecovery.from_envelope(
        butter(4, [.1, 30], btype="bandpass", fs=250, output="sos"),
        250, BANDPASS_PREPROCESSING, ("Cz",), [np.geomspace(1e-8, 1e-3, 8)],
        tolerance_v=1e-6, fit_provenance={"scope": "known_box"})
    record = model.record()
    record["mode_identity"][0][0] += 5e-13
    restored = epoch_recovery_from_record(record)
    assert restored.record() == record
    assert restored.recovery_samples() == model.recovery_samples()
    np.testing.assert_array_equal(restored.epoch_error_bound(1000), model.epoch_error_bound(1000))
    changed = copy.deepcopy(record)
    changed["mode_identity"][0][0] += 1e-6
    with pytest.raises(ValueError, match="modal identity"):
        epoch_recovery_from_record(changed)
    changed = copy.deepcopy(record)
    changed["mode_identity"].reverse()
    with pytest.raises(ValueError, match="modal identity"):
        epoch_recovery_from_record(changed)


@pytest.mark.parametrize("change", [
    {"participant_key": "OTHER"}, {"source_reference": "OTHER"},
    {"calibration_session_ids": ["future"]}, {"calibration_end_timestamp_s": 20.},
])
def test_nested_recovery_authority_is_checked_before_raw_processing(change):
    from data.continuous_qc import ContinuousQCSpec, fit_continuous_qc
    from data.recovery import EpochStateRecovery

    raw = mne.io.RawArray(np.sin(np.arange(1000)/20)[None,:]*1e-6,
                         mne.create_info(["Cz"],250,"eeg"), verbose=False)
    provenance = {"participant_key": "p1", "source_reference": "A2",
                  "calibration_session_ids": ["cal"], "calibration_end_timestamp_s": 10.}
    recovery = EpochStateRecovery.from_envelope(butter(4,[.1,30],btype="bandpass",fs=250,output="sos"),
        250, BANDPASS_PREPROCESSING, ("Cz",), np.full((1,8),1e-4), tolerance_v=1e-6,
        fit_provenance=provenance)
    with pytest.raises(ValueError, match="nested epoch recovery"):
        fit_continuous_qc(raw, spec=ContinuousQCSpec(12,1e-7,25,channel_local=True),
            fit_sample_mask=np.ones(raw.n_times,dtype=bool), fit_provenance=provenance,
            epoch_recovery=replace(recovery, fit_provenance={**provenance, **change}))
