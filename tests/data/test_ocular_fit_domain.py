"""EOG coupling must use the same band/support as its downstream EEG."""

import mne
import numpy as np
import pytest

from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from data.ocular import PRESERVE_ACQUIRED_EOG, fit_eog_regression


def _calibration(offsets=True):
    rate, seconds = 250., 200
    t = np.arange(int(rate*seconds))/rate
    eye = 80e-6*np.sin(2*np.pi*.8*t)
    brain = 8e-6*np.sin(2*np.pi*6*t)
    mains = 1000e-6*np.sin(2*np.pi*50*t)
    e = eye+mains
    x = brain+.4*eye-7.5*mains
    if offsets:
        e += np.where(t<100, .01, -.01)
        x += np.where(t<100, -.05, .05)
    raw = mne.io.RawArray(np.vstack([x,np.zeros_like(x),-.2*eye+brain,e]),
        mne.create_info(["Fz","Cz","Pz","EOG"],rate,["eeg","eeg","eeg","eog"]),verbose=False)
    raw.annotations.append(100.,0.,"EDGE boundary")
    return raw


def _fit(raw, mask=None, **kwargs):
    return fit_eog_regression(raw, participant_key="fixture:P1", source_reference="nose",
        reference_channel="Cz", eog_reference_policy=PRESERVE_ACQUIRED_EOG,
        fit_sample_mask=np.ones(raw.n_times,bool) if mask is None else mask,
        fit_provenance={"role":"source_or_C_only"}, **kwargs)


def test_matched_fit_rejects_dc_and_out_of_band_covariance():
    raw = _calibration()
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    old = _fit(raw)
    assert old.coefficients[0][0] < -4
    fixed = _fit(raw, fit_domain="matched_passband_scopes", preprocessing=spec)
    np.testing.assert_allclose(np.asarray(fixed.coefficients)[:,0],[.4,-.2],atol=.015)
    assert fixed.fit_provenance["fit_domain"] == "matched_passband_scopes"
    assert fixed.fit_provenance["fit_preprocessing"] == spec.record()
    assert fixed.fit_provenance["samples"] < raw.n_times
    assert raw.get_channel_types() == ["eeg","eeg","eeg","eog"]


def test_segment_offset_and_unauthorised_raw_cannot_change_fitted_coupling():
    raw = _calibration(False)
    mask = np.arange(raw.n_times)<45000
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    first = _fit(raw,mask,fit_domain="matched_passband_scopes",preprocessing=spec)
    shifted = raw.copy()
    shifted._data[[0,2,3],:25000] += np.array([[.08],[-.03],[.02]])
    shifted._data[[0,2,3],25000:45000] += np.array([[-.06],[.04],[-.015]])
    shifted._data[:,45000:] = 1e4*np.arange(raw.n_times-45000)
    second = _fit(shifted,mask,fit_domain="matched_passband_scopes",preprocessing=spec)
    np.testing.assert_allclose(first.coefficients,second.coefficients,atol=1e-8,rtol=1e-8)
    # Instantaneous deployment still preserves the acquired EOG channel exactly.
    corrected = second.apply(raw,source_reference="nose")
    np.testing.assert_array_equal(corrected.get_data(picks=["EOG"]),raw.get_data(picks=["EOG"]))


def test_corrected_policy_is_distinct_and_enforces_acquisition_edges():
    from data.eog_preprocessing import EOGPreprocessingPolicy

    policy = EOGPreprocessingPolicy.corrected()
    record = policy.record()
    assert record["EOG_fit_domain"] == "matched_passband_scopes"
    assert record["continuous_qc"]["boundary_action"] == "all_edges"
    assert EOGPreprocessingPolicy.from_record(record) == policy
    legacy = EOGPreprocessingPolicy("QC_eog_ols_v1",True,"record_unavailable",continuous_qc=True)
    assert legacy.record()["continuous_qc"]["boundary_action"] == "artifact_edges"
    assert EOGPreprocessingPolicy.from_record(legacy.record()) == legacy
    assert legacy.digest() != policy.digest()


def test_frozen_matched_model_recovers_independent_neural_signal_and_reloads(tmp_path):
    from dataclasses import replace

    import pytest

    from data.ocular import load_eog_regression, save_eog_regression, validate_eog_fit_preprocessing
    from data.preprocess import filter_continuous

    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    model = _fit(_calibration(),fit_domain="matched_passband_scopes",preprocessing=spec)
    held = _calibration(False)
    t = np.arange(held.n_times)/250.
    extra_eye = 4*80e-6*np.sin(2*np.pi*.8*t)
    held._data[3] += extra_eye
    held._data[0] += .4*extra_eye
    held._data[2] += -.2*extra_eye
    corrected = model.apply(held,source_reference="nose")
    clean = held.copy()
    clean._data[[0,2]] = 8e-6*np.sin(2*np.pi*6*t)

    def filtered(raw):
        return filter_continuous(raw,l_freq=spec.l_freq,h_freq=spec.h_freq,order=spec.filter_order,
            phase=spec.filter_phase,causal_iir_initial_state=spec.causal_iir_initial_state,
            notch_freqs_hz=spec.notch_freqs_hz,notch_quality_factor=spec.notch_quality_factor,
            zero_phase_edge_policy=spec.zero_phase_edge_policy,copy=True).get_data(picks=["Fz","Pz"])

    valid = ((t>30)&(t<70))|((t>130)&(t<170))
    assert np.sqrt(np.mean((filtered(corrected)[:,valid]-filtered(clean)[:,valid])**2)) < 1e-6
    restored = load_eog_regression(save_eog_regression(tmp_path/"model.json",model))
    np.testing.assert_array_equal(restored.apply(held,source_reference="nose").get_data(),corrected.get_data())
    with pytest.raises(ValueError,match="passband differs"):
        validate_eog_fit_preprocessing(model,replace(spec,l_freq=.5))


def test_source_and_target_share_the_exact_new_fit_owner():
    from data.eog_preprocessing import EOGPreprocessingPolicy, prepare_source_eog

    raw = _calibration()
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    mask = np.ones(raw.n_times,bool)
    target = _fit(raw,mask,fit_domain="matched_passband_scopes",preprocessing=spec)
    _,_,source = prepare_source_eog(raw,policy=EOGPreprocessingPolicy.corrected(),
        eeg_names=["Fz","Cz","Pz"],source_reference="nose",participant_key="fixture:P1",session_id="s1",
        fit_mask=mask,preprocessing=spec)
    np.testing.assert_array_equal(source["fitted_model"]["coefficients"],target.coefficients)
    assert source["fitted_model"]["fit_provenance"]["fit_domain"] == "matched_passband_scopes"


def test_neural_signal_also_in_eog_remains_an_explicit_counterexample():
    # Statistical projection cannot identify whether a shared waveform is
    # ocular or neural. A visually smaller output alone is not physiological validation.
    rate=250.
    t=np.arange(50000)/rate
    brain=10e-6*np.sin(2*np.pi*4*t)
    raw=mne.io.RawArray(np.vstack([brain,np.zeros_like(brain),brain,brain]),
        mne.create_info(["Fz","Cz","Pz","EOG"],rate,["eeg","eeg","eeg","eog"]),verbose=False)
    spec=preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    model=_fit(raw,fit_domain="matched_passband_scopes",preprocessing=spec)
    corrected=model.apply(raw,source_reference="nose")
    assert np.std(corrected.get_data(picks=["Fz"])) < .01*np.std(brain)


@pytest.mark.parametrize("piecewise",[False,True])
def test_nonzero_flat_eog_cannot_be_fitted_from_filter_roundoff(piecewise):
    raw=_calibration(False)
    raw._data[3,:25000]=.065
    raw._data[3,25000:]=.08 if piecewise else .065
    spec=preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    with pytest.raises(ValueError,match="flat|numerically"):
        _fit(raw,fit_domain="matched_passband_scopes",preprocessing=spec)


def test_matched_fit_rejects_too_short_authorized_interior():
    raw=_calibration(False)
    mask=np.zeros(raw.n_times,bool)
    mask[:3]=True
    spec=preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    with pytest.raises(ValueError,match="supported"):
        _fit(raw,mask,fit_domain="matched_passband_scopes",preprocessing=spec)


def test_native_legacy_fit_also_rejects_nonzero_constant_eog():
    raw=_calibration(False)
    raw._data[3]=.065
    with pytest.raises(ValueError,match="flat"):
        _fit(raw)
