"""Known clean voltage is the counterfactual, not a smoother-looking waveform."""

from functools import partial
from types import SimpleNamespace

import mne
import numpy as np
import pytest

from data.brainsync import load_brainsync_session
from data.brainsync_contract import validate_brainsync_bids_session
from data.continuous_qc import ContinuousQCSpec, fit_continuous_qc
from data.contract import BRAIN_SYNC_CAUSAL_DATA_CONTRACT
from data.epochs import materialize_dataset_identity, preprocessing_spec_from_contract
from data.qc_features import effective_observed_mask
from data.recovery import fit_epoch_state_recovery
from tests.support.brainsync import write_bids_session

CAUSAL = preprocessing_spec_from_contract(BRAIN_SYNC_CAUSAL_DATA_CONTRACT)


@pytest.fixture(scope="module")
def recovery_fixture(tmp_path_factory):
    flashes = tuple(range(1, 10)) * 9
    subject = "P001"
    root = write_bids_session(tmp_path_factory.mktemp("recovery") / "raw", session_id="injection",
        subject_id=subject, target=4, started_utc="2026-09-06T00:00:00Z", blocks=(flashes,))
    source = load_brainsync_session(root, preprocessing=CAUSAL)
    participant = materialize_dataset_identity(source).record_for(subject).authority_key("source_or_global")
    session = validate_brainsync_bids_session(root)
    names = session.bids.channel_names
    rate = session.bids.sample_rate_hz
    count = round(session.bids.duration_seconds * rate)
    time = np.arange(count) / rate
    rng = np.random.default_rng(826)
    calibration = np.zeros((len(names), count))
    for frequency, amplitude in ((.15, 20), (.45, 10), (1.3, 7), (6, 2), (12, 3), (25, 1)):
        calibration += amplitude * 1e-6 * np.sin(2*np.pi*frequency*time[None, :] + rng.uniform(-np.pi, np.pi, (len(names), 1)))
    info = mne.create_info(list(names), rate, [kind.lower() for kind in session.bids.channel_types])
    raw = mne.io.RawArray(calibration, info, verbose=False)
    scope = {"participant_key": participant, "source_reference": session.bids.source_reference,
             "calibration_session_ids": [session.session_id], "calibration_end_timestamp_s": session.ended_timestamp_s,
             "purpose": "known synthetic calibration, independent of injected corruptions"}
    mask = np.ones(count, dtype=bool)
    recovery = fit_epoch_state_recovery(raw, preprocessing=CAUSAL,
        channels=names, fit_sample_mask=mask, tolerance_v=1e-6, inflation=2.,
        warmup_samples=round(30 * rate), probe_step_samples=round(.5 * rate), fit_provenance=scope)
    epoch_duration = (CAUSAL.tmax_ms - CAUSAL.tmin_ms) / 1000
    model = fit_continuous_qc(raw, spec=ContinuousQCSpec(
        difference_z=12, difference_floor_v=1e-7, settle_samples=round(.1 * rate), channel_local=True,
        include_eog=True, reference_channel="Cz", minimum_channel_fraction=.5,
        range_window_samples=round(epoch_duration * rate) + 1),
        fit_sample_mask=mask, fit_provenance=scope, epoch_recovery=recovery)
    neural = 4e-6 * np.sin(2*np.pi*.8*time)[None, :] * np.linspace(.8, 1.2, len(names))[:, None]
    p300 = sum(20e-6*np.exp(-.5*((time-onset-.35)/.16)**2) for onset in np.arange(4, time[-1]-3, 6))
    scalp_weights = {"Fz": .3, "Cz": .4, "P3": .8, "Pz": 1., "P4": .8, "PO8": .4, "EOG": 0., "Oz": .7}
    neural += np.array([scalp_weights[name] for name in names])[:,None] * p300[None,:]
    neural += np.linspace(.2,.8,len(names))[:,None] * (15e-6*np.exp(-.5*((time-50)/2.5)**2))[None,:]
    clean = mne.io.RawArray(neural, raw.info.copy(), verbose=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("data.brainsync.read_raw", lambda *args, **kwargs: clean.copy())
        expected = load_brainsync_session(root, preprocessing=CAUSAL, continuous_qc_model=model)
    assert expected.n_epochs == len(flashes)
    assert expected.trial_channel_mask is None
    return SimpleNamespace(root=root, clean=clean, time=time, model=model,
                           expected=expected, n_flashes=len(flashes))


def _rectangular_pulses(time, *, starts, width):
    return sum((time >= start) & (time < start + width) for start in starts)


def _exponential_step(time, *, decay_seconds):
    elapsed = np.maximum(time - 30.0, 0.0)
    return (time >= 30.0) * np.exp(-elapsed / decay_seconds)


def _load_injected_voltage(fixture, monkeypatch, waveform, *, channel, record_property):
    contaminated = fixture.clean.copy()
    contaminated._data[contaminated.ch_names.index(channel)] += .0142 * waveform(fixture.time)
    monkeypatch.setattr("data.brainsync.read_raw", lambda *args, **kwargs: contaminated.copy())
    actual = load_brainsync_session(fixture.root, preprocessing=CAUSAL, continuous_qc_model=fixture.model)
    assert actual.n_epochs >= 30
    rows = actual.metadata["scheduled_event_index"].to_numpy(dtype=int)
    observed = effective_observed_mask(n_epochs=actual.n_epochs,
        channel_mask=actual.channel_mask, trial_channel_mask=actual.trial_channel_mask)
    error = np.abs(actual.X - fixture.expected.X[rows])
    max_error_v = float(error[observed].max())
    record_property("retained_epochs", actual.n_epochs)
    record_property("max_retained_channel_error_uv", max_error_v * 1e6)
    assert max_error_v <= 1e-6
    return actual


@pytest.mark.parametrize("waveform", [
    pytest.param(lambda time: time == 30.0, id="pulse"),
    pytest.param(partial(_rectangular_pulses, starts=(30.0,), width=1.0), id="plateau"),
    pytest.param(lambda time: time >= 30.0, id="step"),
    pytest.param(partial(_exponential_step, decay_seconds=.35), id="exponential"),
    pytest.param(partial(_exponential_step, decay_seconds=2.0), id="exponential_2s"),
    pytest.param(partial(_exponential_step, decay_seconds=8.0), id="exponential_8s"),
    pytest.param(partial(_rectangular_pulses, starts=(30.0, 36.0, 51.0), width=.7), id="multiple"),
])
def test_local_contamination_preserves_other_channels_and_voltage_budget(
    recovery_fixture, monkeypatch, waveform, record_property,
):
    actual = _load_injected_voltage(recovery_fixture, monkeypatch, waveform,
                                   channel="Pz", record_property=record_property)
    assert actual.n_epochs == recovery_fixture.n_flashes
    unaffected = actual.channel_names.index("FZ")
    np.testing.assert_array_equal(actual.X[:, unaffected], recovery_fixture.expected.X[:, unaffected])


def test_reference_contamination_preserves_retained_voltage_budget(
    recovery_fixture, monkeypatch, record_property,
):
    _load_injected_voltage(recovery_fixture, monkeypatch,
        partial(_rectangular_pulses, starts=(30.0,), width=1.0),
        channel="Cz", record_property=record_property)
