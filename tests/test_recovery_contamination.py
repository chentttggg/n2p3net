"""Known clean voltage is the counterfactual, not a smoother-looking waveform."""

import mne
import numpy as np
import pytest

from data.brainsync import BRAIN_SYNC_PREPROCESSING, load_brainsync_session
from data.brainsync_contract import validate_brainsync_bids_session
from data.continuous_qc import ContinuousQCSpec, fit_continuous_qc
from data.epochs import materialize_dataset_identity
from data.recovery import fit_epoch_state_recovery
from tests.test_brainsync import _write_bids_session


@pytest.fixture(scope="module")
def recovery_fixture(tmp_path_factory):
    root = _write_bids_session(tmp_path_factory.mktemp("recovery") / "raw", session_id="injection",
        target=4, started_utc="2026-09-06T00:00:00Z", blocks=(tuple(range(1, 10)) * 9,))
    source = load_brainsync_session(root)
    participant = materialize_dataset_identity(source).record_for("P001").authority_key("source_or_global")
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
    recovery = fit_epoch_state_recovery(raw, preprocessing=BRAIN_SYNC_PREPROCESSING,
        channels=names, fit_sample_mask=mask, tolerance_v=1e-6, inflation=2.,
        warmup_samples=7500, probe_step_samples=125, fit_provenance=scope)
    model = fit_continuous_qc(raw, spec=ContinuousQCSpec(12, 1e-7, 25, channel_local=True,
        include_eog=True, reference_channel="Cz", minimum_channel_fraction=.5, range_window_samples=351),
        fit_sample_mask=mask, fit_provenance=scope, epoch_recovery=recovery)
    neural = 4e-6 * np.sin(2*np.pi*.8*time)[None, :] * np.linspace(.8, 1.2, len(names))[:, None]
    p300 = sum(20e-6*np.exp(-.5*((time-onset-.35)/.16)**2) for onset in np.arange(4, time[-1]-3, 6))
    neural += np.array([.3,.4,.8,1.,.8,.4,0.,.7])[:,None] * p300[None,:]
    neural += np.linspace(.2,.8,len(names))[:,None] * (15e-6*np.exp(-.5*((time-50)/2.5)**2))[None,:]
    return root, raw, neural, time, model


@pytest.mark.parametrize("artifact", ["pulse", "plateau", "step", "exponential", "exponential_2s", "exponential_8s", "multiple", "reference"])
def test_injected_contamination_is_excluded_and_retained_voltage_stays_within_budget(recovery_fixture, monkeypatch, artifact, record_property):
    root, raw, neural, time, model = recovery_fixture
    clean = mne.io.RawArray(neural, raw.info.copy(), verbose=False)
    monkeypatch.setattr("data.brainsync.read_raw", lambda *args, **kwargs: clean.copy())
    expected = load_brainsync_session(root, continuous_qc_model=model)
    assert expected.n_epochs == 81
    assert expected.trial_channel_mask is None
    contaminated = clean.copy()
    channel = 1 if artifact == "reference" else 3
    start = 30 * 250
    if artifact == "pulse":
        contaminated._data[channel, start] += .0142
    elif artifact in {"plateau", "reference"}:
        contaminated._data[channel, start:start+250] += .0142
    elif artifact == "step":
        contaminated._data[channel, start:] += .0142
    elif artifact.startswith("exponential"):
        decay = {"exponential": .35, "exponential_2s": 2., "exponential_8s": 8.}[artifact]
        contaminated._data[channel, start:] += .0142 * np.exp(-(time[start:]-30)/decay)
    else:
        for begin in (30, 36, 51):
            contaminated._data[channel, begin*250:begin*250+175] += .0142
    monkeypatch.setattr("data.brainsync.read_raw", lambda *args, **kwargs: contaminated.copy())
    actual = load_brainsync_session(root, continuous_qc_model=model)
    assert actual.n_epochs >= 30
    rows = actual.metadata["scheduled_event_index"].to_numpy(dtype=int)
    observed = np.broadcast_to(actual.channel_mask, actual.X.shape[:2]) if actual.trial_channel_mask is None else actual.trial_channel_mask
    error = np.abs(actual.X - expected.X[rows])
    record_property("retained_epochs", actual.n_epochs)
    record_property("max_retained_channel_error_uv", float(error[observed].max()) * 1e6)
    assert float(error[observed].max()) <= 1e-6
    if artifact != "reference":
        assert actual.n_epochs == 81
        unaffected = actual.channel_names.index("FZ")
        np.testing.assert_array_equal(actual.X[:, unaffected], expected.X[:, unaffected])
