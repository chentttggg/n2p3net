import mne
import numpy as np

from data.brainsync import load_brainsync_session
from data.brainsync_contract import validate_brainsync_bids_session
from data.continuous_qc import ContinuousQCSpec, fit_continuous_qc, signal_dependency_matrix
from data.contract import BRAIN_SYNC_CAUSAL_DATA_CONTRACT
from data.epochs import (
    load_epoch_dataset,
    materialize_dataset_identity,
    preprocessing_spec_from_contract,
    save_epoch_dataset,
)
from tests.support.brainsync import write_bids_session

CAUSAL = preprocessing_spec_from_contract(BRAIN_SYNC_CAUSAL_DATA_CONTRACT)


def test_local_channel_fault_retains_other_final_reference_channels_and_cache(tmp_path, monkeypatch):
    root = write_bids_session(tmp_path / "raw", session_id="local-qc", target=4,
                              started_utc="2026-09-06T00:00:00Z")
    session = validate_brainsync_bids_session(root)
    source = load_brainsync_session(root, preprocessing=CAUSAL)
    participant = materialize_dataset_identity(source).record_for("P001").authority_key("source_or_global")
    raw = mne.io.read_raw_edf(session.bids.raw_path, preload=True, verbose=False)
    raw.set_channel_types(dict(zip(raw.ch_names, [kind.lower() for kind in session.bids.channel_types], strict=True)))
    model = fit_continuous_qc(raw,
        spec=ContinuousQCSpec(12, 1e-7, 10, channel_local=True, reference_channel="Cz",
                              minimum_channel_fraction=.5, include_eog=True),
        fit_sample_mask=np.ones(raw.n_times, dtype=bool), fit_provenance={
            "participant_key": participant, "source_reference": session.bids.source_reference,
            "calibration_session_ids": [session.session_id], "calibration_end_timestamp_s": session.ended_timestamp_s})
    monkeypatch.setattr("data.brainsync.read_raw", lambda *args, **kwargs: raw.copy())
    expected = load_brainsync_session(root, preprocessing=CAUSAL, continuous_qc_model=model)
    raw._data[2, 100:] += .02
    actual = load_brainsync_session(root, preprocessing=CAUSAL, continuous_qc_model=model)
    assert actual.n_epochs == expected.n_epochs == 9
    assert actual.n_channels == 6 and actual.provenance["source_reference"] == "measured_channel:CZ"
    affected = actual.channel_names.index("P3")
    assert not actual.trial_channel_mask[:, affected].any()
    assert np.all(actual.trial_channel_mask.sum(axis=1) == 5)
    np.testing.assert_array_equal(actual.X[:, actual.trial_channel_mask[0]], expected.X[:, actual.trial_channel_mask[0]])
    np.testing.assert_array_equal(actual.X[:, affected], 0)
    reloaded = load_epoch_dataset(save_epoch_dataset(tmp_path / "local.npz", actual))
    np.testing.assert_array_equal(actual.trial_channel_mask, reloaded.trial_channel_mask)
    np.testing.assert_array_equal(actual.X, reloaded.X)


def test_frozen_eog_reference_dependencies_propagate_by_the_actual_linear_map():
    from data.ocular import FittedEOGRegression

    ocular = FittedEOGRegression(("Pz",), ("EOG",), ((.25,),), (0.,),
        "test:p1", "measured_channel:CZ", {"role": "synthetic", "acquisition_source_reference": "A2"},
        reference_channel="Cz", eog_derivation={"EOG": {
            "kind": "single_ended", "placement": "left_eye_corner", "reference": "A2"}})
    matrix = signal_dependency_matrix(("Cz", "Pz", "EOG"), ("Pz",), ocular_model=ocular)
    np.testing.assert_array_equal(matrix, [[-.75, 1., -.25]])


def test_saved_uppercase_native_channels_keep_raw_annotation_identity(tmp_path, monkeypatch):
    root = write_bids_session(tmp_path / "raw", session_id="native-case", target=4,
                              started_utc="2026-09-06T00:00:00Z")
    session = validate_brainsync_bids_session(root)
    source = load_brainsync_session(root, preprocessing=CAUSAL)
    participant = materialize_dataset_identity(source).record_for("P001").authority_key("source_or_global")
    raw = mne.io.read_raw_edf(session.bids.raw_path, preload=True, verbose=False)
    raw.set_channel_types(dict(zip(raw.ch_names, [kind.lower() for kind in session.bids.channel_types], strict=True)))
    model = fit_continuous_qc(raw,
        spec=ContinuousQCSpec(12, 1e-7, 10, channel_local=True, reference_channel=None,
                              minimum_channel_fraction=.5, include_eog=True),
        fit_sample_mask=np.ones(raw.n_times, dtype=bool), fit_provenance={
            "participant_key": participant, "source_reference": session.bids.source_reference,
            "calibration_session_ids": [session.session_id], "calibration_end_timestamp_s": session.ended_timestamp_s})
    raw._data[7, 100:] += .02
    monkeypatch.setattr("data.brainsync.read_raw", lambda *args, **kwargs: raw.copy())
    native = load_brainsync_session(root, preprocessing=CAUSAL, continuous_qc_model=model)
    restored = load_brainsync_session(root, preprocessing=CAUSAL, continuous_qc_model=model, channels=source.channel_names)
    assert native.n_channels == restored.n_channels == 7
    np.testing.assert_array_equal(restored.X, native.X)
    np.testing.assert_array_equal(restored.trial_channel_mask, native.trial_channel_mask)
