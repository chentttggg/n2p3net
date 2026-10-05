"""Independent recordings relax chronology without relaxing fit ownership."""

from __future__ import annotations

import copy
from dataclasses import replace

import mne
import numpy as np
import pytest

from data import brainsync_contract
from data.brainsync import (
    IndependentRecordingScope,
    load_brainsync_inference_session,
    load_brainsync_session,
    load_brainsync_sessions,
    load_brainsync_sessions_resilient,
)
from data.brainsync_contract import validate_brainsync_bids_session
from data.continuous_qc import (
    ContinuousQCSpec,
    fit_continuous_qc,
    load_continuous_qc,
    save_continuous_qc,
)
from data.contract import BRAIN_SYNC_CAUSAL_DATA_CONTRACT
from data.epochs import materialize_dataset_identity, preprocessing_spec_from_contract
from data.ocular import fit_eog_regression, load_eog_regression, save_eog_regression
from research.contracts import semantic_sha256
from tests.support.brainsync import write_bids_session

CAUSAL = preprocessing_spec_from_contract(BRAIN_SYNC_CAUSAL_DATA_CONTRACT)


@pytest.fixture(scope="module")
def recording_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("independent_raw_scope")
    sessions = {f"R{digit}": write_bids_session(
        root / f"R{digit}", session_id=f"R{digit}", target=digit,
        started_utc=f"2026-09-0{digit}T00:00:00Z",
    ) for digit in (2, 3, 4, 5, 6)}
    sessions["other_person"] = write_bids_session(
        root / "other_person", session_id="R3", target=3, subject_id="P002",
        started_utc="2026-09-03T00:00:00Z",
    )
    validated = {name: validate_brainsync_bids_session(path) for name, path in sessions.items()}
    before = load_brainsync_session(sessions["R2"], preprocessing=CAUSAL)
    participant = materialize_dataset_identity(before).record_for("P001").authority_key("source_or_global")
    calibration = []
    for name in ("R2", "R4"):
        session = validated[name]
        raw = mne.io.read_raw_edf(session.bids.raw_path, preload=True, verbose=False)
        raw.set_channel_types(dict(zip(raw.ch_names,
            [kind.lower() for kind in session.bids.channel_types], strict=True)))
        calibration.append(raw)
    raw = mne.concatenate_raws(calibration, verbose=False)
    provenance = {
        "participant_key": participant, "source_reference": validated["R2"].bids.source_reference,
        "calibration_session_ids": ["R2", "R4"],
        "calibration_end_timestamp_s": validated["R4"].ended_timestamp_s,
        "purpose": "synthetic independent recording holdout contract",
    }
    ocular = fit_eog_regression(
        raw, participant_key=participant, source_reference=provenance["source_reference"],
        fit_provenance=provenance, fit_sample_mask=np.ones(raw.n_times, dtype=bool),
        eog_derivation=validated["R2"].bids.eog_derivation,
    )
    ocular = load_eog_regression(save_eog_regression(root / "ocular.json", ocular))
    continuous = fit_continuous_qc(
        raw, spec=ContinuousQCSpec(12, 1., 25, channel_local=True, include_eog=True),
        fit_sample_mask=np.ones(raw.n_times, dtype=bool), fit_provenance=provenance,
    )
    continuous = load_continuous_qc(save_continuous_qc(root / "continuous.json", continuous))
    return {"sessions": sessions, "validated": validated, "ocular_model": ocular,
            "continuous_qc_model": continuous,
            "scope": IndependentRecordingScope(("R2", "R4"), ("R3", "R5"))}


@pytest.mark.parametrize("model_name", ["ocular_model", "continuous_qc_model"])
@pytest.mark.parametrize("loader", [load_brainsync_session, load_brainsync_inference_session])
def test_default_prospective_loader_rejects_future_fit(recording_fixture, model_name, loader):
    fixture = recording_fixture
    with pytest.raises(ValueError, match="future data"):
        loader(fixture["sessions"]["R3"], preprocessing=CAUSAL, **{model_name: fixture[model_name]})


@pytest.mark.parametrize("model_name", ["ocular_model", "continuous_qc_model"])
def test_independent_scope_allows_earlier_test_and_preserves_frozen_model(recording_fixture, model_name):
    fixture = recording_fixture
    model = fixture[model_name]
    original_record = copy.deepcopy(model.record())
    scope_record = fixture["scope"].record()
    args = {model_name: model, "recording_scope": fixture["scope"]}
    earlier = load_brainsync_session(fixture["sessions"]["R3"], preprocessing=CAUSAL, **args)
    later = load_brainsync_session(fixture["sessions"]["R5"], preprocessing=CAUSAL, **args)
    prospective_later = load_brainsync_session(fixture["sessions"]["R5"], preprocessing=CAUSAL, **{model_name: model})
    assert earlier.n_epochs == later.n_epochs == 9
    assert earlier.provenance["raw_calibration_scope"] == scope_record
    assert later.provenance["raw_calibration_scope"] == scope_record
    np.testing.assert_array_equal(later.X, prospective_later.X)
    assert "raw_calibration_scope" not in prospective_later.provenance
    assert model.record() == original_record
    assert semantic_sha256(model.record()) == semantic_sha256(original_record)
    assert fixture["scope"].record() == scope_record
    assert model.fit_provenance["calibration_session_ids"] == ["R2", "R4"]


@pytest.mark.parametrize("model_name", ["ocular_model", "continuous_qc_model"])
@pytest.mark.parametrize("fit_sessions", [("R2", "R3"), ("R2", "R6"), ("R3",)])
def test_test_recording_or_undeclared_fit_cannot_enter_calibration(recording_fixture, model_name, fit_sessions):
    fixture = recording_fixture
    model = fixture[model_name]
    contaminated = replace(model, fit_provenance={**model.fit_provenance,
                                                "calibration_session_ids": list(fit_sessions)})
    with pytest.raises(ValueError, match="fit exceeds.*calibration scope"):
        load_brainsync_session(fixture["sessions"]["R3"], preprocessing=CAUSAL, recording_scope=fixture["scope"],
                               **{model_name: contaminated})


@pytest.mark.parametrize("model_name", ["ocular_model", "continuous_qc_model"])
def test_independent_scope_keeps_participant_ownership(recording_fixture, model_name):
    fixture = recording_fixture
    with pytest.raises(ValueError, match="different participant_key"):
        load_brainsync_session(fixture["sessions"]["other_person"], preprocessing=CAUSAL, recording_scope=fixture["scope"],
                               **{model_name: fixture[model_name]})


@pytest.mark.parametrize("loader", [load_brainsync_session, load_brainsync_inference_session])
def test_scope_rejects_undeclared_recording_even_without_fitted_preprocessing(recording_fixture, loader):
    fixture = recording_fixture
    with pytest.raises(ValueError, match="outside the declared independent scope"):
        loader(fixture["sessions"]["R6"], preprocessing=CAUSAL, recording_scope=fixture["scope"])


@pytest.mark.parametrize(
    "model_name,change_reference",
    [
        ("ocular_model", lambda model: replace(model, source_reference="incorrect_reference")),
        (
            "continuous_qc_model",
            lambda model: replace(
                model,
                fit_provenance={**model.fit_provenance, "source_reference": "incorrect_reference"},
            ),
        ),
    ],
)
def test_independent_scope_does_not_waive_reference_identity(
    recording_fixture, model_name, change_reference
):
    fixture = recording_fixture
    model = fixture[model_name]
    changed = change_reference(model)
    with pytest.raises(ValueError, match="reference"):
        load_brainsync_session(
            fixture["sessions"]["R3"], preprocessing=CAUSAL, recording_scope=fixture["scope"], **{model_name: changed}
        )


@pytest.mark.parametrize("model_name", ["ocular_model", "continuous_qc_model"])
def test_independent_scope_still_requires_finite_original_fit_end(recording_fixture, model_name):
    fixture = recording_fixture
    model = fixture[model_name]
    changed = replace(model, fit_provenance={**model.fit_provenance,
                                           "calibration_end_timestamp_s": float("nan")})
    with pytest.raises(ValueError, match="end time|end timestamp"):
        load_brainsync_session(fixture["sessions"]["R3"], preprocessing=CAUSAL, recording_scope=fixture["scope"],
                               **{model_name: changed})


@pytest.mark.parametrize("calibration,evaluation", [
    (("R2", "R4"), ("R3", "R4")), (("R2", "R2"), ("R3",)),
    ((), ("R3",)), (("R2",), ()), ((" ",), ("R3",)),
])
def test_independent_scope_requires_nonempty_unique_disjoint_session_sets(calibration, evaluation):
    with pytest.raises(ValueError, match="unique named sessions|must be disjoint"):
        IndependentRecordingScope(calibration, evaluation)


def test_scope_copies_mutable_input_session_lists():
    calibration, evaluation = ["R2", "R4"], ["R3", "R5"]
    scope = IndependentRecordingScope(calibration, evaluation)
    calibration.append("R3")
    evaluation.append("R6")
    assert scope.calibration_session_ids == ("R2", "R4")
    assert scope.evaluation_session_ids == ("R3", "R5")


def test_batch_and_resilient_loaders_propagate_independent_scope(recording_fixture):
    fixture = recording_fixture
    args = {"ocular_model": fixture["ocular_model"], "continuous_qc_model": fixture["continuous_qc_model"],
            "recording_scope": fixture["scope"]}
    paths = [fixture["sessions"][name] for name in ("R5", "R3")]
    batch = load_brainsync_sessions(paths, preprocessing=CAUSAL, **args)
    resilient = load_brainsync_sessions_resilient(paths, preprocessing=CAUSAL, invalid_session_policy="skip", **args)
    np.testing.assert_array_equal(batch.X, resilient.dataset.X)
    assert resilient.failures == ()
    assert batch.provenance["raw_calibration_scope"] == fixture["scope"].record()
    assert batch.provenance["session_ids"] == ["R3", "R5"]


def test_unlabelled_scope_never_accesses_target_fields_and_preserves_signal(recording_fixture, monkeypatch):
    fixture = recording_fixture
    args = {"ocular_model": fixture["ocular_model"], "continuous_qc_model": fixture["continuous_qc_model"],
            "recording_scope": fixture["scope"]}
    labelled = load_brainsync_session(fixture["sessions"]["R3"], preprocessing=CAUSAL, **args)
    original_loader = brainsync_contract._load_json_object

    class TruthGuard(dict):
        def __init__(self, values, blocked):
            super().__init__(values)
            self.blocked = blocked

        def __getitem__(self, key):
            if key in self.blocked:
                raise AssertionError(f"target-free loader accessed {key}")
            return super().__getitem__(key)

        def get(self, key, default=None):
            if key in self.blocked:
                raise AssertionError(f"target-free loader accessed {key}")
            return super().get(key, default)

    def guarded_loader(path):
        manifest = original_loader(path)
        if path.name == "session.json":
            manifest["experiment"] = TruthGuard(manifest["experiment"],
                {"thought_digit", "target_label_status"})
            return TruthGuard(manifest, {"target_label"})
        return manifest

    monkeypatch.setattr(brainsync_contract, "_load_json_object", guarded_loader)
    unlabelled = load_brainsync_inference_session(fixture["sessions"]["R3"], preprocessing=CAUSAL, **args)
    assert unlabelled.y is None
    np.testing.assert_array_equal(unlabelled.X, labelled.X)
    assert not unlabelled.event_timeline.has_candidate_sets
    assert "target_candidate_id" not in unlabelled.metadata
    assert "target_digits" not in unlabelled.provenance
    assert unlabelled.provenance["raw_calibration_scope"] == fixture["scope"].record()
