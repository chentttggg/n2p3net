"""A frozen calibration QC model can admit supplemental training evidence."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from data.artifact import FoldLocalArtifactPolicy
from research.contracts import semantic_sha256
from tests.test_calibration_quality_ingress import _quality_pair
from transfer.calibration_candidates import (
    CalibrationCandidateFitCache,
    fit_calibration_candidate,
    select_calibration_candidate,
    waveform_lda_recipe,
)
from transfer.quality import fit_epoch_quality


def _frozen():
    dataset, _, policy = _quality_pair()
    frozen = fit_epoch_quality(dataset, np.arange(dataset.n_epochs) < 20, policy)
    return dataset, frozen, policy


def test_explicit_model_never_refits_on_supplemental_values_or_labels(monkeypatch):
    import transfer.quality as quality

    dataset, frozen, _ = _frozen()
    record = copy.deepcopy(frozen.record())
    training = np.arange(dataset.n_epochs) < 40
    values, labels = dataset.X.copy(), dataset.y.copy()
    values[20:40] *= 1.01
    labels[20:40] = 1 - labels[20:40]

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit model mode must never fit QC thresholds")

    monkeypatch.setattr(quality, "fit_epoch_quality", forbidden)
    first = fit_calibration_candidate(dataset, waveform_lda_recipe(), training, epoch_quality_model=frozen)
    second = fit_calibration_candidate(replace(dataset, X=values, y=labels), waveform_lda_recipe(), training,
                                       epoch_quality_model=frozen)
    assert first.epoch_quality.record() == second.epoch_quality.record() == frozen.record() == record
    assert first.epoch_quality is not frozen
    assert first.fit_history["epoch_quality"]["lifecycle"] == "frozen_explicit"
    assert first.fit_history["training_data_sha256"] != second.fit_history["training_data_sha256"]
    assert any(not np.array_equal(value.numpy(), second.state[key].numpy()) for key, value in first.state.items())


@pytest.mark.parametrize("entry", ["fit", "cache", "selector"])
def test_explicit_model_is_mutually_exclusive_with_policy(entry):
    dataset, frozen, policy = _frozen()
    training = np.arange(dataset.n_epochs) < 40
    kwargs = {"epoch_quality_model": frozen, "epoch_quality_policy": policy}
    with pytest.raises(ValueError, match="mutually exclusive"):
        if entry == "fit":
            fit_calibration_candidate(dataset, waveform_lda_recipe(), training, **kwargs)
        elif entry == "cache":
            CalibrationCandidateFitCache().fit(dataset, waveform_lda_recipe(), training, phase="inner", **kwargs)
        else:
            select_calibration_candidate(dataset, (waveform_lda_recipe(),), calibration_mask=training, **kwargs)


@pytest.mark.parametrize("field,value", [
    ("participant_key", "other"), ("source_reference", "other-reference"),
    ("channel_names", ["Pz", "Cz", "Fz"]), ("preprocessing", {}),
    ("session_ids", ["future-session"]), ("training_event_ids_digest", "f" * 64),
])
def test_explicit_quality_contract_or_fit_scope_mismatch_is_rejected(field, value):
    dataset, frozen, _ = _frozen()
    scope = {**frozen.fit_provenance, field: value}
    with pytest.raises(ValueError, match="epoch quality|epoch QC"):
        fit_calibration_candidate(dataset, waveform_lda_recipe(), np.arange(dataset.n_epochs) < 40,
                                  epoch_quality_model=replace(frozen, fit_provenance=scope))


@pytest.mark.parametrize("mode", ["same_session_later_event", "future_session", "unknown_event", "duplicate_event"])
def test_explicit_quality_cannot_use_events_outside_authorized_training(mode):
    dataset, frozen, policy = _frozen()
    training = np.arange(dataset.n_epochs) < (10 if mode == "same_session_later_event" else 40)
    if mode == "future_session":
        frozen = fit_epoch_quality(dataset, np.arange(dataset.n_epochs) >= 40, policy)
    if mode in {"unknown_event", "duplicate_event"}:
        scope = copy.deepcopy(frozen.fit_provenance)
        scope["training_event_ids"][0] = "unknown-future-event" if mode == "unknown_event" else scope["training_event_ids"][1]
        scope["training_event_ids_digest"] = semantic_sha256(scope["training_event_ids"])
        frozen = replace(frozen, fit_provenance=scope)
    with pytest.raises(ValueError, match="epoch quality|epoch QC"):
        fit_calibration_candidate(dataset, waveform_lda_recipe(), training, epoch_quality_model=frozen)


def test_cache_binds_explicit_model_digest_and_selector_retains_exact_frozen_state():
    dataset, frozen, policy = _frozen()
    training = np.arange(dataset.n_epochs) < 40
    cache = CalibrationCandidateFitCache()
    first = cache.fit(dataset, waveform_lda_recipe(), training, phase="inner", epoch_quality_model=frozen)
    reused = cache.fit(dataset, waveform_lda_recipe(), training, phase="inner", epoch_quality_model=frozen)
    alternate = fit_epoch_quality(dataset, np.arange(dataset.n_epochs) < 20,
                                  replace(policy, global_scale_mad_z=5.))
    changed = cache.fit(dataset, waveform_lda_recipe(), training, phase="inner", epoch_quality_model=alternate)
    assert first.fit_history["fit_reuse"]["fit_key_sha256"] == reused.fit_history["fit_reuse"]["fit_key_sha256"]
    assert first.fit_history["fit_reuse"]["fit_key_sha256"] != changed.fit_history["fit_reuse"]["fit_key_sha256"]
    selected, result = select_calibration_candidate(
        dataset, (waveform_lda_recipe(),), calibration_mask=training, selector_mode="no_transfer",
        epoch_quality_model=frozen, recording_order=("session-a", "session-b", "session-c"),
    )
    assert selected.epoch_quality.record() == frozen.record()
    assert result["refit_history"]["epoch_quality"]["model_digest"] == semantic_sha256(frozen.record())
    assert all(row["fit_history"]["epoch_quality"]["model_digest"] == semantic_sha256(frozen.record())
               for row in result["recipes"] if row["status"] == "completed")


def test_selector_rejects_model_fitted_on_its_inner_validation_before_fallback():
    dataset, _, policy = _frozen()
    frozen = fit_epoch_quality(dataset, (np.arange(dataset.n_epochs) >= 20) & (np.arange(dataset.n_epochs) < 40), policy)
    cache = CalibrationCandidateFitCache()
    with pytest.raises(ValueError, match="epoch quality|epoch QC"):
        select_calibration_candidate(dataset, (waveform_lda_recipe(),),
            calibration_mask=np.arange(dataset.n_epochs) < 40, selector_mode="no_transfer",
            epoch_quality_model=frozen, fit_cache=cache,
            recording_order=("session-a", "session-b", "session-c"))
    assert cache.record()["fit_calls"] == 0


def test_epoch_quality_json_preserves_record_digest_and_is_immutable(tmp_path):
    from transfer.quality import load_epoch_quality, save_epoch_quality

    dataset, frozen, policy = _frozen()
    before = semantic_sha256(frozen.record())
    assert before == semantic_sha256(fit_epoch_quality(dataset, np.arange(dataset.n_epochs) < 20, policy).record())
    path = save_epoch_quality(tmp_path / "epoch_quality.json", frozen)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == json.loads(json.dumps({"model": frozen.record(), "digest": before}))
    assert semantic_sha256(load_epoch_quality(path).record()) == before
    with pytest.raises(FileExistsError):
        save_epoch_quality(path, frozen)
    payload["model"]["selected_bad_channel_fraction"] = .01
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        load_epoch_quality(path)


@pytest.mark.parametrize("changed", ["voltage", "mask", "qc_features"])
def test_frozen_fit_rows_cannot_be_reprocessed_under_identical_event_ids(changed):
    from data.qc_features import compute_epoch_qc_features

    dataset, frozen, _ = _frozen()
    values = dataset.X.copy()
    observed = np.ones(dataset.X.shape[:2], dtype=bool)
    features = compute_epoch_qc_features(values, channel_mask=dataset.channel_mask)
    if changed == "voltage":
        values[2, 0, 3] += 1e-6
    elif changed == "mask":
        observed[2, 0] = False
        values[~observed] = 0.
        features = compute_epoch_qc_features(values, channel_mask=dataset.channel_mask, trial_channel_mask=observed)
    else:
        features.relative_ptp[2, 0] += .01
    changed_dataset = replace(dataset, X=values, trial_channel_mask=observed, qc_features=features)
    with pytest.raises(ValueError, match="fit data digest"):
        fit_calibration_candidate(changed_dataset, waveform_lda_recipe(), np.arange(dataset.n_epochs) < 40,
                                  epoch_quality_model=frozen)


def test_frozen_quality_fit_digest_excludes_classifier_labels():
    dataset, frozen, _ = _frozen()
    labels = dataset.y.copy()
    labels[:20] = 1 - labels[:20]
    fitted = fit_calibration_candidate(replace(dataset, y=labels), waveform_lda_recipe(),
                                       np.arange(dataset.n_epochs) < 40, epoch_quality_model=frozen)
    assert fitted.epoch_quality.record() == frozen.record()


def test_calibrated_explicit_model_survives_preflight_inner_refit_and_reload(tmp_path, monkeypatch):
    import transfer.quality as quality
    from tests.test_calibration_quality_ingress import _masked_bids_calibration
    from transfer.calibrated import (
        CalibrationRecipe,
        fit_calibrated_predictor,
        load_calibrated_predictor,
        preflight_calibrated_predictor,
        save_calibrated_predictor,
    )

    dataset = _masked_bids_calibration(tmp_path)
    frozen = fit_epoch_quality(dataset, dataset.metadata["session"].to_numpy() == "cal-0",
                               FoldLocalArtifactPolicy(min_clean_epochs=2))
    digest = semantic_sha256(frozen.record())
    plan = preflight_calibrated_predictor(dataset, selector_mode="no_transfer", epoch_quality_model=frozen)
    assert plan["execution_ready"] is True

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit QC must not refit during selection or full refit")

    monkeypatch.setattr(quality, "fit_epoch_quality", forbidden)
    predictor = fit_calibrated_predictor(dataset, selector_mode="no_transfer",
        recipes=(CalibrationRecipe("scratch_small", 1, 1e-3, initialization="scratch"),),
        epoch_quality_model=frozen)
    assert predictor.metadata["epoch_quality"]["model_digest"] == digest
    assert predictor.metadata["epoch_quality"]["lifecycle"] == "frozen_explicit"
    restored = load_calibrated_predictor(save_calibrated_predictor(tmp_path / "frozen.pt", predictor))
    assert semantic_sha256(restored.epoch_quality.record()) == digest
    np.testing.assert_array_equal(restored.predict_logit(dataset), predictor.predict_logit(dataset))


def test_preflight_explicit_model_rejects_inner_validation_fit_scope(tmp_path):
    from tests.test_calibration_quality_ingress import _masked_bids_calibration
    from transfer.calibrated import preflight_calibrated_predictor

    dataset = _masked_bids_calibration(tmp_path)
    frozen = fit_epoch_quality(dataset, dataset.metadata["session"].to_numpy() == "cal-2",
                               FoldLocalArtifactPolicy(min_clean_epochs=2))
    plan = preflight_calibrated_predictor(dataset, selector_mode="no_transfer", epoch_quality_model=frozen)
    assert plan["execution_ready"] is False
    assert any("epoch quality fit events" in error for error in plan["execution_blockers"])
