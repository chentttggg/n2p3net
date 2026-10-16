"""A frozen calibration QC model can admit supplemental training evidence."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from data.artifact import FoldLocalArtifactPolicy
from research.contracts import semantic_sha256
from tests.support.io import read_json, write_json
from tests.support.quality import masked_bids_calibration, quality_pair
from transfer.calibration_candidates import (
    CalibrationCandidateFitCache,
    fit_calibration_candidate,
    select_calibration_candidate,
    waveform_lda_recipe,
)
from transfer.quality import fit_epoch_quality


def _session_rows(dataset, *sessions):
    return dataset.metadata["session"].isin(sessions).to_numpy()


def _frozen():
    dataset, _, policy = quality_pair()
    frozen = fit_epoch_quality(dataset, _session_rows(dataset, "session-a"), policy)
    return dataset, frozen, policy


def test_explicit_model_never_refits_on_supplemental_values_or_labels(monkeypatch):
    import transfer.quality as quality

    dataset, frozen, _ = _frozen()
    record = copy.deepcopy(frozen.record())
    training = _session_rows(dataset, "session-a", "session-b")
    values, labels = dataset.X.copy(), dataset.y.copy()
    supplemental = _session_rows(dataset, "session-b")
    values[supplemental] *= 1.01
    labels[supplemental] = 1 - labels[supplemental]

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit model mode must never fit QC thresholds")

    monkeypatch.setattr(quality, "fit_epoch_quality", forbidden)
    first = fit_calibration_candidate(
        dataset, waveform_lda_recipe(), training, epoch_quality_model=frozen
    )
    second = fit_calibration_candidate(
        replace(dataset, X=values, y=labels),
        waveform_lda_recipe(),
        training,
        epoch_quality_model=frozen,
    )
    assert (
        first.epoch_quality.record() == second.epoch_quality.record() == frozen.record() == record
    )
    assert first.epoch_quality is not frozen
    assert first.fit_history["epoch_quality"]["lifecycle"] == "frozen_explicit"
    assert first.fit_history["training_data_sha256"] != second.fit_history["training_data_sha256"]
    assert any(
        not np.array_equal(value.numpy(), second.state[key].numpy())
        for key, value in first.state.items()
    )


@pytest.mark.parametrize(
    "fit",
    [
        pytest.param(fit_calibration_candidate, id="fit"),
        pytest.param(
            lambda data, recipe, rows, **options: CalibrationCandidateFitCache().fit(
                data, recipe, rows, phase="inner", **options
            ),
            id="cache",
        ),
        pytest.param(
            lambda data, recipe, rows, **options: select_calibration_candidate(
                data, (recipe,), calibration_mask=rows, **options
            ),
            id="selector",
        ),
    ],
)
def test_explicit_model_is_mutually_exclusive_with_policy(fit):
    dataset, frozen, policy = _frozen()
    training = _session_rows(dataset, "session-a", "session-b")
    with pytest.raises(ValueError, match="mutually exclusive"):
        fit(
            dataset,
            waveform_lda_recipe(),
            training,
            epoch_quality_model=frozen,
            epoch_quality_policy=policy,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("participant_key", "other"),
        ("source_reference", "other-reference"),
        ("channel_names", ["Pz", "Cz", "Fz"]),
        ("preprocessing", {}),
        ("session_ids", ["future-session"]),
        ("training_event_ids_digest", "f" * 64),
    ],
)
def test_explicit_quality_contract_or_fit_scope_mismatch_is_rejected(field, value):
    dataset, frozen, _ = _frozen()
    scope = {**frozen.fit_provenance, field: value}
    with pytest.raises(ValueError, match="epoch quality|epoch QC"):
        fit_calibration_candidate(
            dataset,
            waveform_lda_recipe(),
            _session_rows(dataset, "session-a", "session-b"),
            epoch_quality_model=replace(frozen, fit_provenance=scope),
        )


@pytest.mark.parametrize(
    "fit_sessions,training_sessions,fraction",
    [
        pytest.param(("session-a",), ("session-a",), 0.5, id="later-event-in-same-session"),
        pytest.param(("session-c",), ("session-a", "session-b"), 1.0, id="future-session"),
    ],
)
def test_explicit_quality_fit_must_stay_within_authorized_rows(
    fit_sessions, training_sessions, fraction
):
    dataset, _, policy = _frozen()
    frozen = fit_epoch_quality(dataset, _session_rows(dataset, *fit_sessions), policy)
    requested = np.flatnonzero(_session_rows(dataset, *training_sessions))
    training = np.zeros(dataset.n_epochs, dtype=bool)
    training[requested[: int(len(requested) * fraction)]] = True
    with pytest.raises(ValueError, match="epoch quality|epoch QC"):
        fit_calibration_candidate(
            dataset, waveform_lda_recipe(), training, epoch_quality_model=frozen
        )


@pytest.mark.parametrize(
    "replacement",
    [lambda events: "unknown-future-event", lambda events: events[1]],
    ids=["unknown-event", "duplicate-event"],
)
def test_explicit_quality_rejects_changed_event_identity_even_with_a_matching_digest(replacement):
    dataset, frozen, _ = _frozen()
    scope = copy.deepcopy(frozen.fit_provenance)
    scope["training_event_ids"][0] = replacement(scope["training_event_ids"])
    scope["training_event_ids_digest"] = semantic_sha256(scope["training_event_ids"])
    with pytest.raises(ValueError, match="epoch quality|epoch QC"):
        fit_calibration_candidate(
            dataset,
            waveform_lda_recipe(),
            _session_rows(dataset, "session-a", "session-b"),
            epoch_quality_model=replace(frozen, fit_provenance=scope),
        )


def test_cache_binds_explicit_model_digest_and_selector_retains_exact_frozen_state():
    dataset, frozen, policy = _frozen()
    training = _session_rows(dataset, "session-a", "session-b")
    cache = CalibrationCandidateFitCache()
    first = cache.fit(
        dataset, waveform_lda_recipe(), training, phase="inner", epoch_quality_model=frozen
    )
    reused = cache.fit(
        dataset, waveform_lda_recipe(), training, phase="inner", epoch_quality_model=frozen
    )
    alternate = fit_epoch_quality(
        dataset, _session_rows(dataset, "session-a"), replace(policy, global_scale_mad_z=5.0)
    )
    changed = cache.fit(
        dataset, waveform_lda_recipe(), training, phase="inner", epoch_quality_model=alternate
    )
    assert (
        first.fit_history["fit_reuse"]["fit_key_sha256"]
        == reused.fit_history["fit_reuse"]["fit_key_sha256"]
    )
    assert (
        first.fit_history["fit_reuse"]["fit_key_sha256"]
        != changed.fit_history["fit_reuse"]["fit_key_sha256"]
    )
    selected, result = select_calibration_candidate(
        dataset,
        (waveform_lda_recipe(),),
        calibration_mask=training,
        selector_mode="no_transfer",
        epoch_quality_model=frozen,
        recording_order=("session-a", "session-b", "session-c"),
    )
    assert selected.epoch_quality.record() == frozen.record()
    assert result["refit_history"]["epoch_quality"]["model_digest"] == semantic_sha256(
        frozen.record()
    )
    assert all(
        row["fit_history"]["epoch_quality"]["model_digest"] == semantic_sha256(frozen.record())
        for row in result["recipes"]
        if row["status"] == "completed"
    )


def test_selector_rejects_model_fitted_on_its_inner_validation_before_fallback():
    dataset, _, policy = _frozen()
    frozen = fit_epoch_quality(dataset, _session_rows(dataset, "session-b"), policy)
    cache = CalibrationCandidateFitCache()
    with pytest.raises(ValueError, match="epoch quality|epoch QC"):
        select_calibration_candidate(
            dataset,
            (waveform_lda_recipe(),),
            calibration_mask=_session_rows(dataset, "session-a", "session-b"),
            selector_mode="no_transfer",
            epoch_quality_model=frozen,
            fit_cache=cache,
            recording_order=("session-a", "session-b", "session-c"),
        )
    assert cache.record()["fit_calls"] == 0


def test_epoch_quality_json_preserves_record_digest_and_is_immutable(tmp_path):
    from transfer.quality import load_epoch_quality, save_epoch_quality

    dataset, frozen, policy = _frozen()
    before = semantic_sha256(frozen.record())
    assert before == semantic_sha256(
        fit_epoch_quality(dataset, _session_rows(dataset, "session-a"), policy).record()
    )
    path = save_epoch_quality(tmp_path / "epoch_quality.json", frozen)
    payload = read_json(path)
    assert payload == json.loads(json.dumps({"model": frozen.record(), "digest": before}))
    assert semantic_sha256(load_epoch_quality(path).record()) == before
    with pytest.raises(FileExistsError):
        save_epoch_quality(path, frozen)
    payload["model"]["selected_bad_channel_fraction"] = 0.01
    write_json(path, payload)
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
        values[~observed] = 0.0
        features = compute_epoch_qc_features(
            values, channel_mask=dataset.channel_mask, trial_channel_mask=observed
        )
    else:
        features.relative_ptp[2, 0] += 0.01
    changed_dataset = replace(dataset, X=values, trial_channel_mask=observed, qc_features=features)
    with pytest.raises(ValueError, match="fit data digest"):
        fit_calibration_candidate(
            changed_dataset,
            waveform_lda_recipe(),
            _session_rows(dataset, "session-a", "session-b"),
            epoch_quality_model=frozen,
        )


def test_frozen_quality_fit_digest_excludes_classifier_labels():
    dataset, frozen, _ = _frozen()
    labels = dataset.y.copy()
    quality_fit_rows = _session_rows(dataset, "session-a")
    labels[quality_fit_rows] = 1 - labels[quality_fit_rows]
    fitted = fit_calibration_candidate(
        replace(dataset, y=labels),
        waveform_lda_recipe(),
        _session_rows(dataset, "session-a", "session-b"),
        epoch_quality_model=frozen,
    )
    assert fitted.epoch_quality.record() == frozen.record()


def test_calibrated_explicit_model_survives_preflight_inner_refit_and_reload(tmp_path, monkeypatch):
    import transfer.quality as quality
    from transfer.calibrated import (
        CalibrationRecipe,
        fit_calibrated_predictor,
        load_calibrated_predictor,
        preflight_calibrated_predictor,
        save_calibrated_predictor,
    )

    dataset = masked_bids_calibration(tmp_path)
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
    from transfer.calibrated import preflight_calibrated_predictor

    dataset = masked_bids_calibration(tmp_path)
    frozen = fit_epoch_quality(dataset, dataset.metadata["session"].to_numpy() == "cal-2",
                               FoldLocalArtifactPolicy(min_clean_epochs=2))
    plan = preflight_calibrated_predictor(dataset, selector_mode="no_transfer", epoch_quality_model=frozen)
    assert plan["execution_ready"] is False
    assert any("epoch quality fit events" in error for error in plan["execution_blockers"])
