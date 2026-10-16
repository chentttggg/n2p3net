"""Epoch-QC selected caches cannot become an unscoped personalized fit input."""

from dataclasses import replace

import numpy as np
import pytest

from tests.support.brainsync import write_bids_session
from tests.support.quality import (
    declare_continuous_quality,
    masked_bids_calibration,
    personal_dataset,
    quality_pair,
)
from transfer.calibration_candidates import (
    CalibrationCandidateFitCache,
    fit_calibration_candidate,
    select_calibration_candidate,
    waveform_lda_recipe,
)
from transfer.quality import fit_epoch_quality


@pytest.mark.parametrize("refit_quality", [False, True])
def test_public_candidate_rejects_already_selected_epochs_with_or_without_threshold_refit(refit_quality):
    _, selected, policy = quality_pair()
    training = np.arange(selected.n_epochs) < 35
    with pytest.raises(ValueError, match="already.*epoch QC|epoch QC.*already"):
        fit_calibration_candidate(
            selected, waveform_lda_recipe(), training,
            epoch_quality_policy=policy if refit_quality else None,
        )


def test_selector_rejects_selected_cache_before_candidate_failure_fallback():
    _, selected, _ = quality_pair()
    cache = CalibrationCandidateFitCache()
    with pytest.raises(ValueError, match="already.*epoch QC|epoch QC.*already"):
        select_calibration_candidate(
            selected, (waveform_lda_recipe(),), calibration_mask=np.ones(selected.n_epochs, dtype=bool),
            selector_mode="no_transfer", fit_cache=cache,
        )
    assert cache.record() == {"fit_calls": 0, "cache_hits": 0, "entries": 0}


def test_fit_cache_rejects_applied_quality_marker_before_reusing_matching_tensor_key():
    dataset, selected, _ = quality_pair()
    cache = CalibrationCandidateFitCache()
    training = np.ones(dataset.n_epochs, dtype=bool)
    cache.fit(dataset, waveform_lda_recipe(), training, phase="full_prefix")
    marked = replace(dataset, provenance={
        **dataset.provenance,
        "epoch_artifact_processing": selected.provenance["epoch_artifact_processing"],
    })
    before = cache.record()
    with pytest.raises(ValueError, match="already.*epoch QC|epoch QC.*already"):
        cache.fit(marked, waveform_lda_recipe(), training, phase="full_prefix")
    assert cache.record() == before


def test_internal_quality_fit_retains_frozen_state_and_reuses_it_with_original_input():
    dataset, _, policy = quality_pair()
    training = np.arange(dataset.n_epochs) < 40
    expected = fit_epoch_quality(dataset, training, policy)
    cache = CalibrationCandidateFitCache()
    first = cache.fit(dataset, waveform_lda_recipe(), training, phase="inner", epoch_quality_policy=policy)
    reused = cache.fit(dataset, waveform_lda_recipe(), training, phase="inner", epoch_quality_policy=policy)

    assert first.epoch_quality.record() == expected.record()
    assert reused.epoch_quality.record() == expected.record()
    assert first.fit_history["epoch_quality"]["requested_training_rows"] == 40
    assert first.fit_history["epoch_quality"]["retained_training_rows"] < 40
    assert np.isnan(reused.predict_logit(dataset.X * 1e6)).all()
    assert cache.record() == {"fit_calls": 1, "cache_hits": 1, "entries": 1}


@pytest.mark.parametrize("status", [None, "not_applied"])
def test_absent_or_unapplied_epoch_quality_allows_continuous_recovery_inputs(status):

    dataset = declare_continuous_quality(personal_dataset())
    if status is not None:
        dataset = replace(dataset, provenance={
            **dataset.provenance, "epoch_artifact_processing": {"status": status},
        })
    fitted = fit_calibration_candidate(
        dataset, waveform_lda_recipe(), np.ones(dataset.n_epochs, dtype=bool),
    )
    assert fitted.channel_quality_availability is True
    assert fitted.epoch_quality is None


@pytest.mark.parametrize("preflight", [False, True])
def test_legacy_personalized_ingress_rejects_applied_epoch_quality_cache(tmp_path, preflight):
    from data.brainsync import load_brainsync_sessions
    from transfer.calibrated import fit_calibrated_predictor, preflight_calibrated_predictor

    root = write_bids_session(
        tmp_path / "calibration", session_id="calibration", target=4,
        started_utc="2026-09-06T00:00:00Z", blocks=(tuple(range(1, 10)) * 5,),
    )
    dataset = load_brainsync_sessions([root])
    # A no-rejection QC pass can preserve every array while still having a fitted scope.
    marked = replace(dataset, provenance={
        **dataset.provenance, "epoch_artifact_processing": {"status": "applied"},
    })
    with pytest.raises(ValueError, match="already.*epoch QC|epoch QC.*already"):
        (preflight_calibrated_predictor if preflight else fit_calibrated_predictor)(marked)


def test_preflight_reports_the_same_complete_union_lda_rows_and_channels_as_fitting(tmp_path):
    from transfer.calibrated import preflight_calibrated_predictor

    dataset = masked_bids_calibration(tmp_path)
    plan = preflight_calibrated_predictor(dataset, selector_mode="no_transfer")
    fitted = fit_calibration_candidate(dataset, waveform_lda_recipe(), np.ones(dataset.n_epochs, dtype=bool))
    full = plan["full_calibration_lda_gate"]
    assert full["legal"] is True
    assert full["requested_training_rows"] == dataset.n_epochs
    assert full["training_rows"] == fitted.fit_history["fit_scope"]["training_rows"]
    assert full["observed_channel_indices"] == fitted.waveform_channels.tolist()
    assert full["observed_channel_count"] == dataset.n_channels
    assert full["feature_count"] == fitted.state["coefficient"].numel()
    assert full["waveform_training_policy"] == "complete_observations_of_fixed_training_channel_union"
    inner = next(row for row in plan["candidate_fit_gates"] if row["candidate_id"] == "waveform_lda")
    assert inner["fit_eligible"] is True
    assert inner["requested_training_rows"] == len(plan["inner_training_rows"])
    assert inner["training_rows"] == len(plan["inner_training_rows"]) - dataset.n_channels
    assert inner["observed_channel_indices"] == fitted.waveform_channels.tolist()


def test_preflight_rejects_complete_lda_rows_that_lost_all_target_labels(tmp_path):
    from transfer.calibrated import preflight_calibrated_predictor

    dataset = masked_bids_calibration(tmp_path, drop_all_targets=True)
    plan = preflight_calibrated_predictor(dataset, selector_mode="no_transfer")
    full = plan["full_calibration_lda_gate"]
    assert full["legal"] is False
    assert full["training_rows"] == int((dataset.y == 0).sum())
    assert full["observed_channel_indices"] == list(range(dataset.n_channels))
    assert "both classes" in full["reason"]
    inner = next(row for row in plan["candidate_fit_gates"] if row["candidate_id"] == "waveform_lda")
    assert inner["fit_eligible"] is False
    with pytest.raises(ValueError, match="both classes"):
        fit_calibration_candidate(dataset, waveform_lda_recipe(), np.ones(dataset.n_epochs, dtype=bool))


def test_legacy_preflight_keeps_strict_no_quality_channel_gate(tmp_path):
    from transfer.calibrated import preflight_calibrated_predictor

    dataset = masked_bids_calibration(tmp_path, drop_all_targets=True, continuous=False)
    plan = preflight_calibrated_predictor(dataset)
    assert plan["full_calibration_lda_gate"]["legal"] is False
    assert plan["execution_ready"] is False
    assert "all selected EEG channels" in plan["full_calibration_lda_gate"]["reason"]


@pytest.mark.parametrize(
    "scope_change",
    [
        pytest.param({"calibration_session_ids": ["cal-2"]}, id="validation-session"),
        pytest.param({"calibration_session_ids": ["future-cal"]}, id="future-session"),
        pytest.param(
            {"participant_key": "n2p3-origin://BrainSync-GTN/other-participant"}, id="other-person"
        ),
    ],
)
def test_preflight_blocks_continuous_quality_fitted_outside_selected_inner_scope(
    tmp_path, scope_change
):
    import copy

    from research.contracts import semantic_sha256
    from transfer.calibrated import preflight_calibrated_predictor

    dataset = masked_bids_calibration(tmp_path)
    processing = copy.deepcopy(dataset.provenance["continuous_artifact_processing"])
    scope = processing["model"]["fit_provenance"]
    scope.update(scope_change)
    processing["model_digest"] = semantic_sha256(processing["model"])
    changed = replace(
        dataset, provenance={**dataset.provenance, "continuous_artifact_processing": processing}
    )
    plan = preflight_calibrated_predictor(changed, selector_mode="no_transfer")

    assert plan["execution_ready"] is False
    assert any(
        "continuous QC used sessions outside" in message for message in plan["execution_blockers"]
    )


def test_preflight_uses_full_continuous_quality_budget_for_fixed_lda_without_inner_selection(tmp_path):
    import copy

    from research.contracts import semantic_sha256
    from transfer.calibrated import preflight_calibrated_predictor

    dataset = masked_bids_calibration(tmp_path)
    processing = copy.deepcopy(dataset.provenance["continuous_artifact_processing"])
    processing["model"]["fit_provenance"]["calibration_session_ids"] = ["cal-2"]
    processing["model_digest"] = semantic_sha256(processing["model"])
    changed = replace(dataset, provenance={**dataset.provenance, "continuous_artifact_processing": processing})
    plan = preflight_calibrated_predictor(changed)

    assert plan["inner_selection_planned"] is False
    assert plan["execution_ready"] is True


def test_raw_fit_session_membership_does_not_certify_a_partial_recording():
    from transfer.calibration_candidates import validate_calibration_raw_fit_scope

    dataset = declare_continuous_quality(personal_dataset())
    complete = np.ones(dataset.n_epochs, dtype=bool)
    validate_calibration_raw_fit_scope(dataset, complete)
    partial = complete.copy()
    partial[-1] = False
    with pytest.raises(ValueError, match="partial-recording"):
        validate_calibration_raw_fit_scope(dataset, partial)
