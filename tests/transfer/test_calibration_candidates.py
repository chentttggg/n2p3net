"""Calibration candidate scopes, quality, joint fitting and exact fit reuse."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from data.artifact import FoldLocalArtifactPolicy
from data.brainsync import load_brainsync_sessions
from data.epochs import select_scheduled_events
from models.n2p3net import N2P3Net
from tests.support.brainsync import write_bids_session
from tests.support.candidate import candidate_dataset
from tests.support.datasets import binary_dataset
from transfer.calibration_candidates import calibration_decision_groups, fit_calibration_candidate
from transfer.recipes import unified_target_recipes


def test_structured_erp_candidate_keeps_fit_scope_and_numeric_state():
    from baselines.structured_erp import StructuredERP
    from transfer.recipes import erp_target_recipe

    data = binary_dataset()
    mask = np.arange(data.n_epochs) < 20
    recipe = erp_target_recipe("xdawn_erp", bin_width=2)
    fitted = fit_calibration_candidate(data, recipe, mask)
    changed = copy.deepcopy(data)
    changed.X[~mask] *= 37
    changed.y[~mask] = 1 - changed.y[~mask]
    repeated = fit_calibration_candidate(changed, recipe, mask)
    np.testing.assert_array_equal(fitted.predict_logit(data.X), repeated.predict_logit(data.X))
    record = fitted.fit_history["erp_artifact"]
    restored = StructuredERP.from_state(record, {k.removeprefix("erp."): v.numpy() for k, v in fitted.state.items() if k.startswith("erp.")})
    np.testing.assert_array_equal(fitted.predict_logit(data.X), restored.predict_logit(data.X[:, fitted.waveform_channels]))


def test_temporal_fit_cache_binds_basis_and_declared_source_role():
    from baselines.temporal_transfer import TemporalBasis
    from data.identity import DatasetIdentityTable
    from transfer.calibration_candidates import CalibrationCandidateFitCache
    from transfer.recipes import erp_target_recipe

    source = binary_dataset()
    target = copy.deepcopy(source)
    target.identity_table = DatasetIdentityTable.from_source_rows(["p1"], ["independent_target"])
    mask = np.arange(target.n_epochs) < 20
    first = TemporalBasis.fit_source([(source, np.arange(20))], rank=2)
    changed = copy.deepcopy(source)
    changed.X[0, 0] += np.linspace(0, 1e-5, changed.n_times)
    second = TemporalBasis.fit_source([(changed, np.arange(20))], rank=2)
    recipe = erp_target_recipe("temporal_lda", temporal_basis_kind="source_supervised", temporal_rank=2)
    cache = CalibrationCandidateFitCache()
    a = cache.fit(target, recipe, mask, source_temporal_basis=first, phase="full_prefix")
    repeated = cache.fit(target, recipe, mask, source_temporal_basis=first, phase="full_prefix")
    b = cache.fit(target, recipe, mask, source_temporal_basis=second, phase="full_prefix")
    assert cache.record()["fit_calls"] == 2 and cache.record()["cache_hits"] == 1
    np.testing.assert_array_equal(a.predict_logit(target.X), repeated.predict_logit(target.X))
    assert not np.allclose(a.predict_logit(target.X), b.predict_logit(target.X))
    wrong_role = erp_target_recipe("temporal_lda", temporal_basis_kind="fixed_cosine", temporal_rank=2)
    with pytest.raises(ValueError, match="cannot consume"):
        fit_calibration_candidate(target, wrong_role, mask, source_temporal_basis=first)


def test_erp_prediction_requires_selected_physical_channels():
    from transfer.recipes import erp_target_recipe

    data = binary_dataset()
    data.channel_mask[2] = False
    data.X[:, 2] = 0
    fitted = fit_calibration_candidate(data, erp_target_recipe("pooled_lda", bin_width=4),
                                        np.arange(data.n_epochs) < 20)
    assert fitted.waveform_channels.tolist() == [0, 1]
    mask = np.broadcast_to(data.channel_mask, data.X.shape[:2]).copy()
    mask[0, 1] = False
    with pytest.raises(ValueError, match="every selected channel"):
        fitted.predict_logit(data.X, trial_channel_mask=mask)
    with pytest.raises(ValueError, match="boolean"):
        fitted.predict_logit(data.X, trial_channel_mask=np.full(data.X.shape[:2], np.nan))


def test_four_selectors_share_exact_fits_without_changing_outputs(monkeypatch):
    import transfer.calibration_candidates as candidates

    dataset = binary_dataset()
    source = N2P3Net(n_channels=3, n_times=dataset.n_times,
        sfreq=dataset.preprocessing.sfreq, tmin_s=dataset.preprocessing.tmin_ms / 1000,
        temporal_filters=2, mst_features_per_scale=2)
    allowed = np.arange(dataset.n_epochs) < 20
    source_stats = (np.zeros((1, 3, 1)), np.ones((1, 3, 1)))
    real_fit = candidates.fit_calibration_candidate
    calls = []

    def tracked_fit(*args, **kwargs):
        calls.append(1)
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(candidates, "fit_calibration_candidate", tracked_fit)
    recipes = {mode: tuple(replace(recipe, epochs=1) for recipe in
        candidates.calibration_candidate_recipes(mode, batch_size=512))
        for mode in ("legacy_product", "transfer_only", "no_transfer", "five_candidates")}

    def execute(cache):
        return [candidates.select_calibration_candidate(dataset, recipes[mode],
            calibration_mask=allowed, source_trunk=source, source_input_stats=source_stats,
            selector_mode=mode, fit_cache=cache) for mode in recipes]

    uncached = execute(None)
    assert len(calls) == 16
    calls.clear()
    cache = candidates.CalibrationCandidateFitCache()
    cached = execute(cache)
    assert len(calls) == 10
    assert cache.record()["fit_calls"] == 10
    assert cache.record()["cache_hits"] == 6
    for (first, first_record), (second, second_record) in zip(uncached, cached, strict=True):
        assert first.recipe == second.recipe
        assert first_record["refit_state_sha256"] == second_record["refit_state_sha256"]
        np.testing.assert_array_equal(first.predict_logit(dataset.X), second.predict_logit(dataset.X))
        assert [item["primary_score"] for item in first_record["recipes"]] == [
            item["primary_score"] for item in second_record["recipes"]]
    cache.clear()
    assert cache.record()["entries"] == 0


def test_fit_reuse_isolates_state_phase_recipe_training_source_and_statistics():
    from transfer.calibration_candidates import CalibrationCandidateFitCache
    from transfer.recipes import TargetRecipe

    dataset = binary_dataset()
    source = N2P3Net(n_channels=3, n_times=dataset.n_times,
        sfreq=dataset.preprocessing.sfreq, tmin_s=dataset.preprocessing.tmin_ms / 1000,
        temporal_filters=2, mst_features_per_scale=2)
    recipe = TargetRecipe("source", epochs=1, batch_size=512, input_statistics="source")
    training = np.arange(dataset.n_epochs) < 20
    stats = (np.zeros((1, 3, 1)), np.ones((1, 3, 1)))
    cache = CalibrationCandidateFitCache()

    def fit(data=dataset, treatment=recipe, trunk=source, statistics=stats, phase="inner"):
        return cache.fit(data, treatment, training, source_trunk=trunk,
                         source_input_stats=statistics, phase=phase)

    first = fit()
    state = copy.deepcopy(first.state)
    with torch.no_grad():
        next(first.adapter.trunk.parameters()).add_(1)
    first.fit_history["mutated"] = True
    same = fit()
    assert all(torch.equal(value, same.state[name]) for name, value in state.items())
    assert "mutated" not in same.fit_history
    assert same.fit_history["fit_reuse"]["cache_hit"]
    final_changed = copy.deepcopy(dataset)
    final_changed.X[20:] *= 100
    final_changed.y[20:] = 1 - final_changed.y[20:]
    assert fit(data=final_changed).fit_history["fit_reuse"]["cache_hit"]
    training_changed = copy.deepcopy(dataset)
    training_changed.X[0] += 1e-3
    source_changed = copy.deepcopy(source)
    with torch.no_grad():
        next(source_changed.parameters()).add_(1)
    for fitted in (
        fit(data=training_changed), fit(treatment=replace(recipe, seed=9)),
        fit(treatment=replace(recipe, bn_buffers="source")), fit(trunk=source_changed),
        fit(statistics=(stats[0] + 1, stats[1])), fit(phase="full_prefix"),
    ):
        assert not fitted.fit_history["fit_reuse"]["cache_hit"]
    assert cache.record()["fit_calls"] == 7
    assert cache.record()["cache_hits"] == 2


def test_validation_is_rescored_and_only_full_refit_changes_for_fixed_budget_candidates():
    from transfer.calibration_candidates import (
        CalibrationCandidateFitCache,
        select_calibration_candidate,
        waveform_lda_recipe,
    )

    dataset = binary_dataset()
    training = np.arange(dataset.n_epochs) < 20
    cache = CalibrationCandidateFitCache()
    recipe = waveform_lda_recipe(batch_size=512)
    _, first = select_calibration_candidate(dataset, (recipe,), calibration_mask=training,
                                            fit_cache=cache, selector_mode="no_transfer")
    changed = copy.deepcopy(dataset)
    changed.y[16:20] = 1 - changed.y[16:20]
    _, second = select_calibration_candidate(changed, (recipe,), calibration_mask=training,
                                             fit_cache=cache, selector_mode="no_transfer")
    assert second["fit_execution"] == {"requested_fits": 2, "fit_calls": 1, "cache_hits": 1}
    assert first["recipes"][0]["fitted_state_sha256"] == second["recipes"][0]["fitted_state_sha256"]
    assert second["validation_auc"] == pytest.approx(1 - first["validation_auc"])
    final_changed = copy.deepcopy(changed)
    final_changed.X[20:] *= 100
    final_changed.y[20:] = 1 - final_changed.y[20:]
    _, third = select_calibration_candidate(final_changed, (recipe,), calibration_mask=training,
                                            fit_cache=cache, selector_mode="no_transfer")
    assert third["fit_execution"] == {"requested_fits": 2, "fit_calls": 0, "cache_hits": 2}


def test_epoch_selection_validation_changes_cache_key_and_failed_fits_remain_attempts():
    from transfer.calibration_candidates import CalibrationCandidateFitCache
    from transfer.recipes import TargetRecipe

    dataset = binary_dataset()
    recipe = TargetRecipe("scratch", initialization="scratch", epochs=1, batch_size=512)
    cache = CalibrationCandidateFitCache()
    training = np.arange(dataset.n_epochs) < 16
    validation = (np.arange(dataset.n_epochs) >= 16) & (np.arange(dataset.n_epochs) < 20)
    first = cache.fit(dataset, recipe, training, phase="inner", validation_mask=validation)
    changed = copy.deepcopy(dataset)
    changed.X[validation] *= 100
    second = cache.fit(changed, recipe, training, phase="inner", validation_mask=validation)
    assert not first.fit_history["fit_reuse"]["cache_hit"]
    assert not second.fit_history["fit_reuse"]["cache_hit"]
    for _ in range(2):
        with pytest.raises(ValueError, match="more than two rows"):
            cache.fit(dataset, recipe, np.arange(dataset.n_epochs) < 2, phase="inner")
    assert cache.record() == {"fit_calls": 4, "cache_hits": 0, "entries": 2}


@pytest.fixture
def direct_dataset(tmp_path):
    root = write_bids_session(tmp_path / "direct", session_id="cal", target=2,
        started_utc="2026-09-06T08:00:00Z", blocks=(tuple(range(1, 10)) * 5,) * 3)
    return load_brainsync_sessions((root,))


def test_direct_budget_does_not_complete_a_partial_r5_with_future_labels(direct_dataset):
    dataset = direct_dataset
    training = np.arange(dataset.n_epochs) < 50
    groups, targets, weights, ledger = calibration_decision_groups(dataset, training)
    assert len(groups) == 1 and targets == (1,)
    np.testing.assert_array_equal(groups[0].evidence_indices, np.arange(45))
    np.testing.assert_array_equal(weights, [1.])
    assert any(item["reason"] == "outside_training_budget" for item in ledger["excluded_groups"])
    # A target outside the authorized complete group cannot affect its fit scope.
    changed = dataset.event_timeline.target_candidate_ids.copy()
    changed[45:] = "unread_future_target"
    altered = replace(dataset, event_timeline=replace(dataset.event_timeline, target_candidate_ids=changed))
    other = calibration_decision_groups(altered, training)
    assert groups[0].record() == other[0][0].record() and targets == other[1]


def test_row_column_groups_retain_physical_selection_and_ignore_future_truth():
    dataset = candidate_dataset(n_selections=2, repetitions_per_selection=2)
    timeline = replace(dataset.event_timeline, complete=True,
        candidate_ids=dataset.metadata["candidate_id"].to_numpy(dtype=str),
        repetition_indices=dataset.metadata["repetition_index"].to_numpy(dtype=np.int64))
    dataset = replace(dataset, event_timeline=timeline)
    training = np.arange(dataset.n_epochs) < 30
    groups, targets, weights, _ = calibration_decision_groups(dataset, training)
    assert len(groups) == 1 and groups[0].protocol == "row_column"
    assert groups[0].repetitions == 2 and targets == (2,)
    np.testing.assert_array_equal(weights, [1.])
    metadata = dataset.metadata.copy()
    metadata.loc[24:, "target_col"] = 99
    other = calibration_decision_groups(replace(dataset, metadata=metadata), training)
    assert groups[0].record() == other[0][0].record() and targets == other[1]


def test_observed_only_or_binary_evidence_does_not_invent_decisions():
    data = binary_dataset()
    groups, targets, weights, ledger = calibration_decision_groups(data, np.ones(data.n_epochs, bool))
    assert groups == targets == () and len(weights) == 0
    assert ledger["reason"] == "membership_unavailable"
    data = candidate_dataset(n_selections=1)
    assert calibration_decision_groups(data, np.ones(data.n_epochs, bool))[3]["reason"] == "incomplete_physical_timeline"


def test_joint_candidate_fit_keeps_requested_risk_after_qc_for_d0_and_d1(direct_dataset, monkeypatch):
    import transfer.quality as quality

    dataset = direct_dataset

    def drop_first_block(data, fitted):
        return select_scheduled_events(data, np.arange(data.event_timeline.n_events) >= 45,
                                       reason="frozen_test_quality", preserve_schedule=True)

    monkeypatch.setattr(quality, "apply_epoch_quality", drop_first_block)
    architecture = N2P3Net(dataset.n_channels, n_times=dataset.n_times,
        sfreq=dataset.preprocessing.sfreq, temporal_filters=2, spatial_depth_multiplier=1,
        mst_features_per_scale=2, mst_kernel_sizes=(3,))
    declared = {recipe.name: recipe for recipe in unified_target_recipes(
        membership_available=True, source_prior_weight=0., max_updates=1, batch_size=45, seed=19)}
    recipes = (declared["unified_q0"], declared["unified_q1"])
    fits = [fit_calibration_candidate(dataset, recipe, np.ones(dataset.n_epochs, bool),
        source_trunk=architecture, epoch_quality_policy=FoldLocalArtifactPolicy(min_clean_epochs=2))
        for recipe in recipes]
    first, second = (fit.fit_history for fit in fits)
    assert first["initial_state_sha256"] == second["initial_state_sha256"]
    assert first["decision_scope"] == second["decision_scope"]
    assert first["decision_training"] == second["decision_training"]
    assert first["candidate_loss_active"] is False and second["candidate_loss_active"] is True
    assert first["decision_training"]["requested_groups"] == first["decision_training"]["authorized_groups"] == 3
    scope = first["decision_scope"]
    assert scope["requested_groups"] == 3 and scope["post_quality_complete_groups"] == 2
    np.testing.assert_allclose(scope["risk_weights"], np.repeat(1 / 3, 3))
    np.testing.assert_array_equal(scope["post_quality_groups"][0]["evidence_indices"], np.repeat(-1, 45))
    np.testing.assert_array_equal(scope["post_quality_groups"][1]["evidence_indices"], np.arange(45))
    for fit in fits:
        assert fit.fit_history["fit_budget"]["updates"] == 1
        assert len(fit.fit_history["candidate_losses"]) == 1
        assert np.isfinite(fit.adapter.predict_logit(dataset.X)).all()


def test_candidates_are_compared_on_their_common_validation_support():
    from transfer.calibration_candidates import _score_on_common_support

    labels = np.array([0, 0, 0, 1, 1, 1])
    # A scores every row but misranks rows 2 and 3; B abstains exactly there.
    logits = {"A": np.array([0.0, 0.1, 0.9, 0.2, 0.8, 0.9]),
              "B": np.array([0.0, 0.1, np.nan, np.nan, 0.8, 0.9])}
    completed = [{"candidate_id": "A", "primary_score": 7 / 9}, {"candidate_id": "B", "primary_score": 1.0}]
    assert _score_on_common_support(completed, logits, labels, "binary_auc")
    assert completed[0]["primary_score"] == completed[1]["primary_score"] == 1.0
    assert completed[0]["common_support_rows"] == 4
    one_class = [{"candidate_id": "A"}, {"candidate_id": "B"}]
    assert not _score_on_common_support(one_class, logits, np.array([0, 0, 0, 0, 1, 1]) * 0, "binary_auc")
