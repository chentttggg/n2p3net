"""Exact fit reuse across selectors with independently rescored validation."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from models.n2p3net import N2P3Net
from tests.test_target_experiments import binary_dataset


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
