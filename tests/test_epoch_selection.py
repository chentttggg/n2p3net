"""Target epoch selection must preserve fit scopes and account for full refits."""
import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from models.n2p3net import N2P3Net
from tests.test_target_experiments import binary_dataset
from transfer.calibration_candidates import CalibrationCandidateFitCache, fit_calibration_candidate
from transfer.epoch_selection import EpochSelectionPolicy, fit_with_epoch_selection
from transfer.recipes import TargetRecipe


def source_for(dataset):
    return N2P3Net(3, n_times=dataset.n_times, sfreq=dataset.preprocessing.sfreq,
        tmin_s=dataset.preprocessing.tmin_ms / 1000, temporal_filters=2, mst_features_per_scale=2)


def test_patience_stops_actual_training_but_full_trajectory_keeps_scanning_and_refits_from_origin():
    dataset = binary_dataset()
    source = source_for(dataset)
    allowed = np.arange(dataset.n_epochs) < 20
    recipe = TargetRecipe("high", epochs=8, batch_size=512, seed=12)
    cache = CalibrationCandidateFitCache()
    fitted, stopped = fit_with_epoch_selection(dataset, recipe, allowed,
        policy=EpochSelectionPolicy("patience", patience=2, min_delta=10), source_trunk=source, fit_cache=cache)
    full, scanned = fit_with_epoch_selection(dataset, recipe, allowed,
        policy=EpochSelectionPolicy("full_trajectory", patience=2, min_delta=10), source_trunk=source, fit_cache=cache)
    assert stopped["inner_epochs_run"] == 3 and scanned["inner_epochs_run"] == 8
    assert stopped["selected_epochs"] == scanned["selected_epochs"] == 1
    assert stopped["total_optimizer_updates"] == 4 and scanned["total_optimizer_updates"] == 9
    assert stopped["total_trial_draws"] == 68 and scanned["total_trial_draws"] == 148
    assert cache.record()["fit_calls"] == 3 and cache.record()["cache_hits"] == 1
    expected = fit_calibration_candidate(dataset, replace(recipe, epochs=1), allowed, source_trunk=source)
    for name, value in expected.state.items():
        torch.testing.assert_close(fitted.state[name], value, rtol=0, atol=0)
        torch.testing.assert_close(full.state[name], value, rtol=0, atol=0)
    assert fitted.fit_history["fit_scope"]["training_rows"] == 20


@pytest.mark.parametrize("mode", ["patience", "full_trajectory"])
def test_final_test_changes_never_change_epoch_selection_or_refit(mode):
    dataset = binary_dataset()
    source = source_for(dataset)
    allowed = np.arange(dataset.n_epochs) < 20
    recipe = TargetRecipe("high", epochs=4, batch_size=512, seed=12)
    policy = EpochSelectionPolicy(mode, patience=2)
    first, selection = fit_with_epoch_selection(dataset, recipe, allowed, policy=policy, source_trunk=source)
    changed = copy.deepcopy(dataset)
    changed.X[~allowed] *= 1000
    changed.y[~allowed] = 1 - changed.y[~allowed]
    other, repeated = fit_with_epoch_selection(changed, recipe, allowed, policy=policy, source_trunk=source)
    assert selection["selected_epochs"] == repeated["selected_epochs"]
    assert selection["inner_history"]["val_losses"] == repeated["inner_history"]["val_losses"]
    for name, value in first.state.items():
        torch.testing.assert_close(other.state[name], value, rtol=0, atol=0)


def test_illegal_inner_split_falls_back_to_same_fixed_recipe_without_selection_claim():
    dataset = binary_dataset()
    dataset.y[:20] = 0
    dataset.y[[0, 2, 4, 6]] = 1
    source = source_for(dataset)
    allowed = np.arange(dataset.n_epochs) < 20
    recipe = TargetRecipe("high", epochs=2, batch_size=512)
    fitted, record = fit_with_epoch_selection(dataset, recipe, allowed,
        policy=EpochSelectionPolicy("patience"), source_trunk=source)
    expected = fit_calibration_candidate(dataset, recipe, allowed, source_trunk=source)
    assert not record["selection_evidence"] and record["effective_mode"] == "fixed"
    assert record["fallback"] == "fixed_requested_recipe"
    assert record["inner_epochs_run"] == 0 and record["selected_epochs"] == 2
    for name, value in expected.state.items():
        torch.testing.assert_close(fitted.state[name], value, rtol=0, atol=0)


def test_validation_changes_leave_full_trajectory_training_losses_and_initialization_unchanged():
    dataset = binary_dataset()
    source = source_for(dataset)
    allowed = np.arange(dataset.n_epochs) < 20
    recipe = TargetRecipe("high", epochs=5, batch_size=512, seed=12)
    policy = EpochSelectionPolicy("full_trajectory")
    _, before = fit_with_epoch_selection(dataset, recipe, allowed, policy=policy, source_trunk=source)
    changed = copy.deepcopy(dataset)
    changed.X[16:20] *= 50
    changed.y[16:20] = 1 - changed.y[16:20]
    _, after = fit_with_epoch_selection(changed, recipe, allowed, policy=policy, source_trunk=source)
    for field in ("train_losses", "initial_state_sha256", "training_data_sha256"):
        assert before["inner_history"][field] == after["inner_history"][field]
    assert before["inner_history"]["val_losses"] != after["inner_history"]["val_losses"]


@pytest.mark.parametrize("parameters", [{"patience": 0}, {"patience": True}, {"min_delta": float("nan")},
                                        {"min_delta": float("inf")}, {"mode": "test_auc"}])
def test_epoch_policy_rejects_invalid_settings(parameters):
    with pytest.raises(ValueError):
        EpochSelectionPolicy(**parameters)
