"""Target epoch selection must preserve fit scopes and account for full refits."""
import copy
import hashlib
from dataclasses import replace

import numpy as np
import pytest
import torch

from models.n2p3net import N2P3Net
from tests.support.checkpoints import write_checkpoint
from tests.support.datasets import binary_dataset
from train.early_stopping import GlobalEarlyStoppingPolicy
from transfer.binary import binary_time_split
from transfer.calibration_candidates import CalibrationCandidateFitCache, fit_calibration_candidate
from transfer.epoch_selection import EpochSelectionPolicy, fit_with_epoch_selection
from transfer.recipes import TargetRecipe
from transfer.result_reuse import TargetResultReuse
from transfer.target_experiments import build_target_tasks, execute_target_tasks


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
    with pytest.raises(ValueError, match="requires a legal"):
        fit_with_epoch_selection(dataset, recipe, allowed, source_trunk=source,
            policy=EpochSelectionPolicy("patience", require_validation=True))
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


def _global_manifest_for(data, policy):
    split = binary_time_split(np.arange(40) * 2., np.arange(40) * 2. + .8,
        calibration_trials=20, maximum_calibration_trials=20, tmin_s=-.2)
    return build_target_tasks(data, {"p1": (np.arange(40), {20: split})}, stage="P1",
        batch_size=512, global_early_stopping=policy.record())


def test_global_policy_reaches_every_neural_arm_and_removes_fixed_updates():
    policy = GlobalEarlyStoppingPolicy(patience=9, target_max_epochs=100)
    manifest = _global_manifest_for(binary_dataset(), policy)
    neural = [task for task in manifest["tasks"] if task["recipe"]["weights"] != "none"]
    assert len(neural) == 8
    for task in neural:
        assert task["epoch_selection"] == policy.target_selection()
        assert task["recipe"]["max_updates"] is None
        assert task["recipe"]["epochs"] == 100
        assert task["validation_rows"] > 0
    lda = manifest["tasks"][-1]
    assert "epoch_selection" not in lda and lda["budget"]["updates"] == 0


def test_global_selection_reuse_preserves_selected_recipe_and_physical_cost(tmp_path):
    data = binary_dataset()
    checkpoint = write_checkpoint(tmp_path / "source.pt", data, source_snapshot_sha256="a" * 64)
    payload = torch.load(checkpoint, weights_only=False)
    policy = GlobalEarlyStoppingPolicy(patience=1, min_delta=100., target_max_epochs=4)
    manifest = _global_manifest_for(data, policy)
    # Policy propagation across all arms is checked above. Reuse execution needs
    # the plain and residual neural state forms plus the non-neural LDA form.
    arms = {"p1_full_fine", "p1_residual_head", "p1_waveform_lda"}
    manifest["tasks"] = [task for task in manifest["tasks"] if task["arm"] in arms]
    manifest["task_count"] = len(manifest["tasks"])
    assert manifest["task_count"] == 3
    context = dict(target_study="synthetic_p300", source_seed=1,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), target_cache_sha256="b" * 64,
        evaluation_snapshot={"archive_sha256": "c" * 64}, target_protocol={"kind": "single_recording_prefix"},
        evidence_scope="software_correctness")
    reuse = TargetResultReuse()
    first = execute_target_tasks(data, payload, manifest, output=tmp_path / "a", context=context,
                                 device="cpu", result_reuse=reuse)
    second = execute_target_tasks(data, payload, manifest, output=tmp_path / "b", context=context,
                                  device="cpu", result_reuse=reuse)
    assert first["failed_cells"] == second["failed_cells"] == 0
    assert second["fit_execution"]["fit_calls"] == 0
    assert second["fit_execution"]["cache_hits"] == 5
    for index, cell in enumerate(second["cells"][:-1]):
        assert cell["recipe"]["epochs"] == 1
        assert cell["requested_recipe"]["epochs"] == 4
        assert cell["epoch_selection_result"]["inner_epochs_run"] == 2
        assert cell["epoch_selection_result"]["stop_reason"] == "patience"
        assert cell["epoch_selection_result"]["physical_optimizer_updates"] == 0
        assert cell["epoch_selection_result"]["physical_trial_draws"] == 0
        saved = torch.load(tmp_path / f"b/{index:05d}.pt", weights_only=False)
        assert saved["recipe"] == cell["recipe"]
        assert ("feature_transfer" in saved["architecture"]) == (cell["arm"] == "p1_residual_head")
        assert cell["fit_execution"]["fit_calls"] == 0
    changed = _global_manifest_for(data, GlobalEarlyStoppingPolicy(patience=2, min_delta=100., target_max_epochs=4))
    changed["tasks"] = changed["tasks"][:1]
    third = execute_target_tasks(data, payload, changed, output=tmp_path / "c", context=context,
                                 device="cpu", result_reuse=reuse)
    assert third["failed_cells"] == 0 and third["fit_execution"]["fit_calls"] == 2
    assert third["cells"][0]["epoch_selection_result"]["inner_epochs_run"] == 3
