"""Global configuration, independent counters, source isolation and full refit."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from baselines.deep import DeepConfig
from research.source_selection import fit_source_trajectory
from tests.support.source_programs import source_early_stopping_config, source_early_stopping_data
from train.early_stopping import PatienceStopper, UnifiedTrainingBudget


def test_reference_policy_is_independent_of_fixed_target_budget():
    budget = UnifiedTrainingBudget()
    assert budget.target_epochs == 90
    assert budget.reference_policy() == replace(budget, target_epochs=60).reference_policy()
    assert budget.reference_policy().source_max_epochs == 12
    with pytest.raises(ValueError):
        UnifiedTrainingBudget(target_epochs=0)
    with pytest.raises(ValueError):
        UnifiedTrainingBudget(development_people_per_domain=True)


def run_source(data, cfg):
    return fit_source_trajectory(data, np.arange(32), config=cfg,
        source_cache_sha256="a" * 64, source_snapshot_sha256="b" * 64,
        excluded_source_datasets=["outer"], device=torch.device("cpu"))


def test_global_value_does_not_share_counter_state_and_resets_on_improvement():
    a, b = PatienceStopper(2, .01), PatienceStopper(2, .01)
    assert a.observe(1.)
    assert not a.observe(.995)
    assert a.observe(.98)
    assert a.bad_checks == 0 and b.checks == 0
    assert not a.observe(.985)
    assert not a.exhausted
    assert not a.observe(.986)
    assert a.exhausted and a.record()["stop_reason"] == "patience"
    assert b.observe(2.) and not b.exhausted
    with pytest.raises(ValueError, match="finite"):
        b.observe(float("nan"))


def test_source_early_stopping_refits_selected_epochs_from_origin():
    data, cfg = source_early_stopping_data(), source_early_stopping_config()
    model, payloads = run_source(data, cfg)
    assert set(payloads) == {2}
    payload = payloads[2]
    selection = payload["source_epoch_selection"]
    assert selection["stop_reason"] == "patience"
    assert selection["validation_checks"] == 3 and selection["selected_epochs"] == 1
    assert selection["inner_history"]["optimizer_updates"] == 3
    assert selection["total_optimizer_updates"] == model.last_history["optimizer_updates"] == 5
    train_people = set(data.subject_ids[selection["training_rows"]])
    validation_people = set(data.subject_ids[selection["validation_rows"]])
    assert train_people.isdisjoint(validation_people)
    assert len(payload["source_fit_ledger"]["fit_rows"]) == 32
    expected, _ = run_source(data, replace(cfg, global_early_stopping=None, fixed_updates=2))
    for name, value in expected.model_.state_dict().items():
        torch.testing.assert_close(value, payload["trunk_state_dict"][name], rtol=0, atol=0)


def test_source_validation_and_outside_rows_cannot_enter_inner_training():
    data, cfg = source_early_stopping_data(), source_early_stopping_config()
    _, original = run_source(data, cfg)
    before = original[2]["source_epoch_selection"]
    changed = copy.deepcopy(data)
    rows = before["validation_rows"]
    changed.X[rows] *= 50
    changed.y[rows] = 1 - changed.y[rows]
    changed.X[32:] *= 10000
    _, after = run_source(changed, cfg)
    assert before["inner_history"]["update_losses"] == after[2]["source_epoch_selection"]["inner_history"]["update_losses"]
    assert before["validation_curve"] != after[2]["source_epoch_selection"]["validation_curve"]


def test_fixed_and_global_budgets_cannot_be_combined():
    with pytest.raises(ValueError, match="cannot also request fixed"):
        DeepConfig(fixed_updates=200, global_early_stopping={})


def test_source_exact_reuse_keeps_stopping_evidence_and_changes_policy_invalidate_cache(tmp_path):
    from experiments.run_source_training_chain import fit_or_reuse_source_trajectory

    common = dict(registry={}, dataset=source_early_stopping_data(), rows=np.arange(32), config=source_early_stopping_config(),
        source_cache_sha256="a" * 64, source_snapshot_sha256="b" * 64,
        excluded_source_datasets=["outer"], device="cpu", evidence_scope="software_correctness")
    first = fit_or_reuse_source_trajectory(**common, destination=tmp_path / "a", task_id="a")
    second = fit_or_reuse_source_trajectory(**common, destination=tmp_path / "b", task_id="b")
    assert first["physical_optimizer_updates"] == 5
    assert second["physical_optimizer_updates"] == 0
    saved = torch.load(second["checkpoints"][2], weights_only=False)
    assert saved["training_contract"]["validation"]["early_stopping"]["selected_epochs"] == 1
    third = fit_or_reuse_source_trajectory(**{**common, "config": replace(source_early_stopping_config(),
        global_early_stopping=replace(source_early_stopping_config().global_early_stopping, patience=3))},
        destination=tmp_path / "c", task_id="c")
    assert third["physical_optimizer_updates"] == 6
