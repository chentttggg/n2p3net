"""Zero-fit counterexamples for target forward budgets and event-label authority."""
from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from data.epochs import (
    load_epoch_dataset,
    materialize_dataset_identity,
    save_epoch_dataset,
    select_scheduled_events,
)
from data.preprocess import RawSampleScope
from tests.support.datasets import binary_dataset, reference_dataset
from transfer.binary import BinaryTimeSplit
from transfer.calibration_candidates import calibration_decision_groups
from transfer.epoch_selection import (
    EpochSelectionPolicy,
    IndependentEpochSelectionData,
    epoch_selection_budget,
    fit_with_epoch_selection,
)
from transfer.recipes import TargetRecipe, target_batch_budget
from transfer.subject_adapter import SubjectAdapter
from transfer.target_experiments import (
    apply_correctness_budget,
    build_target_tasks,
    selector_fit_budget,
)


def direct_data(candidates=9, groups=3):
    count = candidates * 5 * groups
    data = reference_dataset(np.zeros((count, 3, 4), np.float32), reference="average")
    shown = np.tile(np.arange(candidates).astype(str), count // candidates)
    owner = np.repeat(np.arange(groups).astype(str), candidates * 5)
    data = replace(data, y=(shown == "1").astype(np.int64),
        metadata=pd.DataFrame({"block_id": owner}),
        event_timeline=replace(data.event_timeline, candidate_ids=shown,
            target_candidate_ids=np.repeat("1", count), group_ids=owner,
            repetition_indices=np.tile(np.repeat(np.arange(5), candidates), groups)),
        provenance={**data.provenance, "candidate_vocabulary": np.arange(candidates).astype(str).tolist()})
    materialize_dataset_identity(data)
    return data


def unified_plan(data):
    count = data.n_epochs * 2 // 3
    rows = np.arange(data.n_epochs)
    split = BinaryTimeSplit(rows[:count], rows[count:], np.empty(0, dtype=int))
    return build_target_tasks(data, {"s": (rows, {count: split})}, stage="unified",
        batch_size=32, source_prior_weight=.2,
        unified_training_budget={"target_epochs": 3})


@pytest.mark.parametrize("candidates", [4, 9])
def test_unified_plan_counts_actual_decision_forward_even_for_d0(candidates):
    data = direct_data(candidates)
    plan = unified_plan(data)
    count = data.n_epochs * 2 // 3
    updates = 3 * int(np.ceil(count / 32))
    for task in plan["tasks"]:
        if task["recipe"]["weights"] == "none":
            assert task["budget"]["draws"] == 0
            continue
        assert task["budget"]["updates"] == updates
        assert task["budget"]["draws"] == 3 * count + updates * (candidates * 5)
        assert task["budget"]["ordinary_draws"] == 3 * count
        assert task["budget"]["candidate_draws_upper_bound"] == updates * candidates * 5
    before = copy.deepcopy(plan)
    apply_correctness_budget(plan, epochs=1)
    for old, task in zip(before["tasks"], plan["tasks"], strict=True):
        if task["recipe"]["weights"] != "none":
            assert task["budget"]["draws"] * 3 == old["budget"]["draws"]
            assert task["budget"]["candidate_rows_per_update_upper_bound"] == candidates * 5


@pytest.mark.parametrize("weight", [0., 1.])
def test_actual_joint_forward_preserves_duplicate_rows_and_one_complete_request(weight):
    data = direct_data(groups=1)
    groups, targets, weights, _ = calibration_decision_groups(data, np.ones(data.n_epochs, bool))
    seen = []

    def head(values):
        seen.extend(values[:, 0].tolist())
        return torch.stack((values[:, 0] * 0., values[:, 0] * .01), dim=1)

    holder = SimpleNamespace(config=SimpleNamespace(batch_size=8,
        candidate_loss_temperature=1., candidate_loss_weight=weight), device="cpu", head=head)
    features = torch.arange(data.n_epochs, dtype=torch.float32).reshape(-1, 1)
    rows = torch.tensor([0, 1, 2, 3])
    context = (list(zip(groups, targets, strict=True)), weights)
    _, loss, diagnostics = SubjectAdapter._joint_training_batch(holder, features, None, rows,
        context, np.random.default_rng(7), live_forward=False)
    assert seen == [0., 1., 2., 3., *map(float, range(45))]
    assert diagnostics["requested"] == diagnostics["eligible"] == 1
    assert diagnostics["draws"] == 45 and torch.isfinite(loss)


def test_selector_and_epoch_selection_include_both_actual_forward_streams():
    recipe = TargetRecipe("joint", epochs=2, batch_size=8, candidate_loss_weight=0.,
                          bn_buffers="calibration")
    parameters = dict(calibration_rows=20, inner_rows=11, inner_legal=True,
                      calibration_decision_rows=9, inner_decision_rows=4)
    # Refit: 6 updates, 40 CE + 54 extra; inner: 4 updates, 22 CE + 16 extra.
    for result in (selector_fit_budget([recipe.record()], **parameters),
                   epoch_selection_budget(recipe, EpochSelectionPolicy("patience"), **parameters)):
        assert result["optimizer_updates_upper_bound"] == 10
        assert result["trial_draws_upper_bound"] == 132
    capped = replace(recipe, max_updates=4)
    result = selector_fit_budget([capped.record()], **parameters)
    assert result["optimizer_updates_upper_bound"] == 8
    assert result["trial_draws_upper_bound"] == 28 + 22 + 36 + 16
    ordinary = target_batch_budget(20, 8, epochs=2)
    assert ordinary["draws"] == 40 and "candidate_draws_upper_bound" not in ordinary


def test_diagnostic_matrix_preserves_budget_for_zero_and_positive_decision_weights():
    from transfer.decision_experiments import expand_decision_diagnostics

    data = direct_data()
    plan = unified_plan(data)
    result = expand_decision_diagnostics(plan, {"schema": "n2p3_prior_decision_diagnostic/1",
        "full_joint_interpretation": "exploratory_not_low_G_sufficiency", "decision_weight": 1.})
    for task in result["tasks"]:
        if task["recipe"]["weights"] == "none":
            continue
        assert task["budget"]["draws"] == 3 * 90 + 9 * 45


def test_correctness_recalculation_uses_independent_inner_rows_and_decision_roster():
    recipe = TargetRecipe("joint", epochs=5, batch_size=8, candidate_loss_weight=0.,
                          bn_buffers="calibration")
    plan = {"tasks": [{"scope_id": "s:20", "arm": "joint", "recipe": recipe.record(),
        "training_rows": 20, "epoch_selection": EpochSelectionPolicy("patience").record()}],
        "scopes": {"s:20": {"inner_training_rows": list(range(16)),
            "inner_split": {"legal": True, "training_rows": 11},
            "decision_rows_per_update": {"calibration": 9, "inner_training": 4}}}}
    apply_correctness_budget(plan, epochs=2)
    assert plan["tasks"][0]["selection_budget"]["trial_draws_upper_bound"] == 132


def test_variable_row_column_roster_uses_authorized_maximum_without_held_labels():
    from tests.support.candidate import candidate_dataset
    from transfer.calibration_candidates import calibration_decision_rows_per_update

    data = candidate_dataset(n_selections=3, repetitions_per_selection=(1, 3, 2))
    data = replace(data, event_timeline=replace(data.event_timeline, complete=True,
        candidate_ids=data.metadata["candidate_id"].to_numpy(dtype=str),
        repetition_indices=data.metadata["repetition_index"].to_numpy(dtype=np.int64)))
    allowed = np.arange(data.n_epochs) < 48
    assert calibration_decision_rows_per_update(data, allowed) == 36
    # Both 12-row and 36-row groups are legal; the held 24-row truth is unread.
    data.metadata.loc[48:, "target_col"] = 999
    data.y[48:] = 999
    assert calibration_decision_rows_per_update(data, allowed) == 36
    budget = target_batch_budget(48, 32, epochs=2, decision_rows_per_update=36)
    assert budget["draws"] == 96 + 4 * 36


def _cached_view(root, data, start, stop):
    view = select_scheduled_events(data,
        (np.arange(data.n_epochs) >= start) & (np.arange(data.n_epochs) < stop), reason="synthetic label authority")
    scope = RawSampleScope("synthetic", start * 100, stop * 100, tuple(range(start, stop)))
    view.provenance["source_preparation"] = {"raw_scope": scope.record()}
    path = root / f"view_{start}_{stop}.npz"
    save_epoch_dataset(path, view)
    return load_epoch_dataset(path, validation="attested")


@pytest.fixture
def cached_views(tmp_path):
    data = binary_dataset()
    parent = _cached_view(tmp_path, data, 0, 20)
    train = _cached_view(tmp_path, data, 0, 14)
    valid = _cached_view(tmp_path, data, 14, 20)
    return parent, train, valid


@pytest.mark.parametrize("changed", ["training", "validation"])
def test_view_constructor_rejects_changed_event_label_with_both_classes_retained(cached_views, changed):
    parent, train, valid = cached_views
    target = train if changed == "training" else valid
    target.y[0] = 1 - target.y[0]
    assert set(target.y) == {0, 1}
    with pytest.raises(ValueError, match="labels differ.*authorized calibration"):
        IndependentEpochSelectionData.from_views(train, valid, parent)


@pytest.mark.parametrize("row", [0, 17])
def test_runtime_manual_selection_cannot_bypass_parent_event_labels(monkeypatch, row):
    import transfer.epoch_selection as selection

    parent = binary_dataset()
    allowed = np.arange(parent.n_epochs) < 20
    inner = select_scheduled_events(parent, np.arange(parent.n_epochs) < 20, reason="authorized inner")
    inner.y[row] = 1 - inner.y[row]
    train = np.arange(20) < 14
    explicit = IndependentEpochSelectionData(inner, train, ~train, {"legal": True})

    def forbidden(*args, **kwargs):
        pytest.fail("event-label mismatch must be rejected before any fit")

    monkeypatch.setattr(selection, "fit_calibration_candidate", forbidden)
    with pytest.raises(ValueError, match="labels differ.*authorized calibration"):
        fit_with_epoch_selection(parent, TargetRecipe("source"), allowed,
            policy=EpochSelectionPolicy("patience"), selection_data=explicit)


def test_event_alignment_accepts_independent_values_and_row_order(cached_views, tmp_path):
    parent, train, valid = cached_views
    # Swap every opposite-label pair and explicitly remap physical event rows.
    order = np.arange(train.n_epochs).reshape(-1, 2)[:, ::-1].ravel()
    inverse = np.argsort(order)
    train = replace(train, X=train.X[order] * 3., y=train.y[order],
        metadata=train.metadata.iloc[order].reset_index(drop=True),
        event_timeline=replace(train.event_timeline,
            evidence_indices=inverse[train.event_timeline.evidence_indices]))
    save_epoch_dataset(tmp_path / "reordered.npz", train)
    train = load_epoch_dataset(tmp_path / "reordered.npz", validation="attested")
    selection = IndependentEpochSelectionData.from_views(train, valid, parent)
    assert selection.training_mask.sum() == 14 and selection.validation_mask.sum() == 6


def test_event_alignment_accepts_independently_dropped_rows(cached_views, tmp_path):
    parent, train, valid = cached_views
    train = select_scheduled_events(train, np.arange(train.event_timeline.n_events) != 4,
        reason="independent boundary/QC retention", preserve_schedule=True)
    save_epoch_dataset(tmp_path / "retained.npz", train)
    train = load_epoch_dataset(tmp_path / "retained.npz", validation="attested")
    selection = IndependentEpochSelectionData.from_views(train, valid, parent)
    assert selection.training_mask.sum() == 13 and selection.validation_mask.sum() == 6


def test_runtime_rejects_outer_event_before_fit(monkeypatch):
    import transfer.epoch_selection as selection

    parent = binary_dataset()
    train = np.arange(parent.n_epochs) < 14
    explicit = IndependentEpochSelectionData(parent, train, ~train, {"legal": True})
    monkeypatch.setattr(selection, "fit_calibration_candidate", lambda *a, **k: pytest.fail("no fit"))
    with pytest.raises(ValueError, match="escape authorized calibration events"):
        fit_with_epoch_selection(parent, TargetRecipe("source"), np.arange(parent.n_epochs) < 20,
            policy=EpochSelectionPolicy("patience"), selection_data=explicit)


def test_censor_basis_first_axis_is_response_support():
    from transfer.continuous_decision import _event_design, smooth_erp_basis

    basis = smooth_erp_basis(100., response_seconds=.8, n_basis=8)
    assert basis.shape == (80, 8) and len(basis) == 80
    onsets = np.array([0, 90])
    censored = int(np.sum(onsets + len(basis) > 100))
    assert censored == 1
    design = _event_design(onsets, np.array([0, 1]), 100, basis)
    np.testing.assert_array_equal(design[90:, 8:], basis[:10])
