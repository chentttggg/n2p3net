"""Structural adaptation contracts exercised through fitting and persistence."""

import copy
import hashlib
from dataclasses import replace

import numpy as np
import pytest
import torch

from models.n2p3net import N2P3Net
from tests.support.checkpoints import write_checkpoint
from tests.support.datasets import binary_dataset
from transfer.binary import binary_time_split
from transfer.recipes import TargetRecipe, p1_target_recipes
from transfer.subject_adapter import SubjectAdapter


@pytest.mark.parametrize("weights,expected", [
    ("head", {"readout"}), ("spatial", {"spatial", "readout"}),
    ("temporal", {"temporal", "readout"}),
    ("non_bn", {"spatial", "temporal", "readout"}),
])
def test_structural_fit_changes_only_declared_groups_and_predict_is_read_only(weights, expected):
    dataset = binary_dataset()
    torch.manual_seed(13)
    model = N2P3Net(3, n_times=dataset.n_times)
    before = {name: tuple(p.detach().clone() for p in values)
              for name, values in model.adaptation_parameter_groups().items()}
    recipe = TargetRecipe("probe", weights=weights, bn_buffers="source", bn_affine="frozen",
                          dropout="off", max_updates=3, batch_size=16)
    fitted = SubjectAdapter(model, config=recipe.adapter_config()).fit(dataset.X, dataset.y)
    changed = {name for name, values in model.adaptation_parameter_groups().items()
               if any(not torch.equal(a, b) for a, b in zip(before[name], values, strict=True))}
    assert changed == expected
    assert (fitted.last_history["feature_cache_key"] is not None) == (weights == "head")
    state = fitted.fitted_state_digest()
    scores = fitted.predict_logit(dataset.X)
    np.testing.assert_array_equal(scores, fitted.predict_logit(dataset.X))
    assert state == fitted.fitted_state_digest()


def test_unknown_parameter_cannot_silently_enter_a_structural_fit():
    dataset = binary_dataset()
    model = N2P3Net(3, n_times=dataset.n_times)
    model.unowned = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError, match="own every"):
        SubjectAdapter(model, config=TargetRecipe("probe", weights="spatial").adapter_config()).fit(
            dataset.X, dataset.y)


def test_partial_refit_never_reads_outside_the_explicit_calibration_prefix():
    dataset = binary_dataset()
    torch.manual_seed(8)
    source = N2P3Net(3, n_times=dataset.n_times)
    recipe = TargetRecipe("refit", weights="temporal", bn_affine="frozen", dropout="off",
                          epochs=2, batch_size=8, refit_full_prefix=True)
    index = np.arange(dataset.n_epochs)
    kwargs = dict(training_mask=index < 16, validation_mask=(index >= 16) & (index < 24),
                  prefix_mask=index < 28)
    first = SubjectAdapter(copy.deepcopy(source), config=recipe.adapter_config()).fit(
        dataset.X, dataset.y, **kwargs)
    changed = dataset.X.copy()
    changed[28:] *= 1000
    labels = dataset.y.copy()
    labels[28:] = 1 - labels[28:]
    second = SubjectAdapter(copy.deepcopy(source), config=recipe.adapter_config()).fit(
        changed, labels, **kwargs)
    assert first.fitted_state_digest() == second.fitted_state_digest()
    assert first.last_history["fit_scope"]["refit_rows"] == 28
    assert first.last_history["parameter_groups"]["spatial"]["trainable_parameters"] == 0


def test_p1_matrix_runs_all_arms_and_saves_the_residual_architecture(tmp_path):
    from transfer.target_experiments import (
        apply_correctness_budget,
        build_target_tasks,
        execute_target_tasks,
    )

    dataset = binary_dataset()
    source = write_checkpoint(tmp_path / "source.pt", dataset, source_snapshot_sha256="a" * 64)
    split = binary_time_split(np.arange(40) * 2., np.arange(40) * 2. + .8,
        calibration_trials=20, maximum_calibration_trials=20, tmin_s=-.2)
    plan = build_target_tasks(dataset, {"p1": (np.arange(40), {20: split})},
                              stage="P1", batch_size=512, seed=9)
    assert plan["task_count"] == 9
    assert len(p1_target_recipes()) == 9
    residual = next(t for t in plan["tasks"] if t["arm"] == "p1_residual_head")
    assert TargetRecipe(**residual["recipe"]).record() == residual["recipe"]
    apply_correctness_budget(plan, epochs=1)
    context = dict(target_study="synthetic_p300", source_seed=1,
        checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), target_cache_sha256="b" * 64,
        evaluation_snapshot={"archive_sha256": "c" * 64},
        target_protocol={"kind": "single_recording_prefix"})
    report = execute_target_tasks(dataset, torch.load(source, weights_only=False), plan,
                                  output=tmp_path / "run", context=context, device="cpu")
    assert report["failed_cells"] == 0, [cell.get("failure") for cell in report["cells"]]
    for index, cell in enumerate(report["cells"]):
        saved = torch.load(tmp_path / f"run/{index:05d}.pt", weights_only=False)
        assert saved["recipe"] == cell["recipe"]
        if cell["arm"] == "p1_residual_head":
            assert "feature_transfer" in saved["architecture"]
            groups = cell["fit_history"]["parameter_groups"]
            assert groups["target_adapter"]["trainable_parameters"] > 0
            assert groups["source_extensions"]["trainable_parameters"] == 0
            assert groups["temporal"]["trainable_parameters"] == 0


def test_p1_source_and_scratch_have_the_same_budget_and_factorial_is_bn_matched():
    recipes = {r.name: r for r in p1_target_recipes(seed=31)}
    full = recipes["p1_full_fine"].record()
    scratch = recipes["p1_scratch"].record()
    assert {key for key in full if full[key] != scratch[key]} == {"name", "initialization"}
    factorial = [recipes[f"p1_s{s}t{t}"] for s in (0, 1) for t in (0, 1)]
    assert all(r.bn_buffers == "online" and r.bn_affine == "frozen" and r.dropout == "off"
               for r in factorial)
    assert all(r.max_updates == 200 for r in factorial)
    with pytest.raises(ValueError, match="requires adapter"):
        replace(recipes["p1_residual_head"], weights="head")


def test_p2_reuses_scratch_and_lda_across_distinct_sources_but_refits_transfer(tmp_path):
    from transfer.result_reuse import TargetResultReuse
    from transfer.target_experiments import (
        apply_correctness_budget,
        build_target_tasks,
        execute_target_tasks,
    )

    dataset = binary_dataset()
    source = write_checkpoint(tmp_path / "source.pt", dataset, source_snapshot_sha256="a" * 64)
    payload = torch.load(source, weights_only=False)
    split = binary_time_split(np.arange(40) * 2., np.arange(40) * 2. + .8,
        calibration_trials=20, maximum_calibration_trials=20, tmin_s=-.2)
    plan = build_target_tasks(dataset, {"p1": (np.arange(40), {20: split})},
                              stage="P2", batch_size=512, seed=9)
    apply_correctness_budget(plan, epochs=1)
    context = dict(target_study="synthetic_p300", source_seed=1,
        checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), target_cache_sha256="b" * 64,
        evaluation_snapshot={"archive_sha256": "c" * 64},
        target_protocol={"kind": "single_recording_prefix"}, plan={"source_arm": "first"})
    cache = TargetResultReuse()
    first = execute_target_tasks(dataset, payload, plan, output=tmp_path / "first", context=context,
                                 device="cpu", result_reuse=cache)
    changed = copy.deepcopy(payload)
    changed["trunk_state_dict"]["classifier.weight"] += .25
    second = execute_target_tasks(dataset, changed, plan, output=tmp_path / "second",
        context={**context, "checkpoint_sha256": "d" * 64, "plan": {"source_arm": "second"}},
        device="cpu", result_reuse=cache)
    assert first["failed_cells"] == second["failed_cells"] == 0
    assert second["fit_execution"] == {"requested_fits": 3, "fit_calls": 1, "cache_hits": 2}
    assert all(cell["source_arm"] == "second" for cell in second["cells"])
    for left, right in zip(first["cells"], second["cells"], strict=True):
        if "scratch" in left["arm"] or "lda" in left["arm"]:
            assert left["fitted_state_sha256"] == right["fitted_state_sha256"]
            assert right["prediction_reuse"]["fit_executed"] is False
