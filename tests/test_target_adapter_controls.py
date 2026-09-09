"""E1 state invariants and B0 exposure accounting through SubjectAdapter.fit."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from models.n2p3net import N2P3Net
from transfer.recipes import TargetRecipe
from transfer.subject_adapter import SubjectAdapter


def _trunk():
    return N2P3Net(
        n_channels=3,
        n_times=128,
        sfreq=128.0,
        tmin_s=-0.2,
        pooling_mode="ms_flatten",
        temporal_filters=2,
        mst_features_per_scale=2,
    )


def _data(n=12):
    return np.random.default_rng(101).normal(3.0, 2.0, (n, 3, 128)).astype("float32"), np.arange(
        n
    ) % 2


def test_statistics_only_updates_sequential_bn_population_moments_without_parameters():
    X, y = _data()
    source = _trunk()
    original = {name: value.detach().clone() for name, value in source.named_parameters()}
    recipe = TargetRecipe(
        name="stats",
        input_statistics="target_prefix",
        bn_buffers="calibration",
        bn_affine="frozen",
        weights="none",
        dropout="off",
        epochs=1,
    )
    adapter = SubjectAdapter(copy.deepcopy(source), config=recipe.adapter_config()).fit(X, y)
    assert all(
        torch.equal(original[name], value) for name, value in adapter.trunk.named_parameters()
    )
    assert adapter.parameter_count() == 0
    assert adapter.last_history["fit_budget"]["updates"] == 0
    records = adapter.last_history["bn_reestimation"]
    assert records and all(item["variance_convention"] == "population_ddof0" for item in records)
    # Independent full-calibration forward observes each BN's final upstream transform.
    mean, std = adapter.fitted_input_statistics()
    standardized = torch.tensor((X - mean) / std)
    captures = {}

    def collect(name):
        def hook(_module, args):
            captures[name] = args[0].detach().clone()

        return hook

    hooks = [
        module.register_forward_pre_hook(collect(name))
        for name, module in adapter.trunk.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    with torch.no_grad():
        adapter.trunk(standardized)
    for hook in hooks:
        hook.remove()
    for name, module in adapter.trunk.named_modules():
        if name in captures:
            values = captures[name].double().movedim(1, 0).flatten(1)
            torch.testing.assert_close(
                module.running_mean.double(), values.mean(1), atol=2e-6, rtol=2e-5
            )
            torch.testing.assert_close(
                module.running_var.double(), values.var(1, unbiased=False), atol=2e-6, rtol=2e-5
            )
    state = copy.deepcopy(adapter.trunk.state_dict())
    assert np.array_equal(adapter.predict_logit(X), adapter.predict_logit(X))
    assert all(
        torch.equal(value, adapter.trunk.state_dict()[name]) for name, value in state.items()
    )


def test_bn_reestimation_is_invariant_to_collection_chunk_size():
    X, y = _data()
    source = _trunk()
    recipe = TargetRecipe(
        name="stats",
        bn_buffers="calibration",
        bn_affine="frozen",
        weights="none",
        dropout="off",
        epochs=1,
    )
    adapters = [
        SubjectAdapter(
            copy.deepcopy(source),
            config=replace(recipe.adapter_config(), bn_reestimation_batch_size=chunk),
        ).fit(X, y)
        for chunk in (1, 5, len(X))
    ]
    for other in adapters[1:]:
        for name, tensor in adapters[0].trunk.state_dict().items():
            torch.testing.assert_close(tensor, other.trunk.state_dict()[name], atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(adapters[0].predict_logit(X), other.predict_logit(X), atol=2e-5)


def test_bn_affine_and_dropout_are_independent_and_cannot_change_batch_permutations():
    X, y = _data()
    source = _trunk()
    base = TargetRecipe(
        name="matrix",
        bn_buffers="source",
        bn_affine="frozen",
        dropout="off",
        epochs=2,
        batch_size=5,
    )
    adapters = [
        SubjectAdapter(copy.deepcopy(source), config=recipe.adapter_config()).fit(X, y)
        for recipe in (
            base,
            replace(base, bn_affine="trainable"),
            replace(base, dropout="checkpoint"),
        )
    ]
    bn_parameters = {
        f"{name}.{key}": parameter.detach().clone()
        for name, module in source.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        for key, parameter in module.named_parameters(recurse=False)
    }
    frozen = dict(adapters[0].trunk.named_parameters())
    affine = dict(adapters[1].trunk.named_parameters())
    assert all(torch.equal(value, frozen[name]) for name, value in bn_parameters.items())
    assert any(not torch.equal(value, affine[name]) for name, value in bn_parameters.items())
    for adapter in adapters[1:]:
        assert (
            adapter.last_history["fit_budget"]["batch_sequence_sha256"]
            == (adapters[0].last_history["fit_budget"]["batch_sequence_sha256"])
        )
    assert not np.array_equal(adapters[0].predict_logit(X), adapters[2].predict_logit(X))


def test_fixed_updates_records_partial_final_epoch_without_duplicate_padding():
    X, y = _data()
    recipe = TargetRecipe(name="updates", epochs=1, max_updates=5, batch_size=5)
    adapter = SubjectAdapter(_trunk(), config=recipe.adapter_config()).fit(X, y)
    budget = adapter.last_history["fit_budget"]
    assert budget["updates"] == 5
    assert budget["draws"] == 22
    assert budget["batch_sizes"] == [5, 5, 2, 5, 5]
    assert budget["epoch_draws"] == [12, 10]
    assert len(adapter.last_history["train_losses"]) == 2


def test_compact_head_only_freezes_backbone_parameters_but_updates_online_bn():
    from transfer.recipes import compact_target_recipes

    X, y = _data()
    source = _trunk()
    parameters = {name: value.detach().clone() for name, value in source.named_parameters()}
    buffers = {name: value.detach().clone() for name, value in source.named_buffers()}
    recipes = {recipe.name: replace(recipe, epochs=2) for recipe in compact_target_recipes()}
    head = SubjectAdapter(copy.deepcopy(source),
        config=recipes["compact_head_only"].adapter_config()).fit(X, y)
    full = SubjectAdapter(copy.deepcopy(source),
        config=recipes["compact_full_fine"].adapter_config()).fit(X, y)
    assert all(torch.equal(parameters[name], value)
               for name, value in head.trunk.named_parameters()
               if not name.startswith("classifier."))
    assert any(not torch.equal(parameters[name], value)
               for name, value in head.trunk.named_parameters()
               if name.startswith("classifier."))
    assert {name for name, value in head.trunk.named_parameters() if value.requires_grad} == {
        name for name in parameters if name.startswith("classifier.")}
    assert any(not torch.equal(buffers[name], value)
               for name, value in head.trunk.named_buffers()
               if name.endswith("running_mean"))
    for key in ("updates", "draws", "batch_sequence_sha256", "batch_sizes"):
        assert head.last_history["fit_budget"][key] == full.last_history["fit_budget"][key]
    state = copy.deepcopy(head.trunk.state_dict())
    head.predict_logit(X)
    assert all(torch.equal(value, head.trunk.state_dict()[name]) for name, value in state.items())


def test_refit_recomputes_bn_and_input_statistics_from_initial_source_on_authorized_prefix():
    X, y = _data(16)
    source = _trunk()
    recipe = TargetRecipe(
        name="refit",
        bn_buffers="calibration",
        bn_affine="frozen",
        dropout="off",
        epochs=2,
        batch_size=5,
        refit_full_prefix=True,
    )
    prefix = np.arange(16) < 12
    training = np.arange(16) < 6
    validation = (np.arange(16) >= 8) & prefix
    selected = SubjectAdapter(copy.deepcopy(source), config=recipe.adapter_config()).fit(
        X, y, training_mask=training, validation_mask=validation, prefix_mask=prefix
    )
    expected = SubjectAdapter(
        copy.deepcopy(source),
        config=replace(
            recipe.adapter_config(),
            epochs=selected.last_history["best_epoch"],
            refit_full_prefix=False,
        ),
    ).fit(X[prefix], y[prefix])
    np.testing.assert_array_equal(selected.predict_logit(X), expected.predict_logit(X))
    assert selected.last_history["refit_budget"]["n_fit"] == 12
    assert selected.last_history["initial_bn_reestimation"][0]["calibration_rows"] == 6
    assert selected.last_history["bn_reestimation"][0]["calibration_rows"] == 12


@pytest.mark.parametrize("failure", ["input", "update", "validation", "refit", "interrupt"])
def test_failed_adapter_fit_invalidates_predictions_and_fitted_records(monkeypatch, failure):
    X, y = _data()
    initial = _trunk()
    recipe = TargetRecipe("lifecycle", epochs=1, batch_size=4, refit_full_prefix=True)
    adapter = SubjectAdapter(copy.deepcopy(initial), config=recipe.adapter_config())
    scope = dict(training_mask=np.arange(12) < 8, validation_mask=np.arange(12) >= 8,
                 prefix_mask=np.ones(12, dtype=bool))
    adapter.fit(X, y, **scope)
    assert np.isfinite(adapter.predict_logit(X)).all()
    original_step, original_forward = torch.optim.AdamW.step, adapter._forward_head
    steps = 0

    def step(optimizer, *args, **kwargs):
        nonlocal steps
        steps += 1
        if steps == (4 if failure == "refit" else 2) and failure in {"update", "refit", "interrupt"}:
            if failure == "interrupt":
                raise KeyboardInterrupt("injected fit interruption")
            raise RuntimeError("injected fit failure")
        return original_step(optimizer, *args, **kwargs)

    def forward(*args, **kwargs):
        if failure == "validation" and not adapter.trunk.training:
            raise RuntimeError("injected validation failure")
        return original_forward(*args, **kwargs)

    with monkeypatch.context() as injected:
        injected.setattr(torch.optim.AdamW, "step", step)
        injected.setattr(adapter, "_forward_head", forward)
        broken = X.copy()
        if failure == "input":
            broken[0, 0, 0] = np.nan
        expected_error = KeyboardInterrupt if failure == "interrupt" else (
            ValueError if failure == "input" else RuntimeError)
        with pytest.raises(expected_error):
            adapter.fit(broken, y, **scope)
    with pytest.raises(RuntimeError, match="fit"):
        adapter.predict_logit(X)
    with pytest.raises(RuntimeError, match="fit"):
        adapter.predict_features(X)
    with pytest.raises(RuntimeError, match="fit"):
        adapter.fitted_input_statistics()
    assert adapter.last_history == {}
    assert adapter.calibration_logits_ is None
    adapter.fit(X, y, **scope)
    expected = SubjectAdapter(copy.deepcopy(initial), config=recipe.adapter_config()).fit(X, y, **scope)
    np.testing.assert_allclose(adapter.predict_logit(X), expected.predict_logit(X), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "max_updates,expected_epochs,expected_updates,expected_draws",
    [
        (None, 3, 9, 36),
        (5, 2, 5, 22),
    ],
)
def test_diagnostic_validation_cannot_select_epochs_change_weights_or_refit(
    max_updates,
    expected_epochs,
    expected_updates,
    expected_draws,
):
    X, y = _data(16)
    source = _trunk()
    recipe = TargetRecipe("diagnostic", epochs=3, max_updates=max_updates, batch_size=5)
    baseline = SubjectAdapter(copy.deepcopy(source), config=recipe.adapter_config()).fit(
        X[:12], y[:12]
    )
    diagnostic = replace(
        recipe.adapter_config(),
        validation_behavior="diagnostics_only",
        early_stop_patience=1,
        early_stop_min_delta=1e9,
        refit_full_prefix=True,
    )
    first = SubjectAdapter(copy.deepcopy(source), config=diagnostic).fit(
        X,
        y,
        training_mask=np.arange(16) < 12,
        validation_mask=np.arange(16) >= 12,
        prefix_mask=np.ones(16, bool),
    )
    changed_X, changed_y = X.copy(), y.copy()
    changed_X[12:] *= 100
    changed_y[12:] = 1 - changed_y[12:]
    changed = SubjectAdapter(copy.deepcopy(source), config=diagnostic).fit(
        changed_X,
        changed_y,
        training_mask=np.arange(16) < 12,
        validation_mask=np.arange(16) >= 12,
        prefix_mask=np.ones(16, bool),
    )
    for fitted in (first, changed):
        assert fitted.last_history["best_epoch"] is None
        assert fitted.last_history["best_val_loss"] is None
        assert not fitted.last_history["refit_full_prefix"]
        assert len(fitted.last_history["train_losses"]) == expected_epochs
        assert len(fitted.last_history["validation_curve"]) == expected_epochs
        assert fitted.last_history["fit_budget"]["updates"] == expected_updates
        assert fitted.last_history["fit_budget"]["draws"] == expected_draws
        assert fitted.fitted_state_digest() == baseline.fitted_state_digest()
        np.testing.assert_array_equal(fitted.predict_logit(X), baseline.predict_logit(X))
    assert first.last_history["val_losses"] != changed.last_history["val_losses"]
