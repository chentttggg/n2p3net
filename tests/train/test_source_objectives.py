"""Identity, gradient, fixed-dose and checkpoint tests for source objectives."""

import copy
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from baselines.deep import DeepConfig
from baselines.n2p3net import N2P3NetBaseline
from transfer.source_objective import (
    SourceObjective,
    SourceObjectiveConfig,
    source_objective_recipes,
    validate_source_objective_contract,
)


def test_combined_source_objective_matches_weighted_values_and_gradients():
    labels = np.arange(16) % 2
    domains = np.repeat(["a", "b"], 8)
    rows = torch.tensor([*range(16), 0, 3])
    original = torch.randn(16, 4, 7)
    features = original[rows].clone().requires_grad_()
    kwargs = dict(labels=labels, domains=domains, participants=domains, device="cpu")
    combined = SourceObjective(
        SourceObjectiveConfig(
            supcon_weight=0.17, mmd_weight=0.29, feature_location="post_residual"
        ),
        **kwargs,
    )
    supcon = SourceObjective(
        SourceObjectiveConfig(supcon_weight=0.17, feature_location="post_residual"), **kwargs
    )
    mmd = SourceObjective(
        SourceObjectiveConfig(mmd_weight=0.29, feature_location="post_residual"), **kwargs
    )
    actual, diagnostics = combined.loss(features, rows, feature_location="post_residual")
    left, _ = supcon.loss(features, rows, feature_location="post_residual")
    right, _ = mmd.loss(features, rows, feature_location="post_residual")
    torch.testing.assert_close(actual, left + right, rtol=0, atol=0)
    actual_gradient = torch.autograd.grad(actual, features, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(left + right, features)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)
    assert diagnostics["unique_rows"] == 16
    assert diagnostics["active_anchors"] == 16
    assert diagnostics["active_mmd_terms"] == 2
    assert torch.count_nonzero(actual_gradient[-2:]) == 0


def test_ce_only_objective_is_a_zero_without_auxiliary_identity_requirements():
    objective = SourceObjective(
        SourceObjectiveConfig(),
        labels=np.array([0, 1]),
        domains=None,
        participants=None,
        device="cpu",
    )
    features = torch.randn(2, 4, requires_grad=True)
    loss, diagnostics = objective.loss(features, torch.arange(2), feature_location="post_mst")
    assert loss.item() == 0 and not objective.config.enabled
    loss.backward()
    assert torch.equal(features.grad, torch.zeros_like(features))
    assert diagnostics["active_anchors"] == diagnostics["active_mmd_terms"] == 0


def test_source_feature_location_is_checked_at_the_loss_boundary():
    objective = SourceObjective(
        SourceObjectiveConfig(supcon_weight=0.05, feature_location="post_residual"),
        labels=np.array([0, 1, 0, 1]),
        domains=["a", "a", "b", "b"],
        participants=["p", "p", "q", "q"],
        device="cpu",
    )
    with pytest.raises(ValueError, match="feature location"):
        objective.loss(torch.randn(4, 3), torch.arange(4), feature_location="post_mst")


@pytest.mark.parametrize(
    "field,value",
    [
        ("supcon_weight", -1.0),
        ("supcon_weight", float("inf")),
        ("mmd_weight", float("nan")),
        ("mmd_weight", True),
        ("temperature", 0.0),
        ("feature_location", "guessed_features"),
    ],
)
def test_combination_config_rejects_invalid_weights_and_locations(field, value):
    with pytest.raises(ValueError):
        SourceObjectiveConfig(**{field: value})


@pytest.mark.parametrize("rows", [[], [0], [0, 0, 0], [0, 1]])
def test_combined_inactive_batch_keeps_finite_zero_gradient(rows):
    objective = SourceObjective(
        SourceObjectiveConfig(supcon_weight=0.05, mmd_weight=0.05),
        labels=np.array([0, 1, 0, 1]),
        domains=["a", "a", "b", "b"],
        participants=["p", "p", "q", "q"],
        device="cpu",
    )
    features = torch.ones(len(rows), 4, requires_grad=True)
    loss, diagnostics = objective.loss(
        features, torch.tensor(rows, dtype=torch.long), feature_location="post_mst"
    )
    assert loss.item() == 0
    assert diagnostics["supcon_inactive_batches"] == diagnostics["mmd_inactive_batches"] == 1
    assert diagnostics["inactive_batches"] == 1
    loss.backward()
    assert torch.equal(features.grad, torch.zeros_like(features))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_combination_cannot_hide_nonfinite_duplicate_features(bad):
    objective = SourceObjective(
        SourceObjectiveConfig(supcon_weight=0.05, mmd_weight=0.05),
        labels=np.array([0, 1, 0, 1]),
        domains=["a", "a", "b", "b"],
        participants=["p", "p", "q", "q"],
        device="cpu",
    )
    with pytest.raises(ValueError, match="finite"):
        objective.loss(
            torch.tensor([[1.0, 2.0], [bad, 3.0]]),
            torch.tensor([0, 0]),
            feature_location="post_mst",
        )


def _combined_checkpoint_record():
    config = SourceObjectiveConfig(
        supcon_weight=0.05, mmd_weight=0.03, feature_location="post_residual"
    )
    objective = SourceObjective(
        config,
        labels=np.arange(16) % 2,
        domains=np.repeat(["a", "b"], 8),
        participants=np.repeat(["p", "q"], 8),
        device="cpu",
    )
    _, diagnostics = objective.loss(
        torch.randn(16, 4, 7), torch.arange(16), feature_location="post_residual"
    )
    return {"config": {"source_objective": asdict(config)}, "actual_updates": 1}, objective.record(
        diagnostics, 1
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("supcon_weight", 0.4),
        ("feature_location", "post_mst"),
        ("mean_mmd_loss", float("nan")),
        ("mean_auxiliary_loss", 999.0),
        ("mmd_inactive_batches", 1),
        ("negative_mmd_terms", -1),
        ("unique_rows", True),
    ],
)
def test_combined_checkpoint_cross_checks_config_components_and_exposure(field, value):
    optimizer, record = _combined_checkpoint_record()
    validate_source_objective_contract(optimizer, {"source_auxiliary": record})
    changed = {**record, field: value}
    with pytest.raises(ValueError):
        validate_source_objective_contract(optimizer, {"source_auxiliary": changed})


def test_individual_components_can_be_inactive_without_hiding_other_component():
    config = SourceObjectiveConfig(supcon_weight=0.05, mmd_weight=0.05)
    objective = SourceObjective(
        config,
        labels=np.array([0, 1, 0, 1]),
        domains=["a", "a", "b", "b"],
        participants=["p", "p", "q", "q"],
        device="cpu",
    )
    _, diagnostics = objective.loss(torch.randn(4, 3), torch.arange(4), feature_location="post_mst")
    assert diagnostics["active_anchors"] == 4
    assert diagnostics["mmd_inactive_batches"] == 1
    assert diagnostics["supcon_inactive_batches"] == diagnostics["inactive_batches"] == 0
    validate_source_objective_contract(
        {"config": {"source_objective": asdict(config)}, "actual_updates": 1},
        {"source_auxiliary": objective.record(diagnostics, 1)},
    )


@pytest.mark.parametrize("fixed_updates", [None, 2])
def test_deep_training_validation_and_prediction_keep_observed_channel_masks(
    monkeypatch, fixed_updates
):
    from models.n2p3net import DEFAULT_N2P3_ARCHITECTURE, N2P3Net

    values = np.random.default_rng(913).normal(size=(32, 3, 64)).astype(np.float32)
    labels = np.arange(32) % 2
    masks = np.ones((32, 3), dtype=bool)
    masks[np.arange(32), np.arange(32) % 3] = False
    observed_calls = []
    original = N2P3Net._observed_temporal_features

    def checked(model, inputs, channel_mask, **kwargs):
        assert channel_mask is not None and channel_mask.shape == inputs.shape[:2]
        assert not bool(channel_mask.all())
        assert torch.count_nonzero(inputs[~channel_mask]) == 0
        observed_calls.append(model.training)
        return original(model, inputs, channel_mask, **kwargs)

    monkeypatch.setattr(N2P3Net, "_observed_temporal_features", checked)
    residual = (
        {"bottleneck_channels": 4, "kernel_size": 9, "max_residual": 0.25}
        if fixed_updates
        else None
    )
    architecture = replace(
        DEFAULT_N2P3_ARCHITECTURE, observed_channel_norm=True, feature_residual=residual
    )
    auxiliary = (
        SourceObjectiveConfig(supcon_weight=0.05, mmd_weight=0.03, feature_location="post_residual")
        if fixed_updates
        else None
    )
    config = DeepConfig(
        epochs=2,
        fixed_updates=fixed_updates,
        batch_size=8,
        compile_mode=None,
        val_group_frac=0.25,
        val_groups_min=1,
        val_groups_max=2,
        source_objective=auxiliary,
    )
    model = N2P3NetBaseline(
        3, 64, 128, config=config, architecture=architecture, device=torch.device("cpu")
    )
    fit_kwargs = (
        {"source_domain_ids": np.repeat(["a", "b"], 16), "risk_unit_ids": np.repeat(["p", "q"], 16)}
        if fixed_updates
        else {"group_ids": np.repeat(np.arange(8), 4)}
    )
    model.fit(values, labels, trial_channel_mask=masks, **fit_kwargs)
    assert True in observed_calls
    if fixed_updates:
        assert model.last_history["source_objective"]["feature_location"] == "post_residual"
        assert model.last_history["source_objective"]["supcon_weight"] == 0.05
        assert model.last_history["source_objective"]["mmd_weight"] == 0.03
    else:
        assert False in observed_calls  # The validation pass received its own masks.
    state = copy.deepcopy(model.model_.state_dict())
    expected = model.predict_logit(values, trial_channel_mask=masks)
    changed = values.copy()
    changed[~masks] = 1e9
    actual = model.predict_logit(changed, trial_channel_mask=masks)
    np.testing.assert_array_equal(expected, actual)
    assert all(torch.equal(value, model.model_.state_dict()[key]) for key, value in state.items())


def test_source_training_refuses_feature_location_inconsistent_with_actual_model():
    values = np.random.default_rng(31).normal(size=(8, 3, 64)).astype(np.float32)
    config = DeepConfig(
        fixed_updates=1,
        compile_mode=None,
        source_objective=SourceObjectiveConfig(
            supcon_weight=0.05, feature_location="post_residual"
        ),
    )
    model = N2P3NetBaseline(3, 64, 128, config=config, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="feature location"):
        model.fit(
            values,
            np.arange(8) % 2,
            source_domain_ids=np.repeat(["a", "b"], 4),
            risk_unit_ids=np.repeat(["p", "q"], 4),
        )
    assert not model._fitted


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_combined_source_objective_and_observation_masks_execute_on_cuda():
    from models.n2p3net import DEFAULT_N2P3_ARCHITECTURE

    values = np.random.default_rng(51).normal(size=(16, 3, 64)).astype(np.float32)
    mask = np.ones((16, 3), dtype=bool)
    mask[np.arange(16), np.arange(16) % 3] = False
    config = DeepConfig(
        fixed_updates=2,
        batch_size=8,
        compile_mode=None,
        precision="fp32",
        source_objective=SourceObjectiveConfig(
            supcon_weight=0.05, mmd_weight=0.03, feature_location="post_residual"
        ),
    )
    architecture = replace(
        DEFAULT_N2P3_ARCHITECTURE,
        observed_channel_norm=True,
        feature_residual={"bottleneck_channels": 4, "kernel_size": 9, "max_residual": 0.25},
    )
    model = N2P3NetBaseline(
        3, 64, 128, config=config, architecture=architecture, device=torch.device("cuda")
    )
    model.fit(
        values,
        np.arange(16) % 2,
        trial_channel_mask=mask,
        source_domain_ids=np.repeat(["a", "b"], 8),
        risk_unit_ids=np.repeat(["p", "q"], 8),
    )
    record = model.last_history["source_objective"]
    validate_source_objective_contract(
        {"config": asdict(config), "actual_updates": 2}, {"source_auxiliary": record}
    )
    assert np.isfinite(model.predict_logit(values, trial_channel_mask=mask)).all()
    assert model.last_history["trial_draws"] == 16


@pytest.mark.parametrize("kind", ["cross_domain_supcon", "conditional_rbf_mmd"])
def test_frozen_v1_objective_is_read_only_and_cannot_be_relabelled_as_v2(kind):
    raw = {"kind": kind, "weight": 0.05, "temperature": 0.1, "min_cell_count": 2}
    record = {
        "schema": "n2p3_source_objective/1",
        **raw,
        "projection": "parameter_free_l2_normalized_flattened_post_mst",
        "fit_scope": "source_training_rows_only",
        "duplicate_policy": "first_occurrence_per_physical_training_row",
        "positive_policy": "same_class_different_domain_and_participant"
        if kind == "cross_domain_supcon"
        else "class_conditional_domain_pairs",
        "negative_policy": "different_class_only" if kind == "cross_domain_supcon" else None,
        "ce_sampling_changed": False,
        "updates": 3,
        "mean_auxiliary_loss": 0.25,
    }
    optimizer = {"config": {"source_objective": raw}, "actual_updates": 3}
    before = copy.deepcopy((optimizer, record))
    validate_source_objective_contract(optimizer, {"source_auxiliary": record})
    assert (optimizer, record) == before
    with pytest.raises(ValueError, match="differs"):
        validate_source_objective_contract(
            optimizer, {"source_auxiliary": {**record, "schema": "n2p3_source_objective/2"}}
        )
    with pytest.raises(TypeError):
        SourceObjectiveConfig(**raw)  # Historical declarations are not new training recipes.
    with pytest.raises(ValueError, match="legacy"):
        validate_source_objective_contract(
            {"config": {"source_objective": {**raw, "feature_location": "post_residual"}}},
            {"source_auxiliary": record},
        )


def _config(kind, *, weight=0.05):
    field = "supcon_weight" if kind == "cross_domain_supcon" else "mmd_weight"
    return SourceObjectiveConfig(**{field: weight})


def _objective(kind="cross_domain_supcon", **kwargs):
    return SourceObjective(
        _config(kind, weight=1.0),
        labels=np.array([0, 1, 0, 1]),
        domains=np.array(["A", "A", "B", "B"]),
        participants=np.array(["p0", "p0", "p1", "p1"]),
        device="cpu",
        **kwargs,
    )


def test_supcon_matches_explicit_cross_domain_pairs_and_duplicate_rows_have_no_effect():
    objective = _objective()
    features = torch.tensor([[1.0, 0.2], [-0.4, 1.0], [0.3, 0.9], [0.8, -0.3]], requires_grad=True)
    actual, diagnostics = objective.loss(features, torch.arange(4), feature_location="post_mst")
    z = F.normalize(features, dim=1)
    similarity = z @ z.T / 0.1
    reference = torch.stack(
        [
            torch.logsumexp(similarity[i, torch.arange(4) != i], 0) - similarity[i, (i + 2) % 4]
            for i in range(4)
        ]
    ).mean()
    torch.testing.assert_close(actual, reference)
    assert diagnostics["active_anchors"] == 4
    assert diagnostics["positive_pairs"] == 4
    duplicate_rows = torch.tensor([0, 1, 2, 3, 2, 3])
    repeated, extra = objective.loss(
        features[duplicate_rows], duplicate_rows, feature_location="post_mst"
    )
    torch.testing.assert_close(actual, repeated)
    assert extra["unique_rows"] == 4
    actual.backward()
    assert torch.isfinite(features.grad).all() and features.grad.abs().sum() > 0


def test_inactive_single_domain_batch_has_finite_zero_loss_and_gradient():
    objective = _objective()
    features = torch.randn(2, 6, requires_grad=True)
    loss, diagnostics = objective.loss(features, torch.tensor([0, 1]), feature_location="post_mst")
    assert loss.item() == 0 and diagnostics["inactive_batches"] == 1
    loss.backward()
    assert torch.isfinite(features.grad).all() and features.grad.count_nonzero() == 0


@pytest.mark.parametrize("kind", ["conditional_rbf_mmd", "cross_domain_supcon"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_invalid_original_features_cannot_hide_in_inactive_duplicate_rows(kind, bad):
    features = torch.tensor([[1.0, 2.0], [bad, 2.0]], requires_grad=True)
    with pytest.raises(ValueError, match="finite"):
        _objective(kind).loss(features, torch.tensor([0, 0]), feature_location="post_mst")


@pytest.mark.parametrize("count", [0, 1, 3])
def test_conditional_domain_empty_terms_return_graph_connected_zero(count):
    from transfer.alignment import conditional_domain_rbf_mmd_loss

    features = torch.ones(count, 4, requires_grad=True)
    labels = torch.zeros(count, dtype=torch.long)
    loss, info = conditional_domain_rbf_mmd_loss(features, labels, labels, 2, min_cell_count=2)
    assert loss.item() == 0 and info["alignment_inactive"]
    loss.backward()
    assert torch.equal(features.grad, torch.zeros_like(features))


@pytest.mark.parametrize(
    "field,value",
    [
        ("domain_count", True),
        ("min_cell_count", 1),
        ("bandwidth", float("nan")),
    ],
)
def test_empty_domain_terms_do_not_bypass_configuration_validation(field, value):
    from transfer.alignment import conditional_domain_rbf_mmd_loss

    options = dict(domain_count=2, min_cell_count=2, bandwidth=1.0)
    options[field] = value
    with pytest.raises(ValueError):
        conditional_domain_rbf_mmd_loss(
            torch.ones(1, 4),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            **options,
        )


def test_domain_mmd_rejects_fractional_class_codes_even_without_estimable_pairs():
    from transfer.alignment import conditional_domain_rbf_mmd_loss

    with pytest.raises(ValueError, match="integer"):
        conditional_domain_rbf_mmd_loss(
            torch.ones(3, 4),
            torch.tensor([0.0, 0.5, 1.0]),
            torch.zeros(3, dtype=torch.long),
            2,
            min_cell_count=2,
        )


def test_inactive_source_mmd_keeps_ce_optimizer_rng_and_batch_trajectory(monkeypatch):
    torch.manual_seed(711)
    X = np.random.default_rng(711).normal(size=(8, 3, 64)).astype(np.float32)
    y = np.arange(8) % 2
    domains = np.repeat(["A", "B"], 4)
    config = DeepConfig(fixed_updates=5, batch_size=1, compile_mode=None, seed=23)
    models, trajectories = [], []
    original_step = torch.optim.Adam.step

    def capture_step(optimizer, *args, **kwargs):
        gradients = [
            None if p.grad is None else p.grad.detach().clone()
            for group in optimizer.param_groups
            for p in group["params"]
        ]
        random_state = torch.get_rng_state().clone()
        result = original_step(optimizer, *args, **kwargs)
        trace.append((gradients, random_state, copy.deepcopy(optimizer.state_dict())))
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", capture_step)
    for auxiliary in (None, _config("conditional_rbf_mmd")):
        trace = []
        model = N2P3NetBaseline(
            3,
            64,
            128,
            config=replace(config, source_objective=auxiliary),
            device=torch.device("cpu"),
        )
        model.fit(X, y, source_domain_ids=domains, risk_unit_ids=domains)
        models.append(model)
        trajectories.append(trace)
    assert models[0].last_history["batch_sizes"] == models[1].last_history["batch_sizes"] == [1] * 5
    assert models[1].last_history["source_objective"]["inactive_batches"] == 5
    for name, value in models[0].model_.state_dict().items():
        assert torch.equal(value, models[1].model_.state_dict()[name]), name
    assert len(trajectories[0]) == len(trajectories[1]) == 5
    torch.testing.assert_close(trajectories[0], trajectories[1], rtol=0, atol=0)


def test_same_person_in_multiple_domains_is_not_a_positive_pair():
    objective = SourceObjective(
        _config("cross_domain_supcon"),
        labels=np.tile([0, 1], 3),
        domains=np.repeat(["A", "B", "C"], 2),
        participants=np.repeat(["one", "one", "two"], 2),
        device="cpu",
    )
    _, diagnostics = objective.loss(torch.randn(6, 9), torch.arange(6), feature_location="post_mst")
    assert diagnostics["positive_pairs"] == 8  # A-B same-person pairs are excluded.


@pytest.mark.parametrize(
    "field,values",
    [
        ("domains", None),
        ("participants", None),
        ("domains", ["a"] * 4),
        ("participants", ["p"] * 4),
        ("participants", ["p0", "", "p1", "p1"]),
    ],
)
def test_auxiliary_training_refuses_missing_or_degenerate_identity(field, values):
    kwargs = dict(
        labels=np.array([0, 1, 0, 1]),
        domains=["a", "a", "b", "b"],
        participants=["p0", "p0", "p1", "p1"],
        device="cpu",
    )
    kwargs[field] = values
    with pytest.raises(ValueError):
        SourceObjective(_config("cross_domain_supcon"), **kwargs)


@pytest.mark.parametrize("kind", ["cross_domain_supcon", "conditional_rbf_mmd"])
def test_auxiliary_fixed_updates_preserve_sampling_and_bind_each_snapshot_prefix(kind):
    X = np.random.default_rng(3).normal(size=(32, 3, 64)).astype(np.float32)
    labels = np.arange(32) % 2
    domains = np.repeat(["a", "b"], 16)
    people = np.repeat(["person-a", "person-b"], 16)
    config = DeepConfig(
        fixed_updates=3,
        snapshot_updates=(1, 3),
        batch_size=32,
        sampling="epoch_permutation",
        compile_mode=None,
        seed=14,
    )

    def fit(cfg):
        model = N2P3NetBaseline(3, 64, 128, config=cfg, device=torch.device("cpu"))
        return model.fit(X, labels, source_domain_ids=domains, risk_unit_ids=people)

    plain = fit(config)
    active = fit(replace(config, source_objective=_config(kind)))
    assert (
        active.last_history["batch_sequence_sha256"] == plain.last_history["batch_sequence_sha256"]
    )
    assert active.last_history["trial_draws"] == plain.last_history["trial_draws"] == 96
    assert any(
        not torch.equal(value, active.model_.state_dict()[key])
        for key, value in plain.model_.state_dict().items()
    )
    for update in (1, 3):
        record = active.candidate_snapshots_[update]["source_objective"]
        assert record["updates"] == update
        assert record["unique_rows"] == update * 32
        validate_source_objective_contract(
            {"config": asdict(active.cfg), "actual_updates": update}, {"source_auxiliary": record}
        )
        corrupted = copy.deepcopy(record)
        corrupted["supcon_weight"] = 0.7
        with pytest.raises(ValueError, match="differs"):
            validate_source_objective_contract(
                {"config": asdict(active.cfg), "actual_updates": update},
                {"source_auxiliary": corrupted},
            )
    standalone = fit(replace(active.cfg, fixed_updates=1, snapshot_updates=()))
    for key, tensor in standalone.model_.state_dict().items():
        torch.testing.assert_close(
            tensor, active.candidate_snapshots_[1]["model_state_dict"][key], rtol=0, atol=0
        )


def test_ce_control_is_bitwise_unchanged_and_invalid_configs_fail_early():
    X = np.random.default_rng(2).normal(size=(8, 3, 64)).astype(np.float32)
    config = DeepConfig(fixed_updates=2, batch_size=8, compile_mode=None)
    models = [
        N2P3NetBaseline(3, 64, 128, config=cfg, device=torch.device("cpu")).fit(X, np.arange(8) % 2)
        for cfg in (config, replace(config, source_objective=SourceObjectiveConfig()))
    ]
    for key, value in models[0].model_.state_dict().items():
        torch.testing.assert_close(value, models[1].model_.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="fixed-update"):
        DeepConfig(source_objective=_config("cross_domain_supcon"))
    with pytest.raises(ValueError, match="CE control"):
        source_objective_recipes({"only_supcon": {"supcon_weight": 0.05}})
    assert source_objective_recipes()["ce"] is None


@pytest.mark.parametrize("count", [1, 12, 64])
@pytest.mark.parametrize("constant", [False, True])
def test_batched_mmd_matches_reference_values_gradients_and_inactive_cells(count, constant):
    from transfer.alignment import conditional_domain_rbf_mmd_loss

    torch.manual_seed(413)
    labels = np.arange(96) % 2
    domains = np.repeat(np.arange(3), 32)
    objective = SourceObjective(
        _config("conditional_rbf_mmd", weight=1.0),
        labels=labels,
        domains=domains,
        participants=domains,
        device="cpu",
    )
    rows = torch.randperm(96)[:count]
    if count > 1:
        rows[-1] = rows[0]  # Repeated physical draw must not change the auxiliary population.
    x = (torch.ones(count, 7) if constant else torch.randn(count, 7)).requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    value, metrics = objective.loss(x, rows, feature_location="post_mst")
    seen = set()
    keep = []
    for i, row in enumerate(rows.tolist()):
        if row not in seen:
            keep.append(i)
            seen.add(row)
    z = torch.nn.functional.normalize(reference_x, dim=1)[keep]
    if count == 1:
        assert value == 0 and metrics["active_mmd_terms"] == 0
        value.backward()
        assert torch.equal(x.grad, torch.zeros_like(x))
        return
    reference, diagnostics = conditional_domain_rbf_mmd_loss(
        z, objective.labels[rows[keep]], objective.domains[rows[keep]], 3, min_cell_count=2
    )
    torch.testing.assert_close(value, reference, atol=5e-7, rtol=1e-4)
    assert metrics["active_mmd_terms"] == diagnostics["active_terms"]
    if reference.requires_grad:
        reference.backward()
        value.backward()
        torch.testing.assert_close(x.grad, reference_x.grad, atol=5e-7, rtol=1e-4)
    else:
        value.backward()
        assert torch.equal(x.grad, torch.zeros_like(x))


def test_source_objective_checkpoint_gate_rejects_forged_metadata_even_with_a_valid_digest(
    tmp_path,
):
    from research.contracts import semantic_sha256
    from tests.support.checkpoints import write_checkpoint
    from tests.support.datasets import binary_dataset
    from transfer.checkpoint import checkpoint_training_contract

    payload = torch.load(
        write_checkpoint(tmp_path / "source.pt", binary_dataset(), source_snapshot_sha256="a" * 64),
        weights_only=False,
    )
    payload["training_contract"]["objective"]["source_auxiliary"] = _config(
        "cross_domain_supcon"
    ).record()
    payload["training_contract_digest"] = semantic_sha256(payload["training_contract"])
    with pytest.raises(ValueError, match="source auxiliary"):
        checkpoint_training_contract(payload)


def test_source_trajectory_serializes_prefix_objectives_and_loads_for_an_excluded_person():
    from data.epochs import concatenate_epoch_datasets
    from research.source_selection import fit_source_trajectory
    from tests.support.datasets import source_dataset
    from transfer.checkpoint import load_n2p3_trunk_checkpoint

    views = [source_dataset(library=key, namespace=key) for key in ("a", "b")]
    dataset = concatenate_epoch_datasets(
        views,
        name="aux_source",
        provenance={
            "source_reference": "average",
            "source_domain_axis": {
                **views[0].provenance["source_domain_axis"],
                "domains": ["a", "b"],
            },
        },
    )
    rows = np.array([0, 1, 2, 3, 6, 7, 8, 9])
    config = DeepConfig(
        fixed_updates=2,
        snapshot_updates=(1, 2),
        batch_size=8,
        compile_mode=None,
        val_group_frac=None,
        source_objective={"supcon_weight": 0.05},
    )
    _, payloads = fit_source_trajectory(
        dataset,
        rows,
        config=config,
        source_cache_sha256="a" * 64,
        source_snapshot_sha256="b" * 64,
        excluded_source_datasets=["outer"],
        device=torch.device("cpu"),
        evidence_scope="software_correctness",
    )
    for update, payload in payloads.items():
        assert payload["training_contract"]["objective"]["source_auxiliary"]["updates"] == update
        loaded, _ = load_n2p3_trunk_checkpoint(payload, dataset, target_subject="a::s3")
        assert torch.isfinite(loaded(torch.from_numpy(dataset.X[:2]))).all()


def test_supervised_cli_executes_auxiliary_fixed_updates_and_exports_prefix_contracts(tmp_path):
    from data.contract import SOURCE_COHORT_DATA_CONTRACTS
    from data.domain import annotate_source_domain
    from data.epochs import (
        concatenate_epoch_datasets,
        preprocessing_spec_from_contract,
        save_epoch_dataset,
    )
    from data.identity import DatasetIdentityTable
    from experiments.freeze_source_snapshot import freeze_source
    from experiments.run_pretrain_supervised import main
    from tests.support.datasets import binary_dataset
    from transfer.checkpoint import checkpoint_training_contract

    views = []
    spec = preprocessing_spec_from_contract(SOURCE_COHORT_DATA_CONTRACTS["causal"])
    for number, library in enumerate(("a", "b")):
        dataset = binary_dataset(count=600)
        dataset.preprocessing = spec
        dataset.X = (
            np.random.default_rng(number).normal(size=(600, 3, spec.n_times)).astype(np.float32)
            * 1e-6
        )
        dataset.identity_table = DatasetIdentityTable.from_source_rows(
            dataset.subject_ids.tolist(), [library] * dataset.n_epochs
        )
        from data.domain import namespace_epoch_dataset

        views.append(annotate_source_domain(namespace_epoch_dataset(dataset, library), library))
    merged = concatenate_epoch_datasets(
        views,
        name="cli_source",
        provenance={
            "source_reference": "average",
            "source_domain_axis": {
                **views[0].provenance["source_domain_axis"],
                "domains": ["a", "b"],
            },
        },
    )
    cache = save_epoch_dataset(tmp_path / "source.npz", merged)
    snapshot = freeze_source(tmp_path / "snapshot")
    output = tmp_path / "model.pt"
    main(
        [
            "--source-cache",
            str(cache),
            "--source-snapshot-manifest",
            str(snapshot),
            "--checkpoint",
            str(output),
            "--device",
            "cpu",
            "--fixed-updates",
            "2",
            "--snapshot-updates",
            "1",
            "--batch-size",
            "32",
            "--compile-mode",
            "none",
            "--cohort",
            "causal",
            "--source-supcon-weight",
            "0.05",
            "--source-mmd-weight",
            "0.03",
            "--source-feature-location",
            "post_residual",
            "--observed-channel-norm",
            "--feature-residual-config",
            '{"bottleneck_channels":4,"kernel_size":9,"max_residual":0.25}',
        ]
    )
    for path, updates in ((output, 2), (tmp_path / "model.u1.pt", 1)):
        payload = torch.load(path, weights_only=False)
        contract = checkpoint_training_contract(payload)
        assert contract.objective["source_auxiliary"]["updates"] == updates
        assert contract.objective["source_auxiliary"]["schema"] == "n2p3_source_objective/2"
        assert contract.objective["source_auxiliary"]["supcon_weight"] == 0.05
        assert contract.objective["source_auxiliary"]["mmd_weight"] == 0.03
        assert contract.objective["source_auxiliary"]["feature_location"] == "post_residual"
        assert contract.architecture["channel_masking"] == "observed_temporal_bn/1"
        assert contract.architecture["feature_residual"]["bottleneck_channels"] == 4
